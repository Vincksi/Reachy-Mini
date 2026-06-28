"""
Hand Tracking pour Reachy Mini
================================
Architecture propre :
  - MediaPipe HandLandmarker VIDEO mode  : tracking temporel stable (21 landmarks)
  - Filtre alpha-beta sur cx/cy          : gere aussi bien les mains lentes que rapides
  - Controller tete : look_at_world()    : utilise le vrai champ de vue camera
  - Controller corps : accumulation angulaire propre sur angle reel (pas pixel brut)
  - Commande unique set_target()         : tete + corps + antennes en un seul message WS
  - Gestures : 6 gestes -> commande antennes

Usage :
    python hand_tracking.py               # USB-C (defaut)
    python hand_tracking.py --wifi        # WiFi
    python hand_tracking.py --host IP
"""

import argparse
import math
import os
import time
import urllib.request
from collections import deque
from enum import Enum, auto
from typing import List, Optional
import asyncio
import websockets
import json
import base64
import queue
import threading
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Stream config ───────────────────────────────────────────────────────────────
SERVER_IP  = "192.168.8.208"
SERVER_PORT = 8000
ROBOT_ID   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
_send_queue: queue.Queue = queue.Queue(maxsize=2)

def _encode_frame(frame: np.ndarray) -> str:
    small = cv2.resize(frame, (320, 180))
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return base64.b64encode(buf).decode('utf-8')

async def _ws_sender() -> None:
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}/robot/{ROBOT_ID}"
    while True:
        try:
            async with websockets.connect(uri, max_size=10_000_000) as ws:
                print(f"[Stream] Robot {ROBOT_ID} connecté au serveur !")
                while True:
                    try:
                        data = _send_queue.get(timeout=0.5)
                        await ws.send(json.dumps(data))
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"[Stream] Reconnexion... {e}")
            await asyncio.sleep(1)

def _ws_thread() -> None:
    asyncio.run(_ws_sender())

threading.Thread(target=_ws_thread, daemon=True).start()

# ────────────────────────── Constantes ─────────────────────────────────────────

FRAME_W, FRAME_H = 640, 480

# Champ de vue de la camera Reachy Mini (approx)
CAM_HFOV_RAD = math.radians(69)
CAM_VFOV_RAD = math.radians(42)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"

# Antennes
ANT_NEUTRAL = 0.0
ANT_HAPPY   = -0.8
ANT_SAD     =  0.8

IDLE_DELAY = 2.0       # secondes sans main avant retour neutre

# Corps : gains du controleur integrant
# K_BODY est en rad de rotation par rad d'erreur angulaire par frame
# Plus grand = rotation plus rapide du corps
K_BODY        = 0.12   # rad de yaw accumule par rad d'erreur par frame
BODY_DEADZONE = 0.06   # en dessous de cette erreur angulaire (rad), le corps ne tourne pas
BODY_MAX      = math.pi  # rotation maximale (+-180 deg)

# Connexion
DEFAULT_HOST = "localhost"
DEFAULT_CONN = "localhost_only"


# ────────────────────────── Filtre Alpha-Beta ───────────────────────────────────

class AlphaBetaFilter:
    """
    Tracker de 1er ordre : lisse la position ET estime la vitesse.
    Avantage sur EMA : faible retard pour les mouvements rapides.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 0.08, dt: float = 1 / 20):
        self.alpha = alpha
        self.beta  = beta
        self.dt    = dt
        self._x: Optional[float] = None
        self._v = 0.0

    @property
    def value(self) -> float:
        return self._x if self._x is not None else 0.0

    def update(self, measurement: float) -> float:
        if self._x is None:
            self._x = measurement
            return self._x
        x_pred   = self._x + self._v * self.dt
        residual = measurement - x_pred
        self._x  = x_pred + self.alpha * residual
        self._v  = self._v + (self.beta / self.dt) * residual
        return self._x

    def reset(self, v: float) -> None:
        self._x = v
        self._v = 0.0


# ────────────────────────── Gestes ─────────────────────────────────────────────

class Gesture(Enum):
    UNKNOWN     = auto()
    OPEN_HAND   = auto()
    FIST        = auto()
    THUMBS_UP   = auto()
    THUMBS_DOWN = auto()
    PEACE       = auto()
    POINTING    = auto()


def _ext(lm: list, tip: int, pip: int) -> bool:
    return lm[tip].y < lm[pip].y


def classify_gesture(lm: list) -> Gesture:
    if not lm or len(lm) < 21:
        return Gesture.UNKNOWN
    thumb_up = lm[4].y < lm[3].y
    n = sum([_ext(lm, 8, 6), _ext(lm, 12, 10), _ext(lm, 16, 14), _ext(lm, 20, 18)])
    if n == 0 and not thumb_up:                              return Gesture.FIST
    if n == 4:                                               return Gesture.OPEN_HAND
    if thumb_up and n == 0:
        return Gesture.THUMBS_UP if lm[4].y < lm[0].y else Gesture.THUMBS_DOWN
    if _ext(lm, 8, 6) and _ext(lm, 12, 10) and n == 2:     return Gesture.PEACE
    if _ext(lm, 8, 6) and n == 1:                           return Gesture.POINTING
    return Gesture.UNKNOWN


def gesture_to_antenna(lm: list, gesture: Gesture) -> float:
    if not lm or len(lm) < 21:
        return ANT_NEUTRAL
    dx = lm[9].x - lm[0].x
    dy = lm[9].y - lm[0].y
    angle = -math.atan2(-dy, dx) * 0.7
    if gesture == Gesture.OPEN_HAND:     angle = max(angle, ANT_HAPPY)
    elif gesture == Gesture.FIST:        angle = min(angle, ANT_SAD)
    elif gesture == Gesture.THUMBS_UP:   angle = ANT_HAPPY
    elif gesture == Gesture.THUMBS_DOWN: angle = ANT_SAD
    return float(np.clip(angle, -math.pi, math.pi))


# ────────────────────────── Squelette MediaPipe ─────────────────────────────────

HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17),
]

GESTURE_COLORS = {
    Gesture.OPEN_HAND:   (0,  230, 120),
    Gesture.FIST:        (0,   60, 255),
    Gesture.THUMBS_UP:   (0,  255, 220),
    Gesture.THUMBS_DOWN: (60,  60, 255),
    Gesture.PEACE:       (255, 200,  0),
    Gesture.POINTING:    (255, 140,  0),
    Gesture.UNKNOWN:     (180, 180, 180),
}


# ────────────────────────── HandTracker ────────────────────────────────────────

class HandTracker:
    """
    Boucle principale du hand tracking.

    La tete et le corps du robot suivent la main detectee par MediaPipe.
    Le corps tourne indefiniment pour garder la main au centre de l'image
    (tracking a 360 deg).
    """

    def __init__(self, host: str, conn_mode: str, show_display: bool):
        self.host         = host
        self.conn_mode    = conn_mode
        self.show_display = show_display

        # Filtres alpha-beta pour la position de la main dans l'image (px)
        dt = 1 / 20
        self._fx = AlphaBetaFilter(alpha=0.35, beta=0.08, dt=dt)
        self._fy = AlphaBetaFilter(alpha=0.35, beta=0.08, dt=dt)

        # Filtres pour les antennes
        self._f_ant_l = AlphaBetaFilter(alpha=0.25, beta=0.04, dt=dt)
        self._f_ant_r = AlphaBetaFilter(alpha=0.25, beta=0.04, dt=dt)

        # Accumulateur de yaw corps (angle absolu en rad)
        self._body_yaw = 0.0

        # Timestamp MediaPipe (doit etre strictement croissant)
        self._mp_ts = 0

        # Etat detecte
        self._robot       = None
        self._running     = False
        self._t_last_hand = 0.0
        self._fps_buf: deque = deque(maxlen=30)

        self._gestures : List[Gesture]          = [Gesture.UNKNOWN, Gesture.UNKNOWN]
        self._landmarks: List[Optional[list]]   = [None, None]
        self._sides    : List[str]              = ["Unknown", "Unknown"]

    # ── Modele ─────────────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if not os.path.exists(MODEL_PATH):
            print("[HandTracker] Telechargement du modele MediaPipe...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[HandTracker] Modele OK.")

    def _build_detector(self) -> mp_vision.HandLandmarker:
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.HandLandmarker.create_from_options(opts)

    def _next_ts(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._mp_ts:
            ts = self._mp_ts + 1
        self._mp_ts = ts
        return ts

    # ── Camera ─────────────────────────────────────────────────────────────────

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        if self._robot is not None:
            try:
                self._robot.release_media()
            except Exception:
                pass

        for idx in range(4):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[HandTracker] Camera index={idx}")
                return cap
            cap.release()
        return None

    # ── Robot ───────────────────────────────────────────────────────────────────

    def _connect_robot(self) -> None:
        try:
            from reachy_mini import ReachyMini
            self._robot = ReachyMini(
                host=self.host,
                connection_mode=self.conn_mode,
                media_backend="no_media",
                log_level="WARNING",
            )
            self._robot.wake_up()
            self._robot.enable_motors()
            print("[HandTracker] Robot connecte.")
        except Exception as exc:
            print(f"[HandTracker] Robot indisponible : {exc}")
            print("[HandTracker] Mode demo (affichage seul).")
            self._robot = None

    def _shutdown_robot(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.set_target(antennas=[ANT_NEUTRAL, ANT_NEUTRAL], body_yaw=0.0)
            time.sleep(0.3)
            self._robot.goto_sleep()
        except Exception:
            pass
        finally:
            try:
                self._robot.__exit__(None, None, None)
            except Exception:
                pass

    # ── Controller ─────────────────────────────────────────────────────────────

    def _send_commands(self, n_hands: int, cx: float, cy: float,
                       antennas: List[float]) -> None:
        """
        Calcule et envoie en une seule commande :
          - La pose de tete (look_at_world avec projection optique reelle)
          - Le yaw du corps (accumule l'erreur angulaire pour rotation 360 deg)
          - Les antennes
        """
        if self._robot is None:
            return

        now = time.time()

        # ── Aucune main : retour neutre apres delai ─────────────────────────
        if n_hands == 0:
            if now - self._t_last_hand > IDLE_DELAY:
                self._body_yaw = 0.0
                head = self._robot.look_at_world(1.0, 0.0, 0.0, duration=0,
                                                 perform_movement=False)
                try:
                    self._robot.set_target(
                        head=head,
                        antennas=[ANT_NEUTRAL, ANT_NEUTRAL],
                        body_yaw=0.0,
                    )
                except Exception as exc:
                    print(f"[HandTracker] {exc}")
            return

        # ── Au moins 1 main ─────────────────────────────────────────────────
        # Erreur angulaire reelle en utilisant le vrai champ de vue camera
        err_h = ((cx - FRAME_W / 2) / (FRAME_W / 2)) * (CAM_HFOV_RAD / 2)
        err_v = ((cy - FRAME_H / 2) / (FRAME_H / 2)) * (CAM_VFOV_RAD / 2)

        # ── Tete : asservissement visuel ferme ──────────────────────────────
        try:
            T_body_head = self._robot.get_current_head_pose()
            T_body_cam = T_body_head @ self._robot.T_head_cam
            
            # Dans le repere camera: Z est en avant, X est a droite, Y est en bas
            ray_cam = np.array([math.tan(err_h), math.tan(err_v), 1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            
            # Transformation du rayon dans le repere du corps du robot
            ray_body = T_body_cam[:3, :3] @ ray_cam
            t_body_cam = T_body_cam[:3, 3]
            
            # Point cible dans le repere robot (a 1.0m devant la camera)
            target_pos = t_body_cam + ray_body * 1.0
            head = self._robot.look_at_world(
                target_pos[0], target_pos[1], target_pos[2], duration=0, perform_movement=False
            )
        except Exception:
            # Fallback si get_current_head_pose() echoue
            ray_y = -math.tan(err_h)
            ray_z = -math.tan(err_v)
            head = self._robot.look_at_world(
                1.0, ray_y, ray_z, duration=0, perform_movement=False
            )

        # ── Corps : accumulation angulaire reelle ───────────────────────────
        # La main est a err_h radians du centre horizontal.
        # On tourne le corps en sens inverse pour recentrer la main.
        # Zone morte BODY_DEADZONE evite les micro-oscillations.
        if abs(err_h) > BODY_DEADZONE:
            self._body_yaw -= err_h * K_BODY
            self._body_yaw  = float(np.clip(self._body_yaw, -BODY_MAX, BODY_MAX))

        # ── Envoi unique ─────────────────────────────────────────────────────
        try:
            self._robot.set_target(
                head=head,
                body_yaw=self._body_yaw,
                antennas=antennas,
            )
        except Exception as exc:
            print(f"[HandTracker] {exc}")

    # ── Parsing des resultats MediaPipe ─────────────────────────────────────────

    def _parse_result(self, result) -> int:
        n = 0
        if result.hand_landmarks:
            n = min(len(result.hand_landmarks), 2)
            for i in range(n):
                lm   = result.hand_landmarks[i]
                info = result.handedness[i][0]
                raw_cx = float(np.mean([p.x for p in lm])) * FRAME_W
                raw_cy = float(np.mean([p.y for p in lm])) * FRAME_H
                self._landmarks[i] = lm
                self._sides[i]     = info.category_name
                self._gestures[i]  = classify_gesture(lm)
                if i == 0:
                    self._fx.update(raw_cx)
                    self._fy.update(raw_cy)

        for i in range(n, 2):
            self._landmarks[i] = None

        return n

    def _compute_antennas(self, n_hands: int) -> List[float]:
        if n_hands == 0:
            return [ANT_NEUTRAL, ANT_NEUTRAL]
        if n_hands == 1:
            lm  = self._landmarks[0]
            ant = self._f_ant_l.update(gesture_to_antenna(lm, self._gestures[0]))
            return [ant, -ant]
        lm_l = next((self._landmarks[i] for i in range(2) if self._sides[i] == "Left"),  self._landmarks[0])
        lm_r = next((self._landmarks[i] for i in range(2) if self._sides[i] == "Right"), self._landmarks[1])
        g_l  = next((self._gestures[i]  for i in range(2) if self._sides[i] == "Left"),  self._gestures[0])
        g_r  = next((self._gestures[i]  for i in range(2) if self._sides[i] == "Right"), self._gestures[1])
        return [
            self._f_ant_l.update(gesture_to_antenna(lm_l, g_l)),
            self._f_ant_r.update(gesture_to_antenna(lm_r, g_r)),
        ]

    # ── Rendu ───────────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, result, n_hands: int, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)

        if result.hand_landmarks:
            for i, lm in enumerate(result.hand_landmarks):
                for p1, p2 in HAND_BONES:
                    x1, y1 = int(lm[p1].x * w), int(lm[p1].y * h)
                    x2, y2 = int(lm[p2].x * w), int(lm[p2].y * h)
                    cv2.line(frame, (w - x1, y1), (w - x2, y2), (180, 180, 180), 1, cv2.LINE_AA)
                for pt in lm:
                    px, py = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (w - px, py), 4, (50, 50, 255), -1)
                if i < 2 and self._landmarks[i] is not None:
                    color = GESTURE_COLORS[self._gestures[i]]
                    cx_d  = w - int(np.mean([p.x for p in lm]) * w)
                    cy_d  = int(np.mean([p.y for p in lm]) * h)
                    cv2.circle(frame, (cx_d, cy_d), 12, color, -1)
                    cv2.circle(frame, (cx_d, cy_d), 14, (255, 255, 255), 2)
                    cv2.putText(frame, f"{self._sides[i]}: {self._gestures[i].name}",
                                (cx_d - 70, cy_d - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 70), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        fps_col = (0, 230, 60) if fps > 24 else (0, 200, 230) if fps > 14 else (0, 50, 255)
        cv2.putText(frame, f"FPS {fps:4.0f}   Mains: {n_hands}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_col, 2, cv2.LINE_AA)

        yaw_deg  = math.degrees(self._body_yaw)
        robot_ok = self._robot is not None
        status   = f"Robot: {'OK' if robot_ok else 'DEMO'}   Corps: {yaw_deg:+.0f} deg"
        cv2.putText(frame, status, (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 255, 80) if robot_ok else (80, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q = quitter", (w - 145, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        return frame

    # ── Boucle principale ───────────────────────────────────────────────────────

    def run(self) -> None:
        self._ensure_model()
        self._connect_robot()
        detector = self._build_detector()

        cap = self._open_camera()
        if cap is None:
            print("[HandTracker] Impossible d'ouvrir la camera.")
            return

        self._running = True
        print("[HandTracker] Demarre — appuyez sur Q pour quitter.")

        try:
            while self._running:
                t0 = time.perf_counter()

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    result = detector.detect_for_video(mp_img, self._next_ts())
                except RuntimeError:
                    continue

                n_hands  = self._parse_result(result)
                if n_hands > 0:
                    self._t_last_hand = time.time()

                cx       = self._fx.value
                cy       = self._fy.value
                antennas = self._compute_antennas(n_hands)
                self._send_commands(n_hands, cx, cy, antennas)

                dt = time.perf_counter() - t0
                self._fps_buf.append(max(dt, 1e-6))
                fps = len(self._fps_buf) / sum(self._fps_buf)

                # ── Stream vers serveur ─────────────────────────────────────
                try:
                    _send_queue.put_nowait({
                        "robot_id":  ROBOT_ID,
                        "timestamp": time.time(),
                        "objects":   {"hands": n_hands},
                        "frame":     _encode_frame(frame)
                    })
                except queue.Full:
                    pass

                if self.show_display:
                    disp = self._draw(frame, result, n_hands, fps)
                    cv2.imshow(f"R{ROBOT_ID} — Hand Tracking", disp)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break

        finally:
            self._running = False
            cap.release()
            if self.show_display:
                cv2.destroyAllWindows()
            self._shutdown_robot()
            print("[HandTracker] Arrete.")


# ────────────────────────── Entree ──────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Hand Tracking pour Reachy Mini")
    ap.add_argument("--host",       default=DEFAULT_HOST)
    ap.add_argument("--wifi",       action="store_true")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    host      = args.host
    conn_mode = DEFAULT_CONN
    if args.wifi:
        host      = args.host if args.host != DEFAULT_HOST else "reachy-mini.local"
        conn_mode = "network"

    print(f"[HandTracker] Mode {'WiFi' if args.wifi else 'USB-C'} -> {host}")
    HandTracker(host=host, conn_mode=conn_mode, show_display=not args.no_display).run()


if __name__ == "__main__":
    main()