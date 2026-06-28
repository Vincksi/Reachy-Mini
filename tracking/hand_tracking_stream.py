from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys
import asyncio
import websockets
import json
import base64
import threading

SERVER_IP = "192.168.8.208"
ROBOT_ID  = int(sys.argv[1]) if len(sys.argv) > 1 else 1

FRAME_W, FRAME_H = 1280, 720
HEAD_MAX_YAW   = 12
HEAD_MAX_PITCH = 8
BODY_K         = 0.12
BODY_DEADZONE  = 0.06
IDLE_DELAY     = 2.0
ANT_NEUTRAL    = 0.0

# File partagée entre thread robot et thread websocket
frame_queue = []
objects_queue = []
import queue
send_queue = queue.Queue(maxsize=2)

class ABFilter:
    def __init__(self, a=0.35, b=0.08, dt=1/20):
        self.a, self.b, self.dt = a, b, dt
        self._x, self._v = None, 0.0
    def update(self, m):
        if self._x is None:
            self._x = m; return m
        xp = self._x + self._v * self.dt
        r  = m - xp
        self._x = xp + self.a * r
        self._v = self._v + (self.b / self.dt) * r
        return self._x
    @property
    def value(self): return self._x or 0.0

def classify(lm):
    if not lm or len(lm) < 21: return "UNKNOWN"
    n = sum([lm[t].y < lm[p].y for t,p in [(8,6),(12,10),(16,14),(20,18)]])
    thumb = lm[4].y < lm[3].y
    if n == 0 and not thumb: return "FIST"
    if n == 4:               return "OPEN"
    if thumb and n == 0:     return "THUMBS_UP" if lm[4].y < lm[0].y else "THUMBS_DOWN"
    if n == 2:               return "PEACE"
    if n == 1:               return "POINTING"
    return "UNKNOWN"

def gesture_to_ant(lm, gesture):
    if not lm: return ANT_NEUTRAL
    dx = lm[9].x - lm[0].x
    dy = lm[9].y - lm[0].y
    angle = -math.atan2(-dy, dx) * 0.7
    if gesture == "THUMBS_UP":   return -0.8
    if gesture == "THUMBS_DOWN": return  0.8
    return float(np.clip(angle, -math.pi, math.pi))

def encode_frame(frame):
    small = cv2.resize(frame, (320, 180))
    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return base64.b64encode(buf).decode('utf-8')

# Thread WebSocket — envoie en arrière-plan
async def ws_sender():
    uri = f"ws://{SERVER_IP}:8000/robot/{ROBOT_ID}"
    while True:
        try:
            async with websockets.connect(uri, max_size=10_000_000) as ws:
                print(f"Robot {ROBOT_ID} connecté au serveur !")
                while True:
                    try:
                        data = send_queue.get(timeout=0.5)
                        await ws.send(json.dumps(data))
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Reconnexion... {e}")
            await asyncio.sleep(1)

def ws_thread():
    asyncio.run(ws_sender())

threading.Thread(target=ws_thread, daemon=True).start()

# Main — robot + hand tracking
fx = ABFilter(); fy = ABFilter()
f_ant_l = ABFilter(0.25, 0.04)
f_ant_r = ABFilter(0.25, 0.04)
body_yaw    = 0.0
t_last_hand = 0.0

GESTURE_COLORS = {
    "OPEN":(0,230,120),"FIST":(0,60,255),"THUMBS_UP":(0,255,220),
    "THUMBS_DOWN":(60,60,255),"PEACE":(255,200,0),
    "POINTING":(255,140,0),"UNKNOWN":(180,180,180),
}

hands_mp = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

print(f"Robot {ROBOT_ID} — Hand Tracking + Stream. Q pour quitter")

with ReachyMini(media_backend="default") as mini:
    while True:
        raw = mini.media.get_frame()
        if raw is None: continue

        frame = raw.copy()
        rgb   = np.ascontiguousarray(frame[:,:,::-1])
        res   = hands_mp.process(rgb)

        now     = time.time()
        n_hands = 0
        gestures, landmarks, sides = [], [], []

        if res.multi_hand_landmarks:
            n_hands     = len(res.multi_hand_landmarks)
            t_last_hand = now

            for i, lm in enumerate(res.multi_hand_landmarks):
                lm_list = lm.landmark
                gestures.append(classify(lm_list))
                landmarks.append(lm_list)
                side = res.multi_handedness[i].classification[0].label
                sides.append(side)

                cx = np.mean([p.x for p in lm_list]) * FRAME_W
                cy = np.mean([p.y for p in lm_list]) * FRAME_H
                if i == 0:
                    fx.update(cx); fy.update(cy)

                color = GESTURE_COLORS.get(gestures[i], (180,180,180))
                for conn in mp.solutions.hands.HAND_CONNECTIONS:
                    p1,p2 = lm_list[conn[0]], lm_list[conn[1]]
                    cv2.line(frame,
                             (int(p1.x*FRAME_W),int(p1.y*FRAME_H)),
                             (int(p2.x*FRAME_W),int(p2.y*FRAME_H)),
                             (180,180,180),1)
                for pt in lm_list:
                    cv2.circle(frame,(int(pt.x*FRAME_W),int(pt.y*FRAME_H)),4,(50,50,255),-1)
                cv2.circle(frame,(int(cx),int(cy)),12,color,-1)
                cv2.putText(frame,f"{side}:{gestures[i]}",(int(cx)-60,int(cy)-18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        # Commande robot
        if n_hands == 0:
            if now - t_last_hand > IDLE_DELAY:
                body_yaw = 0.0
                try:
                    mini.goto_target(
                        head=create_head_pose(x=0,y=0,z=0,mm=True),
                        antennas=np.array([ANT_NEUTRAL,ANT_NEUTRAL]),
                        body_yaw=0.0, duration=0.5
                    )
                except: pass
        else:
            err_h = (fx.value-FRAME_W/2)/(FRAME_W/2)*math.radians(34)
            err_v = (fy.value-FRAME_H/2)/(FRAME_H/2)*math.radians(21)
            yaw   = float(np.clip(-math.degrees(err_h)*0.35, -HEAD_MAX_YAW, HEAD_MAX_YAW))
            pitch = float(np.clip( math.degrees(err_v)*0.25, -HEAD_MAX_PITCH, HEAD_MAX_PITCH))
            if abs(err_h) > BODY_DEADZONE:
                body_yaw -= err_h * BODY_K
                body_yaw  = float(np.clip(body_yaw, -math.pi, math.pi))
            if n_hands == 1:
                ant = f_ant_l.update(gesture_to_ant(landmarks[0], gestures[0]))
                antennas = np.array([ant, -ant])
            else:
                idx_l = next((i for i,s in enumerate(sides) if s=="Left"),  0)
                idx_r = next((i for i,s in enumerate(sides) if s=="Right"), 1)
                antennas = np.array([
                    f_ant_l.update(gesture_to_ant(landmarks[idx_l], gestures[idx_l])),
                    f_ant_r.update(gesture_to_ant(landmarks[idx_r], gestures[idx_r]))
                ])
            try:
                mini.goto_target(
                    head=create_head_pose(x=round(pitch,1),y=round(yaw,1),mm=True),
                    body_yaw=body_yaw,
                    antennas=antennas,
                    duration=0.08, method="linear"
                )
            except: pass

        # Envoie au serveur (non bloquant)
        try:
            send_queue.put_nowait({
                "robot_id":  ROBOT_ID,
                "timestamp": now,
                "objects":   {"hands": n_hands},
                "frame":     encode_frame(frame)
            })
        except queue.Full:
            pass

        # HUD
        overlay = frame.copy()
        cv2.rectangle(overlay,(0,0),(FRAME_W,55),(0,0,0),-1)
        cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
        cv2.putText(frame,
                    f"R{ROBOT_ID} | Hands:{n_hands} | Body:{math.degrees(body_yaw):+.0f}°",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(100,220,255),2)

        cv2.imshow(f"Robot {ROBOT_ID} — Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
