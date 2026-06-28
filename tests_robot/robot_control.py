from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
import numpy as np
import time

HEAD_STEP  = 2    # mm
BODY_STEP  = 10   # degrés
ANT_STEP   = 15   # degrés

head_x, head_y, head_z = 0, 0, 0
body_angle = 0
ant_left   = 0
ant_right  = 0

def move(mini, head=True, body=False, ant=False, dur=0.2):
    try:
        kwargs = {"duration": dur, "method": "minjerk"}
        if head:
            kwargs["head"] = create_head_pose(
                x=head_x, y=head_y, z=head_z, mm=True)
        if body:
            kwargs["body_yaw"] = np.deg2rad(body_angle)
        if ant:
            kwargs["antennas"] = np.deg2rad([ant_left, ant_right])
        mini.goto_target(**kwargs)
    except Exception as e:
        print(f"⚠️ {e}")

def draw_hud(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Titre
    cv2.putText(frame, "REACHY MINI CONTROLLER",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,220,255), 2)

    # Tête
    cv2.putText(frame,
                f"HEAD  x:{head_x:+d}  y:{head_y:+d}  z:{head_z:+d} mm",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)

    # Corps
    cv2.putText(frame, f"BODY  {body_angle:+d} deg",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,80), 1)

    # Antennes
    cv2.putText(frame,
                f"ANT   L:{ant_left:+d}°  R:{ant_right:+d}°",
                (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,255), 1)

    # Contrôles
    controls = "ARROWS:head | A/D:body | Z/X:antL | C/V:antR | R:360 | SPACE:reset | Q:quit"
    cv2.putText(frame, controls,
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160,160,160), 1)

    return frame

print("Reachy Mini Controller — fenêtre caméra doit être active")

with ReachyMini(media_backend="default") as mini:
    last_key_time = 0
    COOLDOWN = 0.06

    while True:
        raw = mini.media.get_frame()
        if raw is None:
            continue

        frame = raw.copy()
        frame = draw_hud(frame)
        cv2.imshow("Reachy Mini Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        if now - last_key_time < COOLDOWN:
            continue

        moved = False

        if key == ord('q'):
            break

        # --- RESET ---
        elif key == ord(' '):
            head_x = head_y = head_z = 0
            body_angle = 0
            ant_left = ant_right = 0
            mini.goto_target(
                head=create_head_pose(x=0, y=0, z=0, mm=True),
                body_yaw=0.0,
                antennas=np.deg2rad([0, 0]),
                duration=0.8
            )
            moved = True

        # --- TÊTE : flèches ---
        elif key in [81, 2]:   # ←
            head_y += HEAD_STEP
            move(mini, head=True, dur=0.12)
            moved = True
        elif key in [83, 3]:   # →
            head_y -= HEAD_STEP
            move(mini, head=True, dur=0.12)
            moved = True
        elif key in [82, 0]:   # ↑
            head_z += HEAD_STEP
            move(mini, head=True, dur=0.12)
            moved = True
        elif key in [84, 1]:   # ↓
            head_z -= HEAD_STEP
            move(mini, head=True, dur=0.12)
            moved = True

        # --- CORPS : A / D ---
        elif key == ord('a'):
            body_angle -= BODY_STEP
            move(mini, head=False, body=True, dur=0.2)
            moved = True
        elif key == ord('d'):
            body_angle += BODY_STEP
            move(mini, head=False, body=True, dur=0.2)
            moved = True

        # --- 360° ---
        elif key == ord('r'):
            print("Rotation 360°")
            for a in range(body_angle, body_angle + 361, 8):
                mini.goto_target(body_yaw=np.deg2rad(a), duration=0.05)
                time.sleep(0.05)
                r2 = mini.media.get_frame()
                if r2 is not None:
                    f2 = r2.copy()
                    cv2.putText(f2, f"360° — {a % 360}°",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0,255,0), 2)
                    cv2.imshow("Reachy Mini Controller", f2)
                    cv2.waitKey(1)
            body_angle = body_angle % 360
            moved = True

        # --- ANTENNE GAUCHE : Z / X ---
        elif key == ord('z'):
            ant_left = min(ant_left + ANT_STEP, 120)
            move(mini, head=False, ant=True, dur=0.15)
            moved = True
        elif key == ord('x'):
            ant_left = max(ant_left - ANT_STEP, -120)
            move(mini, head=False, ant=True, dur=0.15)
            moved = True

        # --- ANTENNE DROITE : C / V ---
        elif key == ord('c'):
            ant_right = min(ant_right + ANT_STEP, 120)
            move(mini, head=False, ant=True, dur=0.15)
            moved = True
        elif key == ord('v'):
            ant_right = max(ant_right - ANT_STEP, -120)
            move(mini, head=False, ant=True, dur=0.15)
            moved = True

        # --- HAPPY / SAD émotions ---
        elif key == ord('h'):
            mini.goto_target(
                antennas=np.deg2rad([45, 45]),
                duration=0.3
            )
            ant_left = ant_right = 45
            moved = True

        elif key == ord('s'):
            mini.goto_target(
                antennas=np.deg2rad([-45, -45]),
                duration=0.3
            )
            ant_left = ant_right = -45
            moved = True

        if moved:
            last_key_time = now

cv2.destroyAllWindows()
