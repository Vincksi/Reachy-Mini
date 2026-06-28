from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from ultralytics import YOLO
import cv2
import numpy as np
import time

FRAME_W = 1280
FRAME_H = 720
HEAD_MAX_YAW   = 12
HEAD_MAX_PITCH = 8
BODY_STEP      = 10
CENTER_ZONE    = 0.25

model = YOLO("yolo11n.pt")
locked_id    = None
locked_label = None
body_angle   = 0
last_body    = 0

def move(mini, yaw=0, pitch=0, body=None, dur=0.1):
    try:
        kwargs = {
            "head": create_head_pose(x=round(pitch,1), y=round(yaw,1), mm=True),
            "duration": dur,
            "method": "linear"
        }
        if body is not None:
            kwargs["body_yaw"] = np.deg2rad(body)
        mini.goto_target(**kwargs)
    except: pass

print("SPACE:lock nearest  U:unlock  0:reset  Q:quit")

with ReachyMini(media_backend="default") as mini:
    while True:
        raw = mini.media.get_frame()
        if raw is None:
            continue

        frame = raw.copy()

        # ByteTrack — ID stable par objet
        results = model.track(frame, persist=True,
                              tracker="bytetrack.yaml",
                              verbose=False, conf=0.4)

        hud = frame.copy()
        now = time.time()
        target = None

        boxes  = results[0].boxes
        left_z  = int(FRAME_W * CENTER_ZONE)
        right_z = int(FRAME_W * (1 - CENTER_ZONE))

        # Zones
        cv2.line(hud,(left_z,0),(left_z,FRAME_H),(40,40,40),1)
        cv2.line(hud,(right_z,0),(right_z,FRAME_H),(40,40,40),1)
        cv2.line(hud,(FRAME_W//2-20,FRAME_H//2),(FRAME_W//2+20,FRAME_H//2),(255,255,255),1)
        cv2.line(hud,(FRAME_W//2,FRAME_H//2-20),(FRAME_W//2,FRAME_H//2+20),(255,255,255),1)

        for box in boxes:
            if box.id is None:
                continue
            tid  = int(box.id)
            name = model.names[int(box.cls)]
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            cx,cy = (x1+x2)//2,(y1+y2)//2

            if tid == locked_id:
                target = (cx, cy, x1, y1, x2, y2, name)
                # Box verte + coins
                cv2.rectangle(hud,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(hud,(cx,cy),6,(0,255,0),-1)
                cv2.line(hud,(FRAME_W//2,FRAME_H//2),(cx,cy),(0,200,255),1)
                l=16
                for (px,py),(dx,dy) in [((x1,y1),(1,1)),((x2,y1),(-1,1)),
                                         ((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
                    cv2.line(hud,(px,py),(px+dx*l,py),(0,255,0),2)
                    cv2.line(hud,(px,py),(px,py+dy*l),(0,255,0),2)
                cv2.putText(hud,f"ID:{tid} {name}",
                            (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            else:
                cv2.rectangle(hud,(x1,y1),(x2,y2),(60,60,60),1)
                cv2.putText(hud,f"ID:{tid} {name}",
                            (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(60,60,60),1)

        if target:
            cx,cy,x1,y1,x2,y2,name = target

            # 1. Tête suit proportionnellement
            yaw   = -(cx - FRAME_W/2) / (FRAME_W/2) * HEAD_MAX_YAW
            pitch =  (cy - FRAME_H/2) / (FRAME_H/2) * HEAD_MAX_PITCH
            
            # 2. Corps corrige si hors zone centrale
            if cx < left_z and now - last_body > 0.3:
                offset = abs(cx - left_z) / left_z
                body_angle = (body_angle - BODY_STEP * offset) % 360
                move(mini, yaw=0, pitch=0, body=body_angle, dur=0.3)
                last_body = now
                cv2.putText(hud,"◄ CENTERING",(10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,120,255),2)
            elif cx > right_z and now - last_body > 0.3:
                offset = abs(cx - right_z) / (FRAME_W - right_z)
                body_angle = (body_angle + BODY_STEP * offset) % 360
                move(mini, yaw=0, pitch=0, body=body_angle, dur=0.3)
                last_body = now
                cv2.putText(hud,"CENTERING ►",(10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,120,255),2)
            else:
                move(mini, yaw=yaw, pitch=pitch)

            cv2.putText(hud,f"TRACKING ID:{locked_id} {name}  body:{int(body_angle)}°",
                        (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            # Barre centrage
            offset_bar = int((cx - FRAME_W/2) / (FRAME_W/2) * 100)
            bx = FRAME_W//2 + offset_bar
            cv2.rectangle(hud,(FRAME_W//2-100,FRAME_H-25),
                          (FRAME_W//2+100,FRAME_H-12),(30,30,30),-1)
            color_bar = (0,255,0) if left_z < cx < right_z else (0,120,255)
            cv2.rectangle(hud,(FRAME_W//2,FRAME_H-23),(bx,FRAME_H-14),color_bar,-1)

        elif locked_id is not None:
            # Perdu → cherche en tournant
            cv2.putText(hud,f"SEARCHING ID:{locked_id}...",
                        (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,60,255),2)
            if now - last_body > 0.4:
                body_angle = (body_angle + 18) % 360
                move(mini, body=body_angle, dur=0.3)
                last_body = now
        else:
            cv2.putText(hud,"SPACE to lock nearest object",
                        (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),1)

        cv2.putText(hud,f"Body:{int(body_angle)}°  SPACE:lock  U:unlock  0:reset  Q:quit",
                    (10,FRAME_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,(120,120,120),1)

        cv2.imshow("ByteTrack", hud)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            # Lock l'objet le plus proche du centre
            best_id, best_name, best_dist = None, None, float('inf')
            for box in boxes:
                if box.id is None: continue
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                cx,cy = (x1+x2)//2,(y1+y2)//2
                dist = ((cx-FRAME_W//2)**2+(cy-FRAME_H//2)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id   = int(box.id)
                    best_name = model.names[int(box.cls)]
            if best_id is not None:
                locked_id    = best_id
                locked_label = best_name
                print(f"Locked ID:{locked_id} {locked_label}")

        elif key == ord('u'):
            locked_id = locked_label = None
            move(mini, body=body_angle, dur=0.5)

        elif key == ord('0'):
            locked_id = locked_label = None
            body_angle = 0
            move(mini, yaw=0, pitch=0, body=0, dur=0.8)

cv2.destroyAllWindows()
