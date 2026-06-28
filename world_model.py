from reachy_mini import ReachyMini
from ultralytics import YOLO
import cv2
import numpy as np
import time

FRAME_W = 1280
FRAME_H = 720
REF_HEIGHT_PX = 350
REF_DISTANCE_M = 1.0

model = YOLO("yolo11n.pt")

def estimate_distance(box_h):
    if box_h < 10:
        return None
    return max(0.3, min((REF_HEIGHT_PX * REF_DISTANCE_M) / box_h, 8.0))

def estimate_angle(box_cx):
    return (box_cx - FRAME_W / 2) / FRAME_W * 55

def box_to_world(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    h  = y2 - y1
    dist = estimate_distance(h)
    if dist is None:
        return None
    angle_rad = np.radians(estimate_angle(cx))
    return dist * np.sin(angle_rad), dist * np.cos(angle_rad), dist

def iso_project(wx, wz, cx, robot_y, scale=85):
    """Projection isométrique style Tesla."""
    px = int(cx + wx * scale)
    py = int(robot_y - wz * scale * 0.55)
    return px, py

def draw_shadow(img, px, py, rx, ry, color=(200,200,200)):
    """Ombre portée elliptique."""
    cv2.ellipse(img, (px + 4, py + 4), (rx, ry), 0, 0, 360, color, -1)

def draw_person_3d(img, px, py, scale=1.0):
    """Silhouette humaine 3D style Tesla."""
    s = max(0.3, scale)
    # Ombre
    draw_shadow(img, px, py + int(18*s), int(12*s), int(5*s), (195,195,195))
    # Jambes
    leg_w = int(4*s)
    leg_h = int(14*s)
    cv2.rectangle(img, (px - leg_w - 1, py + int(4*s)),
                  (px - 1, py + int(4*s) + leg_h), (160,160,160), -1)
    cv2.rectangle(img, (px + 1, py + int(4*s)),
                  (px + leg_w + 1, py + int(4*s) + leg_h), (160,160,160), -1)
    # Corps
    body_w = int(10*s)
    body_h = int(16*s)
    cv2.rectangle(img, (px - body_w, py - body_h),
                  (px + body_w, py + int(4*s)), (175,175,175), -1)
    # Épaules
    cv2.ellipse(img, (px, py - body_h), (int(11*s), int(5*s)),
                0, 0, 360, (185,185,185), -1)
    # Tête
    head_r = int(8*s)
    cv2.circle(img, (px, py - body_h - head_r), head_r, (185,185,185), -1)
    # Contour léger
    cv2.circle(img, (px, py - body_h - head_r), head_r, (210,210,210), 1)

def draw_object_3d(img, px, py, label, scale=1.0):
    """Objet 3D générique style Tesla."""
    s = max(0.3, scale)
    w = int(16*s)
    h = int(10*s)
    d = int(6*s)  # profondeur isométrique

    # Ombre
    draw_shadow(img, px, py + h + 2, w, int(4*s), (200,200,200))

    # Face avant
    pts_front = np.array([
        [px - w, py],
        [px + w, py],
        [px + w, py - h*2],
        [px - w, py - h*2]
    ], np.int32)
    cv2.fillPoly(img, [pts_front], (175,175,175))

    # Face dessus (isométrique)
    pts_top = np.array([
        [px - w,     py - h*2],
        [px + w,     py - h*2],
        [px + w + d, py - h*2 - d],
        [px - w + d, py - h*2 - d],
    ], np.int32)
    cv2.fillPoly(img, [pts_top], (195,195,195))

    # Face côté
    pts_side = np.array([
        [px + w,     py],
        [px + w + d, py - d],
        [px + w + d, py - h*2 - d],
        [px + w,     py - h*2],
    ], np.int32)
    cv2.fillPoly(img, [pts_side], (155,155,155))

def draw_reachy_3d(img, cx, robot_y):
    """Reachy Mini style Tesla — couleur vive."""
    # Ombre
    cv2.ellipse(img, (cx, robot_y + 5), (22, 8), 0, 0, 360, (190,190,190), -1)
    # Base
    cv2.ellipse(img, (cx, robot_y), (20, 9), 0, 0, 360, (220, 100, 255), -1)
    cv2.ellipse(img, (cx, robot_y), (20, 9), 0, 0, 360, (180, 60, 220), 2)
    # Corps
    cv2.rectangle(img, (cx-12, robot_y-28), (cx+12, robot_y), (220,100,255), -1)
    cv2.rectangle(img, (cx-12, robot_y-28), (cx+12, robot_y), (180,60,220), 1)
    # Tête
    cv2.circle(img, (cx, robot_y - 36), 12, (220,100,255), -1)
    cv2.circle(img, (cx, robot_y - 36), 12, (180,60,220), 2)
    # Yeux
    cv2.circle(img, (cx-4, robot_y-38), 3, (255,255,255), -1)
    cv2.circle(img, (cx+4, robot_y-38), 3, (255,255,255), -1)
    cv2.circle(img, (cx-4, robot_y-38), 1, (50,50,50), -1)
    cv2.circle(img, (cx+4, robot_y-38), 1, (50,50,50), -1)

def draw_scene(objects, size=700):
    # Fond blanc avec vignette grise sur les bords
    img = np.ones((size, size, 3), dtype=np.uint8) * 248

    # Vignette
    for i in range(60):
        alpha = int(i * 2.5)
        cv2.rectangle(img, (i, i), (size-i, size-i),
                      (248-alpha//4, 248-alpha//4, 248-alpha//4), 1)

    cx = size // 2
    robot_y = size - 80
    scale = 85

    # Sol (ellipse perspective)
    cv2.ellipse(img, (cx, robot_y - 10), (300, 140), 0, 0, 360, (238,238,238), -1)

    # Lignes de sol en perspective
    for i in range(-4, 5):
        bx = cx + i * 75
        hx = cx + i * 15
        cv2.line(img, (hx, int(robot_y - 280)), (bx, robot_y),
                 (225, 225, 225), 1)

    # Repères distance
    for d in [1, 2, 3, 4]:
        y = robot_y - int(d * scale * 0.55)
        spread = int(d * 75)
        if y > 0:
            cv2.ellipse(img, (cx, y), (spread, int(spread*0.3)),
                        0, 0, 360, (225, 225, 225), 1)
            cv2.putText(img, f"{d}m", (cx + spread + 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1)

    # Objets triés par distance (loin d'abord)
    objects_sorted = sorted(objects, key=lambda o: -o[2])

    for label, wx, wz, dist in objects_sorted:
        px, py = iso_project(wx, wz, cx, robot_y, scale)
        if not (15 < px < size-15 and 15 < py < robot_y - 5):
            continue
        obj_scale = max(0.4, min(1.4, 1.2 / max(dist, 0.5)))
        if label == "person":
            draw_person_3d(img, px, py, obj_scale)
        else:
            draw_object_3d(img, px, py, label, obj_scale * 0.8)
        # Label discret
        cv2.putText(img, label[:7], (px - 15, py - int(32*obj_scale) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150,150,150), 1)

    # Robot
    draw_reachy_3d(img, cx, robot_y)

    # HUD minimal
    n_persons = sum(1 for l,_,_,_ in objects if l=="person")
    ts = time.strftime("%H:%M:%S")
    cv2.putText(img, ts, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)
    cv2.putText(img, f"{n_persons} persons  |  {len(objects)} objects",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150,150,150), 1)

    return img

with ReachyMini(media_backend="default") as mini:
    print("World Model Tesla Style — 'q' pour quitter")
    while True:
        frame = mini.media.get_frame()
        if frame is None:
            continue
        results = model(frame, verbose=False)
        objects = []
        for box, cls in zip(results[0].boxes.xyxy.tolist(),
                            results[0].boxes.cls.tolist()):
            label = model.names[int(cls)]
            pos = box_to_world(box)
            if pos:
                wx, wz, dist = pos
                objects.append((label, wx, wz, dist))

        cv2.imshow("Camera", results[0].plot())
        cv2.imshow("World Model", draw_scene(objects))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
