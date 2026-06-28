import cv2
import numpy as np
import sqlite3
import json
import time

DB = "world_model.db"
W, H = 1200, 800

ROBOT_POSITIONS = {
    1: (-4, 4),
    2: (4, 4),
    3: (-4, -1),
    4: (4, -1),
}

ROBOT_COLORS = {
    1: (255, 100, 220),
    2: (100, 220, 255),
    3: (100, 255, 150),
    4: (255, 180, 80),
}

OBJ_COLORS = {
    "person":     (200, 230, 255),
    "chair":      (150, 150, 200),
    "laptop":     (100, 255, 180),
    "cup":        (255, 200, 100),
    "bottle":     (255, 130, 80),
    "cell phone": (180, 100, 255),
    "default":    (160, 160, 160),
}

def get_latest():
    try:
        conn = sqlite3.connect(DB)
        rows = conn.execute("""
            SELECT robot_id, objects FROM events
            WHERE timestamp > ?
            GROUP BY robot_id
            HAVING timestamp = MAX(timestamp)
        """, (time.time() - 5,)).fetchall()
        conn.close()
        return rows
    except:
        return []

def project(wx, wz, cx, base_y, scale=70):
    px = int(cx + wx * scale)
    py = int(base_y - wz * scale * 0.45)
    return px, py

def draw_person(img, px, py, s, color):
    s = max(0.5, s)
    # Ombre
    cv2.ellipse(img, (px+3, py+int(20*s)), (int(13*s), int(5*s)),
                0, 0, 360, (40,40,50), -1)
    # Jambes
    for dx in [-int(5*s), int(5*s)]:
        cv2.rectangle(img,
                      (px+dx-int(3*s), py+int(5*s)),
                      (px+dx+int(3*s), py+int(20*s)),
                      tuple(int(c*0.7) for c in color), -1)
    # Corps
    cv2.rectangle(img,
                  (px-int(11*s), py-int(18*s)),
                  (px+int(11*s), py+int(5*s)),
                  color, -1)
    # Tête
    cv2.circle(img, (px, py-int(27*s)), int(9*s), color, -1)
    # Yeux
    cv2.circle(img, (px-int(3*s), py-int(29*s)), int(2*s), (20,20,30), -1)
    cv2.circle(img, (px+int(3*s), py-int(29*s)), int(2*s), (20,20,30), -1)

def draw_box(img, px, py, s, color):
    s = max(0.4, s)
    w, h, d = int(16*s), int(12*s), int(7*s)
    # Ombre
    cv2.ellipse(img, (px+3, py+3), (w, int(h*0.3)),
                0, 0, 360, (40,40,50), -1)
    # Face avant
    cv2.fillPoly(img, [np.array([
        [px-w, py], [px+w, py],
        [px+w, py-h*2], [px-w, py-h*2]], np.int32)], color)
    # Dessus
    cv2.fillPoly(img, [np.array([
        [px-w, py-h*2], [px+w, py-h*2],
        [px+w+d, py-h*2-d], [px-w+d, py-h*2-d]], np.int32)],
        tuple(min(255, c+40) for c in color))
    # Côté
    cv2.fillPoly(img, [np.array([
        [px+w, py], [px+w+d, py-d],
        [px+w+d, py-h*2-d], [px+w, py-h*2]], np.int32)],
        tuple(max(0, c-40) for c in color))

def draw_robot(img, px, py, color, rid):
    # Ombre
    cv2.ellipse(img, (px+4, py+6), (22, 8), 0, 0, 360, (30,30,40), -1)
    # Base
    cv2.ellipse(img, (px, py), (20, 8), 0, 0, 360, color, -1)
    # Corps
    pts = np.array([[px-12,py],[px+12,py],[px+10,py-28],[px-10,py-28]], np.int32)
    cv2.fillPoly(img, [pts], color)
    # Tête
    cv2.circle(img, (px, py-38), 13, color, -1)
    cv2.circle(img, (px, py-38), 13, (255,255,255), 2)
    # Yeux
    cv2.circle(img, (px-4, py-40), 3, (255,255,255), -1)
    cv2.circle(img, (px+4, py-40), 3, (255,255,255), -1)
    cv2.circle(img, (px-4, py-40), 1, (20,20,20), -1)
    cv2.circle(img, (px+4, py-40), 1, (20,20,20), -1)
    # Label
    cv2.putText(img, f"R{rid}", (px-12, py+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,255,255), 1, cv2.LINE_AA)

def draw_scene(robot_data):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)  # fond sombre

    cx = W // 2
    base_y = H - 80
    scale = 70

    # Sol ellipse
    cv2.ellipse(img, (cx, base_y-20), (480, 200), 0, 0, 360, (28,28,38), -1)
    cv2.ellipse(img, (cx, base_y-20), (480, 200), 0, 0, 360, (45,45,60), 1)

    # Grille perspective
    for i in range(-6, 7):
        x1 = cx + i * 18
        x2 = int(cx + i * scale * 1.5)
        cv2.line(img, (x1, base_y-280), (x2, base_y), (35,35,48), 1)
    for d in [1, 2, 3, 4, 5]:
        y = base_y - int(d * scale * 0.45)
        sp = int(d * scale * 1.1)
        if y > 0:
            cv2.ellipse(img, (cx, y), (sp, int(sp*0.28)),
                        0, 0, 360, (40,40,55), 1)
            cv2.putText(img, f"{d}m", (cx+sp+8, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (80,80,100), 1, cv2.LINE_AA)

    # Scan lines effet
    for y in range(0, H, 4):
        cv2.line(img, (0, y), (W, y), (0,0,0), 1)
        img[y] = (img[y] * 0.92).astype(np.uint8)

    total_persons = 0
    total_objects = 0

    for robot_id, objects_json in robot_data:
        rx, rz = ROBOT_POSITIONS.get(robot_id, (0,0))
        color = ROBOT_COLORS.get(robot_id, (180,180,180))
        rpx, rpy = project(rx, rz, cx, base_y, scale)

        objects = json.loads(objects_json)
        total_objects += sum(objects.values())

        # Place objets autour du robot
        offset = 0
        for label, count in objects.items():
            obj_color = OBJ_COLORS.get(label, OBJ_COLORS["default"])
            for i in range(min(count, 5)):
                angle = np.radians(offset * 50 + i * 25)
                dist = 1.3 + i * 0.5
                ox = rx + dist * np.sin(angle)
                oz = rz + dist * np.cos(angle) * 0.6
                opx, opy = project(ox, oz, cx, base_y, scale)
                if 15 < opx < W-15 and 15 < opy < base_y:
                    obj_scale = 0.7
                    if label == "person":
                        draw_person(img, opx, opy, obj_scale, obj_color)
                        total_persons += 1
                    else:
                        draw_box(img, opx, opy, obj_scale * 0.7, obj_color)
                    cv2.putText(img, label[:6],
                                (opx-15, opy-int(38*obj_scale)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                (180,180,200), 1, cv2.LINE_AA)
            offset += 1

        draw_robot(img, rpx, rpy, color, robot_id)

    # HUD
    overlay = img.copy()
    cv2.rectangle(overlay, (12, 10), (340, 80), (25,25,35), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (12, 10), (340, 80), (60,60,90), 1)

    ts = time.strftime("%H:%M:%S")
    active = len(robot_data)
    cv2.putText(img, f"WORLD MODEL  {ts}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100,200,255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{active} robots  |  {total_persons} persons  |  {total_objects} objects",
                (20, 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (150,200,180), 1, cv2.LINE_AA)
    cv2.putText(img, "LIVE",
                (290, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80,255,120), 1, cv2.LINE_AA)
    # Point clignotant
    if int(time.time()) % 2 == 0:
        cv2.circle(img, (278, 28), 5, (80,255,120), -1)

    return img

print("World Model — 'q' pour quitter")
while True:
    data = get_latest()
    cv2.imshow("World Model", draw_scene(data))
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
