from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
import time
import os

os.makedirs("photos", exist_ok=True)

angles = []
for yaw in range(-25, 26, 5):      # -25 à +25 degrés
    for pitch in [-5, 0, 5]:        # 3 hauteurs
        angles.append((yaw, pitch))

with ReachyMini(media_backend="default") as mini:
    print(f"Capture de {len(angles)} photos...")
    for i, (yaw, pitch) in enumerate(angles):
        try:
            mini.goto_target(
                head=create_head_pose(x=pitch, y=yaw, mm=True),
                duration=1.0
            )
            time.sleep(1.2)
            frame = mini.media.get_frame()
            if frame is not None:
                path = f"photos/frame_{i:03d}_y{yaw}_p{pitch}.jpg"
                cv2.imwrite(path, frame)
                print(f"✅ {i+1}/{len(angles)} — {path}")
        except Exception as e:
            print(f"⚠️ Skip {yaw},{pitch}: {e}")

print(f"Done — {len(os.listdir('photos'))} photos dans ./photos/")
