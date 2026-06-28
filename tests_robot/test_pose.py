from reachy_mini import ReachyMini
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")

with ReachyMini(media_backend="default") as mini:
    while True:
        frame = mini.media.get_frame()
        if frame is not None:
            results = model(frame, verbose=False)
            
            # Affiche tous les keypoints
            if results[0].keypoints is not None:
                kps = results[0].keypoints.xy.tolist()
                for i, person in enumerate(kps):
                    print(f"Personne {i}:")
                    print(f"  Poignet gauche (9): {person[9] if len(person)>9 else 'N/A'}")
                    print(f"  Poignet droit (10): {person[10] if len(person)>10 else 'N/A'}")
            else:
                print("Aucun keypoint détecté")
            
            # Visualise
            annotated = results[0].plot()
            cv2.imshow("Pose", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
