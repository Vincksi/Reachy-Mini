from reachy_mini import ReachyMini
import cv2

with ReachyMini(media_backend="default") as mini:
    print("Live camera - appuie sur 'q' pour quitter")
    
    while True:
        frame = mini.media.get_frame()
        if frame is not None:
            cv2.imshow("Reachy Mini - Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
