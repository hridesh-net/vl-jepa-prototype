import cv2

def list_cameras(max_devices=10):
    print("Scanning cameras...")
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i}: AVAILABLE")
            cap.release()
        else:
            print(f"Camera index {i}: not available")

list_cameras()