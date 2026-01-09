import cv2
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
])

def stream_video(source="webcam", path=None, camera_index=1, device="mps"):
    if source == "webcam":
        cap = cv2.VideoCapture(camera_index)
    elif source == "video":
        if path is None:
            raise ValueError("Video path must be provided for video source")
        cap = cv2.VideoCapture(path)
    else:
        raise ValueError("source must be 'webcam' or 'video'")

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)

        yield tensor, frame

    cap.release()