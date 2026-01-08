import cv2
import torch
from torchvision import transforms
from torchvision.transforms import Normalize

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
])

def webcam_stream(device="mps"):
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        yield tensor, frame

    cap.release()