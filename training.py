import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from src.models.vision_encoder import VisionEncoder
from src.models.predictor import Predictor
from src.models.query_encoder import QueryEncoder
from src.models.y_encoder import YEncoder

# --------------------------------------------------
# Device (Mac-safe)
# --------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Training on:", device)

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_DIR = "data/train"
EPOCHS = 10
LR = 1e-4
SAVE_PATH = "predictor.pt"

PROMPT = "What objects are visible?"

# --------------------------------------------------
# Image transform (CLIP compatible)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
])

# --------------------------------------------------
# Models
# --------------------------------------------------
vision = VisionEncoder(device=device)
predictor = Predictor().to(device).train()

q_encoder = QueryEncoder()
y_encoder = YEncoder()

optimizer = torch.optim.Adam(predictor.parameters(), lr=LR)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
samples = []
for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        samples.append((os.path.join(class_dir, fname), label))

print(f"Loaded {len(samples)} training samples")

# --------------------------------------------------
# Encode prompt ONCE
# --------------------------------------------------
q_emb = q_encoder.encode([PROMPT]).to(device)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
for epoch in range(EPOCHS):
    total_loss = 0.0

    for img_path, label in tqdm(samples, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        # Target embedding (teacher)
        target_emb = y_encoder.encode([label]).to(device)

        # Forward
        sv = vision(img)
        pred_emb = predictor(sv, q_emb)

        # Cosine loss
        loss = 1 - F.cosine_similarity(pred_emb, target_emb).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(samples)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# --------------------------------------------------
# Save predictor
# --------------------------------------------------
torch.save(predictor.state_dict(), SAVE_PATH)
print("Saved predictor to", SAVE_PATH)