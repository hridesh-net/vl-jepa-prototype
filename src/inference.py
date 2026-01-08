from video_stream import webcam_stream
from models.vision_encoder import VisionEncoder
from models.y_encoder import YEncoder
from models.predictor import Predictor
from models.query_encoder import QueryEncoder
import torch.nn.functional as F

import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# --------------------------------------------------
# Models
# --------------------------------------------------
vision = VisionEncoder(device=device)
y_encoder = YEncoder()          # keep text encoders on CPU (best on mac)
q_encoder = QueryEncoder()

predictor = Predictor().to(device).eval()

# --------------------------------------------------
# Object vocabulary
# --------------------------------------------------
OBJECTS = [
    "person",
    "laptop",
    "phone",
    "cup",
    "table",
    "chair",
    "keyboard",
    "screen",
    "book",
]

prompt = "What objects are visible?"

# Encode ONCE (important)
q_emb = q_encoder.encode([prompt]).to(device)
object_embs = y_encoder.encode(OBJECTS).to(device)

# --------------------------------------------------
# Webcam loop
# --------------------------------------------------
for img_tensor, frame in webcam_stream(device=device):
    with torch.no_grad():
        # img_tensor already on device
        sv = vision(img_tensor)                 # [1, T, D]
        sy_hat = predictor(sv, q_emb)           # [1, D]

        sims = torch.matmul(
            F.normalize(sy_hat, dim=-1),
            F.normalize(object_embs, dim=-1).T
        )  # [1, N]

        # topk = sims.topk(3, dim=-1)
        # detected = [OBJECTS[i] for i in topk.indices[0].tolist()]
        
        best_idx = sims.argmax(dim=1).item()
        detected = OBJECTS[best_idx]

    print("Detected:", detected)
