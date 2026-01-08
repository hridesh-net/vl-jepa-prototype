import cv2
import torch
from collections import deque
import torch.nn.functional as F

from src.video_stream import webcam_stream
from src.models.vision_encoder import VisionEncoder
from src.models.y_encoder import YEncoder
from src.models.predictor import Predictor
from src.models.query_encoder import QueryEncoder

embedding_buffer = deque(maxlen=5)

COLOR_STABLE = (200, 200, 200)   # light gray (calm)
COLOR_CHANGE = (0, 165, 255)     # orange (high contrast, readable)

# --------------------------------------------------
# Device selection (Mac safe)
# --------------------------------------------------
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
y_encoder = YEncoder()
q_encoder = QueryEncoder()
predictor = Predictor().to(device)
predictor.load_state_dict(torch.load("predictor.pt", map_location=device))
predictor.eval()

print("Loaded trained predictor ✅")

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

q_emb = q_encoder.encode([prompt]).to(device)
object_embs = y_encoder.encode(OBJECTS).to(device)

# --------------------------------------------------
# OpenCV Window
# --------------------------------------------------
WINDOW_NAME = "VL-JEPA – Live Perception"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# --------------------------------------------------
# Webcam loop
# --------------------------------------------------

last_label = None
last_embedding = None
text_color = COLOR_STABLE

CHANGE_THRESHOLD = 0.15  # semantic change sensitivity

# for img_tensor, frame in webcam_stream(device=device):
#     with torch.no_grad():
#         sv = vision(img_tensor)
#         sy_hat = predictor(sv, q_emb)

#         sims = torch.matmul(
#             F.normalize(sy_hat, dim=-1),
#             F.normalize(object_embs, dim=-1).T
#         )

#         best_idx = sims.argmax(dim=-1).item()
#         detected = OBJECTS[best_idx]

#     # --------------------------------------------------
#     # Overlay text on frame
#     # --------------------------------------------------
#     overlay_text = f"I see: {detected}"

#     cv2.putText(
#         frame,
#         overlay_text,
#         (20, 40),                     # position
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.0,                          # font scale
#         (0, 255, 0),                  # color (green)
#         2,                            # thickness
#         cv2.LINE_AA
#     )

#     # Show window
#     cv2.imshow(WINDOW_NAME, frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
changed = False
for img_tensor, frame in webcam_stream(device=device):
    with torch.no_grad():
        sv = vision(img_tensor)
        sy_hat = predictor(sv, q_emb)  # [1, D]

        # Smooth embedding over last few frames
        embedding_buffer.append(sy_hat)
        stable_emb = torch.mean(torch.stack(list(embedding_buffer)), dim=0)

        # Compare with previous embedding
        
        if last_embedding is not None:
            delta = 1 - F.cosine_similarity(stable_emb, last_embedding)
            if delta.item() > CHANGE_THRESHOLD:
                changed = True

        last_embedding = stable_emb.clone()

        # Object classification (argmax)
        sims = torch.matmul(
            F.normalize(stable_emb, dim=-1),
            F.normalize(object_embs, dim=-1).T
        )
        confidence = sims.max().item()
        

        best_idx = sims.argmax(dim=-1).item()
        current_label = OBJECTS[best_idx]
        print("Detected:", current_label, "| confidence:", f"{confidence:.2f}")

        # Detect semantic change
        if changed and current_label != last_label:
            overlay_text = f"New object: {current_label}"
            text_color = COLOR_CHANGE
            last_label = current_label
        else:
            overlay_text = f"I see: {last_label}"
            text_color = COLOR_STABLE

        # --------------------------------------------------
        # Draw overlay
        # --------------------------------------------------
        cv2.putText(
            frame,
            overlay_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow("VL-JEPA – Live Perception", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()