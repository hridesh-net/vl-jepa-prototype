import argparse
import cv2
import torch
import torch.nn.functional as F
from collections import deque

from video_stream import stream_video
from models.vision_encoder import VisionEncoder
from models.predictor import Predictor
from models.query_encoder import QueryEncoder
from models.y_encoder import YEncoder

# --------------------------------------------------
# Scene description helper
# --------------------------------------------------
def describe_scene(current, previous, changed, prompt):
    if previous is None:
        return f"{prompt} \n I see {current}."
    if changed and current != previous:
        return f"A {current} has appeared in the scene."
    return f"{prompt} \n I still see {current}."

# --------------------------------------------------
# Main inference
# --------------------------------------------------
def run_inference(args):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    # Models
    vision = VisionEncoder(device=device)
    predictor = Predictor().to(device)
    predictor.load_state_dict(torch.load(args.weights, map_location=device))
    predictor.eval()

    q_encoder = QueryEncoder()
    y_encoder = YEncoder()

    # Objects
    OBJECTS = args.objects.split(",")

    prompt = "What objects are visible?"
    q_emb = q_encoder.encode([prompt]).to(device)
    object_embs = y_encoder.encode(OBJECTS).to(device)

    # State
    embedding_buffer = deque(maxlen=5)
    last_embedding = None
    previous_label = None

    TEXT_COLOR = (0, 0, 0)
    OUTLINE_COLOR = (255, 255, 255)
    BG_COLOR = (255, 255, 255)
    COLOR_CHANGE = (0, 165, 255)
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.2     # large
    THICKNESS = 5       # bold text
    OUTLINE_THICKNESS = 5
    PADDING = 14

    # Window
    window_name = "VL-JEPA – Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Stream
    for img_tensor, frame in stream_video(
        source=args.source,
        path=args.path,
        camera_index=args.camera_id,
        device=device
    ):
        with torch.no_grad():
            sv = vision(img_tensor)
            sy_hat = predictor(sv, q_emb)

            embedding_buffer.append(sy_hat)
            if len(embedding_buffer) < 2:
                stable_emb = sy_hat
            else:
                stable_emb = torch.mean(
                    torch.stack(list(embedding_buffer)), dim=0
                )

            changed = False
            if last_embedding is not None:
                delta = 1 - F.cosine_similarity(stable_emb, last_embedding)
                if delta.item() > args.change_threshold:
                    changed = True

            last_embedding = stable_emb.clone()

            sims = torch.matmul(
                F.normalize(stable_emb, dim=-1),
                F.normalize(object_embs, dim=-1).T
            )

            best_idx = sims.argmax(dim=-1).item()
            current_label = OBJECTS[best_idx]

            sentence = describe_scene(
                current=current_label,
                previous=previous_label,
                changed=changed,
                prompt=prompt
            )

            color = COLOR_CHANGE if changed else TEXT_COLOR
            previous_label = current_label

        
        h, w, _ = frame.shape

        (tw, th), baseline = cv2.getTextSize(
            sentence, FONT, FONT_SCALE, THICKNESS
        )

        x = 20
        y = 60
        
        cv2.rectangle(
            frame,
            (x - PADDING, y - th - PADDING),
            (x + tw + PADDING, y + baseline + PADDING),
            BG_COLOR,
            thickness=-1
        )
        
        
        # cv2.putText(
        #     frame,
        #     sentence,
        #     (20, 30),
        #     FONT,
        #     FONT_SCALE,
        #     OUTLINE_COLOR,
        #     OUTLINE_THICKNESS,
        #     cv2.LINE_AA
        # )
        
        # Main black text (draw on top)
        cv2.putText(
            frame,
            sentence,
            (20, 60),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            THICKNESS,
            cv2.LINE_AA
        )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["webcam", "video"], default="webcam")
    parser.add_argument("--path", type=str, help="Path to video file")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--weights", type=str, default="predictor.pt")
    parser.add_argument(
        "--objects",
        type=str,
        default="person,laptop,phone,cup,table,chair,ujjwal,vanshika,shakti"
    )
    parser.add_argument("--change-threshold", type=float, default=0.15)

    args = parser.parse_args()
    run_inference(args)