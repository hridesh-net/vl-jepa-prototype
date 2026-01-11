import os
import cv2
import torch
import json
from collections import deque
from datetime import datetime
import torch.nn.functional as F

from video_stream import stream_video
from models.vision_encoder import VisionEncoder
from models.predictor import Predictor
from models.query_encoder import QueryEncoder
from models.y_encoder import YEncoder

# --------------------------------------------------
# Config
# --------------------------------------------------
CORRECTIONS_FILE = "data/corrections.jsonl"
CHANGE_THRESHOLD = 0.15

TEXT_COLOR = (0, 0, 0)           # black
OUTLINE_COLOR = (255, 255, 255)  # white
BG_COLOR = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.6
THICKNESS = 4
OUTLINE_THICKNESS = 8
PADDING = 14

# --------------------------------------------------
# Utils
# --------------------------------------------------
def draw_centered_text(frame, text):
    h, w, _ = frame.shape
    (tw, th), baseline = cv2.getTextSize(
        text, FONT, FONT_SCALE, THICKNESS
    )

    x = (w - tw) // 2
    y = (h + th) // 2

    cv2.rectangle(
        frame,
        (x - PADDING, y - th - PADDING),
        (x + tw + PADDING, y + baseline + PADDING),
        BG_COLOR,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        FONT_SCALE,
        OUTLINE_COLOR,
        OUTLINE_THICKNESS,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        THICKNESS,
        cv2.LINE_AA
    )


def draw_prompt_text(frame, prompt):
    h, w, _ = frame.shape

    text = f"Prompt: {prompt}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 8

    (tw, th), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x = (w - tw) // 2
    y = 40  # top margin

    # background box
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        (0, 0, 0),
        -1
    )

    # text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def save_correction(
    prompt,
    predicted,
    correct,
    embedding,
    frame
):
    os.makedirs("data/feedback", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"data/feedback/frame_{timestamp}.jpg"

    # Save frame image
    cv2.imwrite(image_path, frame)

    record = {
        "prompt": prompt,
        "predicted": predicted,
        "correct": correct,
        "embedding": embedding.detach().cpu().tolist(),
        "image_path": image_path,
        "timestamp": timestamp
    }

    with open("data/corrections.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def describe_scene(current, previous, changed):
    if previous is None:
        return f"I see a {current}."
    if changed and current != previous:
        return f"A {current} has appeared in the scene."
    return f"I still see a {current}."


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
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
    predictor.load_state_dict(torch.load("predictor.pt", map_location=device))
    predictor.eval()

    q_encoder = QueryEncoder()
    y_encoder = YEncoder()

    OBJECTS = [
        "person", "cup", "laptop", "phone",
        "shakti", "ujjwal", "vanshika"
    ]

    object_embs = y_encoder.encode(OBJECTS).to(device)

    # Prompt (dynamic)
    current_prompt = "What objects are visible?"
    q_emb = q_encoder.encode([current_prompt]).to(device)

    print("Prompt:", current_prompt)
    print("Press 'p' → change prompt")
    print("Press 'c' → correct model")
    print("Press 'q' → quit")

    # State
    embedding_buffer = deque(maxlen=5)
    last_embedding = None
    previous_label = None
    display_sentence = "Detecting scene..."

    # Window
    window = "VL-JEPA – Interactive"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # --------------------------------------------------
    # Loop
    # --------------------------------------------------
    for img_tensor, frame in stream_video(device=device):
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
                if delta.item() > CHANGE_THRESHOLD:
                    changed = True

            last_embedding = stable_emb.clone()

            sims = torch.matmul(
                F.normalize(stable_emb, dim=-1),
                F.normalize(object_embs, dim=-1).T
            )

            best_idx = sims.argmax(dim=-1).item()
            current_label = OBJECTS[best_idx]

            display_sentence = describe_scene(
                current_label, previous_label, changed
            )

            if changed:
                previous_label = current_label

        draw_prompt_text(frame, current_prompt)
        draw_centered_text(frame, display_sentence)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF

        # Change prompt
        if key == ord("p"):
            new_prompt = input("\nEnter new prompt: ")
            current_prompt = new_prompt
            q_emb = q_encoder.encode([current_prompt]).to(device)
            print("Updated prompt:", current_prompt)

        # Correction
        if key == ord("c"):
            print("\nModel incorrect.")
            correct = input("Enter correct label: ")

            record = {
                "prompt": current_prompt,
                "predicted": current_label,
                "correct": correct,
                "embedding": stable_emb,
                "frame": frame
            }

            save_correction(**record)
            print("Correction saved.")

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()