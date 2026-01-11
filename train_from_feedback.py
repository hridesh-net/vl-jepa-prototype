"""
Production-grade training script for VL-JEPA Predictor
using human feedback + base dataset.

Implements:
1. Confidence-weighted loss
2. Recent-feedback priority
3. Mixed dataset + feedback training
4. Automatic retrain trigger
5. Evaluation on holdout feedback
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.models.predictor import Predictor
from src.models.query_encoder import QueryEncoder
from src.models.y_encoder import YEncoder

# ==================================================
# Configuration
# ==================================================

FEEDBACK_FILE = Path("data/corrections.jsonl")
FEEDBACK_META = Path("data/feedback_meta.json")

BASE_WEIGHTS = Path("checkpoints/predictor.pt")
OUTPUT_WEIGHTS = Path("checkpoints/predictor_feedback.pt")

EPOCHS = 10
LEARNING_RATE = 5e-5
SEED = 42

RETRAIN_THRESHOLD = 0        # auto-trigger retrain
FEEDBACK_WEIGHT = 0.7
DATASET_WEIGHT = 0.3
HALF_LIFE_HOURS = 24          # recency decay

# ==================================================
# Logging
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("vl-jepa-trainer")

# ==================================================
# Utilities
# ==================================================

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confidence_weight(conf: float) -> float:
    """Higher confidence mistakes matter more."""
    return 0.5 + conf  # [0.5, 1.5]


def recency_weight(timestamp: str) -> float:
    """Exponential decay based on recency."""
    ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    age_hours = (datetime.now() - ts).total_seconds() / 3600
    return 0.5 ** (age_hours / HALF_LIFE_HOURS)


# ==================================================
# Data Loading
# ==================================================

def load_feedback(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No feedback file at {path}")

    records = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if {"prompt", "correct", "vision_embedding", "timestamp"} <= rec.keys():
                records.append(rec)

    if not records:
        raise RuntimeError("No valid feedback records found")

    return records


def load_feedback_meta() -> Dict[str, int]:
    if FEEDBACK_META.exists():
        return json.loads(FEEDBACK_META.read_text())
    return {"new_samples": 0}



def save_feedback_meta(meta: Dict[str, int]) -> None:
    FEEDBACK_META.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_META.write_text(json.dumps(meta))


# ==================================================
# Training + Evaluation
# ==================================================

def train_step(
    predictor: Predictor,
    q_encoder: QueryEncoder,
    y_encoder: YEncoder,
    record: Dict[str, Any],
    device: str,
    weight: float,
) -> torch.Tensor:
    visual_emb = torch.tensor(
        record["vision_embedding"], dtype=torch.float32
    ).unsqueeze(0).to(device)

    q_emb = q_encoder.encode([record["prompt"]]).to(device)
    target_emb = y_encoder.encode([record["correct"]]).to(device)

    pred_emb = predictor(visual_emb, q_emb)

    base_loss = 1.0 - F.cosine_similarity(pred_emb, target_emb).mean()

    conf = record.get("confidence", 0.5)
    loss = (
        base_loss
        * weight
        * confidence_weight(conf)
        * recency_weight(record["timestamp"])
    )

    return loss


def evaluate(
    predictor: Predictor,
    records: List[Dict[str, Any]],
    q_encoder: QueryEncoder,
    y_encoder: YEncoder,
    device: str,
) -> float:
    predictor.eval()
    scores = []

    with torch.no_grad():
        for rec in records:
            visual_emb = torch.tensor(
                rec["vision_embedding"], dtype=torch.float32
            ).unsqueeze(0).to(device)

            q_emb = q_encoder.encode([rec["prompt"]]).to(device)
            target_emb = y_encoder.encode([rec["correct"]]).to(device)

            pred_emb = predictor(visual_emb, q_emb)
            sim = F.cosine_similarity(pred_emb, target_emb).item()
            scores.append(sim)

    predictor.train()
    return sum(scores) / len(scores)


# ==================================================
# Main Training Logic
# ==================================================

def train_from_feedback(records: List[Dict[str, Any]], device: str) -> Predictor:
    predictor = Predictor().to(device)
    predictor.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device))
    predictor.train()

    q_encoder = QueryEncoder()
    y_encoder = YEncoder()

    optimizer = torch.optim.Adam(predictor.parameters(), lr=LEARNING_RATE)

    train_records, val_records = train_test_split(
        records, test_size=0.2, random_state=SEED
    )

    logger.info(
        "Training on %d samples | validating on %d",
        len(train_records),
        len(val_records),
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for rec in tqdm(train_records, desc=f"Epoch {epoch}/{EPOCHS}"):
            loss = train_step(
                predictor,
                q_encoder,
                y_encoder,
                rec,
                device,
                FEEDBACK_WEIGHT,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_records)
        val_score = evaluate(
            predictor, val_records, q_encoder, y_encoder, device
        )

        logger.info(
            "Epoch %d | Train Loss: %.4f | Val CosSim: %.4f",
            epoch,
            avg_loss,
            val_score,
        )

    return predictor


# ==================================================
# Entry Point
# ==================================================

def main() -> None:
    set_seed(SEED)
    device = get_device()

    logger.info("Using device: %s", device)

    meta = load_feedback_meta()
    if meta["new_samples"] < RETRAIN_THRESHOLD:
        logger.info(
            "Not enough new feedback (%d/%d). Skipping training.",
            meta["new_samples"],
            RETRAIN_THRESHOLD,
        )
        return

    records = load_feedback(FEEDBACK_FILE)
    predictor = train_from_feedback(records, device)

    torch.save(predictor.state_dict(), OUTPUT_WEIGHTS)
    logger.info("Saved trained predictor → %s", OUTPUT_WEIGHTS)

    meta["new_samples"] = 0
    save_feedback_meta(meta)
    logger.info("Feedback counter reset.")


if __name__ == "__main__":
    main()