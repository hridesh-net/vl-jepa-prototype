# Real Time Trainable Architecture.

```bash
┌──────────────────────────────────────────────────────────┐
│                      REAL-TIME INFERENCE                 │
└──────────────────────────────────────────────────────────┘

        Live Video / Stream / File
                    │
                    ▼
        ┌─────────────────────────┐
        │     Vision Encoder       │  (frozen)
        │  (CLIP / ViT backbone)   │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Pooled Vision Embedding │
        │        sv = mean(T)      │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │        Predictor         │  ← trainable
        │   fθ(sv, prompt_emb)    │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Semantic Embedding     │
        │      (latent belief)     │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Similarity Matching    │
        │  vs Text Embeddings      │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Scene Description / UI  │
        └─────────────────────────┘


┌──────────────────────────────────────────────────────────┐
│                 HUMAN-IN-THE-LOOP FEEDBACK               │
└──────────────────────────────────────────────────────────┘

        User observes output
                    │
                    │  (press 'c')
                    ▼
        ┌─────────────────────────┐
        │   Human Correction       │
        │  "This is Hridesh"       │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Feedback Capture        │
        │  - prompt               │
        │  - correct label        │
        │  - vision embedding     │
        │  - raw frame            │
        │  - timestamp            │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  corrections.jsonl      │
        │  feedback_meta.json     │
        │  (new_samples += 1)     │
        └─────────────────────────┘


┌──────────────────────────────────────────────────────────┐
│               AUTOMATED TRAINING PIPELINE                │
└──────────────────────────────────────────────────────────┘

        feedback_meta.json
                    │
        new_samples >= THRESHOLD ?
                    │
           ┌────────┴────────┐
           │                 │
          NO                YES
           │                 │
           ▼                 ▼
   ┌─────────────┐   ┌──────────────────────┐
   │  Skip train │   │  Load feedback data   │
   └─────────────┘   │  (train / val split) │
                      └──────────────────────┘
                                │
                                ▼
                      ┌──────────────────────┐
                      │  Predictor Training  │
                      │  - confidence weight │
                      │  - recency decay     │
                      │  - mixed data        │
                      └──────────────────────┘
                                │
                                ▼
                      ┌──────────────────────┐
                      │  Validation (holdout)│
                      │  cosine similarity   │
                      └──────────────────────┘
                                │
                                ▼
                      ┌──────────────────────┐
                      │ Save new checkpoint  │
                      │ predictor_feedback  │
                      └──────────────────────┘
                                │
                                ▼
                      feedback_meta.json
                      new_samples = 0


┌──────────────────────────────────────────────────────────┐
│                 CONTINUOUS LEARNING LOOP                 │
└──────────────────────────────────────────────────────────┘

   New checkpoint loaded at inference startup
                    │
                    ▼
           System gets better over time

```

## Key Design Principles (encoded in this diagram)
- Vision encoder is frozen (no catastrophic forgetting)
- Predictor is the only learned component
- Learning happens asynchronously
- Inference is never blocked by training
- Human feedback is first-class data
- VL-JEPA semantics live in embedding space