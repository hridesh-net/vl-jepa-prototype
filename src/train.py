from models.vision_encoder import VisionEncoder
from models.y_encoder import YEncoder
from models.predictor import Predictor
from models.loss import info_nce

import torch
from torch.optim import Adam

vision = VisionEncoder()
y_enc = YEncoder()
predictor = Predictor().cuda()

optimizer = Adam(predictor.parameters(), lr=1e-4)

for step in range(1000):
    images = torch.randn(8, 3, 224, 224).cuda()
    questions = ["What is happening?"] * 8
    answers = ["A person is cooking food"] * 8

    sv = vision(images)
    q_emb = y_enc.encode(questions).cuda()
    sy = y_enc.encode(answers).cuda()

    sy_hat = predictor(sv, q_emb)
    loss = info_nce(sy_hat, sy)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step} | loss {loss.item():.4f}")