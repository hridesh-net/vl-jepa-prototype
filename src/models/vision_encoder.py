import torch
from transformers import CLIPVisionModel

class VisionEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="mps"):
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, images):
        return self.model(pixel_values=images).last_hidden_state