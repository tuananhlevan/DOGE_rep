import os
from typing import Tuple, Union
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
from transformers import CLIPVisionModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_mnist")
vision_model = model.vision_model
peft_model = PeftModel.from_pretrained(vision_model, "hoffman-lab/KnOTS-ViT-B-32_lora_R16_mnist", is_trainable=False)
peft_model = peft_model.merge_and_unload()
peft_model.to(device)

transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),  # 1 → 3 channels
    transforms.ToTensor(),
])

# --- dataset & dataloader ---
test_dataset = datasets.MNIST(
    root="./dataset",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

with torch.no_grad():
    correct = 0
    total = 0
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)

        features = vision_model(x).pooler_output
        logits = classifier_head(features)
        
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)
    print(correct / total)