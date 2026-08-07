import os
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

print("Carregando modelo Danbooru Tagger...")

model_path = hf_hub_download(
    repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
    filename="model.onnx",
    cache_dir="./model_cache"
)

csv_path = hf_hub_download(
    repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
    filename="selected_tags.csv",
    cache_dir="./model_cache"
)

# Carrega tags + categoria (0 = geral, 4 = personagem, 9 = rating)
tags = []
categories = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tags.append(row['name'])
        categories.append(int(row['category']))

CHARACTER_CATEGORY = 4
CHARACTER_THRESHOLD = 0.5
GENERAL_THRESHOLD = 0.3

session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_shape = session.get_inputs()[0].shape
target_size = input_shape[1] if isinstance(input_shape[1], int) else 448

waifu_dir = Path("waifu")
renamed_count = 0

print("=" * 60)
print("Processando com Danbooru Tagger...")
print("=" * 60)


def preprocess(image_bgr, size):
    """Redimensiona com padding quadrado e monta o tensor NHWC em BGR
    (formato e ordem de cor que o modelo espera - sem converter pra RGB)."""
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = image_bgr

    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32)
    tensor = np.expand_dims(tensor, axis=0)  # (1, size, size, 3) - NHWC, BGR
    return tensor


for idx, image_file in enumerate(sorted(waifu_dir.glob("*")), 1):
    if not image_file.is_file():
        continue

    ext = image_file.suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        continue

    old_name = image_file.name
    print(f"[{idx}] {old_name}")

    try:
        image = cv2.imread(str(image_file))
        if image is None:
            print("    Nao foi possivel ler a imagem")
            continue

        tensor = preprocess(image, target_size)
        output = session.run([output_name], {input_name: tensor})[0][0]

        character_tags = [
            tags[i] for i in range(len(tags))
            if categories[i] == CHARACTER_CATEGORY and output[i] > CHARACTER_THRESHOLD
        ]

        character_name = None
        if character_tags:
            char_indices = [i for i in range(len(tags)) if categories[i] == CHARACTER_CATEGORY]
            best_idx = max(char_indices, key=lambda i: output[i])
            character_name = tags[best_idx]
            print(f"    Tags de personagem: {character_tags}")
        else:
            general_indices = [i for i in range(len(tags)) if categories[i] == 0]
            top_indices = sorted(general_indices, key=lambda i: output[i], reverse=True)[:5]
            for i in top_indices:
                if output[i] > GENERAL_THRESHOLD:
                    character_name = tags[i]
                    break

        if not character_name:
            print("    Nao identificado")
            continue

        clean_name = "".join(c if c.isalnum() or c in ('-', '_') else '' for c in character_name)
        clean_name = clean_name[:50]

        if len(clean_name) < 2:
            print("    Nome invalido")
            continue

        new_name = f"{clean_name}{ext}"
        new_path = waifu_dir / new_name

        if new_path.exists() and new_path != image_file:
            counter = 1
            while new_path.exists():
                new_name = f"{clean_name}_{counter}{ext}"
                new_path = waifu_dir / new_name
                counter += 1

        subprocess.run(["git", "mv", str(image_file), str(new_path)], check=True)
        print(f"    Renomeado para: {new_name}")
        renamed_count += 1

    except Exception as e:
        print(f"    Erro: {e}")
        continue

print("=" * 60)
print(f"Total renomeados: {renamed_count}")
print("=" * 60)

result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

if result.stdout.strip():
    subprocess.run(["git", "config", "user.name", "danbooru-bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@github.com"], check=True)
    subprocess.run(["git", "commit", "-m", f"Danbooru Tagger: {renamed_count} personagens identificados"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("Commit realizado!")
else:
    print("Nenhuma mudanca")
