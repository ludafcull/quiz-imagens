import csv
import subprocess
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

print("Carregando modelo Danbooru Tagger...")

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.onnx", cache_dir="./model_cache")
csv_path = hf_hub_download(repo_id=MODEL_REPO, filename="selected_tags.csv", cache_dir="./model_cache")

tags = []
categories = []

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tags.append(row['name'])
        categories.append(int(row['category']))

# Categoria 1 = Character (Personagem)
CHARACTER_CATEGORY = 1
# THRESHOLD MAIS BAIXO: Captura mais personagens sem perder muita precisão
CHARACTER_THRESHOLD = 0.22  

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
target_size = 448

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_image_exact(image_path, size=448):
    """Pré-processamento BGR/RGB padronizado para o modelo WD EVA-02."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Redimensiona mantendo a proporção com padding branco
    h, w = img.shape[:2]
    max_dim = max(h, w)
    padded = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
    
    # Centraliza a imagem
    dx = (max_dim - w) // 2
    dy = (max_dim - h) // 2
    padded[dy:dy+h, dx:dx+w] = img

    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)
    # Modelo espera RGB de 0.0 a 1.0 (float32)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32)
    return np.expand_dims(tensor, axis=0)

waifu_dir = Path("waifu")
renamed_count = 0

print("=" * 60)
print("Processando imagens com alta sensibilidade...")
print("=" * 60)

valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

for idx, image_file in enumerate(sorted(waifu_dir.glob("*")), 1):
    if not image_file.is_file() or image_file.suffix.lower() not in valid_exts:
        continue

    old_name = image_file.name
    print(f"[{idx}] {old_name}")

    try:
        tensor = preprocess_image_exact(image_file, target_size)
        if tensor is None:
            print("    Erro ao abrir imagem.")
            continue

        raw_output = session.run([output_name], {input_name: tensor})[0][0]

        if np.min(raw_output) < 0.0 or np.max(raw_output) > 1.0:
            probs = sigmoid(raw_output)
        else:
            probs = raw_output

        char_matches = [
            (tags[i], probs[i]) for i in range(len(tags))
            if categories[i] == CHARACTER_CATEGORY and probs[i] >= CHARACTER_THRESHOLD
        ]

        if not char_matches:
            print("    Nenhum personagem identificado acima do limite.")
            continue

        char_matches.sort(key=lambda x: x[1], reverse=True)
        best_character, score = char_matches[0]
        print(f"    -> Personagem: {best_character} ({score:.1%})")

        clean_name = "".join(c if c.isalnum() or c in ('-', '_') else '' for c in best_character)
        clean_name = clean_name[:50]

        if len(clean_name) < 2:
            continue

        ext = image_file.suffix.lower()
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

if renamed_count > 0:
    subprocess.run(["git", "config", "user.name", "danbooru-bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@github.com"], check=True)
    subprocess.run(["git", "commit", "-m", f"Danbooru Tagger: {renamed_count} personagens identificados"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("Commit realizado com sucesso!")
else:
    print("Nenhuma alteracao para enviar.")
