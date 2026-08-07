import csv
import subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

print("Carregando modelo Danbooru Tagger...")

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# Download do modelo e CSV
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model.onnx",
    cache_dir="./model_cache"
)

csv_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="selected_tags.csv",
    cache_dir="./model_cache"
)

# Carrega tags + categoria
# 0 = general | 1 = character | 3 = rating | 4 = copyright
tags = []
categories = []

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tags.append(row['name'])
        categories.append(int(row['category']))

# Configurações do Personagem
CHARACTER_CATEGORY = 1  # 1 é a categoria correta para Personagem
CHARACTER_THRESHOLD = 0.50  # Probabilidade mínima para considerar

# Inicializa ONNX Runtime
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
target_size = 448

# Parâmetros de normalização ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_image(image_path, size=448):
    """Abre em RGB, redimensiona e aplica normalização padrão."""
    img = Image.open(image_path).convert("RGB")
    
    # Redimensiona mantendo proporção com padding
    img.thumbnail((size, size), Image.BICUBIC)
    new_img = Image.new("RGB", (size, size), (255, 255, 255))
    new_img.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    
    # Normalização
    arr = np.array(new_img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = np.expand_dims(arr, axis=0)
    return tensor

waifu_dir = Path("waifu")
renamed_count = 0

print("=" * 60)
print("Processando imagens...")
print("=" * 60)

for idx, image_file in enumerate(sorted(waifu_dir.glob("*")), 1):
    if not image_file.is_file():
        continue

    ext = image_file.suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        continue

    old_name = image_file.name
    print(f"[{idx}] {old_name}")

    try:
        tensor = preprocess_image(image_file, target_size)
        logits = session.run([output_name], {input_name: tensor})[0][0]
        probs = sigmoid(logits)  # Converte logits para probabilidades (0.0 a 1.0)

        # Filtra apenas tags da categoria Personagem (categoria 1) acima do threshold
        char_matches = [
            (tags[i], probs[i]) for i in range(len(tags))
            if categories[i] == CHARACTER_CATEGORY and probs[i] >= CHARACTER_THRESHOLD
        ]

        if not char_matches:
            print("    Nenhum personagem identificado acima do limite.")
            continue

        # Seleciona o personagem com maior probabilidade
        char_matches.sort(key=lambda x: x[1], reverse=True)
        best_character, score = char_matches[0]
        print(f"    Personagem detectado: {best_character} ({score:.2%})")

        # Limpa o nome para salvar no sistema de arquivos
        clean_name = "".join(c if c.isalnum() or c in ('-', '_') else '' for c in best_character)
        clean_name = clean_name[:50]

        if len(clean_name) < 2:
            print("    Nome limpo é inválido.")
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
        print(f"    Erro ao processar: {e}")
        continue

print("=" * 60)
print(f"Total renomeados com sucesso: {renamed_count}")
print("=" * 60)

# Commit Git
result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

if result.stdout.strip():
    subprocess.run(["git", "config", "user.name", "danbooru-bot"], check=True)
    subprocess.run(["git", "config", "user.email", "bot@github.com"], check=True)
    subprocess.run(["git", "commit", "-m", f"Danbooru Tagger: {renamed_count} personagens identificados"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("Commit e Push realizados!")
else:
    print("Nenhuma alteração a ser enviada.")
