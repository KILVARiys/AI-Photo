from diffusers import AutoPipelineForText2Image
import torch
import os
# Импортируем модуль для генерации уникальных идентификаторов
import uuid

# Загрузка модели на процессор
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32
).to("cpu")  # Изменено с "cuda" на "cpu"

# Генерация изображения
prompt = input('Enter the text from which the picture will be created (Example - Astronaut in the jungle, cold color palette, muted colors, detail, 8k):\n')

image = pipeline(
    prompt=prompt,
    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
).images[0]

# Папка загрузок
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

# Генерация случайного названия файла
random_filename = f"{uuid.uuid4()}.png"
image_path = os.path.join(downloads_folder, random_filename)

# Сохранение изображения
image.save(image_path)

# Пишем путь сохранения файла
print(f"Image saved to {image_path}")
