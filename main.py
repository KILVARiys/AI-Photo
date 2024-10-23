from diffusers import AutoPipelineForText2Image
import torch

# Загрузка модели на процессор
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32
).to("cpu")  # Изменено с "cuda" на "cpu"

# Генерация изображения
image = pipeline(
    prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
).images[0]

# Сохранение изображения
image.save("FHOTO.png")