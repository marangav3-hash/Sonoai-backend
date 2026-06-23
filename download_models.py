import os
from huggingface_hub import hf_hub_download

REPO_ID = "Sonoai-guy/sonoai-models"
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
os.makedirs(MODELS_DIR, exist_ok=True)

models = [
    "biometry_v1.pt",
    "emergency_flag_v1.pt",
]

for model in models:
    target = os.path.join(MODELS_DIR, model)
    if not os.path.exists(target):
        print(f"Downloading {model}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=model,
            local_dir=MODELS_DIR,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Downloaded {model} to {target}")
    else:
        print(f"{model} already exists, skipping.")

print("All models ready.")
