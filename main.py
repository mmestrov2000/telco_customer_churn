from src.train import train_model
from src.evaluate import evaluate_model
from src.optimize import run_optimization
import yaml
import torch
import shutil
import boto3
import os

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Train and evaluate
model, val_loader = train_model(config)
evaluate_model(model, val_loader, device=config["model"]["device"], threshold=config["model"]["threshold"])

# Save model locally
local_model_dir = "models/logistic_regression_v1"
os.makedirs(local_model_dir, exist_ok=True)

model_path = f"{local_model_dir}/logistic_regression_v1.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved locally at: {model_path}")

config_copy_path = f"{local_model_dir}/logistic_regression_v1_config.yaml"
shutil.copy("config/config.yaml", config_copy_path)
print(f"✅ Config copied to: {config_copy_path}")

# 🪣 Upload to S3
bucket_name = "telco-model-bucket-marin"
s3_prefix = "logistic_regression_v1/"  # Optional folder inside the bucket

s3 = boto3.client("s3")

# Upload model
s3.upload_file(model_path, bucket_name, f"{s3_prefix}logistic_regression_v1.pth")
print(f"📤 Model uploaded to s3://{bucket_name}/{s3_prefix}logistic_regression_v1.pth")

# Upload config
s3.upload_file(config_copy_path, bucket_name, f"{s3_prefix}logistic_regression_v1_config.yaml")
print(f"📤 Config uploaded to s3://{bucket_name}/{s3_prefix}logistic_regression_v1_config.yaml")

# Uncomment to re-run hyperparameter tuning
# run_optimization()