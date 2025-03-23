from src.train import train_model
from src.evaluate import evaluate_model
from src.optimize import run_optimization
import yaml
import torch
import shutil

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Train and evaluate
model, val_loader = train_model(config)
evaluate_model(model, val_loader, device=config["model"]["device"], threshold=config["model"]["threshold"])

# Save model weights
model_path = "models/logistic_regression_v1/logistic_regression_v1.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

# Save the config alongside the model
config_copy_path = "models/logistic_regression_v1/logistic_regression_v1_config.yaml"
shutil.copy("config/config.yaml", config_copy_path)
print(f"✅ Config saved to {config_copy_path}")

# Uncomment to re-run hyperparameter tuning
# run_optimization()