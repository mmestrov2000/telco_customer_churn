import optuna
from src.train import train_model
from sklearn.metrics import roc_auc_score
import torch

def objective(trial):
    threshold = trial.suggest_float("threshold", 0.4, 0.6)

    config = {
        "model": {
            "input_dim": 28,  # update based on your data
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "max_epochs": 50,
            "patience": 5,
            "device": "cpu",
            "threshold": threshold,
            "seed": 42
        },
        "data": {
            "path": "data/processed/telco_churn_processed.csv"
        }
    }

    # Train model
    model, val_loader = train_model(config)

    # Evaluate
    model.eval()
    probs = []
    targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(config["model"]["device"]), yb.to(config["model"]["device"])
            pred_probs = model(xb).squeeze()
            probs.extend(pred_probs.cpu().numpy())
            targets.extend(yb.cpu().numpy())

    auc = roc_auc_score(targets, probs)  # ✅ AUC on probabilities (correct)
    return auc


def run_optimization():
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=40)
    print("Best Trial:", study.best_trial.params)
