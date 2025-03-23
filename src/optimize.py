import optuna
from src.train import train_model
from sklearn.metrics import roc_auc_score
import torch

def objective(trial):
    threshold = trial.suggest_float("threshold", 0.4, 0.6)

    config = {
        "model": {
            "input_dim": 28,  # update based on your data
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_epochs": 100,
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
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            probs = model(xb).squeeze()
            predicted = (probs > threshold).int()
            preds.extend(predicted.numpy())
            targets.extend(yb.numpy())

    # You can switch this to recall_score, f1_score, etc. if needed
    auc = roc_auc_score(targets, preds)
    return auc


def run_optimization():
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=40)
    print("Best Trial:", study.best_trial.params)
