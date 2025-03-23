import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from src.model import LogisticRegressionModel
from sklearn.model_selection import train_test_split

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(config):
    set_seed(config["model"]["seed"])

    # Load and preprocess data
    df = pd.read_csv(config["data"]["path"])
    X = df.drop("Churn", axis=1).values.astype(np.float32)
    y = df["Churn"].values.astype(np.float32)

    # Train-test split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config["model"]["seed"], stratify=y
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["model"]["batch_size"])

    model = LogisticRegressionModel(input_dim=X.shape[1])
    model.to(config["model"]["device"])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["model"]["lr"],
        weight_decay=config["model"]["weight_decay"]
    )

    # --- Early Stopping Parameters ---
    max_epochs = config["model"].get("max_epochs", 50)
    patience = config["model"].get("patience", 5)
    best_auc = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(config["model"]["device"]), yb.to(config["model"]["device"])
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation AUC after each epoch ---
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(config["model"]["device"]), yb.to(config["model"]["device"])
                probs = model(xb).squeeze()
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(all_targets, all_preds)
        print(f"Epoch {epoch+1}/{max_epochs} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Feature importance
    weights = model.linear.weight.detach().cpu().numpy().flatten()
    columns = df.drop("Churn", axis=1).columns
    importance = pd.Series(weights, index=columns).sort_values(key=abs, ascending=False)
    print("Top Feature Weights (by magnitude):\n", importance.head(10))

    return model, val_loader
