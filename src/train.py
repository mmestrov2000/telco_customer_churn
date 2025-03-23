import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config["model"]["seed"], stratify=y)


    # Convert to TensorDataset
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["model"]["batch_size"])

    model = LogisticRegressionModel(input_dim=X.shape[1])
    model.to(config["model"]["device"])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["lr"], weight_decay=config["model"]["weight_decay"])

    for epoch in range(config["model"]["num_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(config["model"]["device"]), yb.to(config["model"]["device"])
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    weights = model.linear.weight.detach().cpu().numpy().flatten()
    columns = pd.read_csv(config["data"]["path"]).drop("Churn", axis=1).columns
    importance = pd.Series(weights, index=columns).sort_values(key=abs, ascending=False)
    print("Top Feature Weights (by magnitude):\n", importance.head(10))


    return model, val_loader