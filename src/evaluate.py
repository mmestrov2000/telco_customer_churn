from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import torch

def evaluate_model(model, val_loader, device="cpu", threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            probs = model(xb).squeeze()
            all_probs.extend(probs.cpu().numpy())  # Use for AUC
            preds = (probs > threshold).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, all_preds))
    print("AUC:", roc_auc_score(all_labels, all_probs))  # AUC on probabilities!
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
