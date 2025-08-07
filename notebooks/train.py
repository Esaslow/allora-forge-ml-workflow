import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

class BasicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(BasicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def corr_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return np.corrcoef(y_true, y_pred)[0, 1]

def run_training(X_train, y_train, X_val, y_val, X_test, y_test=None):
    print("🧹 Preprocessing...")

    feature_cols = [f for f in list(X_train) if 'feature' in f]
    X_train_f = X_train[feature_cols].values.astype(np.float32)
    X_val_f = X_val[feature_cols].values.astype(np.float32)
    X_test_f = X_test[feature_cols].values.astype(np.float32)
    y_train_f = y_train.values.astype(np.float32).reshape(-1, 1)
    y_val_f = y_val.values.astype(np.float32).reshape(-1, 1)

    input_dim = X_train_f.shape[1]
    lr = 0.001
    epochs = 100
    batch_size = 128
    patience = 10
    device = torch.device("cpu")

    model = BasicMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_f), torch.tensor(y_train_f)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_f), torch.tensor(y_val_f)), batch_size=batch_size)

    best_model_state = None
    best_corr = -np.inf
    best_epoch = -1
    epochs_no_improve = 0

    print("🔁 Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(yb.numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        corr = corr_score(val_targets, val_preds)

        if corr > best_corr:
            best_corr = corr
            best_epoch = epoch
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            improved = "✅"
        else:
            epochs_no_improve += 1
            improved = "  "

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Val Corr: {corr:.5f} {improved}")

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}. Best Corr = {best_corr:.5f} (epoch {best_epoch+1})")
            break

    print(f"\n📦 Reloading best model from epoch {best_epoch+1}...")
    X_full = np.vstack([X_train_f, X_val_f])
    y_full = np.vstack([y_train_f, y_val_f])
    full_loader = DataLoader(TensorDataset(torch.tensor(X_full), torch.tensor(y_full)), batch_size=batch_size, shuffle=True)

    final_model = BasicMLP(input_dim).to(device)
    final_model.load_state_dict(best_model_state)
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    final_model.train()

    print("🔁 Fine-tuning on full dataset...")
    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in full_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = final_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Fine-tune Epoch {epoch+1:02d} | Loss: {total_loss / len(full_loader.dataset):.6f}")

    final_model.eval()
    with torch.no_grad():
        test_preds = final_model(torch.tensor(X_test_f).to(device)).cpu().numpy().flatten()
        test_preds = pd.Series(test_preds, index=X_test.index)

    print("\n✅ Prediction complete.")
    return test_preds
