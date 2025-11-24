import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------- 配置区（你可以按需修改） ----------------
HD_PATH = "./instruction_hd.json"  # 上一步生成的文件

MODEL_FAMILY = "llamabase"
MODEL_TYPE = "7b"

INPUT_SIZE = 4096 * 2      # LLaMA-2-7B hidden size = 4096, 拼接两个向量
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OUT_DIR = f"./auto-labeled/output/{MODEL_FAMILY}{MODEL_TYPE}_instruction/train_log"
os.makedirs(OUT_DIR, exist_ok=True)
# --------------------------------------------------------


class InstDataset(Dataset):
    def __init__(self, feats, labels):
        self.X = feats
        self.y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    """
    与 detection_score.py 里的 Model 结构保持一致：
    dropout -> 256 -> 128 -> 64 -> 2
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    with open(HD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    feats = []
    labels = []

    for item in data:
        vec = item["hd_last_token"] + item["hd_last_mean"]
        feats.append(vec)
        labels.append(int(item["label"]))  # 0=normal, 1=conflict

    feats = torch.tensor(feats, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    print(f"[INFO] Loaded {feats.shape[0]} samples, feat_dim = {feats.shape[1]}")
    return feats, labels


def train():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    train_ds = InstDataset(X_train, y_train)
    val_ds = InstDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP(INPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_loss / len(train_ds)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)

        val_acc = correct / total if total > 0 else 0.0
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    save_path = os.path.join(OUT_DIR, "best_acc_model.pt")
    torch.save({"model_state_dict": best_state}, save_path)
    print(f"[INFO] Best val_acc = {best_acc:.4f}")
    print(f"[INFO] Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    train()
