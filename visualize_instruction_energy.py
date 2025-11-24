import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU2（可换）

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ============== 1. 复用训练时 MLP 结构 ==============
class MLP(nn.Module):
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


class EnergyModel(nn.Module):
    def __init__(self, input_size, ckpt_path, device="cuda:0"):
        super().__init__()
        self.device = device
        self.mlp = MLP(input_size).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.mlp.load_state_dict(ckpt["model_state_dict"])
        self.mlp.eval()

    def energy(self, x_tensor):
        with torch.no_grad():
            logits = self.mlp(x_tensor)
            E = -torch.logsumexp(logits, dim=1)
        return E.cpu().numpy()


# ============== 2. 加载 instruction_hd.json ==============
def load_hd(path="./instruction_hd.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feats, labels = [], []
    for item in data:
        feats.append(item["hd_last_token"] + item["hd_last_mean"])
        labels.append(int(item["label"]))

    feats = torch.tensor(feats, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    print(f"[INFO] loaded feats: {feats.shape}, labels dist:", torch.bincount(labels))
    return feats, labels


# ============== 3. 可视化（直方图 + KDE） ==============
def visualize_energy(normal_E, conflict_E):
    plt.figure(figsize=(9, 5))

    # ------ KDE 曲线 ------
    xs = np.linspace(min(normal_E.min(), conflict_E.min()),
                     max(normal_E.max(), conflict_E.max()), 500)

    kde_normal = gaussian_kde(normal_E)
    kde_conflict = gaussian_kde(conflict_E)

    plt.plot(xs, kde_normal(xs), label="Normal KDE", color="blue")
    plt.plot(xs, kde_conflict(xs), label="Conflict KDE", color="red")

    # ------ 直方图 ------
    plt.hist(normal_E, bins=40, alpha=0.3, density=True, color="blue")
    plt.hist(conflict_E, bins=40, alpha=0.3, density=True, color="red")

    plt.legend()
    plt.xlabel("Energy E = -logsumexp(logits)")
    plt.ylabel("Density")
    plt.title("Energy Distribution (KDE + Histogram)")

    plt.tight_layout()
    plt.savefig("instruction_energy_kde.png", dpi=300)
    plt.show()
    print("📌 KDE 图已保存为 instruction_energy_kde.png")


# ============== 4. 主程序 ==============
if __name__ == "__main__":
    HD_PATH = "./instruction_hd.json"
    CKPT_PATH = "./auto-labeled/output/llamabase7b_instruction/train_log/best_acc_model.pt"
    INPUT_SIZE = 4096 * 2

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    feats, labels = load_hd(HD_PATH)
    model = EnergyModel(INPUT_SIZE, CKPT_PATH, device=device)

    normal_feats = feats[labels == 0].to(device)
    conflict_feats = feats[labels == 1].to(device)

    print("[INFO] normal samples:", normal_feats.shape[0])
    print("[INFO] conflict samples:", conflict_feats.shape[0])

    normal_E = model.energy(normal_feats)
    conflict_E = model.energy(conflict_feats)

    visualize_energy(normal_E, conflict_E)
