import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
from tqdm import tqdm

import torch
from utils.model import get_model  # 复用你现有的 get_model

# ---------------- 配置区（你可以按需修改） ----------------
model_type = "7b"
model_family = "llamabase"

# 指令数据所在目录，建议你建一个，例如 ./instructions
INSTRUCTION_DIR = "./instructions"

# 你准备好的指令数据文件名，可按需增减
INSTRUCTION_FILES = [
    "case_instruction.json",        # CAS 系列整理后
    "digits_instruction.json",      # DIG 系列
    "json_instruction.json",        # JSN 系列
    "language_instruction.json",    # LAN 系列
    "list_instruction.json",
    "permutation_case_instruction.json",
    "permutation_digits_instruction.json",
    "permutation_json_instruction.json",
    "permutation_language_instruction.json",
    "permutation_list_instruction.json",
    "permutation_quotepresence_instruction.json",
    "permutation_sentencecount_instruction.json",
    "permutation_wordlength_instruction.json",
    "quotepresence_instruction.json",
    "sentencecount_instruction.json",
    "wordlength_instruction.json"
]

# 输出文件
OUT_PATH = "./instruction_hd.json"
# --------------------------------------------------------


def build_prompt(item):
    """
    把 system + user + task 拼成一个简单的 prompt 文本
    You can change this template if you want a different style.
    """
    sys_msg = item.get("system_message", "").strip()
    user_msg = item.get("user_message", "").strip()
    task = item.get("task", "").strip()

    # 这里用一个非常直观的模板，你可以以后改成 chat 格式
    parts = []
    if sys_msg:
        parts.append(f"[SYSTEM] {sys_msg}")
    if user_msg:
        parts.append(f"[USER] {user_msg}")
    if task:
        parts.append(f"[TASK] {task}")
    prompt = "\n".join(parts)
    return prompt


def main():
    print(f"[INFO] loading model: {model_family}{model_type}")
    model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)
    model.eval()
    torch.set_grad_enabled(False)

    all_samples = []

    for fname in INSTRUCTION_FILES:
        path = os.path.join(INSTRUCTION_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] file not found, skip: {path}")
            continue

        print(f"[INFO] loading instruction file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in tqdm(data, desc=f"encode {fname}"):
            prompt = build_prompt(item)
            if not prompt.strip():
                continue

            inputs = tokenizer(prompt.strip(), return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            # hidden_states 是一个 list，[-1] 是最后一层，形状 [batch, seq_len, hidden_size]
            last_layer = outputs.hidden_states[-1][0]  # [seq_len, hidden_size]

            hd_last_token = last_layer[-1].detach().cpu().tolist()
            hd_last_mean = last_layer.mean(dim=0).detach().cpu().tolist()

            # 把文字标签映射成数字：normal -> 0, conflict -> 1
            label_str = item.get("label", "normal")
            label = 1 if label_str.lower() == "conflict" else 0

            all_samples.append(
                {
                    "id": item.get("id", ""),
                    "src_file": fname,
                    "label": label,
                    "hd_last_token": hd_last_token,
                    "hd_last_mean": hd_last_mean,
                }
            )

    print(f"[INFO] total samples: {len(all_samples)}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f)

    print(f"[INFO] saved hidden states to: {OUT_PATH}")


if __name__ == "__main__":
    main()
