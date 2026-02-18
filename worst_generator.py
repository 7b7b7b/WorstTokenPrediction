import random
import time
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
ASCII_SYMBOL_RE = re.compile(r"^[\s\[\]{}()<>\"'`~!@#$%^&*+=|\\,.;:?_\-\/]+$")


def has_any_digit(token: str) -> bool:
    # 覆盖 ASCII 与全角/其他 Unicode 数字。
    return any(ch.isdigit() for ch in token)


def build_weighted_prior_and_mask(
    tokenizer: AutoTokenizer, vocab_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # GPT2 字节级分词下，中文常拆成非 ASCII token。
    # 这里用 token 字面形式分层加权：
    # 1) 英文 token 直接禁用
    # 2) 数字 token 直接禁用（尽量避免数字污染）
    # 3) 非 ASCII token 高权重
    weights = torch.zeros(vocab_size, dtype=torch.float32, device=device)

    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id) or ""
        token_clean = token.replace("Ġ", "").replace("Ċ", "").replace("ĉ", "").strip()

        if has_any_digit(token_clean):
            weight = 0.0
        elif ASCII_LETTER_RE.search(token_clean):
            weight = 0.0
        elif token_clean == "":
            weight = 0.20
        elif any(ord(ch) > 127 for ch in token_clean):
            weight = 1.00
        elif ASCII_SYMBOL_RE.fullmatch(token_clean):
            weight = 0.20
        else:
            weight = 0.10

        weights[token_id] = weight

    allowed_mask = (weights > 0).float().unsqueeze(0)
    prior = weights.unsqueeze(0)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    return prior, allowed_mask


# =========================
# 随机种子（每次不同）
# =========================
seed = int(time.time() * 1000) % 1_000_000
print("Random Seed:", seed)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# 加载中文模型（使用 safetensors，规避低版本 torch 加载 .bin 限制）
# =========================
model_name = "IDEA-CCNL/Wenzhong-GPT2-110M"
model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.sep_token is not None:
        tokenizer.pad_token = tokenizer.sep_token

eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
model.eval()

# =========================
# Prompt
# =========================
prompt = "关于本文的方法创新，我们提出了"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# =========================
# 渐进崩塌参数（按你原方案：逐步向均匀分布靠拢）
# =========================
alpha_start = 0.0
alpha_max = 0.95          # 越接近 1，越接近完全均匀

# 长度控制：优先目标总长度，至少生成 min_new_tokens。
target_total_tokens = 360
min_new_tokens = 260
prompt_tokens = input_ids.size(1)
max_new_tokens = max(min_new_tokens, target_total_tokens - prompt_tokens)

# 崩塌调度：先预热，再缓慢增加 alpha。
collapse_warmup_ratio = 0.30
collapse_reach_alpha_max_ratio = 0.98
collapse_curve_power = 1.8  # >1 时前期更慢，后期再明显崩塌
collapse_warmup_steps = max(1, int(max_new_tokens * collapse_warmup_ratio))
collapse_reach_steps = max(2, int(max_new_tokens * collapse_reach_alpha_max_ratio))
collapse_steps = max(1, collapse_reach_steps - collapse_warmup_steps)

temperature = 1.0

entropy_values = []
vocab_size = model.get_output_embeddings().weight.size(0)
uniform_prior, allowed_mask = build_weighted_prior_and_mask(tokenizer, vocab_size, device)

# =========================
# 生成循环
# =========================
with torch.no_grad():
    for step in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :].float()

        # 原始概率
        probs = F.softmax(logits / temperature, dim=-1)
        effective_step = max(0, step - collapse_warmup_steps)
        progress = min(1.0, effective_step / collapse_steps)
        alpha = alpha_start + (alpha_max - alpha_start) * (progress ** collapse_curve_power)

        # 关键：向加权先验渐进（entropy 会逐步变高）
        mixed_probs = (1 - alpha) * probs + alpha * uniform_prior

        # 最终分布整体约束到允许集合，避免生成中混入英文词。
        mixed_probs = mixed_probs * allowed_mask

        if eos_id is not None:
            mixed_probs[:, eos_id] = 0.0

        total_prob = mixed_probs.sum(dim=-1, keepdim=True)
        if torch.any(total_prob <= 0):
            mixed_probs = probs * allowed_mask
            if eos_id is not None:
                mixed_probs[:, eos_id] = 0.0
            total_prob = mixed_probs.sum(dim=-1, keepdim=True)
        mixed_probs = mixed_probs / total_prob

        entropy_probs = mixed_probs.clamp_min(1e-12)
        entropy = -(entropy_probs * entropy_probs.log()).sum().item()
        entropy_values.append(entropy)

        next_token = torch.multinomial(mixed_probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

# =========================
# 输出文本
# =========================
output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

print("\n==== Generated Text ====\n")
try:
    print(output_text)
except UnicodeEncodeError:
    # Windows 控制台常见 gbk 编码下，随机乱码字符可能不可编码。
    encoding = sys.stdout.encoding or "utf-8"
    print(output_text.encode(encoding, errors="replace").decode(encoding, errors="replace"))

# =========================
# 绘制 entropy 曲线
# =========================
plt.figure()
plt.plot(entropy_values)
plt.xlabel("Generation Step")
plt.ylabel("Entropy")
plt.title("Entropy vs Generation Step (Gradual Uniform Collapse)")
plt.tight_layout()
plt.savefig("entropy_curve.png", dpi=150)
print("Saved entropy plot: entropy_curve.png")
plt.show()
