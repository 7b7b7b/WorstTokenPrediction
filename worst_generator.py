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


def first_visible_char(text: str) -> str:
    for ch in text:
        if not ch.isspace():
            return ch
    return ""


def update_run_state(last_char: str, run_len: int, fragment: str) -> tuple[str, int]:
    # 按生成后的可见字符更新“连续重复字符”状态。
    # 空白字符会打断连续计数。
    for ch in fragment:
        if ch.isspace():
            last_char = ""
            run_len = 0
            continue
        if ch == last_char:
            run_len += 1
        else:
            last_char = ch
            run_len = 1
    return last_char, run_len


def max_visible_run(text: str) -> int:
    max_run = 0
    prev = ""
    cur = 0
    for ch in text:
        if ch.isspace():
            prev = ""
            cur = 0
            continue
        if ch == prev:
            cur += 1
        else:
            prev = ch
            cur = 1
        if cur > max_run:
            max_run = cur
    return max_run


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
        token_decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        if has_any_digit(token_clean) or has_any_digit(token_decoded):
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


def build_repeat_start_index(
    tokenizer: AutoTokenizer, vocab_size: int, device: str
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    # 预先建立“token 首个可见字符 -> token id 列表”，
    # 便于在生成时高效施加连续字符惩罚。
    char_to_ids: dict[str, list[int]] = {}
    char_to_prefix_runs: dict[str, list[int]] = {}
    token_run_penalty = torch.ones(vocab_size, dtype=torch.float32, device=device)
    for token_id in range(vocab_size):
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        text = token_text.lstrip()
        if not text:
            continue

        ch = text[0]
        prefix_run = 1
        while prefix_run < len(text) and text[prefix_run] == ch:
            prefix_run += 1

        char_to_ids.setdefault(ch, []).append(token_id)
        char_to_prefix_runs.setdefault(ch, []).append(prefix_run)

        token_max_run = max_visible_run(text)
        if token_max_run >= 8:
            token_run_penalty[token_id] = 0.0
        elif token_max_run >= 6:
            token_run_penalty[token_id] = 0.005
        elif token_max_run == 5:
            token_run_penalty[token_id] = 0.04
        elif token_max_run == 4:
            token_run_penalty[token_id] = 0.15

    repeat_start_meta = {
        ch: (
            torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(char_to_prefix_runs[ch], dtype=torch.long, device=device),
        )
        for ch, ids in char_to_ids.items()
    }
    return repeat_start_meta, token_run_penalty.unsqueeze(0)


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

# 连续字符惩罚：当同一字符连续出现过长时，显著降低“继续同字符”的概率。
repeat_penalty_start = 3
repeat_penalty_base = 0.06
repeat_hard_block_start = 6

entropy_values = []
vocab_size = model.get_output_embeddings().weight.size(0)
uniform_prior, allowed_mask = build_weighted_prior_and_mask(tokenizer, vocab_size, device)
repeat_start_meta, token_run_penalty = build_repeat_start_index(
    tokenizer, vocab_size, device
)
last_char = ""
last_char_run_len = 0

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
        mixed_probs = mixed_probs * token_run_penalty

        if last_char_run_len >= repeat_penalty_start and last_char in repeat_start_meta:
            penalty_ids, prefix_runs = repeat_start_meta[last_char]
            projected_runs = last_char_run_len + prefix_runs

            hard_mask = projected_runs >= repeat_hard_block_start
            if torch.any(hard_mask):
                mixed_probs[:, penalty_ids[hard_mask]] = 0.0

            soft_mask = ~hard_mask
            if torch.any(soft_mask):
                exponents = (
                    projected_runs[soft_mask] - repeat_penalty_start + 1
                ).clamp_min(1)
                penalty_scales = repeat_penalty_base ** exponents.float()
                mixed_probs[:, penalty_ids[soft_mask]] *= penalty_scales.unsqueeze(0)

        if eos_id is not None:
            mixed_probs[:, eos_id] = 0.0

        total_prob = mixed_probs.sum(dim=-1, keepdim=True)
        if torch.any(total_prob <= 0):
            mixed_probs = probs * allowed_mask
            mixed_probs = mixed_probs * token_run_penalty
            if eos_id is not None:
                mixed_probs[:, eos_id] = 0.0
            total_prob = mixed_probs.sum(dim=-1, keepdim=True)
        mixed_probs = mixed_probs / total_prob

        entropy_probs = mixed_probs.clamp_min(1e-12)
        entropy = -(entropy_probs * entropy_probs.log()).sum().item()
        entropy_values.append(entropy)

        next_token = torch.multinomial(mixed_probs, num_samples=1)
        next_fragment = tokenizer.decode(
            next_token[0].tolist(), clean_up_tokenization_spaces=False
        )
        last_char, last_char_run_len = update_run_state(
            last_char, last_char_run_len, next_fragment
        )
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
steps = np.arange(len(entropy_values))
entropy_arr = np.asarray(entropy_values, dtype=np.float32)
if entropy_arr.size >= 7:
    kernel = np.ones(7, dtype=np.float32) / 7.0
    smooth_entropy = np.convolve(entropy_arr, kernel, mode="same")
else:
    smooth_entropy = entropy_arr

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=150)
ax.axvspan(0, collapse_warmup_steps, color="#cfe8ff", alpha=0.35, label="Warmup")
ax.axvspan(
    collapse_warmup_steps,
    min(len(entropy_values), collapse_reach_steps),
    color="#ffe0b2",
    alpha=0.30,
    label="Collapse ramp",
)
if collapse_reach_steps < len(entropy_values):
    ax.axvspan(
        collapse_reach_steps,
        len(entropy_values),
        color="#ffd6d6",
        alpha=0.24,
        label="Saturated collapse",
    )

ax.plot(steps, entropy_arr, color="#8aa5c2", linewidth=1.2, alpha=0.55, label="Raw entropy")
ax.plot(
    steps,
    smooth_entropy,
    color="#1f5aa6",
    linewidth=2.4,
    label="Smoothed entropy",
)
if smooth_entropy.size > 0:
    ax.fill_between(steps, smooth_entropy, smooth_entropy.min(), color="#1f5aa6", alpha=0.10)

ax.set_xlabel("Generation step")
ax.set_ylabel("Entropy")
ax.set_title("Entropy Trajectory Under Gradual Worst-Token Collapse", pad=10)
ax.legend(loc="upper left", fontsize=8, frameon=True)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
fig.tight_layout()
fig.savefig("entropy_curve.png", dpi=220, bbox_inches="tight")
print("Saved entropy plot: entropy_curve.png")
plt.close(fig)
