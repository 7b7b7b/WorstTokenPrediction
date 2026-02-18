# Worst Token Prediction

An experiment script that gradually shifts generation from normal next-token prediction toward a worst-token style distribution, while tracking entropy growth and text collapse behavior.

## What This Repo Contains

- `worst_generator.py`: main experiment script

## Environment

- Python 3.10+ (recommended)
- Optional but recommended: CUDA GPU

Install dependencies:

```bash
pip install torch transformers matplotlib numpy
```

## Run

```bash
python worst_generator.py
```

The script will:

1. load `IDEA-CCNL/Wenzhong-GPT2-110M`
2. generate text with gradual collapse scheduling
3. print generated text
4. save `entropy_curve.png`

## Main Tunable Parameters

In `worst_generator.py`, adjust these variables:

- `prompt`: initial prompt text
- `target_total_tokens`, `min_new_tokens`: output length control
- `alpha_start`, `alpha_max`: collapse strength range
- `collapse_warmup_ratio`, `collapse_reach_alpha_max_ratio`, `collapse_curve_power`: collapse speed/schedule
- `repeat_penalty_start`, `repeat_penalty_base`, `repeat_hard_block_start`: repeated-character suppression

## Notes

- First run may take longer because model weights are downloaded.
- Behavior is intentionally unstable in later steps (semantic drift, gibberish, symbol noise).
