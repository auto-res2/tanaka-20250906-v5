"""
evaluate.py – validation perplexity + lm-eval-harness glue
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import torch
from lm_eval import evaluator
from lm_eval.tasks import task_dict  # noqa: F401 – required by lm-eval

from preprocess import build_dataloader, load_tokenizer


@torch.no_grad()
def evaluate_model(model, tokenizer_path: Path, cfg) -> dict:  # noqa: D401 – simple util
    """1) validation perplexity (1M tokens)
    2) Sub-set of lm-eval-harness tasks.
    """
    tokenizer = load_tokenizer(Path(tokenizer_path))

    val_loader = build_dataloader(cfg.max_seq_len, batch_size_per_gpu=2, tokenizer_path=Path(tokenizer_path))
    n_tok, loss_sum = 0, 0.0
    for i, batch in enumerate(val_loader):
        if n_tok >= 1_000_000:
            break
        batch = batch.cuda() if torch.cuda.is_available() else batch
        out = model(batch, labels=batch)
        loss_sum += out.loss.item() * batch.numel()
        n_tok += batch.numel()
    ppl = math.exp(loss_sum / max(1, n_tok))

    eval_tasks: List[str] = ["mmlu", "hellaswag", "arc_easy", "winogrande"]
    results, _ = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        tasks=eval_tasks,
        bootstrap_iters=0,
        batch_size=2,
        max_batch_size=4,
    )
    acc = {t: results[t]["acc"] for t in eval_tasks}
    metrics = {"ppl": ppl, **acc}
    return metrics
