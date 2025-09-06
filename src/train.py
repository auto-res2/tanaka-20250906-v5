"""
train.py – contains model construction, RL-controller and the training
loop.  All heavy-lifting lives here so that the remaining modules can stay
light-weight and dependency-free.

NOTE
----
Only the logic that already existed in the original single-file reference
implementation is retained.  No new features have been added – the code is
merely reorganised so it can be imported as a proper Python module.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Tuple

import deepspeed  # noqa: F401 – heavy dependency but needed at runtime
import torch
from tqdm import tqdm

# Local modules (part of the STRICT 6-file setup)
from preprocess import build_dataloader
from evaluate import evaluate_model

# -----------------------------------------------------------------------------
# 1)  Configuration objects – identical to the original implementation
# -----------------------------------------------------------------------------


@dataclass
class OptimConfig:
    peak_lr: float = 2.4e-4
    warmup_steps: int = 3_750
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class PPOConfig:
    clip: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_env_steps: int = 2_048
    sgd_epochs: int = 3
    temperature: float = 1.0


@dataclass
class ExperimentConfig:
    name: str
    variant: str
    seed: int
    dataset_name: str
    dataset_url: str
    max_tokens: int
    max_pflops: float
    model_url: str
    start_n_layers: int
    d_model: int
    n_heads: int
    vocab_size: int = 32_000
    max_seq_len: int = 4_096
    deepspeed_config: str = "ds_config_zero3.json"
    ctrl_interval_tokens: int = 2_000_000
    optim: OptimConfig = field(default_factory=OptimConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        """Utility/helper so we can load configs directly from YAML."""
        d = d.copy()
        d["optim"] = OptimConfig(**d.get("optim", {}))
        d["ppo"] = PPOConfig(**d.get("ppo", {}))
        return ExperimentConfig(**d)

    # ------------------------------------------------------------------
    # A few convenient helpers
    # ------------------------------------------------------------------
    def asdict(self):  # noqa: D401 – short helper
        d = asdict(self)
        d["optim"] = asdict(self.optim)
        d["ppo"] = asdict(self.ppo)
        return d


# -----------------------------------------------------------------------------
# 2)  (Mini)-Model wrapper with depth-growth / seq-extension / compression hooks
# -----------------------------------------------------------------------------
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


def load_llama_model(model_name_or_path: str, device: torch.device) -> torch.nn.Module:  # noqa: D401
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model.to(device)
    return model


# ----------------------------  GROW ----------------------------

def apply_depth_growth(model: torch.nn.Module, factor: int) -> None:
    assert factor in (2, 4), "Growth factor must be 2 or 4"
    decoder = model.model.layers  # transformer blocks list for LLaMA/OPT style models
    current_layers = len(decoder)
    target_layers = int(current_layers * factor)
    last_block = decoder[-1]
    for _ in range(target_layers - current_layers):
        # Create a brand-new copy of the last block (weights re-initialised)
        cloned = type(last_block)(last_block.config).to(last_block.weight.dtype).to(last_block.weight.device)
        decoder.append(cloned)
    model.config.num_hidden_layers = target_layers
    print(f"[AutoSlim] GROW applied – depth {current_layers} → {target_layers}")


# -------------------------  EXTEND SEQ -------------------------

def apply_seq_extension(model: torch.nn.Module, new_len: int) -> None:
    current = model.config.max_position_embeddings
    if new_len <= current:
        return  # no-op
    model.config.max_position_embeddings = new_len
    print(f"[AutoSlim] ExtendSeq applied – max_pos {current} → {new_len}")


# -------------------------- COMPRESS ---------------------------
from peft import LoraConfig, get_peft_model  # noqa: E402


def apply_compression(model: torch.nn.Module, group_size: int) -> torch.nn.Module:
    lora_cfg = LoraConfig(r=group_size, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"[AutoSlim] Compress applied – LoRA group size {group_size}")
    return model


# -----------------------------------------------------------------------------
# 3)  RLlib PPO Controller (unchanged – merely inlined)
# -----------------------------------------------------------------------------
import numpy as np  # noqa: E402
import ray  # noqa: E402
from ray.rllib.algorithms import ppo  # noqa: E402
from ray.rllib.env.policy_server_base import PolicyServerBase  # noqa: E402
from ray.rllib.env import EnvContext  # noqa: E402

STATE_KEYS = [
    "step",
    "loss_slope",
    "grad_var",
    "gpu_mem",
    "flops_per_tok",
    "depth",
    "seq_len",
]
ACTION_SPACE = [
    ("grow", 2),
    ("grow", 4),
    ("seq", 512),
    ("seq", 1024),
    ("seq", 2048),
    ("seq", 4096),
    ("compress", 4),
    ("compress", 8),
    ("compress", 16),
]


class _SchedulerEnv(PolicyServerBase):
    """Tiny stub-env: exposes state → reward for RLlib PPO."""

    def __init__(self, ctx: EnvContext):
        super().__init__(ctx)
        self._state: Dict[str, float] = {k: 0.0 for k in STATE_KEYS}
        self.last_compute: float = 0.0
        self.last_loss: float = math.inf
        self.max_pflops: float = float(ctx.get("max_pflops", 1e9))

    def run(self):  # pragma: no cover – not used, exists to satisfy interface
        raise RuntimeError("_SchedulerEnv must be used with RLlib's in-proc usage, not policy-server mode.")


# ------------------------------------------------------------------

def _build_ppo_trainer(cfg: ExperimentConfig):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    env_conf = {"max_pflops": cfg.max_pflops}
    algo_conf = (
        ppo.PPOConfig()
        .environment(env=_SchedulerEnv, env_config=env_conf)
        .rollouts(num_rollout_workers=0)
        .training(
            gamma=cfg.ppo.gamma,
            lambda_=cfg.ppo.gae_lambda,
            lr=1e-4,
            grad_clip=None,
            train_batch_size=cfg.ppo.batch_env_steps,
            sgd_minibatch_size=512,
            num_sgd_iter=cfg.ppo.sgd_epochs,
            clip_param=cfg.ppo.clip,
        )
        .framework("torch")
    )
    return algo_conf.build()


class AutoSlimController:
    """Thin wrapper around RLlib PPO that maps states ↔ actions."""

    def __init__(self, cfg: ExperimentConfig):
        self.algo = _build_ppo_trainer(cfg)
        self.cfg = cfg

    def act(self, state: Dict[str, float]):
        obs = np.asarray([state[k] for k in STATE_KEYS], dtype=np.float32)
        action_idx, _, _ = self.algo.compute_single_action(obs)
        return ACTION_SPACE[action_idx]

    def record(self, state: Dict[str, float], reward: float, done: bool):
        obs = np.asarray([state[k] for k in STATE_KEYS], dtype=np.float32)
        self.algo.learn_on_batch({"obs": [obs], "rewards": [reward], "dones": [done]})


# -----------------------------------------------------------------------------
# 4)  Misc helpers
# -----------------------------------------------------------------------------

def _now() -> str:  # noqa: D401
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _save_json(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(data, fp, indent=2)


# -----------------------------------------------------------------------------
# 5)  Pre-trainer – the actual training loop (DeepSpeed)
# -----------------------------------------------------------------------------


class PreTrainer:  # noqa: D101 – docstring not vital for refactor
    def __init__(self, cfg: ExperimentConfig, tokenizer_path: Path):
        import yaml  # local import to keep global namespace minimal

        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer_path = tokenizer_path

        # -------------- build model & DeepSpeed engine --------------------
        self.model = load_llama_model(cfg.model_url, self.device)
        ds_cfg_path = Path(cfg.deepspeed_config)
        if not ds_cfg_path.exists():
            ds_cfg_path.write_text(
                json.dumps(
                    {
                        "train_batch_size": 16,
                        "train_micro_batch_size_per_gpu": 2,
                        "zero_optimization": {
                            "stage": 3,
                            "offload_optimizer": {"device": "none"},
                            "offload_param": {"device": "none"},
                        },
                        "bf16": {"enabled": True},
                    }
                )
            )
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model, model_parameters=self.model.parameters(), config=str(ds_cfg_path)
        )

        # -------------- controller (AutoSlim only) ------------------------
        self.controller = AutoSlimController(cfg) if cfg.variant.startswith("autoslim") else None

        # -------------- data-loader --------------------------------------
        self.dataloader = build_dataloader(cfg.max_seq_len, batch_size_per_gpu=2, tokenizer_path=tokenizer_path)
        self.data_iter = iter(self.dataloader)

        # -------------- trackers -----------------------------------------
        self.step = 0
        self.token_consumed = 0
        self.pflops_consumed = 0.0
        self.running_loss = 0.0

        torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch.to(self.device)

    def _compute_pflops(self, seq_len: int, batch_tokens: int) -> float:
        n_params = sum(p.numel() for p in self.model.parameters())
        return 2 * n_params * seq_len * batch_tokens / 1e15

    def _maybe_controller_step(self):
        if self.controller is None:
            return
        if self.token_consumed % self.cfg.ctrl_interval_tokens != 0:
            return

        # build observation
        grad_means = [p.grad.mean() for p in self.model.parameters() if p.grad is not None]
        grad_var = float(torch.var(torch.stack(grad_means))) if grad_means else 0.0
        state = {
            "step": self.step,
            "loss_slope": self.running_loss,
            "grad_var": grad_var,
            "gpu_mem": torch.cuda.max_memory_allocated() / 1e9,
            "flops_per_tok": self.pflops_consumed / max(1, self.token_consumed),
            "depth": self.model.config.num_hidden_layers,
            "seq_len": self.cfg.max_seq_len,
        }
        action_type, arg = self.controller.act(state)
        if action_type == "grow":
            apply_depth_growth(self.model, arg)
        elif action_type == "seq":
            self.cfg.max_seq_len = arg
            apply_seq_extension(self.model, arg)
            self.dataloader = build_dataloader(arg, batch_size_per_gpu=2, tokenizer_path=self.tokenizer_path)
            self.data_iter = iter(self.dataloader)
        elif action_type == "compress":
            self.model = apply_compression(self.model, arg)
        else:
            raise ValueError(action_type)

        # DeepSpeed needs to point to the updated model
        self.model_engine.module = self.model
        self.controller.record(state, reward=0.0, done=False)

    # ------------------------------------------------------------------
    def train(self):  # noqa: D401 – imperative method
        pbar = tqdm(total=self.cfg.max_tokens, desc=self.cfg.name)
        while self.token_consumed < self.cfg.max_tokens and self.pflops_consumed < self.cfg.max_pflops:
            batch = self._next_batch()
            loss = self.model_engine(batch, labels=batch).loss
            self.model_engine.backward(loss)
            self.model_engine.step()

            batch_tokens = batch.numel()
            seq_len = batch.shape[1]
            self.token_consumed += batch_tokens
            self.pflops_consumed += self._compute_pflops(seq_len, batch_tokens)
            self.step += 1
            self.running_loss = (
                0.95 * self.running_loss + 0.05 * loss.item() if self.step > 1 else loss.item()
            )

            self._maybe_controller_step()
            pbar.update(batch_tokens)
            pbar.set_postfix(loss=f"{self.running_loss:.3f}", pf=f"{self.pflops_consumed:.2f}")
        pbar.close()

        # ---------------- evaluation & persistence ---------------------
        metrics = evaluate_model(self.model, self.tokenizer_path, self.cfg)
        metrics.update(
            {
                "loss": self.running_loss,
                "pf_consumed": self.pflops_consumed,
                "tokens": self.token_consumed,
            }
        )
        out_path = Path(".research/iteration1") / f"{self.cfg.name}_{_now()}.json"
        _save_json(metrics, out_path)
        print("\n================ EXPERIMENT SUMMARY ================")
        print(json.dumps(metrics, indent=2))
        return metrics
