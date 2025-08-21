import logging
import os
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    """Generic training helper for PyTorch models.

    The design mirrors :class:`tf_base_model.TFBaseModel` used in the original
    TensorFlow implementation.  It handles optimisation set-up, training and
    validation loops, early stopping with restarts and checkpointing.  Metrics
    can be supplied via a dictionary mapping names to callables that accept the
    model and a batch and return a scalar tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        num_training_steps: int = 20000,
        learning_rates: List[float] | None = None,
        batch_sizes: List[int] | None = None,
        patiences: List[int] | None = None,
        beta1_decays: List[float] | None = None,
        optimizer: str = "adam",
        grad_clip: float = 5.0,
        keep_prob: float = 1.0,
        enable_parameter_averaging: bool = False,
        warm_start_init_step: int = 0,
        log_interval: int = 20,
        min_steps_to_checkpoint: int = 100,
        loss_averaging_window: int = 100,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        metrics: Optional[Dict[str, Callable[[nn.Module, Iterable], float]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        learning_rates = learning_rates or [1e-3]
        batch_sizes = batch_sizes or [64]
        patiences = patiences or [3000]
        beta1_decays = beta1_decays or [0.9]
        assert len(learning_rates) == len(batch_sizes) == len(patiences) == len(beta1_decays)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_training_steps = num_training_steps
        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.patiences = patiences
        self.beta1_decays = beta1_decays
        self.optimizer_name = optimizer
        self.grad_clip = grad_clip
        self.keep_prob = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging
        self.warm_start_init_step = warm_start_init_step
        self.log_interval = log_interval
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.loss_averaging_window = loss_averaging_window
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.metrics = metrics or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.restart_idx = 0
        self.num_restarts = len(self.batch_sizes) - 1
        self.update_train_params()

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.enable_parameter_averaging:
            os.makedirs(self.checkpoint_dir + "_avg", exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "train.log")),
                logging.StreamHandler(),
            ],
        )

        self.model.to(self.device)
        self.optimizer = self._init_optimizer()
        if self.enable_parameter_averaging:
            self.averaged_params = [p.clone().detach() for p in self.model.parameters()]
        else:
            self.averaged_params = None

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    def update_train_params(self) -> None:
        self.batch_size = self.batch_sizes[self.restart_idx]
        self.learning_rate = self.learning_rates[self.restart_idx]
        self.beta1_decay = self.beta1_decays[self.restart_idx]
        self.early_stopping_steps = self.patiences[self.restart_idx]

    def _init_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        if self.optimizer_name.lower() == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, betas=(self.beta1_decay, 0.999))
        if self.optimizer_name.lower() == "rms":
            return torch.optim.RMSprop(params, lr=self.learning_rate)
        if self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate)
        raise ValueError(f"Unknown optimizer {self.optimizer_name}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _checkpoint_path(self, step: int, averaged: bool = False) -> str:
        directory = self.checkpoint_dir + ("_avg" if averaged else "")
        return os.path.join(directory, f"model-{step}.pt")

    def save(self, step: int, averaged: bool = False) -> None:
        path = self._checkpoint_path(step, averaged)
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step,
        }
        if averaged and self.averaged_params is not None:
            state["averaged_params"] = [p.cpu() for p in self.averaged_params]
        torch.save(state, path)
        logging.info("Saved checkpoint %s", path)

    def restore(self, step: int, averaged: bool = False) -> int:
        path = self._checkpoint_path(step, averaged)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if averaged and "averaged_params" in state:
            self.averaged_params = [p.to(self.device) for p in state["averaged_params"]]
        logging.info("Restored checkpoint %s", path)
        return state.get("step", step)

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        losses = []
        metric_sums: Dict[str, float] = {k: 0.0 for k in self.metrics}
        with torch.no_grad():
            for batch in self.val_loader:
                batch = [b.to(self.device) for b in batch]
                _, loss = self.model(*batch)
                losses.append(loss.item())
                for name, fn in self.metrics.items():
                    metric_sums[name] += float(fn(self.model, batch))
        mean_loss = sum(losses) / max(len(losses), 1)
        mean_metrics = {name: metric_sums[name] / max(len(losses), 1) for name in metric_sums}
        return mean_loss, mean_metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self) -> None:
        step = self.warm_start_init_step
        if step:
            self.restore(step)
        best_val_loss = float("inf")
        best_step = step
        train_iter = iter(self.train_loader)
        train_loss_history: deque[float] = deque(maxlen=self.loss_averaging_window)
        val_loss_history: deque[float] = deque(maxlen=self.loss_averaging_window)
        metric_histories: Dict[str, deque] = {
            name: deque(maxlen=self.loss_averaging_window) for name in self.metrics
        }

        while step < self.num_training_steps:
            self.model.train()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            batch = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            _, loss = self.model(*batch)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.enable_parameter_averaging and self.averaged_params is not None:
                with torch.no_grad():
                    for p, avg in zip(self.model.parameters(), self.averaged_params):
                        avg.mul_(0.999).add_(p.data, alpha=0.001)
            train_loss_history.append(loss.item())

            if step % self.log_interval == 0:
                val_loss, val_metrics = self.evaluate()
                val_loss_history.append(val_loss)
                for k, v in val_metrics.items():
                    metric_histories[k].append(v)
                train_loss = sum(train_loss_history) / max(len(train_loss_history), 1)
                logging.info(
                    "step %d train_loss=%.4f val_loss=%.4f %s",
                    step,
                    train_loss,
                    val_loss,
                    " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    if step >= self.min_steps_to_checkpoint:
                        self.save(step)
                        if self.enable_parameter_averaging:
                            self.save(step, averaged=True)
                elif step - best_step >= self.early_stopping_steps:
                    if self.restart_idx < self.num_restarts:
                        logging.info("Restarting from step %d with new hyper-parameters", best_step)
                        self.restart_idx += 1
                        self.update_train_params()
                        self.optimizer = self._init_optimizer()
                        self.restore(best_step)
                        train_iter = iter(self.train_loader)
                        step = best_step
                        continue
                    else:
                        logging.info("Early stopping at step %d", step)
                        break
            step += 1

        # save final state
        self.save(step)
        if self.enable_parameter_averaging:
            self.save(step, averaged=True)
