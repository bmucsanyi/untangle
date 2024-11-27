"""Checkpoint saver.

Tracks top-n training checkpoints.
"""

import logging
import operator
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CheckpointSaver:
    """Checkpoint saver that tracks top-n training checkpoints.

    Args:
        model: The model to save checkpoints for.
        optimizer: The optimizer to save state for.
        amp_scaler: The amp scaler to save state for, if used.
        decreasing: If True, a lower metric is better.
        max_history: Maximum number of checkpoints to keep.
        checkpoint_dir: Directory to save checkpoints in.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        amp_scaler: torch.cuda.amp.GradScaler | None,
        decreasing: bool,
        max_history: int,
        checkpoint_dir: Path,
    ) -> None:
        # Objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.amp_scaler = amp_scaler

        # State
        # (filename, metric) tuples in order of decreasing performance
        self.checkpoint_files = []
        self.best_epoch = None
        self.best_metric = None

        # Config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = "checkpoint"
        self.extension = "pt"
        self.decreasing = decreasing  # A lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt
        self.max_history = max_history

    def save_checkpoint(self, epoch: int, metric: float) -> None:
        """Save a checkpoint.

        Args:
            epoch: The current epoch number.
            metric: The metric value for this checkpoint.
        """
        # Save as last checkpoint
        tmp_filename = f"{self.checkpoint_prefix}_tmp.{self.extension}"
        last_filename = f"{self.checkpoint_prefix}_last.{self.extension}"
        tmp_save_path = self.checkpoint_dir / tmp_filename
        last_save_path = self.checkpoint_dir / last_filename
        self._save(tmp_save_path, epoch, metric)

        last_save_path.unlink(missing_ok=True)
        tmp_save_path.rename(last_save_path)

        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None

        if len(self.checkpoint_files) < self.max_history or self.cmp(
            metric, worst_file[1]
        ):
            # Save as among-the-best checkpoint
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            top_filename = f"{self.checkpoint_prefix}_{epoch}.{self.extension}"
            top_save_path = self.checkpoint_dir / top_filename
            top_save_path.hardlink_to(last_save_path)
            self.checkpoint_files.append((top_save_path, metric))
            self.checkpoint_files.sort(
                key=operator.itemgetter(1), reverse=not self.decreasing
            )

            checkpoints_str = "Current checkpoints:\n"
            for checkpoint_path, checkpoint_metric in self.checkpoint_files:
                checkpoints_str += f"\t{checkpoint_path}: {checkpoint_metric:.4f}\n"
            logger.info(checkpoints_str)

            if self.best_metric is None or self.cmp(metric, self.best_metric):
                self.best_epoch = epoch
                self.best_metric = metric
                best_filename = f"{self.checkpoint_prefix}_best.{self.extension}"
                best_save_path = self.checkpoint_dir / best_filename
                best_save_path.unlink(missing_ok=True)
                best_save_path.hardlink_to(last_save_path)

    def _save(self, save_path: Path, epoch: int, metric: float) -> None:
        """Internal method to save the checkpoint.

        Args:
            save_path: Path to save the checkpoint to.
            epoch: The current epoch number.
            metric: The metric value for this checkpoint.
        """
        save_state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metric": metric,
        }

        if self.amp_scaler is not None:
            save_state["amp_scaler"] = self.amp_scaler.state_dict()

        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, num_checkpoints_to_delete: int) -> None:
        """Clean up old checkpoints.

        Args:
            num_checkpoints_to_delete: Number of checkpoints to remove.
        """
        start_delete_index = self.max_history - num_checkpoints_to_delete

        if start_delete_index < 0 or len(self.checkpoint_files) <= start_delete_index:
            return

        checkpoints_to_delete = self.checkpoint_files[start_delete_index:]

        for checkpoint_path, checkpoint_metric in checkpoints_to_delete:
            logger.info(f"Cleaning checkpoint: {checkpoint_path}: {checkpoint_metric}")
            checkpoint_path.unlink()

        self.checkpoint_files = self.checkpoint_files[:start_delete_index]
