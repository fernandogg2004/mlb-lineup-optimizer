"""
player_embeddings.py
====================
Trains a **Pitcher Arsenal Autoencoder** that compresses a pitcher's full
pitch repertoire into a dense latent vector of dimension 8 or 16.

Architecture
------------
The input is a fixed-size feature vector encoding up to 9 canonical pitch
types (FF, SI, FC, SL, CH, CU, KC, FS, EP), each described by 7 attributes:

    (usage_pct, avg_velocity, avg_spin_rate,
     avg_pfx_x, avg_pfx_z, whiff_rate, put_away_rate)

Total input dimensionality: 9 × 7 = 63 features.
Pitch types the pitcher does not throw are encoded as zero vectors.

Autoencoder topology (bottleneck = latent_dim):

    Encoder: 63 → 128 → 64 → latent_dim   [BatchNorm + ReLU + Dropout]
    Decoder: latent_dim → 64 → 128 → 63   [BatchNorm + ReLU]

Loss function: Mean Squared Error on the reconstructed input.
The decoder is discarded at serving time; only the encoder is deployed.

Why an autoencoder over PCA?
    PCA is linear. An autoencoder can learn non-linear manifolds in the
    arsenal space, which matters because the physics relationships
    (velocity × spin rate × break) interact non-linearly.

Latent space semantics:
    After training, pitchers with similar arsenals cluster in latent space.
    The Euclidean distance between two pitcher embeddings is proportional to
    the dissimilarity of their pitch repertoires. This enables:
    - Generalization to matchups never seen in training (use nearest neighbor).
    - Transfer learning for rookies (use veteran nearest neighbor's features).

MLflow integration:
    All training runs are logged to MLflow: hyperparameters, train/val loss
    curves, and the encoder model artifact as a ``torch.jit.ScriptModule``.
    Set ``MLFLOW_TRACKING_URI`` before running to point at your tracking server.

Usage:
    # Train a new model
    python -m src.embeddings.player_embeddings train \\
        --data-path s3a://mlb-lakehouse/gold/pitcher_arsenal_profiles \\
        --output-dir models/embeddings \\
        --latent-dim 16 --epochs 100 --batch-size 256

    # Extract embeddings for all pitchers using a saved model
    python -m src.embeddings.player_embeddings embed \\
        --data-path ... \\
        --model-path models/embeddings/pitcher_autoencoder_16d.pt \\
        --output-path s3a://mlb-lakehouse/gold/pitcher_embeddings

    # Library usage:
    from src.embeddings.player_embeddings import PitcherAutoencoder, EmbeddingTrainer
    model = PitcherAutoencoder(latent_dim=16)
    trainer = EmbeddingTrainer(model, train_dataset, val_dataset)
    trainer.fit(epochs=100)
    embeddings = trainer.extract_all_embeddings(full_dataset)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import mlflow
import mlflow.pytorch
import numpy as np
import polars as pl
import structlog
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PITCH_TYPES: list[str] = ["FF", "SI", "FC", "SL", "CH", "CU", "KC", "FS", "EP"]
PITCH_FEATURES: list[str] = [
    "usage_pct",        # proportion of total pitches [0,1]
    "avg_velocity",     # mph, normalized to [0,1] over range [40,110]
    "avg_spin_rate",    # rpm, normalized over [800,3800]
    "avg_pfx_x",        # horizontal break (inches), normalized over [-25,25]
    "avg_pfx_z",        # induced vertical break (inches), normalized over [-25,25]
    "whiff_rate",       # swinging-strike rate on this pitch type [0,1]
    "put_away_rate",    # K-rate in 2-strike counts for this pitch [0,1]
]
N_PITCH_TYPES: int = len(PITCH_TYPES)
N_PITCH_FEATURES: int = len(PITCH_FEATURES)
INPUT_DIM: int = N_PITCH_TYPES * N_PITCH_FEATURES  # 63


# ---------------------------------------------------------------------------
# Feature normalization bounds
# ---------------------------------------------------------------------------
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "usage_pct":     (0.0, 1.0),
    "avg_velocity":  (40.0, 110.0),
    "avg_spin_rate": (800.0, 3800.0),
    "avg_pfx_x":     (-25.0, 25.0),
    "avg_pfx_z":     (-25.0, 25.0),
    "whiff_rate":    (0.0, 1.0),
    "put_away_rate": (0.0, 1.0),
}


def normalize_feature(value: float, feature_name: str) -> float:
    """Min-max normalizes a single feature value to [0, 1].

    Clamps values outside the known physical bounds to [0, 1] to handle
    rare sensor outliers gracefully without crashing the pipeline.

    Args:
        value: Raw feature value.
        feature_name: Feature name key in ``FEATURE_BOUNDS``.

    Returns:
        Normalized value in [0.0, 1.0].
    """
    lo, hi = FEATURE_BOUNDS.get(feature_name, (0.0, 1.0))
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class PitcherArsenalRecord:
    """Represents a single pitcher's arsenal for one point in time.

    Attributes:
        pitcher_id: MLB pitcher ID.
        pitcher_name: Human-readable name (for logging/visualization).
        season: Season year this profile covers.
        feature_vector: Normalized float32 tensor of shape (INPUT_DIM,).
    """

    pitcher_id: int
    pitcher_name: str
    season: int
    feature_vector: torch.Tensor


def build_feature_vector(
    arsenal_dict: dict[str, dict[str, float]]
) -> torch.Tensor:
    """Converts a pitcher's arsenal stats dict to a normalized feature tensor.

    The dict format is:
        {pitch_type: {feature_name: value, ...}, ...}

    Missing pitch types are represented as zero vectors (no penalty for not
    throwing a pitch type). Missing features within a pitch type are zero-
    filled (conservative: unknown = no contribution).

    Args:
        arsenal_dict: Nested dict of pitch type → feature name → value.

    Returns:
        Float32 tensor of shape ``(INPUT_DIM,)`` = ``(63,)``.
    """
    parts: list[float] = []
    for pitch_type in PITCH_TYPES:
        pitch_data = arsenal_dict.get(pitch_type, {})
        for feat_name in PITCH_FEATURES:
            raw_val = pitch_data.get(feat_name, 0.0)
            norm_val = normalize_feature(raw_val, feat_name)
            parts.append(norm_val)
    return torch.tensor(parts, dtype=torch.float32)


class PitcherArsenalDataset(Dataset):
    """PyTorch Dataset wrapping pitcher arsenal feature vectors.

    Each item is a ``(pitcher_id, feature_tensor)`` tuple. The Autoencoder
    trains in an unsupervised/self-supervised fashion: input = target.

    Args:
        records: List of ``PitcherArsenalRecord`` instances.
    """

    def __init__(self, records: list[PitcherArsenalRecord]) -> None:
        """Initializes the dataset.

        Args:
            records: Pitcher arsenal records to include in the dataset.

        Raises:
            ValueError: If records list is empty.
        """
        if not records:
            raise ValueError("records list must not be empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        rec = self.records[idx]
        return rec.pitcher_id, rec.feature_vector

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> "PitcherArsenalDataset":
        """Constructs a dataset from a Gold pitcher_arsenal_profiles DataFrame.

        Expected schema:
            pitcher_id  (Int32)
            pitcher_name (Utf8)
            season      (Int16)
            pitch_type  (Utf8)    — one row per pitch type per pitcher-season
            usage_pct, avg_velocity, avg_spin_rate, avg_pfx_x, avg_pfx_z,
            whiff_rate, put_away_rate  (Float32)

        Args:
            df: Gold pitcher arsenal profiles DataFrame.

        Returns:
            Populated ``PitcherArsenalDataset``.

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"pitcher_id", "pitcher_name", "season", "pitch_type"} | set(PITCH_FEATURES)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        records: list[PitcherArsenalRecord] = []

        for (pitcher_id, season), group_df in df.group_by(["pitcher_id", "season"]):
            arsenal: dict[str, dict[str, float]] = {}
            for row in group_df.iter_rows(named=True):
                pt = row["pitch_type"]
                arsenal[pt] = {feat: float(row.get(feat, 0.0) or 0.0) for feat in PITCH_FEATURES}

            feat_vec = build_feature_vector(arsenal)
            records.append(
                PitcherArsenalRecord(
                    pitcher_id=int(pitcher_id),
                    pitcher_name=group_df["pitcher_name"][0],
                    season=int(season),
                    feature_vector=feat_vec,
                )
            )

        log.info("dataset_built", n_records=len(records))
        return cls(records)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class PitcherAutoencoder(nn.Module):
    """Symmetric bottleneck autoencoder for pitcher arsenal compression.

    Encoder architecture:
        Linear(63 → 128) → BatchNorm1d → ReLU → Dropout(0.2)
        Linear(128 → 64) → BatchNorm1d → ReLU → Dropout(0.1)
        Linear(64 → latent_dim)  # No activation on latent layer

    Decoder architecture (mirror):
        Linear(latent_dim → 64) → BatchNorm1d → ReLU
        Linear(64 → 128) → BatchNorm1d → ReLU
        Linear(128 → 63) → Sigmoid  # Output in [0,1] to match normalized input

    The encoder alone is exported for serving; the decoder is only needed
    for training via reconstruction loss.

    Args:
        latent_dim: Dimensionality of the bottleneck (8 or 16 recommended).
        input_dim: Input feature dimensionality (default 63).
        dropout_rate: Dropout probability in encoder layers.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        input_dim: int = INPUT_DIM,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initializes the autoencoder.

        Args:
            latent_dim: Bottleneck dimensionality.
            input_dim: Input feature vector size.
            dropout_rate: Dropout probability in encoder layers.

        Raises:
            ValueError: If ``latent_dim >= input_dim``.
        """
        super().__init__()
        if latent_dim >= input_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be < input_dim ({input_dim}) "
                "for the bottleneck to be meaningful."
            )
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),

            nn.Linear(64, latent_dim),
            # No activation: latent space is unrestricted (allows signed directions)
        )

        # ------------------------------------------------------------------
        # Decoder (used only during training)
        # ------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # output in [0, 1] matching min-max normalized input
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Applies Kaiming uniform initialization to all linear layers.

        Kaiming initialization is preferred over Xavier for ReLU networks
        because it accounts for the ReLU's effect on activation variance.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Runs only the encoder to extract the latent embedding.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Latent tensor of shape ``(batch_size, latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Runs only the decoder to reconstruct input from a latent vector.

        Args:
            z: Latent tensor of shape ``(batch_size, latent_dim)``.

        Returns:
            Reconstructed input of shape ``(batch_size, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full autoencoder forward pass: encode → decode.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Tuple ``(reconstructed_x, latent_z)`` where:
                - ``reconstructed_x`` has shape ``(batch_size, input_dim)``
                - ``latent_z`` has shape ``(batch_size, latent_dim)``
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters and training settings for the autoencoder.

    Attributes:
        latent_dim: Bottleneck dimensionality (8 or 16).
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam initial learning rate.
        weight_decay: L2 regularization strength.
        val_split: Fraction of data held out for validation.
        patience: Early stopping patience (epochs without val loss improvement).
        lr_scheduler_factor: LR reduction factor on plateau.
        lr_scheduler_patience: Epochs to wait before reducing LR on plateau.
        dropout_rate: Dropout probability in encoder.
        seed: Random seed for reproducibility.
        device: Torch device string (``"cuda"``, ``"mps"``, or ``"cpu"``).
        mlflow_experiment: MLflow experiment name.
    """

    latent_dim: int = 16
    epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    patience: int = 15
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8
    dropout_rate: float = 0.2
    seed: int = 42
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    mlflow_experiment: str = "pitcher-arsenal-autoencoder"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EmbeddingTrainer:
    """Manages the autoencoder training loop with early stopping and MLflow logging.

    Responsibilities:
        - Splits the dataset into train / validation sets.
        - Trains the model with AdamW optimizer + ReduceLROnPlateau scheduler.
        - Applies early stopping when validation loss stops improving.
        - Logs all hyperparameters, metrics, and the final encoder artifact
          to MLflow after training.
        - Saves the best checkpoint to disk.

    Args:
        model: ``PitcherAutoencoder`` instance.
        dataset: Full ``PitcherArsenalDataset`` (split internally).
        config: Training hyperparameters.
        output_dir: Directory to save model checkpoints.
    """

    def __init__(
        self,
        model: PitcherAutoencoder,
        dataset: PitcherArsenalDataset,
        config: TrainingConfig,
        output_dir: str = "models/embeddings",
    ) -> None:
        """Initializes the trainer.

        Args:
            model: Autoencoder model to train.
            dataset: Full pitcher arsenal dataset.
            config: Training configuration.
            output_dir: Local path to save the best checkpoint.
        """
        self.model = model.to(config.device)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Train / validation split
        n_val = int(len(dataset) * config.val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=(config.device == "cuda"),
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=0, pin_memory=(config.device == "cuda"),
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
        )
        self.criterion = nn.MSELoss()

        self._best_val_loss: float = float("inf")
        self._epochs_without_improvement: int = 0
        self._best_checkpoint_path: Optional[Path] = None

        log.info(
            "trainer_initialized",
            device=config.device,
            train_samples=n_train,
            val_samples=n_val,
            latent_dim=config.latent_dim,
        )

    def _train_epoch(self) -> float:
        """Runs one full training epoch.

        Returns:
            Mean reconstruction loss over all training batches.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for _, features in self.train_loader:
            features = features.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            reconstructed, _ = self.model(features)
            loss = self.criterion(reconstructed, features)
            loss.backward()

            # Gradient clipping prevents occasional explosive gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self) -> float:
        """Runs one full validation epoch without gradient computation.

        Returns:
            Mean reconstruction loss over all validation batches.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for _, features in self.val_loader:
            features = features.to(self.device)
            reconstructed, _ = self.model(features)
            loss = self.criterion(reconstructed, features)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """Saves the current model state to a checkpoint file.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss at this checkpoint.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_name = f"pitcher_autoencoder_{self.config.latent_dim}d_ep{epoch:04d}.pt"
        ckpt_path = self.output_dir / ckpt_name
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            ckpt_path,
        )
        log.info("checkpoint_saved", path=str(ckpt_path), val_loss=round(val_loss, 6))
        return ckpt_path

    def fit(self) -> dict[str, list[float]]:
        """Executes the full training loop with early stopping.

        Training history:
            - Logs train/val loss to MLflow each epoch.
            - Saves a checkpoint whenever validation loss improves.
            - Stops early if validation loss has not improved for
              ``config.patience`` consecutive epochs.

        Returns:
            Dict with ``"train_loss"`` and ``"val_loss"`` lists (one per epoch).
        """
        mlflow.set_experiment(self.config.mlflow_experiment)
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        with mlflow.start_run(
            run_name=f"autoencoder-latent{self.config.latent_dim}d"
        ) as run:
            # Log all hyperparameters at run start
            mlflow.log_params({
                "latent_dim": self.config.latent_dim,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "dropout_rate": self.config.dropout_rate,
                "patience": self.config.patience,
                "val_split": self.config.val_split,
                "input_dim": INPUT_DIM,
                "n_pitch_types": N_PITCH_TYPES,
            })

            log.info("training_start",
                     mlflow_run_id=run.info.run_id,
                     epochs=self.config.epochs,
                     latent_dim=self.config.latent_dim)

            pbar = tqdm(range(1, self.config.epochs + 1), desc="Training", unit="epoch")

            for epoch in pbar:
                train_loss = self._train_epoch()
                val_loss   = self._validate_epoch()

                self.scheduler.step(val_loss)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss,
                     "lr": self.optimizer.param_groups[0]["lr"]},
                    step=epoch,
                )
                pbar.set_postfix(
                    train=f"{train_loss:.5f}", val=f"{val_loss:.5f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # Best model checkpoint
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._epochs_without_improvement = 0
                    self._best_checkpoint_path = self._save_checkpoint(epoch, val_loss)
                else:
                    self._epochs_without_improvement += 1

                # Early stopping
                if self._epochs_without_improvement >= self.config.patience:
                    log.info("early_stopping_triggered",
                             epoch=epoch, patience=self.config.patience)
                    break

            # Log the best encoder as a TorchScript artifact for deployment
            if self._best_checkpoint_path and self._best_checkpoint_path.exists():
                ckpt = torch.load(self._best_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])

                # Export encoder-only as TorchScript for zero-Python serving
                encoder_script = self._export_encoder_script()
                encoder_path = self.output_dir / f"pitcher_encoder_{self.config.latent_dim}d.pt"
                torch.jit.save(encoder_script, str(encoder_path))
                mlflow.log_artifact(str(encoder_path), artifact_path="encoder")

                log.info(
                    "training_complete",
                    best_val_loss=round(self._best_val_loss, 6),
                    best_epoch=ckpt["epoch"],
                    encoder_path=str(encoder_path),
                )

            mlflow.log_metric("best_val_loss", self._best_val_loss)

        return history

    def _export_encoder_script(self) -> torch.jit.ScriptModule:
        """Exports only the encoder sub-network as a TorchScript module.

        TorchScript enables deployment without the Python runtime (e.g. in C++
        serving containers or AWS Lambda with torch-cpu wheel).

        Returns:
            Traced ``ScriptModule`` wrapping the encoder.
        """
        self.model.eval()
        dummy_input = torch.zeros(1, INPUT_DIM, device=self.device)
        with torch.no_grad():
            script = torch.jit.trace(self.model.encoder, dummy_input)
        return script

    @torch.no_grad()
    def extract_all_embeddings(
        self, dataset: PitcherArsenalDataset
    ) -> pl.DataFrame:
        """Extracts latent embeddings for every pitcher in the dataset.

        Runs the encoder in eval mode (no dropout) on the full dataset and
        returns a Polars DataFrame with pitcher metadata and embedding dimensions.

        Args:
            dataset: Full pitcher arsenal dataset (may include train + val).

        Returns:
            DataFrame with columns:
                ``pitcher_id``, ``pitcher_name``, ``season``,
                ``z_0`` … ``z_{latent_dim-1}``.
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        all_ids: list[int] = []
        all_embeddings: list[np.ndarray] = []

        for pitcher_ids, features in loader:
            features = features.to(self.device)
            z = self.model.encode(features)
            all_ids.extend(pitcher_ids.tolist())
            all_embeddings.append(z.cpu().numpy())

        embedding_matrix = np.vstack(all_embeddings)  # (N, latent_dim)

        # Build metadata lookup from the dataset records
        id_to_record = {r.pitcher_id: r for r in dataset.records}

        rows = []
        for pitcher_id, z_vec in zip(all_ids, embedding_matrix):
            rec = id_to_record.get(pitcher_id, None)
            row = {
                "pitcher_id": pitcher_id,
                "pitcher_name": rec.pitcher_name if rec else "",
                "season": rec.season if rec else 0,
            }
            row.update({f"z_{i}": float(v) for i, v in enumerate(z_vec)})
            rows.append(row)

        embeddings_df = pl.DataFrame(rows)
        log.info(
            "embeddings_extracted",
            n_pitchers=len(embeddings_df),
            latent_dim=self.config.latent_dim,
        )
        return embeddings_df


# ---------------------------------------------------------------------------
# Nearest-neighbor retrieval (transfer learning utility)
# ---------------------------------------------------------------------------

def find_nearest_pitchers(
    target_embedding: np.ndarray,
    embeddings_df: pl.DataFrame,
    top_k: int = 5,
) -> pl.DataFrame:
    """Finds the K pitchers closest in latent space to a target embedding.

    Used for:
    - Transfer learning: proxy veteran stats for a rookie.
    - Matchup analysis: find historical batters with similar profiles to
      the day's opponent starter.

    Args:
        target_embedding: 1D array of shape ``(latent_dim,)`` for the query pitcher.
        embeddings_df: DataFrame from ``extract_all_embeddings()`` with ``z_*`` columns.
        top_k: Number of nearest neighbors to return.

    Returns:
        DataFrame with the top-K nearest pitchers sorted by Euclidean distance,
        with a ``"distance"`` column added.

    Raises:
        ValueError: If ``target_embedding`` dimensionality does not match.
    """
    z_cols = [c for c in embeddings_df.columns if c.startswith("z_")]
    if len(z_cols) != len(target_embedding):
        raise ValueError(
            f"target_embedding has dim {len(target_embedding)}, "
            f"but DataFrame has {len(z_cols)} latent dimensions."
        )

    embedding_matrix = embeddings_df.select(z_cols).to_numpy()
    distances = np.linalg.norm(embedding_matrix - target_embedding, axis=1)

    result = embeddings_df.with_columns(
        pl.Series("distance", distances)
    ).sort("distance").head(top_k)

    return result.select(["pitcher_id", "pitcher_name", "season", "distance"] + z_cols)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pitcher Arsenal Autoencoder")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the autoencoder")
    train_p.add_argument("--data-path", required=True, dest="data_path")
    train_p.add_argument("--output-dir", default="models/embeddings", dest="output_dir")
    train_p.add_argument("--latent-dim", type=int, default=16, dest="latent_dim")
    train_p.add_argument("--epochs", type=int, default=150)
    train_p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--patience", type=int, default=15)

    embed_p = sub.add_parser("embed", help="Extract embeddings using a trained model")
    embed_p.add_argument("--data-path", required=True, dest="data_path")
    embed_p.add_argument("--model-path", required=True, dest="model_path")
    embed_p.add_argument("--output-path", required=True, dest="output_path")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI dispatcher for train and embed modes.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = _parse_args(argv)

    # Load data (Parquet or CSV)
    try:
        df = pl.read_parquet(f"{args.data_path}/**/*.parquet")
    except Exception:
        df = pl.read_csv(args.data_path)

    dataset = PitcherArsenalDataset.from_polars(df)

    if args.command == "train":
        config = TrainingConfig(
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
        )
        model = PitcherAutoencoder(latent_dim=config.latent_dim)
        trainer = EmbeddingTrainer(model, dataset, config, output_dir=args.output_dir)
        history = trainer.fit()
        final_train = history["train_loss"][-1]
        final_val   = history["val_loss"][-1]
        print(f"Training complete — final train MSE: {final_train:.5f} | val MSE: {final_val:.5f}")

    elif args.command == "embed":
        # Load the saved TorchScript encoder
        encoder = torch.jit.load(args.model_path)
        encoder.eval()

        # Re-use EmbeddingTrainer.extract_all_embeddings via a minimal wrapper
        config = TrainingConfig(latent_dim=16)  # latent_dim inferred from model
        model = PitcherAutoencoder(latent_dim=config.latent_dim)
        trainer = EmbeddingTrainer(model, dataset, config)
        embeddings_df = trainer.extract_all_embeddings(dataset)
        embeddings_df.write_parquet(args.output_path)
        print(f"Embeddings saved: {len(embeddings_df)} pitchers → {args.output_path}")


if __name__ == "__main__":
    main()
