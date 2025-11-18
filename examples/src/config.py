from dataclasses import dataclass


@dataclass
class DatasetConfig:
    DATA_DIR: str
    BATCH_SIZE: int = 64
    VALIDATION_SIZE: float = 0.2


@dataclass
class ModelConfig:
    EMBEDDING_DIMENSION: int
    PRETRAINED: bool = True


@dataclass
class HyperbolicConfig:
    CURVATURE: float
    TRAINABLE_CURVATURE: bool
    CURVATURE_LEARNING_RATE: float


@dataclass
class TrainingConfig:
    MAX_EPOCHS: int
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 10


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    hyperbolic: HyperbolicConfig
    training: TrainingConfig
