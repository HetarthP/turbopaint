"""Backend-independent image generation types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Inputs that must remain identical when comparing backends."""

    prompt: str
    width: int = 512
    height: int = 512
    inference_steps: int = 1
    guidance_scale: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be divisible by 8")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


class InferenceBackend(ABC):
    """Stable API for PyTorch, TensorRT, and optimized future backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique backend implementation name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier used by this backend."""

    @abstractmethod
    def generate(self, config: GenerationConfig) -> Any:
        """Generate one image from a fully specified configuration."""

