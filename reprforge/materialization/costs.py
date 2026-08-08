"""Measured cost catalog shared by the compiler and workload replay."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostCatalog:
    raw_query_seconds: float
    feature_query_seconds: float
    feature_build_seconds: float
    feature_write_seconds: float
    feature_bytes: float
    retrieval_build_seconds: float
    retrieval_bytes: float
    feature_storage_seconds_per_byte: float = 0.0
    retrieval_storage_seconds_per_byte: float = 0.0

    def validate(self) -> None:
        values = self.__dict__
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.raw_query_seconds <= self.feature_query_seconds:
            raise ValueError("raw query cost must exceed feature query cost")
        if self.feature_build_seconds <= 0.0 or self.feature_bytes <= 0.0:
            raise ValueError("feature construction and bytes must be positive")
        if self.retrieval_build_seconds <= 0.0 or self.retrieval_bytes <= 0.0:
            raise ValueError("retrieval construction and bytes must be positive")

    @property
    def feature_saving_per_use(self) -> float:
        return self.raw_query_seconds - self.feature_query_seconds

    @property
    def offline_feature_cost(self) -> float:
        return (
            self.feature_build_seconds
            + self.feature_write_seconds
            + self.feature_storage_seconds_per_byte * self.feature_bytes
        )

    @property
    def retrieval_cost(self) -> float:
        return (
            self.retrieval_build_seconds
            + self.retrieval_storage_seconds_per_byte * self.retrieval_bytes
        )

    def feature_net_value(self, expected_future_uses: float) -> float:
        self.validate()
        if expected_future_uses < 0.0 or not math.isfinite(expected_future_uses):
            raise ValueError("expected_future_uses must be finite and nonnegative")
        return expected_future_uses * self.feature_saving_per_use - self.offline_feature_cost

    @property
    def feature_break_even_future_uses(self) -> float:
        self.validate()
        return self.offline_feature_cost / self.feature_saving_per_use
