"""Orthogonal page representation states and monotone v0 transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MaterializationAction(str, Enum):
    FEATURE = "materialize_feature"
    RETRIEVAL = "materialize_retrieval"


@dataclass(frozen=True)
class PageState:
    """A page may independently support exact reuse and corpus discovery."""

    has_feature: bool = False
    has_retrieval: bool = False

    def apply(self, action: MaterializationAction) -> "PageState":
        if action is MaterializationAction.FEATURE:
            return PageState(has_feature=True, has_retrieval=self.has_retrieval)
        if action is MaterializationAction.RETRIEVAL:
            return PageState(has_feature=self.has_feature, has_retrieval=True)
        raise ValueError(f"unsupported materialization action: {action}")

    @property
    def label(self) -> str:
        if self.has_feature and self.has_retrieval:
            return "feature+retrieval"
        if self.has_feature:
            return "feature"
        if self.has_retrieval:
            return "retrieval"
        return "raw"
