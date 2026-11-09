"""Alias module providing counterfactual_service for backwards compatibility."""

from app.services.counterfactual import (
    CounterfactualService,
    counterfactual_service,
)

__all__ = ["CounterfactualService", "counterfactual_service"]

