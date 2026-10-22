"""
Model-agnostic configuration system for agent tools
"""

from typing import Any, Dict, Optional


class CapacityTier:
    HIGH_CAPACITY = "high_capacity"
    ULTRA_CAPACITY = "ultra_capacity"
    CONSERVATIVE = "conservative"


DEFAULT_ANALYSIS_CONFIG = {"max_relevant_docs": 10, "max_chars": 10000, "max_per_source": 5}

DEFAULT_REPORT_CONFIG = {"max_content_length": 10000, "max_total_tokens": 120000}

MODEL_OPTIMIZATIONS = {
    CapacityTier.HIGH_CAPACITY: {
        "analysis": {
            "max_relevant_docs": 12,
            "max_chars": 8000,
        },
        "report": {"max_content_length": 8000, "max_total_tokens": 100000},
    },
    CapacityTier.ULTRA_CAPACITY: {
        "analysis": {
            "max_relevant_docs": 15,
            "max_chars": 10000,
        },
        "report": {"max_content_length": 10000, "max_total_tokens": 120000},
    },
    CapacityTier.CONSERVATIVE: {
        "analysis": {
            "max_relevant_docs": 6,
            "max_chars": 2000,
        },
        "report": {"max_content_length": 2000, "max_total_tokens": 25000},
    },
}


def get_analysis_config(capacity_tier: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """
    Get analysis tool configuration

    Args:
        capacity_tier: Optional tier from MODEL_OPTIMIZATIONS ("high_capacity", "ultra_capacity", "conservative")
        **overrides: Direct parameter overrides

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_ANALYSIS_CONFIG.copy()

    if capacity_tier and capacity_tier in MODEL_OPTIMIZATIONS:
        tier_config = MODEL_OPTIMIZATIONS[capacity_tier].get("analysis", {})
        config.update(tier_config)

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config


def get_report_config(capacity_tier: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """
    Get report assembler configuration

    Args:
        capacity_tier: Optional tier from MODEL_OPTIMIZATIONS ("high_capacity", "ultra_capacity", "conservative")
        **overrides: Direct parameter overrides

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_REPORT_CONFIG.copy()

    if capacity_tier and capacity_tier in MODEL_OPTIMIZATIONS:
        tier_config = MODEL_OPTIMIZATIONS[capacity_tier].get("report", {})
        config.update(tier_config)

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config
