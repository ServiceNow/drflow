"""
DRFlow Metrics - Evaluation metrics for DRBench Workflow evaluation.

Available metrics:
    - factuality: Verifies workflow step descriptions against source documents
    - insight_recall: Checks if golden steps are covered by predicted steps
    - insight_precision: Checks if predicted steps match golden steps
    - topology_order: Validates prerequisite ordering of matched steps
    - condition_resolvement: Evaluates proper resolution of conditional paths
    - personalized_comprehensiveness: Assesses detail and personalization level
"""

from drflow.metrics_drflow.factuality import WorkflowFactuality
from drflow.metrics_drflow.insight_recall import WorkflowInsightRecall
from drflow.metrics_drflow.insight_precision import WorkflowInsightPrecision
from drflow.metrics_drflow.topology_order import WorkflowTopologyOrder
from drflow.metrics_drflow.condition_resolvement import (
    WorkflowConditionResolvement,
)
from drflow.metrics_drflow.personalized_comprehensiveness import (
    WorkflowPersonalizedComprehensiveness,
)


DRFLOW_METRICS = {
    "factuality": WorkflowFactuality,
    "insight_recall": WorkflowInsightRecall,
    "insight_precision": WorkflowInsightPrecision,
    "topology_order": WorkflowTopologyOrder,
    "condition_resolvement": WorkflowConditionResolvement,
    "personalized_comprehensiveness": WorkflowPersonalizedComprehensiveness,
}

# Metrics that must run before others (dependency ordering)
# insight_precision must run before topology_order, condition_resolvement,
# and personalized_comprehensiveness because they need precision_matches.
METRIC_DEPENDENCIES = {
    "topology_order": ["insight_precision"],
    "condition_resolvement": ["insight_precision"],
    "personalized_comprehensiveness": ["insight_precision"],
}


def get_drflow_metric(name: str, model: str = "gpt-4o-mini"):
    """
    Get a DRFlow metric instance by name.

    Args:
        name: The metric name. One of: 'factuality', 'insight_recall',
              'insight_precision', 'topology_order', 'condition_resolvement',
              'personalized_comprehensiveness'.
        model: LLM model to use for evaluation.

    Returns:
        An instance of the requested metric class.

    Raises:
        ValueError: If the metric name is unknown.
    """
    if name not in DRFLOW_METRICS:
        available = ", ".join(sorted(DRFLOW_METRICS.keys()))
        raise ValueError(
            f"Unknown DRFlow metric: '{name}'. "
            f"Available metrics: {available}"
        )

    return DRFLOW_METRICS[name](model=model)


def get_all_drflow_metric_names():
    """Return list of all available DRFlow metric names."""
    return list(DRFLOW_METRICS.keys())


def resolve_metric_order(requested_metrics):
    """
    Resolve metric execution order based on dependencies.

    Ensures that dependent metrics run after their prerequisites.
    For example, topology_order runs after insight_precision.

    Args:
        requested_metrics: List of metric names the user wants to run.

    Returns:
        Ordered list of metric names with dependencies satisfied.
    """
    ordered = []
    added = set()

    def _add_metric(name):
        if name in added:
            return
        for dep in METRIC_DEPENDENCIES.get(name, []):
            if dep not in added:
                _add_metric(dep)
        ordered.append(name)
        added.add(name)

    for metric_name in requested_metrics:
        _add_metric(metric_name)

    return ordered
