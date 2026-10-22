"""
Topology Order Metric for DRBench DRFlow Evaluation.

Checks whether the predicted workflow steps that match golden steps
maintain the correct prerequisite ordering. Only considers predicted
steps that pass insight_precision (i.e., are matched to golden steps).

The metric computes a pairwise ordering consistency score:
for every pair of matched steps, it checks whether their relative
order in the predicted workflow is consistent with the golden order.

Score = concordant_pairs / total_pairs  (Kendall's tau-like)
"""

import re
from typing import Any, Dict, List, Tuple

from drflow.metrics.base import DrBenchMetric


class WorkflowTopologyOrder(DrBenchMetric):
    """
    Evaluates whether matched predicted steps maintain the prerequisite
    ordering defined by the golden workflow.

    Algorithm:
    1. Retrieve precision_matches from report_dict (output of insight_precision).
    2. For each match, extract the golden step number (from WORKFLOW_STEP_N)
       and the predicted item number.
    3. For all pairs of matched steps, check if the predicted item ordering
       is consistent with the golden step ordering.
    4. Also verify that the explicit prerequisite chains in predicted steps
       are consistent with the golden order.
    5. Score = fraction of concordant pairs.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(name="workflow_topology_order", model=model)

    def compute(
        self,
        report_dict: Dict[str, Any],
        task_data: Dict[str, Any] = None,
        eval_data: Dict[str, Any] = None,
        **kwargs,
    ) -> dict:
        """
        Compute topology order score.

        Args:
            report_dict: Must contain 'workflow_steps' and 'precision_matches'.
                precision_matches is a list of dicts with 'pred_item' and 'gold_id'.
            task_data: Task config (not directly used).
            eval_data: Evaluation config (not directly used).

        Returns:
            Dict with score, summary, and metric_result.
        """
        workflow_steps = report_dict.get("workflow_steps", [])
        precision_matches = report_dict.get("precision_matches", [])

        if len(precision_matches) < 2:
            return {
                "score": 1.0 if len(precision_matches) == 1 else 0.0,
                "summary": (
                    f"Topology order: {len(precision_matches)} matched steps "
                    f"(need >= 2 for pairwise comparison). "
                    f"Score: {'1.0 (trivially correct)' if len(precision_matches) == 1 else '0.0 (no matches)'}"
                ),
                "metric_result": {
                    "num_matched_steps": len(precision_matches),
                    "concordant_pairs": 0,
                    "discordant_pairs": 0,
                    "total_pairs": 0,
                    "pairwise_details": [],
                    "prerequisite_consistency": [],
                },
            }

        matched_pairs = []
        for match in precision_matches:
            pred_item = match.get("pred_item")
            gold_id = match.get("gold_id", "")

            gold_num = self._extract_step_number(gold_id)

            if pred_item is not None and gold_num is not None:
                matched_pairs.append({
                    "pred_item": int(pred_item) if pred_item else 0,
                    "gold_num": gold_num,
                    "gold_id": gold_id,
                    "pred_title": match.get("pred_title", ""),
                    "gold_title": match.get("gold_title", ""),
                })

        if len(matched_pairs) < 2:
            return {
                "score": 1.0 if len(matched_pairs) == 1 else 0.0,
                "summary": (
                    f"Topology order: {len(matched_pairs)} valid matched pairs "
                    f"(need >= 2 for comparison)."
                ),
                "metric_result": {
                    "num_matched_steps": len(matched_pairs),
                    "concordant_pairs": 0,
                    "discordant_pairs": 0,
                    "total_pairs": 0,
                    "pairwise_details": [],
                    "prerequisite_consistency": [],
                },
            }

        concordant = 0
        discordant = 0
        pairwise_details = []

        for i in range(len(matched_pairs)):
            for j in range(i + 1, len(matched_pairs)):
                pair_i = matched_pairs[i]
                pair_j = matched_pairs[j]

                gold_order = pair_i["gold_num"] < pair_j["gold_num"]
                pred_order = pair_i["pred_item"] < pair_j["pred_item"]

                is_concordant = (gold_order == pred_order)

                if is_concordant:
                    concordant += 1
                else:
                    discordant += 1

                pairwise_details.append({
                    "step_a": {
                        "pred_item": pair_i["pred_item"],
                        "gold_id": pair_i["gold_id"],
                    },
                    "step_b": {
                        "pred_item": pair_j["pred_item"],
                        "gold_id": pair_j["gold_id"],
                    },
                    "gold_order_correct": gold_order,
                    "pred_order_correct": pred_order,
                    "concordant": is_concordant,
                })

        total_pairs = concordant + discordant
        score = concordant / total_pairs if total_pairs > 0 else 0.0

        prerequisite_consistency = self._check_prerequisite_chains(
            workflow_steps, matched_pairs
        )

        summary = (
            f"**Workflow Topology Order:** {score:.4f} "
            f"({concordant}/{total_pairs} concordant pairs)\n\n"
        )
        summary += f"Matched steps used for ordering: {len(matched_pairs)}\n\n"

        if discordant > 0:
            summary += "**Discordant pairs (order violations):**\n"
            for detail in pairwise_details:
                if not detail["concordant"]:
                    a = detail["step_a"]
                    b = detail["step_b"]
                    summary += (
                        f"  - Predicted Step {a['pred_item']} ({a['gold_id']}) "
                        f"vs Step {b['pred_item']} ({b['gold_id']}): "
                        f"order mismatch\n"
                    )
            summary += "\n"

        if prerequisite_consistency:
            summary += "**Prerequisite chain analysis:**\n"
            for pc in prerequisite_consistency:
                status = "Consistent" if pc["consistent"] else "Inconsistent"
                summary += (
                    f"  - Step {pc['pred_item']} prereq={pc['prerequisite']}: "
                    f"{status} ({pc['detail']})\n"
                )

        return {
            "score": score,
            "summary": summary,
            "metric_result": {
                "num_matched_steps": len(matched_pairs),
                "concordant_pairs": concordant,
                "discordant_pairs": discordant,
                "total_pairs": total_pairs,
                "pairwise_details": pairwise_details,
                "prerequisite_consistency": prerequisite_consistency,
            },
        }

    def _extract_step_number(self, gold_id: str) -> int:
        """Extract step number from gold_id like 'WORKFLOW_STEP_3'."""
        match = re.search(r'(\d+)$', gold_id)
        if match:
            return int(match.group(1))
        return None

    def _check_prerequisite_chains(
        self,
        workflow_steps: List[Dict],
        matched_pairs: List[Dict],
    ) -> List[Dict]:
        """
        Check if explicit prerequisites in predicted steps are consistent
        with the golden ordering of matched steps.

        For each matched predicted step that has a prerequisite, check:
        - Does the prerequisite step also match a golden step?
        - If so, is the golden ordering consistent?
        """
        pred_to_gold = {}
        for mp in matched_pairs:
            pred_to_gold[mp["pred_item"]] = mp["gold_num"]

        results = []

        for step in workflow_steps:
            pred_item = step.get("item")
            prerequisite = step.get("prerequisite", "")

            if not prerequisite or pred_item not in pred_to_gold:
                continue

            prereq_items = []
            for p in str(prerequisite).split(","):
                p = p.strip()
                if p.isdigit():
                    prereq_items.append(int(p))

            for prereq_item in prereq_items:
                if prereq_item in pred_to_gold:
                    gold_current = pred_to_gold[pred_item]
                    gold_prereq = pred_to_gold[prereq_item]

                    consistent = gold_prereq <= gold_current

                    results.append({
                        "pred_item": pred_item,
                        "prerequisite": prereq_item,
                        "gold_current": gold_current,
                        "gold_prereq": gold_prereq,
                        "consistent": consistent,
                        "detail": (
                            f"Gold step {gold_prereq} -> {gold_current}: "
                            f"{'correct order' if consistent else 'REVERSED order'}"
                        ),
                    })
                else:
                    results.append({
                        "pred_item": pred_item,
                        "prerequisite": prereq_item,
                        "gold_current": pred_to_gold.get(pred_item),
                        "gold_prereq": None,
                        "consistent": True,
                        "detail": (
                            f"Prerequisite step {prereq_item} not matched "
                            f"to any golden step (cannot verify)"
                        ),
                    })

        return results
