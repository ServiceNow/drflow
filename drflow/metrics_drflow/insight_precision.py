"""
Insight Precision Metric for DRBench DRFlow Evaluation.

Checks whether each predicted personalized workflow step matches
a golden workflow step from eval.json.

Precision = (number of predicted steps matched to a golden step)
            / (total number of predicted steps)

Also produces the matching map (predicted -> golden) which is used
by the topology_order metric.
"""

import json
import re
from typing import Any, Dict, List

from drflow.agents.utils import prompt_llm
from drflow.metrics.base import DrBenchMetric
from drflow.metrics_drflow.workflow_parser import format_golden_steps_text


class WorkflowInsightPrecision(DrBenchMetric):
    """
    Evaluates precision of predicted workflow steps against golden steps.

    For each predicted step, uses an LLM to determine whether it semantically
    matches any golden workflow step. Produces a matching map that downstream
    metrics (topology_order) can consume.

    Defaults are tuned slightly stricter than recall (higher confidence floor,
    extra parse retries) to penalize spurious predicted steps.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 4,
        min_match_confidence: float = 0.8,
    ):
        super().__init__(name="workflow_insight_precision", model=model)
        self.model = model
        self.max_retries = max_retries
        self.min_match_confidence = min_match_confidence

    def compute(
        self,
        report_dict: Dict[str, Any],
        task_data: Dict[str, Any] = None,
        eval_data: Dict[str, Any] = None,
        **kwargs,
    ) -> dict:
        """
        Compute insight precision for workflow steps.

        Args:
            report_dict: Must contain 'workflow_steps'.
            task_data: Task config (not directly used).
            eval_data: Evaluation config with 'dr_report_evaluation_qa'.

        Returns:
            Dict with score, summary, and metric_result including per-step
            matches and the precision_matches list for topology_order.
        """
        workflow_steps = report_dict.get("workflow_steps", [])
        qa_list = eval_data.get("dr_report_evaluation_qa", [])
        golden_steps = [
            qa for qa in qa_list if qa.get("type") == "workflow_step"
        ]

        if not workflow_steps:
            return {
                "score": 0.0,
                "summary": "No predicted workflow steps found in report.",
                "metric_result": {
                    "total_predicted": 0,
                    "matched_predicted": 0,
                    "per_step_results": [],
                    "precision_matches": [],
                },
            }

        if not golden_steps:
            return {
                "score": 0.0,
                "summary": "No golden workflow steps found in eval config.",
                "metric_result": {
                    "total_predicted": len(workflow_steps),
                    "matched_predicted": 0,
                    "per_step_results": [
                        {
                            "pred_item": s.get("item"),
                            "pred_title": s.get("title"),
                            "matched": False,
                            "matched_gold_id": None,
                            "justification": "No golden steps available",
                        }
                        for s in workflow_steps
                    ],
                    "precision_matches": [],
                },
            }

        gold_steps_text = format_golden_steps_text(eval_data)

        per_step_results = []

        for pred_step in workflow_steps:
            pred_item = pred_step.get("item", "?")
            pred_title = pred_step.get("title", "")
            pred_task = pred_step.get("task", "")
            pred_desc = pred_step.get("task_description", "")
            pred_subtask = pred_step.get("subtask_description", "")

            match_result = self._match_predicted_step(
                pred_item=pred_item,
                pred_title=pred_title,
                pred_task=pred_task,
                pred_desc=pred_desc,
                pred_subtask=pred_subtask,
                gold_steps_text=gold_steps_text,
                golden_steps=golden_steps,
            )
            per_step_results.append(match_result)

        per_step_results = self._enforce_match_constraints(
            per_step_results, golden_steps
        )

        matched_count = sum(
            1 for r in per_step_results if r["matched"]
        )
        total = len(workflow_steps)
        score = matched_count / total if total > 0 else 0.0

        precision_matches = []
        for r in per_step_results:
            if r["matched"] and r["matched_gold_id"]:
                precision_matches.append({
                    "pred_item": r["pred_item"],
                    "pred_title": r["pred_title"],
                    "gold_id": r["matched_gold_id"],
                    "gold_title": r.get("matched_gold_title", ""),
                })

        summary = (
            f"**Workflow Insight Precision:** {score:.4f} "
            f"({matched_count}/{total} predicted steps matched to golden)\n\n"
        )
        summary += "---\n\n"

        for r in per_step_results:
            status = "Matched" if r["matched"] else "Not Matched"
            summary += (
                f"**Predicted Step {r['pred_item']} - {r['pred_title']}:** "
                f"{status}\n"
            )
            if r["matched_gold_id"]:
                summary += (
                    f"  Matched to Golden: {r['matched_gold_id']}\n"
                )
            summary += f"  Justification: {r['justification']}\n\n"

        return {
            "score": score,
            "summary": summary,
            "metric_result": {
                "total_predicted": total,
                "matched_predicted": matched_count,
                "precision_score": score,
                "per_step_results": per_step_results,
                "precision_matches": precision_matches,
            },
        }

    def _enforce_match_constraints(
        self,
        per_step_results: List[Dict[str, Any]],
        golden_steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply stricter post-processing so broad or duplicate matches do not score."""
        valid_gold_ids = {
            step.get("id"): step.get("title", "")
            for step in golden_steps
            if step.get("id")
        }
        best_match_by_gold: Dict[str, int] = {}

        for idx, result in enumerate(per_step_results):
            confidence = self._safe_confidence(result.get("confidence"))
            gold_id = result.get("matched_gold_id")

            if not result.get("matched"):
                result["confidence"] = confidence
                continue

            if gold_id not in valid_gold_ids:
                self._mark_unmatched(
                    result,
                    "Rejected: matched golden step ID was missing or invalid.",
                )
                continue

            if confidence < self.min_match_confidence:
                self._mark_unmatched(
                    result,
                    (
                        "Rejected: match confidence "
                        f"{confidence:.2f} is below the strict threshold "
                        f"of {self.min_match_confidence:.2f}."
                    ),
                )
                continue

            result["confidence"] = confidence
            result["matched_gold_title"] = valid_gold_ids[gold_id]

            previous_idx = best_match_by_gold.get(gold_id)
            if previous_idx is None:
                best_match_by_gold[gold_id] = idx
                continue

            previous_confidence = self._safe_confidence(
                per_step_results[previous_idx].get("confidence")
            )
            if confidence > previous_confidence:
                self._mark_unmatched(
                    per_step_results[previous_idx],
                    (
                        "Rejected: another predicted step matched the same golden "
                        "step with higher confidence, so only the strongest "
                        "one-to-one match is retained."
                    ),
                )
                best_match_by_gold[gold_id] = idx
            else:
                self._mark_unmatched(
                    result,
                    (
                        "Rejected: duplicate match to a golden step that already "
                        "has an equal or stronger predicted match."
                    ),
                )

        return per_step_results

    def _mark_unmatched(self, result: Dict[str, Any], justification: str) -> None:
        result["matched"] = False
        result["matched_gold_id"] = None
        result["matched_gold_title"] = ""
        result["confidence"] = self._safe_confidence(result.get("confidence"))
        result["justification"] = justification

    def _safe_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _match_predicted_step(
        self,
        pred_item,
        pred_title: str,
        pred_task: str,
        pred_desc: str,
        pred_subtask: str,
        gold_steps_text: str,
        golden_steps: List[Dict],
    ) -> Dict:
        """
        Use LLM to check if a predicted step matches any golden step.
        """
        prompt = f"""You are evaluating whether a predicted workflow step semantically matches any ground truth (golden) workflow step.

## Matching Rules (precision-oriented — avoid false positives)
Match on **procedure intent**, not on wording. Ignore case-specific instance details (names, IDs, ticket numbers, dates, amounts, customer/vendor strings): those must neither force nor forbid a match.

To count as a match, the predicted step must:
- Address the **same primary obligation** as the golden step (what must be done, in what role/system context), not merely a related or upstream/downstream activity.
- Align on the **same decision or artifact class** when the golden step is conditional or produces a specific record/approval/output.

Reject `matched=true` when the prediction is only loosely related, a generic umbrella ("handle the case"), a prerequisite, a follow-up, merges two distinct golden obligations into one vague step, or differs in system/process family (e.g., intake vs ledger posting vs legal hold).

Reserve **high confidence (≥0.88)** only for clear same-procedure alignment; otherwise use `matched=false` or lower confidence.

## Predicted Step
- Item: {pred_item}
- Title: {pred_title}
- Task: {pred_task}
- Description: {pred_desc}
- Subtask/Conditions: {pred_subtask if pred_subtask else 'None'}

## Golden Workflow Steps
{gold_steps_text}

## Instructions
Pick at most one best golden step. Use `matched=true` only if the predicted step would satisfy a reviewer that it is the **same** procedural beat as that golden step (ignoring instance-specific strings). If in doubt, prefer `matched=false`.
Return ONLY a valid JSON object (no other text):
{{
    "matched": true or false,
    "matched_gold_id": "<ID of the best matching golden step, e.g. WORKFLOW_STEP_1, or null if no match>",
    "matched_gold_title": "<title of matched golden step or null>",
    "confidence": <float 0.0-1.0>,
    "justification": "<brief explanation of why it matches or why no match was found>"
}}"""

        for attempt in range(self.max_retries):
            try:
                response = prompt_llm(prompt, self.model, temperature=0)
                result = self._parse_json_response(response)

                return {
                    "pred_item": pred_item,
                    "pred_title": pred_title,
                    "matched": bool(result.get("matched", False)),
                    "matched_gold_id": result.get("matched_gold_id"),
                    "matched_gold_title": result.get(
                        "matched_gold_title", ""
                    ),
                    "confidence": result.get("confidence", 0.0),
                    "justification": result.get(
                        "justification", "No justification provided"
                    ),
                }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    return {
                        "pred_item": pred_item,
                        "pred_title": pred_title,
                        "matched": False,
                        "matched_gold_id": None,
                        "matched_gold_title": "",
                        "confidence": 0.0,
                        "justification": (
                            f"Failed to parse LLM response after "
                            f"{self.max_retries} attempts: {str(e)}"
                        ),
                    }

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling code blocks."""
        response = response.strip()

        json_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL
        )
        if json_match:
            return json.loads(json_match.group(1))

        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        return json.loads(response)
