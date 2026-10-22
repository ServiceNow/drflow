"""
Insight Recall Metric for DRBench DRFlow Evaluation.

Checks whether the golden workflow steps from eval.json are present
(semantically matched) in the predicted personalized workflow steps.

Recall = (number of golden steps matched by at least one predicted step)
         / (total number of golden steps)
"""

import json
import re
from typing import Any, Dict, List

from drflow.agents.utils import prompt_llm
from drflow.metrics.base import DrBenchMetric
from drflow.metrics_drflow.workflow_parser import format_predicted_steps_text


class WorkflowInsightRecall(DrBenchMetric):
    """
    Evaluates recall of golden workflow steps in the predicted workflow.

    For each golden step, uses an LLM to determine whether any predicted
    step semantically covers the same core action/procedure. A match means
    the predicted step addresses the same key activity, even if wording differs.

    Uses a distinct confidence threshold from insight precision so the two
    metrics are not identical in strictness.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        min_match_confidence: float = 0.8,
    ):
        super().__init__(name="workflow_insight_recall", model=model)
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
        Compute insight recall for workflow steps.

        Args:
            report_dict: Must contain 'workflow_steps'.
            task_data: Task config (not directly used).
            eval_data: Evaluation config with 'dr_report_evaluation_qa'.

        Returns:
            Dict with score, summary, and metric_result including per-step matches.
        """
        workflow_steps = report_dict.get("workflow_steps", [])
        qa_list = eval_data.get("dr_report_evaluation_qa", [])
        golden_steps = [
            qa for qa in qa_list if qa.get("type") == "workflow_step"
        ]

        if not golden_steps:
            return {
                "score": 0.0,
                "summary": "No golden workflow steps found in eval config.",
                "metric_result": {
                    "total_golden": 0,
                    "matched_golden": 0,
                    "per_step_results": [],
                },
            }

        if not workflow_steps:
            return {
                "score": 0.0,
                "summary": "No predicted workflow steps found in report.",
                "metric_result": {
                    "total_golden": len(golden_steps),
                    "matched_golden": 0,
                    "per_step_results": [
                        {
                            "gold_id": gs.get("id"),
                            "gold_title": gs.get("title"),
                            "matched": False,
                            "matched_pred_item": None,
                            "justification": "No predicted steps available",
                        }
                        for gs in golden_steps
                    ],
                },
            }

        pred_steps_text = format_predicted_steps_text(workflow_steps)

        per_step_results = []

        for gold_step in golden_steps:
            gold_id = gold_step.get("id", "")
            gold_title = gold_step.get("title", "")
            gold_task = gold_step.get("task", "")
            gold_answer = gold_step.get("answer", "")

            match_result = self._match_golden_step(
                gold_id=gold_id,
                gold_title=gold_title,
                gold_task=gold_task,
                gold_answer=gold_answer,
                pred_steps_text=pred_steps_text,
                workflow_steps=workflow_steps,
            )
            match_result = self._enforce_match_constraints(
                match_result, workflow_steps
            )
            per_step_results.append(match_result)

        per_step_results = self._enforce_one_to_one_matching(
            per_step_results
        )

        matched_count = sum(
            1 for r in per_step_results if r["matched"]
        )
        total = len(golden_steps)
        score = matched_count / total if total > 0 else 0.0

        summary = (
            f"**Workflow Insight Recall:** {score:.4f} "
            f"({matched_count}/{total} golden steps matched)\n\n"
        )
        summary += "---\n\n"

        for r in per_step_results:
            status = "Matched" if r["matched"] else "Not Matched"
            summary += f"**{r['gold_id']} - {r['gold_title']}:** {status}\n"
            if r["matched_pred_item"] is not None:
                summary += (
                    f"  Matched to Predicted Step {r['matched_pred_item']}\n"
                )
            summary += f"  Justification: {r['justification']}\n\n"

        return {
            "score": score,
            "summary": summary,
            "metric_result": {
                "total_golden": total,
                "matched_golden": matched_count,
                "recall_score": score,
                "per_step_results": per_step_results,
            },
        }

    def _enforce_match_constraints(
        self,
        match_result: Dict[str, Any],
        workflow_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Reject low-confidence or invalid matches so recall is less forgiving."""
        confidence = self._safe_confidence(match_result.get("confidence"))
        valid_pred_items = {
            step.get("item") for step in workflow_steps if step.get("item") is not None
        }

        match_result["confidence"] = confidence
        matched_pred_item = self._normalize_pred_item(
            match_result.get("matched_pred_item")
        )
        match_result["matched_pred_item"] = matched_pred_item

        if not match_result.get("matched"):
            return match_result

        if matched_pred_item not in valid_pred_items:
            match_result["matched"] = False
            match_result["matched_pred_item"] = None
            match_result["justification"] = (
                "Rejected: matched predicted step item was missing or invalid."
            )
            return match_result

        if confidence < self.min_match_confidence:
            match_result["matched"] = False
            match_result["matched_pred_item"] = None
            match_result["justification"] = (
                "Rejected: match confidence "
                f"{confidence:.2f} is below the strict threshold "
                f"of {self.min_match_confidence:.2f}."
            )

        return match_result

    def _enforce_one_to_one_matching(
        self,
        per_step_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ensure each predicted step is matched to at most one golden step.

        When multiple golden steps claim the same predicted step, only the
        golden step with the highest confidence keeps the match; the others
        are marked unmatched.
        """
        best_by_pred: Dict[Any, int] = {}

        for idx, result in enumerate(per_step_results):
            if not result.get("matched"):
                continue
            pred_item = result.get("matched_pred_item")
            if pred_item is None:
                continue

            prev_idx = best_by_pred.get(pred_item)
            if prev_idx is None:
                best_by_pred[pred_item] = idx
                continue

            prev_conf = self._safe_confidence(
                per_step_results[prev_idx].get("confidence")
            )
            curr_conf = self._safe_confidence(result.get("confidence"))

            if curr_conf > prev_conf:
                per_step_results[prev_idx]["matched"] = False
                per_step_results[prev_idx]["matched_pred_item"] = None
                per_step_results[prev_idx]["justification"] = (
                    "Rejected: another golden step matched the same "
                    "predicted step with higher confidence, so only "
                    "the strongest one-to-one match is retained."
                )
                best_by_pred[pred_item] = idx
            else:
                result["matched"] = False
                result["matched_pred_item"] = None
                result["justification"] = (
                    "Rejected: another golden step already matched the "
                    "same predicted step with equal or higher confidence."
                )

        return per_step_results

    def _safe_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _normalize_pred_item(self, value: Any) -> Any:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def _match_golden_step(
        self,
        gold_id: str,
        gold_title: str,
        gold_task: str,
        gold_answer: str,
        pred_steps_text: str,
        workflow_steps: List[Dict],
    ) -> Dict:
        """
        Use LLM to check if a golden step is covered by any predicted step.
        """
        gold_answer_truncated = gold_answer[:800] if len(gold_answer) > 800 else gold_answer

        prompt = f"""You are evaluating whether a ground truth workflow step is semantically covered by any of the predicted workflow steps.

## Matching Rules (recall-oriented — demand real coverage)
The golden step is the reference. Ignore instance-only strings (people names, account numbers, dates, free-text identifiers): predictions may use different wording or entities and still match.

For `matched=true`, some predicted step must **materially implement** what the golden step requires:
- The predicted step's task/description should make it plausible that the golden obligation would be completed, not just mentioned in passing or implied by a distant step.
- Title overlap alone is insufficient if the predicted action is a different kind of work (e.g., "notify stakeholder" vs "obtain written approval").

Reject when the closest prediction is only adjacent (prerequisite, packaging, logging) or a superset that omits the golden step's distinctive requirement. Use confidence **≥0.83** only when coverage is substantive; otherwise `matched=false` or lower confidence.

## Ground Truth Step
- ID: {gold_id}
- Title: {gold_title}
- Task: {gold_task}
- Expected Answer: {gold_answer_truncated}

## Predicted Workflow Steps
{pred_steps_text}

## Instructions
If multiple predictions partially relate, pick the **single** predicted item that best covers this golden step. If none clearly does, answer `matched=false`. Do not stretch synonyms across different process families.
Return ONLY a valid JSON object (no other text):
{{
    "matched": true or false,
    "matched_pred_item": <item number of the best matching predicted step, or null if no match>,
    "confidence": <float 0.0-1.0>,
    "justification": "<brief explanation of why it matches or why no match was found>"
}}"""

        for attempt in range(self.max_retries):
            try:
                response = prompt_llm(prompt, self.model, temperature=0)
                result = self._parse_json_response(response)

                return {
                    "gold_id": gold_id,
                    "gold_title": gold_title,
                    "matched": bool(result.get("matched", False)),
                    "matched_pred_item": result.get("matched_pred_item"),
                    "confidence": result.get("confidence", 0.0),
                    "justification": result.get(
                        "justification", "No justification provided"
                    ),
                }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    return {
                        "gold_id": gold_id,
                        "gold_title": gold_title,
                        "matched": False,
                        "matched_pred_item": None,
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
