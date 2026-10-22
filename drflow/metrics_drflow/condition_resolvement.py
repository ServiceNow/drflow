"""
Condition Resolvement Metric for DRBench DRFlow Evaluation.

Evaluates whether the personalized workflow properly resolved the conditions
specified in the golden workflow steps. Golden steps may have a "conditions"
field describing multiple possible paths (e.g., "If X -> do A; If Y -> do B").
The "answer" field shows the correct resolution for the user's situation.

This metric checks whether the predicted steps:
1. Properly routed to the correct condition path (score 1.0)
2. Mentioned conditions but didn't clearly resolve them (score 0.5)
3. Didn't address conditions at all, or failed to match a conditional
   golden step (score 0.0)

Only golden steps with non-empty "conditions" are evaluated.
Each such golden step must have a precision-matched predicted step to
be eligible for non-zero credit; otherwise it automatically scores 0.0.
Score = average condition resolvement across all golden steps with conditions.
"""

import json
import re
from typing import Any, Dict

from drflow.agents.utils import prompt_llm
from drflow.metrics.base import DrBenchMetric


class WorkflowConditionResolvement(DrBenchMetric):
    """
    Evaluates whether predicted workflow steps properly resolve the
    conditions specified in golden steps.

    For each golden step with non-empty conditions:
    1. Identifies the matching predicted step (from precision_matches),
       if one exists.
    2. Uses an LLM to evaluate whether the predicted step properly
       resolved the conditions by choosing the correct path and providing
       appropriate justification, as described in the golden "answer".
    3. Assigns score 0.0 automatically when no predicted step matched
       that conditional golden step.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 3):
        super().__init__(name="workflow_condition_resolvement", model=model)
        self.model = model
        self.max_retries = max_retries

    def compute(
        self,
        report_dict: Dict[str, Any],
        task_data: Dict[str, Any] = None,
        eval_data: Dict[str, Any] = None,
        **kwargs,
    ) -> dict:
        """
        Compute condition resolvement score.

        Args:
            report_dict: Must contain 'workflow_steps' and 'precision_matches'.
            task_data: Task config (not directly used).
            eval_data: Evaluation config with 'dr_report_evaluation_qa'.

        Returns:
            Dict with score, summary, and metric_result.
        """
        eval_data = eval_data or {}
        workflow_steps = report_dict.get("workflow_steps", [])
        precision_matches = report_dict.get("precision_matches", [])
        qa_list = eval_data.get("dr_report_evaluation_qa", [])

        gold_to_pred = {}
        for match in precision_matches:
            gold_id = match.get("gold_id")
            pred_item = match.get("pred_item")
            if gold_id and pred_item is not None:
                gold_to_pred[gold_id] = pred_item

        golden_with_conditions = [
            qa for qa in qa_list
            if qa.get("type") == "workflow_step"
            and qa.get("conditions", "").strip()
        ]

        if not golden_with_conditions:
            return {
                "score": 1.0,
                "summary": (
                    "No golden workflow steps with conditions found. "
                    "Score: 1.0 (trivially satisfied)."
                ),
                "metric_result": {
                    "total_conditions": 0,
                    "resolved_conditions": 0,
                    "partially_resolved": 0,
                    "unresolved_conditions": 0,
                    "per_condition_results": [],
                },
            }

        pred_step_map = {}
        for step in workflow_steps:
            item = step.get("item")
            if item is not None:
                pred_step_map[item] = step
                pred_step_map[str(item)] = step

        per_condition_results = []

        for gold_step in golden_with_conditions:
            gold_id = gold_step.get("id", "")
            gold_title = gold_step.get("title", "")
            gold_conditions = gold_step.get("conditions", "")
            gold_answer = gold_step.get("answer", "")
            gold_justification = gold_step.get("justification", "")

            pred_item = gold_to_pred.get(gold_id)
            if pred_item is None:
                per_condition_results.append({
                    "gold_id": gold_id,
                    "gold_title": gold_title,
                    "gold_conditions": gold_conditions,
                    "score": 0.0,
                    "level": "unresolved",
                    "justification": (
                        "No precision-matched predicted step was found "
                        "for this condition-containing golden step."
                    ),
                    "pred_step_item": None,
                })
                continue

            pred_step = pred_step_map.get(pred_item) or pred_step_map.get(str(pred_item))

            if pred_step is None:
                per_condition_results.append({
                    "gold_id": gold_id,
                    "gold_title": gold_title,
                    "gold_conditions": gold_conditions,
                    "score": 0.0,
                    "level": "unresolved",
                    "justification": (
                        "Predicted step passed precision but could not "
                        "be resolved from workflow data."
                    ),
                    "pred_step_item": pred_item,
                })
                continue

            result = self._evaluate_condition_resolvement(
                gold_id=gold_id,
                gold_title=gold_title,
                gold_conditions=gold_conditions,
                gold_answer=gold_answer,
                gold_justification=gold_justification,
                pred_step=pred_step,
            )
            per_condition_results.append(result)

        if per_condition_results:
            total_score = sum(r["score"] for r in per_condition_results)
            score = total_score / len(per_condition_results)
        else:
            score = 0.0

        resolved = sum(1 for r in per_condition_results if r["score"] == 1.0)
        partial = sum(1 for r in per_condition_results if r["score"] == 0.5)
        unresolved = sum(1 for r in per_condition_results if r["score"] == 0.0)

        summary = (
            f"**Workflow Condition Resolvement:** {score:.4f}\n"
            f"  Fully Resolved: {resolved}/{len(per_condition_results)}\n"
            f"  Partially Resolved: {partial}/{len(per_condition_results)}\n"
            f"  Unresolved: {unresolved}/{len(per_condition_results)}\n\n"
        )
        summary += "---\n\n"

        for r in per_condition_results:
            level_label = {
                "resolved": "Fully Resolved",
                "partially_resolved": "Partially Resolved",
                "unresolved": "Unresolved",
            }.get(r["level"], r["level"])

            summary += (
                f"**{r['gold_id']} - {r['gold_title']}:** {level_label} "
                f"(score: {r['score']:.1f})\n"
            )
            summary += f"  Conditions: {r['gold_conditions'][:150]}...\n"
            summary += f"  Justification: {r['justification']}\n\n"

        return {
            "score": score,
            "summary": summary,
            "metric_result": {
                "total_conditions": len(per_condition_results),
                "resolved_conditions": resolved,
                "partially_resolved": partial,
                "unresolved_conditions": unresolved,
                "per_condition_results": per_condition_results,
            },
        }

    def _evaluate_condition_resolvement(
        self,
        gold_id: str,
        gold_title: str,
        gold_conditions: str,
        gold_answer: str,
        gold_justification: str,
        pred_step: Dict,
    ) -> Dict:
        """
        Use LLM to evaluate if the predicted step properly resolves conditions.
        """
        pred_item = pred_step.get("item", "?")
        pred_title = pred_step.get("title", "")
        pred_task = pred_step.get("task", "")
        pred_desc = pred_step.get("task_description", "")
        pred_subtask = pred_step.get("subtask_description", "")

        gold_answer_trunc = gold_answer[:1000] if len(gold_answer) > 1000 else gold_answer
        gold_justification_trunc = gold_justification[:500] if len(gold_justification) > 500 else gold_justification

        prompt = f"""You are evaluating whether a predicted workflow step properly resolves the conditions described in a golden (ground truth) workflow step.

## Golden Step Conditions
The golden step "{gold_title}" ({gold_id}) has the following conditions that require resolution:
{gold_conditions}

## Expected Resolution (from golden answer)
The correct resolution for the user's specific situation is:
{gold_answer_trunc}

## Justification for the correct resolution
{gold_justification_trunc}

## Predicted Step
The predicted workflow step that corresponds to this golden step is:
- Item: {pred_item}
- Title: {pred_title}
- Task: {pred_task}
- Description: {pred_desc}
- Subtask/Conditions stated: {pred_subtask if pred_subtask else 'None'}

## Evaluation Criteria
Score the condition resolvement based on these levels:

1. **Score 1.0 (Fully Resolved)**: The predicted step correctly resolves the conditions by:
   - Choosing the appropriate condition path for the user's situation
   - Providing specific justification or evidence for the chosen path
   - Not just listing all possible conditions, but actually routing through one

2. **Score 0.5 (Partially Resolved)**: The predicted step mentions the conditions but:
   - Lists multiple possible paths without clearly choosing one
   - Mentions the correct path but lacks specific justification
   - Shows awareness of conditions but doesn't fully personalize the resolution

3. **Score 0.0 (Unresolved)**: The predicted step:
   - Does not address the conditions at all
   - Only provides generic description without condition awareness
   - Completely ignores the conditional routing

Return ONLY a valid JSON object (no other text):
{{
    "score": <0.0, 0.5, or 1.0>,
    "level": "<resolved, partially_resolved, or unresolved>",
    "justification": "<detailed explanation of the evaluation>"
}}"""

        for attempt in range(self.max_retries):
            try:
                response = prompt_llm(prompt, self.model, temperature=0)
                result = self._parse_json_response(response)

                raw_score = float(result.get("score", 0.0))
                if raw_score >= 0.75:
                    final_score = 1.0
                    level = "resolved"
                elif raw_score >= 0.25:
                    final_score = 0.5
                    level = "partially_resolved"
                else:
                    final_score = 0.0
                    level = "unresolved"

                return {
                    "gold_id": gold_id,
                    "gold_title": gold_title,
                    "gold_conditions": gold_conditions,
                    "score": final_score,
                    "level": level,
                    "justification": result.get(
                        "justification", "No justification provided"
                    ),
                    "pred_step_item": pred_step.get("item"),
                }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    return {
                        "gold_id": gold_id,
                        "gold_title": gold_title,
                        "gold_conditions": gold_conditions,
                        "score": 0.0,
                        "level": "unresolved",
                        "justification": (
                            f"Failed to parse LLM response after "
                            f"{self.max_retries} attempts: {str(e)}"
                        ),
                        "pred_step_item": pred_step.get("item"),
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
