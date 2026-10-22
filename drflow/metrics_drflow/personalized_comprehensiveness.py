"""
Personalized Comprehensiveness Metric for DRBench DRFlow Evaluation.

Evaluates how comprehensive and personalized the predicted workflow step
descriptions are compared to the golden "task" and "answer" fields.

Only considers predicted steps that passed insight precision (i.e., that
are present in precision_matches).

Checks whether predicted steps:
- Mention specific names (forms, people, institutions, etc.)
- Include specific what-to-do instructions
- Reference specific data categories, study types, procedures
- Are personalized to the user's situation rather than generic

Score = average comprehensiveness across precision-matched predicted-golden pairs.
"""

import json
import re
from typing import Any, Dict, List

from drflow.agents.utils import prompt_llm
from drflow.metrics.base import DrBenchMetric


class WorkflowPersonalizedComprehensiveness(DrBenchMetric):
    """
    Evaluates the comprehensiveness and personalization of predicted
    workflow step descriptions compared to the golden standard.

    For each matched pair (predicted step <-> golden step):
    1. Compares the predicted task + task_description to the golden task + answer.
    2. Uses an LLM to assess:
       - Specificity: mentions of specific names, forms, procedures
       - Personalization: tailored to the user's situation
       - Completeness: covers the key action items from the golden answer
       - Actionability: provides clear, actionable instructions
    3. Returns a score from 0.0 to 1.0.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        missing_detail_penalty: float = 0.01,
        max_missing_detail_penalty: float = 0.03,
    ):
        super().__init__(
            name="workflow_personalized_comprehensiveness", model=model
        )
        self.model = model
        self.max_retries = max_retries
        self.missing_detail_penalty = missing_detail_penalty
        self.max_missing_detail_penalty = max_missing_detail_penalty

    def compute(
        self,
        report_dict: Dict[str, Any],
        task_data: Dict[str, Any] = None,
        eval_data: Dict[str, Any] = None,
        **kwargs,
    ) -> dict:
        """
        Compute personalized comprehensiveness score.

        Only evaluates predicted steps that passed insight precision
        (present in precision_matches). This is consistent with the
        topology_order and condition_resolvement metrics.

        Args:
            report_dict: Must contain 'workflow_steps' and 'precision_matches'.
            task_data: Task config (not directly used).
            eval_data: Evaluation config with 'dr_report_evaluation_qa'.

        Returns:
            Dict with score, summary, and metric_result.
        """
        workflow_steps = report_dict.get("workflow_steps", [])
        precision_matches = report_dict.get("precision_matches", [])
        qa_list = eval_data.get("dr_report_evaluation_qa", [])

        if not precision_matches:
            return {
                "score": 0.0,
                "summary": (
                    "No predicted steps passed insight precision. "
                    "Cannot evaluate personalized comprehensiveness."
                ),
                "metric_result": {
                    "total_pairs": 0,
                    "average_score": 0.0,
                    "per_pair_results": [],
                },
            }

        pred_step_map = {}
        for step in workflow_steps:
            item = step.get("item")
            if item is not None:
                pred_step_map[item] = step
                pred_step_map[str(item)] = step

        gold_step_map = {}
        for qa in qa_list:
            if qa.get("type") == "workflow_step":
                gold_step_map[qa.get("id")] = qa

        per_pair_results = []

        for match in precision_matches:
            pred_item = match.get("pred_item")
            gold_id = match.get("gold_id")

            pred_step = pred_step_map.get(pred_item) or pred_step_map.get(str(pred_item))
            gold_step = gold_step_map.get(gold_id)

            if pred_step is None or gold_step is None:
                per_pair_results.append({
                    "pred_item": pred_item,
                    "gold_id": gold_id,
                    "score": 0.0,
                    "justification": "Could not find matching step data.",
                    "missing_details": [],
                })
                continue

            result = self._evaluate_comprehensiveness(
                pred_step=pred_step,
                gold_step=gold_step,
            )
            per_pair_results.append(result)

        if per_pair_results:
            total_score = sum(r["score"] for r in per_pair_results)
            score = total_score / len(per_pair_results)
        else:
            score = 0.0

        summary = (
            f"**Workflow Personalized Comprehensiveness:** {score:.4f}\n"
            f"  Evaluated pairs: {len(per_pair_results)}\n\n"
        )
        summary += "---\n\n"

        for r in per_pair_results:
            summary += (
                f"**Predicted Step {r['pred_item']} <-> "
                f"Golden {r['gold_id']}:** Score {r['score']:.2f}\n"
            )
            summary += f"  Justification: {r['justification']}\n"
            if r.get("missing_details"):
                summary += (
                    f"  Missing Details: "
                    f"{', '.join(r['missing_details'][:5])}\n"
                )
            summary += "\n"

        return {
            "score": score,
            "summary": summary,
            "metric_result": {
                "total_pairs": len(per_pair_results),
                "average_score": score,
                "per_pair_results": per_pair_results,
            },
        }

    def _evaluate_comprehensiveness(
        self,
        pred_step: Dict,
        gold_step: Dict,
    ) -> Dict:
        """
        Use LLM to evaluate comprehensiveness of a predicted step.
        """
        pred_item = pred_step.get("item", "?")
        pred_title = pred_step.get("title", "")
        pred_task = pred_step.get("task", "")
        pred_desc = pred_step.get("task_description", "")
        pred_subtask = pred_step.get("subtask_description", "")

        gold_id = gold_step.get("id", "")
        gold_title = gold_step.get("title", "")
        gold_task = gold_step.get("task", "")
        gold_answer = gold_step.get("answer", "")
        gold_justification = gold_step.get("justification", "")

        gold_answer_trunc = (
            gold_answer[:1500] if len(gold_answer) > 1500 else gold_answer
        )

        prompt = f"""You are evaluating the comprehensiveness and personalization of a predicted workflow step description compared to a golden (ground truth) standard.

Do not reward loose thematic similarity, generic workflow language, or partially overlapping content as highly comprehensive. A high score should be reserved for steps that preserve the concrete details, user-specific tailoring, and operational guidance from the ground truth.

## Golden Step (Ground Truth)
- ID: {gold_id}
- Title: {gold_title}
- Task: {gold_task}
- Expected Detailed Answer: {gold_answer_trunc}
- Justification: {gold_justification[:1000] if gold_justification else 'N/A'}

## Predicted Step
- Item: {pred_item}
- Title: {pred_title}
- Task: {pred_task}
- Description: {pred_desc}
- Subtask/Conditions: {pred_subtask if pred_subtask else 'None'}

## Evaluation Dimensions
Rate the predicted step on a scale of 0.0 to 1.0 based on these criteria:

1. **Specificity** (0.0-1.0): Compared to the golden step answer, does the predicted description mention specific names, form numbers, data categories, institutions, study types, procedures, thresholds, and other concrete details from the golden answer? Or is it vague and generic?

2. **Personalization** (0.0-1.0): Compared to the golden step answer, is the description tailored to the user's specific situation (e.g., mentioning the specific study type, specific forms, specific data categories, specific numbers, specific PI or coordinator)? Or could it apply to many generic situations?

3. **Completeness** (0.0-1.0): Compared to the golden step answer, does the description cover the key action items and important details from the golden answer? Are major steps, conditions, branches, approvals, or required artifacts missing?

4. **Actionability** (0.0-1.0): Compared to the golden step answer, does the description provide clear, actionable instructions that a user could follow? Or is it too high-level to be useful?

Scoring guidance:
- Reserve 0.90-1.00 for near-complete, highly specific, clearly personalized steps with very few meaningful omissions.
- Use 0.70-0.89 for good steps that are still missing some concrete details or tailoring.
- Use 0.40-0.69 for partially correct but generic/incomplete steps.
- Use 0.00-0.39 when the step is mostly generic, weakly personalized, or missing major workflow content.

Compute an overall score as the average of these four dimensions before any penalties for missing details.

Return ONLY a valid JSON object (no other text):
{{
    "specificity": <float 0.0-1.0>,
    "personalization": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "actionability": <float 0.0-1.0>,
    "overall_score": <float 0.0-1.0, average of above>,
    "justification": "<concise explanation covering all dimensions>",
    "missing_details": ["<list of important details from golden answer missing in predicted step>"]
}}"""

        for attempt in range(self.max_retries):
            try:
                response = prompt_llm(prompt, self.model, temperature=0)
                result = self._parse_json_response(response)

                specificity = self._safe_score(result.get("specificity", 0.0))
                personalization = self._safe_score(
                    result.get("personalization", 0.0)
                )
                completeness = self._safe_score(result.get("completeness", 0.0))
                actionability = self._safe_score(result.get("actionability", 0.0))
                missing_details = self._normalize_missing_details(
                    result.get("missing_details", [])
                )

                dimension_average = (
                    specificity
                    + personalization
                    + completeness
                    + actionability
                ) / 4.0
                model_overall = self._safe_score(result.get("overall_score", 0.0))
                overall = min(model_overall, dimension_average)

                penalty = min(
                    len(missing_details) * self.missing_detail_penalty,
                    self.max_missing_detail_penalty,
                )
                overall = max(0.0, overall - penalty)

                if specificity < 0.6 or personalization < 0.6:
                    overall = min(overall, 0.7)
                if completeness < 0.6:
                    overall = min(overall, 0.65)
                if actionability < 0.5:
                    overall = min(overall, 0.75)

                return {
                    "pred_item": pred_item,
                    "gold_id": gold_id,
                    "score": overall,
                    "specificity": specificity,
                    "personalization": personalization,
                    "completeness": completeness,
                    "actionability": actionability,
                    "justification": result.get(
                        "justification", "No justification provided"
                    ),
                    "missing_details": missing_details,
                    "missing_detail_penalty": penalty,
                }
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    return {
                        "pred_item": pred_item,
                        "gold_id": gold_id,
                        "score": 0.0,
                        "specificity": 0.0,
                        "personalization": 0.0,
                        "completeness": 0.0,
                        "actionability": 0.0,
                        "justification": (
                            f"Failed to parse LLM response after "
                            f"{self.max_retries} attempts: {str(e)}"
                        ),
                        "missing_details": [],
                        "missing_detail_penalty": 0.0,
                    }

    def _safe_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def _normalize_missing_details(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

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
