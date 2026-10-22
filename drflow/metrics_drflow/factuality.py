"""
Workflow Factuality Metric for DRBench DRFlow Evaluation.

Checks whether each personalized workflow step's description is factually
supported by the cited source documents. Adapts the citation-based factuality
approach from drflow.metrics.factuality_v2 for workflow steps.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tqdm

from drflow import task_loader
from drflow.agents.utils import get_factuality_verdict_multi
from drflow.metrics.base import DrBenchMetric
from drflow.metrics_drflow.workflow_parser import extract_step_citations

logger = logging.getLogger(__name__)


class WorkflowFactuality(DrBenchMetric):
    """
    Evaluates factuality of personalized workflow steps.

    For each predicted workflow step, the metric:
    1. Concatenates the step's title, task, and task_description into a claim.
    2. Extracts citation references embedded in the step text (e.g., [^3]).
    3. Resolves citations to actual file names via the report's references map.
    4. Preprocesses enterprise citations (Mattermost, email): decodes
       ``b'…'`` bytes-literal subjects and resolves display names to
       internal identifiers before content lookup.
    5. Verifies the claim against source document content using
       get_factuality_verdict_multi.
    6. Steps without citations are marked as unfactual.

    Score = fraction of steps that are factually supported.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(name="workflow_factuality", model=model)
        self.model = model

    def compute(
        self,
        report_dict: Dict[str, Any],
        task_data: Dict[str, Any] = None,
        eval_data: Dict[str, Any] = None,
        **kwargs,
    ) -> dict:
        """
        Compute factuality scores for workflow steps.

        Args:
            report_dict: Must contain 'workflow_steps' and 'references_map'.
            task_data: Task config for loading environment files.
            eval_data: Evaluation config (not directly used here).

        Returns:
            Dict with score, summary, and metric_result.
        """
        workflow_steps = report_dict.get("workflow_steps", [])
        references_map = report_dict.get("references_map", {})

        if not workflow_steps:
            return {
                "score": 0.0,
                "summary": "No workflow steps found in the report.",
                "metric_result": {
                    "factual_steps": [],
                    "unfactual_steps": [],
                    "total_steps": 0,
                    "detailed_results": [],
                },
            }

        env_files = task_loader.from_task_config_to_env_files(task_data)

        team_map, channel_map = self._build_mattermost_name_mappings(env_files)

        factuality_results = []

        for step in tqdm.tqdm(
            workflow_steps,
            desc="Checking workflow step factuality...",
            leave=False,
        ):
            item = step.get("item", "?")
            title = step.get("title", "")
            task = step.get("task", "")
            task_desc = step.get("task_description", "")
            justification = step.get("justification", "")

            claim = f"{title}. {task}. {task_desc}".strip()

            step_text = (
                f"{title} {task} {task_desc} {step.get('subtask_description', '')} {justification}"
            ).strip()
            citations = extract_step_citations(step_text, references_map)

            if citations:
                preprocessed = [
                    self._preprocess_citation(c, team_map, channel_map)
                    for c in citations
                ]
                verdict = get_factuality_verdict_multi(
                    claim,
                    preprocessed,
                    file_list=env_files,
                    model=self.model,
                )
            else:
                verdict = {
                    "is_factual": False,
                    "explanation": "No citations found for this workflow step",
                    "source_details": [],
                }

            verdict["step_item"] = item
            verdict["step_title"] = title
            verdict["citations"] = citations
            factuality_results.append(verdict)

        factual_steps = [r for r in factuality_results if r["is_factual"]]
        unfactual_steps = [r for r in factuality_results if not r["is_factual"]]
        total = len(factuality_results)
        score = len(factual_steps) / total if total > 0 else 0.0

        summary = (
            f"**Workflow Factuality Score:** {score:.4f} "
            f"({len(factual_steps)}/{total} steps factually supported)\n\n"
        )
        summary += "---\n\n"

        for r in factuality_results:
            status = "Factual" if r["is_factual"] else "Unfactual"
            summary += (
                f"**Step {r['step_item']} - {r['step_title']}:** {status}\n"
            )
            summary += f"  Citations: {r.get('citations', [])}\n"
            summary += f"  Explanation: {r.get('explanation', 'N/A')}\n\n"

        return {
            "score": score,
            "summary": summary,
            "metric_result": {
                "factual_steps": [
                    r["step_title"] for r in factual_steps
                ],
                "unfactual_steps": [
                    r["step_title"] for r in unfactual_steps
                ],
                "total_steps": total,
                "factuality_percentage": score * 100,
                "detailed_results": factuality_results,
            },
        }


    @staticmethod
    def _build_mattermost_name_mappings(
        env_files: Dict[str, Any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Scan Mattermost JSONL files for team/channel metadata and build
        display_name (lowercase) → internal_name mappings.

        Returns:
            (team_display_to_internal, channel_display_to_internal)
        """
        team_map: Dict[str, str] = {}
        channel_map: Dict[str, str] = {}

        if not env_files:
            return team_map, channel_map

        for fname, fpath in env_files.items():
            if not (
                fname.lower().startswith("mattermost_chat")
                and str(fname).endswith(".jsonl")
            ):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        rec_type = data.get("type")
                        if rec_type == "team":
                            info = data.get("team", {})
                            internal = info.get("name", "")
                            display = info.get("display_name", "")
                            if internal and display:
                                team_map[display.lower()] = internal.lower()
                        elif rec_type == "channel":
                            info = data.get("channel", {})
                            internal = info.get("name", "")
                            display = info.get("display_name", "")
                            if internal and display:
                                channel_map[display.lower()] = internal.lower()
                        elif rec_type == "post":
                            break
            except Exception as exc:
                logger.debug("Could not read Mattermost JSONL %s: %s", fname, exc)

        if team_map or channel_map:
            logger.info(
                "Mattermost name mappings: %d teams, %d channels",
                len(team_map),
                len(channel_map),
            )
        return team_map, channel_map

    @staticmethod
    def _decode_bytes_literal(text: str) -> str:
        """Decode Python bytes-literal representations embedded in citations.

        Report generation sometimes produces email subjects via
        ``str(bytes_obj)``, yielding strings like::

            b'Re: Subject \\xe2\\x80\\x94 details'

        This method strips the ``b'…'`` wrapper and converts ``\\xNN`` hex
        escapes back to proper UTF-8 characters so downstream matching
        against JSONL source data succeeds.
        """
        if not text:
            return text

        def _decode_inner(m: re.Match) -> str:
            inner = m.group(1)

            def _hex_to_chr(hm: re.Match) -> str:
                return chr(int(hm.group(1), 16))

            decoded = re.sub(r"\\x([0-9a-fA-F]{2})", _hex_to_chr, inner)
            try:
                return decoded.encode("latin-1").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                return decoded

        result = re.sub(r"b'((?:[^'\\]|\\.)*)'", _decode_inner, text)
        result = re.sub(r'b"((?:[^"\\]|\\.)*)"', _decode_inner, result)
        return result

    @staticmethod
    def _preprocess_citation(
        citation: str,
        team_map: Dict[str, str],
        channel_map: Dict[str, str],
    ) -> str:
        """Preprocess a citation before content lookup.

        Applies two transformations:

        1. **Bytes-literal decoding** – strips ``b'…'`` wrappers and
           converts ``\\xNN`` hex escapes to proper UTF-8 so email subject
           matching works (e.g. ``\\xe2\\x80\\x94`` → ``—``).
        2. **Mattermost display-name resolution** – replaces human-readable
           team/channel display names with the internal slugs used in JSONL
           post records so ``get_content`` can match them.

        Non-Mattermost citations that lack bytes-literal encoding are
        returned unchanged after step 1 (which is a no-op when there is
        nothing to decode).
        """
        if not citation:
            return citation

        citation = WorkflowFactuality._decode_bytes_literal(citation)

        citation_lower = citation.lower()
        is_mattermost = (
            "mattermost" in citation_lower
            and ("enterprise chat" in citation_lower or "user:" in citation_lower)
        )
        if not is_mattermost:
            return citation

        if not team_map and not channel_map:
            return citation

        match = re.search(
            r"User:\s*([^,\)]+).*?Team:\s*([^,\)]+).*?Channel:\s*([^,\)]+)",
            citation,
            re.IGNORECASE,
        )
        if not match:
            return citation

        user = match.group(1).strip()
        team_display = match.group(2).strip()
        channel_display = match.group(3).strip()

        team_resolved = team_map.get(team_display.lower(), team_display)
        channel_resolved = channel_map.get(channel_display.lower(), channel_display)

        return (
            f"Mattermost Message - Enterprise Chat "
            f"(User: {user}, Team: {team_resolved}, "
            f"Channel: {channel_resolved})"
        )
