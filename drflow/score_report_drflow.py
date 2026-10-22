"""
DRFlow Score Report - Main scoring orchestrator for DRBench Workflow evaluation.

This module provides the `score_report_drflow` function that:
1. Parses personalized workflow steps from the generated report.
2. Runs each requested DRFlow metric (with dependency resolution).
3. Saves detailed evaluation reports (Markdown + JSON).

Usage (from agent output):
    from drflow.score_report_drflow import score_report_drflow

    scores = score_report_drflow(
        predicted_report=report_text_or_dict,
        task=task,
        metrics=["insight_recall", "insight_precision", "topology_order",
                 "condition_resolvement", "personalized_comprehensiveness",
                 "factuality"],
        savedir="results/drflow_eval",
    )

Usage (from a saved report file):
    scores = score_report_drflow(
        predicted_report_file="outputs/research_workspace/research_report.md",
        task=task,
        savedir="results/drflow_eval",
    )
"""

import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import dotenv

dotenv.load_dotenv()
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from drflow import task_loader
from drflow.metrics_drflow import (
    get_drflow_metric,
    resolve_metric_order,
    get_all_drflow_metric_names,
)
from drflow.metrics_drflow.workflow_parser import parse_workflow_from_report

MAX_REPORT_LENGTH = 80000


def score_report_drflow(
    predicted_report_text: str = None,
    task_config: Dict[str, Any] = None,
    eval_config: Dict[str, Any] = None,
    metrics: List[str] = None,
    savedir: str = None,
    verbose: bool = True,
    model: str = "gpt-4o-mini",
    predicted_report: Any = None,
    predicted_report_dict: Dict[str, Any] = None,
    predicted_report_file: str = None,
    task=None,
    timing_dict: Dict = None,
) -> dict:
    """
    Evaluate a generated report's personalized workflow against golden steps.

    This function:
    1. Extracts personalized workflow steps from the report text.
    2. Runs each requested DRFlow metric in dependency order.
    3. Optionally saves detailed Markdown and JSON reports.

    The report can be provided in several ways (checked in this order):
    - ``predicted_report_file``: path to a markdown file on disk.
    - ``predicted_report``: a string (report text) or dict with
      ``report_text`` / ``workflow_steps`` keys.
    - ``predicted_report_text``: raw report string.
    - ``predicted_report_dict``: pre-parsed workflow dict.

    Args:
        predicted_report_text: The full report text (markdown format).
        task_config: Task configuration dict (from task.get_task_config()).
        eval_config: Evaluation config dict (from task.get_eval_config()).
        metrics: List of metric names to compute. If None, runs all metrics.
        savedir: Directory to save evaluation reports. If None, no files saved.
        verbose: If True, print progress and scores.
        model: LLM model name for metric computation.
        predicted_report: Convenience arg - can be str (report text) or dict.
        predicted_report_dict: Pre-parsed workflow dict (optional).
        predicted_report_file: Path to a markdown/text file containing the
            generated report. The file is read and used as predicted_report_text.
        task: A Task object (alternative to passing task_config/eval_config).
        timing_dict: Optional dict with setup_time and generate_report_time.

    Returns:
        Dict mapping metric names to scores, plus metadata.
    """
    scoring_start_time = time.perf_counter()


    if predicted_report_file is not None:
        if verbose:
            print(f"Reading report from file: {predicted_report_file}")
        with open(predicted_report_file, "r", encoding="utf-8") as f:
            predicted_report_text = f.read()

    elif predicted_report is not None:
        if isinstance(predicted_report, str):
            predicted_report_text = predicted_report
        elif isinstance(predicted_report, dict):
            predicted_report_text = predicted_report.get("report_text", "")
            if not predicted_report_text:
                predicted_report_text = predicted_report.get("text", "")
            predicted_report_dict = predicted_report
        else:
            raise ValueError(
                "predicted_report must be a string or a dictionary"
            )

    if task is not None:
        task_config = task.get_task_config()
        eval_config = task.get_eval_config()

    if predicted_report_text is None and predicted_report_dict is None:
        raise ValueError(
            "Either predicted_report_text, predicted_report, or "
            "predicted_report_file must be provided."
        )

    if predicted_report_text and len(predicted_report_text) > MAX_REPORT_LENGTH:
        if verbose:
            print(
                f"WARNING: Report text ({len(predicted_report_text)} chars) "
                f"exceeds {MAX_REPORT_LENGTH} limit. Truncating."
            )
        predicted_report_text = predicted_report_text[-MAX_REPORT_LENGTH:]

    if metrics is None:
        metrics = get_all_drflow_metric_names()

    if verbose:
        print("Parsing personalized workflow from report...")

    if predicted_report_dict is not None and "workflow_steps" in predicted_report_dict:
        workflow_dict = predicted_report_dict.copy()
        if "report_text" not in workflow_dict and predicted_report_text:
            workflow_dict["report_text"] = predicted_report_text
    elif predicted_report_text:
        workflow_dict = parse_workflow_from_report(predicted_report_text)
    else:
        raise ValueError(
            "Cannot parse workflow: no report text or workflow dict provided."
        )

    workflow_steps = workflow_dict.get("workflow_steps", [])
    if verbose:
        print(
            f"  Found {len(workflow_steps)} personalized workflow steps."
        )
        if workflow_dict.get("workflow_flowchart"):
            print("  Found personalized flowchart.")
        print(
            f"  Found {len(workflow_dict.get('references_map', {}))} "
            f"reference mappings."
        )

    if not workflow_steps:
        if verbose:
            print(
                "WARNING: No personalized workflow steps found in the report. "
                "All metric scores will be 0."
            )

    ordered_metrics = resolve_metric_order(metrics)

    if verbose:
        print(f"\nMetrics to compute (ordered): {ordered_metrics}")

    score_dict = {}
    metric_results = []

    for metric_name in ordered_metrics:
        if verbose:
            print(f"\nComputing {metric_name}...")

        metric = get_drflow_metric(metric_name, model=model)
        metric_result = metric.compute(
            report_dict=workflow_dict,
            task_data=task_config,
            eval_data=eval_config,
        )

        score_dict[metric_name] = metric_result["score"]
        metric_result["name"] = metric_name
        metric_results.append(metric_result)

        if metric_name == "insight_precision":
            precision_matches = metric_result.get(
                "metric_result", {}
            ).get("precision_matches", [])
            workflow_dict["precision_matches"] = precision_matches
            if verbose:
                print(
                    f"  -> {len(precision_matches)} precision matches "
                    f"stored for dependent metrics."
                )

        if verbose:
            print(f"  {metric_name}: {score_dict[metric_name]:.4f}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if savedir:
        os.makedirs(savedir, exist_ok=True)
        _save_markdown_report(
            savedir=savedir,
            timestamp=timestamp,
            task_config=task_config,
            predicted_report_text=predicted_report_text,
            workflow_dict=workflow_dict,
            score_dict=score_dict,
            metric_results=metric_results,
        )
        _save_json_report(
            savedir=savedir,
            timestamp=timestamp,
            task_config=task_config,
            eval_config=eval_config,
            predicted_report_text=predicted_report_text,
            workflow_dict=workflow_dict,
            score_dict=score_dict,
            metric_results=metric_results,
            timing_dict=timing_dict,
            scoring_start_time=scoring_start_time,
        )
        if verbose:
            print(f"\nDRFlow evaluation reports saved to: {savedir}")

    score_dict["predicted_report_text"] = predicted_report_text
    score_dict["task_id"] = task_config.get("task_id", "unknown")
    score_dict["question"] = task_config.get("dr_question", "")
    score_dict["num_predicted_steps"] = len(workflow_steps)
    score_dict["num_golden_steps"] = len([
        qa for qa in eval_config.get("dr_report_evaluation_qa", [])
        if qa.get("type") == "workflow_step"
    ])

    return score_dict


def _save_markdown_report(
    savedir: str,
    timestamp: str,
    task_config: Dict,
    predicted_report_text: str,
    workflow_dict: Dict,
    score_dict: Dict,
    metric_results: List[Dict],
):
    """Save detailed Markdown evaluation report."""
    filepath = os.path.join(savedir, "drflow_evaluation_report.md")

    with open(filepath, "w") as f:
        f.write("# DRFlow Workflow Evaluation Report\n\n")
        f.write(f"**Timestamp:** {timestamp}\n\n")
        f.write(f"**Task ID:** `{task_config.get('name', task_config.get('task_id', 'unknown'))}`\n\n")
        f.write(f"**Question:** {task_config.get('dr_question', 'N/A')}\n\n")

        f.write("## Overall Scores\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        for mr in metric_results:
            name = mr["name"]
            score = mr["score"]
            f.write(f"| {name} | {score:.4f} |\n")
        f.write("\n")

        f.write("## Extracted Personalized Workflow Steps\n\n")
        workflow_steps = workflow_dict.get("workflow_steps", [])
        if workflow_steps:
            for step in workflow_steps:
                f.write(
                    f"### Step {step.get('item', '?')}: "
                    f"{step.get('title', 'N/A')}\n\n"
                )
                f.write(f"**Task:** {step.get('task', 'N/A')}\n\n")
                f.write(
                    f"**Description:** {step.get('task_description', 'N/A')}\n\n"
                )
                subtask = step.get("subtask_description", "")
                if subtask:
                    f.write(f"**Subtask/Conditions:** {subtask}\n\n")
                prereq = step.get("prerequisite", "")
                if prereq:
                    f.write(f"**Prerequisite:** Step {prereq}\n\n")
                f.write("---\n\n")
        else:
            f.write("No personalized workflow steps found.\n\n")

        flowchart = workflow_dict.get("workflow_flowchart", "")
        if flowchart:
            f.write("## Personalized Flowchart\n\n")
            f.write(f"```mermaid\n{flowchart}\n```\n\n")

        f.write("## Detailed Metric Results\n\n")
        for mr in metric_results:
            f.write(f"### {mr['name']}\n\n")
            f.write(f"**Score:** {mr['score']:.4f}\n\n")
            f.write(f"{mr.get('summary', 'No summary available.')}\n\n")
            f.write("---\n\n")


def _save_json_report(
    savedir: str,
    timestamp: str,
    task_config: Dict,
    eval_config: Dict,
    predicted_report_text: str,
    workflow_dict: Dict,
    score_dict: Dict,
    metric_results: List[Dict],
    timing_dict: Dict = None,
    scoring_start_time: float = None,
):
    """Save detailed JSON evaluation report."""
    filepath = os.path.join(savedir, "drflow_evaluation_report.json")

    scoring_time = None
    if scoring_start_time:
        scoring_time = time.perf_counter() - scoring_start_time

    serializable_workflow = {}
    for key, value in workflow_dict.items():
        if key != "report_text":
            serializable_workflow[key] = value

    full_report = {
        "task_id": task_config.get("task_id", "unknown"),
        "question": task_config.get("dr_question", ""),
        "timestamp": timestamp,
        "overall_scores": {
            mr["name"]: mr["score"] for mr in metric_results
        },
        "num_predicted_steps": len(
            workflow_dict.get("workflow_steps", [])
        ),
        "num_golden_steps": len([
            qa
            for qa in (eval_config or {}).get(
                "dr_report_evaluation_qa",
                workflow_dict.get("_eval_qa", []),
            )
            if qa.get("type") == "workflow_step"
        ]),
        "workflow_data": serializable_workflow,
        "metrics_details": [],
    }

    if timing_dict is not None and scoring_time is not None:
        total_time = (
            timing_dict.get("setup_time", 0)
            + timing_dict.get("generate_report_time", 0)
            + scoring_time
        )
        full_report["timing"] = {
            "setup_time": timing_dict.get("setup_time", 0),
            "generate_report_time": timing_dict.get(
                "generate_report_time", 0
            ),
            "score_report_time": scoring_time,
            "total_time": total_time,
        }

    for mr in metric_results:
        metric_detail = {
            "name": mr["name"],
            "score": mr["score"],
            "summary": mr.get("summary", ""),
        }
        metric_detail.update(mr.get("metric_result", {}))
        full_report["metrics_details"].append(metric_detail)

    with open(filepath, "w") as f:
        json.dump(full_report, f, indent=4, default=str)
