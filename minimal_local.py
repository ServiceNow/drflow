"""Generate and evaluate a DRFLOW workflow using a task's local files."""

import argparse
import logging
from datetime import datetime
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="b2b_04", help="Bundled task ID (default: b2b_04)")
    parser.add_argument("--model", default="openrouter/openai/gpt-5.2")
    parser.add_argument("--embedding-model", default="openrouter/openai/text-embedding-ada-002")
    parser.add_argument("--eval-model", default="gpt-4o-mini")
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")

    from drflow import task_loader
    from drflow.agents.drflow_agent.drflow_agent import DrFlowAgent
    from drflow.score_report_drflow import score_report_drflow

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    task = task_loader.get_task_from_id(task_id=args.task_id)
    print(task.summary())
    dr_question = task.get_task_config()["dr_question"]
    local_files = task.get_local_files_list()
    print(f"DR Question: {dr_question}")
    print(f"Local files: {len(local_files)}")

    model_slug = args.model.split("/")[-1].replace(":", "_")
    run_id = f"{args.task_id}_{model_slug}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    result_dir = f"outputs/research_workspace/{run_id}"
    eval_save_dir = f"results/minimal_local_drflow/{run_id}"
    dr_agent = DrFlowAgent(
        model=args.model,
        embedding_model=args.embedding_model,
        max_iterations=args.max_iterations,
        concurrent_actions=1,
        verbose=True,
        workspace_dir=result_dir,
        vector_store_base_dir=f"outputs/vector_stores/{run_id}",
    )
    report = dr_agent.generate_report(query=dr_question, local_files=local_files, results_dir=result_dir)

    drflow_metrics = [
        "factuality",
        "insight_recall",
        "insight_precision",
        "topology_order",
        "condition_resolvement",
        "personalized_comprehensiveness",
    ]
    scores = score_report_drflow(
        predicted_report=report,
        task=task,
        metrics=drflow_metrics,
        model=args.eval_model,
        savedir=eval_save_dir,
        verbose=True,
    )

    print(f"\n--- DRFLOW Scores Summary for task {args.task_id} ---")
    for metric_name in drflow_metrics:
        if metric_name in scores:
            print(f"  {metric_name}: {scores[metric_name]:.4f}")
    print(f"  Predicted Steps: {scores.get('num_predicted_steps', 0)}")
    print(f"  Golden Steps: {scores.get('num_golden_steps', 0)}")
    print(f"Research report directory: {result_dir}")
    print(f"Evaluation directory: {eval_save_dir}")


if __name__ == "__main__":
    main()
