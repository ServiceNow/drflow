import warnings

warnings.filterwarnings("ignore")

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from drflow import task_loader
from drflow.score_report_drflow import score_report_drflow


if __name__ == "__main__":
    task = task_loader.get_task_from_id(task_id="b2b_04")
    print(task.summary())

    dr_question = task.get_task_config()["dr_question"]
    print(f"DR Question: {dr_question}")
    report_file = "outputs/research_workspace/research_report.md"
    print("\n" + "=" * 60)
    print(f"DRFlow Workflow Evaluation (from file: {report_file})")
    print("=" * 60)

    drflow_metrics = [
        "factuality",
        "insight_recall",
        "insight_precision",
        "topology_order",
        "condition_resolvement",
        "personalized_comprehensiveness",
    ]
    
    drflow_score_dict_file = score_report_drflow(
        predicted_report_file=report_file,
        task=task,
        metrics=drflow_metrics,
        model="gpt-4o",
        savedir="results/minimal_drflow_from_file",
        verbose=True,
    )
    
    print("\n--- DRFlow Scores Summary (from file) ---")
    for metric_name in drflow_metrics:
        if metric_name in drflow_score_dict_file:
            print(f"  {metric_name}: {drflow_score_dict_file[metric_name]:.4f}")
