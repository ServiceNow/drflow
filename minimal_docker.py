import warnings
import time
import logging


from drflow import drflow_enterprise_space
from drflow import task_loader
from drflow.agents.drflow_agent.drflow_agent import DrFlowAgent
from drflow.score_report_drflow import score_report_drflow

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


if __name__ == "__main__":

    TASK_ID = "b2b_04"
    task = task_loader.get_task_from_id(task_id=TASK_ID)
    print(task.summary())

    dr_question = task.get_task_config()["dr_question"]
    print(f"DR Question: {dr_question}")
    env = drflow_enterprise_space.DrFlowEnterpriseSearchSpace(
        task=task.get_path(),
        start_container=True,
        auto_ports=True,
    )

    print(env.get_available_apps())

    model_name = "openrouter/openai/gpt-5.2"

    model_slug = model_name.split("/")[-1]
    run_id = f"{TASK_ID}_{model_slug}_{time.strftime('%Y%m%d_%H%M%S')}"
    vector_store_dir = f"./outputs/vector_stores/{run_id}"
    result_dir = f"./outputs/research_workspace/{run_id}"
    eval_save_dir = f"./results/minimal_drflow/{run_id}"
    dr_agent = DrFlowAgent(
        model=model_name,
        max_iterations=3,
        concurrent_actions=1,
        verbose=True,
        vector_store_base_dir=vector_store_dir,
    )

    report = dr_agent.generate_report(query=dr_question, env=env, results_dir=result_dir)

    print("\n" + "=" * 60)
    print("DRFlow Workflow Evaluation (from agent output)")
    print("=" * 60)

    drflow_metrics = [
        "factuality",
        "insight_recall",
        "insight_precision",
        "topology_order",
        "condition_resolvement",
        "personalized_comprehensiveness",
    ]

    drflow_score_dict = score_report_drflow(
        predicted_report=report,
        task=task,
        metrics=drflow_metrics,
        savedir=eval_save_dir,
        verbose=True,
    )

    print(f"\n--- DRFlow Scores Summary for task id {TASK_ID} ---")
    for metric_name in drflow_metrics:
        if metric_name in drflow_score_dict:
            print(f"  {metric_name}: {drflow_score_dict[metric_name]:.4f}")

    print(f"Research Report directory: {result_dir}")
