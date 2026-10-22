import json, os
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


import warnings

ROOT_DIR = Path(__file__).parent.parent


def get_all_subset_files_in_dir(dir_path: str):
    path = os.path.join(ROOT_DIR, "drflow", "data", "subsets")
    return glob.glob(os.path.join(path, "*.jsonl"))


def get_data_path(data_name) -> str:
    """Get path to a data resource"""
    if isinstance(data_name, str):
        data_name = Path(data_name)

    path = Path(ROOT_DIR) / data_name
    if not path.exists():
        warnings.warn(f"Data directory not found at {path}")

    return str(path)


def get_all_subsets() -> List[str]:
    """
    Get all subsets.
    """
    return [
        f.split("/")[-1].split(".")[0].replace("subset_", "")
        for f in get_all_subset_files()
    ]


def get_task_config(task: dict):
    task_id = task["id"]
    subset = task["subset"]
    with open(get_data_path(f"{subset}/{task_id}/task.json"), "r") as f:
        task_config = json.load(f)

    with open(get_data_path(f"{subset}/{task_id}/eval.json"), "r") as f:
        eval_qa = json.load(f)

    return task_config, eval_qa


class Task:
    """
    A class to represent a task that allows us to access the task and eval configs.
    """

    def __init__(self, task_path: Union[str, Path], ignore_config=False):
        self.task_path = str(task_path)
        task_path = self.task_path

        config_path = get_data_path(f"{task_path}")

        if ignore_config:
            task_config = None
            eval_config = None
            env_config = None
        else:
            assert Path(
                config_path
            ).exists(), f"Task configuration file not found at {config_path}"
            with open(get_data_path(f"{config_path}/task.json"), "r") as f:
                task_config = json.load(f)

            eval_file = get_data_path(f"{config_path}/eval.json")
            if not Path(eval_file).exists():
                return task_config, None

            with open(get_data_path(f"{config_path}/eval.json"), "r") as f:
                eval_config = json.load(f)

            env_file = get_data_path(f"{config_path}/env.json")
            with open(env_file, "r") as f:
                env_config = json.load(f)

        self.task_config = task_config
        self.eval_config = eval_config
        self.env_config = env_config

    def get_task_and_eval(self) -> Tuple[Dict, Optional[Dict]]:
        return self.task_config, self.eval_config

    def get_local_files_list(self) -> str:
        env_files = self.get_env_files()
        file_list = [str(env_files[file]) for file in env_files]

        for file in file_list:
            assert Path(file).exists(), f"File {file} does not exist"
        return file_list

    def get_task_config(self) -> Dict:
        return self.task_config

    def get_eval_config(self) -> Dict:
        return self.eval_config

    def get_path(self) -> str:
        return get_data_path(self.task_path)

    def get_id(self) -> str:
        return self.task_config["task_id"]

    def get_dr_question(self) -> str:
        task_config = self.get_task_config()
        return task_config["dr_question"]

    def get_per_insight_eval_config(self) -> Dict:
        eval_config = self.get_eval_config()
        return eval_config["dr_report_evaluation_qa"]

    def __repr__(self) -> str:
        return f"<Task {self.get_id()}>"

    def get_stats(self) -> Dict:
        eval_config = self.get_eval_config()
        stats = {}

        stats["dr_question"] = self.get_dr_question()
        stats_dict = {}
        stats_dict["insights"] = len(eval_config["dr_report_evaluation_qa"])

        for qa in eval_config["dr_report_evaluation_qa"]:
            for file_path in qa.get("supporting_file_paths", []):
                file_extension = file_path.split(".")[-1]
                if file_extension not in stats_dict:
                    stats_dict[file_extension] = 0
                stats_dict[file_extension] += 1
        stats["stats"] = " | ".join([f"{v} {k}" for k, v in stats_dict.items()])
        stats["url"] = self.get_supporting_urls()
        return stats

    def save_supporting_files(self, file_paths: List[str], dst_dir: str):
        import shutil

        for file_path in file_paths:
            shutil.copy(file_path, dst_dir)

    def summary(self, include_eval=False, return_dict=False) -> str:
        task_config = self.get_task_config()
        env_config = self.get_env_config()
        persona_name = task_config["persona"]["name"]
        persona_department = task_config["persona"]["department"]
        persona_username = task_config["persona"]["username"]
        persona_password = task_config["persona"]["password"]
        company_name = task_config["company_info"]["name"]
        company_industry = task_config["company_info"]["industry"]
        task_difficulty = task_config["level"]

        env_files = env_config["env_files"]

        count_file_type_by_app = {}
        file2app = {}
        for f in env_files:
            app_name = f["app"]
            file2app[f["destination"]] = app_name
            if app_name not in count_file_type_by_app:
                count_file_type_by_app[app_name] = {}

            file_extension = f["destination"].split(".")[-1]
            if file_extension not in count_file_type_by_app[app_name]:
                count_file_type_by_app[app_name][file_extension] = 0
            count_file_type_by_app[app_name][file_extension] += 1
        count_file_type_by_app_str = ""
        for app_name, file_type_count in count_file_type_by_app.items():
            count_file_type_by_app_str += f"{app_name}:\n"
            for file_extension, count in file_type_count.items():
                count_file_type_by_app_str += f"    - {file_extension}: {count}\n"

        summary = f"""
Task: {self.get_id()}
Path: {self.get_path()}
--------------------------------
(a) DR Question:
{self.get_dr_question()}

***

(b) Company Structure:
{company_name} - {company_industry}

***

(c) Persona:
{persona_name} - {persona_department}
username: {persona_username}
password: {persona_password}


***
(d) Env Files:
{count_file_type_by_app_str}
        """
        if include_eval:
            eval_config = self.get_eval_config()
            insight_meta = {"n_insights": len(eval_config["dr_report_evaluation_qa"])}
            app_count = {}
            file_extension_count = {}
            dest2source = {
                f["destination"]: f["source"] for f in task_config["env_files"]
            }
            distractors = list(dest2source.keys())
            for qa in eval_config["dr_report_evaluation_qa"]:
                for file_path in qa.get("supporting_file_paths", []):
                    file_extension = file_path.split(".")[-1]
                    insight_key = f"{file_extension}_{file2app.get(file_path, 'tba')}"
                    if insight_key not in insight_meta:
                        insight_meta[insight_key] = 0
                    insight_meta[insight_key] += 1
                    app_name = file2app.get(file_path, "tba")
                    if app_name not in insight_meta:
                        app_count[app_name] = 0
                    app_count[app_name] += 1
                    if file_extension not in file_extension_count:
                        file_extension_count[file_extension] = 0
                    file_extension_count[file_extension] += 1
                    if file_path in distractors:
                        distractors.remove(file_path)
            insight_meta["distractor_count"] = len(distractors)
            insight_meta_str = ""
            for file_extension, count in insight_meta.items():
                insight_meta_str += f"    - {file_extension}: {count}\n"
            summary += f"""
***
(d) Eval Config:
{insight_meta_str}
"""
        if return_dict:
            summary_dict = {
                "dr_question": self.get_dr_question(),
                "persona_department": task_config["persona"]["department"],
                "persona_responsibilities": task_config["persona"]["responsibilities"],
                "difficulty": task_difficulty,
                "app_count": len(app_count.keys()),
                "file_extension_count": len(file_extension_count.keys()),
            }
            summary_dict.update(insight_meta)
            summary_dict["task_id"] = self.get_id()
            return summary, summary_dict
        return summary

    def get_insights(self) -> str:
        task_config = self.get_task_config()
        return task_config["insights"]

    def get_info(self) -> Dict:
        info_file = get_data_path(f"{os.path.dirname(self.task_path)}/info.json")
        with open(info_file, "r") as f:
            info = json.load(f)
        return info

    def __str__(self) -> str:
        return self.__repr__()

    def get_supporting_urls(self) -> List[str]:
        eval_config = self.get_eval_config()
        urls = []
        for qa in eval_config["dr_report_evaluation_qa"]:
            for file_path in qa.get("supporting_urls", []):
                if file_path.startswith("http"):
                    urls.append(file_path)
        assert len(urls) <= 1, "Only one supporting url is allowed"
        if len(urls) == 0:
            return "N/A"
        return urls[0]

    def get_env_config(self) -> Dict:
        return self.env_config

    def get_env_files(self) -> List[str]:
        return from_task_config_to_env_files(self.get_task_config())

    def get_supporting_files(self) -> List[str]:
        eval_config = self.get_eval_config()
        return eval_config["supporting_file_paths"]

    def load_supporting_file(self, file_path: str) -> str:
        env_files = {}
        task_dict = self.get_task_config()
        for env_file in task_dict["env_files"]:
            env_files[env_file["destination"]] = get_data_path(env_file["source"])

        with open(env_files[file_path], "r") as f:
            content = f.read()
        return content

    def get_context(self) -> str:
        task_config = self.get_task_config()

        with open(
            get_data_path(task_config["context"].replace("research_questions_v2/", "")),
            "r",
        ) as f:
            context = json.load(f)
        return context

    def get_supporting_files(self) -> List[str]:
        eval_config = self.get_eval_config()
        return eval_config["supporting_file_paths"]

    def get_task_summary(self) -> Dict:
        task_config = self.get_task_config()
        summary = f"# {task_config['name']}\n\n"
        summary += f"## Task Description\n\n"
        summary += f"{task_config['description']}\n\n"
        summary += f"## Task Configuration\n\n"
        summary += f"{json.dumps(task_config, indent=2)}\n\n"
        return summary


def from_task_config_to_env_files(task_config: Dict) -> Dict:
    task = get_task_from_id(task_config["task_id"])
    env_config = task.get_env_config()
    env_files = {}
    for env_file in env_config["env_files"]:
        path = Path(get_data_path(env_file["source"]))
        env_files[path.name] = path
    return env_files


class TaskLoader:
    def __init__(self, subset: str):
        self.subset = subset
        self.tasks = get_tasks_from_subset(subset)

    def get_task_from_id(self, task_id: str) -> Task:
        return get_task_from_id(task_id)

    def __getitem__(self, task_idx: int) -> Task:
        return self.tasks[task_idx]


def get_task_from_id(task_id: str) -> Task:
    path = Path(__file__).parent / "data" / "tasks" / task_id / "config"
    task = Task(path)
    return task


def save_all_subset_to_csv() -> List[str]:
    files = get_all_subset_files_in_dir("validation")

    for file in files:
        parent_dir = os.path.dirname(os.path.dirname(file))
        subset_dir = os.path.join(parent_dir, "subsets")
        fname = os.path.basename(file).replace(".jsonl", ".csv")
        csv_path = os.path.join(subset_dir, fname)
        os.makedirs(subset_dir, exist_ok=True)
        tasks = get_tasks_from_subset_file(file)
        summary_list = [
            task.summary(return_dict=True, include_eval=True)[1] for task in tasks
        ]
        df = pd.DataFrame(summary_list)
        df.fillna(0, inplace=True)
        columns = []
        for col in df.columns:
            if col not in [
                "task_id",
                "persona_department",
                "persona_responsibilities",
            ]:
                columns.append(col)
        columns.append("persona_department")
        columns.append("persona_responsibilities")
        columns.append("task_id")
        df = df[columns]
        df.to_csv(csv_path, index=True)


def get_all_subset_files() -> List[str]:
    return get_all_subset_files_in_dir("validation")


def get_tasks_from_subset_file(file_path: str) -> List[Task]:
    with open(file_path, "r") as f:
        all_tasks = [json.loads(line) for line in f]
    return [Task(task["path"]) for task in all_tasks]


def get_tasks_from_subset(subset: str) -> List[Task]:
    """
    Get all tasks from a specific subset.

    Args:
        subset (str): The subset of tasks (e.g., "train", "dev", "test").

    Returns:
        List[Dict]: A list of task configurations for the specified subset.
    """

    subset_path = get_data_path(f"drflow/data/subsets/{subset}.jsonl")
    if not Path(subset_path).exists():
        raise FileNotFoundError(
            f"Subset file not found at {subset_path} - available subsets {get_all_subset_files()}"
        )

    with open(subset_path, "r") as f:
        all_tasks = [Task(json.loads(line)["path"]) for line in f]

    return all_tasks


def get_task_ids_from_subset(subset: str) -> List[str]:
    all_tasks = get_tasks_from_subset(subset)
    return [task.get_id() for task in all_tasks]


def get_all_tasks(verbose: bool = False):
    files = get_all_subset_files()
    tasks_list = []
    for file in files:
        tasks = get_tasks_from_subset_file(file)
        tasks_list.extend(tasks)
        if verbose:
            print(f"\n- Loaded {len(tasks)} tasks from {file}")

    seen_task_ids = set()
    unique_tasks = []
    for task in tasks_list:
        task_id = task.get_id()
        if task_id not in seen_task_ids:
            seen_task_ids.add(task_id)
            unique_tasks.append(task)

    if verbose:
        print(
            f"Total unique tasks: {len(unique_tasks)} (removed {len(tasks_list) - len(unique_tasks)} duplicates)"
        )

    return unique_tasks


def get_tasks_df(subset: str = None, saveto=None):
    """
    Load tasks from the tasks folder and create a DataFrame with relevant columns.

    Args:
        subset (str): If provided, load only tasks from this subset (e.g., 'val', 'train', 'test').
                     If None, load all tasks from the tasks directory.
        saveto (str): If provided, save the DataFrame to this CSV file path.

    Returns:
        pd.DataFrame: DataFrame with columns matching the table structure
    """
    task_data = []

    if subset is not None:
        try:
            tasks = get_tasks_from_subset(subset)
            task_ids_to_process = [task.get_id() for task in tasks]
        except FileNotFoundError as e:
            raise ValueError(f"Error loading subset '{subset}': {e}")

    else:
        tasks_dir = ROOT_DIR / Path("drflow/data/tasks")
        task_ids_to_process = [
            task_dir.name for task_dir in tasks_dir.iterdir() if task_dir.is_dir()
        ]

    for task_id in task_ids_to_process:
        task_dir = ROOT_DIR / Path("drflow/data/tasks") / task_id
        dr_question_file = task_dir / "dr_question.json"
        info_file = task_dir / "info.json"
        context_file = task_dir / "context.json"

        if not (dr_question_file.exists() and info_file.exists()):
            continue

        try:
            with open(dr_question_file, "r") as f:
                dr_question_data = json.load(f)

            with open(info_file, "r") as f:
                info_data = json.load(f)

            context_data = {}
            if context_file.exists():
                with open(context_file, "r") as f:
                    context_data = json.load(f)

            files = info_data.get("files", [])
            n_supporting = sum(1 for f in files if f.get("type") == "supporting")
            n_distractor = sum(1 for f in files if f.get("type") == "distractor")

            persona_info = context_data.get("persona", {})
            persona_name = persona_info.get("name", "")

            company_info = context_data.get("company_info", {})
            company_name = company_info.get("name", "")

            url_info = context_data.get("url", {})
            url_date = url_info.get("date", "")

            row = {
                "task_id": task_id,
                "dr_question": dr_question_data.get("dr_question", ""),
                "justification": persona_info.get("justification", ""),
                "url": dr_question_data.get("url", ""),
                "url_date": url_date,
                "persona": persona_name,
                "industry": dr_question_data.get("industry", ""),
                "company": company_name,
                "domain": dr_question_data.get("domain", ""),
                "n_supporting": n_supporting,
                "n_distractor": n_distractor,
            }

            task_data.append(row)

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error processing task {task_id}: {e}")

    df = pd.DataFrame(task_data)

    df = df.sort_values("task_id").reset_index(drop=True)

    if saveto:
        del df["justification"]
        df.to_csv(saveto, index=False)

    return df


def get_facts_df(task_id, saveto=None):
    """
    Load facts from qa_dict.json files for a specific task.

    Args:
        task_id (str): The task ID (e.g., 'DR0001')

    Returns:
        pd.DataFrame: DataFrame with fact information from qa_dict.json files
    """
    task_dir = ROOT_DIR / Path(f"drflow/data/tasks/{task_id}")
    files_dir = task_dir / "files"

    if not files_dir.exists():
        print(f"Files directory not found for task {task_id}")
        return pd.DataFrame()

    facts_data = []

    for subdir in files_dir.iterdir():
        if not subdir.is_dir():
            continue

        qa_dict_file = subdir / "qa_dict.json"

        if not qa_dict_file.exists():
            continue

        try:
            with open(qa_dict_file, "r") as f:
                qa_data = json.load(f)

            insight_id = qa_data.get("insight_id", "")
            qa_type = qa_data.get("qa_type", "")

            type_mapping = {"insight": "supporting", "distractor": "distractor"}
            fact_type = type_mapping.get(qa_type, qa_type)

            external_or_internal = "internal"

            row = {
                "fact_id": insight_id,
                "type": fact_type,
                "external_or_internal": external_or_internal,
                "question": qa_data.get("specific_question", ""),
                "answer": qa_data.get("answer", ""),
                "justification": qa_data.get("justification", ""),
                "dr_question": qa_data.get("dr_question", ""),
            }

            facts_data.append(row)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing qa_dict in {subdir.name}: {e}")
            continue

    df = pd.DataFrame(facts_data)

    if not df.empty:
        df = df.sort_values("fact_id").reset_index(drop=True)

    if saveto:
        del df["justification"]
        df.to_csv(saveto, index=False)

    return df


if __name__ == "__main__":
    df = get_tasks_df(
        subset="val", saveto=ROOT_DIR / Path("drflow/data/summary/dr_questions.csv")
    )
    print(df)

    for task_id in df["task_id"]:
        path = ROOT_DIR / Path(f"drflow/data/summary/facts/{task_id}_facts.csv")
        os.makedirs(path.parent, exist_ok=True)
        df = get_facts_df(
            task_id,
            saveto=path,
        )
        print(df)
