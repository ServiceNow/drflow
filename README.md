# DRFLOW: A Deep Research Benchmark for Personalized Workflow Prediction

<p align="center">
  <img src="assets/drflow_image.png" alt="DRFLOW logo" width="640">
</p>

[![Paper on arXiv](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.18191)
[![Dataset on Hugging Face](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-FFD21E.svg)](https://huggingface.co/datasets/ServiceNow/DRFlow)

**`DRFLOW`** is a deep research benchmark designed to evaluate agents on
**personalized workflow prediction** for complex enterprise tasks.

It tests an agent's ability to connect evidence across **heterogeneous data
sources** and turn that evidence into an **ordered sequence of action steps**
tailored to a user's goals and constraints.

## News

- DRFLOW has been accepted at EMNLP 2026! 🎉
- The DRFLOW dataset is now available on [Hugging Face](https://huggingface.co/datasets/ServiceNow/DRFlow)! 🤗

## Quick Start

Run the following commands from this directory. Python 3.10 or later is required.

### Install Requirements

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
cp env_template .env
```

Populate `.env` with your credentials. The default local and Docker examples use
`OPENROUTER_API_KEY` for report generation, embeddings, and evaluation. Set
`SERPER_API_KEY` to enable internet search. Keep `.env` private; it is ignored by
Git. Values in `.env` override existing shell environment variables.

### Run Locally Without Docker

```bash
python minimal_local.py
```

This runs only the DRFLOW agent on the bundled `b2b_04` task, using the task's
local files directly, then evaluates the generated workflow with all six DRFLOW
metrics. Docker installation and service startup are unnecessary for this mode.
Model APIs still require an internet connection and credentials.

Choose another bundled task or change the model settings:

```bash
python minimal_local.py --task-id education_10
python minimal_local.py --task-id legal_01 --max-iterations 5
python minimal_local.py --help
```

Each run writes its report and workflow spreadsheet under
`outputs/research_workspace/<run_id>/`, its vector store under
`outputs/vector_stores/<run_id>/`, and its Markdown and JSON evaluation reports
under `results/minimal_local_drflow/<run_id>/`.

### Run with Docker Services

To use the simulated enterprise applications, install
[Docker](https://www.docker.com/get-started/) and build the services image:

```bash
cd services
make local-build
cd ..
python minimal_docker.py
```

The initial image build takes around 30 minutes and only needs to be done once.
This example also uses the bundled `b2b_04` task.

## Citation

If you use DRFLOW in your work, please cite:

```bibtex
@inproceedings{khondaker-etal-2026-drflow,
    title = "{DRFLOW}: A Deep Research Benchmark for Personalized Workflow Prediction",
    author = "Khondaker, Md Tawkat Islam  and
      Li, Raymond  and
      Abdul-Mageed, Muhammad  and
      Lakshmanan, Laks V. S.  and
      Laradji, Issam H.",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1344/",
    doi = "10.18653/2026.findings-emnlp.1344",
    pages = "28492--28558",
    ISBN = "979-8-89176-462-0"
}
```

## Acknowledgement

This repository is built on [DRBench](https://github.com/ServiceNow/drbench).
