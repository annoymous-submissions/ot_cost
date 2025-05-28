# A Universal Metric of Dataset Similarity for Cross-silo Federated Learning

## Project Overview

This repository contains a Python-based research framework for conducting Federated Learning (FL) experiments and analyzing the similarity between client data distributions using Optimal Transport (OT). The framework supports:

*   **Federated Learning:** Running FL experiments using algorithms like FedAvg, FedProx, PFedMe, Ditto, and Local training across various datasets.
*   **Data Heterogeneity:** Simulating different types of data heterogeneity, including:
    *   Label Distribution Skew (Dirichlet distribution)
    *   Feature Distribution Shift (Mean shift, scale shift, rotation/tilt for tabular; Geometric rotation for images)
    *   Concept Shift (Changing decision boundaries)
    *   Pre-defined splits based on real-world site data.
*   **Optimal Transport Analysis:** Calculating the similarity/distance between client data representations (activations) using Direct OT with options for incorporating label distribution information through methods like Hellinger distance.


## Directory Structure
```bash
├── code/ # Main source code
│ ├── evaluation/ # Scripts for running FL experiments (run.py)
│ ├── ot/ # OT analysis scripts (run_ot_analysis.py)
│ ├── clients.py
│ ├── configs.py # Central configuration file
│ ├── data_loading.py
│ ├── data_partitioning.py
│ ├── data_processing.py
│ ├── data_sets.py
│ ├── helper.py # Core utilities, dataclasses, metrics
│ ├── losses.py
│ ├── models.py # Model architectures
│ ├── ot_calculators.py # OT cost calculation logic
│ ├── ot_configs.py # OT configuration definitions
│ ├── ot_data_manager.py # Loads activations/performance for OT
│ ├── ot_pipeline_runner.py # Runs the OT analysis pipeline
│ ├── ot_utils.py # OT mathematical helpers
│ ├── pipeline.py # FL experiment orchestration
│ ├── results_manager.py # Handles loading/saving FL results/models
│ ├── results_utils.py # Utility functions for result analysis
│ ├── servers.py
│ └── synthetic_data.py # Generates synthetic data and shifts
├── pipeline_tools/ # Orchestration and submission scripts
│ ├── orchestrate_all.py # Run orchestration for multiple datasets/clients
│ ├── orchestrate.py # Orchestrate FL experiments in sequence
│ ├── status.py # Check status of experiments
│ ├── submit_evaluation.sh # Slurm script for FL experiments
│ ├── submit_ot_analysis.sh # Slurm script for OT analysis
│ └── logs/ # Logs from orchestration runs
├── data/ # Raw and processed datasets (organized by dataset name)
│ ├── CIFAR/
│ ├── Credit/
│ └── ... # Other dataset directories
├── results/ # Stores FL experiment results when using 'score' metric
│ ├── lr_tuning/
│ ├── reg_param_tuning/
│ ├── evaluation/
│ └── ot_analysis/
├── results_loss/ # Stores FL experiment results when using 'loss' metric
│ ├── lr_tuning/
│ ├── reg_param_tuning/
│ ├── evaluation/
│ └── ot_analysis/
├── saved_models/ # Saved model checkpoints for 'score' metric
│ ├── CIFAR/
│ ├── Credit/
│ └── ...
├── saved_models_loss/ # Saved model checkpoints for 'loss' metric
│ ├── CIFAR/
│ ├── Credit/
│ └── ...
├── activations/ # Cached activation data for 'score' metric OT analysis
│ ├── CIFAR/
│ ├── Credit/
│ └── ...
├── activations_loss/ # Cached activation data for 'loss' metric OT analysis
│ ├── CIFAR/
│ ├── Credit/
│ └── ...
└── logs/ # Slurm logs
    ├── outputs_loss/
    └── errors_loss/
```

## Setup

1.  **Environment:** Ensure you have a Python environment (e.g., Conda) with necessary packages installed. Use the `requirements` file and install using `pip install -r requirements`. Key dependencies include PyTorch, NumPy, Pandas, Scipy, POT (Python Optimal Transport), scikit-learn, Matplotlib, Seaborn, tabulate, etc.
2.  **CUDA (Optional):** If using GPUs, ensure CUDA toolkit and compatible PyTorch versions are installed. The code will automatically use CUDA if available.
3.  **Data:** Place datasets in the `data/` directory, following the expected structure for each dataset loader (see `data_loading.py` and `configs.py`). Some datasets (e.g., CIFAR, EMNIST) may be downloaded automatically on first run if not found.

## Configuration (`configs.py`)

This is the central file for defining experiment parameters:

*   **`DEFAULT_PARAMS`:** Contains a dictionary where keys are dataset names. Each dataset has sub-dictionaries defining:
    *   FL parameters (learning rates, rounds, epochs, batch size, primary metric for model selection).
    *   Data handling (data source, partitioning strategy, sampling, shift types/parameters).
    *   Model and Dataset class names.
    *   Configuration for hyperparameter tuning runs (servers_tune_lr, servers_tune_reg).
    *   selection_criterion_key: Specifies which metric (e.g., val_scores, val_losses) is used to select the "best" model during training/tuning.
    *   activation_extractor_type: Defines which method to use for extracting model activations (e.g., hook_based, rep_vector).
*   **`DATASET_COSTS`:** Defines the list of heterogeneity parameters (cost) to iterate over for each dataset during experiments. The meaning of cost depends on the dataset configuration (e.g., Dirichlet alpha, shift intensity, site pair name).
*   **Paths:** Defines root directories. RESULTS_DIR, MODEL_SAVE_DIR, ACTIVATION_DIR are dynamically configured by configure_paths(metric) based on the chosen primary metric for an experiment run (e.g., "score" or "loss"), allowing for separate tracking of results.
*   **Global Settings:** Device preference (DEVICE), number of workers (N_WORKERS).
*   **Algorithms:**
    *   ALGORITHMS: Lists FL algorithms for which final evaluation runs are performed (e.g., ['local', 'fedavg']).
    *   REG_ALGORITHMS: Lists algorithms that typically require regularization parameter tuning (e.g., ['fedprox', 'pfedme', 'ditto']).

**Before running experiments, review and potentially adjust the settings in `configs.py` for your target datasets and experimental setup.** The choice of metric via command-line arguments to the submission scripts will influence which set of paths are used for results and models.

# Running Experiments

## 1. Using the Orchestration Pipeline (Recommended)

The pipeline tools provide a streamlined way to run complete experiment pipelines from learning rate tuning through to OT analysis.

### Orchestrate a Single Dataset

```bash
python pipeline_tools/orchestrate.py --dataset DATASET_NAME --num_clients NUM_CLIENTS --metric [score|loss]
```

This will run the following phases in sequence for one dataset:
1. Learning rate tuning
2. Regularization parameter tuning
3. Final evaluation
4. OT analysis

The script automatically tracks progress and only runs phases that haven't been completed.

### Orchestrate Multiple Datasets

```bash
python pipeline_tools/orchestrate_all.py --datasets DATASET1,DATASET2 --num-clients NUM_CLIENTS --metric [score|loss]
```

Options:
- `--datasets`: Comma-separated list of datasets or 'all'
- `--num-clients`: Number of clients to use (single value or comma-separated list)
- `--metric`: Evaluation metric ('score' or 'loss')
- `--force`: Force rerun of all phases
- `--force-phases`: Comma-separated list of phases to force (learning_rate, reg_param, evaluation, ot_analysis)
- `--dry-run`: Print commands without executing
- `--summary-only`: Only print status summary

### Check Experiment Status

```bash
python pipeline_tools/status.py --dataset DATASET_NAME --num_clients NUM_CLIENTS --metric [score|loss]
```

This shows a detailed table of experiment status for each phase, including progress, timestamps, and error counts.

## 2. Running Individual Components

### Federated Learning Experiments

Entry point: `code/evaluation/run.py`

```bash
python code/evaluation/run.py -ds <DATASET_NAME> -exp <EXPERIMENT_TYPE> [-nc <NUM_CLIENTS>] [-mc <METRIC>]
```

Arguments:
- `-ds`, `--dataset`: Required. The name of the dataset to use (must match a key in configs.DEFAULT_PARAMS).
- `-exp`, `--experiment_type`: Required. The type of experiment:
  - `learning_rate`: Performs learning rate tuning runs.
  - `reg_param`: Performs regularization parameter tuning runs (for algorithms like FedProx).
  - `evaluation`: Runs the final evaluation using the best hyperparameters found during tuning (or defaults).
- `-nc`, `--num_clients`: Optional. An integer to override the default number of clients.
- `-mc`, `--metric`: Optional. The primary metric ('score' or 'loss') to use for model selection and path configuration.

Examples:
```bash
python code/evaluation/run.py -ds CIFAR -exp evaluation
python code/evaluation/run.py -ds Synthetic_Feature -exp learning_rate -nc 10 -mc loss
```

### Optimal Transport Analysis

Entry point: `code/ot/run_ot_analysis.py`

```bash
python code/ot/run_ot_analysis.py -ds <DATASET_NAME> [-nc <NUM_FL_CLIENTS>] [-mt <MODEL_TYPE>] [-al <ACTIVATION_LOADER>] [-mc <METRIC>] [-far]
```

Arguments:
- `-ds`, `--dataset`: Required. The name of the dataset to analyze.
- `-nc`, `--num_fl_clients`: Optional. Number of FL clients in the run to analyze.
- `-mt`, `--model_type`: Optional. Type of model to load for activations: round0, best, or final. Defaults to round0.
- `-al`, `--activation_loader`: Optional. DataLoader type for activation extraction: train, val, or test. Defaults to val.
- `-mc`, `--metric`: Optional. The primary metric ('score' or 'loss') from the FL run. Defaults to 'score'.
- `-far`, `--force_activation_regen`: Optional. Force regeneration of activation cache.

Example:
```bash
python code/ot/run_ot_analysis.py -ds CIFAR -nc 5 -mt round0 -al val -mc score
```

OT analysis now uses a **single direct OT method** applied to model activations. Programmatic usage:

```python
from ot_pipeline_runner import OTPipelineRunner 

runner = OTPipelineRunner(num_target_fl_clients=5) 
results_df = runner.run_pipeline(
    dataset_name='CIFAR',
    model_type_to_analyze='round0',
    activation_loader_type='val',
    performance_metric_key='score',
    force_activation_regen=False
)
print(results_df.head())
```

### Batch Submission (Slurm)

The `pipeline_tools` directory includes Slurm submission scripts:

**Submit FL Evaluation Jobs:**
```bash
bash pipeline_tools/submit_evaluation.sh --datasets=CIFAR,EMNIST --exp-types=evaluation --num-clients=5 --metric=score
```

**Submit OT Analysis Jobs:**
```bash
bash pipeline_tools/submit_ot_analysis.sh --datasets=CIFAR,EMNIST --fl-num-clients=5 --model-types=round0 --activation-loaders=val --metric=score
```

## OT Analysis Details

The current implementation focuses on Direct OT with several configuration options:

- **Direct OT with Label Distribution Distance:** Combines feature space OT with label distribution distance metrics (Hellinger, Wasserstein Gaussian)
- **Within-Class OT:** Analyzes feature similarity separately for each class, avoiding cross-class comparisons
- **Feature-Only OT:** Standard OT on feature representations with different distance metrics (cosine, euclidean)

OT configurations are defined in `ot_configs.py` and include parameters like:
- Feature/label weights
- Distance metrics
- Normalization options
- Regularization parameters

## Data Requirements

Datasets should be placed in the `data/` directory, organized into subdirectories named after the dataset key used in `configs.py` (e.g., `data/CIFAR/`, `data/Credit/`).

The specific file structure required within each dataset directory depends on the corresponding loader function in `data_loading.py`. Check the loader implementation for details (e.g., load_credit_raw expects creditcard.csv inside data/Credit/).

Torchvision datasets (CIFAR, EMNIST) will attempt to download automatically if not found in the specified data_dir.


## Citation
If you use this code or the concepts presented in your research, please cite our paper
