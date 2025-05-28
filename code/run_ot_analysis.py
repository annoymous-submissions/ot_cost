#!/usr/bin/env python
"""
Entry point script for running OT analysis on Federated Learning models.

Parses command-line arguments to select the dataset, number of FL clients,
model type, and activation loader type, then runs the OT pipeline.
"""
import argparse
import os
import sys
import logging
import traceback
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from directories import configure, paths
# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description='Run Optimal Transport (OT) analysis on FL models.'
)
parser.add_argument(
    "-ds", "--dataset",
    required=True,
    help="Select the dataset for analysis."
)
parser.add_argument(
    "-nc", "--num_fl_clients",
    type=int,
    default=None,
    help="Number of FL clients in the run to analyze. If not provided, uses the default from configs."
)
parser.add_argument(
    "-mt", "--model_type",
    default="round0",
    choices=["round0", "best", "final"],
    help="Type of model to analyze: 'round0', 'best', or 'final'."
)
parser.add_argument(
    "-al", "--activation_loader",
    default="val",
    choices=["train", "val", "test"],
    help="DataLoader type to use for activation extraction: 'train', 'val', or 'test'."
)
parser.add_argument(
    "-mc", "--metric",
    default="score",
    choices=["score", "loss"],
    help="Metric to use for performance comparison: 'score' or 'loss'."
)
parser.add_argument(
    "-far", "--force_activation_regen",
    action="store_true",
    help="Force regeneration of activation cache."
)

args = parser.parse_args()
# --- Directory Setup ---
configure(args.metric)
dir_paths = paths()
ROOT_DIR = dir_paths.root_dir
RESULTS_DIR = dir_paths.results_dir
ACTIVATION_DIR = dir_paths.activation_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, 'logs', 'ot_analysis.log'))
    ]
)
logger = logging.getLogger(__name__)

from configs import DEFAULT_PARAMS
from ot_pipeline_runner import OTPipelineRunner

def main():
    """
    Parses command-line arguments and runs the OT analysis pipeline.
    """
    # --- Resolve num_fl_clients ---
    num_fl_clients = args.num_fl_clients
    if num_fl_clients is None:
        num_fl_clients = DEFAULT_PARAMS[args.dataset].get('default_num_clients', 5)

    
    # --- Run OT Pipeline ---
    try:
        logger.info(f"Starting OT analysis for dataset '{args.dataset}' with {num_fl_clients} FL clients")
        logger.info(f"Model type: {args.model_type}, Activation loader: {args.activation_loader}")
        logger.info(f"Performance metric: {args.metric}")
        try:
            os.makedirs(ACTIVATION_DIR, exist_ok=True)
            print(f"Results will be saved in subdirectories under: {ACTIVATION_DIR}")
        except OSError as e:
            print(f"Error creating results directories: {e}", file=sys.stderr)
            sys.exit(1)
            
        runner = OTPipelineRunner(
            num_target_fl_clients=num_fl_clients, 
            activation_dir=ACTIVATION_DIR
        )
        
        results_df = runner.run_pipeline(
            dataset_name=args.dataset,
            model_type_to_analyze=args.model_type,
            activation_loader_type=args.activation_loader,
            performance_metric_key=args.metric,
            force_activation_regen=args.force_activation_regen,
        )
        
        logger.info(f"OT analysis complete for dataset '{args.dataset}'. {len(results_df)} records processed.")
        logger.info(f"Results saved successfully.")
        
    except ValueError as ve:
        logger.error(f"Configuration or Value Error: {ve}")
        traceback.print_exc()
        sys.exit(1)
    except FileNotFoundError as fnf:
        logger.error(f"File Not Found Error: {fnf}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during OT analysis: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()