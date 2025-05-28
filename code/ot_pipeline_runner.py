"""
OT Pipeline Runner - Orchestrates OT analysis over FL runs, costs, client pairs, and OT configurations.
"""
import os
import pandas as pd
import numpy as np
import logging
import traceback
from dataclasses import replace
from itertools import combinations
from typing import List, Dict, Any, Optional, Union

# Import OT-related components
from directories import paths
dir_paths = paths()
ROOT_DIR = dir_paths.root_dir
from configs import DATASET_COSTS, DEFAULT_PARAMS
from ot_configs import OTConfig, all_configs
from ot_data_manager import OTDataManager
from ot_calculators import OTCalculatorFactory
from results_manager import OTAnalysisRecord, ResultsManager

# Configure module logger
logger = logging.getLogger(__name__)

class OTPipelineRunner:
    """Orchestrates OT similarity analysis pipeline across runs, costs, and client pairs."""

    def __init__(self, num_target_fl_clients: int, activation_dir: Optional[str] = None):
        """
        Initializes the OT pipeline runner.

        Args:
            num_target_fl_clients: Number of FL clients in the run to analyze
            activation_dir: Path to activation cache (optional, defaults to config path)
        """
        self.num_target_fl_clients = num_target_fl_clients
        
        # Initialize DataManager with target FL client count
        self.data_manager = OTDataManager(
            num_target_fl_clients=num_target_fl_clients,
            activation_dir=activation_dir
        )
        logger.info(f"OT Pipeline runner initialized for {num_target_fl_clients} FL clients")

    def run_pipeline(
            self,
            dataset_name: str,
            model_type_to_analyze: str = 'round0',
            activation_loader_type: str = 'val',
            performance_metric_key: str = 'score',
            performance_aggregation: str = 'mean',
            force_activation_regen: bool = False,
            ot_configurations: Optional[List[OTConfig]] = None
        ) -> pd.DataFrame:
            """
            Runs the full OT analysis pipeline.
            Only processes missing OT configurations to avoid recomputing existing results.

            Args:
                dataset_name: Name of the dataset to analyze
                model_type_to_analyze: Type of FL model to extract activations from ('round0', 'best', 'final')
                activation_loader_type: Dataloader type ('train', 'val', 'test')
                performance_metric_key: Key for evaluating performance ('score', 'loss')
                performance_aggregation: How to aggregate performance metrics ('mean', 'median')
                force_activation_regen: Whether to force regeneration of activations
                ot_configurations: Custom list of OT configurations to use (defaults to all configs)

            Returns:
                DataFrame with OT analysis results
            """
            # Import ExperimentType for proper enum usage
            from helper import ExperimentType

            # Initialize results manager and parameters
            ot_results_saver = ResultsManager(ROOT_DIR, dataset_name, self.num_target_fl_clients)
            fl_costs_list = DATASET_COSTS.get(dataset_name, [])
            if not fl_costs_list:
                logger.warning(f"No costs defined in DATASET_COSTS for dataset '{dataset_name}'. Aborting.")
                return pd.DataFrame()
                
            dataset_fl_params = DEFAULT_PARAMS.get(dataset_name, {})
            num_fl_runs = dataset_fl_params.get('runs', 1)
            base_fl_seed = dataset_fl_params.get('base_seed', 42)
            num_classes = dataset_fl_params.get('fixed_classes')  # Can be None for segmentation
            ot_configs_to_run = ot_configurations or all_configs

            # Load existing results to determine what's already completed
            existing_records, existing_metadata = ot_results_saver.load_results(ExperimentType.OT_ANALYSIS)
            completed_work_units = set()
            
            # Build set of completed work units from existing results
            for record in existing_records:
                if isinstance(record, dict) and record.get('status') == 'Success':
                    cost = record.get('fl_cost_param')
                    run_idx = record.get('fl_run_idx')
                    client_pair = record.get('client_pair')
                    method_name = record.get('ot_method_name')
                    
                    if all(x is not None for x in [cost, run_idx, client_pair, method_name]):
                        completed_work_units.add((cost, run_idx, client_pair, method_name))
            
            logger.info(f"Found {len(completed_work_units)} completed work units from existing results")

            # Storage for new results only
            new_ot_analysis_records = []

            # Process all FL runs
            logger.info(f"Starting OT Analysis for dataset '{dataset_name}', {self.num_target_fl_clients} clients")
            
            # Iterate through FL runs
            for fl_run_idx in range(num_fl_runs):
                current_fl_seed = base_fl_seed + fl_run_idx
                logger.info(f"Processing FL run {fl_run_idx + 1}/{num_fl_runs} (seed: {current_fl_seed})")
                
                # Iterate through FL heterogeneity costs
                for fl_cost_param in fl_costs_list:
                    logger.info(f"  Processing FL cost parameter: {fl_cost_param}")
                    
                    # Get FL performance (Local vs FedAvg) - only compute if needed
                    local_perf, fedavg_perf, perf_delta = None, None, None
                    performance_computed = False
                    
                    # Generate all possible client pairs for analysis
                    client_ids = [f'client_{i+1}' for i in range(self.num_target_fl_clients)]
                    client_pairs = list(combinations(client_ids, 2))
                    
                    # Process each client pair
                    for cid1, cid2 in client_pairs:
                        pair_id = f"{cid1}_vs_{cid2}"
                        
                        # Process each OT configuration
                        for ot_config in ot_configs_to_run:
                            # Check if this work unit is already completed
                            work_unit_key = (fl_cost_param, fl_run_idx, pair_id, ot_config.name)
                            if work_unit_key in completed_work_units:
                                continue  # Skip already completed work
                            
                            # Compute performance metrics only when needed
                            if not performance_computed:
                                logger.info(f"  Computing performance metrics for cost: {fl_cost_param}")
                                local_perf, fedavg_perf = self.data_manager.get_performance(
                                    dataset_name, 
                                    fl_cost_param, 
                                    current_fl_seed,
                                    performance_aggregation, 
                                    performance_metric_key
                                )
                                
                                # Calculate performance delta (direction depends on metric type)
                                if performance_metric_key == 'loss':
                                    # For loss, lower is better: Local - FedAvg (positive means FedAvg is better)
                                    perf_delta = local_perf - fedavg_perf if np.isfinite(local_perf) and np.isfinite(fedavg_perf) else np.nan
                                else:
                                    # For score, higher is better: FedAvg - Local (positive means FedAvg is better)
                                    perf_delta = fedavg_perf - local_perf if np.isfinite(local_perf) and np.isfinite(fedavg_perf) else np.nan

                                if not np.isfinite(perf_delta):
                                    logger.warning(f"  Skipping FL cost {fl_cost_param} for run {fl_run_idx + 1}: invalid performance delta")
                                    break  # Skip this entire cost
                                
                                performance_computed = True
                            
                            logger.info(f"    Processing NEW: {pair_id} -> {ot_config.name}")
                            
                            # Create a base record with common information
                            record = OTAnalysisRecord(
                                dataset_name=dataset_name,
                                fl_cost_param=fl_cost_param,
                                fl_run_idx=fl_run_idx,
                                client_pair=pair_id,
                                ot_method_name=ot_config.name,
                                ot_cost_value=None,  # Will be set below if successful
                                fl_local_metric=local_perf,
                                fl_fedavg_metric=fedavg_perf,
                                fl_performance_delta=perf_delta,
                                status="Pending",
                                error_message=None,
                                ot_method_specific_results={}
                            )
                            
                            # Get activations for this client pair
                            processed_activations = self.data_manager.get_activations(
                                dataset_name=dataset_name,
                                fl_cost=fl_cost_param,
                                fl_seed=current_fl_seed,
                                client_id_1=cid1,
                                client_id_2=cid2,
                                num_classes=num_classes,
                                loader_type=activation_loader_type,
                                force_regenerate=force_activation_regen,
                                model_type=model_type_to_analyze,
                                use_loss_weighting_hint=ot_config.params.get('use_loss_weighting', False)
                            )
                            
                            # Handle activation failure
                            if processed_activations is None or cid1 not in processed_activations or cid2 not in processed_activations:
                                reason = "No activations generated" if processed_activations is None else f"Missing data for client(s): {cid1 if cid1 not in processed_activations else ''} {cid2 if cid2 not in processed_activations else ''}"
                                updated_record = replace(record, 
                                    status="ActivationError", 
                                    error_message=f"Failed to get activations: {reason}"
                                )
                                new_ot_analysis_records.append(updated_record)
                                continue
                            
                            # Extract client data
                            data1 = processed_activations[cid1]
                            data2 = processed_activations[cid2]
                            
                            # Create calculator using the factory
                            calculator = OTCalculatorFactory.create_calculator(
                                config=ot_config,
                                client_id_1=cid1,
                                client_id_2=cid2,
                                num_classes=num_classes
                            )
                            
                            if calculator is None:
                                updated_record = replace(record, 
                                    status="CalculatorCreationError", 
                                    error_message=f"Failed to create calculator for method: {ot_config.method_type}"
                                )
                                new_ot_analysis_records.append(updated_record)
                                continue
                                
                            # Calculate OT cost
                            try:
                                calculator.calculate_similarity(data1, data2, ot_config.params)
                                ot_calc_results_dict = calculator.get_results()
                                
                                # Extract primary OT cost based on method type
                                primary_ot_cost = None
                                if ot_config.method_type == 'direct_ot':
                                    primary_ot_cost = ot_calc_results_dict.get('direct_ot_cost')
                                else:
                                    # Try to find any "cost" in the results
                                    for k in ot_calc_results_dict:
                                        if "cost" in k.lower() and isinstance(ot_calc_results_dict[k], (int, float)):
                                            primary_ot_cost = ot_calc_results_dict[k]
                                            break
                                
                                # Create successful record
                                updated_record = replace(record, 
                                    status="Success", 
                                    ot_cost_value=primary_ot_cost, 
                                    ot_method_specific_results=ot_calc_results_dict
                                )
                                
                            except Exception as e:
                                # Handle OT calculation error
                                error_details = f"{str(e)}\n{traceback.format_exc()}"
                                logger.error(f"OT calculation error for {ot_config.name}: {error_details}")
                                updated_record = replace(record, 
                                    status="OTError", 
                                    error_message=f"Error during OT calculation: {str(e)}"
                                )
                            
                            # Add the new record to results
                            new_ot_analysis_records.append(updated_record)
            
            # Combine existing and new results for saving
            if new_ot_analysis_records or existing_records:
                logger.info(f"Merging {len(existing_records)} existing + {len(new_ot_analysis_records)} new records")
                
                # Convert new records to dicts
                new_record_dicts = [record.to_dict() for record in new_ot_analysis_records]
                
                # Combine existing and new records
                all_records = existing_records + new_record_dicts
                
                # Prepare updated metadata
                ot_metadata = {
                    'dataset_analyzed': dataset_name,
                    'fl_clients_in_run': self.num_target_fl_clients,
                    'model_type_activations': model_type_to_analyze,
                    'activation_loader_type': activation_loader_type,
                    'performance_metric_key': performance_metric_key,
                    'ot_methods_used': [config.name for config in ot_configs_to_run],
                    'last_fl_run_idx_processed': num_fl_runs - 1,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'incremental_update': len(new_ot_analysis_records) > 0,
                    'new_records_added': len(new_ot_analysis_records)
                }
                
                # Preserve important metadata from existing results
                if existing_metadata:
                    ot_metadata['original_timestamp'] = existing_metadata.get('timestamp', 'unknown')
                
                # Save combined results
                ot_results_saver.save_results(
                    all_records, 
                    ExperimentType.OT_ANALYSIS,
                    ot_metadata
                )
                
                logger.info(f"OT analysis complete: {len(new_ot_analysis_records)} new records added to {len(existing_records)} existing records")
            else:
                logger.info("No new OT analysis records to process")
            
            # Return DataFrame with all results (existing + new)
            all_records_for_return = existing_records + [record.to_dict() for record in new_ot_analysis_records]
            return pd.DataFrame(all_records_for_return)