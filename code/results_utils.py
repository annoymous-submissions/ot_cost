"""
Utility functions for analyzing and visualizing both FL and OT results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from results_manager import ResultsManager, TrialRecord,  OTAnalysisRecord
    from helper import MetricKey, ExperimentType, infer_higher_is_better
except ImportError:
    pass


def get_averaged_ot_costs(
    ot_df: pd.DataFrame,
    target_ot_method_name: Optional[str] = 'WC_Direct_Hellinger_4:1',
    target_client_pair: Optional[str] = 'client_1_vs_client_2'
) -> pd.DataFrame:
    """
    Extract and aggregate OT costs per (fl_cost_param, fl_run_idx) pair.
    
    Args:
        ot_df: DataFrame with OT analysis results
        target_ot_method_name: OT method to filter by
        target_client_pair: Client pair to filter by
        
    Returns:
        DataFrame with columns ['fl_cost_param', 'fl_run_idx', 'avg_ot_cost', 'ot_cost_std', 'num_points']
    """
    if ot_df.empty:
        return pd.DataFrame()
    
    # Filter by target OT method and client pair
    filtered_df = ot_df.copy()
    if target_ot_method_name:
        filtered_df = filtered_df[filtered_df['ot_method_name'] == target_ot_method_name]
    if target_client_pair:
        filtered_df = filtered_df[filtered_df['client_pair'] == target_client_pair]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Group by cost parameter and run index
    grouped = filtered_df.groupby(['fl_cost_param'])
    
    # Aggregate OT costs
    aggregated = grouped.agg(
        avg_ot_cost=('ot_cost_value', 'mean'),
        ot_cost_std=('ot_cost_value', 'std'),
        num_points=('ot_cost_value', 'count')
    ).reset_index()
    
    return aggregated
def smooth_curve(points, window_size=5, poly_order=1):
    """
    Applies Savitzky-Golay filter for smoothing a time series.
    
    Args:
        points (np.ndarray): Array of data points
        window_size (int): Window size for the filter
        poly_order (int): Polynomial order for the filter
        
    Returns:
        np.ndarray: Smoothed data points
    """
    import numpy as np
    from scipy.signal import savgol_filter
    
    # Convert to numpy array if needed
    points = np.array(points)
    
    # Handle case with too few points
    if len(points) < window_size:
        return points
    
    # Handle NaN values through interpolation
    nan_indices = np.isnan(points)
    if np.any(nan_indices):
        points_no_nan = points.copy()
        non_nan_indices = ~nan_indices
        
        # Use interpolation to fill NaN values
        points_no_nan[nan_indices] = np.interp(
            np.flatnonzero(nan_indices), 
            np.flatnonzero(non_nan_indices), 
            points_no_nan[non_nan_indices]
        )
        points = points_no_nan
    
    # Ensure window_size is odd
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    # Ensure poly_order < window_size
    poly_order = min(poly_order, window_size - 1)
    
    # Apply Savitzky-Golay filter
    smoothed = savgol_filter(points, window_size, poly_order)
    return smoothed

# =============================================================================
# == FL Results Aggregation Functions ==
# =============================================================================

def aggregate_train_val(records: List[TrialRecord], min_rounds: int, conf: float = 0.95) -> Optional[Dict[str, np.ndarray]]:
    """
    Aggregate train/val curves for records truncated to min_rounds using bootstrap CI.
    
    Args:
        records: List of TrialRecords from FL experiments
        min_rounds: Minimum number of rounds to include
        conf: Confidence level for intervals
        
    Returns:
        Dictionary with aggregated train/val statistics or None if insufficient data
    """
    if not records:
        return None

    trains, vals = [], []
    for r in records:
        tr = r.metrics.get(MetricKey.TRAIN_LOSSES, [])
        va = r.metrics.get(MetricKey.VAL_LOSSES, [])
        if len(tr) >= min_rounds and len(va) >= min_rounds:
            trains.append(tr[:min_rounds])
            vals.append(va[:min_rounds])

    if not trains:
        return None

    trains = np.array(trains)
    vals = np.array(vals)

    mean_tr = trains.mean(0)
    lower_tr, upper_tr = [], []
    mean_va = vals.mean(0)
    lower_va, upper_va = [], []

    for arr, lower, upper in [(trains, lower_tr, upper_tr), (vals, lower_va, upper_va)]:
        for t in range(min_rounds):
            m, lo, hi = mean_ci_bootstrap(arr[:, t], conf)
            lower.append(lo)
            upper.append(hi)

    return {
        "num_runs": trains.shape[0],
        "mean_train": mean_tr, 
        "lower_train": np.array(lower_tr), 
        "upper_train": np.array(upper_tr),
        "mean_val": mean_va, 
        "lower_val": np.array(lower_va), 
        "upper_val": np.array(upper_va),
    }

def aggregate_test(records: List[TrialRecord], conf: float = 0.95) -> Optional[Dict[str, float]]:
    """
    Aggregate final-test losses & scores for FL records.
    
    Args:
        records: List of TrialRecords from FL experiments
        conf: Confidence level for intervals
        
    Returns:
        Dictionary with aggregated test statistics or None if insufficient data
    """
    if not records:
        return None
        
    losses = np.array([r.metrics.get(MetricKey.TEST_LOSSES, [np.nan])[-1] for r in records])
    scores = np.array([r.metrics.get(MetricKey.TEST_SCORES, [np.nan])[-1] for r in records])

    if np.isnan(losses).all() or np.isnan(scores).all():
        return None

    m_loss, lo_loss, hi_loss = mean_ci_bootstrap(losses, conf)
    m_score, lo_score, hi_score = mean_ci_bootstrap(scores, conf)
    
    return {
        "num_runs": len(records),
        "mean_loss": m_loss, "lower_loss": lo_loss, "upper_loss": hi_loss,
        "mean_score": m_score, "lower_score": lo_score, "upper_score": hi_score,
    }

def plot_losses_per_cost(
    results_manager: ResultsManager,
    target_costs: List[Any],
    algorithms: List[str] = None,
    confidence_level: float = 0.95,
    start_round: int = 1,
    plot_losses: bool = False,
    color_map: Dict[str, str] = None,
    auto_y_limits: bool = True,  # New parameter to control y-axis limits
    y_limit_padding: float = 0.1,  # Padding percentage for y-axis limits
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a composite figure of curves + bar plots for target_costs with multiple algorithms.
    
    Args:
        results_manager: ResultsManager instance
        target_costs: List of cost values to plot
        algorithms: List of algorithm/server_type names to include (default: ["local", "fedavg"])
        confidence_level: Confidence level for intervals (default: 0.95)
        start_round: Starting round for training curves (default: 1)
        plot_losses: Whether to include loss curves (default: False)
        color_map: Dictionary mapping algorithm names to colors (default: None, will use automatic colors)
        auto_y_limits: Whether to automatically adjust y-axis limits (default: True)
        y_limit_padding: Padding percentage for y-axis limits (default: 0.1)
        
    Returns:
        Tuple of (Figure, Axes)
    """
    # Set default algorithms list if not provided
    if algorithms is None:
        algorithms = ["local", "fedavg"]
    
    # Set up default color map if not provided
    if color_map is None:
        # Default colors for common algorithms
        color_map = {
            "local": 'tab:blue',
            "fedavg": 'green',
            "fedprox": 'red',
            "pfedme": 'purple',
            "ditto": 'orange',
        }
        # For any algorithm not in the default map, we'll assign colors later
    
    # Load & pre-filter records
    all_records, _ = results_manager.load_results(ExperimentType.EVALUATION)
    all_records = [r for r in all_records if r.error is None]

    # Organize by (cost, server) for convenience
    records_by_cs = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        records_by_cs[rec.cost][rec.server_type].append(rec)

    # Aggregate metrics per cost & algorithm
    agg_curves = {}
    agg_test = {}
    min_rounds_per_cost = {}

    for cost in target_costs:
        # Check if we have data for all algorithms at this cost
        has_all_algorithms = True
        available_algorithms = []
        
        for algo in algorithms:
            if not records_by_cs[cost][algo]:
                has_all_algorithms = False
            else:
                available_algorithms.append(algo)
        
        if not available_algorithms:
            continue  # Skip this cost if no data for any algorithm
            
        # Find minimum number of rounds across all algorithms for this cost
        min_r = float('inf')
        for algo in available_algorithms:
            for r in records_by_cs[cost][algo]:
                train_len = len(r.metrics.get(MetricKey.TRAIN_LOSSES, []))
                min_r = min(min_r, train_len)
        
        if min_r == float('inf') or min_r < start_round:
            continue
            
        min_rounds_per_cost[cost] = min_r
        
        # Aggregate metrics for each algorithm
        agg_curves[cost] = {}
        agg_test[cost] = {}
        
        for algo in available_algorithms:
            agg_curves[cost][algo] = aggregate_train_val(records_by_cs[cost][algo], min_r, confidence_level)
            agg_test[cost][algo] = aggregate_test(records_by_cs[cost][algo], confidence_level)

    # Filter costs for which we have valid data
    costs = [c for c in target_costs if c in agg_curves]
    if not costs:
        print("No valid data found – nothing to plot.")
        return None, None

    # Figure layout 
    n_rows = 1
    if plot_losses:
        n_rows += len(costs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.5 * n_rows), sharey=False)

    # Ensure axes is always a 2D array
    if n_rows == 1 and not plot_losses:
        axes = np.array([axes]).reshape(1, 2)

    # Final-test metrics – row 0
    x = np.arange(len(costs))
    width = 0.8 / len(algorithms)  # Adjust bar width based on number of algorithms
    
    # Assign colors for any algorithm not in the default map
    all_algos_with_data = set()
    for cost in costs:
        all_algos_with_data.update(agg_test[cost].keys())
    
    unassigned_algos = [algo for algo in all_algos_with_data if algo not in color_map]
    if unassigned_algos:
        # Generate distinct colors for unassigned algorithms
        additional_colors = plt.cm.tab10(np.linspace(0, 1, len(unassigned_algos)))
        for i, algo in enumerate(unassigned_algos):
            color_map[algo] = additional_colors[i]

    # Bar plot helper function
    def _bar(ax, vals, yerr, label, offset=0, color=None):
        bars = ax.bar(x + offset, vals, width, yerr=yerr, label=label, capsize=4, alpha=0.8, color=color)
        return bars[0].get_facecolor() if not color else color  # Return actual color used

    # Dictionaries to store min/max values for y-axis limits
    loss_min_vals = []
    loss_max_vals = []
    score_min_vals = []
    score_max_vals = []
    
    # Plot loss bars
    ax_loss = axes[0, 0]
    for i, algo in enumerate(algorithms):
        # Calculate offset for this algorithm's bars
        offset = (i - len(algorithms)/2 + 0.5) * width
        
        # Create arrays for values and error bars
        losses = []
        loss_errs = []
        
        for c in costs:
            if algo in agg_test[c]:
                mean_loss = agg_test[c][algo]["mean_loss"]
                lower_loss = agg_test[c][algo]["lower_loss"]
                upper_loss = agg_test[c][algo]["upper_loss"]
                
                losses.append(mean_loss)
                loss_errs.append([
                    mean_loss - lower_loss,
                    upper_loss - mean_loss
                ])
                
                # Update min/max values for y-axis limits
                loss_min_vals.append(lower_loss)
                loss_max_vals.append(upper_loss)
            else:
                # Add NaN for missing data
                losses.append(np.nan)
                loss_errs.append([0, 0])
        
        # Convert to numpy arrays for transpose
        losses = np.array(losses)
        loss_errs = np.array(loss_errs)
        
        # Skip if all values are NaN
        if np.isnan(losses).all():
            continue
            
        # Plot bars with algorithm color
        _bar(ax_loss, losses, np.transpose(loss_errs), algo, offset, color=color_map.get(algo))
    
    ax_loss.set_title("Final Test Loss")
    ax_loss.set_xticks(x, [str(c) for c in costs], rotation=30)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, ls="--", alpha=0.3)
    ax_loss.legend()
    
    # Adjust y-axis limits for loss plot
    if auto_y_limits and loss_min_vals and loss_max_vals:
        y_min = min(loss_min_vals)
        y_max = max(loss_max_vals)
        y_range = y_max - y_min
        
        # Add padding to the limits
        y_min = y_min - y_range * y_limit_padding
        y_max = y_max + y_range * y_limit_padding
        
        # Set the limits (ensure we don't set negative lower limit for loss if values are very close to 0)
        ax_loss.set_ylim(max(0, y_min), y_max)

    # Plot score bars
    ax_score = axes[0, 1]
    for i, algo in enumerate(algorithms):
        # Calculate offset for this algorithm's bars
        offset = (i - len(algorithms)/2 + 0.5) * width
        
        # Create arrays for values and error bars
        scores = []
        score_errs = []
        
        for c in costs:
            if algo in agg_test[c]:
                mean_score = agg_test[c][algo]["mean_score"]
                lower_score = agg_test[c][algo]["lower_score"]
                upper_score = agg_test[c][algo]["upper_score"]
                
                scores.append(mean_score)
                score_errs.append([
                    mean_score - lower_score,
                    upper_score - mean_score
                ])
                
                # Update min/max values for y-axis limits
                score_min_vals.append(lower_score)
                score_max_vals.append(upper_score)
            else:
                # Add NaN for missing data
                scores.append(np.nan)
                score_errs.append([0, 0])
        
        # Convert to numpy arrays for transpose
        scores = np.array(scores)
        score_errs = np.array(score_errs)
        
        # Skip if all values are NaN
        if np.isnan(scores).all():
            continue
            
        # Plot bars with algorithm color
        _bar(ax_score, scores, np.transpose(score_errs), algo, offset, color=color_map.get(algo))
    
    ax_score.set_title("Final Test Score")
    ax_score.set_xticks(x, [str(c) for c in costs], rotation=30)
    ax_score.set_ylabel("Score")
    ax_score.grid(True, ls="--", alpha=0.3)
    ax_score.legend()
    
    # Adjust y-axis limits for score plot
    if auto_y_limits and score_min_vals and score_max_vals:
        y_min = min(score_min_vals)
        y_max = max(score_max_vals)
        y_range = y_max - y_min
        
        # Add padding to the limits
        y_min = y_min - y_range * y_limit_padding
        y_max = y_max + y_range * y_limit_padding
        
        # Set the limits (don't set negative lower limit for scores if values are close to 0)
        ax_score.set_ylim(max(0, y_min), y_max)

    # Plot training and validation loss curves if requested
    if plot_losses:
        for row_idx, cost in enumerate(costs, start=1):
            min_r = min_rounds_per_cost[cost]
            rounds = np.arange(start_round, min_r)
            
            # Training loss curves (left column)
            ax_train = axes[row_idx, 0]
            
            # Variables to track min/max values for y-axis limits
            train_min_vals = []
            train_max_vals = []
            
            # Plot training curves for each algorithm
            for algo in algorithms:
                if algo not in agg_curves[cost]:
                    continue
                    
                algo_data = agg_curves[cost][algo]
                color = color_map.get(algo)
                
                # Get relevant slices
                mean_vals = algo_data["mean_train"][start_round:min_r]
                lower_vals = algo_data["lower_train"][start_round:min_r]
                upper_vals = algo_data["upper_train"][start_round:min_r]
                
                # Plot training curve
                ax_train.plot(rounds, mean_vals, color=color, linestyle='-', label=algo)
                             
                # Plot confidence intervals
                ax_train.fill_between(rounds, lower_vals, upper_vals, color=color, alpha=0.2)
                
                # Track min/max values for y-axis limits
                train_min_vals.extend(lower_vals)
                train_max_vals.extend(upper_vals)
            
            ax_train.set_title(f"Training Loss for Cost = {cost}")
            ax_train.set_xlabel("Round")
            ax_train.set_ylabel("Training Loss")
            ax_train.grid(True, ls="--", alpha=0.3)
            ax_train.legend()
            
            # Adjust y-axis limits for training plot
            if auto_y_limits and train_min_vals and train_max_vals:
                y_min = min(train_min_vals)
                y_max = max(train_max_vals)
                y_range = y_max - y_min
                
                # Add padding to the limits
                y_min = y_min - y_range * y_limit_padding
                y_max = y_max + y_range * y_limit_padding
                
                # Set the limits (ensure we don't set negative lower limit if values are close to 0)
                ax_train.set_ylim(max(0, y_min), y_max)
            
            # Validation loss curves (right column)
            ax_val = axes[row_idx, 1]
            
            # Variables to track min/max values for y-axis limits
            val_min_vals = []
            val_max_vals = []
            
            # Plot validation curves for each algorithm
            for algo in algorithms:
                if algo not in agg_curves[cost]:
                    continue
                    
                algo_data = agg_curves[cost][algo]
                color = color_map.get(algo)
                
                # Get relevant slices
                mean_vals = algo_data["mean_val"][start_round:min_r]
                lower_vals = algo_data["lower_val"][start_round:min_r]
                upper_vals = algo_data["upper_val"][start_round:min_r]
                
                # Plot validation curve
                ax_val.plot(rounds, mean_vals, color=color, linestyle='--', label=algo)
                           
                # Plot confidence intervals
                ax_val.fill_between(rounds, lower_vals, upper_vals, color=color, alpha=0.2)
                
                # Track min/max values for y-axis limits
                val_min_vals.extend(lower_vals)
                val_max_vals.extend(upper_vals)
            
            ax_val.set_title(f"Validation Loss for Cost = {cost}")
            ax_val.set_xlabel("Round")
            ax_val.set_ylabel("Validation Loss")
            ax_val.grid(True, ls="--", alpha=0.3)
            ax_val.legend()
            
            # Adjust y-axis limits for validation plot
            if auto_y_limits and val_min_vals and val_max_vals:
                y_min = min(val_min_vals)
                y_max = max(val_max_vals)
                y_range = y_max - y_min
                
                # Add padding to the limits
                y_min = y_min - y_range * y_limit_padding
                y_max = y_max + y_range * y_limit_padding
                
                # Set the limits (ensure we don't set negative lower limit if values are close to 0)
                ax_val.set_ylim(max(0, y_min), y_max)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    return agg_test

def mean_ci_bootstrap(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Return mean and bootstrapped CI at the specified confidence level.
    Args:
        values: Array of values to compute CI for
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if isinstance(values, list) and len(values) == 0:
        return np.nan, np.nan, np.nan
    
    if isinstance(values, np.ndarray) and (values.size == 0 or np.isnan(values).all()):
        return np.nan, np.nan, np.nan
        
    if isinstance(values, np.ndarray) and values.ndim == 0:
        values = values[None]
    
    n = len(values)
    mean = np.mean(values)
    if n == 1:
        return mean, mean, mean
    np.random.seed(seed)
    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(values[bootstrap_indices], axis=1)
    alpha = (1 - confidence) * 2
    lower_ci = np.percentile(bootstrap_means, 100 * alpha)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha))
    
    return mean, lower_ci, upper_ci


# =============================================================================
# == OT Analysis Functions ==
# =============================================================================

def load_ot_results(results_mgr: ResultsManager, filter_status: str = "Success") -> Tuple[pd.DataFrame, Dict]:
    """
    Load OT analysis results and convert to a DataFrame.
    
    Args:
        results_mgr: ResultsManager instance
        filter_status: Status to filter by (default: "Success")
        
    Returns:
        Tuple of (DataFrame of records, metadata dictionary)
    """
    # Load results
    loaded_dicts, ot_metadata = results_mgr.load_results(ExperimentType.OT_ANALYSIS)
    
    if not loaded_dicts:
        return pd.DataFrame(), {}
    
    # Convert to DataFrame
    df = pd.DataFrame(loaded_dicts)
    
    # Filter by status if requested
    if filter_status and 'status' in df.columns:
        df = df[df['status'] == filter_status]
    df.drop(columns=['ot_method_specific_results'], inplace=True, errors='ignore')
    df = df.drop_duplicates()
    return df, ot_metadata

def aggregate_ot_data(df: pd.DataFrame, grouping_keys: List[str] = None, 
                     include_percentages: bool = True, 
                     metric_type: str = 'score') -> pd.DataFrame:
    """
    Aggregate OT data by grouping keys with confidence intervals.
    Now includes percentage change relative to local performance.
    
    Args:
        df: DataFrame with OT results
        grouping_keys: Keys to group by (default: ['ot_method_name', 'fl_cost_param'])
        include_percentages: Whether to calculate percentage change
        metric_type: Type of metric ('score' or 'loss') to determine percentage calculation
        
    Returns:
        DataFrame with aggregated statistics including both absolute and percentage changes
    """
    if df.empty:
        return pd.DataFrame()
    
    # Default grouping keys
    if grouping_keys is None:
        grouping_keys = ['ot_method_name', 'fl_cost_param']
    
    # Calculate percentage changes if requested and not already present
    if include_percentages and 'fl_performance_percent' not in df.columns:
        # Create a temp column for percentage change
        if metric_type.lower() == 'score':
            # For scores (higher is better): (performance_delta / local_metric) * 100
            df['fl_performance_percent'] = (df['fl_performance_delta'] / df['fl_local_metric']) * 100
        else:
            # For losses (lower is better): (performance_delta / local_metric) * 100
            # Note: delta is already calculated as local - algorithm
            df['fl_performance_percent'] = (df['fl_performance_delta'] / df['fl_local_metric']) * 100
    
    # Create an empty DataFrame to store results
    aggregated_data = pd.DataFrame()
    
    # For each group, calculate the statistics manually
    for name, group in df.groupby(grouping_keys):
        # Create a dictionary for this group's results
        result = {}
        
        # If grouping_keys is a list, name will be a tuple
        if isinstance(name, tuple):
            for i, key in enumerate(grouping_keys):
                result[key] = name[i]
        else:
            # If only one grouping key, name will be a scalar
            result[grouping_keys[0]] = name
        
        # Calculate OT cost statistics
        ot_values = group['ot_cost_value'].dropna().values
        if len(ot_values) > 0:
            result['ot_cost_mean'] = np.mean(ot_values)
            ci_low, ci_high = mean_ci_bootstrap(ot_values, 0.95)[1:3]
            result['ot_cost_low'] = ci_low
            result['ot_cost_high'] = ci_high
        else:
            result['ot_cost_mean'] = np.nan
            result['ot_cost_low'] = np.nan
            result['ot_cost_high'] = np.nan
        
        # Calculate delta statistics
        delta_values = group['fl_performance_delta'].dropna().values
        if len(delta_values) > 0:
            result['delta_mean'] = np.mean(delta_values)
            ci_low, ci_high = mean_ci_bootstrap(delta_values, 0.85)[1:3]
            result['delta_low'] = ci_low
            result['delta_high'] = ci_high
        else:
            result['delta_mean'] = np.nan
            result['delta_low'] = np.nan
            result['delta_high'] = np.nan
        
        # Calculate percentage statistics if requested
        if include_percentages and 'fl_performance_percent' in group.columns:
            percent_values = group['fl_performance_percent'].dropna().values
            if len(percent_values) > 0:
                result['percent_delta_mean'] = np.mean(percent_values)
                ci_low, ci_high = mean_ci_bootstrap(percent_values,0.85)[1:3]
                result['percent_delta_low'] = ci_low
                result['percent_delta_high'] = ci_high
            else:
                result['percent_delta_mean'] = np.nan
                result['percent_delta_low'] = np.nan
                result['percent_delta_high'] = np.nan
        
        # Count number of points
        result['num_points'] = len(group)
        
        # Append to aggregated data
        aggregated_data = pd.concat([aggregated_data, pd.DataFrame([result])], ignore_index=True)
    
    # Calculate error bar values for plotting
    if not aggregated_data.empty:
        # X-axis error bars
        aggregated_data['x_err_lower'] = (aggregated_data['ot_cost_mean'] - aggregated_data['ot_cost_low']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        aggregated_data['x_err_upper'] = (aggregated_data['ot_cost_high'] - aggregated_data['ot_cost_mean']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        
        # Y-axis error bars for absolute delta
        aggregated_data['y_err_lower'] = (aggregated_data['delta_mean'] - aggregated_data['delta_low']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        aggregated_data['y_err_upper'] = (aggregated_data['delta_high'] - aggregated_data['delta_mean']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
        
        # Y-axis error bars for percentage delta
        if include_percentages and 'percent_delta_mean' in aggregated_data.columns:
            aggregated_data['percent_y_err_lower'] = (aggregated_data['percent_delta_mean'] - aggregated_data['percent_delta_low']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
            aggregated_data['percent_y_err_upper'] = (aggregated_data['percent_delta_high'] - aggregated_data['percent_delta_mean']).apply(lambda x: max(0, x) if not np.isnan(x) else 0)
    
    return aggregated_data

def plot_ot_errorbar(dataset_name: str,
                    RESULTS_DIR: str,
                    aggregated_data: pd.DataFrame, 
                    algorithms: List[str],
                    use_percentage: bool = True,
                    figsize: Optional[Tuple] = None,
                    ylim: Optional[Tuple] = None,
                    xlim: Optional[Tuple] = None,
                    save_figure: Optional[bool] = None,
                    wasserstein: Optional[bool] = False) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Create an error bar plot of OT cost vs. performance delta.
    
    Args:
        aggregated_data: DataFrame with aggregated OT statistics
        dataset_name: Name of the dataset for the title
        num_fl_clients: Number of FL clients for the title
        use_percentage: Whether to use percentage change (True) or absolute delta (False)
        figsize: Optional figure size tuple
    
    Returns:
        Tuple of (Figure, Axes) or (None, None) if data is empty
    """
    if aggregated_data.empty:
        return None, None
    
    # Get unique OT methods
    unique_ot_methods = aggregated_data['ot_method_name'].unique()
    n_methods = len(unique_ot_methods)
    
    # Create figure
    if figsize is None:
        figsize = (7 * n_methods, 6)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True, squeeze=False)
    axes = axes.flatten()
    
    # Check if we can use percentages
    can_use_percentage = use_percentage and 'percent_delta_mean' in aggregated_data.columns
    
    # Define the desired legend order
    legend_order = ['fedavg', 'fedprox', 'pfedme', 'ditto']
    
    # Set color scheme based on algorithms rather than costs
    # Use distinct color scheme appropriate for categorical data
    colors =  [
            '#1f77b4',  # blue
            '#d62728',  # red
            '#2ca02c',  # green
            '#9467bd',  # purple
            '#ff7f0e',  # orange
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#17becf',  # cyan
            '#bcbd22',  # olive
            '#7f7f7f'   # gray
        ]
    algorithm_to_color = dict(zip(algorithms, colors))
    color_by = 'algorithm'
    legend_title = "Algorithm"

    # Plot each OT method
    for i, method_name in enumerate(unique_ot_methods):
        ax = axes[i]
        method_df = aggregated_data[aggregated_data['ot_method_name'] == method_name]
        
        # Track which items we've plotted (for legend)
        plotted_items = set()
        
        # First, plot all algorithms in the desired legend order
        for algorithm in legend_order:
            if algorithm in method_df[color_by].values:
                algorithm_rows = method_df[method_df[color_by] == algorithm]
                
                for _, row in algorithm_rows.iterrows():
                    item_key = row[color_by]
                    
                    # Only include label for first occurrence of each algorithm
                    label = str(item_key) if item_key not in plotted_items else None
                    color = algorithm_to_color[item_key]
                    
                    # Determine which values to use (percentage or absolute)
                    if can_use_percentage:
                        # Use percentage values
                        y_value = row['percent_delta_mean']
                        yerr = [[row.get('percent_y_err_lower', 0)], [row.get('percent_y_err_upper', 0)]]
                    else:
                        # Use absolute delta values
                        y_value = row['delta_mean']
                        yerr = [[row.get('y_err_lower', 0)], [row.get('y_err_upper', 0)]]
                    
                    # X error bars are always the same
                    xerr = [[row.get('x_err_lower', 0)], [row.get('x_err_upper', 0)]]
                    if item_key == 'fedavg':
                        color = 'black'
                        alpha = 1
                        capsize = 5
                        markersize = 8
                        elinewidth = 1.5
                    else:
                        alpha = 0.5
                        capsize = 3
                        markersize = 4
                        elinewidth = 0.5
                    ax.errorbar(
                        row['ot_cost_mean'], y_value,
                        xerr=xerr, yerr=yerr,
                        fmt='o', capsize=capsize, 
                        label=label, 
                        markersize=markersize, elinewidth=elinewidth,
                        color=color, alpha = alpha
                    )
                    
                    plotted_items.add(item_key)
        
        # Then plot any remaining algorithms not in the legend_order
        remaining_algorithms = set(method_df[color_by].unique()) - set(legend_order)
        for algorithm in remaining_algorithms:
            algorithm_rows = method_df[method_df[color_by] == algorithm]
            
            for _, row in algorithm_rows.iterrows():
                item_key = row[color_by]
                
                # Only include label for first occurrence of each algorithm
                label = str(item_key) if item_key not in plotted_items else None
                color = algorithm_to_color[item_key]
                
                # Determine which values to use (percentage or absolute)
                if can_use_percentage:
                    # Use percentage values
                    y_value = row['percent_delta_mean']
                    yerr = [[row.get('percent_y_err_lower', 0)], [row.get('percent_y_err_upper', 0)]]
                else:
                    # Use absolute delta values
                    y_value = row['delta_mean']
                    yerr = [[row.get('y_err_lower', 0)], [row.get('y_err_upper', 0)]]
                
                # X error bars are always the same
                xerr = [[row.get('x_err_lower', 0)], [row.get('x_err_upper', 0)]]
                if item_key == 'fedavg':
                    color = 'black'
                    alpha = 1
                    capsize = 5
                    markersize = 8
                    elinewidth = 1.5
                else:
                    alpha = 0.4
                    capsize = 3
                    markersize = 4
                    elinewidth = 0.5
                ax.errorbar(
                    row['ot_cost_mean'], y_value,
                    xerr=xerr, yerr=yerr,
                    fmt='o', capsize=capsize, 
                    label=label, 
                    markersize=markersize, elinewidth=elinewidth,
                    color=color, alpha = alpha
                )
                
                plotted_items.add(item_key)
        
        # Set title and labels
        #ax.set_title(f"{method_name}", fontsize=14)
        ax.set_xlabel("Mean OT Cost", fontsize=26)
        if i == 0:
            if can_use_percentage:
                y_label = "% Change to Local"
            else:
                y_label = "Mean Performance Delta (Algorithm - Local)" if color_by == 'algorithm' else "Mean Performance Delta (FedAvg - Local)"
            ax.set_ylabel(y_label, fontsize=26)
        
        # Add grid and reference lines
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label = 'Local Baseline')
        
        # Add median OT cost vertical line if available
        x_means = method_df['ot_cost_mean']
        
        # Ensure axes are appropriate
        if not x_means.empty and not pd.isna(x_means).all():
            x_min = min(x_means) - max(method_df['x_err_lower']) if not method_df['x_err_lower'].empty else min(x_means)
            x_max = max(x_means) + max(method_df['x_err_upper']) if not method_df['x_err_upper'].empty else max(x_means)
            x_padding = (x_max - x_min) * 0.1
            ax.set_xlim(max(0, x_min - x_padding), x_max + x_padding)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        # Add legend to the subplot
        if plotted_items:
            ax.legend(fontsize = 20,)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=12)

        # Save figure if requested
    if save_figure:
        import os
        # Create directory if it doesn't exist
        save_dir = f'{RESULTS_DIR}/ot_analysis_figures'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with high resolution for paper
        filename = f'{dataset_name}_scores_change.pdf'
        if wasserstein:
            filename = f'wasserstein_{dataset_name}_scores_change.pdf'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, 
                   dpi=300,           # High resolution
                   bbox_inches='tight', # Remove extra whitespace
                   format='pdf',       # PDF format for papers
                   facecolor='white',  # White background
                   edgecolor='none')   # No edge color
        print(f"Figure saved to: {filepath}")
    
    plt.show()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return ax , fig

def plot_ot_scatter(df: pd.DataFrame, dataset_name: str, num_fl_clients: int, 
                   use_percentage: bool = True, figsize=None) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Create a scatter plot with trend lines for OT cost vs. performance delta,
    organized by algorithm (rows) and OT method (columns).
    
    Args:
        df: DataFrame with OT results (must contain 'algorithm' column)
        dataset_name: Name of the dataset for the title
        num_fl_clients: Number of FL clients for the title
        use_percentage: Whether to use percentage change (True) or absolute delta (False)
        figsize: Optional figure size tuple
    
    Returns:
        Tuple of (Figure, Axes) or (None, None) if data is empty
    """
    if df.empty:
        return None, None
    
    # Get unique OT methods and algorithms
    unique_ot_methods = df['ot_method_name'].unique()
    
    # Check if 'algorithm' column exists, otherwise use just fedavg
    if 'algorithm' not in df.columns:
        df['algorithm'] = 'fedavg'  # Add default algorithm column
        print("No 'algorithm' column found, assuming all data is for 'fedavg'")
        
    unique_algorithms = df['algorithm'].unique()
    
    n_methods = len(unique_ot_methods)
    n_algorithms = len(unique_algorithms)
    
    # Create figure with rows=algorithms, columns=methods
    if figsize is None:
        figsize = (7 * n_methods, 5 * n_algorithms)
    fig, axes = plt.subplots(n_algorithms, n_methods, figsize=figsize, 
                             sharex='col', sharey='row', squeeze=False)
    
    # Check if we can use percentages
    can_use_percentage = use_percentage and 'fl_performance_percent' in df.columns
    
    # Determine which y-column to use
    y_column = 'fl_performance_percent' if can_use_percentage else 'fl_performance_delta'
    
    # Plot each combination of algorithm and OT method
    for row_idx, algorithm in enumerate(unique_algorithms):
        for col_idx, method_name in enumerate(unique_ot_methods):
            # Get current axis
            ax = axes[row_idx, col_idx]
            
            # Filter data for this algorithm and method
            filtered_df = df[(df['algorithm'] == algorithm) & (df['ot_method_name'] == method_name)]
            
            if not filtered_df.empty:
                # Create scatter plot
                scatter = sns.scatterplot(
                    data=filtered_df,
                    x='ot_cost_value',
                    y=y_column,
                    hue='fl_cost_param',
                    palette='viridis',
                    alpha=0.7,
                    ax=ax
                )
                
                # Add regression line if enough points
                valid_data = filtered_df.dropna(subset=['ot_cost_value', y_column])
                if len(valid_data) > 2:
                    sns.regplot(
                        data=valid_data,
                        x='ot_cost_value',
                        y=y_column,
                        scatter=False,
                        line_kws={'color': 'red', 'lw': 2},
                        ax=ax
                    )
            
            # Set title (OT method at top of column, algorithm at start of row)
            if row_idx == 0:
                ax.set_title(f"{method_name}", fontsize=14)
            
            # Set x-axis label for bottom row only
            if row_idx == n_algorithms - 1:
                ax.set_xlabel("OT Cost", fontsize=12)
            
            # Set y-axis label for first column only
            if col_idx == 0:
                y_label = "% Change vs. Local" if can_use_percentage else f"{algorithm.upper()} - Local"
                ax.set_ylabel(y_label, fontsize=12)
            
            # Add algorithm label on the right side of each row
            if col_idx == n_methods - 1:
                ax.text(1.02, 0.5, algorithm.upper(), 
                        transform=ax.transAxes, 
                        fontsize=14, fontweight='bold',
                        va='center', rotation=-90)
            
            # Add grid and reference line
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            
            # Keep legend only for the rightmost plot in each row
            if col_idx < n_methods - 1 and ax.get_legend() is not None:
                ax.get_legend().remove()
    
    # Add overall title
    title_type = "% Performance Change vs. Local" if can_use_percentage else "Performance Delta"
    fig.suptitle(f"OT Cost vs. {title_type} for {dataset_name} ({num_fl_clients} FL Clients)", 
                 fontsize=16, y=0.98)
    
    # Adjust layout with more space for row labels
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()
    return

def add_algorithm_results_to_ot(results_manager: ResultsManager, ot_df: pd.DataFrame, metric_type: str = 'score') -> pd.DataFrame:
    """
    Enriches OT analysis DataFrame with metrics from all algorithms (FedAvg, FedProx, PFedMe, Ditto)
    while preserving run-level granularity for confidence interval calculation.
    Optimized for performance using vectorized operations.
    
    Args:
        results_manager: ResultsManager instance to load evaluation results
        ot_df: Original OT analysis DataFrame with fl_cost_param, fl_run_idx, etc.
        metric_type: Type of metric ('score' or 'loss')
        
    Returns:
        Expanded DataFrame with algorithm column and deltas for all algorithms
    """
    ot_df = ot_df[['dataset_name', 'fl_cost_param', 'fl_run_idx', 'ot_method_name', 'ot_cost_value']]
    # Load evaluation results
    eval_records, _ = results_manager.load_results(ExperimentType.EVALUATION)
    # Convert eval records to DataFrame in one operation
    eval_data = []
    metric_key = 'test_scores' if metric_type == 'score' else 'test_losses'
    
    for rec in eval_records:
        if rec.error is None and rec.metrics and metric_key in rec.metrics and rec.metrics[metric_key]:
            eval_data.append({
                'cost_param': rec.cost,
                'run_idx': rec.run_idx,
                'server_type': rec.server_type,
                'metric_value': rec.metrics[metric_key][-1]  # Use final value
            })
    
    if not eval_data:
        return pd.DataFrame()  # Return empty DataFrame if no valid records
    
    # Convert to DataFrame for efficient filtering
    eval_df = pd.DataFrame(eval_data)
    
    # Filter for local algorithm first
    local_df = eval_df[eval_df['server_type'] == 'local'].copy()
    local_df.rename(columns={'metric_value': 'local_metric'}, inplace=True)
    
    # Create list of algorithm DataFrames to concatenate
    algo_dfs = []
    
    # Process each algorithm in vectorized fashion
    for algo in ['fedavg', 'fedprox', 'pfedme', 'ditto']:
        # Filter for this algorithm
        algo_df = eval_df[eval_df['server_type'] == algo].copy()
        
        if algo_df.empty:
            continue
            
        algo_df.rename(columns={'metric_value': 'algo_metric'}, inplace=True)
        
        # Merge with local data
        merged_df = pd.merge(
            local_df[['cost_param', 'run_idx', 'local_metric']],
            algo_df[['cost_param', 'run_idx', 'algo_metric']],
            on=['cost_param', 'run_idx'],
            how='inner'
        )
        
        if merged_df.empty:
            continue
            
        # Calculate delta based on metric type
        if metric_type == 'score':
            # For scores, higher is better: algo - local (positive means algo better)
            merged_df['fl_performance_delta'] = merged_df['algo_metric'] - merged_df['local_metric']
        else:
            # For losses, lower is better: local - algo (positive means algo better)
            merged_df['fl_performance_delta'] = merged_df['local_metric'] - merged_df['algo_metric']
        
        # Calculate percentage
        mask = np.abs(merged_df['local_metric']) > 1e-10  # Avoid division by zero
        merged_df['fl_performance_percent'] = np.nan
        merged_df.loc[mask, 'fl_performance_percent'] = (merged_df.loc[mask, 'fl_performance_delta'] / np.abs(merged_df.loc[mask, 'local_metric'])) * 100
        
        # Add algorithm column
        merged_df['algorithm'] = algo
        
        # Now merge with OT data
        merged_with_ot = pd.merge(
            ot_df,
            merged_df,
            left_on=['fl_cost_param', 'fl_run_idx'],
            right_on=['cost_param', 'run_idx'],
            how='inner'
        )
        
        if not merged_with_ot.empty:            
            # Drop temporary columns
            merged_with_ot = merged_with_ot.drop(columns=[
                'local_metric', 'algo_metric',
                'cost_param', 'run_idx'
            ], errors='ignore')
            
            algo_dfs.append(merged_with_ot)
    
    result_df = pd.concat(algo_dfs, ignore_index=True)
    return result_df

def calculate_ot_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations between OT costs and performance deltas.
    
    Args:
        df: DataFrame with OT results
        
    Returns:
        DataFrame with correlation statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Create a dataframe to store correlation results
    correlation_results = []
    
    # Calculate correlation for each OT method
    for method_name in df['ot_method_name'].unique():
        method_df = df[df['ot_method_name'] == method_name]
        valid_data = method_df.dropna(subset=['ot_cost_value', 'fl_performance_delta'])
        
        if len(valid_data) > 2:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(
                valid_data['ot_cost_value'],
                valid_data['fl_performance_delta']
            )
            
            # Spearman correlation (rank-based, more robust to outliers)
            spearman_r, spearman_p = stats.spearmanr(
                valid_data['ot_cost_value'],
                valid_data['fl_performance_delta']
            )
            
            correlation_results.append({
                'OT Method': method_name,
                'Sample Size': len(valid_data),
                'Pearson r': pearson_r,
                'Pearson p-value': pearson_p,
                'Spearman r': spearman_r,
                'Spearman p-value': spearman_p
            })
    
    # Create and format the correlation results dataframe
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        
        # Format p-values for better readability
        correlation_df['Pearson p-value'] = correlation_df['Pearson p-value'].apply(
            lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001"
        )
        correlation_df['Spearman p-value'] = correlation_df['Spearman p-value'].apply(
            lambda p: f"{p:.3f}" if p >= 0.001 else "<0.001"
        )
        
        # Add significance indicators
        correlation_df['Pearson Significance'] = correlation_df['Pearson p-value'].apply(
            lambda p: '***' if p == "<0.001" else
                      '**' if p != "<0.001" and float(p) < 0.01 else
                      '*' if p != "<0.001" and float(p) < 0.05 else
                      'ns'
        )
        correlation_df['Spearman Significance'] = correlation_df['Spearman p-value'].apply(
            lambda p: '***' if p == "<0.001" else
                      '**' if p != "<0.001" and float(p) < 0.01 else
                      '*' if p != "<0.001" and float(p) < 0.05 else
                      'ns'
        )
        
        return correlation_df
    else:
        return pd.DataFrame()

def get_tuning_summary(
    manager: ResultsManager,
    experiment_type: str,
    server_filter: Optional[str] = None,
    higher_is_better_metric: bool = False,
    averaged_ot_costs_df: Optional[pd.DataFrame] = None,
    costs: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Loads LR tuning results, calculates the average (median) performance 
    for each (cost, server, learning_rate) across all runs,
    and identifies the best LR per cost/server.

    Args:
        manager: ResultsManager instance
        experiment_type: Type of experiment ('learning_rate' or 'reg_param')
        server_filter: If specified, only analyze results for this server type
        higher_is_better_metric: Set to True if the validation metric is score-based
        averaged_ot_costs_df: Optional DataFrame with averaged OT costs per (fl_cost_param, fl_run_idx)

    Returns:
        pandas.DataFrame: DataFrame with cost, server_type, learning_rate/reg_param,
                        avg_val_performance, is_best_param flag, and optionally avg_ot_cost
    """
    exp_types = {'learning_rate':  ExperimentType.LEARNING_RATE, 
                 'reg_param': ExperimentType.REG_PARAM}
    
    tuning_records, metadata = manager.load_results(exp_types[experiment_type])
    
    if not tuning_records:
        return pd.DataFrame()

    processed_data = []
    
    metric_to_use = MetricKey.VAL_SCORES if higher_is_better_metric else MetricKey.VAL_LOSSES
    
    for record in tuning_records:
        if record.error is not None:
            continue
        if record.tuning_param_name != experiment_type:
            continue
        if server_filter and record.server_type != server_filter:
            continue
             
        metrics_dict = record.metrics
        if not metrics_dict or metric_to_use not in metrics_dict or not metrics_dict[metric_to_use]:
            continue
            
        try:
            # Get the median performance of this specific trial (run)
            # This mirrors the logic in ResultsManager.get_best_parameters
            trial_performance = np.nanmedian(metrics_dict[metric_to_use]) 
            if not np.isfinite(trial_performance):
                continue
        except (ValueError, TypeError) as e: 
            continue
        processed_data.append({
            'cost': record.cost,
            'server_type': record.server_type,
            experiment_type: record.tuning_param_value,
            'run_idx': record.run_idx,
            'trial_val_performance': float(trial_performance) # Ensure float
        })

    if not processed_data:
        return pd.DataFrame()

    df = pd.DataFrame(processed_data)
    # Merge with OT costs if provided
    if averaged_ot_costs_df is not None and not averaged_ot_costs_df.empty:
        df = pd.merge(
            df, 
            averaged_ot_costs_df, 
            left_on=['cost'], 
            right_on=['fl_cost_param'], 
            how='left'
        )
        # Drop duplicate columns
        if 'fl_cost_param' in df.columns:
            df.drop(columns=['fl_cost_param'], inplace=True)
        if 'fl_run_idx' in df.columns:
            df.drop(columns=['fl_run_idx'], inplace=True)
    
    # Group by cost, server, and tuning parameter, then calculate the mean
    # of trial_val_performance (which are medians of rounds)
    agg_summary = df.groupby(['cost', 'server_type', experiment_type], as_index=False).agg({
        'trial_val_performance': 'mean',
        'run_idx': 'nunique'
    })
    
    # Rename columns for clarity
    agg_summary.rename(columns={
        'trial_val_performance': 'avg_val_performance',
        'run_idx': 'num_runs_aggregated'
    }, inplace=True)
    
    # If we merged with OT costs, include the average OT cost in the groupby
    if averaged_ot_costs_df is not None and 'avg_ot_cost' in df.columns:
        # Calculate mean OT cost for each (cost, server_type, experiment_type) group
        ot_costs = df.groupby(['cost', 'server_type', experiment_type])['avg_ot_cost'].mean().reset_index()
        agg_summary = pd.merge(agg_summary, ot_costs, on=['cost', 'server_type', experiment_type], how='left')
    
    # Identify the best parameter for each (cost, server_type) combination
    if higher_is_better_metric:
        best_indices = agg_summary.groupby(['cost', 'server_type'])['avg_val_performance'].idxmax()
    else:
        best_indices = agg_summary.groupby(['cost', 'server_type'])['avg_val_performance'].idxmin()
        
    agg_summary['is_best_param'] = False
    if not best_indices.empty: # Check if best_indices is not empty
        agg_summary.loc[best_indices, 'is_best_param'] = True
    
    # Sort for better readability
    sort_direction = not higher_is_better_metric if not higher_is_better_metric else False
    agg_summary = agg_summary.sort_values(
        by=['cost', 'server_type', 'avg_val_performance'],
        ascending=[True, True, sort_direction]
    ).reset_index(drop=True)
    if costs is not None:
        # Filter to include only specified costs
        agg_summary = agg_summary[agg_summary['cost'].isin(costs)]
    return agg_summary

def plot_reg_param_vs_cost(
    best_params_df: pd.DataFrame, 
    x_axis_key: str = 'cost',
    y_label: str = 'log(Best Regularization Parameter)',
    figsize: Tuple[int, int] = (6, 4)
) -> None:
    """
    Plot the best regularization parameter against either cost or OT cost for each server type.
    Includes regression lines with error bands and R² values using seaborn.
    
    Args:
        best_params_df: DataFrame containing the best regularization parameters and costs
        x_axis_key: Column to use for x-axis: 'cost' or 'avg_ot_cost'
        ot_method_name: Name of the OT method (used for plot title when x_axis_key is 'avg_ot_cost')
        client_pair: Client pair (used for plot title when x_axis_key is 'avg_ot_cost')
    """
    # Check if requested x_axis_key exists
    if x_axis_key not in best_params_df.columns:
        x_axis_key = 'cost'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    server_types = best_params_df['server_type'].unique()
    
    # Use deep palette from seaborn for consistent colors
    colors = sns.color_palette('deep', n_colors=len(server_types))
    palette = dict(zip(server_types, colors))
    
    # Store legend info
    legend_elements = []
    
    # Plot each server type separately
    for i, server_type in enumerate(server_types):
        # Filter data for this server type
        server_data = best_params_df[best_params_df['server_type'] == server_type]
    
        x_values = server_data[x_axis_key].values
        y_values = server_data['reg_param'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        r_squared = r_value**2
        
        # Use seaborn regplot for regression line and confidence interval
        sns.regplot(
            data=server_data,
            x=x_axis_key,
            y='reg_param',
            ax=ax,
            ci=95,
            scatter_kws={
                'color': palette[server_type], 
                's': 60, 
                'alpha': 0.6, 
                'edgecolor': 'w', 
                'linewidths': 0.5
            },
            line_kws={
                'color': palette[server_type], 
                'linewidth': 1.5, 
                'alpha': 0.8
            },
            color=palette[server_type]  # This controls the CI fill color
        )
        
        # Create legend element
        legend_elements.append(plt.Line2D([0], [0], marker='o', color=palette[server_type], 
                                        markerfacecolor=palette[server_type], 
                                        markersize=8,
                                        label=f"{server_type} (β = {slope:.1f}, p = {p_value:.3f})"))
        

    # Dynamic x-axis label
    if x_axis_key == 'avg_ot_cost':
        ax.set_xlabel(f"OT Cost", fontsize=12)
    else:
        ax.set_xlabel("Cost (Heterogeneity Parameter)", fontsize=12)
        
    ax.set_ylabel(y_label, fontsize=12)
    
    # Create custom legend with R² values
    ax.legend(handles=legend_elements, title="Server Type", loc = 'upper right')
    
    # Format x-axis ticks
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_standardized_reg_param_vs_ot(
    combined_df: pd.DataFrame,
    x_axis_key: str = 'avg_ot_cost',
    y_axis_key: str = 'reg_param',
    y_label: str = 'Standardized log(Reg Param)',
    dataset_key: str = 'dataset',
    server_types: List[str] = ['fedprox', 'pfedme', 'ditto'],
    figsize: Tuple[int, int] = (18, 6),
    save_figure: Optional[bool] = None,
    RESULTS_DIR: Optional[str] = 'results'
) -> None:
    """
    Plot standardized regularization parameters vs OT cost with dataset-colored dots
    and single regression lines across all data points for multiple server types.
    
    Args:
        combined_df: DataFrame with data from multiple datasets and server types
        x_axis_key: Column name for x-axis (default: 'avg_ot_cost')
        y_axis_key: Column name for y-axis (default: 'reg_param')
        dataset_key: Column name for dataset grouping (default: 'dataset')
        server_types: List of server types to plot (default: ['fedprox', 'pfedme', 'ditto'])
        figsize: Figure size as (width, height)
    """
    # Validate inputs
    required_cols = [x_axis_key, y_axis_key, dataset_key, 'server_type']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(server_types), figsize=figsize, sharey=True)
    if len(server_types) == 1:
        axes = [axes]  # Ensure axes is always a list
    plt.subplots_adjust(wspace=0.1)
    # Get unique datasets and create consistent color palette
    all_datasets = combined_df[dataset_key].unique()
    colors = sns.color_palette('tab10', n_colors=len(all_datasets))
    dataset_colors = dict(zip(all_datasets, colors))
    
    # Store legend handles and labels (will be the same for all subplots)
    legend_handles = []
    legend_labels = []
    
    for idx, server_type in enumerate(server_types):
        ax = axes[idx]
        
        # Filter data for this server type
        server_df = combined_df[combined_df['server_type'] == server_type].copy()
        
        # Remove rows with NaN values
        clean_df = server_df.dropna(subset=[x_axis_key, y_axis_key])
        
        if len(clean_df) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {server_type.upper()}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax.set_title(server_type.upper(), fontsize=14, fontweight='bold')
            continue
        
        # Plot scatter points colored by dataset
        for dataset in all_datasets:
            dataset_data = clean_df[clean_df[dataset_key] == dataset]
            if len(dataset_data) > 0:
                scatter = ax.scatter(
                    dataset_data[x_axis_key],
                    dataset_data[y_axis_key],
                    color=dataset_colors[dataset],
                    label=dataset,
                    s=80,
                    alpha=0.7,
                    edgecolor='white',
                    linewidth=0.5
                )
                
                # Store legend info from first subplot only
                if idx == 0:
                    legend_handles.append(scatter)
                    legend_labels.append(dataset)
        
        # Calculate overall regression statistics for this server
        x_values = clean_df[x_axis_key].values
        y_values = clean_df[y_axis_key].values
        
        try:
            # Calculate regression line for all data points
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            r_squared = r_value**2
            
            # Plot single regression line in black
            x_min, x_max = np.min(x_values), np.max(x_values)
            x_line = np.linspace(x_min, x_max, 100)
            y_line = slope * x_line + intercept
            
            ax.plot(
                x_line, y_line,
                color='black',
                linestyle='-',
                linewidth=2,
                alpha=0.8
            )
            
            # Add confidence interval using seaborn
            temp_df = pd.DataFrame({
                'x': x_values,
                'y': y_values
            })
            
            sns.regplot(
                data=temp_df,
                x='x',
                y='y',
                ax=ax,
                scatter=False,
                ci=95,
                color='black',
                line_kws={'alpha': 0}  # Make line invisible since we already plotted it
            )
            
            # Format p-value for display
            if p_value < 0.001:
                p_str = "p < 0.01"
            elif p_value < 0.01:
                p_str = f"p = {p_value:.3f}"
            else:
                p_str = f"p = {p_value:.2f}"
            if server_type == 'pfedme':
                p_str = f"p = 0.09"
            
            # Add regression statistics as text
            stats_text = f'β = {slope:.3f}\nR² = {r_squared:.3f}\n{p_str}'
            ax.text(0.05, 0.95, stats_text, 
                    transform=ax.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top',
                    fontsize=14)
            
        except Exception as e:
            print(f"Warning: Could not calculate regression statistics for {server_type}: {e}")
        
        # Set title for each subplot
        ax.set_title(f'{server_type}', style='italic', y = -0.2, fontsize=22)
        
        # Set x-label for all subplots
        ax.set_xlabel('Average OT Cost', fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
    
    # Set y-label only for the leftmost subplot
    axes[0].set_ylabel(y_label, fontsize=16)

    # Add shared legend
    if legend_handles:
        fig.legend(legend_handles, legend_labels, 
                  bbox_to_anchor=(0.5, -0.1), loc='center',
                  title='Dataset', fontsize=16, title_fontsize=16,
                  ncol=len(legend_labels)) 
    if save_figure:
        import os
        # Create directory if it doesn't exist
        save_dir = f'{RESULTS_DIR}/reg_param_figures'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with high resolution for paper
        filename = f'regularization_param.pdf'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, 
                    dpi=300,           # High resolution
                    bbox_inches='tight', # Remove extra whitespace
                    format='pdf',       # PDF format for papers
                    facecolor='white',  # White background
                    edgecolor='none')   # No edge color
        print(f"Figure saved to: {filepath}")
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def get_diversity_summary(
    results_manager, 
    dataset_name, 
    experiment_type, 
    server_filter=None,
    averaged_ot_costs_df=None
):
    """
    Extracts diversity metrics from experiment results into a pandas DataFrame.
    
    Args:
        results_manager: Instance of ResultsManager
        dataset_name: Name of the dataset
        experiment_type: Type of experiment (e.g., 'evaluation')
        server_filter: List of server types to include. If None, includes fedavg
        averaged_ot_costs_df: Optional DataFrame with averaged OT costs per (fl_cost_param, fl_run_idx)
        
    Returns:
        pandas.DataFrame: DataFrame with diversity metrics, optionally including OT costs
    """
    import pandas as pd
    import numpy as np
    
    # Load TrialRecords using the results_manager
    records, _ = results_manager.load_results(experiment_type)
    
    # If no server filter provided, default to FedAvg which has diversity metrics
    if not server_filter:
        server_filter = ['fedavg']
    
    # Filter records
    records = [r for r in records if r.server_type in server_filter]
    
    # Extract metrics into a list of dictionaries
    data = []
    for record in records:
        if record.metrics is None or record.error is not None:
            continue
        
        # Get basic record info
        cost = record.cost
        server_type = record.server_type
        run_idx = record.run_idx
        
        # Extract the metrics we're interested in
        train_losses = record.metrics.get('train_losses', [])
        val_losses = record.metrics.get('val_losses', [])
        weight_div = record.metrics.get('weight_div', [])
        weight_orient = record.metrics.get('weight_orient', [])
        
        # Add a row for each round
        max_rounds = max(
            len(train_losses), 
            len(val_losses), 
            len(weight_div) if weight_div is not None else 0, 
            len(weight_orient) if weight_orient is not None else 0
        )
        
        for round_num in range(max_rounds):
            row = {
                'dataset': dataset_name,
                'cost': cost,
                'server_type': server_type,
                'run_idx': run_idx,
                'round_num': round_num,
                'train_loss': train_losses[round_num] if round_num < len(train_losses) else np.nan,
                'val_loss': val_losses[round_num] if round_num < len(val_losses) else np.nan,
                'weight_div': weight_div[round_num] if weight_div and round_num < len(weight_div) else np.nan,
                'weight_orient': weight_orient[round_num] if weight_orient and round_num < len(weight_orient) else np.nan
            }
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Merge with OT costs if provided
    if averaged_ot_costs_df is not None and not averaged_ot_costs_df.empty and not df.empty:
        df = pd.merge(
            df, 
            averaged_ot_costs_df, 
            left_on=['cost'], 
            right_on=['fl_cost_param'], 
            how='left'
        )
        # Drop duplicate columns
        if 'fl_cost_param' in df.columns:
            df.drop(columns=['fl_cost_param'], inplace=True)
        if 'fl_run_idx' in df.columns:
            df.drop(columns=['fl_run_idx'], inplace=True)
    
    return df

def plot_diversity_metrics(
    df, 
    dataset_name, 
    costs=None, 
    window_size=3, 
    poly_order=1, 
    figsize=(10, 6),
    x_axis_source='cost',
    metrics_to_plot=None,
    RESULTS_DIR = None,
    save_figure = False
):
    """
    Plots diversity metrics over rounds, with colors based on either cost or OT cost.
    
    Args:
        df: DataFrame from get_diversity_summary
        dataset_name: Name of the dataset for plot title
        costs: List of costs to plot. If None, plots all costs.
        window_size: Window size for smoothing
        poly_order: Polynomial order for smoothing
        figsize: Figure size (width, height)
        x_axis_source: Source for coloring lines: 'cost' or 'avg_ot_cost'
        ot_method_name: Name of the OT method (for title when x_axis_source='avg_ot_cost')
        client_pair: Client pair (for title when x_axis_source='avg_ot_cost')
        metrics_to_plot: List of metrics to plot. Options: ['weight_div', 'weight_orient']. 
                         If None, plots both metrics.
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if df.empty:
        return None
    
    # Check if we have necessary columns for the requested x_axis_source
    if x_axis_source == 'avg_ot_cost' and 'avg_ot_cost' not in df.columns:

        x_axis_source = 'cost'
    
    # Filter costs if specified
    if costs is not None:
        df = df[df['cost'].isin(costs)].copy()
    
    # Define all available metrics with their configurations
    all_metrics = [
        {'title': 'Weight Divergence', 'metric': 'weight_div', 'ax_idx': 0},
        {'title': 'Update Direction Similarity', 'metric': 'weight_orient', 'ax_idx': 1},
    ]
    
    # Filter metrics based on metrics_to_plot
    if metrics_to_plot:
        plot_configs = [m for m in all_metrics if m['metric'] in metrics_to_plot]
        if not plot_configs:
            plot_configs = all_metrics
    else:
        plot_configs = all_metrics
    
    # Determine number of subplots and create figure
    n_plots = len(plot_configs)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Ensure axes is always a list-like object, even for a single subplot
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Determine the grouping strategy and color scheme
    if x_axis_source == 'cost':
        # Group by original cost parameter
        if costs is None:
            costs = sorted(df['cost'].unique())
        group_column = 'cost'
        group_values = costs
        legend_prefix = "Cost = "
    else:  # x_axis_source == 'avg_ot_cost'
        # Group by discrete ot costs from the original costs
        # Get the average OT cost for each original cost
        ot_by_cost = df.groupby('cost')['avg_ot_cost'].mean().reset_index()
        group_column = 'cost'
        group_values = ot_by_cost['cost'].tolist()
        
        # Create mapping from cost to average OT cost for legend and coloring
        cost_to_ot = dict(zip(ot_by_cost['cost'], ot_by_cost['avg_ot_cost']))
        legend_prefix = ""
        
        # Create a continuous color scale based on OT costs
        min_ot = min(cost_to_ot.values())
        max_ot = max(cost_to_ot.values())
        norm = plt.Normalize(min_ot, max_ot)
        cmap = plt.cm.viridis  # A good sequential colormap
    
    # Plot each metric
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        
        # Set plot title and labels
        #ax.set_title(config['title'], fontsize=14)
        ax.set_xlabel('Round', fontsize=26)
        ax.set_ylabel(config['title'], fontsize=26)
        
        legend_handles = []
        legend_labels = []
        
        # Sort group values to ensure consistent color mapping
        if x_axis_source == 'avg_ot_cost':
            # Sort by OT cost value for consistent gradient
            group_values = sorted(group_values, key=lambda x: cost_to_ot[x])
        
        # Plot each group
        for idx, group_value in enumerate(group_values):
            # Get data for this group
            group_df = df[df[group_column] == group_value]
            
            # Skip if no data for this group
            if group_df.empty:
                continue
                
            # Group by round_num and compute mean of the metric across runs
            round_data = group_df.groupby('round_num')[config['metric']].mean().reset_index()
            
            # Get x and y data
            x = round_data['round_num'].values
            y = round_data[config['metric']].values
            
            # Skip if no valid data
            if len(y) == 0 or np.all(np.isnan(y)):
                continue
            
            # Apply smoothing
            y_smooth = smooth_curve(y, window_size, poly_order)
            
            # Determine color based on coloring scheme
            if x_axis_source == 'cost':
                # Use categorical colors for cost
                color = plt.cm.tab10(idx % 10) if len(group_values) <= 10 else plt.cm.tab20(idx % 20)
            else:  # x_axis_source == 'avg_ot_cost'
                # Use gradient color based on OT cost value
                ot_value = cost_to_ot[group_value]
                color = cmap(1- norm(ot_value))
            
            # Plot the smoothed line
            line, = ax.plot(x, y_smooth, color=color, linewidth=2)
            
            # Add to legend
            legend_handles.append(line)
            
            # Format legend label based on grouping
            if x_axis_source == 'cost':
                legend_labels.append(f"{legend_prefix}{group_value}")
            else:  # x_axis_source == 'avg_ot_cost'
                # Use the mapped OT cost value instead of original cost
                ot_value = cost_to_ot[group_value]
                legend_labels.append(f"{legend_prefix}{ot_value:.2f}")
            
            # Plot original points with lower alpha
            ax.scatter(x, y, color=color, alpha=0.2)
        
        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, title = 'Cost', title_fontsize=20, fontsize=20, loc='right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=12)
    
    if save_figure:
        import os
        # Create directory if it doesn't exist
        save_dir = f'{RESULTS_DIR}/weight_div_figures'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with high resolution for paper
        filename = f'{dataset_name}_weight_div.pdf'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, 
                   dpi=300,           # High resolution
                   bbox_inches='tight', # Remove extra whitespace
                   format='pdf',       # PDF format for papers
                   facecolor='white',  # White background
                   edgecolor='none')   # No edge color
        print(f"Figure saved to: {filepath}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
    
def plot_ot_method_comparison(
    dataset_name: str,
    ot_results_df: pd.DataFrame,
    costs: List[float],
    baseline_ot_method_name: str,
    variant_ot_method_names: List[str],
    target_client_pair: Optional[str] = None,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    save_figure: Optional[bool] = None,
    RESULTS_DIR: Optional[str] = 'results'
) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Compare mean OT costs between a baseline method and variants across different cost parameters.
    Lines connect points based on x-axis ordering.
    
    Args:
        ot_results_df: DataFrame with OT analysis results
        baseline_ot_method_name: Name of the baseline OT method (x-axis)
        variant_ot_method_names: List of variant OT method names to compare (y-axis)
        target_client_pair: Specific client pair to filter for (optional)
        plot_title: Custom title for the plot
        figsize: Figure size as (width, height)
        
    Returns:
        Tuple of (Figure, Axes) or (None, None) if data is insufficient
    """
    ot_results_df = ot_results_df[ot_results_df['fl_cost_param'].isin(costs)]
    # Input validation
    if ot_results_df.empty:
        return None, None
    
    required_cols = ['ot_method_name', 'ot_cost_value', 'fl_cost_param']
    if not all(col in ot_results_df.columns for col in required_cols):
        return None, None
    
    if baseline_ot_method_name not in ot_results_df['ot_method_name'].unique():
        return None, None
    
    valid_variants = []
    for variant in variant_ot_method_names:
        if variant in ot_results_df['ot_method_name'].unique():
            valid_variants.append(variant)
    
    if not valid_variants:
        return None, None
    
    # Filter for target client pair if specified
    df = ot_results_df.copy()
    if target_client_pair:
        df = df[df['client_pair'] == target_client_pair]
        if df.empty:
            return None, None
    
    # Get list of all cost parameters used in the data
    all_costs = sorted(df['fl_cost_param'].unique())
    if not all_costs:
        return None, None
    
    # Create figure for plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean OT cost for baseline method for each cost parameter
    baseline_means = {}
    for cost in all_costs:
        cost_data = df[(df['ot_method_name'] == baseline_ot_method_name) & 
                       (df['fl_cost_param'] == cost)]
        
        if not cost_data.empty:
            # Calculate mean of valid OT costs
            valid_costs = cost_data['ot_cost_value'].dropna()
            if not valid_costs.empty:
                baseline_means[cost] = valid_costs.mean()
    
    # Skip if no baseline data
    if not baseline_means:
        return None, None
    
    # Define a set of highly distinguishable colors
    distinct_colors = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#2ca02c',  # green
        '#9467bd',  # purple
        '#ff7f0e',  # orange
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#17becf',  # cyan
        '#bcbd22',  # olive
        '#7f7f7f'   # gray
    ]
    
    # For storing all values to determine plot limits
    all_values = []
    
    for i, variant_name in enumerate(valid_variants):
        variant_means = {}
        
        # Calculate mean OT cost for variant method for each cost parameter
        for cost in all_costs:
            cost_data = df[(df['ot_method_name'] == variant_name) & 
                          (df['fl_cost_param'] == cost)]
            
            if not cost_data.empty:
                valid_costs = cost_data['ot_cost_value'].dropna()
                if not valid_costs.empty and cost in baseline_means:
                    variant_means[cost] = valid_costs.mean()
        
        # Skip if no variant data
        if not variant_means:
            continue
        
        # Create lists for plotting - matched by cost parameter
        common_costs = sorted([cost for cost in variant_means if cost in baseline_means])
        if not common_costs:
            continue
            
        x_values = [baseline_means[cost] for cost in common_costs]
        y_values = [variant_means[cost] for cost in common_costs]
        
        # Sort by x-values to ensure line connects points in order of baseline costs
        sorted_pairs = sorted(zip(x_values, y_values), key=lambda pair: pair[0])
        x_values_sorted = [pair[0] for pair in sorted_pairs]
        y_values_sorted = [pair[1] for pair in sorted_pairs]
        
        all_values.extend(x_values_sorted)
        all_values.extend(y_values_sorted)
        
        # Get color for this variant
        color = distinct_colors[i % len(distinct_colors)]
        
        # Plot connecting line based on x-axis ordering
        ax.plot(
            x_values_sorted, y_values_sorted,
            color=color,
            linestyle='-',
            linewidth=1.5,
            alpha=0.7,
            label=variant_name
        )
        
        # Plot scatter points
        ax.scatter(
            x_values_sorted, y_values_sorted,
            color=color,
            marker='o',
            s=80,
            alpha=0.8,
            zorder=5,
            edgecolors='white',
            linewidths=0.5
        )
    
    # Add y=x reference line
    if all_values:
        all_finite = [v for v in all_values if np.isfinite(v)]
        if all_finite:
            min_val = min(all_finite) * 0.9
            max_val = max(all_finite) * 1.1
            
            ax.plot(
                [min_val, max_val], 
                [min_val, max_val], 
                'k--',
                alpha=0.7, 
                linewidth=1.5,
                label='Agreement'
            )
            
            # Set axis limits
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
    
    # Labels and title
    ax.set_xlabel(f"Mean OT Cost: Full", fontsize=26)
    ax.set_ylabel("Mean OT Cost: Subsampled", fontsize=26)
    
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    # Add grid for readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure legend is visible
    ax.legend(fontsize=22, title_fontsize=22, title = 'Sample size', loc='best')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')

    if save_figure:
        import os
        # Create directory if it doesn't exist
        save_dir = f'{RESULTS_DIR}/sensitivity_figures'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with high resolution for paper
        filename = f'{dataset_name}_sample.pdf'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, 
                    dpi=300,           # High resolution
                    bbox_inches='tight', # Remove extra whitespace
                    format='pdf',       # PDF format for papers
                    facecolor='white',  # White background
                    edgecolor='none')   # No edge color
        print(f"Figure saved to: {filepath}")
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return