# --- START OF FILE plotter.py ---

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd

# --- Plotting Libraries ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
    MaxNLocator = None

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    confusion_matrix = None
    ConfusionMatrixDisplay = None
    auc = None

try:
    import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    tabulate = None

# Assuming logger_utils is accessible
try:
    from .logger_utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)


# --- Helper Functions ---
# _check_plotting_libs, _create_output_dir, _save_show_plot remain unchanged
def _check_plotting_libs() -> bool:
    libs_ok = True
    if not MATPLOTLIB_AVAILABLE: logger.error("Matplotlib is required for plotting. Please install it (`pip install matplotlib seaborn`)."); libs_ok = False
    if not SKLEARN_METRICS_AVAILABLE: logger.warning("Scikit-learn metrics (for Confusion Matrix, AUC) not found. Some plots/metrics may be skipped.")
    if not TABULATE_AVAILABLE: logger.warning("Tabulate library not found. Metrics tables will not be generated.")
    if sns is None: logger.warning("Seaborn library not found. Some plot aesthetics might be affected.")
    return libs_ok

def _create_output_dir(base_path: Path) -> Optional[Path]:
    try: base_path = Path(base_path); plot_dir = base_path.parent / f"{base_path.stem}_plots"; plot_dir.mkdir(parents=True, exist_ok=True); return plot_dir
    except Exception as e: logger.error(f"Failed to create plot directory near {base_path}: {e}"); return None

def _save_show_plot(fig, output_path: Path, show_plots: bool):
    if not MATPLOTLIB_AVAILABLE: return
    try:
        fig.tight_layout(pad=1.5); fig.savefig(output_path, dpi=300, bbox_inches='tight'); logger.info(f"Plot saved to: {output_path}")
        if show_plots: plt.show()
        plt.close(fig)
    except Exception as e: logger.error(f"Failed to save/show plot {output_path}: {e}");
    if fig: plt.close(fig)


# --- Results Plotter Class ---

class ResultsPlotter:
    """
    Provides static methods to plot results stored in JSON files generated
    by the ClassificationPipeline.
    """

    @staticmethod
    def _load_results_data_and_setup_plot_dir(json_path: Union[str, Path]) -> Tuple[Optional[Dict], Optional[Path], Optional[str]]:
        """Loads JSON data and sets up plot directory. Returns (data, plot_dir, run_id) or (None, None, None)."""
        # ... (Implementation unchanged) ...
        json_path = Path(json_path)
        if not json_path.is_file(): logger.error(f"Results file not found: {json_path}"); return None, None, None
        if not _check_plotting_libs(): return None, None, None
        plot_dir = _create_output_dir(json_path)
        if not plot_dir: return None, None, None
        try:
            with open(json_path, 'r', encoding='utf-8') as f: results_data = json.load(f)
            run_id = results_data.get('run_id', json_path.parent.name)
            return results_data, plot_dir, run_id
        except Exception as e: logger.error(f"Failed to load or parse JSON file {json_path}: {e}"); return None, None, None

    # --- Plotting Helper Methods ---

    @staticmethod
    def _plot_learning_curves(history: List[Dict[str, Any]], title: str, output_path: Path, show_plots: bool, ax_loss=None, ax_acc=None):
        """ Plots learning curves, potentially on existing axes. Returns True if plotted."""
        # ... (Implementation unchanged - uses ax_loss, ax_acc if provided) ...
        if not history or not MATPLOTLIB_AVAILABLE: return False
        try:
            df = pd.DataFrame(history)
            if 'epoch' not in df.columns: logger.warning(f"Cannot plot learning curves for '{title}': 'epoch' column missing."); return False
            standalone_plot = ax_loss is None or ax_acc is None
            if standalone_plot: fig, axes = plt.subplots(1, 2, figsize=(14, 5)); fig.suptitle(title, fontsize=14); ax_loss_local, ax_acc_local = axes
            else: ax_loss_local, ax_acc_local = ax_loss, ax_acc; fig = None
            plot_occurred_loss = False; plot_occurred_acc = False
            has_train_loss = 'train_loss' in df.columns and df['train_loss'].notna().any(); has_valid_loss = 'valid_loss' in df.columns and df['valid_loss'].notna().any()
            if has_train_loss: ax_loss_local.plot(df['epoch'], df['train_loss'], marker='o', markersize=3, linestyle='-', label='Train Loss')
            if has_valid_loss: ax_loss_local.plot(df['epoch'], df['valid_loss'], marker='x', markersize=4, linestyle='--', label='Valid Loss')
            if has_train_loss or has_valid_loss:
                plot_occurred_loss = True; ax_loss_local.set_xlabel('Epoch'); ax_loss_local.set_ylabel('Loss'); ax_loss_local.set_title('Loss vs. Epoch')
                if MaxNLocator: ax_loss_local.xaxis.set_major_locator(MaxNLocator(integer=True));
                if standalone_plot: ax_loss_local.legend(); ax_loss_local.grid(True, linestyle='--', alpha=0.6)
            else: ax_loss_local.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=ax_loss_local.transAxes); ax_loss_local.set_title('Loss Data Missing')
            has_train_acc = 'train_acc' in df.columns and df['train_acc'].notna().any(); has_valid_acc = 'valid_acc' in df.columns and df['valid_acc'].notna().any()
            if has_train_acc: ax_acc_local.plot(df['epoch'], df['train_acc'], marker='o', markersize=3, linestyle='-', label='Train Acc')
            if has_valid_acc: ax_acc_local.plot(df['epoch'], df['valid_acc'], marker='x', markersize=4, linestyle='--', label='Valid Acc')
            if has_train_acc or has_valid_acc:
                plot_occurred_acc = True; ax_acc_local.set_xlabel('Epoch'); ax_acc_local.set_ylabel('Accuracy')
                try: ymin, ymax = ax_acc_local.get_ylim(); ax_acc_local.set_ylim(bottom=max(0, ymin), top=min(1.05, ymax))
                except: ax_acc_local.set_ylim(bottom=0, top=1.05)
                ax_acc_local.set_title('Accuracy vs. Epoch');
                if MaxNLocator: ax_acc_local.xaxis.set_major_locator(MaxNLocator(integer=True));
                if standalone_plot: ax_acc_local.legend(); ax_acc_local.grid(True, linestyle='--', alpha=0.6)
            else: ax_acc_local.text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', transform=ax_acc_local.transAxes); ax_acc_local.set_title('Accuracy Data Missing')
            plot_occurred = plot_occurred_loss or plot_occurred_acc
            if standalone_plot and fig is not None:
                if plot_occurred: _save_show_plot(fig, output_path, show_plots)
                else: plt.close(fig)
            return plot_occurred
        except Exception as e: logger.error(f"Error plotting learning curves for '{title}': {e}", exc_info=True)
        if standalone_plot and 'fig' in locals() and fig is not None: plt.close(fig); return False

    @staticmethod
    def _generate_metrics_table(metrics_data: Dict[str, Any], output_path: Optional[Path] = None) -> Optional[str]:
        """Generates a formatted table of per-class and macro metrics."""
        # ... (Implementation remains the same) ...
        if not TABULATE_AVAILABLE: return None
        per_class_metrics = metrics_data.get('per_class'); macro_metrics = metrics_data.get('macro_avg'); overall_acc = metrics_data.get('overall_accuracy')
        if not per_class_metrics or not macro_metrics: logger.warning("Cannot generate metrics table: 'per_class' or 'macro_avg' data missing."); return None
        headers = ["Metric"] + list(per_class_metrics.keys()) + ["Macro Avg"]; table_data = []
        metric_keys = ['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc']
        for key in metric_keys:
            row = [key.replace('_', ' ').title()]
            for class_name in per_class_metrics.keys(): class_val = per_class_metrics[class_name].get(key, np.nan); row.append(f"{class_val:.4f}" if not np.isnan(class_val) else "N/A")
            macro_val = macro_metrics.get(key, np.nan); row.append(f"{macro_val:.4f}" if not np.isnan(macro_val) else "N/A"); table_data.append(row)
        acc_row = ["Overall Acc"] + ["-"] * len(per_class_metrics) + [f"{overall_acc:.4f}" if overall_acc is not None else "N/A"]; table_data.append(acc_row)
        try:
            table_str = tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f", stralign="center")
            if output_path: output_path.write_text(table_str, encoding='utf-8'); logger.info(f"Metrics table saved to: {output_path}")
            return table_str
        except Exception as e: logger.error(f"Error generating metrics table: {e}"); return None


    @staticmethod
    def _plot_confusion_matrix(y_true: List, y_pred: List, classes: List[str], title: str, output_path: Path, show_plots: bool, ax=None):
        """Plots a confusion matrix, potentially on a given axis. Returns True if plotted."""
        # ... (Implementation remains the same) ...
        if not y_true or not y_pred or not classes or not SKLEARN_METRICS_AVAILABLE or not MATPLOTLIB_AVAILABLE: logger.warning(f"Cannot plot confusion matrix for '{title}': Missing data or libraries."); return False
        standalone_plot = ax is None
        try:
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes))); disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            if standalone_plot: fig, ax_local = plt.subplots(figsize=(max(7, len(classes)*0.7), max(5, len(classes)*0.6)))
            else: ax_local = ax; fig = ax_local.figure
            disp.plot(cmap=plt.cm.Blues, ax=ax_local, xticks_rotation=45, colorbar=standalone_plot); ax_local.set_title(title, fontsize=10 if not standalone_plot else 12)
            if standalone_plot: _save_show_plot(fig, output_path, show_plots)
            return True
        except Exception as e: logger.error(f"Error plotting confusion matrix for '{title}': {e}", exc_info=True);
        if standalone_plot and 'fig' in locals(): plt.close(fig); return False


    @staticmethod
    def _plot_roc_curves(roc_data: Dict[str, Dict[str, List]], title_prefix: str, output_path: Path, show_plots: bool, ax=None):
        """Plots ROC curves (per-class) for a single eval or fold, potentially on a given axis. Returns True if plotted."""
        # ... (Implementation remains the same) ...
        if not roc_data or not MATPLOTLIB_AVAILABLE: return False
        standalone_plot = ax is None
        try:
            if standalone_plot: fig, ax_local = plt.subplots(figsize=(8, 7))
            else: ax_local = ax; fig = ax_local.figure
            ax_local.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)' if standalone_plot else None, alpha=0.7); num_classes = len(roc_data)
            try: colors = plt.cm.get_cmap('tab20', num_classes)
            except ValueError: colors = plt.cm.get_cmap('tab10', num_classes)
            plot_occurred = False
            for i, (class_name, curve_points) in enumerate(roc_data.items()):
                 fpr = curve_points.get('fpr'); tpr = curve_points.get('tpr'); roc_auc_val = curve_points.get('roc_auc', np.nan)
                 if fpr and tpr and isinstance(fpr, list) and isinstance(tpr, list):
                     plot_occurred = True;
                     if np.isnan(roc_auc_val) and auc and len(fpr) > 1 and len(tpr) > 1:
                          try: roc_auc_val = auc(fpr, tpr)
                          except: roc_auc_val = np.nan
                     label = f'{class_name} (AUC={roc_auc_val:.2f})' if not np.isnan(roc_auc_val) else class_name
                     ax_local.plot(fpr, tpr, color=colors(i % colors.N), lw=1.5, label=label, alpha=0.8)
            if not plot_occurred:
                 if standalone_plot: plt.close(fig); logger.warning(f"No valid ROC curve data found for {title_prefix}"); return False
                 else: ax_local.text(0.5, 0.5, 'No ROC Data', ha='center', va='center', transform=ax_local.transAxes); return False # Indicate nothing plotted
            ax_local.set_xlabel('False Positive Rate'); ax_local.set_ylabel('True Positive Rate (Recall)'); ax_local.set_title(f'{title_prefix} - ROC Curves'); ax_local.legend(loc='lower right', fontsize='small'); ax_local.grid(True, linestyle='--', alpha=0.6); ax_local.set_xlim([-0.05, 1.05]); ax_local.set_ylim([-0.05, 1.05])
            if standalone_plot: _save_show_plot(fig, output_path, show_plots)
            return True
        except Exception as e: logger.error(f"Error plotting ROC curves for '{title_prefix}': {e}", exc_info=True);
        if standalone_plot and 'fig' in locals(): plt.close(fig); return False

    @staticmethod
    def _plot_pr_curves(pr_data: Dict[str, Dict[str, List]], title_prefix: str, output_path: Path, show_plots: bool, ax=None):
        """Plots Precision-Recall curves (per-class) for a single eval or fold, potentially on a given axis. Returns True if plotted."""
        # ... (Implementation remains the same) ...
        if not pr_data or not MATPLOTLIB_AVAILABLE: return False
        standalone_plot = ax is None
        try:
            if standalone_plot: fig, ax_local = plt.subplots(figsize=(8, 7))
            else: ax_local = ax; fig = ax_local.figure
            num_classes = len(pr_data);
            try: colors = plt.cm.get_cmap('tab20', num_classes)
            except ValueError: colors = plt.cm.get_cmap('tab10', num_classes)
            plot_occurred = False
            for i, (class_name, curve_points) in enumerate(pr_data.items()):
                 precision = curve_points.get('precision'); recall = curve_points.get('recall'); pr_auc_val = curve_points.get('pr_auc', np.nan)
                 if precision and recall and isinstance(precision, list) and isinstance(recall, list):
                     plot_occurred = True;
                     if np.isnan(pr_auc_val) and auc and len(recall) > 1 and len(precision) > 1:
                          try: order = np.argsort(recall); pr_auc_val = auc(np.array(recall)[order], np.array(precision)[order])
                          except: pr_auc_val = np.nan
                     label = f'{class_name} (AUPRC={pr_auc_val:.2f})' if not np.isnan(pr_auc_val) else class_name
                     ax_local.plot(recall, precision, color=colors(i % colors.N), lw=1.5, label=label, alpha=0.8)
            if not plot_occurred:
                 if standalone_plot: plt.close(fig); logger.warning(f"No valid PR curve data found for {title_prefix}"); return False
                 else: ax_local.text(0.5, 0.5, 'No PR Data', ha='center', va='center', transform=ax_local.transAxes); return False # Indicate nothing plotted
            ax_local.set_xlabel('Recall (True Positive Rate)'); ax_local.set_ylabel('Precision'); ax_local.set_title(f'{title_prefix} - Precision-Recall Curves'); ax_local.legend(loc='lower left', fontsize='small'); ax_local.grid(True, linestyle='--', alpha=0.6); ax_local.set_xlim([-0.05, 1.05]); ax_local.set_ylim([-0.05, 1.05])
            if standalone_plot: _save_show_plot(fig, output_path, show_plots)
            return True
        except Exception as e: logger.error(f"Error plotting PR curves for '{title_prefix}': {e}", exc_info=True);
        if standalone_plot and 'fig' in locals(): plt.close(fig); return False

    @staticmethod
    def _plot_cv_aggregated_metrics(aggregated_metrics: Dict[str, Dict[str, float]], title: str, output_path: Path, show_plots: bool):
        """Plots aggregated CV metrics with confidence intervals."""
        # ... (Implementation remains the same) ...
        if not aggregated_metrics or not MATPLOTLIB_AVAILABLE: return
        metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']; plot_data = {'metric': [], 'mean': [], 'ci_margin': [], 'ci_lower': [], 'ci_upper': []}
        for key in metrics_to_plot:
            metric_stats = aggregated_metrics.get(key);
            if metric_stats and isinstance(metric_stats, dict): mean = metric_stats.get('mean', np.nan)
            if not np.isnan(mean): plot_data['metric'].append(key.replace('_', ' ').title()); plot_data['mean'].append(mean); margin = metric_stats.get('margin_of_error', 0.0); margin = 0.0 if margin is None or np.isnan(margin) else margin; plot_data['ci_margin'].append(margin); plot_data['ci_lower'].append(metric_stats.get('ci_lower', mean - margin)); plot_data['ci_upper'].append(metric_stats.get('ci_upper', mean + margin))
        if not plot_data['metric']: logger.warning(f"No valid aggregated metrics found to plot for '{title}'."); return
        df = pd.DataFrame(plot_data)
        try:
            fig, ax = plt.subplots(figsize=(10, max(6, len(df['metric']) * 0.6))); y_pos = np.arange(len(df['metric'])); error = df['ci_margin']
            ax.barh(y_pos, df['mean'], xerr=error, align='center', alpha=0.7, ecolor='black', capsize=5); ax.set_yticks(y_pos); ax.set_yticklabels(df['metric']); ax.invert_yaxis()
            ax.set_xlabel('Score'); ax.set_title(title); ax.set_xlim(left=max(0, df['ci_lower'].min() - 0.05) if df['ci_lower'].notna().any() else 0, right=min(1.05, df['ci_upper'].max() + 0.05) if df['ci_upper'].notna().any() else 1.05); ax.grid(True, axis='x', linestyle='--', alpha=0.6)
            for i, v in enumerate(df['mean']): ax.text(v + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), i, f"{v:.3f}", va='center', color='black', fontsize=9)
            _save_show_plot(fig, output_path, show_plots)
        except Exception as e: logger.error(f"Error plotting CV aggregated metrics for '{title}': {e}", exc_info=True)


    @staticmethod
    def _generate_grid_search_table(cv_results: Dict[str, Any], output_path: Path, top_n: int = 15):
        """Generates a table summary of GridSearchCV results."""
        # ... (Implementation remains the same) ...
        if not TABULATE_AVAILABLE or not cv_results: return None
        if not isinstance(cv_results, dict): logger.warning("Cannot generate grid search table: cv_results is not a dictionary."); return None
        try:
            df = pd.DataFrame(cv_results); param_cols = [col for col in df.columns if col.startswith('param_')]; score_cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']; time_cols = ['mean_fit_time', 'mean_score_time']; cols_to_show = score_cols + time_cols + param_cols; cols_to_show = [col for col in cols_to_show if col in df.columns]
            if not cols_to_show: logger.warning("No relevant columns found in cv_results for grid search table."); return None
            if 'rank_test_score' not in df.columns: logger.warning("Cannot rank grid search results: 'rank_test_score' missing. Showing first rows."); valid_cols = [c for c in cols_to_show if c in df.columns]; df_top = df[valid_cols].head(top_n) if valid_cols else pd.DataFrame()
            else: valid_cols = [c for c in cols_to_show if c in df.columns]; df_top = df.nsmallest(top_n, 'rank_test_score')[valid_cols] if valid_cols else pd.DataFrame()
            if df_top.empty: logger.warning("Grid search DataFrame is empty after filtering columns."); return None
            df_top.columns = [col.replace('param_', '') if col.startswith('param_') else col for col in df_top.columns]; table_str = tabulate.tabulate(df_top, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".4f")
            output_path.write_text(table_str, encoding='utf-8'); logger.info(f"Grid search summary table saved to: {output_path}")
            return table_str
        except Exception as e: logger.error(f"Error generating grid search table: {e}"); return None


    @staticmethod
    def _generate_nested_cv_param_table(best_params_list: List[Optional[Dict]], best_scores_list: List[Optional[float]], output_path: Path):
        """ Generates table for best params and scores per outer fold of nested CV."""
        # ... (Implementation remains the same) ...
        if not TABULATE_AVAILABLE or not best_params_list or not best_scores_list: return None
        if len(best_params_list) != len(best_scores_list): logger.warning("Mismatch between best_params and best_scores list lengths for nested CV table."); return None
        try:
            table_data = []; all_param_keys = set()
            for params in best_params_list:
                if isinstance(params, dict): all_param_keys.update(params.keys())
            sorted_param_keys = sorted([key.replace('module__', '') for key in all_param_keys])
            headers = ["Outer Fold", "Inner Best Score"] + sorted_param_keys
            for i, (params, score) in enumerate(zip(best_params_list, best_scores_list)):
                row = [f"Fold {i+1}"]; row.append(f"{score:.4f}" if score is not None and not np.isnan(score) else "N/A")
                if isinstance(params, dict):
                    for short_key in sorted_param_keys: original_key = 'module__' + short_key if 'module__' + short_key in params else short_key; row.append(params.get(original_key, "N/A"))
                else: row.extend(["N/A"] * len(sorted_param_keys))
                table_data.append(row)
            table_str = tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f"); output_path.write_text(table_str, encoding='utf-8'); logger.info(f"Nested CV best params per fold table saved to: {output_path}")
            return table_str
        except Exception as e: logger.error(f"Error generating nested CV params table: {e}"); return None


    @staticmethod
    def _plot_macro_roc_point(macro_metrics: Dict[str, float], title: str, output_path: Path, show_plots: bool):
        """ Plots Macro Avg ROC point connected to corners. """
        # ... (Implementation remains the same) ...
        if not macro_metrics or not MATPLOTLIB_AVAILABLE: return
        try:
            recall = macro_metrics.get('recall', np.nan); specificity = macro_metrics.get('specificity', np.nan)
            if np.isnan(recall) or np.isnan(specificity): logger.warning(f"Cannot plot macro ROC point for '{title}': Missing recall or specificity."); return
            fpr_macro = 1.0 - specificity; tpr_macro = recall; fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Line'); ax.scatter(fpr_macro, tpr_macro, marker='o', color='red', s=100, label=f'Macro Avg.\n(FPR={fpr_macro:.3f}, TPR={tpr_macro:.3f})', zorder=3)
            ax.plot([0, fpr_macro], [0, tpr_macro], color='red', linestyle=':', linewidth=1, alpha=0.6); ax.plot([fpr_macro, 1], [tpr_macro, 1], color='red', linestyle=':', linewidth=1, alpha=0.6)
            ax.set_xlabel('Macro Average False Positive Rate (1 - Specificity)'); ax.set_ylabel('Macro Average True Positive Rate (Recall)'); ax.set_title(title); ax.legend(loc='lower right', fontsize='small'); ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim([-0.05, 1.05]); ax.set_ylim([-0.05, 1.05])
            _save_show_plot(fig, output_path, show_plots)
        except Exception as e: logger.error(f"Error plotting macro ROC point for '{title}': {e}", exc_info=True)


    @staticmethod
    def _plot_cv_macro_roc_points(fold_details: List[Dict], title: str, output_path: Path, show_plots: bool):
        """ Plots Macro Avg ROC points per fold and connects mean point to corners."""
        if not fold_details or not MATPLOTLIB_AVAILABLE: return
        try:
            fpr_points = []; tpr_points = []; fold_indices = []
            for i, fold_data in enumerate(fold_details):
                macro_metrics = None # Initialize here
                if isinstance(fold_data, dict):
                    macro_metrics = fold_data.get('macro_avg')
                if isinstance(macro_metrics, dict): # Check if macro_metrics is a dict
                    recall = macro_metrics.get('recall', np.nan)
                    specificity = macro_metrics.get('specificity', np.nan)
                    if not np.isnan(recall) and not np.isnan(specificity):
                        fpr_points.append(1.0 - specificity)
                        tpr_points.append(recall)
                        fold_indices.append(i + 1)

            if not fpr_points:
                logger.warning(f"No valid macro ROC points found across CV folds for '{title}'.")
                return

            fig, ax = plt.subplots(figsize=(7, 6.5))
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Line')
            scatter = ax.scatter(fpr_points, tpr_points, marker='o', c=fold_indices, cmap='viridis', s=60, label='Fold Macro Avg.', zorder=3, alpha=0.7)
            mean_fpr = np.mean(fpr_points); mean_tpr = np.mean(tpr_points)
            ax.scatter(mean_fpr, mean_tpr, marker='^', color='red', s=120, label=f'Mean of Folds\n(FPR={mean_fpr:.3f}, TPR={mean_tpr:.3f})', zorder=4)
            ax.plot([0, mean_fpr], [0, mean_tpr], color='red', linestyle=':', linewidth=1, alpha=0.8)
            ax.plot([mean_fpr, 1], [mean_tpr, 1], color='red', linestyle=':', linewidth=1, alpha=0.8)
            ax.set_xlabel('Macro Average False Positive Rate (1 - Specificity)'); ax.set_ylabel('Macro Average True Positive Rate (Recall)'); ax.set_title(title)
            if len(fold_indices) > 1: cbar = fig.colorbar(scatter, ax=ax, ticks=np.linspace(min(fold_indices), max(fold_indices), min(len(fold_indices), 10), dtype=int)); cbar.set_label('Fold Number')
            ax.legend(loc='lower right', fontsize='small'); ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim([-0.05, 1.05]); ax.set_ylim([-0.05, 1.05])
            _save_show_plot(fig, output_path, show_plots)
        except Exception as e:
            logger.error(f"Error plotting CV macro ROC points for '{title}': {e}", exc_info=True)


    # --- Main Public Plotting Methods ---

    @staticmethod
    def plot_single_train_results(json_path: Union[str, Path], show_plots: bool = False):
        """Plots results from a 'single_train' JSON file."""
        results_data, plot_dir, run_id = ResultsPlotter._load_results_data_and_setup_plot_dir(json_path)
        if not results_data or not plot_dir: return
        try:
            # ... (Implementation unchanged) ...
            logger.info(f"Plotting results for single_train: {run_id}"); history = results_data.get('training_history')
            if history: ResultsPlotter._plot_learning_curves(history, f"Single Train Learning Curves ({run_id})", plot_dir / "learning_curves.png", show_plots)
            else: logger.warning("No 'training_history' found for learning curve plot.")
            logger.info(f"Finished plotting for single_train: {run_id}")
        except Exception as e: logger.error(f"Failed to plot single_train results from {json_path}: {e}", exc_info=True)


    @staticmethod
    def plot_single_eval_results(json_path: Union[str, Path], class_names: List[str], show_plots: bool = False):
        """Plots results from a 'single_eval' JSON file."""
        results_data, plot_dir, run_id = ResultsPlotter._load_results_data_and_setup_plot_dir(json_path)
        if not results_data or not plot_dir: return
        try:
            # ... (Implementation unchanged - calls _plot_macro_roc_point, removed PR call) ...
            logger.info(f"Plotting results for single_eval: {run_id}"); title_prefix = f"Single Eval ({run_id})"
            if TABULATE_AVAILABLE: ResultsPlotter._generate_metrics_table(results_data, output_path=plot_dir / "metrics_table.txt")
            macro_metrics = results_data.get('macro_avg')
            if isinstance(macro_metrics, dict): ResultsPlotter._plot_macro_roc_point(macro_metrics, f"{title_prefix}\nMacro Average ROC Point", plot_dir / "macro_roc_point.png", show_plots)
            detailed_data = results_data.get('detailed_data')
            if not detailed_data: logger.warning(f"Skipping detailed plots for '{title_prefix}': 'detailed_data' missing or incomplete."); logger.info(f"Finished plotting basic info for single_eval: {run_id}"); return
            y_true = detailed_data.get('y_true'); y_pred = detailed_data.get('y_pred')
            if y_true and y_pred and class_names and SKLEARN_METRICS_AVAILABLE: ResultsPlotter._plot_confusion_matrix(y_true, y_pred, class_names, f"{title_prefix}\nConfusion Matrix", plot_dir / "confusion_matrix.png", show_plots)
            roc_curve_points = detailed_data.get('roc_curve_points');
            if roc_curve_points: ResultsPlotter._plot_roc_curves(roc_curve_points, title_prefix, plot_dir / "roc_curves.png", show_plots)
            pr_curve_points = detailed_data.get('pr_curve_points');
            if pr_curve_points: ResultsPlotter._plot_pr_curves(pr_curve_points, title_prefix, plot_dir / "pr_curves.png", show_plots)
            logger.info(f"Finished plotting for single_eval: {run_id}")
        except Exception as e: logger.error(f"Failed to plot single_eval results from {json_path}: {e}", exc_info=True)

    @staticmethod
    def plot_non_nested_cv_results(json_path: Union[str, Path], show_plots: bool = False):
        """Plots results from a 'non_nested_grid_search' JSON file."""
        # ... (Implementation unchanged) ...
        results_data, plot_dir, run_id = ResultsPlotter._load_results_data_and_setup_plot_dir(json_path)
        if not results_data or not plot_dir: return
        try:
            logger.info(f"Plotting results for non_nested_cv: {run_id}"); title_prefix = f"Non-Nested Search ({run_id})"
            history = results_data.get('best_refit_model_history');
            if history: ResultsPlotter._plot_learning_curves(history, f"{title_prefix}\nLearning Curves (Best Refit Model)", plot_dir / "best_refit_learning_curves.png", show_plots)
            else: logger.warning("No 'best_refit_model_history' found for learning curve plot.")
            cv_results = results_data.get('cv_results');
            if cv_results and TABULATE_AVAILABLE: ResultsPlotter._generate_grid_search_table(cv_results, plot_dir / "grid_search_summary.txt", top_n=20)
            elif not cv_results: logger.warning("No 'cv_results' found for grid search table.")
            logger.info(f"Finished plotting for non_nested_cv: {run_id}")
        except Exception as e: logger.error(f"Failed to plot non_nested_cv results from {json_path}: {e}", exc_info=True)

    @staticmethod
    def plot_nested_cv_results(json_path: Union[str, Path], show_plots: bool = False):
        """Plots results from a 'nested_grid_search' JSON file."""
        # ... (Implementation unchanged) ...
        results_data, plot_dir, run_id = ResultsPlotter._load_results_data_and_setup_plot_dir(json_path)
        if not results_data or not plot_dir: return
        try:
            logger.info(f"Plotting results for nested_cv: {run_id}"); title_prefix = f"Nested CV ({run_id})"
            outer_scores = results_data.get('outer_cv_scores')
            if outer_scores and isinstance(outer_scores, dict) and sns:
                 metrics_to_plot = [k for k in outer_scores.keys() if k.startswith('test_')]; valid_metrics_data = {m: np.array(outer_scores[m]).astype(float) for m in metrics_to_plot}; valid_metrics_data = {m: v[~np.isnan(v)] for m, v in valid_metrics_data.items() if len(v[~np.isnan(v)]) > 0}; valid_metrics_to_plot = list(valid_metrics_data.keys())
                 if valid_metrics_to_plot:
                     num_metrics = len(valid_metrics_to_plot); fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5), sharey=True);
                     if num_metrics == 1: axes = [axes]
                     fig.suptitle(f"{title_prefix} - Outer Fold Score Distributions", fontsize=16)
                     for ax, metric in zip(axes, valid_metrics_to_plot): metric_data = valid_metrics_data[metric]; sns.boxplot(y=metric_data, ax=ax, showmeans=True); sns.stripplot(y=metric_data, ax=ax, color=".25", size=4); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.set_title(metric.replace('test_', '').replace('_', ' ').title()); ax.grid(True, axis='y', linestyle='--', alpha=0.6)
                     _save_show_plot(fig, plot_dir / "outer_fold_score_distributions.png", show_plots)
                 else: logger.warning(f"No valid outer fold scores found to plot for {title_prefix}.")
            fold_histories = results_data.get('outer_fold_best_model_histories')
            if fold_histories and isinstance(fold_histories, list):
                 logger.info(f"Plotting learning curves for {len(fold_histories)} outer folds...")
                 for i, history in enumerate(fold_histories):
                     if history: ResultsPlotter._plot_learning_curves(history, f"{title_prefix}\nLearning Curves (Outer Fold {i+1} Best Model)", plot_dir / f"outer_fold_{i+1}_learning_curves.png", show_plots)
                     else: logger.warning(f"Skipping learning curve plot for outer fold {i+1}: No history found.")
            best_params_per_fold = results_data.get('outer_fold_best_params_found'); inner_scores_per_fold = results_data.get('outer_fold_inner_cv_best_score')
            if best_params_per_fold and inner_scores_per_fold and TABULATE_AVAILABLE: ResultsPlotter._generate_nested_cv_param_table(best_params_list=best_params_per_fold, best_scores_list=inner_scores_per_fold, output_path=plot_dir / "nested_cv_best_params_per_fold.txt")
            else: logger.warning("Could not generate nested CV best params table: Data missing.")
            logger.info(f"Finished plotting for nested_cv: {run_id}")
        except Exception as e: logger.error(f"Failed to plot nested_cv results from {json_path}: {e}", exc_info=True)


    @staticmethod
    def plot_cv_model_evaluation_results(json_path: Union[str, Path], class_names: List[str], show_plots: bool = False):
        """Plots results from a 'cv_model_evaluation' JSON file, grouping plots by fold."""
        results_data, plot_dir, run_id = ResultsPlotter._load_results_data_and_setup_plot_dir(json_path)
        if not results_data or not plot_dir: return
        try:
            evaluated_on = results_data.get('evaluated_on', 'unknown')
            n_folds_processed = results_data.get('n_folds_processed', 0)
            conf_level = results_data.get('confidence_level', 0.95)
            title_prefix = f"CV Model Eval ({evaluated_on}, {n_folds_processed} Folds, {run_id})"
            logger.info(f"Plotting results for cv_model_evaluation: {run_id}")

            # --- Plots summarizing across all folds ---
            aggregated_metrics = results_data.get('aggregated_metrics')
            if aggregated_metrics: ResultsPlotter._plot_cv_aggregated_metrics(aggregated_metrics, f"{title_prefix}\nAggregated Scores ({conf_level*100:.0f}% CI)", plot_dir / "aggregated_metrics_ci.png", show_plots)
            fold_scores = results_data.get('cv_fold_scores')
            if fold_scores and isinstance(fold_scores, dict) and sns:
                 metrics_to_plot = [k for k in fold_scores.keys() if k != 'error']; valid_metrics_data = {m: np.array(fold_scores[m]).astype(float) for m in metrics_to_plot}; valid_metrics_data = {m: v[~np.isnan(v)] for m, v in valid_metrics_data.items() if len(v[~np.isnan(v)]) > 0}; valid_metrics_to_plot = list(valid_metrics_data.keys())
                 if valid_metrics_to_plot:
                     num_metrics = len(valid_metrics_to_plot); n_cols = min(num_metrics, 3); n_rows = (num_metrics + n_cols - 1) // n_cols; fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False); axes_flat = axes.flatten(); fig.suptitle(f"{title_prefix} - Fold Score Distributions", fontsize=16)
                     plot_idx = 0;
                     for metric in valid_metrics_to_plot:
                         if plot_idx < len(axes_flat): ax = axes_flat[plot_idx]; metric_data = valid_metrics_data[metric]; sns.boxplot(y=metric_data, ax=ax, showmeans=True); sns.stripplot(y=metric_data, ax=ax, color=".25", size=4); ax.set_ylabel("Score"); ax.set_xlabel(""); ax.set_title(metric.replace('_', ' ').title()); ax.grid(True, axis='y', linestyle='--', alpha=0.6); plot_idx += 1
                     for j in range(plot_idx, len(axes_flat)): axes_flat[j].set_visible(False)
                     _save_show_plot(fig, plot_dir / "fold_score_distributions.png", show_plots)
                 else: logger.warning(f"No valid fold scores found to plot for {title_prefix}.")

            # --- Detailed fold results section ---
            fold_details = results_data.get('fold_detailed_results')
            fold_histories = results_data.get('fold_training_histories')

            if not fold_details or not isinstance(fold_details, list):
                logger.warning(f"Skipping detailed fold plots for '{title_prefix}': 'fold_detailed_results' missing or invalid.")
            else:
                n_folds_actual = len(fold_details)
                if n_folds_actual == 0: logger.warning("No fold details found to plot."); return

                logger.info(f"Generating grouped plots for {n_folds_actual} folds...")

                # --- Grouped Learning Curves (2 plots per row) ---
                if fold_histories and isinstance(fold_histories, list) and len(fold_histories) == n_folds_actual and any(h for h in fold_histories if h):
                    n_cols_lc = 2 # Fixed: 2 columns (Loss, Accuracy)
                    n_rows_lc = n_folds_actual # One row per fold
                    fig_lc, axes_lc = plt.subplots(n_rows_lc, n_cols_lc, figsize=(12, 4 * n_rows_lc), sharex=True, squeeze=False)
                    fig_lc.suptitle(f"{title_prefix} - Learning Curves per Fold", fontsize=16)
                    plotted_any_lc = False
                    for fold_idx, history in enumerate(fold_histories):
                        if fold_idx >= n_rows_lc: break
                        if history:
                            ax_loss = axes_lc[fold_idx, 0]
                            ax_acc = axes_lc[fold_idx, 1]
                            plot_success = ResultsPlotter._plot_learning_curves(history, f"Fold {fold_idx+1}", plot_dir / "dummy.png", False, ax_loss=ax_loss, ax_acc=ax_acc)
                            if plot_success: plotted_any_lc = True
                            # Set shared X label only on last row
                            if fold_idx == n_rows_lc - 1:
                                ax_loss.set_xlabel('Epoch')
                                ax_acc.set_xlabel('Epoch')
                            else:
                                ax_loss.set_xlabel('')
                                ax_acc.set_xlabel('')
                        else:
                            logger.warning(f"Skipping learning curves for Fold {fold_idx+1}: History data missing.")
                            if fold_idx < n_rows_lc:
                                axes_lc[fold_idx, 0].text(0.5, 0.5, 'No History', ha='center', va='center', transform=axes_lc[fold_idx, 0].transAxes); axes_lc[fold_idx, 0].set_title(f'Fold {fold_idx+1} - Loss Missing')
                                axes_lc[fold_idx, 1].text(0.5, 0.5, 'No History', ha='center', va='center', transform=axes_lc[fold_idx, 1].transAxes); axes_lc[fold_idx, 1].set_title(f'Fold {fold_idx+1} - Acc Missing')
                    if plotted_any_lc:
                        plt.subplots_adjust(hspace=0.4, wspace=0.2) # Adjust spacing
                        _save_show_plot(fig_lc, plot_dir / "learning_curves_all_folds.png", show_plots)
                    else: plt.close(fig_lc)

                # --- Grouped Confusion Matrices (Max 2 per row) ---
                if SKLEARN_METRICS_AVAILABLE:
                    n_cols_cm = 2 # Max 2 CMs per row
                    n_rows_cm = (n_folds_actual + n_cols_cm - 1) // n_cols_cm
                    fig_cm, axes_cm = plt.subplots(n_rows_cm, n_cols_cm, figsize=(6 * n_cols_cm, 5 * n_rows_cm + 1), squeeze=False)
                    fig_cm.suptitle(f"{title_prefix} - Confusion Matrices per Fold", fontsize=16)
                    axes_cm_flat = axes_cm.flatten()
                    plot_idx_cm = 0
                    for fold_idx, fold_data in enumerate(fold_details):
                        if plot_idx_cm >= len(axes_cm_flat): break
                        ax = axes_cm_flat[plot_idx_cm]
                        det_data_cm = fold_data.get('detailed_data')
                        plot_success_cm = False
                        if det_data_cm and isinstance(det_data_cm, dict):
                            y_true = det_data_cm.get('y_true'); y_pred = det_data_cm.get('y_pred')
                            if y_true and y_pred and class_names:
                                plot_success_cm = ResultsPlotter._plot_confusion_matrix(y_true, y_pred, class_names, f"Fold {fold_idx+1}", plot_dir / "dummy.png", False, ax=ax)
                        if not plot_success_cm:
                            ax.text(0.5, 0.5, 'No CM Data', ha='center', va='center', transform=ax.transAxes); ax.set_title(f"Fold {fold_idx+1}")
                        plot_idx_cm += 1 # Increment regardless to hide unused axes correctly
                    for idx in range(plot_idx_cm, len(axes_cm_flat)): axes_cm_flat[idx].set_visible(False)
                    if any(ax.has_data() for ax in axes_cm_flat[:plot_idx_cm]): # Check if anything was actually plotted
                       _save_show_plot(fig_cm, plot_dir / "confusion_matrices_all_folds.png", show_plots)
                    else: plt.close(fig_cm)

                # --- Grouped Per-Fold ROC Curves (Max 2 per row) ---
                n_cols_roc = 2 # Max 2 ROC plots per row
                n_rows_roc = (n_folds_actual + n_cols_roc - 1) // n_cols_roc
                fig_roc, axes_roc = plt.subplots(n_rows_roc, n_cols_roc, figsize=(7 * n_cols_roc, 6 * n_rows_roc), sharex=True, sharey=True, squeeze=False)
                fig_roc.suptitle(f"{title_prefix} - ROC Curves per Fold", fontsize=16)
                axes_roc_flat = axes_roc.flatten()
                plot_idx_roc = 0
                for fold_idx, fold_data in enumerate(fold_details):
                     if plot_idx_roc >= len(axes_roc_flat): break
                     ax = axes_roc_flat[plot_idx_roc]
                     det_data_curves = fold_data.get('detailed_data')
                     roc_curve_points_fold = None; plot_success_roc = False
                     if det_data_curves and isinstance(det_data_curves, dict): roc_curve_points_fold = det_data_curves.get('roc_curve_points')
                     if roc_curve_points_fold and isinstance(roc_curve_points_fold, dict): plot_success_roc = ResultsPlotter._plot_roc_curves(roc_curve_points_fold, f"Fold {fold_idx+1}", plot_dir / "dummy.png", False, ax=ax)
                     if not plot_success_roc: ax.text(0.5, 0.5, 'No ROC Data', ha='center', va='center', transform=ax.transAxes); ax.set_title(f"Fold {fold_idx+1}")
                     # Common formatting
                     ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim([-0.05, 1.05]); ax.set_ylim([-0.05, 1.05])
                     if plot_idx_roc >= (n_rows_roc - 1) * n_cols_roc : ax.set_xlabel('False Positive Rate', fontsize=9)
                     else: ax.set_xlabel("")
                     if plot_idx_roc % n_cols_roc == 0: ax.set_ylabel('True Positive Rate', fontsize=9)
                     else: ax.set_ylabel("")
                     ax.tick_params(axis='both', which='major', labelsize=8)
                     plot_idx_roc += 1
                for idx in range(plot_idx_roc, len(axes_roc_flat)): axes_roc_flat[idx].set_visible(False)
                if any(ax.has_data() for ax in axes_roc_flat[:plot_idx_roc]): _save_show_plot(fig_roc, plot_dir / "roc_curves_all_folds.png", show_plots)
                else: plt.close(fig_roc)


                # --- Grouped Per-Fold PR Curves (Max 2 per row) ---
                n_cols_pr = 2 # Max 2 PR plots per row
                n_rows_pr = (n_folds_actual + n_cols_pr - 1) // n_cols_pr
                fig_pr, axes_pr = plt.subplots(n_rows_pr, n_cols_pr, figsize=(7 * n_cols_pr, 6 * n_rows_pr), sharex=True, sharey=True, squeeze=False)
                fig_pr.suptitle(f"{title_prefix} - PR Curves per Fold", fontsize=16)
                axes_pr_flat = axes_pr.flatten()
                plot_idx_pr = 0
                for fold_idx, fold_data in enumerate(fold_details):
                    if plot_idx_pr >= len(axes_pr_flat): break
                    ax = axes_pr_flat[plot_idx_pr]
                    det_data_curves = fold_data.get('detailed_data')
                    pr_curve_points_fold = None; plot_success_pr = False
                    if det_data_curves and isinstance(det_data_curves, dict): pr_curve_points_fold = det_data_curves.get('pr_curve_points')
                    if pr_curve_points_fold and isinstance(pr_curve_points_fold, dict): plot_success_pr = ResultsPlotter._plot_pr_curves(pr_curve_points_fold, f"Fold {fold_idx+1}", plot_dir / "dummy.png", False, ax=ax)
                    if not plot_success_pr: ax.text(0.5, 0.5, 'No PR Data', ha='center', va='center', transform=ax.transAxes); ax.set_title(f"Fold {fold_idx+1}")
                    # Common formatting
                    ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim([-0.05, 1.05]); ax.set_ylim([-0.05, 1.05])
                    if plot_idx_pr >= (n_rows_pr - 1) * n_cols_pr : ax.set_xlabel('Recall', fontsize=9)
                    else: ax.set_xlabel("")
                    if plot_idx_pr % n_cols_pr == 0: ax.set_ylabel('Precision', fontsize=9)
                    else: ax.set_ylabel("")
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plot_idx_pr += 1
                for idx in range(plot_idx_pr, len(axes_pr_flat)): axes_pr_flat[idx].set_visible(False)
                if any(ax.has_data() for ax in axes_pr_flat[:plot_idx_pr]): _save_show_plot(fig_pr, plot_dir / "pr_curves_all_folds.png", show_plots)
                else: plt.close(fig_pr)

                # --- Plot CV Macro Average Points (across all folds) ---
                ResultsPlotter._plot_cv_macro_roc_points(fold_details, f"{title_prefix}\nMacro Average ROC Points per Fold", plot_dir / "cv_macro_roc_points.png", show_plots)
                # No PR macro points plot

            # Generate metrics table for the first fold (optional)
            if fold_details and isinstance(fold_details, list) and len(fold_details) > 0 and fold_details[0] and TABULATE_AVAILABLE:
                 logger.info("Generating metrics table for Fold 1 results.")
                 ResultsPlotter._generate_metrics_table(fold_details[0], output_path=plot_dir / "fold_1_metrics_table.txt")

            logger.info(f"Finished plotting for cv_model_evaluation: {run_id}")

        except Exception as e: logger.error(f"Failed to plot cv_model_evaluation results from {json_path}: {e}", exc_info=True)
