import json
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.utils
from PIL import Image
# Assuming ArtifactRepository can be imported for type hinting if needed
# from ..artifact_repository import ArtifactRepository, LocalFileSystemRepository

# --- Plotting Libraries ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None; sns = None; MaxNLocator = None

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    confusion_matrix = None; ConfusionMatrixDisplay = None; auc = None

try:
    import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    tabulate = None

try:
    from skimage.segmentation import mark_boundaries
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    mark_boundaries = None

try:
    from lime.lime_image import LimeImageExplainer as PlotterLimeImageExplainer
    LIME_PLOTTER_AVAILABLE = True
except ImportError:
    LIME_PLOTTER_AVAILABLE = False
    PlotterLimeImageExplainer = None

try:
    from ..ml.logger_utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers(): logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)


# --- Helper Functions ---
def _check_plotting_libs() -> bool:
    libs_ok = True
    if not MATPLOTLIB_AVAILABLE: logger.error("Matplotlib is required. `pip install matplotlib seaborn`."); libs_ok = False
    if not SKLEARN_METRICS_AVAILABLE: logger.warning("Scikit-learn metrics (CM, AUC) not found. Some plots/metrics skipped.")
    if not TABULATE_AVAILABLE: logger.warning("Tabulate library not found. Metrics tables will not be generated.")
    if sns is None: logger.warning("Seaborn library not found. Plot aesthetics might be affected.")
    return libs_ok


def _save_figure_or_show(fig: plt.Figure,
                         repository: Optional[Any],
                         s3_key_or_local_path: Optional[Union[str, Path]],
                         show_plots: bool):
    if not fig: return
    full_path_for_log = s3_key_or_local_path if s3_key_or_local_path else "Display Only"
    try:
        fig.tight_layout(pad=1.5)
        if repository and s3_key_or_local_path:
            save_id = repository.save_plot_figure(fig, str(s3_key_or_local_path))
            # save_plot_figure should handle logging success/failure
        elif isinstance(s3_key_or_local_path, (str, Path)):
            local_path = Path(s3_key_or_local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(local_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved locally to: {local_path}")
        elif not show_plots:
             logger.debug(f"Plot for '{getattr(fig, '_suptitle_text', 'Figure')}' generated but not saved (no path/repo) and not shown.")
             plt.close(fig)
             return

        if show_plots:
            plt.show()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to save/show plot ({full_path_for_log}): {e}", exc_info=True)
        if fig: plt.close(fig)


class ResultsPlotter:
    @staticmethod
    def _load_results_if_path(results_input: Union[Dict[str, Any], str, Path]) -> Optional[Dict[str, Any]]:
        if isinstance(results_input, dict): return results_input
        try:
            path = Path(results_input)
            if not path.is_file(): logger.error(f"Results file not found: {path}"); return None
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: logger.error(f"Failed to load/parse JSON from {results_input}: {e}"); return None

    @staticmethod
    def _plot_learning_curves(history: List[Dict[str, Any]], title: str,
                              ax_loss_provided=None, ax_acc_provided=None
                              ) -> Optional[plt.Figure]:
        if not history or not MATPLOTLIB_AVAILABLE: return None
        try:
            df = pd.DataFrame(history)
            if 'epoch' not in df.columns:
                logger.warning(f"Plotting LC for '{title}': 'epoch' column missing.");
                return None

            is_standalone_plot = ax_loss_provided is None or ax_acc_provided is None
            fig: Optional[plt.Figure] = None  # Initialize fig

            if is_standalone_plot:
                # Create new figure and axes
                fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))  # Slightly smaller for many folds
                fig.suptitle(title, fontsize=12)  # Standalone title
            else:
                # Use provided axes
                ax_loss, ax_acc = ax_loss_provided, ax_acc_provided
                if ax_loss:
                    fig = ax_loss.figure  # Get figure from provided axis
                elif ax_acc:
                    fig = ax_acc.figure  # Or from the other axis
                # If both axes are provided, they should belong to the same figure

            if fig is None and (
                    ax_loss is not None or ax_acc is not None):  # Should not happen if axes are from subplots
                logger.error("Provided axes for learning curve plot do not have a figure.")
                return None

            plot_occurred_on_loss_ax = False
            plot_occurred_on_acc_ax = False

            # Loss Plot
            has_train_loss = 'train_loss' in df.columns and df['train_loss'].notna().any()
            has_valid_loss = 'valid_loss' in df.columns and df['valid_loss'].notna().any()
            if has_train_loss: ax_loss.plot(df['epoch'], df['train_loss'], marker='o', ms=3, ls='-', label='Train Loss')
            if has_valid_loss: ax_loss.plot(df['epoch'], df['valid_loss'], marker='x', ms=4, ls='--',
                                            label='Valid Loss')
            if has_train_loss or has_valid_loss:
                plot_occurred_on_loss_ax = True
                ax_loss.set_xlabel('Epoch', fontsize=9)
                ax_loss.set_ylabel('Loss', fontsize=9)
                ax_loss.set_title('Loss' if not is_standalone_plot else "Loss vs. Epoch", fontsize=10)
                if MaxNLocator: ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_loss.legend(fontsize='xx-small')  # Always show legend, adjust size for subplots
                ax_loss.grid(True, alpha=0.6)
                ax_loss.tick_params(axis='both', labelsize=8)
            else:
                ax_loss.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=ax_loss.transAxes)
                ax_loss.set_title('Loss Data Missing', fontsize=10)

            # Accuracy Plot
            has_train_acc = 'train_acc' in df.columns and df['train_acc'].notna().any()
            has_valid_acc = 'valid_acc' in df.columns and df['valid_acc'].notna().any()
            if has_train_acc: ax_acc.plot(df['epoch'], df['train_acc'], marker='o', ms=3, ls='-', label='Train Acc')
            if has_valid_acc: ax_acc.plot(df['epoch'], df['valid_acc'], marker='x', ms=4, ls='--', label='Valid Acc')
            if has_train_acc or has_valid_acc:
                plot_occurred_on_acc_ax = True
                ax_acc.set_xlabel('Epoch', fontsize=9)
                ax_acc.set_ylabel('Accuracy', fontsize=9)
                try:  # Set Y limits for accuracy, robustly
                    current_ymin, current_ymax = ax_acc.get_ylim()
                    new_ymin = max(0, current_ymin if current_ymin > -float('inf') else 0)
                    new_ymax = min(1.05, current_ymax if current_ymax < float('inf') else 1.05)
                    ax_acc.set_ylim(bottom=new_ymin, top=new_ymax)
                except Exception:
                    ax_acc.set_ylim(bottom=0, top=1.05)  # Fallback
                ax_acc.set_title('Accuracy' if not is_standalone_plot else "Accuracy vs. Epoch", fontsize=10)
                if MaxNLocator: ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_acc.legend(fontsize='xx-small')
                ax_acc.grid(True, alpha=0.6)
                ax_acc.tick_params(axis='both', labelsize=8)
            else:
                ax_acc.text(0.5, 0.5, 'No Acc Data', ha='center', va='center', transform=ax_acc.transAxes)
                ax_acc.set_title('Acc Data Missing', fontsize=10)

            if not (plot_occurred_on_loss_ax or plot_occurred_on_acc_ax):  # If nothing was plotted on either axis
                if is_standalone_plot and fig:  # Only close if we created it
                    plt.close(fig)
                return None  # Indicate no figure to save/show

            return fig  # Return the figure object (could be newly created or passed via axes)

        except Exception as e:
            logger.error(f"Error plotting learning curves for '{title}': {e}", exc_info=True)
            # Attempt to close figure if it was created in this function call and an error occurred
            if 'fig' in locals() and fig is not None and is_standalone_plot:
                plt.close(fig)
            return None

    @staticmethod
    def _generate_metrics_table(metrics_data: Dict[str, Any], output_path: Optional[Path], repository: Optional[Any]=None, s3_key: Optional[str]=None):
        # ... (Same implementation, but uses repository.save_text_file if repo and s3_key given) ...
        if not TABULATE_AVAILABLE: return None
        per_class_metrics = metrics_data.get('per_class'); macro_metrics = metrics_data.get('macro_avg'); overall_acc = metrics_data.get('overall_accuracy')
        if not per_class_metrics or not macro_metrics: logger.warning("Cannot generate metrics table: 'per_class' or 'macro_avg' data missing."); return None
        headers = ["Metric"] + list(per_class_metrics.keys()) + ["Macro Avg"]; table_data = []
        metric_keys = ['precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc']
        for key_m in metric_keys:
            row = [key_m.replace('_', ' ').title()]
            for class_name in per_class_metrics.keys(): class_val = per_class_metrics[class_name].get(key_m, np.nan); row.append(f"{class_val:.4f}" if not np.isnan(class_val) else "N/A")
            macro_val = macro_metrics.get(key_m, np.nan); row.append(f"{macro_val:.4f}" if not np.isnan(macro_val) else "N/A"); table_data.append(row)
        acc_row = ["Overall Acc"] + ["-"] * len(per_class_metrics) + [f"{overall_acc:.4f}" if overall_acc is not None else "N/A"]; table_data.append(acc_row)
        try:
            table_str = tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f", stralign="center")
            if repository and s3_key: repository.save_text_file(table_str, s3_key)
            elif output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_text(table_str, encoding='utf-8'); logger.info(f"Metrics table saved to: {output_path}")
            return table_str
        except Exception as e: logger.error(f"Error generating metrics table: {e}"); return None

    @staticmethod
    def _generate_aggregated_metrics_table(aggregated_metrics: Dict[str, Dict[str, float]], output_path: Optional[Path], repository: Optional[Any]=None, s3_key: Optional[str]=None):
        if not TABULATE_AVAILABLE or not aggregated_metrics: return None
        # ... (rest of table generation) ...
        headers = ["Metric", "Mean", "Std Dev", "SEM", "CI Margin", "CI Lower", "CI Upper"]; table_data = []; metric_keys_ordered = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']
        for key_m in metric_keys_ordered:
            stats = aggregated_metrics.get(key_m)
            if stats and isinstance(stats, dict):
                row = [key_m.replace('_', ' ').title()];
                row.append(f"{stats.get('mean', np.nan):.4f}" if not np.isnan(stats.get('mean', np.nan)) else "N/A");
                row.append(f"{stats.get('std_dev', np.nan):.4f}" if not np.isnan(stats.get('std_dev', np.nan)) else "N/A");
                row.append(f"{stats.get('sem', np.nan):.4f}" if not np.isnan(stats.get('sem', np.nan)) else "N/A");
                row.append(f"{stats.get('margin_of_error', np.nan):.4f}" if stats.get('margin_of_error') is not None and not np.isnan(stats.get('margin_of_error', np.nan)) else "N/A");
                row.append(f"{stats.get('ci_lower', np.nan):.4f}" if stats.get('ci_lower') is not None and not np.isnan(stats.get('ci_lower', np.nan)) else "N/A");
                row.append(f"{stats.get('ci_upper', np.nan):.4f}" if stats.get('ci_upper') is not None and not np.isnan(stats.get('ci_upper', np.nan)) else "N/A");
                table_data.append(row)
        if not table_data: logger.warning("No data to generate aggregated metrics table."); return None
        try:
            table_str = tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f", stralign="center")
            if repository and s3_key: repository.save_text_file(table_str, s3_key)
            elif output_path: Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_text(table_str, encoding='utf-8'); logger.info(f"Aggregated metrics table saved to: {output_path}")
            return table_str
        except Exception as e: logger.error(f"Error generating aggregated metrics table: {e}"); return None

    @staticmethod
    def _plot_confusion_matrix(y_true: List, y_pred: List, classes: List[str], title: str,
                               ax_provided=None) -> Optional[plt.Figure]:
        if not y_true or not y_pred or not classes or not SKLEARN_METRICS_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning(f"Cannot plot CM for '{title}': Missing data/libs.");
            return None
        is_standalone_plot = ax_provided is None
        fig_to_return: Optional[plt.Figure] = None
        try:
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            if is_standalone_plot:
                fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.7), max(5, len(classes) * 0.6)))
                fig_to_return = fig
            else:
                ax = ax_provided
                fig_to_return = ax.figure  # Get figure from provided axis
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, colorbar=is_standalone_plot)
            ax.set_title(title, fontsize=10 if not is_standalone_plot else 12)
            return fig_to_return
        except Exception as e:
            logger.error(f"Error plotting CM for '{title}': {e}", exc_info=True)
            if is_standalone_plot and fig_to_return: plt.close(fig_to_return)
            return None

    @staticmethod
    def _plot_roc_curves(roc_data: Dict[str, Dict[str, List]], title_prefix: str,
                         ax_provided=None) -> Optional[plt.Figure]:
        if not roc_data or not MATPLOTLIB_AVAILABLE: return None
        is_standalone_plot = ax_provided is None
        fig_to_return: Optional[plt.Figure] = None
        try:
            if is_standalone_plot:
                fig, ax = plt.subplots(figsize=(8, 7))
                fig_to_return = fig
            else:
                ax = ax_provided
                fig_to_return = ax.figure
            ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)' if is_standalone_plot else None, alpha=0.7)
            num_classes = len(roc_data)
            try:
                colors = plt.cm.get_cmap('tab20', num_classes)
            except ValueError:
                colors = plt.cm.get_cmap('tab10', num_classes)
            plotted = False
            for i, (class_name, curve_points) in enumerate(roc_data.items()):
                fpr, tpr, roc_auc_val = curve_points.get('fpr'), curve_points.get('tpr'), curve_points.get('roc_auc',
                                                                                                           np.nan)
                if fpr and tpr and isinstance(fpr, list) and isinstance(tpr, list):
                    plotted = True
                    if np.isnan(roc_auc_val) and auc and len(fpr) > 1 and len(tpr) > 1:
                        try:
                            roc_auc_val = auc(fpr, tpr)
                        except:
                            roc_auc_val = np.nan
                    label = f'{class_name} (AUC={roc_auc_val:.2f})' if not np.isnan(roc_auc_val) else class_name
                    ax.plot(fpr, tpr, color=colors(i % colors.N), lw=1.5, label=label, alpha=0.8)
            if not plotted:
                if is_standalone_plot and fig_to_return: plt.close(fig_to_return)
                logger.warning(f"No valid ROC curve data found for {title_prefix}");
                return None
            ax.set_xlabel('False Positive Rate');
            ax.set_ylabel('True Positive Rate (Recall)');
            ax.set_title(title_prefix)
            ax.legend(loc='lower right', fontsize='small');
            ax.grid(True, alpha=0.6);
            ax.set_xlim([-0.05, 1.05]);
            ax.set_ylim([-0.05, 1.05])
            return fig_to_return
        except Exception as e:
            logger.error(f"Error plotting ROC for '{title_prefix}': {e}", exc_info=True)
        if is_standalone_plot and fig_to_return: plt.close(fig_to_return)
        return None

    @staticmethod
    def _plot_pr_curves(pr_data: Dict[str, Dict[str, List]], title_prefix: str,
                        ax_provided=None) -> Optional[plt.Figure]:
        if not pr_data or not MATPLOTLIB_AVAILABLE: return None
        is_standalone_plot = ax_provided is None
        fig_to_return: Optional[plt.Figure] = None
        try:
            if is_standalone_plot:
                fig, ax = plt.subplots(figsize=(8, 7))
                fig_to_return = fig
            else:
                ax = ax_provided
                fig_to_return = ax.figure
            num_classes = len(pr_data)
            try:
                colors = plt.cm.get_cmap('tab20', num_classes)
            except ValueError:
                colors = plt.cm.get_cmap('tab10', num_classes)
            plotted = False
            for i, (class_name, curve_points) in enumerate(pr_data.items()):
                precision, recall, pr_auc_val = curve_points.get('precision'), curve_points.get(
                    'recall'), curve_points.get('pr_auc', np.nan)
                if precision and recall and isinstance(precision, list) and isinstance(recall, list):
                    plotted = True
                    if np.isnan(pr_auc_val) and auc and len(recall) > 1 and len(precision) > 1:
                        try:
                            order = np.argsort(recall); pr_auc_val = auc(np.array(recall)[order],
                                                                         np.array(precision)[order])
                        except:
                            pr_auc_val = np.nan
                    label = f'{class_name} (AUPRC={pr_auc_val:.2f})' if not np.isnan(pr_auc_val) else class_name
                    ax.plot(recall, precision, color=colors(i % colors.N), lw=1.5, label=label, alpha=0.8)
            if not plotted:
                if is_standalone_plot and fig_to_return: plt.close(fig_to_return)
                logger.warning(f"No valid PR curve data for {title_prefix}");
                return None
            ax.set_xlabel('Recall');
            ax.set_ylabel('Precision');
            ax.set_title(title_prefix)
            ax.legend(loc='lower left', fontsize='small');
            ax.grid(True, alpha=0.6);
            ax.set_xlim([-0.05, 1.05]);
            ax.set_ylim([-0.05, 1.05])
            return fig_to_return
        except Exception as e:
            logger.error(f"Error plotting PR for '{title_prefix}': {e}", exc_info=True)
        if is_standalone_plot and fig_to_return: plt.close(fig_to_return)
        return None

    @staticmethod
    def _plot_cv_aggregated_metrics(aggregated_metrics: Dict[str, Dict[str, float]], title: str) -> Optional[
        plt.Figure]:
        if not aggregated_metrics or not MATPLOTLIB_AVAILABLE: return None
        # ... (Plotting logic for bar chart - unchanged, but returns fig) ...
        metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'specificity_macro',
                           'roc_auc_macro', 'pr_auc_macro'];
        plot_data = {'metric': [], 'mean': [], 'ci_margin': [], 'ci_lower': [], 'ci_upper': []}
        for key_m in metrics_to_plot:
            metric_stats = aggregated_metrics.get(key_m);
            if metric_stats and isinstance(metric_stats, dict): mean = metric_stats.get('mean', np.nan)
            if not np.isnan(mean): plot_data['metric'].append(key_m.replace('_', ' ').title()); plot_data[
                'mean'].append(mean); margin = metric_stats.get('margin_of_error',
                                                                0.0); margin = 0.0 if margin is None or np.isnan(
                margin) else margin; plot_data['ci_margin'].append(margin); plot_data['ci_lower'].append(
                metric_stats.get('ci_lower', mean - margin)); plot_data['ci_upper'].append(
                metric_stats.get('ci_upper', mean + margin))
        if not plot_data['metric']: logger.warning(
            f"No valid aggregated metrics found to plot for '{title}'."); return None
        df = pd.DataFrame(plot_data)
        try:
            fig, ax = plt.subplots(figsize=(10, max(6, len(df['metric']) * 0.6)));
            y_pos = np.arange(len(df['metric']));
            error = df['ci_margin']
            ax.barh(y_pos, df['mean'], xerr=error, align='center', alpha=0.7, ecolor='black', capsize=5);
            ax.set_yticks(y_pos);
            ax.set_yticklabels(df['metric']);
            ax.invert_yaxis()
            ax.set_xlabel('Score');
            ax.set_title(title);
            ax.set_xlim(left=max(0, df['ci_lower'].min() - 0.05) if df['ci_lower'].notna().any() else 0,
                        right=min(1.05, df['ci_upper'].max() + 0.05) if df['ci_upper'].notna().any() else 1.05);
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)
            for i, v_mean in enumerate(df['mean']): ax.text(v_mean + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), i,
                                                            f"{v_mean:.3f}", va='center', color='black', fontsize=9)
            return fig
        except Exception as e:
            logger.error(f"Error plotting CV aggregated metrics for '{title}': {e}", exc_info=True); return None

    @staticmethod
    def _generate_grid_search_table(cv_results: Dict[str, Any],
                                    output_path: Optional[Path] = None,  # For local saving if no repo
                                    repository: Optional[Any] = None,  # ArtifactRepository instance
                                    s3_key: Optional[str] = None,  # S3 object key if using repo
                                    top_n: int = 15):
        if not TABULATE_AVAILABLE or not cv_results: return None
        if not isinstance(cv_results, dict): logger.warning(
            "Cannot generate grid search table: cv_results is not a dictionary."); return None
        try:
            df = pd.DataFrame(cv_results)
            param_cols = [col for col in df.columns if col.startswith('param_')]
            score_cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
            time_cols = ['mean_fit_time', 'mean_score_time']
            cols_to_show = score_cols + time_cols + param_cols
            cols_to_show = [col for col in cols_to_show if col in df.columns]

            if not cols_to_show: logger.warning("No relevant columns in cv_results for grid search table."); return None

            df_top: pd.DataFrame
            if 'rank_test_score' in df.columns:
                df_top = df.nsmallest(top_n, 'rank_test_score')[cols_to_show]
            else:
                logger.warning("Cannot rank grid search results: 'rank_test_score' missing. Showing first N rows.")
                df_top = df[cols_to_show].head(top_n)

            if df_top.empty: logger.warning("Grid search DataFrame is empty after filtering columns."); return None

            df_top.columns = [col.replace('param_', '') if col.startswith('param_') else col for col in df_top.columns]
            table_str = tabulate.tabulate(df_top, headers='keys', tablefmt='fancy_grid', showindex=False,
                                          floatfmt=".4f")

            if repository and s3_key:
                if repository.save_text_file(table_str, s3_key):
                    logger.info(f"Grid search summary table saved via repository to: {s3_key}")
                else:
                    logger.error(f"Failed to save grid search table via repository to: {s3_key}")
            elif output_path:
                output_p = Path(output_path);
                output_p.parent.mkdir(parents=True, exist_ok=True)
                output_p.write_text(table_str, encoding='utf-8')
                logger.info(f"Grid search summary table saved locally to: {output_p}")
            else:
                logger.debug("Grid search table generated but not saved.")
            return table_str
        except Exception as e:
            logger.error(f"Error generating or saving grid search table: {e}"); return None

    @staticmethod
    def _generate_nested_cv_param_table(best_params_list: List[Optional[Dict]],
                                        best_scores_list: List[Optional[float]],
                                        output_path: Optional[Path] = None,  # For local saving
                                        repository: Optional[Any] = None,  # ArtifactRepository
                                        s3_key: Optional[str] = None  # S3 object key
                                        ):
        if not TABULATE_AVAILABLE or not best_params_list or not best_scores_list: return None
        if len(best_params_list) != len(best_scores_list): logger.warning(
            "Mismatch in lengths for nested CV param table."); return None
        try:
            table_data = [];
            all_param_keys = set()
            for params in best_params_list:
                if isinstance(params, dict): all_param_keys.update(params.keys())
            sorted_param_keys = sorted([key.replace('module__', '') for key in all_param_keys])
            headers = ["Outer Fold", "Inner Best Score"] + sorted_param_keys
            for i, (params, score) in enumerate(zip(best_params_list, best_scores_list)):
                row = [f"Fold {i + 1}"];
                row.append(f"{score:.4f}" if score is not None and not np.isnan(score) else "N/A")
                if isinstance(params, dict):
                    for short_key in sorted_param_keys:
                        original_key = 'module__' + short_key if 'module__' + short_key in params else short_key
                        row.append(params.get(original_key, "N/A"))
                else:
                    row.extend(["N/A"] * len(sorted_param_keys))
                table_data.append(row)
            table_str = tabulate.tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")

            if repository and s3_key:
                if repository.save_text_file(table_str, s3_key):
                    logger.info(f"Nested CV best params table saved via repository to: {s3_key}")
                else:
                    logger.error(f"Failed to save nested CV params table via repository to: {s3_key}")
            elif output_path:
                output_p = Path(output_path);
                output_p.parent.mkdir(parents=True, exist_ok=True)
                output_p.write_text(table_str, encoding='utf-8')
                logger.info(f"Nested CV best params table saved locally to: {output_p}")
            else:
                logger.debug("Nested CV params table generated but not saved.")
            return table_str
        except Exception as e:
            logger.error(f"Error generating or saving nested CV params table: {e}"); return None

    @staticmethod
    def _plot_macro_roc_point(macro_metrics: Dict[str, float], title: str) -> Optional[plt.Figure]:
        if not macro_metrics or not MATPLOTLIB_AVAILABLE: return None
        # ... (Plotting logic - unchanged, but returns fig) ...
        try:
            recall = macro_metrics.get('recall', np.nan);
            specificity = macro_metrics.get('specificity', np.nan)
            if np.isnan(recall) or np.isnan(specificity): logger.warning(
                f"Cannot plot macro ROC point for '{title}': Missing recall or specificity."); return None
            fpr_macro = 1.0 - specificity;
            tpr_macro = recall;
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Line');
            ax.scatter(fpr_macro, tpr_macro, marker='o', color='red', s=100,
                       label=f'Macro Avg.\n(FPR={fpr_macro:.3f}, TPR={tpr_macro:.3f})', zorder=3)
            ax.plot([0, fpr_macro], [0, tpr_macro], color='red', linestyle=':', linewidth=1, alpha=0.6);
            ax.plot([fpr_macro, 1], [tpr_macro, 1], color='red', linestyle=':', linewidth=1, alpha=0.6)
            ax.set_xlabel('Macro Avg FPR (1-Specificity)');
            ax.set_ylabel('Macro Avg TPR (Recall)');
            ax.set_title(title);
            ax.legend(loc='lower right', fontsize='small');
            ax.grid(True, alpha=0.6);
            ax.set_xlim([-0.05, 1.05]);
            ax.set_ylim([-0.05, 1.05])
            return fig
        except Exception as e:
            logger.error(f"Error plotting macro ROC point for '{title}': {e}", exc_info=True); return None


    @staticmethod
    def _plot_cv_macro_roc_points(fold_details: List[Dict], title: str) -> Optional[plt.Figure]:
        if not fold_details or not MATPLOTLIB_AVAILABLE: return None
        # ... (Plotting logic - unchanged, but returns fig) ...
        try:
            fpr_points = []; tpr_points = []; fold_indices = []
            for i, fold_data in enumerate(fold_details):
                macro_metrics = None;
                if isinstance(fold_data, dict): macro_metrics = fold_data.get('macro_avg')
                if isinstance(macro_metrics, dict): recall = macro_metrics.get('recall', np.nan); specificity = macro_metrics.get('specificity', np.nan)
                if not np.isnan(recall) and not np.isnan(specificity): fpr_points.append(1.0 - specificity); tpr_points.append(recall); fold_indices.append(i + 1)
            if not fpr_points: logger.warning(f"No valid macro ROC points across CV folds for '{title}'."); return None
            fig, ax = plt.subplots(figsize=(7, 6.5)); ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Line')
            scatter = ax.scatter(fpr_points, tpr_points, marker='o', c=fold_indices, cmap='viridis', s=60, label='Fold Macro Avg.', zorder=3, alpha=0.7)
            mean_fpr = np.mean(fpr_points); mean_tpr = np.mean(tpr_points)
            ax.scatter(mean_fpr, mean_tpr, marker='^', color='red', s=120, label=f'Mean of Folds\n(FPR={mean_fpr:.3f}, TPR={mean_tpr:.3f})', zorder=4)
            ax.plot([0, mean_fpr], [0, mean_tpr], color='red', ls=':', lw=1, alpha=0.8); ax.plot([mean_fpr, 1], [mean_tpr, 1], color='red', ls=':', lw=1, alpha=0.8)
            ax.set_xlabel('Macro Avg FPR (1-Specificity)'); ax.set_ylabel('Macro Avg TPR (Recall)'); ax.set_title(title)
            if len(fold_indices) > 1: cbar = fig.colorbar(scatter, ax=ax, ticks=np.linspace(min(fold_indices), max(fold_indices), min(len(fold_indices), 10), dtype=int)); cbar.set_label('Fold Number')
            ax.legend(loc='lower right', fontsize='small'); ax.grid(True, alpha=0.6); ax.set_xlim([-0.05, 1.05]); ax.set_ylim([-0.05, 1.05])
            return fig
        except Exception as e: logger.error(f"Error plotting CV macro ROC points for '{title}': {e}", exc_info=True); return None

    # --- Main Public Plotting Methods ---

    @staticmethod
    def plot_single_train_results(results_input: Union[Dict[str, Any], str, Path],
                                  plot_save_dir_base: Optional[Union[str, Path]] = None,
                                  repository_for_plots: Optional[Any] = None,
                                  show_plots: bool = False):
        results_data = ResultsPlotter._load_results_if_path(results_input)
        if not results_data: return
        if not _check_plotting_libs(): return  # Essential check at the start

        run_id = results_data.get('run_id', "unknown_run")
        method_name = results_data.get('method', "single_train")
        try:
            logger.info(f"Plotting single_train results for: {run_id}")
            history = results_data.get('training_history')
            if history:
                # _plot_learning_curves will create its own figure here as no axes are passed
                fig_lc = ResultsPlotter._plot_learning_curves(
                    history, f"Single Train Learning Curves ({run_id})"
                )
                if fig_lc:  # Check if a figure was actually returned
                    s3_key_or_local_path_lc: Optional[Union[str, Path]] = None
                    if plot_save_dir_base:  # If a base location for saving is provided
                        s3_key_or_local_path_lc = str(
                            PurePath(plot_save_dir_base) / f"{method_name}_plots" / "learning_curves.png")

                    # _save_figure_or_show will handle if s3_key_or_local_path_lc is None (only show if show_plots)
                    _save_figure_or_show(fig_lc, repository_for_plots, s3_key_or_local_path_lc, show_plots)
                # If fig_lc is None, it means _plot_learning_curves handled closing its own figure if it was empty.
            else:
                logger.warning(f"No 'training_history' found for {run_id} learning curve plot.")
            logger.info(f"Finished plotting for single_train: {run_id}")
        except Exception as e:
            logger.error(f"Failed to plot single_train results for {run_id}: {e}", exc_info=True)

    @staticmethod
    def plot_single_eval_results(results_input: Union[Dict[str, Any], str, Path],
                                 class_names: List[str],
                                 plot_artifact_base_key_or_path: Optional[Union[str, Path]] = None,
                                 repository_for_plots: Optional[Any] = None,
                                 show_plots: bool = False):
        results_data = ResultsPlotter._load_results_if_path(results_input)
        if not results_data: return
        if not _check_plotting_libs(): return

        run_id = results_data.get('run_id', "unknown_run")
        method_name = results_data.get('method', "single_eval")
        try:
            logger.info(f"Plotting single_eval results for: {run_id}")
            title_prefix = f"Single Eval ({run_id})"

            plot_subdir_key_or_path = None
            if plot_artifact_base_key_or_path:
                plot_subdir_key_or_path = str((PurePath(plot_artifact_base_key_or_path) / f"{method_name}_plots").as_posix())

            if TABULATE_AVAILABLE:
                table_key = str(
                    (PurePath(plot_subdir_key_or_path) / "metrics_table.txt").as_posix()) if plot_subdir_key_or_path else None
                ResultsPlotter._generate_metrics_table(results_data, output_path=None, repository=repository_for_plots,
                                                       s3_key=table_key)

            macro_metrics = results_data.get('macro_avg')
            if isinstance(macro_metrics, dict):
                fig_mroc = ResultsPlotter._plot_macro_roc_point(macro_metrics, f"{title_prefix}\nMacro Avg ROC Point")
                if fig_mroc:
                    mroc_key = str(
                        (PurePath(plot_subdir_key_or_path) / "macro_roc_point.png").as_posix()) if plot_subdir_key_or_path else None
                    _save_figure_or_show(fig_mroc, repository_for_plots, mroc_key, show_plots)

            detailed_data = results_data.get('detailed_data')
            if detailed_data:
                y_true, y_pred = detailed_data.get('y_true'), detailed_data.get('y_pred')
                if y_true and y_pred and class_names and SKLEARN_METRICS_AVAILABLE:
                    fig_cm = ResultsPlotter._plot_confusion_matrix(y_true, y_pred, class_names, f"{title_prefix}\nCM")
                    if fig_cm:
                        cm_key = str((PurePath(
                            plot_subdir_key_or_path) / "confusion_matrix.png").as_posix()) if plot_subdir_key_or_path else None
                        _save_figure_or_show(fig_cm, repository_for_plots, cm_key, show_plots)

                roc_curve_points = detailed_data.get('roc_curve_points')
                if roc_curve_points:
                    fig_roc = ResultsPlotter._plot_roc_curves(roc_curve_points, title_prefix)
                    if fig_roc:
                        roc_key = str(
                            (PurePath(plot_subdir_key_or_path) / "roc_curves.png").as_posix()) if plot_subdir_key_or_path else None
                        _save_figure_or_show(fig_roc, repository_for_plots, roc_key, show_plots)

                pr_curve_points = detailed_data.get('pr_curve_points')
                if pr_curve_points:
                    fig_pr = ResultsPlotter._plot_pr_curves(pr_curve_points, title_prefix)
                    if fig_pr:
                        pr_key = str(
                            (PurePath(plot_subdir_key_or_path) / "pr_curves.png").as_posix()) if plot_subdir_key_or_path else None
                        _save_figure_or_show(fig_pr, repository_for_plots, pr_key, show_plots)
            else:
                logger.warning(f"Skipping detailed plots for '{title_prefix}': 'detailed_data' missing.")
            logger.info(f"Finished plotting for single_eval: {run_id}")
        except Exception as e:
            logger.error(f"Failed to plot single_eval for {run_id}: {e}", exc_info=True)

    @staticmethod
    def plot_non_nested_cv_results(results_input: Union[Dict[str, Any], str, Path],
                                   plot_save_dir_base: Optional[Union[str, Path]] = None,
                                   repository_for_plots: Optional[Any] = None,
                                   show_plots: bool = False):
        results_data = ResultsPlotter._load_results_if_path(results_input)
        if not results_data: return
        if not _check_plotting_libs(): return

        run_id = results_data.get('run_id', "unknown_run")
        method_name = results_data.get('method', "non_nested_search")
        try:
            logger.info(f"Plotting non_nested_cv results for: {run_id}")
            title_prefix = f"Non-Nested Search ({run_id})"
            # Learning Curves for best refit model
            history = results_data.get('best_refit_model_history')
            if history:
                fig_lc_refit = ResultsPlotter._plot_learning_curves(history, f"{title_prefix}\nLearning Curves (Best Refit Model)")
                if fig_lc_refit:
                    s3_key_or_local_path_lc: Optional[Union[str, Path]] = None
                    if plot_save_dir_base:  # This is the S3 prefix for the run, or local run dir
                        # key for the artifact repo: <plot_save_dir_base>/<method_name_plots>/<filename>
                        s3_key_or_local_path_lc = str(
                            (PurePath(plot_save_dir_base) / f"{method_name}_plots" / "best_refit_learning_curves.png"
                        ).as_posix())
                    _save_figure_or_show(fig_lc_refit, repository_for_plots, s3_key_or_local_path_lc, show_plots)
            else: logger.warning("No 'best_refit_model_history' for LC plot.")
            # Grid Search Table
            cv_results_data = results_data.get('cv_results')
            if cv_results_data and TABULATE_AVAILABLE:
                key_table = str((PurePath(plot_save_dir_base) / f"{method_name}_plots" / "grid_search_summary.txt").as_posix()) if plot_save_dir_base else None
                ResultsPlotter._generate_grid_search_table(cv_results_data, output_path=None, repository=repository_for_plots, s3_key=key_table, top_n=20)
            elif not cv_results_data: logger.warning("No 'cv_results' data for grid search table.")
            logger.info(f"Finished plotting for non_nested_cv: {run_id}")
        except Exception as e: logger.error(f"Failed to plot non_nested_cv for {run_id}: {e}", exc_info=True)

    @staticmethod
    def plot_nested_cv_results(results_input: Union[Dict[str, Any], str, Path],
                               plot_save_dir_base: Optional[Union[str, Path]] = None,
                               repository_for_plots: Optional[Any] = None,
                               show_plots: bool = False):
        results_data = ResultsPlotter._load_results_if_path(results_input)
        if not results_data: return
        if not _check_plotting_libs(): return

        run_id = results_data.get('run_id', "unknown_run")
        method_name = results_data.get('method', "nested_search")
        try:
            logger.info(f"Plotting nested_cv results for: {run_id}")
            title_prefix = f"Nested CV ({run_id})"
            # Outer fold score distributions
            outer_scores = results_data.get('outer_cv_scores')
            if outer_scores and isinstance(outer_scores, dict) and sns:
                metrics_to_plot = [k for k in outer_scores.keys() if k.startswith('test_')]; valid_metrics_data = {m: np.array(outer_scores[m]).astype(float) for m in metrics_to_plot}; valid_metrics_data = {m: v[~np.isnan(v)] for m, v in valid_metrics_data.items() if len(v[~np.isnan(v)]) > 0}; valid_metrics_to_plot = list(valid_metrics_data.keys())
                if valid_metrics_to_plot:
                    num_metrics = len(valid_metrics_to_plot); fig_dist, axes_dist = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5), sharey=True);
                    if num_metrics == 1: axes_dist = [axes_dist]
                    fig_dist.suptitle(f"{title_prefix} - Outer Fold Score Distributions", fontsize=16)
                    for ax_d, metric_d in zip(axes_dist, valid_metrics_to_plot): metric_data_d = valid_metrics_data[metric_d]; sns.boxplot(y=metric_data_d, ax=ax_d, showmeans=True); sns.stripplot(y=metric_data_d, ax=ax_d, color=".25", size=4); ax_d.set_ylabel("Score"); ax_d.set_xlabel(""); ax_d.set_title(metric_d.replace('test_', '').replace('_', ' ').title()); ax_d.grid(True, axis='y', linestyle='--', alpha=0.6)
                    key_dist = str((PurePath(plot_save_dir_base) / f"{method_name}_plots" / "outer_fold_score_distributions.png").as_posix()) if plot_save_dir_base else None
                    _save_figure_or_show(fig_dist, repository_for_plots, key_dist, show_plots)
            # Learning curves for each outer fold
            fold_histories = results_data.get('outer_fold_best_model_histories')
            if fold_histories and isinstance(fold_histories, list):
                 for i, history in enumerate(fold_histories):
                     if history:
                         fig_lc_fold = ResultsPlotter._plot_learning_curves(history, f"{title_prefix}\nLearning Curves (Outer Fold {i+1})")
                         if fig_lc_fold:
                             s3_key_or_local_path_lc: Optional[Union[str, Path]] = None
                             if plot_save_dir_base:  # This is the S3 prefix for the run, or local run dir
                                 # key for the artifact repo: <plot_save_dir_base>/<method_name_plots>/<filename>
                                 s3_key_or_local_path_lc = str(
                                     (PurePath(plot_save_dir_base) / f"{method_name}_plots" / f"outer_fold_{i+1}_learning_curves.png").as_posix() if plot_save_dir_base else None
                                 )
                             _save_figure_or_show(fig_lc_fold, repository_for_plots, s3_key_or_local_path_lc, show_plots)
            # Table of best params per fold
            best_params_per_fold = results_data.get('outer_fold_best_params_found'); inner_scores_per_fold = results_data.get('outer_fold_inner_cv_best_score')
            if best_params_per_fold and inner_scores_per_fold and TABULATE_AVAILABLE:
                key_table_params = str((PurePath(plot_save_dir_base) / f"{method_name}_plots" / "nested_cv_best_params_per_fold.txt").as_posix()) if plot_save_dir_base else None
                ResultsPlotter._generate_nested_cv_param_table(best_params_list=best_params_per_fold, best_scores_list=inner_scores_per_fold, output_path=None, repository=repository_for_plots, s3_key=key_table_params)
            logger.info(f"Finished plotting for nested_cv: {run_id}")
        except Exception as e: logger.error(f"Failed to plot nested_cv for {run_id}: {e}", exc_info=True)

    @staticmethod
    def plot_cv_model_evaluation_results(results_input: Union[Dict[str, Any], str, Path],
                                         class_names: List[str],
                                         plot_save_dir_base: Optional[Union[str, Path]] = None,
                                         repository_for_plots: Optional[Any] = None,
                                         show_plots: bool = False):
        results_data = ResultsPlotter._load_results_if_path(results_input)
        if not results_data: return
        if not _check_plotting_libs(): return

        run_id = results_data.get('run_id', "unknown_run")
        method_name = results_data.get('method', "cv_model_evaluation")

        plot_base_for_artifacts: Optional[str] = None  # Base S3 key or local dir string for this method's plots
        if plot_save_dir_base:
            plot_base_for_artifacts = str((PurePath(plot_save_dir_base) / f"{method_name}_plots").as_posix())

        try:
            evaluated_on = results_data.get('evaluated_on', 'unknown')
            n_folds_processed = results_data.get('n_folds_processed', 0)
            conf_level = results_data.get('confidence_level', 0.95)
            main_title_prefix = f"CV Model Eval ({evaluated_on}, {n_folds_processed} Folds, {run_id})"
            logger.info(f"Plotting cv_model_evaluation results for: {run_id}")

            # Aggregated Metrics Plot & Table
            aggregated_metrics = results_data.get('aggregated_metrics')
            if aggregated_metrics:
                fig_agg = ResultsPlotter._plot_cv_aggregated_metrics(aggregated_metrics,
                                                                     f"{main_title_prefix}\nAggregated Scores ({conf_level * 100:.0f}% CI)")
                if fig_agg:
                    key_agg_plot = str((PurePath(
                        plot_base_for_artifacts) / "aggregated_metrics_ci.png").as_posix()) if plot_base_for_artifacts else None
                    _save_figure_or_show(fig_agg, repository_for_plots, key_agg_plot, show_plots)
                if TABULATE_AVAILABLE:
                    key_agg_table = str((PurePath(
                        plot_base_for_artifacts) / "aggregated_metrics_table.txt").as_posix()) if plot_base_for_artifacts else None
                    ResultsPlotter._generate_aggregated_metrics_table(aggregated_metrics, output_path=None,
                                                                      repository=repository_for_plots,
                                                                      s3_key=key_agg_table)

            # Fold Score Distributions
            fold_scores = results_data.get('cv_fold_scores')
            if fold_scores and isinstance(fold_scores, dict) and sns:
                # ... (Code to create fig_dist - unchanged) ...
                metrics_to_plot = [k for k in fold_scores.keys() if k != 'error'];
                valid_metrics_data = {m: np.array(fold_scores[m]).astype(float) for m in metrics_to_plot};
                valid_metrics_data = {m: v[~np.isnan(v)] for m, v in valid_metrics_data.items() if
                                      len(v[~np.isnan(v)]) > 0};
                valid_metrics_to_plot = list(valid_metrics_data.keys())
                if valid_metrics_to_plot:
                    num_metrics = len(valid_metrics_to_plot);
                    n_cols = min(num_metrics, 3);
                    n_rows = (num_metrics + n_cols - 1) // n_cols;
                    fig_dist, axes_dist = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False);
                    axes_flat = axes_dist.flatten();
                    fig_dist.suptitle(f"{main_title_prefix} - Fold Score Distributions", fontsize=16);
                    plot_idx = 0;
                    for metric_d in valid_metrics_to_plot:
                        if plot_idx < len(axes_flat): ax = axes_flat[plot_idx]; metric_data = valid_metrics_data[
                            metric_d]; sns.boxplot(y=metric_data, ax=ax, showmeans=True); sns.stripplot(y=metric_data,
                                                                                                        ax=ax,
                                                                                                        color=".25",
                                                                                                        size=4); ax.set_ylabel(
                            "Score"); ax.set_xlabel(""); ax.set_title(metric_d.replace('_', ' ').title()); ax.grid(True,
                                                                                                                   axis='y',
                                                                                                                   linestyle='--',
                                                                                                                   alpha=0.6); plot_idx += 1
                    for j in range(plot_idx, len(axes_flat)): axes_flat[j].set_visible(False)
                    key_dist = str((PurePath(
                        plot_base_for_artifacts) / "fold_score_distributions.png").as_posix()) if plot_base_for_artifacts else None
                    _save_figure_or_show(fig_dist, repository_for_plots, key_dist, show_plots)

            # Detailed fold results section
            fold_details = results_data.get('fold_detailed_results')
            fold_histories = results_data.get('fold_training_histories')

            if fold_details and isinstance(fold_details, list) and len(fold_details) > 0:
                n_folds_actual = len(fold_details)
                logger.info(f"Generating grouped plots per fold for {n_folds_actual} folds...")

                # --- Grouped Learning Curves ---
                if fold_histories and len(fold_histories) == n_folds_actual and any(h for h in fold_histories if h):
                    n_lc_cols = 2;
                    n_lc_rows = n_folds_actual
                    fig_lc, axes_lc = plt.subplots(n_lc_rows, n_lc_cols, figsize=(12, 4 * n_lc_rows + 1), sharex=True,
                                                   squeeze=False)  # +1 for suptitle
                    fig_lc.suptitle(f"{main_title_prefix} - Learning Curves per Fold", fontsize=14,
                                    y=0.99)  # Adjust y for suptitle
                    plotted_any_lc = False
                    for fold_idx, history_item in enumerate(fold_histories):
                        if fold_idx >= n_lc_rows: break
                        ax_loss, ax_acc = axes_lc[fold_idx, 0], axes_lc[fold_idx, 1]
                        if history_item:
                            plot_success = ResultsPlotter._plot_learning_curves(history_item, f"Fold {fold_idx + 1}",
                                                                                ax_loss_provided=ax_loss,
                                                                                ax_acc_provided=ax_acc)
                            if plot_success: plotted_any_lc = True
                        else:
                            logger.warning(f"Skipping learning curves for Fold {fold_idx + 1}: History data missing.")
                            ax_loss.text(0.5, 0.5, 'No History', ha='center', va='center', transform=ax_loss.transAxes);
                            ax_loss.set_title(f'Fold {fold_idx + 1} - Loss Missing', fontsize=10)
                            ax_acc.text(0.5, 0.5, 'No History', ha='center', va='center', transform=ax_acc.transAxes);
                            ax_acc.set_title(f'Fold {fold_idx + 1} - Acc Missing', fontsize=10)
                        if fold_idx == n_lc_rows - 1:
                            ax_loss.set_xlabel('Epoch', fontsize=9); ax_acc.set_xlabel('Epoch', fontsize=9)
                        else:
                            ax_loss.set_xlabel(''); ax_acc.set_xlabel('')
                    if plotted_any_lc:
                        fig_lc.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
                        s3_key_or_local_path_lc = str(
                            (PurePath(plot_base_for_artifacts) / "learning_curves_all_folds.png"
                        ).as_posix()) if plot_base_for_artifacts else None
                        _save_figure_or_show(fig_lc, repository_for_plots, s3_key_or_local_path_lc, show_plots)
                    else:
                        plt.close(fig_lc)

                # --- Grouped Confusion Matrices (Max 2 per row) ---
                if SKLEARN_METRICS_AVAILABLE:
                    n_cm_cols = min(n_folds_actual, 2)  # Max 2 CMs per row
                    n_cm_rows = (n_folds_actual + n_cm_cols - 1) // n_cm_cols
                    fig_cm, axes_cm = plt.subplots(n_cm_rows, n_cm_cols, figsize=(6 * n_cm_cols, 5.5 * n_cm_rows),
                                                   squeeze=False)
                    fig_cm.suptitle(f"{main_title_prefix} - Confusion Matrices per Fold", fontsize=14, y=0.99)
                    axes_cm_flat = axes_cm.flatten()
                    plotted_any_cm = False
                    for fold_idx, fold_data in enumerate(fold_details):
                        if fold_idx >= len(axes_cm_flat): break
                        ax = axes_cm_flat[fold_idx]
                        det_data_cm = fold_data.get('detailed_data')  # Initialize for this scope
                        plot_success_cm = False
                        if det_data_cm and isinstance(det_data_cm, dict):
                            y_true, y_pred = det_data_cm.get('y_true'), det_data_cm.get('y_pred')
                            if y_true and y_pred and class_names:
                                plot_success_cm = ResultsPlotter._plot_confusion_matrix(y_true, y_pred, class_names,
                                                                                        f"Fold {fold_idx + 1}",
                                                                                        ax_provided=ax)
                        if not plot_success_cm:
                            ax.text(0.5, 0.5, 'No CM Data', ha='center', va='center', transform=ax.transAxes);
                            ax.set_title(f"Fold {fold_idx + 1}", fontsize=10)
                        if plot_success_cm: plotted_any_cm = True
                    for k_ax in range(fold_idx + 1, len(axes_cm_flat)): axes_cm_flat[k_ax].set_visible(False)
                    if plotted_any_cm:
                        fig_cm.tight_layout(rect=[0, 0, 1, 0.96])
                        key_cm_all = str((PurePath(
                            plot_base_for_artifacts) / "confusion_matrices_all_folds.png").as_posix()) if plot_base_for_artifacts else None
                        _save_figure_or_show(fig_cm, repository_for_plots, key_cm_all, show_plots)
                    else:
                        plt.close(fig_cm)

                # Helper for grouped ROC/PR plots per fold
                def _plot_grouped_curves_per_fold(curve_type: str):
                    data_key = f"{curve_type.lower()}_curve_points"
                    plot_title_part = curve_type.upper()
                    xlabel, ylabel = ("FPR", "TPR") if curve_type == "ROC" else ("Recall", "Precision")

                    n_curve_cols = min(n_folds_actual, 2)
                    n_curve_rows = (n_folds_actual + n_curve_cols - 1) // n_curve_cols
                    fig_curves, axes_curves = plt.subplots(n_curve_rows, n_curve_cols,
                                                           figsize=(7 * n_curve_cols, 6.5 * n_curve_rows), sharex=True,
                                                           sharey=True, squeeze=False)
                    fig_curves.suptitle(f"{main_title_prefix} - {plot_title_part} Curves per Fold", fontsize=14, y=0.99)
                    axes_curves_flat = axes_curves.flatten()
                    plotted_any_curve_fig = False

                    for fold_idx_curve, fold_data_curve in enumerate(fold_details):
                        if fold_idx_curve >= len(axes_curves_flat): break
                        ax_curve = axes_curves_flat[fold_idx_curve]
                        det_data_cv_curves = fold_data_curve.get('detailed_data')
                        curve_points_fold = None;
                        plot_success_curve = False
                        if det_data_cv_curves and isinstance(det_data_cv_curves, dict):
                            curve_points_fold = det_data_cv_curves.get(data_key)

                        plot_fn = ResultsPlotter._plot_roc_curves if curve_type == "ROC" else ResultsPlotter._plot_pr_curves
                        if curve_points_fold and isinstance(curve_points_fold, dict):
                            plot_success_curve = plot_fn(curve_points_fold, f"Fold {fold_idx_curve + 1}",
                                                         ax_provided=ax_curve)

                        if not plot_success_curve:
                            ax_curve.text(0.5, 0.5, f'No {plot_title_part} Data', ha='center', va='center',
                                          transform=ax_curve.transAxes)
                            ax_curve.set_title(f"Fold {fold_idx_curve + 1}", fontsize=10)
                        if plot_success_curve: plotted_any_curve_fig = True

                        ax_curve.grid(True, alpha=0.6);
                        ax_curve.set_xlim([-0.05, 1.05]);
                        ax_curve.set_ylim([-0.05, 1.05])
                        if fold_idx_curve // n_curve_cols == n_curve_rows - 1:
                            ax_curve.set_xlabel(xlabel, fontsize=9)
                        else:
                            ax_curve.set_xlabel("")
                        if fold_idx_curve % n_curve_cols == 0:
                            ax_curve.set_ylabel(ylabel, fontsize=9)
                        else:
                            ax_curve.set_ylabel("")
                        ax_curve.tick_params(axis='both', labelsize=8)

                    for k_ax_curve in range(fold_idx_curve + 1, len(axes_curves_flat)): axes_curves_flat[
                        k_ax_curve].set_visible(False)
                    if plotted_any_curve_fig:
                        fig_curves.tight_layout(rect=[0, 0, 1, 0.96])
                        key_fig_all_curves = str((PurePath(
                            plot_base_for_artifacts) / f"{data_key}_all_folds.png").as_posix()) if plot_base_for_artifacts else None
                        _save_figure_or_show(fig_curves, repository_for_plots, key_fig_all_curves, show_plots)
                    else:
                        plt.close(fig_curves)

                _plot_grouped_curves_per_fold("ROC")
                _plot_grouped_curves_per_fold("PR")

                # CV Macro ROC Points
                if fold_details and len(fold_details) > 0:
                    fig_cvmroc = ResultsPlotter._plot_cv_macro_roc_points(fold_details,
                                                                          f"{main_title_prefix}\nMacro Avg ROC Points per Fold")
                    if fig_cvmroc:
                        key_cvmroc = str((PurePath(
                            plot_base_for_artifacts) / "cv_macro_roc_points.png").as_posix()) if plot_base_for_artifacts else None
                        _save_figure_or_show(fig_cvmroc, repository_for_plots, key_cvmroc, show_plots)

                # Metrics Tables per Fold
                if TABULATE_AVAILABLE:
                    for i, fold_data_table in enumerate(fold_details):
                        if isinstance(fold_data_table, dict) and fold_data_table.get('per_class'):
                            key_table_fold = str((PurePath(
                                plot_base_for_artifacts) / f"fold_{i + 1}_metrics_table.txt").as_posix()) if plot_base_for_artifacts else None
                            ResultsPlotter._generate_metrics_table(fold_data_table, output_path=None,
                                                                   repository=repository_for_plots,
                                                                   s3_key=key_table_fold)

            logger.info(f"Finished plotting for cv_model_evaluation: {run_id}")
        except Exception as e:
            logger.error(f"Failed to plot cv_model_evaluation for {run_id}: {e}", exc_info=True)

    @staticmethod
    def plot_predictions(predictions_output: List[Dict[str, Any]],
                         image_pil_map: Dict[Any, Image.Image],  # Map of identifier to PIL Image
                         plot_save_dir_base: Optional[Union[str, Path]],
                         repository_for_plots: Optional[Any] = None,
                         show_plots: bool = False,
                         max_cols: int = 4,
                         generate_lime_plots: bool = False,  # Flag from pipeline call
                         lime_num_features_to_display: int = 5  # How many features to highlight in plot
                         ):
        if not predictions_output or not MATPLOTLIB_AVAILABLE:
            logger.warning("No predictions to plot or matplotlib not available.");
            return
        if not _check_plotting_libs(): return  # Includes LIME_PLOTTER_AVAILABLE check if that's added

        # Determine the specific directory or S3 key prefix for *this method's plots*
        # This will be plot_save_dir_base / "predict_images_plots"
        final_plots_location_prefix: Optional[str] = None # For S3 key or local directory path string

        if plot_save_dir_base:
            # Use PurePath for constructing OS-agnostic "paths" that also work as S3 prefixes
            final_plots_location_prefix = str( (PurePath(plot_save_dir_base) / "predict_images_plots").as_posix() )
            # If it's a local path and not a repo, ensure the directory exists
            if not repository_for_plots and isinstance(plot_save_dir_base, (str, Path)):
                Path(final_plots_location_prefix).mkdir(parents=True, exist_ok=True)

        if generate_lime_plots and not SKIMAGE_AVAILABLE:
            logger.warning("LIME plot generation skipped: scikit-image (for mark_boundaries) is not installed.")
            generate_lime_plots = False
        # LIME_PLOTTER_AVAILABLE check isn't strictly needed here if we are just using mark_boundaries
        # and not re-running LimeImageExplainer.explain_instance.

        run_id_for_log = predictions_output[0].get('run_id',
                                                   "prediction_run") if predictions_output else "prediction_run"
        logger.info(
            f"Plotting {len(predictions_output)} predictions for {run_id_for_log}. LIME plots: {generate_lime_plots}")
        num_images_to_plot = len(predictions_output)
        if num_images_to_plot == 0: return

        cols = min(num_images_to_plot, max_cols)
        rows = (num_images_to_plot + cols - 1) // cols
        img_width_unit = 3.5 if not generate_lime_plots else 4.5
        img_height_unit = 4.0 if not generate_lime_plots else 5.0
        fig_width = cols * img_width_unit;
        fig_height = rows * img_height_unit
        max_fig_height = 45;
        fig_height = min(fig_height, max_fig_height)

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.flatten()
        plotted_images_count = 0

        for i, pred_data in enumerate(predictions_output):
            if i >= len(axes_flat): break
            ax = axes_flat[i];
            ax.axis('off')
            identifier = pred_data.get('identifier')
            image_path_str_for_title = pred_data.get('image_path', str(identifier))

            img_pil_for_display: Optional[Image.Image] = image_pil_map.get(identifier)

            try:
                if img_pil_for_display is None:  # Attempt to load if not in map (e.g. direct call to plotter with paths)
                    if isinstance(identifier, (str, Path)) and Path(identifier).is_file() and Path(identifier).exists():
                        img_pil_for_display = Image.open(identifier).convert('RGB')

                lime_explanation_data = pred_data.get('lime_explanation')
                lime_info = ""
                img_to_show_on_axis = np.array(img_pil_for_display) if img_pil_for_display else None  # Default

                if generate_lime_plots and lime_explanation_data and \
                        isinstance(lime_explanation_data, dict) and \
                        'feature_weights' in lime_explanation_data and \
                        'segments_for_render' in lime_explanation_data and \
                        lime_explanation_data.get('segments_for_render') is not None and \
                        SKIMAGE_AVAILABLE and img_pil_for_display is not None:
                    try:
                        img_np_original = np.array(img_pil_for_display)  # For LIME base
                        segments = np.array(lime_explanation_data['segments_for_render'])
                        weights = dict(lime_explanation_data['feature_weights'])

                        lime_mask = np.zeros(segments.shape, dtype=bool)
                        num_features_lime = lime_explanation_data.get('num_features_from_lime_run',
                                                                      lime_num_features_to_display)

                        positive_features = sorted([(seg_id, w) for seg_id, w in weights.items() if w > 0],
                                                   key=lambda x: x[1], reverse=True)

                        features_shown_count = 0
                        for seg_id, weight_val in positive_features:  # Renamed weight to avoid conflict
                            if features_shown_count < num_features_lime:
                                lime_mask[segments == seg_id] = True
                                features_shown_count += 1
                            else:
                                break

                        if np.any(lime_mask):
                            img_norm_for_lime_plot = img_np_original.astype(
                                float) / 255.0 if img_np_original.max() > 1.0 else img_np_original.astype(float)
                            img_to_show_on_axis = mark_boundaries(img_norm_for_lime_plot, lime_mask, color=(1, 0, 0),
                                                                  mode='thick', outline_color=(1, 0, 0))
                            if img_to_show_on_axis.max() <= 1.0 and img_to_show_on_axis.dtype == np.float64:  # mark_boundaries returns float [0,1]
                                img_to_show_on_axis = (img_to_show_on_axis * 255).astype(np.uint8)
                            lime_info = f"\nLIME: {lime_explanation_data.get('explained_class_name', 'N/A')}"
                        else:
                            logger.debug(f"LIME: No positive features for {identifier}, showing original.")
                            lime_info = "\nLIME: (No positive features)"
                    except Exception as e_lime_render:
                        logger.warning(
                            f"Could not render LIME overlay for {identifier}: {e_lime_render}. Showing original.")
                        lime_info = "\n(LIME render error)"
                elif img_pil_for_display:  # If not plotting LIME, or if LIME failed before render
                    lime_info = "\n(LIME data absent)" if generate_lime_plots and not lime_explanation_data else \
                        ("\n(LIME not generated)" if generate_lime_plots else "")
                # else img_to_show_on_axis remains None or original

                if img_to_show_on_axis is not None:
                    ax.imshow(img_to_show_on_axis)
                else:
                    err_msg = pred_data.get('lime_explanation', {}).get('error',
                                                                        "Preview N/A") if generate_lime_plots else "Preview N/A"
                    ax.text(0.5, 0.5, f"Image ID: {identifier}\n({err_msg})", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color='gray')

                title_str = f"Pred: {pred_data['predicted_class_name']} (Conf: {pred_data['confidence']:.2f})"
                top_k = pred_data.get('top_k_predictions')
                if top_k and isinstance(top_k, list) and len(top_k) > 1:
                    if top_k[1][0] != pred_data['predicted_class_name']:
                        title_str += f"\n2nd: {top_k[1][0]} ({top_k[1][1]:.2f})"
                title_str += lime_info
                ax.set_title(title_str, fontsize=8)
                plotted_images_count += 1

            except Exception as e:
                logger.error(f"Error plotting image for identifier {identifier}: {e}", exc_info=True)
                ax.text(0.5, 0.5, "Error Plotting", ha="center", va="center", transform=ax.transAxes, fontsize=8,
                        color='red')
                ax.set_title(f"{Path(str(image_path_str_for_title)).name}", fontsize=7, color='gray', y=0.95)
            finally:
                ax.axis('off')

        for j in range(plotted_images_count, len(axes_flat)): axes_flat[j].set_visible(False)

        if plotted_images_count == 0:
            logger.warning("No images were successfully prepared for the prediction grid plot.")
            if 'fig' in locals() and fig is not None: plt.close(fig)
            return

        if 'fig' in locals() and fig is not None:
            if plotted_images_count > 0:
                try:
                    fig.tight_layout(pad=1.0, h_pad=2.5, w_pad=1.5)
                except ValueError:
                    logger.warning("Could not apply tight_layout to prediction grid.")

                # Construct the final save path/key for the grid plot
                final_grid_plot_save_key_or_path: Optional[Union[str, Path]] = None
                if final_plots_location_prefix: # This is .../run_id/predict_images_plots
                    final_grid_plot_save_key_or_path = str((PurePath(final_plots_location_prefix) / "image_predictions_grid.png").as_posix())

                _save_figure_or_show(fig, repository_for_plots, final_grid_plot_save_key_or_path, show_plots)
            else:
                plt.close(fig)

    @staticmethod
    def plot_image_batch(batch_tensor: torch.Tensor,
                         title: str = "Image Batch",
                         output_path: Optional[Union[str, Path]] = None, # For saving the plot
                         repository_for_plots: Optional[Any] = None,
                         show_plots: bool = True,
                         mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), # For denormalization
                         std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # For denormalization
                        ):
        """
        Plots a batch of images (PyTorch tensor).
        Assumes images are normalized and denormalizes them for display.

        Args:
            batch_tensor: Tensor of shape (B, C, H, W).
            title: Title for the plot.
            output_path: Optional local path or S3 key (if repo provided) to save the plot.
            repository_for_plots: Optional ArtifactRepository instance.
            show_plots: Whether to display the plot.
            mean: Mean used for normalization (for denormalization).
            std: Standard deviation used for normalization (for denormalization).
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping batch plot.")
            return
        if not isinstance(batch_tensor, torch.Tensor) or batch_tensor.ndim != 4:
            logger.warning(f"Invalid batch_tensor for plotting. Expected (B,C,H,W), got {batch_tensor.shape if hasattr(batch_tensor, 'shape') else type(batch_tensor)}")
            return

        try:
            # Denormalize the images for proper display
            # Create a denormalization transform
            denorm_images = batch_tensor.clone() # Work on a copy
            for i in range(denorm_images.shape[0]): # Iterate over batch
                for c in range(denorm_images.shape[1]): # Iterate over channels
                    denorm_images[i, c] = denorm_images[i, c] * std[c] + mean[c]
            denorm_images = torch.clamp(denorm_images, 0, 1) # Clamp to [0, 1] range

            # Make a grid of images
            grid_img = torchvision.utils.make_grid(denorm_images, nrow=4) # Adjust nrow as needed

            fig, ax = plt.subplots(figsize=(12, max(3, 3 * (batch_tensor.size(0) // 4 + 1 )))) # Adjust figsize
            ax.imshow(grid_img.permute(1, 2, 0).cpu().numpy()) # Convert to HWC for matplotlib
            ax.set_title(title, fontsize=14)
            ax.axis('off')

            _save_figure_or_show(fig, repository_for_plots, output_path, show_plots)

        except Exception as e:
            logger.error(f"Error plotting image batch '{title}': {e}", exc_info=True)
