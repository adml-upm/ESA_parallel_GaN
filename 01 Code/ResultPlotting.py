# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.ioff()  # Turn off interactive mode
import numpy as np
import os
from spicelib.editor.base_editor import format_eng, scan_eng
import pandas as pd
import seaborn as sns
from SimConfiguration import LtSimConfiguration

def detect_spacing(values, tol=1e-12):
    arr = np.array(values, dtype=float)
    diffs = np.diff(arr)
    ratios = arr[1:] / arr[:-1]
    if np.allclose(diffs, diffs[0], rtol=tol, atol=tol):
        return "linear"  # Linear spacing: diff should be ~constant  
    elif np.allclose(ratios, ratios[0], rtol=tol, atol=tol):
        return "log"  # Log spacing: ratios should be ~constant
    else:
        return "linear" 


def generate_sweep_plots(params_to_plot, meas_res, skip_branches=None, save_figs=True):
    # Remove first sim of all (base case) if present
    meas_res.pop(sorted(meas_res.keys())[0], None)

    for param_to_plot in params_to_plot:
        # Collect measurement results for entries where param_to_plot is present in config_changes
        x_axis_vals = []
        meas_dict = {}

        for entry in meas_res.values():
            config_changes = entry.get('config_changes', {})
            meas_results = entry.get('meas results', {})
            if param_to_plot in config_changes:
                r_val = config_changes[param_to_plot]
                x_axis_vals.append(r_val)
                for k, v in meas_results.items():
                    if any([skip_branch_str in k for skip_branch_str in skip_branches]):
                        continue  # Skip measurements related to this HEMT branch
                    if k not in meas_dict:
                        meas_dict[k] = []
                    meas_dict[k].append(v)

        # Sort by r_drain_vals for plotting
        sorted_indices = np.argsort(x_axis_vals)
        x_axis_vals = np.array(x_axis_vals)[sorted_indices]
        meas_keys = list(meas_dict.keys())
        meas_matrix = np.array([np.array(meas_dict[k])[sorted_indices] for k in meas_keys]).T

        # Plot measurements with the same unit on the same vertically stacked subplot
        unit_map = {
            'current': ('A', 1),
            '_current': ('A', 1),
            '_v': ('V', 1),
            '_vds': ('V', 1),
            '_vgs': ('V', 1),
            '_e_': ('nJ', 1e9),
            '_e_on': ('nJ', 1e9),
            '_e_off': ('nJ', 1e9),
        }

        # Group keys by unit and coefficient
        unit_groups = {}
        for key in meas_keys:
            found = False
            for k, (unit, coef) in unit_map.items():
                if k in key:
                    unit_coef = (unit, coef)
                    found = True
                    break
            if not found:
                unit_coef = ('Value', 1)
            unit_groups.setdefault(unit_coef, []).append(key)

        n_units = len(unit_groups)
        fig, axes = plt.subplots(n_units, 1, figsize=(10, 3 * n_units), sharex=True)
        if n_units == 1:
            axes = [axes]

        for ax, (unit_coef, keys) in zip(axes, unit_groups.items()):
            unit, coef = unit_coef[0], unit_coef[1]
            for key in keys:
                idx = meas_keys.index(key)
                ax.plot(x_axis_vals, coef * meas_matrix[:, idx], marker='o', label=key)
            ax.set_ylabel(f'[{unit}]')
            ax.set_xscale(detect_spacing(x_axis_vals))
            ax.legend(loc='best')
            ax.grid(True)
        
        # Format x-axis ticks to engineering notation - do not break up this secti
        axes[-1].set_xlabel(f'{param_to_plot}')
        x_ticks = axes[-1].get_xticks()
        x_axis_units = 'Ω' if param_to_plot.startswith('R_') else 'H' if param_to_plot.startswith('L_') else ''
        x_tick_labels = [format_eng(val)+x_axis_units for val in x_ticks]
        axes[-1].set_xticklabels(x_tick_labels)
        
        plt.suptitle(f'Evolution of Measurements vs {param_to_plot}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if not save_figs:
            plt.show()
        else:
            output_folder = './plots'
            os.makedirs(output_folder, exist_ok=True)

            plot_path = os.path.join(output_folder, f'meas_evolution_vs_{param_to_plot}.png')
            plt.savefig(plot_path)
            plt.close()  # Recommended for memory management when generating many plots

def generate_perturbation_barplots(params_to_plot, meas_res, skip_branches=None, base_config=None, save_figs=True):
    # Remove first sim of all (base case) if present
    base_meas_res = meas_res.pop(sorted(meas_res.keys())[0], None)

    for param_to_plot in params_to_plot:
        # Collect measurement results for entries where param_to_plot is present in config_changes
        for entry_key, entry in meas_res.items():
            config_changes = entry.get('config_changes', {})
            meas_results = entry.get('meas results', {})
            if param_to_plot not in config_changes:
                continue

            # Prepare data for bar plot
            categories = []
            percent_variations = []
            base_values = []
            for meas_key, meas_val in meas_results.items():
                if any([skip_branch_str in meas_key for skip_branch_str in (skip_branches or [])]):
                    continue
                base_val = base_meas_res.get('meas results', {}).get(meas_key, None)
                if base_val is None or base_val == 0:
                    continue  # skip if no base value or base is zero (avoid div by zero)
                percent_var = 100 * (meas_val - base_val) / base_val
                categories.append(meas_key)
                percent_variations.append(percent_var)
                base_values.append(base_val)

            if not categories:
                continue  # nothing to plot

            fig, ax = plt.subplots(figsize=(max(6, len(categories)*1.2), 5))
            bars = ax.bar(categories, percent_variations, color='skyblue', edgecolor='k')

            # Annotate base values on top of bars
            for bar, base_val in zip(bars, base_values):
                height = bar.get_height()
                ax.annotate(f'{format_eng(base_val)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5 if height >= 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=9, color='black', rotation=0)

            ax.set_ylabel('% Variation vs Base')
            ax.set_title(f'{param_to_plot} modified from {base_config[param_to_plot]} -> {config_changes[param_to_plot]}. Shown % Variation vs Base')
            ax.axhline(0, color='gray', linewidth=1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if not save_figs:
                plt.show()
            else:
                output_folder = './plots'
                os.makedirs(output_folder, exist_ok=True)
                plot_path = os.path.join(
                    output_folder,
                    f'perturbation_{param_to_plot}_{str(config_changes[param_to_plot]).replace(".","p")}_{entry_key}.png'
                )
                plt.savefig(plot_path)
                plt.close()

def generate_perturbation_heat_map(params_to_plot, meas_res, skip_branches=None, base_config=None, 
                                   k_perturb: float | None=None, save_figs=True):
    # Remove first sim of all (base case) if present
    base_meas_res = meas_res.pop(sorted(meas_res.keys())[0], None)

    variational_dict = {}
    for param_to_plot in params_to_plot:
        if param_to_plot not in variational_dict:
            variational_dict[param_to_plot] = {}

        # Collect measurement results for entries where param_to_plot is present in config_changes
        for sim_file_name, sim_dict in meas_res.items():
            if param_to_plot not in sim_dict['config_changes'].keys():
                continue

            for meas_key, meas_val in sim_dict['meas results'].items():
                if any([str(skip_int) in meas_key for skip_int in skip_branches]):
                    continue
                base_val = base_meas_res['meas results'][meas_key]
                try:
                    percent_var = 100 * (meas_val - base_val) / base_val
                except ZeroDivisionError:
                    continue
                variational_dict[param_to_plot][meas_key] = percent_var
    
    epc_map = LtSimConfiguration().inverse_epc_model_param_mapping
    # Rename rows and columns for better readability
    renamed_variational_dict = {'_'.join([epc_map[k.rsplit('_', 1)[0]],k.rsplit('_', 1)[-1]]): v for k, v in variational_dict.items()}
    

    # Build DataFrame at the end
    variation_df = pd.DataFrame.from_dict(renamed_variational_dict, orient="index")
    print(variation_df.head)

    # Plot seaborn heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        variation_df,
        annot=True, fmt=".2f", cmap="coolwarm", center=0,
        cbar_kws={'label': '% Variation vs Base'}
    )
    
    plt.title(f"Sensitivity Heatmap: Percent variation for K_perturbation={str(100*k_perturb)}%")
    plt.ylabel(f"Modified Variable")
    plt.xlabel("Measured Metric")
    plt.tight_layout()

    if not save_figs:
        plt.show()
    else:
        output_folder = os.path.dirname(os.path.dirname(__file__))
        # os.makedirs(output_folder, exist_ok=True)

        plt.savefig(os.path.join(output_folder, 'plots', 'perturbation_heat_map.png'))
        plt.close()

def plot_wfm_from_all_runs(meas_res, explicit_wfms=[], skip_branches=None, base_config=None, edge=None, save_figs=True):
    """Plot waveforms from all simulation runs, grouping by waveform names.
    Alternatively, only plot specific waveforms if provided.
    if skip_branches is none and meas_res only has data from a single sim, a single sim can be analyzed
    Args:
        meas_res (dict): Measurement results from simulations.
        explicit_wfms (list, optional): List of specific waveform names to plot. Defaults to [] (plot all).
        skip_branches (list, optional): List of branch identifiers to skip in plotting. Defaults to None.
        time_scale (list, optional): Time scale [start, end] in seconds for x-axis. Defaults to [] (full scale).
        save_figs (bool, optional): Whether to save figures or show them. Defaults to True.
    """
    # Collect all waveforms and group them by common keys
    grouped_waveforms = {}
    for sim_name, sim_meas in meas_res.items():
        sim_waveforms = sim_meas.get('waveforms', {})
        param_changes = sim_meas.get('config_changes', {})
        wfm_time = sim_waveforms.get('time', [])
        for wfm_name, wfm_data in sim_waveforms.items():
            if not not explicit_wfms:  # checks if empty list
                if '_'.join(wfm_name.split('_')[:-1]) not in explicit_wfms:
                    continue  # Only plot specified waveforms if provided

            if wfm_name == 'time':
                continue  # Skip time waveform, previously extracted

            if wfm_name not in grouped_waveforms:
                grouped_waveforms[wfm_name] = []
            grouped_waveforms[wfm_name].append((sim_name, wfm_time, wfm_data))


    # Filter out waveforms based on skip_branches
    if skip_branches:
        grouped_waveforms = {
            k: v for k, v in grouped_waveforms.items()
            if not any(k.endswith(f'_{branch}') for branch in skip_branches)
        }

    # Group waveforms by suffix and sort
    suffix_groups = {
        suffix: sorted([wfm for wfm in grouped_waveforms if wfm.endswith(f'_{suffix}')])
        for suffix in sorted(set(wfm.split('_')[-1] for wfm in grouped_waveforms))
    }
    sorted_suffixes = sorted(suffix_groups, key=lambda x: int(x) if x.isdigit() else x)

    # Determine subplot layout
    n_cols, n_rows = len(sorted_suffixes), max(len(suffix_groups[suffix]) for suffix in sorted_suffixes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)

    # Plot waveforms
    for col, suffix in enumerate(sorted_suffixes):
        for row, wfm_name in enumerate(suffix_groups[suffix]):
            ax = axes[row, col]
            for sim_name, wfm_time, wfm_data in grouped_waveforms[wfm_name]:
                ax.plot(np.array(wfm_time) * 1e6, wfm_data, label=f'Sim: {sim_name}')

                if edge == 'rise' and base_config is not None:
                    ti = base_config[f"Vdrive_delay_{col+1}"] + base_config[f"T_sw_{col+1}"]
                    tf = ti + 100e-9  # 100 ns after rising starts
                    ax.set_xlim(ti * 1e6, tf * 1e6)
                elif edge == 'fall':
                    list_of_key = [f"Vdrive_delay_{col+1}", f"T_sw_{col+1}", f"T_on_{col+1}", f"tdrv_rise_{col+1}"]
                    ti = sum([base_config[k] for k in list_of_key])
                    tf = ti + 50e-9  # 100 ns after falling starts
                    ax.set_xlim(ti * 1e6, tf * 1e6)
                else:
                    pass  # Full scale
                
            ax.set(title=wfm_name, xlabel='Time [µs]', ylabel='Amplitude')
            ax.legend(loc='best')
            ax.grid(True)

    # Hide unused subplots
    for row in range(n_rows):
        for col in range(n_cols):
            if col >= len(sorted_suffixes) or row >= len(suffix_groups[sorted_suffixes[col]]):
                axes[row][col].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Waveforms from {"All" if len(meas_res) > 1 else "Selected"} Simulations', y=1.02)

    if not save_figs:
        plt.show()
    else:
        output_folder = './plots'
        os.makedirs(output_folder, exist_ok=True)
        for wfm_name in grouped_waveforms.keys():
            plot_path = os.path.join(output_folder, f'wfm_{wfm_name}_all_runs.png')
            plt.savefig(plot_path)
        plt.close()

def calculate_time_limits_for_plots(meas_res, base_config, edge='rise'):
    """Calculate appropriate time limits for waveform plots based on base configuration.

    Args:
        meas_res (dict): Measurement results from simulations.
        base_config (dict): Base configuration dictionary.
        edge (str, optional): Edge type to consider ('rise' or 'fall'). Defaults to 'rise'.
    Returns:
        tuple: (time_start, time_end) in seconds.
    """
