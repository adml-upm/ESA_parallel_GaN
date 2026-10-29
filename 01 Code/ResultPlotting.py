# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.ioff()  # Turn off interactive mode
import colorsys
import numpy as np
import os
from spicelib.editor.base_editor import format_eng, scan_eng
import pandas as pd
import seaborn as sns
from SimConfiguration import LtSimConfiguration

labeltexts = {
        'i': 'Current [A]',
        'v': 'Voltage [V]',
        'v_lv': 'Low Voltage [V]',
        't': 'Temperature [C]',
    }

_EPC_SPICE_TO_FRIENDLY = LtSimConfiguration().inverse_epc_model_param_mapping


def friendly_name(param):
    """'k2_1' -> 'Vgs_th_1'. Parameters that are not HEMT model params are returned unchanged."""
    base, _, idx = param.rpartition('_')
    if base in _EPC_SPICE_TO_FRIENDLY and idx.isdigit():
        return f'{_EPC_SPICE_TO_FRIENDLY[base]}_{idx}'
    return param


def short_eng(value):
    """Engineering notation rounded to 3 significant digits, for tick and legend labels: 0.5556 -> '556m'."""
    return format_eng(float(f'{value:.3g}'))


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


def generate_sweep_plots(params_to_plot, meas_res, explicit_wfms=[], skip_branches=None, save_figs=True):
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

        # Sort by x_axis_vals for plotting
        sorted_indices = np.argsort(x_axis_vals)
        x_axis_vals = np.array(x_axis_vals)[sorted_indices]
        meas_keys = list(meas_dict.keys())
        if not not explicit_wfms:  # checks if empty list
            meas_keys = [k for k in meas_keys if '_'.join(k.split('_')[1:]) in explicit_wfms]
            
        meas_matrix = np.array([np.array(meas_dict[k])[sorted_indices] for k in meas_keys]).T

        # Plot measurements with the same unit on the same vertically stacked subplot
        # First match wins, so 'ratio' must come before 'cond' (x1_cond_ratio is dimensionless)
        unit_map = {
            'ratio': ('adim.', 1),
            'current': (labeltexts['i'], 1),
            'cond': (labeltexts['i'], 1),
            '_rms_i': (labeltexts['i'], 1),
            # '_v': (labeltexts['v'], 1),
            '_vds': (labeltexts['v'], 1),
            '_vgs': (labeltexts['v_lv'], 1),
            '_e_': ('nJ', 1e9),
            '_e_on': ('nJ', 1e9),
            '_e_off': ('nJ', 1e9),
            'tj': (labeltexts['t'], 1),
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
        axes[-1].set_xlabel(f'{friendly_name(param_to_plot)}')
        x_ticks = axes[-1].get_xticks()
        x_axis_units = 'Ω'   if param_to_plot.startswith('R_') else 'H' if param_to_plot.startswith('L_') else ''
        x_tick_labels = [format_eng(val)+x_axis_units for val in x_ticks]
        axes[-1].set_xticklabels(x_tick_labels)
        
        plt.suptitle(f'Evolution of Measurements vs {friendly_name(param_to_plot)}')
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
            ax.set_title(f'{friendly_name(param_to_plot)} modified from {base_config[param_to_plot]} -> {config_changes[param_to_plot]}. Shown % Variation vs Base')
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
    plt.figure(figsize=(0.8 * variation_df.shape[1], 0.4 * variation_df.shape[0]))
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
        output_folder = './plots'
        os.makedirs(output_folder, exist_ok=True)
        plot_path = os.path.join(output_folder, 'perturbation_heat_map.png')
        plt.savefig(plot_path)
        plt.close()

def plot_wfm_from_all_runs(meas_res, explicit_wfms=[], skip_branches=None, base_config=None, edge=None, save_figs=True):
    """Plot waveforms from all simulation runs, grouping by waveform names.
    Alternatively, only plot specific waveforms if provided.
    if skip_branches is none and meas_res only has data from a single sim, a single sim can be analyzed
    Args:
        meas_res (dict): Measurement results from simulations.
        explicit_wfms (list, optional): List of specific waveform names to plot. Defaults to [] (plot all).
        skip_branches (list, optional): List of branch identifiers to skip in plotting. Defaults to None.
        edge (str, optional): 'rise' or 'fall' to define time window for plotting. Defaults to full sim length.
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
            grouped_waveforms[wfm_name].append((sim_name, param_changes, wfm_time, wfm_data))

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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols + 1.5, 4 * n_rows), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    # Plot waveforms
    for col, suffix in enumerate(sorted_suffixes):
        for row, wfm_name in enumerate(suffix_groups[suffix]):
            ax = axes[row, col]
            
            if edge == 'rise' and base_config is not None:
                ti = base_config[f"Vdrive_delay_{col+1}"] + base_config[f"T_sw_{col+1}"]
                tf = ti + 100e-9  # 100 ns after rising starts
            elif edge == 'fall':
                list_of_key = [f"Vdrive_delay_{col+1}", f"T_sw_{col+1}", f"tdrv_rise_{col+1}"]
                ti = sum([base_config[k] for k in list_of_key]) + base_config[f"D_on_{col+1}"]*base_config[f"T_sw_{col+1}"]
                tf = ti + 100e-9  # 100 ns after falling starts
            else:
                pass  # Full scale
                
            y_min, y_max = float('inf'), float('-inf')
            for sim_name, param_changes, wfm_time, wfm_data in grouped_waveforms[wfm_name]:
                # All changed params, so 2-param runs show both values; the base run has none
                changes_str = ', '.join(f"{friendly_name(k)}={short_eng(v)}" for k, v in param_changes.items())
                plot_label = f"Sim: {sim_name.split('_')[-1]} {changes_str or 'base'}"
                ax.plot(np.array(wfm_time), wfm_data, label=plot_label)
                y_min = min(min(np.array(wfm_data)[(np.array(wfm_time) >= ti) & (np.array(wfm_time) <= tf)]), y_min)
                y_max = max(max(np.array(wfm_data)[(np.array(wfm_time) >= ti) & (np.array(wfm_time) <= tf)]), y_max)
            ax.set_xlim(ti if ti is not None else None, tf if tf is not None else None)
            if row == n_rows - 1:
                ax.set(title=wfm_name, xlabel='Time [ns]')  # Only set xlabel for bottom row
            else:
                ax.set(title=wfm_name)
            ax.set_xticks(ax.get_xticks())  # Keep the same tick positions
            ax.set_xticklabels([round((tick - ax.get_xticks()[0]) * 1e9, 1) for tick in ax.get_xticks()])  # Convert to ns and subtract offset
            
            ax.grid(True)
            ax.set_ylabel(labeltexts[wfm_name[0].lower()])
            ax.set_ylim(y_min-0.05*(y_max-y_min), y_max+0.05*(y_max-y_min))
            _handles, labels = ax.get_legend_handles_labels()
            if len(labels) <= 5:
                ax.legend(loc='best')
            else:
                # Place no individual legends to avoid clutter
                pass

    # Hide unused subplots
    for row in range(n_rows):
        for col in range(n_cols):
            if col >= len(sorted_suffixes) or row >= len(suffix_groups[sorted_suffixes[col]]):
                axes[row][col].axis('off')

    # Global legend on the right, adjust subplot area to fit
    # Shrink subplots to make space for legend and suptitle
    plt.subplots_adjust(right=0.78, top=0.95)  # Reserve space for legend and suptitle
    handles, labels = fig.axes[0].get_legend_handles_labels()
    # Place legend outside the plot area, centered in the reserved space
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.80, 0.5), borderaxespad=0.)
    
    # Set overall title: What waveforms are being shown?
    unique_keys = set('_'.join(key.split('_')[:-1]) for key in grouped_waveforms.keys())
    if len(unique_keys) < 4:
        title_wfms = "-".join(unique_keys)
    else:
        title_wfms = "Several Waveforms"
    num_sims = len(meas_res)  # Nº simulations shown    
    edge_str = f': {edge} edge zoom' if edge else ''
    figure_title = f'{title_wfms} from {num_sims} Simulation{"s" if num_sims != 1 else ""}{edge_str}'
    fig.suptitle(figure_title, y=0.97)  # Place suptitle just above the subplots

    # Only call tight_layout once, and leave space for legend and suptitle
    plt.tight_layout(rect=[0, 0, 0.78, 0.95])

    if not save_figs:
        plt.show()
    else:
        output_folder = './plots'
        os.makedirs(output_folder, exist_ok=True)
        plot_path = os.path.join(output_folder, f'{figure_title.replace(" ", "_").replace(":", "")}.png')
        plt.savefig(plot_path)
        plt.close()

def _heat_map_coord_formatter(grid, x_label, y_label):
    """Hover text for a seaborn heat map: its axes run over cell indices (categorical ticks), so the
    default readout is empty. Map the cursor back to the swept values and the cell's value."""
    def format_coord(x, y):
        col, row = int(np.floor(x)), int(np.floor(y))
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return ''
        value = grid.iat[row, col]
        value_str = 'no data' if pd.isna(value) else f'{value:.4g}'
        return f'{x_label}={short_eng(grid.columns[col])}, {y_label}={short_eng(grid.index[row])}: {value_str}'
    return format_coord

def generate_multiparam_heat_maps(params_to_plot, meas_res, explicit_wfms=[], skip_branches=None,
                                  swap_axes=False, save_figs=True):
    """2-param sweep: one figure per measurement, with one heat map per device over the grid of the
    two swept parameters (first one on x, unless swap_axes). The devices of a figure share the colour
    scale so they can be compared."""
    if len(params_to_plot) != 2:
        raise ValueError("Select exactly two swept parameters in params_to_plot")
    p_x, p_y = reversed(params_to_plot) if swap_axes else params_to_plot

    rows = []
    for entry in meas_res.values():
        config_changes = entry.get('config_changes', {})
        if p_x not in config_changes or p_y not in config_changes:
            continue  # base run
        for meas_key, meas_val in entry.get('meas results', {}).items():
            device, _, meas_name = meas_key.partition('_')  # 'x1_e_on' -> 'x1', 'e_on'
            if any(skip_branch_str in meas_key for skip_branch_str in (skip_branches or [])):
                continue
            if explicit_wfms and meas_name not in explicit_wfms:
                continue
            rows.append({'device': device, 'meas': meas_name, 'x': config_changes[p_x],
                         'y': config_changes[p_y], 'value': meas_val})
    if not rows:
        raise ValueError(f"No measurements found for runs that vary both {p_x} and {p_y}")
    df = pd.DataFrame(rows)

    for meas_name in sorted(df['meas'].unique()):
        meas_df = df[df['meas'] == meas_name]
        devices = sorted(meas_df['device'].unique(), key=lambda d: int(d[1:]) if d[1:].isdigit() else 0)
        vmin, vmax = meas_df['value'].min(), meas_df['value'].max()
        fig, axes = plt.subplots(1, len(devices), figsize=(4.5 * len(devices) + 1, 4), squeeze=False)
        for ax, device in zip(axes[0], devices):
            grid = meas_df[meas_df['device'] == device].pivot_table(index='y', columns='x', values='value')
            grid = grid.sort_index(ascending=False)  # largest y value on top
            annot = grid.map(lambda v: '' if pd.isna(v) else short_eng(v)) if grid.size <= 64 else False
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax, cmap='viridis', annot=annot, fmt='',
                        xticklabels=[short_eng(v) for v in grid.columns],
                        yticklabels=[short_eng(v) for v in grid.index],
                        cbar=device == devices[-1])
            ax.set(title=f'{device}_{meas_name}', xlabel=friendly_name(p_x), ylabel=friendly_name(p_y))
            ax.format_coord = _heat_map_coord_formatter(grid, friendly_name(p_x), friendly_name(p_y))
        fig.suptitle(f'{meas_name} over {friendly_name(p_x)} x {friendly_name(p_y)}')
        plt.tight_layout()

        if not save_figs:
            plt.show()
        else:
            output_folder = './plots'
            os.makedirs(output_folder, exist_ok=True)
            plot_path = os.path.join(output_folder, f'multiparam_{meas_name}_{p_x}_vs_{p_y}.png')
            plt.savefig(plot_path)
            plt.close()

def _edge_window(base_config, dev, edge):
    """(t_start, t_window, t_scale, t_unit) for a 100 ns zoom on the rise/fall edge of the 2nd period of
    device dev, or the full simulation in µs when edge is None or base_config is missing."""
    if edge in ('rise', 'fall') and base_config is not None:
        t_start = base_config[f"Vdrive_delay_{dev}"] + base_config[f"T_sw_{dev}"]
        if edge == 'fall':
            t_start += base_config[f"tdrv_rise_{dev}"] + base_config[f"D_on_{dev}"] * base_config[f"T_sw_{dev}"]
        return t_start, (t_start, t_start + 100e-9), 1e9, 'ns'
    return 0.0, None, 1e6, 'µs'

def _stepped_colorbar(fig, axes, colors, values, label):
    """Colour bar with one step per value of a swept parameter (values sorted, colors in the same order)."""
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    norm = plt.matplotlib.colors.BoundaryNorm(np.arange(len(values) + 1), cmap.N)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label=label,
                        ticks=np.arange(len(values)) + 0.5)
    cbar.ax.set_yticklabels([short_eng(v) for v in values])

def _select_2param_waveforms(params_to_plot, meas_res, explicit_wfms, skip_branches, swap_axes):
    """Runs of a 2-param grid (base run left out) and what to plot of them: the two params (swapped if
    asked), the sorted values of each, the device numbers and the waveform types ('i_drain', ...)."""
    if len(params_to_plot) != 2:
        raise ValueError("Select exactly two swept parameters in params_to_plot")
    p_a, p_b = reversed(params_to_plot) if swap_axes else params_to_plot
    runs = [e for e in meas_res.values()
            if p_a in e.get('config_changes', {}) and p_b in e.get('config_changes', {})]
    if not runs:
        raise ValueError(f"No runs vary both {p_a} and {p_b}")
    a_vals = sorted({e['config_changes'][p_a] for e in runs})
    b_vals = sorted({e['config_changes'][p_b] for e in runs})

    # Waveform names like 'i_drain_3'
    wfm_names = [k for k in runs[0].get('waveforms', {}) if k != 'time']
    if explicit_wfms:
        wfm_names = [k for k in wfm_names if k.rsplit('_', 1)[0] in explicit_wfms]
    devices = sorted({int(k.rsplit('_', 1)[1]) for k in wfm_names} - set(skip_branches or []))
    wfm_types = sorted({k.rsplit('_', 1)[0] for k in wfm_names})
    if not devices or not wfm_types:
        raise ValueError("No waveforms left to plot")
    return p_a, p_b, runs, a_vals, b_vals, devices, wfm_types

def _connect_2param_highlight(fig, axes, key_ax, run_lines, colors, hue_vals, light_vals, p_hue, p_light):
    """Click a curve or a colour key swatch to highlight that run in every subplot and grey out the rest;
    click the same run again, or empty plot space, to restore all colours."""
    frame = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec='black', lw=2.5, visible=False)
    key_ax.add_patch(frame)
    plot_axes = set(axes.flat)
    selected = [None]

    def select(cell):
        selected[0] = None if cell == selected[0] else cell
        for other_cell, lines in run_lines.items():
            for line in lines:
                if selected[0] is None:
                    line.set(color=colors[other_cell], lw=1.3, alpha=0.9, zorder=2)
                elif other_cell == selected[0]:
                    line.set(color=colors[other_cell], lw=2.5, alpha=1.0, zorder=3)
                else:
                    line.set(color='0.75', lw=1.0, alpha=0.35, zorder=2)
        frame.set_visible(selected[0] is not None)
        if selected[0] is not None:
            light_i, hue_i = selected[0]
            frame.set_xy((hue_i - 0.5, light_i - 0.5))
            key_ax.set_title(f'{friendly_name(p_hue)}={short_eng(hue_vals[hue_i])}\n'
                             f'{friendly_name(p_light)}={short_eng(light_vals[light_i])}', fontsize=9)
        else:
            key_ax.set_title('Colour key')
        fig.canvas.draw_idle()

    def on_click(event):
        toolbar = getattr(fig.canvas, 'toolbar', None)
        if event.button != 1 or (toolbar is not None and toolbar.mode):
            return  # leave zoom / pan clicks alone
        if event.inaxes is key_ax and event.xdata is not None:
            cell = (int(round(event.ydata)), int(round(event.xdata)))
            if cell in run_lines:
                select(cell)
        elif event.inaxes in plot_axes:
            # Topmost curve under the cursor: the highlighted one if it was hit, else the last drawn
            hits = [cell for cell, lines in run_lines.items() for line in lines
                    if line.axes is event.inaxes and line.contains(event)[0]]
            if selected[0] in hits:
                select(selected[0])
            elif hits:
                select(hits[-1])
            elif selected[0] is not None:
                select(selected[0])  # empty space: clear

    fig.canvas.mpl_connect('button_press_event', on_click)

def plot_wfm_2param(params_to_plot, meas_res, explicit_wfms=[], skip_branches=None, base_config=None,
                    edge=None, swap_axes=False, save_figs=True):
    """2-param sweep waveforms: one subplot per device (column) and waveform (row). Line hue encodes the
    first swept parameter and lightness the second (dark = smallest value), read off a 2D colour key, so
    even a 10x10 grid needs no per-run legend. swap_axes exchanges the two encodings.
    skip_branches: device numbers to leave out (ints). edge: 'rise' / 'fall' zooms 100 ns on that edge
    of the 2nd period (needs base_config), None shows the full simulation."""
    p_hue, p_light, runs, hue_vals, light_vals, devices, wfm_types = _select_2param_waveforms(
        params_to_plot, meas_res, explicit_wfms, skip_branches, swap_axes)
    # Hue and lightness go by rank, so log-spaced sweeps are spread evenly too. Hues stop short of the
    # full circle so the last one doesn't wrap back to red.
    hues = np.linspace(0, 0.8, len(hue_vals))
    lights = np.linspace(0.22, 0.72, len(light_vals)) if len(light_vals) > 1 else [0.45]
    colors = np.array([[colorsys.hls_to_rgb(h, l, 0.85) for h in hues] for l in lights])  # [light, hue, rgb]

    # Rows by waveform, columns by device; the colour key gets its own narrow subfigure
    fig = plt.figure(figsize=(4.5 * len(devices) + 3, 3.2 * len(wfm_types) + 0.8), layout='constrained')
    plot_fig, key_fig = fig.subfigures(1, 2, width_ratios=[4.5 * len(devices), 3])
    axes = plot_fig.subplots(len(wfm_types), len(devices), sharex=True, squeeze=False)
    run_lines = {}  # (light index, hue index) -> its line in every subplot, for click highlighting
    for col, dev in enumerate(devices):
        t_start, t_window, t_scale, t_unit = _edge_window(base_config, dev, edge)
        for row, wfm_type in enumerate(wfm_types):
            ax = axes[row, col]
            name = f'{wfm_type}_{dev}'
            y_min, y_max = float('inf'), float('-inf')
            for entry in runs:
                t = np.asarray(entry['waveforms']['time'])
                y = np.asarray(entry['waveforms'][name])
                changes = entry['config_changes']
                cell = (light_vals.index(changes[p_light]), hue_vals.index(changes[p_hue]))
                line, = ax.plot((t - t_start) * t_scale, y, color=colors[cell], lw=1.3, alpha=0.9, pickradius=4)
                run_lines.setdefault(cell, []).append(line)
                in_window = (t >= t_window[0]) & (t <= t_window[1]) if t_window else slice(None)
                if y[in_window].size:
                    y_min, y_max = min(y_min, y[in_window].min()), max(y_max, y[in_window].max())
            if t_window:
                ax.set_xlim(0, (t_window[1] - t_window[0]) * t_scale)
                if y_max > y_min:
                    ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
            ax.set_title(name)
            ax.set_ylabel(labeltexts.get(wfm_type[0].lower(), ''))
            ax.grid(True)
            if row == len(wfm_types) - 1:
                ax.set_xlabel(f'Time [{t_unit}]' + (f' from {edge} edge' if t_window else ''))

    # 2D colour key: one swatch per (hue, lightness) pair, smallest lightness value on top
    key_ax = key_fig.subplots()
    key_ax.imshow(colors, aspect='equal')
    key_ax.set_xticks(range(len(hue_vals)), [short_eng(v) for v in hue_vals], rotation=90, fontsize=8)
    key_ax.set_yticks(range(len(light_vals)), [short_eng(v) for v in light_vals], fontsize=8)
    key_ax.set(xlabel=f'Hue: {friendly_name(p_hue)}', ylabel=f'Lightness: {friendly_name(p_light)}',
               title='Colour key')
    _connect_2param_highlight(fig, axes, key_ax, run_lines, colors, hue_vals, light_vals, p_hue, p_light)
    title = (f'{", ".join(wfm_types)} over {friendly_name(p_hue)} x {friendly_name(p_light)}'
             f'{f": {edge} edge zoom" if t_window else ""}')
    fig.suptitle(title)

    if not save_figs:
        plt.show()
    else:
        output_folder = './plots'
        os.makedirs(output_folder, exist_ok=True)
        plot_path = os.path.join(output_folder, f'wfm_2param_{p_hue}_{p_light}_{edge or "full"}.png')
        plt.savefig(plot_path)
        plt.close()

def plot_wfm_2param_3d(params_to_plot, meas_res, explicit_wfms=[], skip_branches=None, base_config=None,
                       edge=None, swap_axes=False, fill=True, save_figs=True):
    """2-param sweep waveforms as a 3D waterfall: one figure per waveform, one 3D subplot per device.
    x = time, z = waveform value, depth (y) = second swept param, colour = first param (colour bar).
    fill shades a translucent ribbon under each curve so the depth slices read as surfaces.
    swap_axes exchanges the two params. skip_branches / edge as in plot_wfm_2param."""
    p_color, p_depth, runs, color_vals, depth_vals, devices, wfm_types = _select_2param_waveforms(
        params_to_plot, meas_res, explicit_wfms, skip_branches, swap_axes)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(color_vals)))

    n_cols = int(np.ceil(np.sqrt(len(devices))))  # near-square grid: 3D panels need the room
    n_rows = -(-len(devices) // n_cols)
    for wfm_type in wfm_types:
        # No constrained layout: it collapses 3D axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols + 1.5, 5.5 * n_rows), squeeze=False,
                                 subplot_kw={'projection': '3d'})
        for ax in axes.flat[len(devices):]:
            ax.remove()
        for ax, dev in zip(axes.flat, devices):
            t_start, t_window, t_scale, t_unit = _edge_window(base_config, dev, edge)
            name = f'{wfm_type}_{dev}'
            curves = []
            for entry in runs:
                t = np.asarray(entry['waveforms']['time'])
                y = np.asarray(entry['waveforms'][name])
                if t_window:  # 3D axes don't clip to their limits, so crop the data instead
                    in_window = (t >= t_window[0]) & (t <= t_window[1])
                    t, y = t[in_window], y[in_window]
                if t.size:
                    changes = entry['config_changes']
                    curves.append(((t - t_start) * t_scale, y, depth_vals.index(changes[p_depth]),
                                   colors[color_vals.index(changes[p_color])]))
            if not curves:
                continue
            # Depth by rank, so log-spaced sweeps are spread evenly too. Ribbons go down to 0 (or the
            # nearest data limit) and stay faint, since a depth slice stacks one ribbon per colour.
            z_min, z_max = min(y.min() for _, y, _, _ in curves), max(y.max() for _, y, _, _ in curves)
            z_base = min(max(0.0, z_min), z_max)
            for x, y, depth, color in curves:
                if fill:
                    ax.fill_between(x, depth, y, x, depth, z_base, color=color, alpha=0.6 / len(color_vals),
                                    linewidth=0)
                ax.plot(x, np.full_like(x, depth), y, color=color, lw=1.1, alpha=0.9)
            ax.set_yticks(range(len(depth_vals)), [short_eng(v) for v in depth_vals], fontsize=7)
            ax.set(title=name, xlabel=f'Time [{t_unit}]' + (f' from {edge} edge' if t_window else ''),
                   ylabel=friendly_name(p_depth), zlabel=labeltexts.get(wfm_type[0].lower(), ''))
            ax.set_box_aspect((2, 1.3, 1), zoom=0.9)
            ax.view_init(elev=22, azim=-55)
        _stepped_colorbar(fig, list(axes.flat[:len(devices)]), colors, color_vals, friendly_name(p_color))
        fig.suptitle(f'{wfm_type} over {friendly_name(p_color)} (colour) x {friendly_name(p_depth)} (depth)'
                     f'{f": {edge} edge zoom" if edge in ("rise", "fall") and base_config is not None else ""}')

        if not save_figs:
            plt.show()
        else:
            output_folder = './plots'
            os.makedirs(output_folder, exist_ok=True)
            plot_path = os.path.join(output_folder, f'wfm_2param_3d_{wfm_type}_{p_color}_{p_depth}_{edge or "full"}.png')
            plt.savefig(plot_path)
            plt.close()
