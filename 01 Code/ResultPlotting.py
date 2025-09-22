import matplotlib.pyplot as plt
import numpy as np
import os
from spicelib.editor.base_editor import format_eng

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


def generate_sweep_plots(params_to_plot, all_meas_results, skippable_branches=None, save_figs=True):
    for param_to_plot in params_to_plot:
        # Collect measurement results for entries where param_to_plot is present in config_changes
        x_axis_vals = []
        meas_dict = {}

        for entry in all_meas_results.values():
            config_changes = entry.get('config_changes', {})
            meas_results = entry.get('meas results', {})
            if param_to_plot in config_changes:
                r_val = config_changes[param_to_plot]
                x_axis_vals.append(r_val)
                for k, v in meas_results.items():
                    if any([skip_branch_str in k for skip_branch_str in skippable_branches]):
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
