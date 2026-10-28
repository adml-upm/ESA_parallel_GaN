"""Script version of the SimGUI flow: edit the settings below and run.
The base config and the run itself come from SimWorkflow, so defaults live in one place
(the JSON files in configs/, or SimWorkflow.build_default_base_config). Works from any CWD.
"""
import copy

import SimWorkflow as W
from ResultPlotting import (generate_sweep_plots, generate_perturbation_heat_map,
                            generate_multiparam_heat_maps, plot_wfm_from_all_runs)

# ---------------------------------------------------------------
# Settings
BASE_CONFIG_FILE = W.CONFIG_DIR / 'base_config_4dev.json'  # None -> W.build_default_base_config(N_DEVICES)
N_DEVICES = 4             # only used when BASE_CONFIG_FILE is None
SIM_PATH = W.DEFAULT_SIM_PATH
SIM_TYPE = 'sweep'        # 'sweep', 'perturbation', 'multiparam'
K_PERTURB = 0.1           # default relative perturbation (p.u.), shown in the heat map title
SAVE_FIGS = False
RESULT_FILE = None        # e.g. W.RESULTS_DIR / 'my_study.pkl' to reopen the results in SimGUI later

# Sweeps (also used by 'multiparam', which needs exactly two): (param, start, stop, points, log_space)
SWEEPS = [
    ('Vgs_th_1', 1.852 - 0.44, 1.852 + 0.44, 10, False),
    # ('R_drain_1', 1e-3, 10e-3, 5, False),
    # ('L_drain_1', 500e-12, 50e-9, 5, True),
    # ('Q_Tj_1', 25, 150, 11, False),
    # ('Vdrv_on_1', 4.5, 5.5, 5, False),
]

# Perturbations: (param, amount, 'rel' | 'abs')
PERTURBATIONS = [
    ('Vgs_th_1', 0.1, 'rel'),
    ('k_gm_factor_1', 0.5, 'rel'),
    ('Vgs_th_k_corner_1', 0.1, 'rel'),
    ('Rds_on_base_1', 0.1, 'rel'),
    ('gm_Tc_1', 0.1, 'rel'),
    ('Rds_on_Tc_1', 0.1, 'rel'),
    ('Vgs_th_Tc_1', 0.1, 'rel'),
    ('Rgate_int_1', 0.1, 'abs'),
    # ('R_drain_1', K_PERTURB, 'rel'),
    # ('L_common_source_1', K_PERTURB, 'rel'),
    # ('Vdrv_on_1', 0.1, 'rel'),
    # ('T_sw_1', 0.1, 'rel'),
]

# ---------------------------------------------------------------
base_config = (W.load_base_config(BASE_CONFIG_FILE) if BASE_CONFIG_FILE
               else W.build_default_base_config(N_DEVICES))
n_devices = int(base_config['N_devices'])

if SIM_TYPE == 'perturbation':
    study = W.Study(sim_type=SIM_TYPE, perturbations=PERTURBATIONS, k_perturb=K_PERTURB)
elif SIM_TYPE in ('sweep', 'multiparam'):
    study = W.Study(sim_type=SIM_TYPE, sweeps=SWEEPS, k_perturb=K_PERTURB)
else:
    raise ValueError("Invalid SIM_TYPE. Choose from 'sweep', 'perturbation', or 'multiparam'.")

result = W.run_study(SIM_PATH, base_config, W.default_saved_traces(n_devices), study,
                     on_progress=lambda done, failed, total: print(f'\rSimulations: {done}/{total} done, '
                                                                  f'{failed} failed', end=''))
print(f"\nSuccessful/Total Simulations: {result.ok}/{result.total}")
print(f"Execution time: {result.elapsed_s:.2f} s (simulating {result.sim_s:.2f} s, "
      f"longest run {result.longest_sim_s:.2f} s, reading results {result.read_s:.2f} s)")
if result.meas_csv:
    print(f"Measurements table: {result.meas_csv}")
if RESULT_FILE:
    W.save_result(result, RESULT_FILE)

# ---------------------------------------------------------------
# Plots. ResultPlotting pops the base run from meas_res, so each call gets its own copy.
varied_runs = {k: v for k, v in result.meas_res.items() if v['config_changes']}  # base run has no label


def plot_edges(explicit_wfms, skip_branches):
    for edge in ('rise', 'fall'):
        plot_wfm_from_all_runs(meas_res=varied_runs, explicit_wfms=explicit_wfms, skip_branches=skip_branches,
                               base_config=base_config, edge=edge, save_figs=SAVE_FIGS)


if SIM_TYPE == 'sweep':
    generate_sweep_plots(params_to_plot=result.params_to_plot, meas_res=copy.deepcopy(result.meas_res),
                         skip_branches=['x3', 'x4'], save_figs=SAVE_FIGS)
    plot_edges(explicit_wfms=['i_drain', 'v_gs'], skip_branches=[])
elif SIM_TYPE == 'perturbation':
    skip = list(range(3, n_devices + 1))
    generate_perturbation_heat_map(params_to_plot=result.params_to_plot, meas_res=copy.deepcopy(result.meas_res),
                                   skip_branches=skip, base_config=base_config, k_perturb=K_PERTURB,
                                   save_figs=True)
    plot_edges(explicit_wfms=[], skip_branches=skip)
elif SIM_TYPE == 'multiparam':
    generate_multiparam_heat_maps(params_to_plot=result.params_to_plot, meas_res=result.meas_res,
                                  explicit_wfms=['e_on', 'e_off', 'cond_ratio'], save_figs=SAVE_FIGS)
    plot_edges(explicit_wfms=['i_drain', 'v_gs'], skip_branches=[])
