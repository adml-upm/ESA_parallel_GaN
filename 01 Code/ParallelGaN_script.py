from PyLTSpice import SimRunner
from PyLTSpice import AscEditor
from PyLTSpice import RawRead
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import time
from SimConfiguration import LtSimConfiguration
from ResultPlotting import generate_sweep_plots, generate_perturbation_barplots, generate_perturbation_heat_map, plot_wfm_from_all_runs
   

# Force another simulatior
simulator = r"C:\Users\Alex\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

# select spice model
LTC = SimRunner(output_folder='temp', simulator=None, parallel_sims=16, timeout=30)

# sim_path = './02 Simulations/02 ParallelGaN/ParGan_subckt_example.asc'
sim_path = './02 Simulations/02 ParallelGaN/ParGan_V3.asc'
netlist = AscEditor(sim_path)




SimConfig = LtSimConfiguration()
SimConfig.N_devices = 4  # Number of parallel devices to simulate
SimConfig.TOTAL_DRAWN_DEVICES = 10

if  SimConfig.N_devices > SimConfig.TOTAL_DRAWN_DEVICES:
    raise ValueError("N_devices exceeds the number of devices drawn in the netlist")
# Electrical testing setpoint:
m_sweep_points = 5
V_DC = 100  # V
I_DC = 10   # A
f_sw = 100e3
T_sw = 1 / f_sw
D_sw = 0.5
T_on = D_sw * T_sw

# ---------------------------------------------------------------

# Get hemt param baseconfig and add to base_config_dict

hemt_base_config = SimConfig.get_hemt_parameters_from_lib()

base_config_dict = {
    'V_DC': V_DC,
    'I_DC': I_DC
}

for i in range(1, SimConfig.N_devices + 1):
    # HEMT device param
    base_config_dict[f'dev_{i}'] = "EPC2305"  # "use default names from lib"
    
    # HEMT cell params
    base_config_dict[f"R_drain_{i}"] = 1e-3
    base_config_dict[f"L_drain_{i}"] = 1e-9
    base_config_dict[f"R_source_{i}"] = 1e-3
    base_config_dict[f"L_source_{i}"] = 1e-9
    # Parasitic PCB params
    base_config_dict[f"R_pcb_branch_{i}"] = 1e-3
    # Driving circuit params
    base_config_dict[f"Lg_uncommon_{i}"] = 0.5e-9
    base_config_dict[f"L_common_source_{i}"] = 200e-12
    base_config_dict[f"R_driver_{i}"] = 2
    base_config_dict[f"R_g_{i}"] = 1
    # Gate Driver params 
    base_config_dict[f"Vdrv_off_{i}"] = 0
    base_config_dict[f"Vdrv_on_{i}"] = 5
    base_config_dict[f"Vdrive_delay_{i}"] = 1e-9
    base_config_dict[f"tdrv_rise_{i}"] = 1e-9
    base_config_dict[f"tdrv_fall_{i}"] = 1e-9
    base_config_dict[f"T_sw_{i}"] = 1 / 100e3
    base_config_dict[f"D_on_{i}"] = 0.5

    # HEMT model params
    hemt_i_base_config = SimConfig.get_hemt_parameters_from_lib()[base_config_dict[f'dev_{i}']][0]
    base_config_dict.update({k + f'_{i}': v for k, v in hemt_i_base_config.items()})

SimConfig.base_cnfg_dict = base_config_dict.copy()

# Define simulation type and parameter variations
SIM_TYPE = 'perturbation'  # 'sweep', 'perturbation', 'multiparam'
if SIM_TYPE == 'sweep':
    SimConfig.add_param_sweep_with_range('R_drain_1', [1e-3, 10e-3], m_points=m_sweep_points)
    SimConfig.add_param_sweep_with_range('L_drain_1', [500e-12, 50e-9], m_points=m_sweep_points, log_space=True)
    # SimConfig.add_param_sweep_with_range('Vdrv_on_1', [4.5, 5.5], m_points=m_points)
elif SIM_TYPE == 'perturbation':    
    SimConfig.default_Kperturb = 0.1  # Modify standard relative perturbation amount (p.u.)

    # Circuit parameter perturbations
    # SimConfig.add_param_perturbation("R_drain_1")
    # SimConfig.add_param_perturbation("L_drain_1")
    # SimConfig.add_param_perturbation("R_source_1")
    # SimConfig.add_param_perturbation("L_source_1")
    # SimConfig.add_param_perturbation("R_pcb_branch_1")
    # SimConfig.add_param_perturbation("Lg_uncommon_1")
    # SimConfig.add_param_perturbation("L_common_source_1")
    # SimConfig.add_param_perturbation("Vdrv_off_1")
    # SimConfig.add_param_perturbation("Vdrv_on_1", perturb_rel=0.1)
    # SimConfig.add_param_perturbation("Vdrive_delay_1")
    # SimConfig.add_param_perturbation("tdrv_rise_1")
    # SimConfig.add_param_perturbation("tdrv_fall_1")
    # SimConfig.add_param_perturbation("T_sw_1", perturb_rel=0.1)

    # HEMT model perturbations
    # SimConfig.add_param_perturbation('Vgs_th_1', perturb_rel=0.1)
    # SimConfig.add_param_perturbation('k_gm_factor_1', perturb_rel=0.1)
    # SimConfig.add_param_perturbation('Vgs_th_k_corner_1', perturb_rel=0.1)
    # SimConfig.add_param_perturbation('Rds_on_base_1', perturb_rel=0.1)
    # SimConfig.add_param_perturbation('gm_Tc_1', perturb_rel=0.1)
    SimConfig.add_param_perturbation('Rds_on_Tc_1', perturb_rel=0.1)
    SimConfig.add_param_perturbation('Vgs_th_Tc_1', perturb_rel=0.1)
    SimConfig.add_param_perturbation('Rgate_int_1', perturb_abs=0.1)
    # SimConfig.add_param_perturbation("D_on_1")
    # SimConfig.add_param_perturbation("R_driver_1")
    # SimConfig.add_param_perturbation("R_g_1")
    # SimConfig.add_param_perturbation('L_drain_1', 1e-3)
elif SIM_TYPE == 'multiparam':
    raise ValueError("multiparam sim not implemented")
else:
    raise ValueError("Invalid SIM_TYPE. Choose from 'sweep', 'perturbation', or 'multiparam'.")

# ---------------------------------------------------------------


# netlist.add_instructions(
#     f".include {os.path.abspath('./02 Simulations/00 ComonLibs/EPCGaNLibrary.lib')}",
#     f".include {os.path.abspath('./02 Simulations/00 ComonLibs/EPCGaN.asy')}"
# )

start_time = time.time()
all_meas_results = {}
for specific_config, changed_params in SimConfig.yield_cnfgs_sequentially(iter_type=SIM_TYPE):
    # Reset netlist + trim number of devices
    SimConfig.prepare_netlist_for_sim(new_cnfg=specific_config, netlist=netlist)
    sim = LTC.run(netlist)
    # Store the changed parameters for this sim
    log_name = sim.netlist_file.name.rstrip('.asc')
    all_meas_results[log_name] = {}
    all_meas_results[log_name]['config_changes']= changed_params
    time.sleep(1.5)  # slight delay to avoid overloading LTSpice

# Wait for all the sims launched. A timeout counter from last completed sim keeps track of stalled sims
LTC.wait_completion()
print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))


for raw, log in LTC:
    # print("Raw file: %s, Log file: %s" % (raw, log))
    raw_data = RawRead(raw)
    # Store the extracted waveforms in the results dictionary
    sim_time = [abs(t) for t in list(raw_data.get_trace('time'))]  # Necessary due to LTSpice bug printing negative time sometimes
    all_meas_results[log.stem]['waveforms'] = {'time': sim_time}
    for i in range(1, SimConfig.N_devices + 1):
        all_meas_results[log.stem]['waveforms'][f'i_drain_{i}'] = list(raw_data.get_trace(f'I(x{i}:L_drain)'))
        vgate = list(raw_data.get_trace(f'V(G{i})'))
        vsource = list(raw_data.get_trace(f'V(S{i})'))
        all_meas_results[log.stem]['waveforms'][f'v_gs_{i}'] = [vgate[j] - vsource[j] for j in range(len(vgate))]
    
    # print(f"Trace length is {len(i_drain_sw1_r)} and they match: {i_drain_sw1_r == i_drain_sw1_l}")
    # # print(raw_data.get_trace_names())
    # time = list(raw_data.get_trace('time'))
    # time = [abs(t) for t in time]  # Necessary due to LTSpice bug printing negative time sometimes
    # print(f"Simulation time is {time[-1]}s, sim length is {len(time)} points")
    
    # Get the names of the variables that were stepped, and the measurements taken
    log_data = LTSpiceLogReader(log)
    # step_names = log_data.get_step_vars()

    # Remove extra logged items, and convert single-item lists to values, save to dict
    meas_results_dict = {k: v for k, v in log_data.dataset.items() if not k.endswith(('_to', '_from', '_at'))}
    meas_results_dict = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in meas_results_dict.items()}
    all_meas_results[log.stem]['meas results'] = meas_results_dict


# Clean up temporary files (asc, net, log, raw)
# LTC.file_cleanup()

end_time = time.time()
print(f"Loop execution time: {end_time - start_time:.2f} seconds")


if SIM_TYPE == 'sweep':
    generate_sweep_plots(params_to_plot=SimConfig.params_to_plot, meas_res=all_meas_results,
                         skip_branches=['x3', 'x4'], save_figs=False)
elif SIM_TYPE == 'perturbation':
    # generate_perturbation_plots(params_to_plot=SimConfig.params_to_plot, meas_res=all_meas_results,
    #                      skip_branches=['x3', 'x4'], base_config=base_config_dict, save_figs=True)
    generate_perturbation_heat_map(params_to_plot=SimConfig.params_to_plot,
                                   meas_res=all_meas_results,
                                   skip_branches=list(range(3, SimConfig.N_devices + 1)),
                                   base_config=base_config_dict,
                                   k_perturb=SimConfig.default_Kperturb,
                                   save_figs=False)
    
    plot_wfm_from_all_runs(meas_res=all_meas_results,
                           explicit_wfms=[],
                           skip_branches=list(range(3, SimConfig.N_devices + 1)),
                           base_config=base_config_dict,
                           edge='rise',
                           save_figs=False)
elif SIM_TYPE == 'multiparam':
    raise Exception("multiparam sim not implemented")
else:
    raise Exception("Wrong SIM_TYPE, choose one from 'sweep', 'perturbation' or 'multiparam'")
