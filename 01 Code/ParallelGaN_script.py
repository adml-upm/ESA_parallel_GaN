from PyLTSpice import SimRunner
from PyLTSpice import AscEditor
# from PyLTSpice import RawRead
from PyLTSpice.log.ltsteps import LTSpiceLogReader
import time
from SimConfiguration import LtSimConfiguration
from ResultPlotting import generate_sweep_plots, generate_perturbation_plots, generate_perturbation_heat_map


def update_netlist_params(new_cnfg: dict, netlist):
    # Update all params & sim command
    netlist.add_instruction(f".tran {2*max([new_cnfg[f"T_sw_{i}"] for i in range(1, N_devices + 1)])*1e6:.3}u")
    for k in sorted(new_cnfg.keys()):
        netlist.set_parameter(k, new_cnfg[k]) 
    # Update gate driving voltage sources
    for i in range(1, N_devices + 1):
        new_cnfg[f"T_on_{i}"] = new_cnfg[f"D_on_{i}"] * new_cnfg[f"T_sw_{i}"]
        
        T_on_corr = new_cnfg[f"T_on_{i}"] - (new_cnfg[f"tdrv_rise_{i}"] + new_cnfg[f"tdrv_fall_{i}"]) / 2
        new_cnfg[f"T_on_corr_{i}"] = T_on_corr
        new_cnfg[f"T_off_{i}"] = new_cnfg[f"T_sw_{i}"] - new_cnfg[f"T_on_{i}"] - new_cnfg[f"tdrv_rise_{i}"] - new_cnfg[f"tdrv_fall_{i}"]
        new_cnfg[f"T_off_corr_{i}"] = new_cnfg[f"T_sw_{i}"] - T_on_corr
        driver_cmd = "PULSE({voff} {von} {d} {tr:.3}n {tf:.3}n {ton:.5}u {tsw:.5}u)".format(voff=new_cnfg[f"Vdrv_off_{i}"],
                                                                                        von=new_cnfg[f"Vdrv_on_{i}"],
                                                                                        d=new_cnfg[f"Vdrive_delay_{i}"],
                                                                                        tr=new_cnfg[f"tdrv_rise_{i}"] * 1e9,
                                                                                        tf=new_cnfg[f"tdrv_fall_{i}"] * 1e9,
                                                                                        ton=new_cnfg[f"T_on_corr_{i}"] * 1e6,
                                                                                        tsw=new_cnfg[f"T_sw_{i}"] * 1e6)
        netlist.set_component_value(f'Vgdrv{i}', driver_cmd)

def introduce_meas_commands(new_cnfg: dict, netlist):
    # Remove existing .meas commands
    netlist.remove_Xinstruction(search_pattern='.meas')
    for i in range(1, N_devices + 1):
        T_sw   = new_cnfg[f"T_sw_{i}"]
        T_on_i = new_cnfg[f"T_on_{i}"]
        tdrv_r = new_cnfg[f"tdrv_rise_{i}"]
        T_off  = new_cnfg[f"T_off_{i}"]

        # Current balance measurements
        t_on_curr_meas = T_sw + tdrv_r + T_on / 2  # Midway through on-time of second cycle
        # Power Loss measurements
        delta_t_loss_meas = min(500e-9, T_on_i * 0.025)
        P_on_times  = [T_sw, T_sw + delta_t_loss_meas]
        P_off_times = [T_sw, T_on_i + tdrv_r + T_on_i + delta_t_loss_meas]
        # Vds Overshoot params
        Vds_pk_times = [T_sw + tdrv_r + T_on_i, T_sw + tdrv_r + T_on_i + T_off / 2]
        # Vgs Overshoot params
        Vgs_pk_times = [T_sw, T_sw + tdrv_r + T_on_i / 2]
    
        loss_param = f"V(G{i})*Ix(X{i}:U1:gatein)+V(D{i})*Ix(X{i}:U1:drainin)+V(X{i}:hemt_s_noLs)*Ix(X{i}:U1:sourcein)"
        netlist.add_instructions(
            f".meas X{i}_cond_current FIND I(X{i}:R_drain) AT {1e6*t_on_curr_meas:.6}u",
            f".meas X{i}_E_on INTEG {loss_param} FROM {1e6*P_on_times[0]:.6}u TO {1e6*P_on_times[1]:.6}u",
            f".meas X{i}_E_off INTEG {loss_param} FROM {1e6*P_off_times[0]:.6}u TO {1e6*P_off_times[1]:.6}u",
            f".meas X{i}_vds_pk MAX V(D{i},X{i}:hemt_s_noLs) FROM {1e6*Vds_pk_times[0]:.6}u TO {1e6*Vds_pk_times[1]:.6}u",
            f".meas X{i}_vgs_pk MAX V(G{i},X{i}:hemt_s_noLs) FROM {1e6*Vgs_pk_times[0]:.6}u TO {1e6*Vgs_pk_times[1]:.6}u",        
        )
    

# Force another simulatior
simulator = r"C:\Users\Alex\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

# select spice model
LTC = SimRunner(output_folder='temp', simulator=None, parallel_sims=16, timeout=30)

sim_path = './02 Simulations/02 ParallelGaN/ParGan_subckt_example.asc'
netlist = AscEditor(sim_path)


# Electrical testing setpoint:
N_devices = 4
m_points = 5
V_DC = 100  # V
I_DC = 10   # A
# netlist.set_component_value('V_DC', f'{V_DC}')
# netlist.set_component_value('I_DC', f'{I_DC}')
f_sw = 100e3
T_sw = 1 / f_sw
D_sw = 0.5
T_on = D_sw * T_sw

# ---------------------------------------------------------------
# Gate Driver setup: 
# Vdrv_off = 0
# Vdrv_on = 5
# Vdrive_delay = 0
# tdrv_rise = 1e-9
# tdrv_fall = 1e-9

# T_off = T_sw - T_on - tdrv_rise - tdrv_fall
# T_off_corr = T_sw - T_on_corr

base_config_dict = {
    'V_DC': V_DC,
    'I_DC': I_DC
}
for i in range(1, N_devices + 1): 
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


SIM_TYPE = 'sweep'  # 'sweep', 'perturbation', 'multiparam'
if SIM_TYPE == 'sweep':
    SimConfig = LtSimConfiguration(base_config=base_config_dict)
    SimConfig.add_param_sweep_with_range('R_drain_1', [1e-3, 10e-3], m_points=m_points)
    SimConfig.add_param_sweep_with_range('L_drain_1', [500e-12, 50e-9], m_points=m_points, log_space=True)
elif SIM_TYPE == 'perturbation':
    SimConfig = LtSimConfiguration(base_config=base_config_dict)
    SimConfig.default_Kperturb = 0.1  # Modify standard relative perturbation amount (p.u.)
    # SimConfig.add_param_perturbation('L_drain_1', 10e-9)
    # SimConfig.add_param_perturbation('R_drain_1', 1e-3)
    SimConfig.add_param_perturbation("R_drain_1")
    SimConfig.add_param_perturbation("L_drain_1")
    SimConfig.add_param_perturbation("R_source_1")
    SimConfig.add_param_perturbation("L_source_1")
    SimConfig.add_param_perturbation("R_pcb_branch_1")
    SimConfig.add_param_perturbation("Lg_uncommon_1")
    SimConfig.add_param_perturbation("L_common_source_1")
    SimConfig.add_param_perturbation("Vdrv_off_1")
    SimConfig.add_param_perturbation("Vdrv_on_1", perturb_rel=0.1)
    SimConfig.add_param_perturbation("Vdrive_delay_1")
    SimConfig.add_param_perturbation("tdrv_rise_1")
    SimConfig.add_param_perturbation("tdrv_fall_1")
    SimConfig.add_param_perturbation("T_sw_1", perturb_rel=0.1)
    SimConfig.add_param_perturbation("D_on_1")
    # SimConfig.add_param_perturbation("R_driver_1")
    # SimConfig.add_param_perturbation("R_g_1")
    # SimConfig.add_param_perturbation('L_drain_1', 1e-3)
elif SIM_TYPE == 'multiparam':
    pass
else:
    raise ValueError("Invalid SIM_TYPE. Choose from 'sweep', 'perturbation', or 'multiparam'.")

# ---------------------------------------------------------------
# .meas setup:


# ---------------------------------------------------------------

# netlist.add_instructions(
#     f".include {os.path.abspath('./02 Simulations/00 ComonLibs/EPCGaNLibrary.lib')}",
#     f".include {os.path.abspath('./02 Simulations/00 ComonLibs/EPCGaN.asy')}"
# )

start_time = time.time()
all_meas_results = {}

for config, changed_params in SimConfig.yield_cnfgs_sequentially(iter_type=SIM_TYPE):
    # Update netlist and run sim    
    update_netlist_params(config, netlist)
    introduce_meas_commands(config, netlist)
    sim = LTC.run(netlist)
    # Store the changed parameters for this sim
    log_name = sim.netlist_file.name.rstrip('.asc')
    all_meas_results[log_name] = {}
    all_meas_results[log_name]['config_changes']= changed_params

# Wait for all the sims launched. A timeout counter from last completed sim keeps track of stalled sims
LTC.wait_completion()
print('Successful/Total Simulations: ' + str(LTC.okSim) + '/' + str(LTC.runno))

for raw, log in LTC:
    # print("Raw file: %s, Log file: %s" % (raw, log))
    # raw_data = RawRead(raw)
    # i_drain_sw1_r = raw_data.get_trace('I(R1)')
    # i_drain_sw1_l = raw_data.get_trace('I(x1:L_drain)')
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
                                   skip_branches=['x3', 'x4'],
                                   base_config=base_config_dict,
                                   k_perturb=SimConfig.default_Kperturb,
                                   save_figs=True)
elif SIM_TYPE == 'multiparam':
    raise Exception("multiparam sim not implemented")
else:
    raise Exception("Wrong SIM_TYPE, choose one from 'sweep', 'perturbation' or 'multiparam'")



