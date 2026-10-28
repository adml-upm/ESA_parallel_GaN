"""Logic layer behind SimGUI.py.

Everything that is not widgets or flow control lives here: config files, .asc node
scanning, turning GUI text into a study definition, running the LTspice batch and
dispatching to ResultPlotting. Mirrors the flow of ParallelGaN_script.py.
No tkinter imports in this file.
"""
import contextlib
import copy
import csv
import dataclasses
import inspect
import json
import os
import pickle
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PyLTSpice import SimRunner, AscEditor, RawRead, LTspice
from PyLTSpice.log.ltsteps import LTSpiceLogReader
from spicelib.editor.base_editor import scan_eng

from SimConfiguration import LtSimConfiguration
from ResultPlotting import (generate_sweep_plots, generate_perturbation_barplots,
                            generate_perturbation_heat_map, generate_multiparam_heat_maps,
                            plot_wfm_from_all_runs, plot_wfm_2param, friendly_name)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIM_PATH = REPO_ROOT / "02 Simulations" / "02 ParallelGaN" / "ParGan_V4.asc"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
OUTPUT_DIR = REPO_ROOT / "temp"
RESULTS_DIR = REPO_ROOT / "results"
LIB_DIR = REPO_ROOT / "02 Simulations" / "00 ComonLibs"

SIM_TYPES = ('sweep', 'perturbation', 'multiparam')
PERTURB_MODES = ('rel', 'abs')
TOTAL_DRAWN_DEVICES = 10  # HEMT branches drawn in ParGan_V4.asc
# Each LTspice run uses every CPU thread, so too many parallel runs thrash and hit SIM_TIMEOUT_S.
# Measured on 32 threads (10x10 grid): 16 parallel 67 s (longest run 20-39 s), 8 parallel 75 s (longest 5.5 s).
PARALLEL_SIMS = max(1, (os.cpu_count() or 4) // 4)
SIM_TIMEOUT_S = 30

# Traces inside each hemt_cell instance X{i}; not visible as FLAGs in the top-level .asc
HEMT_CELL_TRACES = ('I(X{i}:L_drain)', 'V(X{i}:hemt_s_nols)')
# Per-device waveforms derived after the run (used by plot_wfm_from_all_runs)
DERIVED_WAVEFORMS = ('i_drain', 'v_gs', 'v_ds', 'Tj')
_DERIVED_SOURCES = {'i_drain': ('I(X{i}:L_drain)',),
                    'v_gs': ('V(G{i})', 'V(S{i})'),
                    'v_ds': ('V(D{i})', 'V(S{i})'),
                    'Tj': ('V(Tj{i})',)}


# ==================== Base configuration ====================
def build_default_base_config(n_devices: int = 4) -> dict:
    """Base config with the same defaults as ParallelGaN_script.py."""
    sim_config = LtSimConfiguration()
    cfg = {'V_DC': 100, 'I_DC': 10, 'N_devices': n_devices}
    for i in range(1, n_devices + 1):
        cfg.update({
            f'dev_{i}': "EPC2305", f"Q_Tj_{i}": 25,
            f"R_drain_{i}": 1e-3, f"L_drain_{i}": 1e-9, f"R_source_{i}": 1e-3, f"L_source_{i}": 1e-9,
            f"R_pcb_branch_{i}": 1e-3,
            f"Lg_uncommon_{i}": 2.5e-9, f"L_common_source_{i}": 200e-12, f"R_driver_{i}": 1e-3, f"R_g_{i}": 1,
            f"Vdrv_off_{i}": 0, f"Vdrv_on_{i}": 5, f"Vdrive_delay_{i}": 1e-9,
            f"tdrv_rise_{i}": 1e-9, f"tdrv_fall_{i}": 1e-9, f"T_sw_{i}": 1 / 100e3, f"D_on_{i}": 0.5,
        })
        hemt_params = sim_config.get_hemt_parameters_from_lib(cfg[f'dev_{i}'])[cfg[f'dev_{i}']][0]
        cfg.update({k + f'_{i}': v for k, v in hemt_params.items()})
    return cfg


def load_base_config(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_base_config(cfg: dict, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)


def split_config_keys(cfg: dict):
    """Group keys for display.
    returns: (global_keys, per_device, n_devices) with per_device = {base_name: {i: key}}"""
    n_devices = int(cfg.get('N_devices', 0))
    global_keys, per_device = [], {}
    for key in cfg:
        base, _, idx = key.rpartition('_')
        if base and idx.isdigit() and 1 <= int(idx) <= n_devices:
            per_device.setdefault(base, {})[int(idx)] = key
        else:
            global_keys.append(key)
    return global_keys, per_device, n_devices


def variable_params(cfg: dict) -> list[str]:
    """Config keys that can be swept/perturbed (numeric, excluding N_devices)."""
    return [k for k, v in cfg.items() if isinstance(v, (int, float)) and k != 'N_devices']


# HEMT model params are stored by SPICE name ('k2_1'); the GUI shows the friendly name ('Vgs_th_1')
_FRIENDLY_TO_SPICE = LtSimConfiguration().epc_model_param_mapping
_SPICE_TO_FRIENDLY = {v: k for k, v in _FRIENDLY_TO_SPICE.items()}


def display_name(key: str) -> str:
    """'k2_1' -> 'Vgs_th_1', 'k2' -> 'Vgs_th'. Other keys are returned unchanged."""
    return _SPICE_TO_FRIENDLY.get(key) or friendly_name(key)


# Units of the config values, keyed by config base name (no '_<i>' device index). All values are
# plain SI numbers. HEMT model units follow the EPC2305 .subckt equations in EPCGaNLibrary.lib.
PARAM_UNITS = {
    'V_DC': 'V', 'I_DC': 'A', 'Q_Tj': '°C',
    'R_drain': 'Ω', 'L_drain': 'H', 'R_source': 'Ω', 'L_source': 'H', 'R_pcb_branch': 'Ω',
    'Lg_uncommon': 'H', 'L_common_source': 'H', 'R_driver': 'Ω', 'R_g': 'Ω',
    'Vdrv_off': 'V', 'Vdrv_on': 'V', 'Vdrive_delay': 's', 'tdrv_rise': 's', 'tdrv_fall': 's',
    'T_sw': 's', 'D_on': 'p.u.',
    'A1': 'A/V', 'k2': 'V', 'k3': 'V', 'rpara': 'Ω', 'rg_value': 'Ω',
    'aITc': '1/°C', 'arTc': '1/°C', 'k2Tc': '1/°C',
}


def param_unit(key: str) -> str:
    """Unit of a config key or base name ('R_drain_1' or 'R_drain' -> 'Ω'); '' if unknown or unitless."""
    return PARAM_UNITS.get(key) or PARAM_UNITS.get(re.sub(r'_\d+$', '', key), '')


def config_key(name: str) -> str:
    """Inverse of display_name: 'Vgs_th_1' -> 'k2_1'. Other names are returned unchanged."""
    base, _, idx = name.rpartition('_')
    if base in _FRIENDLY_TO_SPICE and idx.isdigit():
        return f'{_FRIENDLY_TO_SPICE[base]}_{idx}'
    return name


def _parse_number(text: str) -> float:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return float(scan_eng(text))  # accepts engineering notation, e.g. 2.5n, 200p


def parse_config_values(raw_values: dict, template: dict) -> dict:
    """Convert GUI text back into config values, using template's types.
    Raises ValueError naming the offending key."""
    cfg = {}
    for key, text in raw_values.items():
        if isinstance(template.get(key), str):
            cfg[key] = text.strip()
            continue
        try:
            value = _parse_number(text)
        except (ValueError, TypeError):
            raise ValueError(f"'{key}': cannot parse '{text}' as a number")
        cfg[key] = int(value) if isinstance(template.get(key), int) and value.is_integer() else value
    return cfg


def validate_base_config(cfg: dict):
    n = int(cfg.get('N_devices', 0))
    if not 1 <= n <= TOTAL_DRAWN_DEVICES:
        raise ValueError(f"N_devices must be between 1 and {TOTAL_DRAWN_DEVICES}")
    missing = [f'{base}_{i}' for base in split_config_keys(cfg)[1] for i in range(1, n + 1)
               if f'{base}_{i}' not in cfg]
    if missing:
        raise ValueError(f"Config has no entries for: {', '.join(missing[:5])}"
                         f"{'...' if len(missing) > 5 else ''}. Load a config made for {n} devices.")
    zero_r = [k for k, v in cfg.items() if param_unit(k) == 'Ω' and v == 0]
    if zero_r:
        raise ValueError(f"LTspice rejects 0 Ω resistors: {', '.join(zero_r)}. Use a small value such as 1m.")


# ==================== .asc viewing and node scan ====================
def open_in_ltspice(asc_path):
    """Open the schematic in the LTspice GUI (non-blocking), to look at it and read node names.
    Uses the LTspice that PyLTSpice auto-detects, else the Windows .asc file association."""
    path = Path(asc_path)
    if not path.is_file():
        raise ValueError(f"Schematic not found: {path}")
    if LTspice.spice_exe:
        subprocess.Popen([*LTspice.spice_exe, str(path)], cwd=path.parent)
    else:
        os.startfile(path)


# ==================== .asc node scan ====================
def scan_asc_nodes(asc_path, n_devices: int) -> list[str]:
    """List traces that can be saved: V() of every net label (FLAG) and the known
    hemt_cell internal traces of every X{i} instance. Nets/instances whose trailing
    index exceeds n_devices are dropped, since those branches get trimmed."""
    text = Path(asc_path).read_text(encoding='utf-8', errors='ignore')
    nets = {m.group(1) for m in re.finditer(r'^FLAG\s+\S+\s+\S+\s+(\S+)', text, re.MULTILINE)}
    instances = {int(m.group(1)) for m in re.finditer(r'^SYMATTR\s+InstName\s+X(\d+)\s*$', text,
                                                      re.MULTILINE | re.IGNORECASE)}

    def _in_range(name):
        m = re.search(r'(\d+)$', name)
        return m is None or int(m.group(1)) <= n_devices

    def _sort_key(name):
        m = re.search(r'(\d+)', name)
        return (int(m.group(1)) if m else 0, name)

    traces = [f'V({net})' for net in nets if net != '0' and _in_range(net)]
    traces += [t.format(i=i) for i in sorted(instances) if i <= n_devices for t in HEMT_CELL_TRACES]
    return sorted(traces, key=_sort_key)


def default_saved_traces(n_devices: int) -> list[str]:
    """Traces needed to build DERIVED_WAVEFORMS (same set the script saves)."""
    return [src.format(i=i) for i in range(1, n_devices + 1)
            for srcs in _DERIVED_SOURCES.values() for src in srcs] + \
           [f'V(X{i}:hemt_s_nols)' for i in range(1, n_devices + 1)]


# ==================== Study definition ====================
@dataclass
class Study:
    sim_type: str
    sweeps: list = field(default_factory=list)          # [(param, start, stop, points, log_space)]
    perturbations: list = field(default_factory=list)   # [(param, amount, 'rel'|'abs')]
    k_perturb: float = 0.1


def build_study(sim_type: str, rows: list[dict], k_perturb_text: str = '0.1') -> Study:
    """rows (raw GUI text): sweep/multiparam -> {param, start, stop, points, log};
    perturbation -> {param, amount, mode}. param may be a display name ('Vgs_th_1');
    the Study stores config keys ('k2_1')."""
    if sim_type not in SIM_TYPES:
        raise ValueError(f"Unknown simulation type '{sim_type}'")
    rows = [r for r in rows if r.get('param')]
    if not rows:
        raise ValueError("Select at least one parameter to vary")
    if len({r['param'] for r in rows}) != len(rows):
        raise ValueError("Each parameter can only be selected once")
    if sim_type == 'multiparam' and len(rows) != 2:
        raise ValueError("2-param study needs exactly two parameters")

    study = Study(sim_type=sim_type, k_perturb=_parse_number(k_perturb_text))
    for r in rows:
        param = config_key(r['param'])
        try:
            if sim_type == 'perturbation':
                study.perturbations.append((param, _parse_number(r['amount']), r['mode']))
            else:
                points = int(_parse_number(r['points']))
                if points < 2:
                    raise ValueError("points must be >= 2")
                start, stop = _parse_number(r['start']), _parse_number(r['stop'])
                if param_unit(param) == 'Ω' and 0 in (start, stop):
                    raise ValueError("LTspice rejects 0 Ω resistors; start or stop at a small value such as 1m")
                study.sweeps.append((param, start, stop, points, bool(r['log'])))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Row '{r['param']}': {exc}")
    return study


# ==================== Running ====================
@dataclass
class StudyResult:
    sim_type: str
    meas_res: dict
    base_config: dict
    params_to_plot: list
    k_perturb: float
    ok: int
    failed: int
    total: int
    elapsed_s: float
    # Time breakdown: launching + waiting for LTspice, longest single LTspice run, reading .raw/.log files
    sim_s: float = 0.0
    longest_sim_s: float = 0.0
    read_s: float = 0.0
    stopped: bool = False   # True if the user stopped the batch early
    sim_path: str = ''
    meas_csv: str = ''      # measurement table written after a 2-param study (export_meas_csv)


def export_meas_csv(result: StudyResult, path):
    """Write one row per run: the varied parameter values, then every .meas value (SI units).
    The base run is the first row, with the base values of the varied parameters."""
    params = list(result.params_to_plot)
    meas_keys = sorted({k for entry in result.meas_res.values() for k in entry['meas results']})
    param_headers = [f"{display_name(p)} [{param_unit(p)}]" if param_unit(p) else display_name(p) for p in params]
    # utf-8-sig so Excel shows the unit symbols (Ω, °C) correctly
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['run'] + param_headers + meas_keys)
        for run_name, entry in result.meas_res.items():
            values = [entry['config_changes'].get(p, result.base_config[p]) for p in params]
            writer.writerow([run_name] + values + [entry['meas results'].get(k, '') for k in meas_keys])


def save_result(result: StudyResult, path):
    """Pickle a StudyResult so its plots can be redone without simulating again."""
    with open(path, 'wb') as f:
        pickle.dump(dataclasses.asdict(result), f)


def load_result(path) -> StudyResult:
    """Load a file written by save_result. Only open files you saved yourself: unpickling runs code."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    known = {f.name for f in dataclasses.fields(StudyResult)}
    return StudyResult(**{k: v for k, v in data.items() if k in known})


def _configure(sim_config: LtSimConfiguration, study: Study):
    sim_config.default_k_perturb = study.k_perturb
    for param, start, stop, points, log_space in study.sweeps:
        sim_config.add_param_sweep_with_range(param, [start, stop], m_points=points, log_space=log_space)
    for param, amount, mode in study.perturbations:
        if mode == 'rel':
            sim_config.add_param_perturbation(param, perturb_rel=amount)
        else:
            sim_config.add_param_perturbation(param, perturb_abs=amount)


def run_study(sim_path, base_config: dict, traces: list[str], study: Study,
              on_progress=None, stop_event: threading.Event | None = None) -> StudyResult:
    """Run the whole batch (blocking; call from a worker thread).
    on_progress(completed, failed, total) is called from this thread.
    Setting stop_event stops launching new runs; runs already started finish (each is capped at
    SIM_TIMEOUT_S) and the result holds what completed."""
    # Custom symbols/subcircuits (gate_cell_V1, hemt_cell_V2, ...) live in 00 ComonLibs, not next to the .asc
    if str(LIB_DIR) not in AscEditor.custom_lib_paths:
        AscEditor.set_custom_library_paths(*AscEditor.custom_lib_paths, str(LIB_DIR))
    # Same working directory as ParallelGaN_script.py (changes the process CWD while running)
    with contextlib.chdir(REPO_ROOT):
        return _run_study(sim_path, base_config, traces, study, on_progress, stop_event)


def _run_study(sim_path, base_config, traces, study, on_progress, stop_event) -> StudyResult:
    validate_base_config(base_config)
    if not traces:
        raise ValueError("Select at least one node to save")
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    sim_config = LtSimConfiguration(sim_output_folder=str(OUTPUT_DIR))
    sim_config.N_devices = int(base_config['N_devices'])
    sim_config.TOTAL_DRAWN_DEVICES = TOTAL_DRAWN_DEVICES
    sim_config.base_cnfg_dict = copy.deepcopy(base_config)
    _configure(sim_config, study)
    configs = list(sim_config.yield_cnfgs_sequentially(iter_type=study.sim_type))
    total = len(configs)

    def _report():
        if on_progress:
            on_progress(runner.okSim + runner.failSim, runner.failSim, total)

    netlist = AscEditor(str(sim_path))
    runner = SimRunner(output_folder=str(OUTPUT_DIR), simulator=None,
                       parallel_sims=PARALLEL_SIMS, timeout=SIM_TIMEOUT_S)
    wfms_to_save = {t: t for t in traces}
    meas_res = {}
    _report()
    stopped = False
    for sim_index, (cfg, changed_params) in enumerate(configs, start=1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            total = len(meas_res)  # only the launched runs count from here on
            break
        sim_config.prepare_netlist_for_sim(new_cnfg=cfg, netlist=netlist, sim_index=sim_index)
        sim_config.define_selective_saving(wfms_to_save=wfms_to_save, netlist=netlist)
        run_name = f"{netlist.asc_file_path.stem}_sim_{sim_index}"
        runner.run(netlist, run_filename=run_name + ".asc")
        meas_res[run_name] = {'config_changes': changed_params}
        _report()

    while runner.active_threads() > 0:
        time.sleep(0.25)
        runner.update_completed()
        _report()
    runner.wait_completion()
    _report()
    sims_done = time.time()

    for raw, log in runner:
        if raw is None or log is None or log.stem not in meas_res:
            continue
        _collect_results(meas_res[log.stem], raw, log, traces, sim_config.N_devices)
    meas_res = {k: v for k, v in meas_res.items() if 'meas results' in v}  # drop failed sims

    run_times = [t.stop_time - t.start_time for t in runner.completed_tasks
                 if t.start_time is not None and t.stop_time is not None]
    end = time.time()
    result = StudyResult(sim_type=study.sim_type, meas_res=meas_res, base_config=copy.deepcopy(base_config),
                         params_to_plot=list(sim_config.params_to_plot), k_perturb=study.k_perturb,
                         ok=runner.okSim, failed=runner.failSim, total=total, elapsed_s=end - start,
                         sim_s=sims_done - start, longest_sim_s=max(run_times, default=0.0),
                         read_s=end - sims_done, stopped=stopped, sim_path=str(sim_path))
    if study.sim_type == 'multiparam' and meas_res:
        RESULTS_DIR.mkdir(exist_ok=True)
        csv_path = RESULTS_DIR / f"{Path(sim_path).stem}_multiparam_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        export_meas_csv(result, csv_path)
        result.meas_csv = str(csv_path)
    return result


def _collect_results(entry: dict, raw, log, traces: list[str], n_devices: int):
    raw_data = RawRead(raw)
    saved = {t: list(raw_data.get_trace(t)) for t in traces}
    entry['traces'] = saved
    waveforms = {'time': raw_data.get_axis()}
    for i in range(1, n_devices + 1):
        src = {name: [s.format(i=i) for s in srcs] for name, srcs in _DERIVED_SOURCES.items()}
        if all(s in saved for s in src['i_drain']):
            waveforms[f'i_drain_{i}'] = saved[src['i_drain'][0]]
        for name in ('v_gs', 'v_ds'):
            if all(s in saved for s in src[name]):
                pos, neg = (saved[s] for s in src[name])
                waveforms[f'{name}_{i}'] = [p - n for p, n in zip(pos, neg)]
        if all(s in saved for s in src['Tj']):
            waveforms[f'Tj_{i}'] = saved[src['Tj'][0]]
    entry['waveforms'] = waveforms

    log_data = LTSpiceLogReader(log)
    meas = {k: v for k, v in log_data.dataset.items() if not k.endswith(('_to', '_from', '_at'))}
    entry['meas results'] = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in meas.items()}


# ==================== Plotting ====================
# Each ResultPlotting function is described by a PlotSpec. Its arguments are read from the
# real function signature: data arguments (AUTO_ARGS) are filled from the StudyResult, and the
# rest are described as PlotArgs so the GUI can build an editor for them generically.
@dataclass
class PlotSpec:
    name: str        # ResultPlotting function name
    title: str
    func: object
    sim_types: tuple
    note: str = ''


@dataclass
class PlotArg:
    name: str
    kind: str        # 'multi' (pick several), 'choice' (pick one), 'bool', 'text'
    default: object
    choices: list = field(default_factory=list)
    help: str = ''


PLOTS = [
    PlotSpec('generate_sweep_plots', 'Measurements vs swept parameter', generate_sweep_plots, ('sweep',),
             "One figure per swept parameter. Empty explicit_wfms = all measurements."),
    PlotSpec('generate_perturbation_heat_map', 'Sensitivity heat map', generate_perturbation_heat_map,
             ('perturbation',), "Only for HEMT model params (k2, A1, rpara, ...)."),
    PlotSpec('generate_perturbation_barplots', '% variation bar plots', generate_perturbation_barplots,
             ('perturbation',), "One figure per perturbed run."),
    PlotSpec('generate_multiparam_heat_maps', 'Measurement heat maps over the 2-param grid',
             generate_multiparam_heat_maps, ('multiparam',),
             "One figure per measurement, one heat map per device. Empty explicit_wfms = all measurements."),
    PlotSpec('plot_wfm_2param', 'Waveforms over the 2-param grid (edge zoom)', plot_wfm_2param, ('multiparam',),
             "Colour = first param (colour bar), line width = second param. swap_axes exchanges them. "
             "Empty explicit_wfms = all."),
    PlotSpec('plot_wfm_from_all_runs', 'Waveforms of all runs (edge zoom)', plot_wfm_from_all_runs,
             ('sweep', 'perturbation'),
             "Overlays derived per-device waveforms, base run excluded. Empty explicit_wfms = all."),
]
AUTO_ARGS = ('meas_res', 'base_config')
WAVEFORM_PLOTS = ('plot_wfm_from_all_runs', 'plot_wfm_2param')  # take waveform names and int skip_branches


def _get_plot(plot_name: str) -> PlotSpec:
    return next(p for p in PLOTS if p.name == plot_name)


def available_plots(result: StudyResult) -> list[PlotSpec]:
    return [p for p in PLOTS if result.sim_type in p.sim_types]


def _meas_names(result: StudyResult) -> list[str]:
    """Measurement names without the device prefix, e.g. 'x1_e_on' -> 'e_on'."""
    return sorted({'_'.join(k.split('_')[1:]) for e in result.meas_res.values() for k in e['meas results']})


def _waveform_names(result: StudyResult) -> list[str]:
    return sorted({k.rsplit('_', 1)[0] for e in result.meas_res.values() for k in e['waveforms'] if k != 'time'})


def plot_args(result: StudyResult, plot_name: str) -> list[PlotArg]:
    """Editable arguments of a plot function, with defaults taken from the result."""
    spec = _get_plot(plot_name)
    n_devices = int(result.base_config['N_devices'])
    args = []
    for name, param in inspect.signature(spec.func).parameters.items():
        if name in AUTO_ARGS:
            continue
        if name == 'params_to_plot':
            varied = [display_name(p) for p in result.params_to_plot]  # converted back in launch_plot
            args.append(PlotArg(name, 'multi', varied, varied, "Varied parameters to plot"))
        elif name == 'explicit_wfms' and spec.name in WAVEFORM_PLOTS:
            choices = _waveform_names(result)
            args.append(PlotArg(name, 'multi', [w for w in ('i_drain', 'v_gs') if w in choices], choices,
                                "Waveforms to plot (none = all)"))
        elif name == 'explicit_wfms':
            choices = _meas_names(result)
            # A heat map figure per measurement adds up quickly, so preselect the main ones
            default = ([m for m in ('e_on', 'e_off', 'cond_ratio') if m in choices]
                       if spec.name == 'generate_multiparam_heat_maps' else [])
            args.append(PlotArg(name, 'multi', default, choices, "Measurements to plot (none = all)"))
        elif name == 'skip_branches':
            args.append(PlotArg(name, 'multi', [], [str(i) for i in range(1, n_devices + 1)],
                                "Devices to leave out"))
        elif name == 'edge':
            args.append(PlotArg(name, 'choice', 'rise', ['rise', 'fall'], "Zoom on the turn-on or turn-off edge"))
        elif name == 'swap_axes':
            args.append(PlotArg(name, 'bool', False, help="Exchange the roles of the two swept parameters"))
        elif name == 'save_figs':
            args.append(PlotArg(name, 'bool', False, help="Save PNGs to plots/ instead of opening windows"))
        elif name == 'k_perturb':
            args.append(PlotArg(name, 'text', str(result.k_perturb), help="Perturbation shown in the title (p.u.)"))
        else:  # argument added to ResultPlotting later: plain text field
            default = '' if param.default is inspect.Parameter.empty else str(param.default)
            args.append(PlotArg(name, 'text', default))
    return args


def _convert_skip_branches(spec: PlotSpec, devices: list[str], result: StudyResult):
    if spec.name in WAVEFORM_PLOTS:
        return [int(i) for i in devices]  # matched against the waveform name's device suffix
    # The other functions match by substring on measurement keys ('x3_e_on'), so pass 'x3_'
    keys = [k for e in result.meas_res.values() for k in e['meas results']]
    prefix = keys[0][0] if keys else 'x'
    return [f'{prefix}{i}_' for i in devices]


def launch_plot(result: StudyResult, plot_name: str, values: dict):
    """Call a ResultPlotting function. values = {arg name: GUI value} for the args in plot_args().
    Shows the figure(s) through plt.show(), or saves them to plots/ when save_figs is set."""
    spec = _get_plot(plot_name)
    kwargs = {}
    for arg in plot_args(result, plot_name):
        value = values.get(arg.name, arg.default)
        if arg.name == 'skip_branches':
            value = _convert_skip_branches(spec, value, result)
        elif arg.name == 'params_to_plot':
            value = [config_key(p) for p in value]
        elif arg.kind == 'text':
            try:
                value = _parse_number(value)
            except (ValueError, TypeError):
                pass  # leave as text
        kwargs[arg.name] = value

    if 'params_to_plot' in kwargs and not kwargs['params_to_plot']:
        raise ValueError("Select at least one parameter in params_to_plot")
    if spec.name == 'generate_perturbation_heat_map':
        epc_params = LtSimConfiguration().inverse_epc_model_param_mapping
        other = [p for p in kwargs['params_to_plot'] if p.rsplit('_', 1)[0] not in epc_params]
        if other:
            raise ValueError(f"Heat map only supports HEMT model params; use the bar plots for {other}")

    meas_res = copy.deepcopy(result.meas_res)  # ResultPlotting pops the base run from the dict
    if spec.name == 'plot_wfm_from_all_runs':
        # Runs are labelled by their first changed param, so the base run is left out
        meas_res = {k: v for k, v in meas_res.items() if v['config_changes']}
    if spec.name in WAVEFORM_PLOTS:
        if not _waveform_names(result):
            raise ValueError(f"No waveforms were derived. Save the nodes they need, e.g. "
                             f"{', '.join(default_saved_traces(1))} for device 1")
    if 'base_config' in inspect.signature(spec.func).parameters:
        kwargs['base_config'] = result.base_config
    with contextlib.chdir(REPO_ROOT):  # ResultPlotting saves into ./plots
        spec.func(meas_res=meas_res, **kwargs)
