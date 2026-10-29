# CLAUDE.md

This file gives Claude Code (claude.ai/code) guidance for working in this repository.

## Project overview

This is research code for an ESA study of **paralleled GaN HEMTs** (EPC2305) in a half-bridge/buck. It covers the whole chain: LTspice simulation of current sharing between N parallel devices, analytical and thermal models, a buck/double-pulse-test (DPT) board driven by an STM32G474, a PC GUI to control that board, and a scope waveform grabber for bench measurements.

The folders are numbered and contain spaces, so always quote paths:

| Folder | What it is |
|---|---|
| `01 Code/` | Python simulation automation (PyLTSpice/spicelib), plotting, analytical models |
| `02 Simulations/` | LTspice schematics (`.asc`), symbols (`.asy`) and the EPC GaN model library |
| `03 PC GUI/` | Tkinter GUI + pyserial link to the STM32 board |
| `04 STM32 Code/` | STM32CubeIDE projects. `02 PWM and ADC/Micro_buck` is the current firmware; `01 Debug Tries` is an early scratch project |
| `05 Oscilloscope Waveform Extractor/` | Grabs waveforms from an R&S MXO44 scope over LAN (RsInstrument) and saves them as CSV/NPZ |
| `temp/`, `plots/` | Generated outputs (gitignored) |

There are no tests and no build system for the Python code. `requirements.txt` at the repo root lists the packages. Python ≥3.12 is assumed: the code uses nested f-string quotes and `X | None` hints. Dependencies: `PyLTSpice`, `spicelib`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pyserial`, `RsInstrument`. You also need LTspice installed (Windows).

## Running things

```powershell
# Simulation GUI (recommended way to run studies; works from any CWD)
python "01 Code/SimGUI.py"

# Same flow as a script, configured by editing the settings at its top (works from any CWD)
python "01 Code/ParallelGaN_script.py"

# PC GUI (run from its folder so `from serial_comm import SerialManager` resolves)
cd "03 PC GUI"; python ESA_ParGan_GUI.py

# Scope waveform acquirer (has its own small Tk GUI; scope IP is hard-coded at the top)
python "05 Oscilloscope Waveform Extractor/RSMXO44_wfm_acquirer.py"

# Foster thermal network fit for EPC2305 Zth(j-c); prints .param lines for LTspice
python "01 Code/ThermalNetwork.py"
```

You build and flash the firmware in **STM32CubeIDE** (`Micro_buck.ioc`, `Micro_buck Debug.launch`). The `Debug/` build output (`.o`, `.elf`, `.map`, makefiles) is committed. Don't edit those files by hand.

## Simulation pipeline (`01 Code`)

The scripts are configured by editing them, not through a CLI. `ParallelGaN_script.py` is a thin driver over `SimWorkflow` (settings at the top: base config JSON, `SIM_TYPE`, `SWEEPS`, `PERTURBATIONS`). The flow, implemented in `SimWorkflow.run_study`:

1. The base config: global `V_DC`, `I_DC`, `N_devices`, plus **per-device parameters suffixed `_<i>`** (`R_drain_1`, `Lg_uncommon_2`, `Vdrv_on_3`, …). HEMT model parameters are pulled from the library and stored by their **SPICE names** (`k2_1`, `A1_1`, `rpara_1`, …).
2. The study type (`'sweep'` | `'perturbation'` | `'multiparam'`) and its parameters, registered with `add_param_sweep_with_range` / `add_param_perturbation`.
3. For each config from `yield_cnfgs_sequentially` (the base config always comes first), calls `prepare_netlist_for_sim` and then `SimRunner.run`, with up to 16 parallel LTspice runs.
4. Reads the `.raw` files (waveforms) and `.log` files (`.meas` results) into `meas_res[<run name>]` (see *Results structure* below).
5. Plots through `ResultPlotting.py` (sweep plots, perturbation heat map/bar plots, 2-param heat maps, overlaid rise/fall edge waveforms).

`SimConfiguration.LtSimConfiguration` holds the main logic:

- **Friendly name ↔ SPICE name mapping** is in `epc_model_param_mapping` (e.g. `Vgs_th`→`k2`, `k_gm_factor`→`A1`, `Rds_on_base`→`rpara`, `Rgate_int`→`rg_value`). The `add_*` methods accept the friendly name (`'Vgs_th_1'`) and convert it to the SPICE name (`'k2_1'`). Any parameter must already exist in `base_cnfg_dict`, or the method raises.
- **Per-device model variants:** when an EPC model parameter of device *i* changes, the device is renamed `EPC2305_<i>`. `reset_and_trim_netlist` then writes a modified `.subckt` copy to `temp/<asc stem>_<sim_index>_EPC2305_<i>.lib` and adds a `.lib` line to the netlist. The source library `02 Simulations/00 ComonLibs/EPCGaNLibrary.lib` is never modified on this path. (`generate_modif_lib`, which rewrites the library in place after a marker comment, is legacy and its call is commented out.)
- **Netlist trimming:** the schematic `ParGan_V4.asc` draws `TOTAL_DRAWN_DEVICES = 10` branches. Branches `N_devices+1..10` are removed by component name (`X{i}`, `XG{i}`, `Vgdrv{i}`, `R{i}`, `R_{i-1}{i}`). If you rename components in the `.asc`, this code must change with them.
- `update_netlist_params` sets `.tran` (2 switching periods) and writes every config key as `.param`. It also rebuilds each gate driver `PULSE(...)` source and each loss behavioural source `B{i}`.
- `introduce_meas_commands` adds the per-device `.meas` lines: `X{i}_E_on`, `_E_off`, `_vds_pk`, `_vgs_pk`, `_rms_i`, `_MaxTj_on/off` and `_cond_ratio`. Plotting code finds results by these names. LTspice reports them in lowercase (`x1_e_on`).
- `define_selective_saving` writes the `.save` lines and **always adds `meas_required_traces()`**. When `.save` is present, LTspice keeps only the saved quantities. It aborts the transient ("Current not found") if a `.meas` uses an unsaved one. If you add or change a `.meas` expression, add its traces to `meas_required_traces()`.
- Node and current names such as `V(G{i})`, `I(X{i}:L_drain)`, `X{i}:hemt_s_noLs`, `xg{i}:Lg_minusLcommonsrc` and `V(Tj{i})` are tied to the hierarchy in `hemt_cell_V*.asc` / `gate_cell_V1.asc` / `ThermalNetwork_EPC2305.asc` in `00 ComonLibs`.

## Simulation GUI (`01 Code/SimGUI.py` + `SimWorkflow.py`)

The GUI runs the same flow as `ParallelGaN_script.py`. The code is split into strict layers:

| File | Role | Rule |
|---|---|---|
| `SimGUI.py` | Tkinter view and flow control only | No simulation or plot logic. Passes raw entry text to `SimWorkflow` and shows the `ValueError` messages it raises |
| `SimWorkflow.py` | Logic layer used by the GUI | No tkinter imports. Orchestrates the steps; delegates to the two modules below |
| `SimConfiguration.py` | Netlist, config and `.meas` logic (`LtSimConfiguration`) | Shared with the script |
| `ResultPlotting.py` | Plot functions | Shared with the script; the GUI calls them unchanged |

Put new logic in `SimWorkflow.py`, or in `LtSimConfiguration` / `ResultPlotting` when the script should get it too. Never put it in `SimGUI.py`.

**Tab 1, Simulation.** It has five sections stacked vertically:

1. **`.asc` file picker.** Defaults to `ParGan_V4.asc`.
2. **Base config editor.**
   - The config is loaded from and saved to JSON files in `01 Code/configs/`. `base_config_4dev.json` loads at startup.
   - **Generate** builds the script's defaults for N devices with `build_default_base_config`, reading HEMT params from the `.lib`.
   - Fields are rebuilt from the dict's keys: global keys in one row, then a table with one row per parameter and one column per device (`split_config_keys`).
   - Values may use engineering notation (`2.5n`). `parse_config_values` converts them back using the loaded dict's types.
3. **Nodes to save.**
   - `scan_asc_nodes` lists `V(<net>)` for every `FLAG`, plus the known `hemt_cell` internal traces (`HEMT_CELL_TRACES`) of each `X{i}` instance.
   - Anything indexed above N_devices is dropped, because those branches get trimmed.
   - `default_saved_traces` pre-selects what the derived waveforms need.
4. **Study type.**
   - The types are sweep, perturbation and 2-param (multiparam, all combinations).
   - The rows of varied parameters change with the type. `build_study` turns them into a `Study`.
5. **Run button.**
   - `run_study` runs in a worker thread. It reports `on_progress(completed, failed, total)` through a `queue.Queue` that the GUI polls with `after()`, which drives the elapsed time, the counters and the progress bar.
   - **Stop** sets the `stop_event`: no new runs are launched, the running ones finish, and the partial result is kept (`StudyResult.stopped`).
   - Tk must only be touched from the main thread. Tk variables freed by the garbage collector in the worker thread cause `RuntimeError: main thread is not in main loop`. That's why `start_run` calls `gc.collect()` and removed study rows are `clear()`ed.

**Tab 2, Plots.** It is a generic editor over `SimWorkflow.PLOTS`:

- 2-param studies get `generate_multiparam_heat_maps` (KPI over the grid, one heat map per device) and `plot_wfm_2param` (hue = first param, lightness = second param, read off a 2D colour key; `swap_axes` exchanges them; clicking a curve or a key swatch highlights that run in every subplot and greys out the rest) and `plot_wfm_2param_3d` (3D waterfall per waveform: x = time, depth = second param, colour = first param, optional translucent `fill` ribbons; 3D axes don't clip, so the edge window crops the data). `plot_wfm_from_all_runs` (one legend entry per run) is only offered for sweep/perturbation.
- After a 2-param (multiparam) study, `run_study` also writes `results/<asc stem>_multiparam_<timestamp>.csv` with `export_meas_csv`: one row per run (base run first), columns = the two parameter values, then every `.meas` value. The path is in `StudyResult.meas_csv`.
- *Save results as...* / *Load results...* pickle the whole `StudyResult` to `results/` (gitignored) with `save_result` / `load_result`, so plots can be redone without simulating.
- `StudyResult` also carries a time breakdown (`sim_s`, `longest_sim_s`, `read_s`), shown in the summary line.

- Each `PlotSpec` wraps one `ResultPlotting` function and lists the study types it applies to.
- `plot_args()` reads the function's **real signature**:
  - `meas_res` and `base_config` (`AUTO_ARGS`) are filled from the `StudyResult`.
  - Every other argument becomes a `PlotArg` (`multi` / `choice` / `bool` / `text`), with choices and defaults taken from the result. The GUI draws a widget per kind.
- `launch_plot()` converts the values and calls the function:
  - It deep-copies `meas_res`, because ResultPlotting pops the base run.
  - It drops the base run for `plot_wfm_from_all_runs`.
  - It converts `skip_branches` to the format each function expects: substring `'x3_'` or int `3`.
  - It runs inside `chdir(REPO_ROOT)` so `save_figs` writes to `plots/`.
- `SimGUI.main()` sets `plt.ion()`, so the `plt.show()` calls inside ResultPlotting open non-blocking windows.
- **To add a plot:** write the function in `ResultPlotting.py` and add a `PlotSpec` to `PLOTS`. Argument names already known to `plot_args()` get proper widgets. Unknown ones fall back to a text field, parsed as a number when possible.

**Results structure.** `StudyResult.meas_res[run_name]` contains:

- `config_changes`
- `meas results`: the `.meas` values.
- `traces`: every saved node, keyed by its expression.
- `waveforms`: `time` plus the derived per-device `i_drain_i`, `v_gs_i`, `v_ds_i`, `Tj_i`. These are built only when their source traces were saved.

Failed runs are dropped.

**Gotchas.**

- `run_study` registers `02 Simulations/00 ComonLibs` as an AscEditor library path. Without it, `gate_cell_V1.asy` is only found when the CWD is the repo root.
- `run_study` also `chdir`s to the repo root while it runs. That changes the CWD of the whole process.
- Every LTspice run uses all CPU threads. `PARALLEL_SIMS = cpu_count // 4` (8 on 32 threads). At 16, runs slowed from ~3 s to over 30 s and were killed by `SIM_TIMEOUT_S`.
- LTspice rejects 0 Ω resistors ("Resistance must not be zero"). `build_study` and `validate_base_config` refuse them.

## Other analysis scripts

`GanLossEstimation.py` is a standalone analytical loss model (`GaNDevice`, datasheet polynomial fits) for EPC2305. `sim_runner_asc_example.py` is a PyLTSpice usage example built on `02 Simulations/01 Examples/Batch_Test.asc`.

## STM32 firmware ↔ GUI protocol

The MCU is an STM32G474RE (Nucleo) on a 170 MHz clock. TIM1 CH3/CH3N generates complementary PWM with hardware deadtime. ADC2 (Iin on PA5, Io on PA6) and ADC5 (Vin on PA8, Vout on PA9) run in DMA mode. USART2 runs at 115200 baud.

Only edit inside the `/* USER CODE BEGIN … */ … /* USER CODE END … */` blocks of CubeMX-generated files. CubeMX regenerates everything else from the `.ioc`.

- **Telemetry (MCU→PC), every 100 ms:** `Vin:%.3fV Vo:%.3fV Iin:%.3fA Io:%.3fA\r\n`. With `SEND_BINARY_ADC_READINGS=1` it sends raw counts with a `b` suffix instead. `serial_comm.SerialManager._parse_telemetry_line` parses both formats. Sensor scaling (divider ratios, shunt × AMC gain × op-amp gain) lives in the `*_factor` constants in `main.c`, so keep those in step with the board design.
- **Commands (PC→MCU):** newline-terminated ASCII, split on `:` in `Parse_SerialCommand`:
  - `PWM:DUTY:<pct>`, `PWM:FREQ:<Hz>`, `PWM:DT:<ns>`. Deadtime is rounded up to 5.88 ns ticks and clamped to 32 ticks (≈188 ns). The GUI shows the same quantisation to the user.
  - `BO:ENABLE:<0|1>`, `MODE:BO:…`, `MODE:DPT:<periodic|single_shot>`
  - `DPT:SINGLE_SHOT`, `DPT:SET;key=val;…`. These are parsed but are **still TODO stubs**: DPT operation isn't implemented yet.
- Known mismatches to watch for:
  - The GUI sends `BO:SET;…` and `MODE:DPT:active`, which the firmware doesn't handle.
  - The firmware RX buffer is `CMD_BUFFER_SIZE 64`. A long `DPT:SET;…` line overflows it and is silently dropped.

## Conventions

- Hard-coded settings sit at the top of each script, as constants (`V_DC`, `SCOPE_IP`, …). Change them there; don't add CLI argument parsing.
- LTspice is auto-detected (`SimRunner(simulator=None)`). Don't hard-code a path to the LTspice executable.
- Device indexing is 1-based everywhere (`_1 … _N`). Some plotting `skip_branches` arguments take strings like `'x3'` and others take integers. `SimWorkflow._convert_skip_branches` deals with this for the GUI.
- Default circuit values live in `SimWorkflow.build_default_base_config` and in the JSON files in `01 Code/configs/`. `ParallelGaN_script.py` has no defaults of its own.
- HEMT model params are stored by SPICE name (`k2_1`) but shown in the GUI and plot labels by friendly name (`Vgs_th_1`). Convert with `SimWorkflow.display_name` / `config_key` (or `ResultPlotting.friendly_name`).
- Config values are plain SI numbers. The GUI labels show the unit from `SimWorkflow.PARAM_UNITS`. When you add a config parameter, add its unit there too.
