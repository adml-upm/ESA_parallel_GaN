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

There are no tests, no build system for the Python code, and no requirements file. Python ≥3.12 is assumed: the code uses nested f-string quotes and `X | None` hints. Dependencies: `PyLTSpice`, `spicelib`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pyserial`, `RsInstrument`. You also need LTspice installed (Windows).

## Running things

```powershell
# LTspice sweep/perturbation study. Run from the REPO ROOT: sim_path ('./02 Simulations/...')
# and output_folder='temp' are relative to the CWD.
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

The scripts are configured by editing them, not through a CLI. `ParallelGaN_script.py` is the driver:

1. Builds `base_config_dict`: global `V_DC`, `I_DC`, `N_devices`, plus **per-device parameters suffixed `_<i>`** (`R_drain_1`, `Lg_uncommon_2`, `Vdrv_on_3`, …). HEMT model parameters are pulled from the library and stored by their **SPICE names** (`k2_1`, `A1_1`, `rpara_1`, …).
2. Picks `SIM_TYPE` (`'sweep'` | `'perturbation'`; `'multiparam'` is unimplemented) and registers parameters with `add_param_sweep_with_range` / `add_param_perturbation`.
3. For each config from `yield_cnfgs_sequentially` (the base config always comes first), calls `prepare_netlist_for_sim` and then `SimRunner.run`, with up to 16 parallel LTspice runs.
4. Reads the `.raw` files (waveforms) and `.log` files (`.meas` results) into `all_meas_results[<run name>]` = `{config_changes, waveforms, 'meas results'}`.
5. Plots through `ResultPlotting.py` (sweep plots, perturbation heat map/bar plots, overlaid rise/fall edge waveforms).

`SimConfiguration.LtSimConfiguration` holds the main logic:

- **Friendly name ↔ SPICE name mapping** is in `epc_model_param_mapping` (e.g. `Vgs_th`→`k2`, `k_gm_factor`→`A1`, `Rds_on_base`→`rpara`, `Rgate_int`→`rg_value`). The `add_*` methods accept the friendly name (`'Vgs_th_1'`) and convert it to the SPICE name (`'k2_1'`). Any parameter must already exist in `base_cnfg_dict`, or the method raises.
- **Per-device model variants:** when an EPC model parameter of device *i* changes, the device is renamed `EPC2305_<i>`. `reset_and_trim_netlist` then writes a modified `.subckt` copy to `temp/<asc stem>_<sim_index>_EPC2305_<i>.lib` and adds a `.lib` line to the netlist. The source library `02 Simulations/00 ComonLibs/EPCGaNLibrary.lib` is never modified on this path. (`generate_modif_lib`, which rewrites the library in place after a marker comment, is legacy and its call is commented out.)
- **Netlist trimming:** the schematic `ParGan_V4.asc` draws `TOTAL_DRAWN_DEVICES = 10` branches. Branches `N_devices+1..10` are removed by component name (`X{i}`, `XG{i}`, `Vgdrv{i}`, `R{i}`, `R_{i-1}{i}`). If you rename components in the `.asc`, this code must change with them.
- `update_netlist_params` sets `.tran` (2 switching periods) and writes every config key as `.param`. It also rebuilds each gate driver `PULSE(...)` source and each loss behavioural source `B{i}`.
- `introduce_meas_commands` adds the per-device `.meas` lines: `X{i}_E_on`, `_E_off`, `_vds_pk`, `_vgs_pk`, `_rms_i`, `_MaxTj_on/off` and `_cond_ratio`. Plotting code finds results by these names.
- Node and current names such as `V(G{i})`, `I(X{i}:L_drain)`, `X{i}:hemt_s_noLs`, `xg{i}:Lg_minusLcommonsrc` and `V(Tj{i})` are tied to the hierarchy in `hemt_cell_V*.asc` / `gate_cell_V1.asc` / `ThermalNetwork_EPC2305.asc` in `00 ComonLibs`.

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
- Device indexing is 1-based everywhere (`_1 … _N`). Some plotting `skip_branches` arguments take strings like `'x3'` and others take integers.
