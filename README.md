# ESA_parallel_GaN
Private repo for ESA analysis of parallel GaN 

## Setup

1. Install Python 3.12 or newer and LTspice (Windows).
2. Install the Python packages from the repo root:

   ```powershell
   pip install -r requirements.txt
   ```

## How to run a study

Start the simulation GUI (it works from any folder):

```powershell
python "01 Code/SimGUI.py"
```

### Tab 1: Simulation

Work through the five numbered sections from top to bottom:

1. **Simulation file.** `ParGan_V4.asc` is selected by default. Use *Browse...* to pick another schematic.
2. **Base configuration.** The operating point and the per-device parameters. `base_config_4dev.json` loads at startup.
   - *Load JSON...* / *Save JSON as...* switch between saved operating points in `01 Code/configs/`.
   - *Generate* builds the default values for the chosen number of devices.
   - Edit any value in the table. Engineering notation is accepted, e.g. `2.5n` or `200p`.
3. **Nodes to save.** These waveforms are kept in the results. *Select defaults* picks the ones the waveform plots need. The traces the measurements need are always saved.
4. **Study.** Pick the study type and add the parameters to vary:
   - *Sweep*: each parameter goes from *Start* to *Stop* in *Points* steps (tick *Log* for log spacing), one parameter at a time.
   - *Perturbation*: each parameter is changed by its *Tolerance* (relative or absolute), one at a time.
   - *2-param sweep*: every combination of the two parameters' sweep points.
5. **Run.** Starts the batch and shows the progress. *Stop* stops launching new simulations. Runs already started finish, and their results are kept.

When the run finishes, the GUI switches to the Plots tab.

### Tab 2: Plots

- Pick a plot on the left, adjust its arguments, and press *Plot*. Figures open in separate windows. Tick `save_figs` to write PNGs to `plots/` instead.
- *Save results as...* stores the whole study in `results/`. *Load results...* reopens it later, so you can redo plots without simulating again.

### Without the GUI

`01 Code/ParallelGaN_script.py` runs the same flow from a script. Edit the settings at the top of the file, then run:

```powershell
python "01 Code/ParallelGaN_script.py"
```
