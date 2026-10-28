"""Tkinter front end for the parallel GaN LTspice studies.

Only widgets and flow control live here; all logic is in SimWorkflow.py.
Tab 1 sets up and runs a study, tab 2 configures and launches ResultPlotting plots.
Run from anywhere: python "01 Code/SimGUI.py"
"""
import gc
import queue
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import SimWorkflow as W

SIM_TYPE_LABELS = {
    'Sweep (one parameter at a time)': 'sweep',
    'Perturbation (one parameter at a time)': 'perturbation',
    '2-param sweep (all combinations)': 'multiparam',
}
POLL_MS = 200


class ScrollableFrame(ttk.Frame):
    """Frame with vertical and horizontal scrollbars; put widgets in .inner"""
    def __init__(self, parent, height):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        vbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        hbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        vbar.grid(row=0, column=1, sticky=tk.NS)
        hbar.grid(row=1, column=0, sticky=tk.EW)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Mouse wheel scrolls only while the pointer is over this frame
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class SimGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parallel GaN Simulation GUI")
        self.root.geometry("1050x900")

        # State
        self.sim_path_var = tk.StringVar(value=str(W.DEFAULT_SIM_PATH))
        self.base_config = {}            # last loaded config, used as type template
        self.config_vars = {}            # config key -> StringVar
        self.config_file_var = tk.StringVar(value="(no config loaded)")
        self.n_devices_var = tk.StringVar(value="4")
        self.sim_type_var = tk.StringVar(value=list(SIM_TYPE_LABELS)[0])
        self.k_perturb_var = tk.StringVar(value="0.1")
        self.study_rows = []             # list of dicts of row widgets/vars
        self.node_names = []
        self.elapsed_var = tk.StringVar(value="Elapsed: 0.0 s")
        self.progress_var = tk.StringVar(value="Completed: 0 / 0   Pending: 0   Failed: 0")
        self.result = None
        self.plot_arg_widgets = {}
        self.plot_status_var = tk.StringVar(value="")

        # Worker thread communication
        self._worker = None
        self._msg_queue = queue.Queue()
        self._run_start = None

        # Layout
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.sim_tab = ttk.Frame(self.notebook)
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_tab, text="Simulation")
        self.notebook.add(self.plot_tab, text="Plots")

        self.create_sim_path_section()
        self.create_config_section()
        self.create_nodes_section()
        self.create_study_section()
        self.create_run_section()
        self.create_plot_tab()

        default_cfg = W.CONFIG_DIR / "base_config_4dev.json"
        if default_cfg.exists():
            self._apply_config(W.load_base_config(default_cfg), default_cfg.name)
        self.scan_nodes()

    # ==================== Tab 1 / 1) Simulation file ====================
    def create_sim_path_section(self):
        frame = ttk.LabelFrame(self.sim_tab, text="1) Simulation file (.asc)", padding=8)
        frame.pack(fill=tk.X, padx=6, pady=(6, 0))
        ttk.Entry(frame, textvariable=self.sim_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X,
                                                                              expand=True, padx=(0, 8))
        ttk.Button(frame, text="Browse...", command=self.browse_sim_path).pack(side=tk.LEFT)

    def browse_sim_path(self):
        current = Path(self.sim_path_var.get())
        path = filedialog.askopenfilename(title="Select LTspice schematic", initialdir=current.parent,
                                          filetypes=[("LTspice schematic", "*.asc"), ("All files", "*.*")])
        if path:
            self.sim_path_var.set(path)
            self.scan_nodes()

    # ==================== Tab 1 / 2) Base configuration ====================
    def create_config_section(self):
        frame = ttk.LabelFrame(self.sim_tab, text="2) Base configuration", padding=8)
        frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        bar = ttk.Frame(frame)
        bar.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(bar, text="Load JSON...", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(bar, text="Save JSON as...", command=self.save_config).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(bar, text="  or defaults for N devices:").pack(side=tk.LEFT)
        ttk.Spinbox(bar, from_=1, to=W.TOTAL_DRAWN_DEVICES, width=4,
                    textvariable=self.n_devices_var).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(bar, text="Generate", command=self.generate_default_config).pack(side=tk.LEFT)
        ttk.Label(bar, textvariable=self.config_file_var, foreground="gray35").pack(side=tk.RIGHT)

        self.config_area = ScrollableFrame(frame, height=230)
        self.config_area.pack(fill=tk.X)

    def load_config(self):
        path = filedialog.askopenfilename(title="Load base configuration", initialdir=W.CONFIG_DIR,
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self._apply_config(W.load_base_config(path), Path(path).name)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def save_config(self):
        try:
            cfg = self._read_config_fields()
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return
        W.CONFIG_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(title="Save base configuration", initialdir=W.CONFIG_DIR,
                                            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            W.save_base_config(cfg, path)
            self.config_file_var.set(Path(path).name)

    def generate_default_config(self):
        try:
            cfg = W.build_default_base_config(int(self.n_devices_var.get()))
        except Exception as exc:
            messagebox.showerror("Generate failed", str(exc))
            return
        self._apply_config(cfg, f"(defaults, {cfg['N_devices']} devices)")

    def _apply_config(self, cfg, source_name):
        self.base_config = cfg
        self.config_file_var.set(source_name)
        self._populate_config_fields()
        self._reset_study_rows()
        self.scan_nodes()

    def _populate_config_fields(self):
        inner = self.config_area.inner
        for child in inner.winfo_children():
            child.destroy()
        self.config_vars = {}
        global_keys, per_device, n_devices = W.split_config_keys(self.base_config)

        # Global parameters on the first row
        col = 0
        for key in global_keys:
            ttk.Label(inner, text=key).grid(row=0, column=col, sticky=tk.E, padx=(6, 2), pady=2)
            self.config_vars[key] = tk.StringVar(value=str(self.base_config[key]))
            ttk.Entry(inner, textvariable=self.config_vars[key], width=12).grid(row=0, column=col + 1, sticky=tk.W)
            col += 2

        # Per-device parameters as a table: one row per parameter, one column per device
        ttk.Label(inner, text="Parameter", font=("TkDefaultFont", 9, "bold")).grid(row=1, column=0, sticky=tk.W,
                                                                                  padx=6, pady=(8, 2))
        for i in range(1, n_devices + 1):
            ttk.Label(inner, text=f"Device {i}", font=("TkDefaultFont", 9, "bold")).grid(row=1, column=i,
                                                                                         pady=(8, 2))
        for row, (base, keys) in enumerate(per_device.items(), start=2):
            ttk.Label(inner, text=base).grid(row=row, column=0, sticky=tk.W, padx=6)
            for i, key in keys.items():
                self.config_vars[key] = tk.StringVar(value=str(self.base_config[key]))
                ttk.Entry(inner, textvariable=self.config_vars[key], width=12).grid(row=row, column=i, padx=1, pady=1)

    def _read_config_fields(self):
        raw = {key: var.get() for key, var in self.config_vars.items()}
        return W.parse_config_values(raw, self.base_config)

    # ==================== Tab 1 / 3) Saved nodes ====================
    def create_nodes_section(self):
        frame = ttk.LabelFrame(self.sim_tab, text="3) Nodes to save", padding=8)
        frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        bar = ttk.Frame(frame)
        bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(bar, text="Scan .asc", command=self.scan_nodes).pack(side=tk.LEFT)
        ttk.Button(bar, text="Select defaults", command=self.select_default_nodes).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar, text="Select all", command=lambda: self.nodes_listbox.select_set(0, tk.END)).pack(
            side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar, text="Clear", command=lambda: self.nodes_listbox.select_clear(0, tk.END)).pack(
            side=tk.LEFT, padx=(6, 0))
        ttk.Label(bar, text="Traces needed by the .meas commands are always saved.",
                  foreground="gray35").pack(side=tk.RIGHT)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.X)
        self.nodes_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=7, exportselection=False)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.nodes_listbox.yview)
        self.nodes_listbox.configure(yscrollcommand=scroll.set)
        self.nodes_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll.pack(side=tk.LEFT, fill=tk.Y)

    def scan_nodes(self):
        n_devices = int(self.base_config.get('N_devices', self.n_devices_var.get()))
        try:
            self.node_names = W.scan_asc_nodes(self.sim_path_var.get(), n_devices)
        except Exception as exc:
            messagebox.showerror("Scan failed", str(exc))
            return
        self.nodes_listbox.delete(0, tk.END)
        for name in self.node_names:
            self.nodes_listbox.insert(tk.END, name)
        self.select_default_nodes()

    def select_default_nodes(self):
        n_devices = int(self.base_config.get('N_devices', self.n_devices_var.get()))
        defaults = set(W.default_saved_traces(n_devices))
        self.nodes_listbox.select_clear(0, tk.END)
        for idx, name in enumerate(self.node_names):
            if name in defaults:
                self.nodes_listbox.select_set(idx)

    # ==================== Tab 1 / 4) Study type and varied parameters ====================
    def create_study_section(self):
        frame = ttk.LabelFrame(self.sim_tab, text="4) Study", padding=8)
        frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        bar = ttk.Frame(frame)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Type:").pack(side=tk.LEFT)
        combo = ttk.Combobox(bar, textvariable=self.sim_type_var, values=list(SIM_TYPE_LABELS),
                             state="readonly", width=38)
        combo.pack(side=tk.LEFT, padx=(4, 12))
        combo.bind("<<ComboboxSelected>>", lambda e: self._reset_study_rows())
        self.k_perturb_label = ttk.Label(bar, text="Default tolerance (p.u.):")
        self.k_perturb_entry = ttk.Entry(bar, textvariable=self.k_perturb_var, width=8)
        self.add_row_button = ttk.Button(bar, text="Add parameter", command=self.add_study_row)
        self.add_row_button.pack(side=tk.RIGHT)

        self.study_header = ttk.Frame(frame)
        self.study_header.pack(fill=tk.X, pady=(6, 0))
        self.study_rows_frame = ttk.Frame(frame)
        self.study_rows_frame.pack(fill=tk.X)

    def _sim_type(self):
        return SIM_TYPE_LABELS[self.sim_type_var.get()]

    def _reset_study_rows(self):
        for row in self.study_rows:
            row['frame'].destroy()
            row.clear()  # break widget<->row reference cycles (see start_run)
        self.study_rows = []
        for child in self.study_header.winfo_children():
            child.destroy()

        is_perturb = self._sim_type() == 'perturbation'
        if is_perturb:
            self.k_perturb_label.pack(side=tk.LEFT)
            self.k_perturb_entry.pack(side=tk.LEFT, padx=(4, 0))
            headers = [("Parameter", 22), ("Base value", 14), ("Tolerance", 10), ("rel/abs", 8)]
        else:
            self.k_perturb_label.pack_forget()
            self.k_perturb_entry.pack_forget()
            headers = [("Parameter", 22), ("Base value", 14), ("Start", 10), ("Stop", 10), ("Points", 7),
                       ("Log", 5)]
        for text, width in headers:
            ttk.Label(self.study_header, text=text, width=width, font=("TkDefaultFont", 9, "bold")).pack(
                side=tk.LEFT, padx=2)

        for _ in range(2 if self._sim_type() == 'multiparam' else 1):
            self.add_study_row()

    def add_study_row(self):
        sim_type = self._sim_type()
        if sim_type == 'multiparam' and len(self.study_rows) >= 2:
            return
        frame = ttk.Frame(self.study_rows_frame)
        frame.pack(fill=tk.X, pady=1)
        row = {'frame': frame, 'param': tk.StringVar(), 'base': tk.StringVar()}

        param_combo = ttk.Combobox(frame, textvariable=row['param'], state="readonly", width=22,
                                   values=W.variable_params(self.base_config))
        param_combo.pack(side=tk.LEFT, padx=2)
        param_combo.bind("<<ComboboxSelected>>", lambda e, r=row: self._show_base_value(r))
        ttk.Label(frame, textvariable=row['base'], width=14, foreground="gray35").pack(side=tk.LEFT, padx=2)

        if sim_type == 'perturbation':
            row['amount'] = tk.StringVar(value=self.k_perturb_var.get())
            row['mode'] = tk.StringVar(value='rel')
            ttk.Entry(frame, textvariable=row['amount'], width=10).pack(side=tk.LEFT, padx=2)
            ttk.Combobox(frame, textvariable=row['mode'], values=W.PERTURB_MODES, state="readonly",
                         width=6).pack(side=tk.LEFT, padx=2)
        else:
            row['start'], row['stop'] = tk.StringVar(), tk.StringVar()
            row['points'], row['log'] = tk.StringVar(value="5"), tk.BooleanVar(value=False)
            for key, width in (('start', 10), ('stop', 10), ('points', 7)):
                ttk.Entry(frame, textvariable=row[key], width=width).pack(side=tk.LEFT, padx=2)
            ttk.Checkbutton(frame, variable=row['log']).pack(side=tk.LEFT, padx=(8, 2))

        if sim_type != 'multiparam':
            ttk.Button(frame, text="Remove", width=8,
                       command=lambda r=row: self._remove_study_row(r)).pack(side=tk.LEFT, padx=(8, 0))
        self.study_rows.append(row)
        self.add_row_button.configure(state=tk.DISABLED if sim_type == 'multiparam' else tk.NORMAL)

    def _remove_study_row(self, row):
        row['frame'].destroy()
        self.study_rows.remove(row)
        row.clear()

    def _show_base_value(self, row):
        var = self.config_vars.get(row['param'].get())
        row['base'].set(var.get() if var else "")

    def _read_study_rows(self):
        rows = []
        for row in self.study_rows:
            values = {k: v.get() for k, v in row.items() if isinstance(v, tk.Variable) and k != 'base'}
            rows.append(values)
        return rows

    # ==================== Tab 1 / 5) Run ====================
    def create_run_section(self):
        frame = ttk.LabelFrame(self.sim_tab, text="5) Run", padding=8)
        frame.pack(fill=tk.X, padx=6, pady=6)
        self.run_button = ttk.Button(frame, text="Run", width=14, command=self.start_run)
        self.run_button.pack(side=tk.LEFT)
        status = ttk.Frame(frame)
        status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))
        ttk.Label(status, textvariable=self.elapsed_var).pack(anchor=tk.W)
        ttk.Label(status, textvariable=self.progress_var).pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(status, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(2, 0))

    def start_run(self):
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            base_config = self._read_config_fields()
            W.validate_base_config(base_config)
            study = W.build_study(self._sim_type(), self._read_study_rows(), self.k_perturb_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        traces = [self.node_names[i] for i in self.nodes_listbox.curselection()]

        # Collect garbage tk Variables here: if the GC frees them inside the worker thread,
        # tkinter raises "main thread is not in main loop"
        gc.collect()
        self.run_button.configure(state=tk.DISABLED)
        self._run_start = time.monotonic()
        self._update_progress(0, 0, 0)
        self._worker = threading.Thread(target=self._run_worker, daemon=True,
                                        args=(self.sim_path_var.get(), base_config, traces, study))
        self._worker.start()
        self.root.after(POLL_MS, self._poll_worker)

    def _run_worker(self, sim_path, base_config, traces, study):
        """Runs in the worker thread: only talks to the GUI through the queue."""
        try:
            result = W.run_study(sim_path, base_config, traces, study,
                                 on_progress=lambda c, f, t: self._msg_queue.put(('progress', c, f, t)))
            self._msg_queue.put(('done', result))
        except Exception:
            self._msg_queue.put(('error', traceback.format_exc()))

    def _poll_worker(self):
        finished = False
        while not self._msg_queue.empty():
            msg = self._msg_queue.get()
            if msg[0] == 'progress':
                self._update_progress(*msg[1:])
            elif msg[0] == 'done':
                finished = True
                self._on_run_done(msg[1])
            elif msg[0] == 'error':
                finished = True
                messagebox.showerror("Simulation failed", msg[1])
        if self._run_start is not None:
            self.elapsed_var.set(f"Elapsed: {time.monotonic() - self._run_start:.1f} s")
        if finished:
            self._run_start = None
            self.run_button.configure(state=tk.NORMAL)
        else:
            self.root.after(POLL_MS, self._poll_worker)

    def _update_progress(self, completed, failed, total):
        self.progress_var.set(f"Completed: {completed} / {total}   Pending: {total - completed}   Failed: {failed}")
        self.progress_bar.configure(maximum=max(total, 1), value=completed)

    def _on_run_done(self, result):
        self._update_progress(result.ok + result.failed, result.failed, result.total)
        if not result.meas_res:
            messagebox.showwarning("No results", "No simulation produced results. Check the logs in temp/.")
            return
        self.result = result
        self._populate_plot_tab()
        self.notebook.select(self.plot_tab)

    # ==================== Tab 2: Plots ====================
    def create_plot_tab(self):
        self.result_summary_var = tk.StringVar(value="Run a study first.")
        ttk.Label(self.plot_tab, textvariable=self.result_summary_var, padding=8).pack(anchor=tk.W)

        body = ttk.Frame(self.plot_tab)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        list_frame = ttk.LabelFrame(body, text="Plot", padding=8)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.plot_listbox = tk.Listbox(list_frame, width=36, exportselection=False)
        self.plot_listbox.pack(fill=tk.Y, expand=True)
        self.plot_listbox.bind("<<ListboxSelect>>", lambda e: self._show_plot_editor())

        editor = ttk.LabelFrame(body, text="Arguments", padding=8)
        editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.plot_args_area = ScrollableFrame(editor, height=600)
        self.plot_args_area.pack(fill=tk.BOTH, expand=True)
        bar = ttk.Frame(editor)
        bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(bar, text="Plot", width=14, command=self.launch_plot).pack(side=tk.LEFT)
        ttk.Button(bar, text="Reset arguments", command=self._show_plot_editor).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bar, text="Close all figures", command=lambda: plt.close('all')).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(bar, textvariable=self.plot_status_var, foreground="gray35").pack(side=tk.LEFT, padx=(12, 0))

    def _populate_plot_tab(self):
        r = self.result
        self.result_summary_var.set(
            f"Study: {r.sim_type}   Runs: {r.total} ({r.ok} ok, {r.failed} failed)   "
            f"Varied: {', '.join(r.params_to_plot)}   Time: {r.elapsed_s:.1f} s")
        self.available_plots = W.available_plots(r)
        self.plot_listbox.delete(0, tk.END)
        for spec in self.available_plots:
            self.plot_listbox.insert(tk.END, spec.title)
        self.plot_listbox.select_set(0)
        self._show_plot_editor()

    def _selected_plot(self):
        sel = self.plot_listbox.curselection()
        return self.available_plots[sel[0]] if sel and self.result else None

    def _show_plot_editor(self):
        inner = self.plot_args_area.inner
        for child in inner.winfo_children():
            child.destroy()
        self.plot_arg_widgets = {}
        spec = self._selected_plot()
        if spec is None:
            return
        ttk.Label(inner, text=f"{spec.title}  ({spec.name})", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(inner, text=spec.note, foreground="gray35").pack(anchor=tk.W, pady=(0, 6))

        for arg in W.plot_args(self.result, spec.name):
            frame = ttk.Frame(inner)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=arg.name, width=16, font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT,
                                                                                            anchor=tk.N)
            if arg.kind == 'multi':
                lb = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=False, width=28,
                                height=max(1, min(len(arg.choices), 6)))
                for idx, choice in enumerate(arg.choices):
                    lb.insert(tk.END, choice)
                    if choice in arg.default:
                        lb.select_set(idx)
                lb.pack(side=tk.LEFT)
                self.plot_arg_widgets[arg.name] = (arg, lb)
            elif arg.kind == 'choice':
                var = tk.StringVar(value=arg.default)
                ttk.Combobox(frame, textvariable=var, values=arg.choices, state="readonly", width=12).pack(side=tk.LEFT)
                self.plot_arg_widgets[arg.name] = (arg, var)
            elif arg.kind == 'bool':
                var = tk.BooleanVar(value=arg.default)
                ttk.Checkbutton(frame, variable=var).pack(side=tk.LEFT)
                self.plot_arg_widgets[arg.name] = (arg, var)
            else:
                var = tk.StringVar(value=arg.default)
                ttk.Entry(frame, textvariable=var, width=14).pack(side=tk.LEFT)
                self.plot_arg_widgets[arg.name] = (arg, var)
            ttk.Label(frame, text=arg.help, foreground="gray35").pack(side=tk.LEFT, padx=(8, 0), anchor=tk.N)

    def _read_plot_args(self):
        values = {}
        for name, (arg, widget) in self.plot_arg_widgets.items():
            if arg.kind == 'multi':
                values[name] = [arg.choices[i] for i in widget.curselection()]
            else:
                values[name] = widget.get()
        return values

    def launch_plot(self):
        spec = self._selected_plot()
        if spec is None:
            return
        values = self._read_plot_args()
        try:
            W.launch_plot(self.result, spec.name, values)
        except (ValueError, NotImplementedError) as exc:
            messagebox.showerror("Cannot plot", str(exc))
            return
        except Exception:
            messagebox.showerror("Plot failed", traceback.format_exc())
            return
        self.plot_status_var.set("Saved to plots/" if values.get('save_figs') else "Opened in a new window")


def main():
    plt.ion()  # plt.show() inside ResultPlotting returns immediately; figure windows share the Tk loop
    root = tk.Tk()
    SimGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
