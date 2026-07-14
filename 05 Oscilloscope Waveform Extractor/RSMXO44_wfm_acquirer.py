from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
import time as pytime
import numpy as np
from RsInstrument import RsInstrument


# Static user configuration for VS Code execution.
SCOPE_IP = "169.254.232.100"
SCOPE_RESOURCE = f"TCPIP::{SCOPE_IP}::INSTR"  # Optional explicit resource, e.g. "TCPIP::192.168.1.100::INSTR"
CHANNELS = ("C1", "C2", "C3", "C4")
RECORD_LENGTH = 50_000
TIMEOUT_MS = 120_000
TRANSFER_TIMEOUT_MS = 15_000
OUTPUT_DIR = Path(__file__).parent / "SavedWfms"
BASE_FILENAME = "EsaParGan"
WRITE_CSV = True
WRITE_NPZ = True
AVERAGE_WAIT_SECONDS = 5.0
VOLTAGE_SUFFIX_OPTIONS = ("vh", "ih", "vl", "il")


def _resource_from_ip(ip: str) -> str:
	# R&S example-style VISA resource over Ethernet.
	return f"TCPIP::{ip}::INSTR"


def _safe_query_float(instr: RsInstrument, command: str) -> float | None:
	try:
		return float(instr.query_str(command).strip())
	except Exception:
		return None


def _channel_number(channel: str) -> int:
	text = channel.strip().upper()
	if not text.startswith("C") or not text[1:].isdigit():
		raise ValueError(f"Invalid channel '{channel}'. Use C1, C2, ...")
	return int(text[1:])


def _sync_voltage_label(label_var: tk.StringVar, suffix_var: tk.StringVar, number_var: tk.StringVar) -> None:
	suffix = suffix_var.get().strip()
	number = number_var.get().strip()
	if suffix and number:
		label_var.set(f"V_{suffix}_{number}")
	elif suffix:
		label_var.set(f"V_{suffix}_")
	else:
		label_var.set("V_")


def _query_channel_trace_float32(
	instr: RsInstrument,
	channel: str,
	transfer_timeout_ms: int,
) -> np.ndarray:
	ch_num = _channel_number(channel)
	previous_timeout = instr.visa_timeout
	instr.visa_timeout = max(int(previous_timeout), int(transfer_timeout_ms))
	try:
		# MXO44 waveform transfer by explicit channel query.
		data = instr.query_bin_or_ascii_float_list(f"CHANnel{ch_num}:WAV1:DATA?")
	finally:
		instr.visa_timeout = previous_timeout
	return np.asarray(data, dtype=np.float32)


def acquire_scope_waveforms(
	resource: str,
	channels: Iterable[str] = ("C1", "C2", "C3", "C4"),
	record_length: int = 50_000,
	timeout_ms: int = 120_000,
	transfer_timeout_ms: int = 600_000,
	average_wait_seconds: float = 4.0,
) -> dict[str, object]:
	instr = RsInstrument(resource, id_query=True, reset=False)
	instr.visa_timeout = timeout_ms

	try:
		idn = instr.query_str("*IDN?").strip()

		# instr.write("*RST")
		instr.write("FORM ASC")
		# instr.write("EXP:WAV:INCX OFF")
		instr.write("ACQuire:POINts:MODE MANual")
		instr.write(f"ACQuire:POINts:VALue {record_length}")

		selected_channels = [ch.strip().upper() for ch in channels]
		for ch in selected_channels:
			ch_num = _channel_number(ch)
			instr.write(f"CHANnel{ch_num}:STATe ON")

		# Switch to averaging mode, let the scope average for a fixed time, then export.
		instr.write("ACQuire:TYPE AVER")
		instr.write("ACQuire:COUNt 128")
		instr.write("RUN")
		pytime.sleep(max(0.0, float(average_wait_seconds)))
		instr.write("STOP")

		traces: dict[str, np.ndarray] = {}
		for ch in selected_channels:
			print(f"Querying ch {ch}")
			traces[ch] = _query_channel_trace_float32(instr, ch, transfer_timeout_ms)

		sample_rate_hz = _safe_query_float(instr, "ACQuire:SRATe?")
		x_start = _safe_query_float(instr, "TIMebase:HORizontal:POSition?")

		min_len = min(len(v) for v in traces.values()) if traces else 0
		if min_len == 0:
			raise RuntimeError("No waveform points received from instrument.")

		for ch in list(traces.keys()):
			traces[ch] = traces[ch][:min_len]

		if sample_rate_hz and sample_rate_hz > 0:
			dt = 1.0 / sample_rate_hz
			t0 = x_start if x_start is not None else 0.0
			time = t0 + np.arange(min_len, dtype=np.float64) * dt
		else:
			time = np.arange(min_len, dtype=np.float64)

		return {
			"idn": idn,
			"resource": resource,
			"channels": selected_channels,
			"record_length_requested": int(record_length),
			"sample_rate_hz": sample_rate_hz,
			"time": time,
			"traces": traces,
		}
	finally:
		try:
			instr.write("ACQuire:TYPE SAMP")
			instr.write("RUN")
		except Exception:
			pass
		instr.close()


def save_waveforms(
	data: dict[str, object],
	output_dir: str | Path,
	base_filename: str,
	write_csv: bool = True,
	write_npz: bool = True,
) -> dict[str, Path]:
	date_folder = datetime.now().strftime("%Y-%m-%d")
	out_dir = Path(output_dir) / date_folder
	out_dir.mkdir(parents=True, exist_ok=True)

	safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_filename).strip("_")
	if not safe_name:
		safe_name = datetime.now().strftime("scope_capture_%Y%m%d_%H%M%S")

	# Auto-increment filename if any output file already exists: abc -> abc_1 -> abc_2 ...
	extensions = ([".csv"] if write_csv else []) + ([".npz"] if write_npz else [])
	if any((out_dir / f"{safe_name}{ext}").exists() for ext in extensions):
		idx = 1
		while any((out_dir / f"{safe_name}_{idx}{ext}").exists() for ext in extensions):
			idx += 1
		safe_name = f"{safe_name}_{idx}"

	time = np.asarray(data["time"], dtype=np.float64)
	traces = data["traces"]
	channels = data["channels"]

	saved: dict[str, Path] = {}

	if write_csv:
		csv_path = out_dir / f"{safe_name}.csv"
		header_cols = ["TIME"] + list(channels)
		cols = [time] + [np.asarray(traces[ch], dtype=np.float64) for ch in channels]
		matrix = np.column_stack(cols)
		np.savetxt(
			csv_path,
			matrix,
			delimiter=",",
			header=",".join(header_cols),
			comments="",
			fmt="%.9g",
		)
		saved["csv"] = csv_path

	if write_npz:
		npz_path = out_dir / f"{safe_name}.npz"
		np.savez_compressed(
			npz_path,
			time=time,
			channels=np.asarray(channels),
			idn=str(data.get("idn", "")),
			sample_rate_hz=(
				np.nan if data.get("sample_rate_hz") is None else float(data["sample_rate_hz"])
			),
			**{str(ch): np.asarray(traces[ch], dtype=np.float32) for ch in channels},
		)
		saved["npz"] = npz_path

	return saved


import tkinter as tk
from tkinter import ttk, scrolledtext


def _run_gui() -> None:
	root = tk.Tk()
	root.title("MXO44 Waveform Acquirer")
	root.resizable(False, False)

	pad = {"padx": 8, "pady": 4}

	# ── Oscilloscope IP ──────────────────────────────────────────────────────
	frm_ip = ttk.LabelFrame(root, text="Oscilloscope")
	frm_ip.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

	ttk.Label(frm_ip, text="IP address:").grid(row=0, column=0, sticky="w", **pad)
	var_ip = tk.StringVar(value=SCOPE_IP)
	ttk.Entry(frm_ip, textvariable=var_ip, width=22).grid(row=0, column=1, sticky="w", **pad)

	# ── Channel labels ───────────────────────────────────────────────────────
	frm_ch = ttk.LabelFrame(root, text="Channel labels (used as column headers in the saved file)")
	frm_ch.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

	ch_defaults = ["I_L", "V_", "V_", "pwm"]
	ch_vars: list[tk.StringVar] = []
	for i, default in enumerate(ch_defaults):
		ttk.Label(frm_ch, text=f"Ch{i + 1}:").grid(row=i, column=0, sticky="w", **pad)
		var = tk.StringVar(value=default)
		entry = ttk.Entry(frm_ch, textvariable=var, width=18)
		entry.grid(row=i, column=1, sticky="w", **pad)

		if i in (1, 2):
			suffix_var = tk.StringVar(value="")
			number_var = tk.StringVar(value="1")

			suffix_frame = ttk.Frame(frm_ch)
			suffix_frame.grid(row=i, column=2, columnspan=4, sticky="w", padx=2, pady=4)

			for col_offset, option in enumerate(VOLTAGE_SUFFIX_OPTIONS):
				ttk.Radiobutton(
					suffix_frame,
					text=option,
					value=option,
					variable=suffix_var,
					command=lambda v=var, s=suffix_var, n=number_var: _sync_voltage_label(v, s, n),
				).grid(row=0, column=col_offset, sticky="w", padx=2)

			ttk.Label(suffix_frame, text="No.").grid(row=0, column=4, sticky="w", padx=(8, 2))
			spin = tk.Spinbox(
				suffix_frame,
				from_=1,
				to=8,
				width=3,
				textvariable=number_var,
				command=lambda v=var, s=suffix_var, n=number_var: _sync_voltage_label(v, s, n),
			)
			spin.grid(row=0, column=5, sticky="w")

			def _on_spin_mousewheel(event: tk.Event, v=var, s=suffix_var, n=number_var) -> str:
				try:
					current = int(n.get())
				except Exception:
					current = 1

				if event.delta > 0:
					current = min(8, current + 1)
				else:
					current = max(1, current - 1)

				n.set(str(current))
				_sync_voltage_label(v, s, n)
				return "break"

			spin.bind("<MouseWheel>", _on_spin_mousewheel)

			def _on_suffix_edit(*_args: object, v=var, s=suffix_var, n=number_var) -> None:
				_sync_voltage_label(v, s, n)

			suffix_var.trace_add("write", _on_suffix_edit)
			number_var.trace_add("write", _on_suffix_edit)

			# Keep the entry editable; the selector updates the prefix and number.
			entry.config(width=14)
		ch_vars.append(var)

	# ── Output filename ──────────────────────────────────────────────────────
	frm_out = ttk.LabelFrame(root, text="Output")
	frm_out.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

	ttk.Label(frm_out, text="Base filename:").grid(row=0, column=0, sticky="w", **pad)
	var_base_fname = tk.StringVar(value=BASE_FILENAME)
	ttk.Entry(frm_out, textvariable=var_base_fname, width=28).grid(row=0, column=1, sticky="w", **pad)
	ttk.Label(frm_out, text=f"Save folder: {OUTPUT_DIR}", foreground="grey").grid(
		row=1, column=0, columnspan=2, sticky="w", padx=8
	)

	# ── Operating point used in generated filename ───────────────────────────
	frm_op = ttk.LabelFrame(root, text="Operating point")
	frm_op.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

	ttk.Label(frm_op, text="Vin:").grid(row=0, column=0, sticky="w", **pad)
	var_vin = tk.StringVar(value="20")
	ttk.Entry(frm_op, textvariable=var_vin, width=12).grid(row=0, column=1, sticky="w", **pad)

	ttk.Label(frm_op, text="Iout:").grid(row=0, column=2, sticky="w", **pad)
	var_iout = tk.StringVar(value="5")
	ttk.Entry(frm_op, textvariable=var_iout, width=12).grid(row=0, column=3, sticky="w", **pad)

	ttk.Label(frm_op, text="fsw (kHz):").grid(row=1, column=0, sticky="w", **pad)
	var_fsw = tk.StringVar(value="100")
	ttk.Entry(frm_op, textvariable=var_fsw, width=12).grid(row=1, column=1, sticky="w", **pad)

	# ── Status log ───────────────────────────────────────────────────────────
	log = scrolledtext.ScrolledText(root, height=7, width=52, state="disabled", font=("Consolas", 9))
	log.grid(row=4, column=0, columnspan=2, **pad)

	def _log(msg: str) -> None:
		log.configure(state="normal")
		log.insert(tk.END, msg + "\n")
		log.see(tk.END)
		log.configure(state="disabled")
		root.update_idletasks()

	# ── Acquire & Save button ────────────────────────────────────────────────
	btn = ttk.Button(root, text="Acquire & Save")
	btn.grid(row=5, column=0, columnspan=2, pady=(4, 10))

	def _on_acquire() -> None:
		btn.configure(state="disabled")
		try:
			ip = var_ip.get().strip()
			if not ip:
				_log("ERROR: IP address is empty.")
				return

			resource = _resource_from_ip(ip)

			# Build channel label list; acquisition still uses C1..C4 internally.
			ch_labels = [v.get().strip() or f"C{i + 1}" for i, v in enumerate(ch_vars)]
			selected_channels = [f"C{i + 1}" for i in range(len(ch_vars))]

			vin_text = var_vin.get().strip()
			iout_text = var_iout.get().strip()
			fsw_text = var_fsw.get().strip()
			if not vin_text or not iout_text or not fsw_text:
				_log("ERROR: Vin, Iout, and fsw are required.")
				return

			try:
				fsw_khz_int = int(float(fsw_text))  # Accepts float input but converts to int for filename.
			except Exception:
				_log("ERROR: fsw must be numeric (in kHz).")
				return

			base_text = var_base_fname.get().strip() or BASE_FILENAME
			generated_filename = f"{base_text}Vin{vin_text}_Io{iout_text}_f{fsw_khz_int}"

			_log(f"Connecting to {ip} ...")
			data = acquire_scope_waveforms(
				resource=resource,
				channels=selected_channels,
				record_length=RECORD_LENGTH,
				timeout_ms=TIMEOUT_MS,
				transfer_timeout_ms=TRANSFER_TIMEOUT_MS,
				average_wait_seconds=AVERAGE_WAIT_SECONDS,
			)
			_log(f"Connected: {data['idn']}")

			# Remap internal channel keys (C1..C4) to user labels before saving.
			relabelled: dict = {
				**data,
				"channels": ch_labels,
				"traces": {ch_labels[i]: data["traces"][ch] for i, ch in enumerate(selected_channels)},
			}

			saved = save_waveforms(
				data=relabelled,
				output_dir=OUTPUT_DIR,
				base_filename=generated_filename,
				write_csv=WRITE_CSV,
				write_npz=WRITE_NPZ,
			)
			_log(f"Filename prefix: {generated_filename}")
			for kind, path in saved.items():
				_log(f"Saved {kind.upper()}: {path.name}")

		except Exception as exc:
			_log(f"ERROR: {exc}")
		finally:
			btn.configure(state="normal")

	btn.configure(command=_on_acquire)
	root.mainloop()


_run_gui()



