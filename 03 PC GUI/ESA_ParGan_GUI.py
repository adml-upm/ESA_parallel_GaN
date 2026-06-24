import math
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from serial_comm import SerialManager


class STM32GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STM32 Communication GUI")
        self.root.geometry("800x600")

        # Serial communication state
        self.serial_manager = SerialManager(
            on_telemetry_callback=self.on_data_received,
            on_status_callback=self._on_serial_status_changed,
            on_raw_line_callback=self._on_serial_raw_line
        )
        self.com_port_var = tk.StringVar(value="")
        self.connection_button_text = tk.StringVar(value="Connect")
        self.connection_status_var = tk.StringVar(value="Disconnected")
        self.monitor_visible = False
        self.monitor_button_text = tk.StringVar(value="Show RX Monitor")
        
        # Variable to track which section is selected (DPT or BO)
        self.selected_part = tk.IntVar(value=2)
        
        # Variable to track operation mode in DPT_section (periodic or single shot)
        self.DPT_section_mode = tk.StringVar(value="periodic")

        # Variable for BO_section Buck enable state and status text
        self.BO_section_enable_var = tk.BooleanVar(value=False)
        self.BO_section_status_text = tk.StringVar(value="STOPPED")
        
        # Create main layout
        self.create_serial_section()
        self.create_serial_monitor_section()
        self.create_TM_section()
        self.create_DPT_BO_sections()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Bind the radio button change to update greyed out state
        self.selected_part.trace("w", self._on_part_selection_change)
        self.DPT_section_mode.trace("w", self._on_DPT_section_mode_change)

    # ==================== Serial Section: COM Port Controls ====================
    def create_serial_section(self):
        """Create COM port controls: scan, select, connect/disconnect."""
        serial_frame = ttk.LabelFrame(self.root, text="Serial Connection", padding=10)
        serial_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        row = ttk.Frame(serial_frame)
        row.pack(fill=tk.X)

        ttk.Label(row, text="COM Port:").pack(side=tk.LEFT, padx=(0, 8))

        self.combobox_ports = ttk.Combobox(
            row,
            textvariable=self.com_port_var,
            state="readonly",
            width=16
        )
        self.combobox_ports.pack(side=tk.LEFT, padx=(0, 8))

        self.button_scan_ports = ttk.Button(
            row,
            text="Scan",
            command=self.scan_com_ports
        )
        self.button_scan_ports.pack(side=tk.LEFT, padx=(0, 8))

        self.button_connect = ttk.Button(
            row,
            textvariable=self.connection_button_text,
            command=self.toggle_com_connection
        )
        self.button_connect.pack(side=tk.LEFT, padx=(0, 12))

        self.button_toggle_monitor = ttk.Button(
            row,
            textvariable=self.monitor_button_text,
            command=self.toggle_serial_monitor
        )
        self.button_toggle_monitor.pack(side=tk.LEFT, padx=(0, 12))

        self.label_connection_status = ttk.Label(
            row,
            textvariable=self.connection_status_var,
            foreground="gray30"
        )
        self.label_connection_status.pack(side=tk.LEFT)

        self.scan_com_ports()

    def create_serial_monitor_section(self):
        """Create a raw serial monitor to inspect incoming lines."""
        self.serial_monitor_frame = ttk.LabelFrame(self.root, text="Raw Serial Monitor (RX)", padding=10)
        if self.monitor_visible:
            self.serial_monitor_frame.pack(fill=tk.BOTH, padx=10, pady=(8, 0))

        button_row = ttk.Frame(self.serial_monitor_frame)
        button_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(button_row, text="Clear", command=self._clear_serial_log).pack(side=tk.LEFT)

        self.serial_log_text = ScrolledText(self.serial_monitor_frame, height=8, wrap=tk.NONE)
        self.serial_log_text.pack(fill=tk.BOTH, expand=True)
        self.serial_log_text.configure(state=tk.DISABLED)

    def toggle_serial_monitor(self):
        """Hide or show the raw serial monitor frame."""
        if self.monitor_visible:
            self.serial_monitor_frame.pack_forget()
            self.monitor_visible = False
            self.monitor_button_text.set("Show RX Monitor")
        else:
            self.serial_monitor_frame.pack(fill=tk.BOTH, padx=10, pady=(8, 0), before=self.TM_section_frame)
            self.monitor_visible = True
            self.monitor_button_text.set("Hide RX Monitor")

    def scan_com_ports(self):
        """Scan available serial ports and update COM selector."""
        ports = SerialManager.list_ports()
        self.combobox_ports["values"] = ports

        if ports:
            if self.com_port_var.get() not in ports:
                self.com_port_var.set(ports[0])
            self.connection_status_var.set("Ports found")
        else:
            self.com_port_var.set("")
            self.connection_status_var.set("No COM ports found")

    def toggle_com_connection(self):
        """Connect or disconnect serial communication."""
        if self.serial_manager.is_connected:
            self.serial_manager.disconnect()
            self._set_connection_ui(False)
            return

        selected_port = self.com_port_var.get().strip()
        if not selected_port:
            messagebox.showwarning("No COM Port", "Please scan and select a COM port first.")
            return

        connected, message = self.serial_manager.connect(selected_port)
        if connected:
            self._set_connection_ui(True)
            self.connection_status_var.set(message)
        else:
            self._set_connection_ui(False)
            self.connection_status_var.set(message)
            messagebox.showerror("Connection Failed", message)

    def _set_connection_ui(self, connected):
        """Update controls for connected/disconnected serial state."""
        if connected:
            self.connection_button_text.set("Disconnect")
            self.button_scan_ports.config(state=tk.DISABLED)
            self.combobox_ports.config(state="disabled")
        else:
            self.connection_button_text.set("Connect")
            self.button_scan_ports.config(state=tk.NORMAL)
            self.combobox_ports.config(state="readonly")

    def _on_serial_status_changed(self, status_text):
        """Thread-safe status update callback from serial layer."""
        self.root.after(0, self.connection_status_var.set, status_text)

    def _on_serial_raw_line(self, line_text):
        """Thread-safe callback to show each raw serial line."""
        self.root.after(0, self._append_serial_log, line_text)

    def _append_serial_log(self, line_text):
        """Append one line to the serial monitor and autoscroll."""
        self.serial_log_text.configure(state=tk.NORMAL)
        self.serial_log_text.insert(tk.END, line_text + "\n")
        self.serial_log_text.see(tk.END)
        self.serial_log_text.configure(state=tk.DISABLED)

    def _clear_serial_log(self):
        """Clear raw serial monitor text area."""
        self.serial_log_text.configure(state=tk.NORMAL)
        self.serial_log_text.delete("1.0", tk.END)
        self.serial_log_text.configure(state=tk.DISABLED)

    def _on_close(self):
        """Gracefully close serial connection before exiting."""
        self.serial_manager.disconnect()
        self.root.destroy()
    
    # ==================== TM_section: Display Fields ====================
    def create_TM_section(self):
        """Create TM_section with display fields (non-editable) in two rows"""
        self.TM_section_frame = ttk.LabelFrame(self.root, text="STM32 Reported Telemetries:", padding=10)
        self.TM_section_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Store display field references for later updates
        self.display_fields = {}
        
        # Row 1 fields
        row1_fields = [
            ("V in (V)", "Waiting..."),
            ("I in (A)", "Waiting..."),
            ("V out (V)", "Waiting..."),
            ("I out (A)", "Waiting..."),
        ]
        
        # Row 2 fields
        row2_fields = [
            ("P in (W)", "Waiting..."),
            ("P out (W)", "Waiting..."),
            ("Eff (%)", "Waiting..."),
            ("Mode", "Waiting..."),
        ]
        
        # Create first row
        row1_frame = ttk.Frame(self.TM_section_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 10))
        
        for label_text, default_value in row1_fields:
            self._create_display_field(row1_frame, label_text, default_value)
        
        # Create second row
        row2_frame = ttk.Frame(self.TM_section_frame)
        row2_frame.pack(fill=tk.X)
        
        for label_text, default_value in row2_fields:
            self._create_display_field(row2_frame, label_text, default_value)
    
    def _create_display_field(self, parent, label_text, default_value):
        """Helper to create a display field with label"""
        field_container = ttk.Frame(parent)
        field_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Label
        label = ttk.Label(field_container, text=label_text, font=("Arial", 10, "bold"))
        label.pack(anchor=tk.W)
        
        # Display field (read-only)
        display_var = tk.StringVar(value=default_value)
        self.display_fields[label_text.lower()] = display_var
        
        display_label = ttk.Label(
            field_container,
            textvariable=display_var,
            background="#f0f0f0",
            relief=tk.SUNKEN,
            borderwidth=2,
            font=("Arial", 11)
        )
        display_label.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    
    # ==================== DPT_section & BO_section: Input Sections ====================
    def create_DPT_BO_sections(self):
        """Create DPT_section and BO_section split horizontally"""
        # Main container for DPT_section and BO_section
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # DPT_section
        self.DPT_section_frame = ttk.LabelFrame(container, text="DPT Operation", padding=10)
        self.DPT_section_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # DPT_section: Radio button
        self.DPT_radio = ttk.Radiobutton(
            self.DPT_section_frame,
            text="Configure DPT Operation",
            variable=self.selected_part,
            value=2,
            command=self._on_DPT_section_selected
        )
        self.DPT_radio.pack(anchor=tk.W, pady=(0, 10))
        
        # DPT_section: Input fields
        self.DPT_section_inputs = {}
        self.DPT_section_action_buttons = []
        DPT_section_fields = ["Turn on time 1 (us)", "Turn off time (us)", "Turn on time 2 (us)", "Cooldown time (us)"]
        for field_name in DPT_section_fields:
            self._create_input_field(self.DPT_section_frame, field_name, "DPT_section")
        
        # DPT_section: Separator
        ttk.Separator(self.DPT_section_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # DPT_section: Operation mode selection
        mode_label = ttk.Label(self.DPT_section_frame, text="Operation Mode:", font=("Arial", 10, "bold"))
        mode_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Periodic radio button
        self.DPT_section_periodic_radio = ttk.Radiobutton(
            self.DPT_section_frame,
            text="Periodic",
            variable=self.DPT_section_mode,
            value="periodic"
        )
        self.DPT_section_periodic_radio.pack(anchor=tk.W)
        
        # Periodic options container
        self.DPT_section_periodic_frame = ttk.Frame(self.DPT_section_frame)
        self.DPT_section_periodic_frame.pack(fill=tk.X, padx=(20, 0), pady=5)
        
        # Period input field
        periodic_input_container = ttk.Frame(self.DPT_section_periodic_frame)
        periodic_input_container.pack(fill=tk.X, pady=5)
        
        ttk.Label(periodic_input_container, text="Period (ms):").pack(side=tk.LEFT, padx=(0, 10))
        self.DPT_section_period_var = tk.StringVar()
        self.DPT_section_period_entry = ttk.Entry(periodic_input_container, textvariable=self.DPT_section_period_var)
        self.DPT_section_period_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Periodic checkbox
        self.DPT_section_periodic_checkbox_var = tk.BooleanVar()
        self.DPT_section_periodic_checkbox = ttk.Checkbutton(
            self.DPT_section_periodic_frame,
            text="Enable periodic operation",
            variable=self.DPT_section_periodic_checkbox_var
        )
        self.DPT_section_periodic_checkbox.pack(anchor=tk.W)
        
        # Single shot radio button
        self.DPT_section_singleshot_radio = ttk.Radiobutton(
            self.DPT_section_frame,
            text="Single Shot",
            variable=self.DPT_section_mode,
            value="single_shot"
        )
        self.DPT_section_singleshot_radio.pack(anchor=tk.W, pady=(10, 0))
        
        # Single shot options container
        self.DPT_section_singleshot_frame = ttk.Frame(self.DPT_section_frame)
        self.DPT_section_singleshot_frame.pack(fill=tk.X, padx=(20, 0), pady=5)
        
        # Single shot button
        self.DPT_section_singleshot_button = ttk.Button(
            self.DPT_section_singleshot_frame,
            text="Run Single Shot",
            command=self.DPT_singleshot_callback
        )
        self.DPT_section_singleshot_button.pack(fill=tk.X)
        
        # Initial visibility
        self._update_DPT_section_mode_visibility()
        
        # BO_section
        self.BO_section_frame = ttk.LabelFrame(container, text="Buck operation", padding=10)
        self.BO_section_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # BO_section: Radio button
        self.BO_radio = ttk.Radiobutton(
            self.BO_section_frame,
            text="Use Buck Operation",
            variable=self.selected_part,
            value=3,
            command=self._on_BO_section_selected
        )
        self.BO_radio.pack(anchor=tk.W, pady=(0, 10))
        
        # BO_section: Input fields
        self.BO_section_inputs = {}
        self.BO_section_action_buttons = []
        self._create_input_field(
            self.BO_section_frame,
            "fsw (kHz)",
            "BO_section",
            default_value="100.0",
            inline_button_text="Send Freq",
            inline_button_command=self.send_pwm_frequency_callback
        )
        self._create_input_field(
            self.BO_section_frame,
            "duty (%.0)",
            "BO_section",
            default_value="50.0",
            inline_button_text="Send Duty",
            inline_button_command=self.send_pwm_duty_callback
        )
        self._create_input_field(
            self.BO_section_frame,
            "T dt on (ns)",
            "BO_section",
            default_value="53",
            inline_button_text="Send DT",
            inline_button_command=self.send_pwm_deadtime_callback
        )
        self._create_input_field(self.BO_section_frame, "T dt off (ns)", "BO_section")

        # Show quantized timer representation for both deadtime fields.
        self.BO_section_dt_on_info_var = tk.StringVar(value="")
        self.BO_section_dt_off_info_var = tk.StringVar(value="")
        ttk.Label(
            self.BO_section_frame,
            textvariable=self.BO_section_dt_on_info_var,
            foreground="gray35"
        ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 4))
        ttk.Label(
            self.BO_section_frame,
            textvariable=self.BO_section_dt_off_info_var,
            foreground="gray35"
        ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 2))

        # STM32 currently supports one deadtime value only.
        # Mirror upper deadtime to lower field and keep lower field read-only.
        self._bind_BO_section_deadtime_fields()

        ttk.Separator(self.BO_section_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # BO_section: Enable toggle and visual status indicator
        BO_section_status_row = ttk.Frame(self.BO_section_frame)
        BO_section_status_row.pack(fill=tk.X, pady=(0, 5))

        self.BO_section_enable_checkbox = ttk.Checkbutton(
            BO_section_status_row,
            text="Enable",
            variable=self.BO_section_enable_var,
            command=self._on_BO_section_enable_toggled
        )
        self.BO_section_enable_checkbox.pack(side=tk.LEFT)

        self.BO_section_status_label = tk.Label(
            BO_section_status_row,
            textvariable=self.BO_section_status_text,
            font=("Arial", 10, "bold"),
            fg="gray40"
        )
        self.BO_section_status_label.pack(side=tk.LEFT, padx=(12, 0))

        self._update_BO_section_status_indicator()
        
        # Initial state: DPT_section is enabled, BO_section is disabled
        self._update_part_states()
    
    def _create_input_field(
        self,
        parent,
        field_name,
        part_id,
        default_value="",
        inline_button_text=None,
        inline_button_command=None
    ):
        """Helper to create an input field with label"""
        field_container = ttk.Frame(parent)
        field_container.pack(fill=tk.X, pady=5)
        
        # Label
        label = ttk.Label(field_container, text=field_name)
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Input field
        input_var = tk.StringVar(value=default_value)
        input_entry = ttk.Entry(field_container, textvariable=input_var)
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        if inline_button_text and inline_button_command:
            inline_button = ttk.Button(
                field_container,
                text=inline_button_text,
                command=inline_button_command
            )
            inline_button.pack(side=tk.LEFT, padx=(8, 0))
            if part_id == "DPT_section":
                self.DPT_section_action_buttons.append(inline_button)
            else:
                self.BO_section_action_buttons.append(inline_button)
        
        # Store reference
        if part_id == "DPT_section":
            self.DPT_section_inputs[field_name.lower()] = (input_var, input_entry)
        else:
            self.BO_section_inputs[field_name.lower()] = (input_var, input_entry)
    
    # ==================== Radio Button Callbacks ====================
    def _on_DPT_section_selected(self):
        """Callback when DPT_section radio button is selected"""
        self._on_part_selection_change()
        self.DPT_mode_change_callback()  # User callback
    
    def _on_BO_section_selected(self):
        """Callback when BO_section radio button is selected"""
        self._on_part_selection_change()
        self.BO_mode_change_callback()  # User callback
    
    def _on_part_selection_change(self, *args):
        """Update UI state when section selection changes"""
        self._update_part_states()
    
    def _update_part_states(self):
        """Enable/disable and grey out sections based on selection"""
        if self.selected_part.get() == 2:
            self._enable_part(self.DPT_section_frame, self.DPT_section_inputs)
            self._disable_part(self.BO_section_frame, self.BO_section_inputs)
        else:
            self._disable_part(self.DPT_section_frame, self.DPT_section_inputs)
            self._enable_part(self.BO_section_frame, self.BO_section_inputs)
    
    def _enable_part(self, frame, inputs_dict):
        """Enable a section and its input fields"""
        for field_name, (input_var, input_entry) in inputs_dict.items():
            if frame == self.BO_section_frame and field_name == "t dt off (ns)":
                input_entry.config(state=tk.DISABLED)
            else:
                input_entry.config(state=tk.NORMAL)
        if frame == self.DPT_section_frame:
            self.DPT_section_periodic_radio.config(state=tk.NORMAL)
            self.DPT_section_singleshot_radio.config(state=tk.NORMAL)
            self.DPT_section_period_entry.config(state=tk.NORMAL)
            self.DPT_section_periodic_checkbox.config(state=tk.NORMAL)
            self.DPT_section_singleshot_button.config(state=tk.NORMAL)
            for button in self.DPT_section_action_buttons:
                button.config(state=tk.NORMAL)
        if frame == self.BO_section_frame:
            for button in self.BO_section_action_buttons:
                button.config(state=tk.NORMAL)
            self.BO_section_enable_checkbox.config(state=tk.NORMAL)
    
    def _disable_part(self, frame, inputs_dict):
        """Disable and grey out a section and its input fields"""
        for input_var, input_entry in inputs_dict.values():
            input_entry.config(state=tk.DISABLED)
        if frame == self.DPT_section_frame:
            self.DPT_section_periodic_radio.config(state=tk.DISABLED)
            self.DPT_section_singleshot_radio.config(state=tk.DISABLED)
            self.DPT_section_period_entry.config(state=tk.DISABLED)
            self.DPT_section_periodic_checkbox.config(state=tk.DISABLED)
            self.DPT_section_singleshot_button.config(state=tk.DISABLED)
            for button in self.DPT_section_action_buttons:
                button.config(state=tk.DISABLED)
        if frame == self.BO_section_frame:
            for button in self.BO_section_action_buttons:
                button.config(state=tk.DISABLED)
            self.BO_section_enable_checkbox.config(state=tk.DISABLED)

    def _on_BO_section_enable_toggled(self):
        """Update BO_section running/stopped indicator from enable checkbox"""
        self._update_BO_section_status_indicator()
        self.on_BO_section_enable_changed()

    def _update_BO_section_status_indicator(self):
        """Refresh BO_section status text and color based on enable state"""
        if self.BO_section_enable_var.get():
            self.BO_section_status_text.set("RUNNING")
            self.BO_section_status_label.config(fg="red")
        else:
            self.BO_section_status_text.set("STOPPED")
            self.BO_section_status_label.config(fg="gray40")
    
    def _on_DPT_section_mode_change(self, *args):
        """Update DPT_section UI when operation mode changes"""
        self._update_DPT_section_mode_visibility()
    
    def _update_DPT_section_mode_visibility(self):
        """Show/hide periodic and single shot options based on selection"""
        if self.DPT_section_mode.get() == "periodic":
            self.DPT_section_periodic_frame.pack(fill=tk.X, padx=(20, 0), pady=5)
            self.DPT_section_singleshot_frame.pack_forget()
        else:
            self.DPT_section_periodic_frame.pack_forget()
            self.DPT_section_singleshot_frame.pack(fill=tk.X, padx=(20, 0), pady=5)

    def _bind_BO_section_deadtime_fields(self):
        """Mirror T dt on value into T dt off and keep T dt off disabled."""
        dt_on_var, _ = self.BO_section_inputs["t dt on (ns)"]
        dt_off_var, dt_off_entry = self.BO_section_inputs["t dt off (ns)"]

        def _mirror_deadtime(*_args):
            requested_text = dt_on_var.get().strip()
            dt_off_var.set(requested_text)
            quantized_info = self._format_deadtime_quantization_info(requested_text)
            self.BO_section_dt_on_info_var.set(quantized_info)
            self.BO_section_dt_off_info_var.set(quantized_info)

        dt_on_var.trace_add("write", _mirror_deadtime)
        _mirror_deadtime()
        dt_off_entry.config(state=tk.DISABLED)

    def _format_deadtime_quantization_info(self, requested_text):
        """Format deadtime timer quantization as '<ticks>clkcycles/<time>ns'."""
        if requested_text == "":
            return ""

        try:
            requested_ns = float(requested_text)
        except ValueError:
            return "Invalid value"

        if requested_ns < 0.0:
            return "Invalid value"

        timer_clk_hz = 170e6
        ticks = int(math.ceil(requested_ns * timer_clk_hz * 1e-9))
        resulting_ns = (ticks / timer_clk_hz) * 1e9
        return f"Deadtime: {ticks} CLK_cycles / {resulting_ns:.1f}ns"
    
    # ==================== Communication Callbacks ====================
    # These are empty callback functions for STM32 communication

    def _send_serial_command(self, command_text):
        """Send one command to STM32 and report if not connected."""
        if not self.serial_manager.is_connected:
            self.connection_status_var.set("Not connected")
            return False

        success = self.serial_manager.send_line(command_text)
        if not success:
            self.connection_status_var.set("Send failed")
            return False

        self.connection_status_var.set(f"TX: {command_text}")
        return True

    def _get_inputs_snapshot(self, inputs_dict):
        """Read current values from one input dictionary."""
        snapshot = {}
        for key, (value_var, _entry) in inputs_dict.items():
            snapshot[key] = value_var.get().strip()
        return snapshot
    
    def DPT_mode_change_callback(self):
        """Callback when Part 2 is selected - implement STM32 communication here"""
        selected_mode = self.DPT_section_mode.get()
        self._send_serial_command(f"MODE:DPT:{selected_mode}")
    
    def BO_mode_change_callback(self):
        """Callback when Part 3 is selected - implement STM32 communication here"""
        self._send_serial_command("MODE:BO:active")

    def on_BO_section_enable_changed(self):
        """Callback when BO_section enable checkbox changes - implement STM32 communication here"""
        enabled = 1 if self.BO_section_enable_var.get() else 0
        self._send_serial_command(f"BO:ENABLE:{enabled}")

    def DPT_singleshot_callback(self):
        """Callback when 'Run Single Shot' button is pressed - implement STM32 communication here"""
        self._send_serial_command("DPT:SINGLE_SHOT")

    def send_pwm_duty_callback(self):
        """Send PWM duty command in percent: PWM:DUTY:<0..100>."""
        duty_text = self.BO_section_inputs["duty (%.0)"][0].get().strip()
        try:
            duty_value = float(duty_text)
        except ValueError:
            messagebox.showwarning("Invalid Duty", "Duty must be a numeric value between 0 and 100.")
            return

        if duty_value < 0.0 or duty_value > 100.0:
            messagebox.showwarning("Invalid Duty", "Duty must be between 0 and 100.")
            return

        self._send_serial_command(f"PWM:DUTY:{duty_value:g}")

    def send_pwm_frequency_callback(self):
        """Send PWM frequency command in Hz: PWM:FREQ:<Hz>."""
        freq_khz_text = self.BO_section_inputs["fsw (khz)"][0].get().strip()
        try:
            freq_khz = float(freq_khz_text)
        except ValueError:
            messagebox.showwarning("Invalid Frequency", "Switching frequency must be numeric (kHz).")
            return

        if freq_khz <= 0.0:
            messagebox.showwarning("Invalid Frequency", "Switching frequency must be greater than 0.")
            return

        freq_hz = int(round(freq_khz * 1000.0))
        self._send_serial_command(f"PWM:FREQ:{freq_hz}")

    def send_pwm_deadtime_callback(self):
        """Send PWM deadtime command in ns: PWM:DT:<ns>."""
        deadtime_text = self.BO_section_inputs["t dt on (ns)"][0].get().strip()
        try:
            deadtime_ns = int(float(deadtime_text))
        except ValueError:
            messagebox.showwarning("Invalid Deadtime", "Deadtime must be numeric (ns).")
            return

        if deadtime_ns < 0:
            messagebox.showwarning("Invalid Deadtime", "Deadtime must be >= 0 ns.")
            return

        self._send_serial_command(f"PWM:DT:{deadtime_ns}")

    def on_data_received(self, telemetry):
        """Update display fields with data received from STM32.

        Expected keys: 'v in (v)', 'i in (a)', 'v out (v)', 'i out (a)',
        'p in (w)', 'p out (w)', 'eff (%)', 'mode'.
        """
        # Serial reads run in a background thread, so push UI updates to main thread.
        self.root.after(0, self._apply_telemetry_update, telemetry)

    def _apply_telemetry_update(self, telemetry):
        """Apply telemetry values to display fields in the Tk main thread."""
        for key, value in telemetry.items():
            normalized_key = key.lower()
            if normalized_key in self.display_fields:
                self.display_fields[normalized_key].set(str(value))
    
    def send_DPT_settings(self):
        """Send Part 2 data to STM32 - implement STM32 communication here"""
        inputs = self._get_inputs_snapshot(self.DPT_section_inputs)
        period_ms = self.DPT_section_period_var.get().strip()
        periodic_enabled = 1 if self.DPT_section_periodic_checkbox_var.get() else 0

        parts = [
            "DPT:SET",
            f"ton1_us={inputs.get('turn on time 1 (us)', '')}",
            f"toff_us={inputs.get('turn off time (us)', '')}",
            f"ton2_us={inputs.get('turn on time 2 (us)', '')}",
            f"cooldown_us={inputs.get('cooldown time (us)', '')}",
            f"mode={self.DPT_section_mode.get()}",
            f"period_ms={period_ms}",
            f"periodic_enable={periodic_enabled}",
        ]
        self._send_serial_command(";".join(parts))
    
    def send_BO_settings(self):
        """Send Part 3 data to STM32 - implement STM32 communication here"""
        inputs = self._get_inputs_snapshot(self.BO_section_inputs)
        enabled = 1 if self.BO_section_enable_var.get() else 0

        parts = [
            "BO:SET",
            f"fsw_khz={inputs.get('fsw (khz)', '')}",
            f"duty_pct={inputs.get('duty (%.0)', '')}",
            f"tdt_on_ns={inputs.get('t dt on (ns)', '')}",
            f"tdt_off_ns={inputs.get('t dt off (ns)', '')}",
            f"enable={enabled}",
        ]
        self._send_serial_command(";".join(parts))


def main():
    root = tk.Tk()
    gui = STM32GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
