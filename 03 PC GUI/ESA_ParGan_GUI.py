import tkinter as tk
from tkinter import ttk


class STM32GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STM32 Communication GUI")
        self.root.geometry("800x600")
        
        # Variable to track which section is selected (DPT or BO)
        self.selected_part = tk.IntVar(value=2)
        
        # Variable to track operation mode in DPT_section (periodic or single shot)
        self.DPT_section_mode = tk.StringVar(value="periodic")

        # Variable for BO_section Buck enable state and status text
        self.BO_section_enable_var = tk.BooleanVar(value=False)
        self.BO_section_status_text = tk.StringVar(value="STOPPED")
        
        # Create main layout
        self.create_TM_section()
        self.create_DPT_BO_sections()
        
        # Bind the radio button change to update greyed out state
        self.selected_part.trace("w", self._on_part_selection_change)
        self.DPT_section_mode.trace("w", self._on_DPT_section_mode_change)
    
    # ==================== TM_section: Display Fields ====================
    def create_TM_section(self):
        """Create TM_section with display fields (non-editable) in two rows"""
        TM_section_frame = ttk.LabelFrame(self.root, text="STM32 Reported Telemetries:", padding=10)
        TM_section_frame.pack(fill=tk.X, padx=10, pady=10)
        
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
        row1_frame = ttk.Frame(TM_section_frame)
        row1_frame.pack(fill=tk.X, pady=(0, 10))
        
        for label_text, default_value in row1_fields:
            self._create_display_field(row1_frame, label_text, default_value)
        
        # Create second row
        row2_frame = ttk.Frame(TM_section_frame)
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
        DPT_section_fields = ["Turn on time 1 (us)", "Turn off time (us)", "Turn on time 2 (us)", "Cooldown time (us)"]
        for field_name in DPT_section_fields:
            self._create_input_field(self.DPT_section_frame, field_name, "DPT_section")
        
        # DPT_section: Separator
        ttk.Separator(self.DPT_section_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # DPT_section: Operation mode selection
        mode_label = ttk.Label(self.DPT_section_frame, text="Operation Mode:", font=("Arial", 10, "bold"))
        mode_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Periodic radio button
        ttk.Radiobutton(
            self.DPT_section_frame,
            text="Periodic",
            variable=self.DPT_section_mode,
            value="periodic"
        ).pack(anchor=tk.W)
        
        # Periodic options container
        self.DPT_section_periodic_frame = ttk.Frame(self.DPT_section_frame)
        self.DPT_section_periodic_frame.pack(fill=tk.X, padx=(20, 0), pady=5)
        
        # Period input field
        periodic_input_container = ttk.Frame(self.DPT_section_periodic_frame)
        periodic_input_container.pack(fill=tk.X, pady=5)
        
        ttk.Label(periodic_input_container, text="Period (ms):").pack(side=tk.LEFT, padx=(0, 10))
        self.DPT_section_period_var = tk.StringVar()
        ttk.Entry(periodic_input_container, textvariable=self.DPT_section_period_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Periodic checkbox
        self.DPT_section_periodic_checkbox_var = tk.BooleanVar()
        ttk.Checkbutton(
            self.DPT_section_periodic_frame,
            text="Enable periodic operation",
            variable=self.DPT_section_periodic_checkbox_var
        ).pack(anchor=tk.W)
        
        # Single shot radio button
        ttk.Radiobutton(
            self.DPT_section_frame,
            text="Single Shot",
            variable=self.DPT_section_mode,
            value="single_shot"
        ).pack(anchor=tk.W, pady=(10, 0))
        
        # Single shot options container
        self.DPT_section_singleshot_frame = ttk.Frame(self.DPT_section_frame)
        self.DPT_section_singleshot_frame.pack(fill=tk.X, padx=(20, 0), pady=5)
        
        # Single shot button
        ttk.Button(
            self.DPT_section_singleshot_frame,
            text="Run Single Shot",
            command=self.DPT_singleshot_callback
        ).pack(fill=tk.X)
        
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
        BO_section_fields = ["fsw (kHz)", "duty (%.0)", "T dt on (ticks)", "T dt off (ticks)"]
        for field_name in BO_section_fields:
            self._create_input_field(self.BO_section_frame, field_name, "BO_section")

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
    
    def _create_input_field(self, parent, field_name, part_id):
        """Helper to create an input field with label"""
        field_container = ttk.Frame(parent)
        field_container.pack(fill=tk.X, pady=5)
        
        # Label
        label = ttk.Label(field_container, text=field_name)
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Input field
        input_var = tk.StringVar()
        input_entry = ttk.Entry(field_container, textvariable=input_var)
        input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
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
        for input_var, input_entry in inputs_dict.values():
            input_entry.config(state=tk.NORMAL)
        if frame == self.BO_section_frame:
            self.BO_section_enable_checkbox.config(state=tk.NORMAL)
    
    def _disable_part(self, frame, inputs_dict):
        """Disable and grey out a section and its input fields"""
        for input_var, input_entry in inputs_dict.values():
            input_entry.config(state=tk.DISABLED)
        if frame == self.BO_section_frame:
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
    
    # ==================== Communication Callbacks ====================
    # These are empty callback functions for STM32 communication
    
    def DPT_mode_change_callback(self):
        """Callback when Part 2 is selected - implement STM32 communication here"""
        pass
    
    def BO_mode_change_callback(self):
        """Callback when Part 3 is selected - implement STM32 communication here"""
        pass

    def on_BO_section_enable_changed(self):
        """Callback when BO_section enable checkbox changes - implement STM32 communication here"""
        pass

    def DPT_singleshot_callback(self):
        """Callback when 'Run Single Shot' button is pressed - implement STM32 communication here"""
        pass

    def on_data_received(self, telemetry):
        """Update display fields with data received from STM32.

        Expected keys: 'v in (v)', 'i in (a)', 'v out (v)', 'i out (a)',
        'p in (w)', 'p out (w)', 'eff (%)', 'mode'.
        """
        for key, value in telemetry.items():
            normalized_key = key.lower()
            if normalized_key in self.display_fields:
                self.display_fields[normalized_key].set(str(value))
    
    def send_DPT_settings(self):
        """Send Part 2 data to STM32 - implement STM32 communication here"""
        pass
    
    def send_BO_settings(self):
        """Send Part 3 data to STM32 - implement STM32 communication here"""
        pass


def main():
    root = tk.Tk()
    gui = STM32GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
