import numpy as np
from itertools import product
from spicelib.editor import base_editor
import re
from pathlib import Path
import shutil

class LtSimConfiguration:
    def __init__(self, base_config: dict = None, sim_output_folder: str = None):
        if base_config:
            self.base_cnfg_dict = base_config.copy()
        else:
            self.base_cnfg_dict = {}
        

        self.sim_output_folder = sim_output_folder

        self.N_devices = 4
        self.TOTAL_DRAWN_DEVICES = 10

        self.config_ranges = {}
        self.perturb_params = {}
        self.params_to_plot = []
        self.default_Kperturb = 0.05  # standard 5% perturbation magnitude

        # HEMT library params
        # project root is parent of "01 Code" -> Path(__file__).resolve().parent.parent
        p = Path(__file__).resolve().parent.parent / "02 Simulations" / "00 ComonLibs" / "EPCGaNLibrary.lib"
        self.hemt_lib_path = str(p)
        # self.out_file_path = r"C:\Users\Alex\Documents\03 Projects\04 ESA parallel GaN\ESA_parallel_GaN\02 Simulations\00 ComonLibs\EPCGaNLibrary_modified.lib"
        self.epc_model_param_mapping = {
            # 'si': 'si',
            # 'so': 'so',
            # 'sr': 'sr',
            # 'aWg': 'aWg',
            # 'Wg': 'Wg',
            'k_gm_factor': 'A1',  # ∝ transconductance gm scaling factor
            'Vgs_th': 'k2',  # Gate threshold voltage
            'Vgs_th_k_corner': 'k3',  # Turn-on softness
            'Rds_on_base': 'rpara',  # Base Parasitic resistance
            # 'rpara_s_factor': 'rpara_s_factor', # Distributes Rdson to source and drain parasitics. Don't touch
            'gm_Tc': 'aITc',  # Temperature coeff for gm, reduces Id per °C increase. (care for polarity)
            'Rds_on_Tc': 'arTc',  # Temperature coeff for Rds_on. (care for polarity)
            'Vgs_th_Tc': 'k2Tc',  # Temp coeff for Vth
            # 'Channel_modulation_0': 'x0_0',
            # 'channel_modulation_0_temp_coef': 'x0_0_TC',
            # 'Channel_modulation_1': 'x0_1',
            # 'channel_modulation_1_temp_coef': 'x0_1_TC',
            'Rgate_int': 'rg_value',
        }
        self.inverse_epc_model_param_mapping = {v: k for k, v in self.epc_model_param_mapping.items()}

    def _is_epc_param(self, param_name: str, type: str='standard') -> bool:
        base_name = re.sub(r'_[0-9]+$', '', param_name)
        if type == 'standard':
            return base_name in self.epc_model_param_mapping.keys()
        elif type == 'spice':
            return base_name in self.epc_model_param_mapping.values()
        else:
            raise ValueError("type must be 'standard' or 'spice'")

    def add_param_sweep_with_range(self, param_name: str, range_vals: list, m_points: int=5, log_space:bool =False):
        if param_name in self.base_cnfg_dict:
            if not log_space:  # Default is linear space
                self.config_ranges[param_name] = np.linspace(range_vals[0], range_vals[1], m_points)
            else:
                self.config_ranges[param_name] = np.geomspace(range_vals[0], range_vals[1], m_points)
            self.params_to_plot.append(param_name)
        else:
            raise ValueError(f"Parameter {param_name} not in base configuration dictionary.")
        
    def add_param_perturbation(self, param_name: str, 
                               perturb_abs: float | None = None, 
                               perturb_rel: float | None = None):
        if self._is_epc_param(param_name, type='standard'):
            stripped_param_name = param_name.rsplit('_', 1)
            param_name = '_'.join([self.epc_model_param_mapping[stripped_param_name[0]], stripped_param_name[-1]])
        if param_name in self.base_cnfg_dict:
            if perturb_abs is not None and perturb_rel is not None:
                raise ValueError("Use either magnitude or percentage, not both")
            elif perturb_abs is None and perturb_rel is not None:
                perturb_abs = perturb_rel * abs(self.base_cnfg_dict[param_name])
            elif perturb_abs is not None and perturb_rel is None:
                pass
            else:
                perturb_abs = self.default_Kperturb * abs(self.base_cnfg_dict[param_name])            
        else:
            raise ValueError(f"Parameter {param_name} not in base configuration dictionary.")
        
        self.perturb_params[param_name] = perturb_abs
        self.params_to_plot.append(param_name)
        
    def perturb_all_params(self):
        """Adds all base_cnfg_dict keys to params to be perturbed. Defaulte default_Kperturb is used as mag. """
        for param in self.base_cnfg_dict.keys():
            self.add_param_perturbation(param_name=param)

    def yield_cnfgs_sequentially(self, iter_type: str='sweep'):
        """Yields configurations one at a time based on the specified iteration type.
         iter_type: 'sweep' for single parameter sweeps, 'perturbation' for single parameter perturbations,
         'multiparam' for multi-parameter sweeps.
        returns: Yields a tuple of (configuration_dict, changed_params_dict) for each configuration."""
        if self.base_cnfg_dict != {}:
            yield self.base_cnfg_dict, {}  # Yield the base configuration first

            if iter_type == 'sweep' and self.config_ranges != {}:
                for config, changed_params in self.single_param_sweep_generator():
                    yield config, changed_params
            elif iter_type == 'perturbation' and self.perturb_params != {}:
                for config, changed_params in self.single_param_perturbation_generator():
                    yield config, changed_params
            elif iter_type == 'multiparam' and self.config_ranges != {}:
                for config, changed_params, in self.multiparam_sweep_generator():
                    yield config, changed_params
            else:
                raise ValueError("Invalid iter_type or config. Choose from 'sweep', 'perturbation', or 'multiparam'.")
        else:
            raise ValueError("Base configuration dictionary is empty. Please set it before yielding configurations.")
        
    def single_param_sweep_generator(self):
        """Generates configurations by sweeping one parameter at a time.
        returns: Yields a tuple of (configuration_dict, changed_params_dict) for each configuration."""
        for param_name, param_range in self.config_ranges.items():
            for param_value in param_range:
                config = self.base_cnfg_dict.copy()
                config[param_name] = param_value
                # Adjust device to modified lib entry if EPC parameter is hemt parameter
                if self._is_epc_param(param_name, type='spice'):
                    subindex = param_name.rsplit('_', 1)[-1]
                    config[f'dev_{subindex}'] += '_' + subindex
                yield config, {param_name: param_value}

    def single_param_perturbation_generator(self):
        """Generates configurations by perturbing one parameter at a time.
        returns: Yields a tuple of (configuration_dict, changed_params_dict) for each configuration."""
        for param_name, perturb_magnitude in self.perturb_params.items():
            if param_name in self.base_cnfg_dict.keys():
                config = self.base_cnfg_dict.copy()
                new_val = config[param_name] + perturb_magnitude
                config[param_name] = new_val
                # Adjust device to modified lib entry if EPC parameter is hemt parameter
                if self._is_epc_param(param_name, type='spice'):
                    subindex = param_name.rsplit('_', 1)[-1]
                    config[f'dev_{subindex}'] += '_' + subindex
                yield config, {param_name: config[param_name]}

    def multiparam_sweep_generator(self):
        """Generates configurations for all combinations of parameters in config_ranges.
        returns: Yields a tuple of (configuration_dict, changed_params_dict) for each configuration."""
        param_names = list(self.config_ranges.keys())
        param_ranges = [self.config_ranges[name] for name in param_names]
        
        for param_values in product(*param_ranges):
            config = self.base_cnfg_dict.copy()
            config.update(dict(zip(param_names, param_values)))
            yield config, dict(zip(param_names, param_values))
    
    def modify_hemt_device_count(self, N_devices: int):
        """Modifies the number of parallel HEMT devices in the base configuration."""
        self.N_devices = N_devices
        raise ValueError("Not yet implemented")
        self.base_cnfg_dict['self.N_devices'] = self.N_devices

    def get_hemt_parameters_from_lib(self, hemt_name: str|list='EPC2305'):
        """Retrieves HEMT parameters from the library file.
        hemt_name: Name of the HEMT device in the library (e.g., 'EPC2305') or list of names.
        returns: Dictionary of HEMT parameters with their values."""

        if isinstance(hemt_name, str):
            hemt_name = [hemt_name]  # make it a list for uniform processing

        with open(self.hemt_lib_path, "r", encoding="utf-8") as f:
            file_text = f.read()

        all_hemt_params = {k: None for k in hemt_name}
        for hemt in hemt_name:
            # find the .subckt block for the requested hemt_name only (e.g. ".subckt EPC2305 gatein drainin sourcein ... .ends")
            device_block_re = re.compile(
                rf'^\s*\.subckt\s+({re.escape(hemt)})\s+gatein\s+drainin\s+sourcein\b(.*?^\s*\.ends\b)',
                re.DOTALL | re.MULTILINE | re.IGNORECASE)

            match = device_block_re.search(file_text)
            device_definition_block = match.group(0)
            if match.group(1) != hemt:
                raise ValueError(f"HEMT device {hemt} not found in the library file.")

            # collect logical .param lines (handle continuation lines starting with '+')
            param_line_matches = []
            lines = device_definition_block.splitlines()
            param_line_matches = []
            current = None
            param_re = re.compile(r'^\s*\.param\b(.*)$', re.IGNORECASE)
            cont_re = re.compile(r'^\s*\+(.*)$')

            for line in lines:
                m = param_re.match(line)
                if m:
                    if current is not None:
                        param_line_matches.append(current.strip())
                    current = m.group(1).strip()
                    continue

                cm = cont_re.match(line)
                if cm and current is not None:
                    current += ' ' + cm.group(1).strip()
                    continue

                # any other line ends the current .param logical line
                if current is not None:
                    param_line_matches.append(current.strip())
                current = None

            # parse key=value pairs from each logical .param line
            epc_model_params = {}
            for line in param_line_matches:
                parts = re.split(r'\s+', line)
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        epc_model_params[k.strip()] = v.strip()

            # return only parameters whose spice-names appear in the mapping values as floats
            NUM_RE = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
            filtered_config = {k: v for k, v in epc_model_params.items() if k in set(self.epc_model_param_mapping.values())}
            for k, v in filtered_config.items():
                try:
                    num_val = float(v)
                except ValueError:
                    m = NUM_RE.search(v[1:-1].strip() if v.startswith('{') and v.endswith('}') else v)
                    num_val = float(m.group(0))
                filtered_config[k] = num_val
            
            all_hemt_params[hemt] = (filtered_config, device_definition_block)  # add this hemt's params to results dict

        return all_hemt_params

    def generate_modif_device_models(self, orig_hemt_name: str='EPC2305', model_changes: dict=None):
        """
        model_changes: dict of {new_model_name: param_changes_dict}
        Each param_changes_dict is {param_spice_name: new_value}
        """
        if model_changes is None or model_changes == {}:
            return []  # Nothing to do

        # Read the original file and extract the original text
        lib_path = Path(self.hemt_lib_path)
        with open(lib_path, 'r', encoding='utf-8') as f:
            orig_lib_text = f.read()
        
        # Find the original device definition block
        device_block_re = re.compile(
            rf'^\s*\.subckt\s+({re.escape(orig_hemt_name)})\s+gatein\s+drainin\s+sourcein\b(.*?^\s*\.ends\b)',
            re.DOTALL | re.MULTILINE | re.IGNORECASE)
        match = device_block_re.search(orig_lib_text)
        if not match:
            raise ValueError(f"HEMT device {orig_hemt_name} not found in the library file.")
        orig_device_definition_block = match.group(0)
        
        # For each new model, create a modified block and collect it
        modified_blocks = []
        for new_hemt_name, hemt_param_changes in model_changes.items():
            # Rename the device in the .subckt line to new_hemt_name
            subckt_name_re = re.compile(r'(^\s*\.subckt\s+)(\S+)', re.IGNORECASE | re.MULTILINE)
            def _rename_subckt(m):
                prefix, name = m.group(1), m.group(2)
                return prefix + new_hemt_name
            modif_dev_def_block = subckt_name_re.sub(_rename_subckt, orig_device_definition_block, count=1)
        
            # Apply parameter changes (reuse original logic)
            for param_spice_name in hemt_param_changes.keys():
                if not self._is_epc_param(param_spice_name, type='spice'):
                    continue    
                new_val_str = repr(hemt_param_changes[param_spice_name])
                pat = re.compile(r'(?<!\w)(' + re.escape(param_spice_name.rsplit('_', 1)[0]) + r')\s*=\s*(\{.*?\}|[^\s]+)', re.DOTALL | re.IGNORECASE)
                def _replace_assign(m):
                    lhs = m.group(1)
                    rhs = m.group(2)
                    if rhs.startswith('{') and rhs.endswith('}'):
                        inner = rhs[1:-1]
                        m2 = re.match(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)$', inner)
                        if m2:
                            rest = m2.group(2) or ''
                            rest = re.sub(r'\s+', '', rest)
                            return lhs + '={' + new_val_str + rest + '}'
                        return lhs + '={' + new_val_str + '}'
                    else:
                        return f"{lhs}={new_val_str}"
                modif_dev_def_block = pat.sub(_replace_assign, modif_dev_def_block)
        
            # Collect the modified block
            modified_blocks.append(modif_dev_def_block)
        
        return modified_blocks

    def generate_modif_lib(self, orig_hemt_name: str='EPC2305', model_changes: dict=None):
        """
        model_changes: dict of {new_model_name: param_changes_dict}
        Each param_changes_dict is {param_spice_name: new_value}
        """
        if model_changes is None or model_changes == {}:
            return  # Nothing to do

        # Read the original file and extract the original text
        lib_path = Path(self.hemt_lib_path)
        with open(lib_path, 'r', encoding='utf-8') as f:
            orig_lib_text = f.read()
    
        # Find the marker and cut everything after it
        marker_re = re.compile(r'(?m)^.*\*\*\*MODIFIED MODELS FOLLOW THIS COMMENT\*\*\*.*$')
        m_marker = marker_re.search(orig_lib_text)
        if not m_marker:
            raise ValueError("Marker for modified models not found in library file. Please add a line with '***MODIFIED MODELS FOLLOW THIS COMMENT***' after all original models")
        cut_pos = m_marker.end()
        out_text = orig_lib_text[:cut_pos] + '\n'
        separator_string = '* ------------------ Modified Model ------------------ \n'
    
        # Find the original device definition block
        device_block_re = re.compile(
            rf'^\s*\.subckt\s+({re.escape(orig_hemt_name)})\s+gatein\s+drainin\s+sourcein\b(.*?^\s*\.ends\b)',
            re.DOTALL | re.MULTILINE | re.IGNORECASE)
        match = device_block_re.search(orig_lib_text)
        if not match:
            raise ValueError(f"HEMT device {orig_hemt_name} not found in the library file.")
        orig_device_definition_block = match.group(0)
    
        # For each new model, create a modified block and append it
        for new_hemt_name, hemt_param_changes in model_changes.items():
            # Rename the device in the .subckt line to new_hemt_name
            subckt_name_re = re.compile(r'(^\s*\.subckt\s+)(\S+)', re.IGNORECASE | re.MULTILINE)
            def _rename_subckt(m):
                prefix, name = m.group(1), m.group(2)
                return prefix + new_hemt_name
            modif_dev_def_block = subckt_name_re.sub(_rename_subckt, orig_device_definition_block, count=1)
    
            # Apply parameter changes (reuse original logic)
            for param_spice_name in hemt_param_changes.keys():
                if not self._is_epc_param(param_spice_name, type='spice'):
                    continue
                new_val_str = repr(hemt_param_changes[param_spice_name])
                pat = re.compile(r'(?<!\w)(' + re.escape(param_spice_name.rsplit('_', 1)[0]) + r')\s*=\s*(\{.*?\}|[^\s]+)', re.DOTALL | re.IGNORECASE)
                def _replace_assign(m):
                    lhs = m.group(1)
                    rhs = m.group(2)
                    if rhs.startswith('{') and rhs.endswith('}'):
                        inner = rhs[1:-1]
                        m2 = re.match(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)$', inner)
                        if m2:
                            rest = m2.group(2) or ''
                            rest = re.sub(r'\s+', '', rest)
                            return lhs + '={' + new_val_str + rest + '}'
                        return lhs + '={' + new_val_str + '}'
                    else:
                        return f"{lhs}={new_val_str}"
                modif_dev_def_block = pat.sub(_replace_assign, modif_dev_def_block)
    
            # Append the modified block
            out_text += separator_string + modif_dev_def_block + '\n'
    
        # Write the modified content back
        with open(lib_path, 'w', encoding='utf-8') as f:
            f.write(out_text)

    def update_netlist_params(self, new_cnfg: dict, netlist):
        # Update all params & sim command
        netlist.add_instruction(f".tran {2*max([new_cnfg[f"T_sw_{i}"] for i in range(1, self.N_devices + 1)])*1e6:.3}u")
        for k, v in sorted(new_cnfg.items()):
            if isinstance(v, str):
                # This can be removed when Nuno fixes the AscEditor issue with string params
                netlist.remove_Xinstruction(search_pattern=rf'^\.param\s+{k}\b')
                netlist.set_parameter(k, f'"{v}"')
            elif isinstance(v, float) or isinstance(v, int):
                netlist.set_parameter(k, v)
            else:
                raise ValueError(f"Unsure if parameter type is supported for {k}: {type(v)}")
            
        # Update gate driving voltage sources
        for i in range(1, self.N_devices + 1):
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

        # Update Power Loss arb. curr. sources:
        for i in range(1, self.N_devices + 1):
            gan_loss_cmd = f"I=V(x{i}:hemt_s_nols)*-I(x{i}:L_common_source)+V(g{i})*I(xg{i}:Lg_minusLcommonsrc)+V(d{i})*I(x{i}:R_drain)"
            netlist.set_component_value(f'B{i}', gan_loss_cmd)

    def reset_and_trim_netlist(self, new_cnfg: dict, netlist, sim_index: int):
        # reset netlist due to removed hemts or changed params
        netlist.reset_netlist()
        # Adjust number of devices in the netlist
        for i in range(self.N_devices + 1, self.TOTAL_DRAWN_DEVICES + 1):
            netlist.remove_component(f'X{i}')
            netlist.remove_component(f'XG{i}')
            netlist.remove_component(f'Vgdrv{i}')
            netlist.remove_component(f'R{i}')
            netlist.remove_component(f'R_{i-1}{i}')
        
        # Check for EPC parameters that differ from the base configuration
        changed_params_by_device = {}
        for i in range(1, self.N_devices + 1):
            # new_cnfg['A1_1']=10
            device_key = new_cnfg[f'dev_{i}']
            changed_params = {k: v for k, v in new_cnfg.items() if k.endswith(f'_{i}') and self.base_cnfg_dict[k] != v}
            if changed_params:
                changed_params_by_device[device_key.split('_')[0] + f'_{i}'] = changed_params
                new_cnfg[f'dev_{i}'] = device_key.split('_')[0] + f'_{i}'
            # Update changed params
        
        # self.generate_modif_lib(orig_hemt_name=device_key, model_changes=changed_params_by_device)
        changed_device_models = self.generate_modif_device_models(orig_hemt_name=device_key, model_changes=changed_params_by_device)
        for block in changed_device_models:
            # Construct the new file name using the asc file name and the second string after a space in the block
            device_name = block.splitlines()[0].split()[1]
            lib_file_path = Path(self.sim_output_folder) / f"{netlist.asc_file_path.stem}_{sim_index}_{device_name}.lib"
            # Write the modified block to the .lib file
            with open(lib_file_path, 'w', encoding='utf-8') as lib_file:
                lib_file.write(block + '\n')
            
            # Add a .lib instruction to the netlist for the newly created .lib file
            netlist.add_instruction(f".lib {lib_file_path.name}")

    def introduce_meas_commands(self, new_cnfg: dict, netlist):
        # Remove existing .meas commands
        netlist.remove_Xinstruction(search_pattern='.meas')
        for i in range(1, self.N_devices + 1):
            T_sw   = new_cnfg[f"T_sw_{i}"]
            T_on_i = new_cnfg[f"T_on_{i}"]
            tdrv_r = new_cnfg[f"tdrv_rise_{i}"] 
            T_off  = new_cnfg[f"T_off_{i}"]

            # Power Loss measurements
            delta_t_loss_meas = min(500e-9, T_on_i * 0.025)
            P_on_times  = [T_sw, T_sw + delta_t_loss_meas]
            P_off_times = [T_sw, T_on_i + tdrv_r + T_on_i + delta_t_loss_meas]
            # Vds Overshoot params
            Vds_pk_times = [T_sw + tdrv_r + T_on_i, T_sw + tdrv_r + T_on_i + T_off / 2]
            # Vgs Overshoot params
            Vgs_pk_times = [T_sw, T_sw + tdrv_r + T_on_i / 2]
            # RMS conduction current measurement triggers
            ich_min = 0.05 * new_cnfg['I_DC'] / self.N_devices
            ich_rise_delay = 10e-6
            vch_min = 0.05 * new_cnfg['V_DC']
            ich_fall_delay = 15e-6
            # Junction temp. measurement window
            Tj_on_times = [T_sw, T_sw + 0.5e-6]
            Tj_off_times = [T_sw + T_on_i, T_sw + T_on_i + 0.5e-6]

            # loss_expression = f"V(G{i})*Ix(X{i}:U1:gatein)+V(D{i})*Ix(X{i}:U1:drainin)+V(X{i}:hemt_s_noLs)*Ix(X{i}:U1:sourcein)"
            gan_loss_cmd = f"V(x{i}:hemt_s_nols)*-I(x{i}:L_common_source)+V(g{i})*I(xg{i}:Lg_minusLcommonsrc)+V(d{i})*I(x{i}:R_drain)"
            loss_expression = gan_loss_cmd
            netlist.add_instructions(
                f".meas X{i}_rms_cond RMS I(x{i}:L_drain) TRIG I(x{i}:L_drain)={ich_min} TD={1e6*ich_rise_delay}u RISE=1 TARG V(D{i},S{i})={vch_min} TD={1e6*ich_fall_delay}u RISE=1",
                # f".meas X{i}_cond_current FIND I(X{i}:R_drain) AT {1e6*t_on_curr_meas:.6}u",
                f".meas X{i}_E_on INTEG {loss_expression} FROM {1e6*P_on_times[0]:.6}u TO {1e6*P_on_times[1]:.6}u",
                f".meas X{i}_E_off INTEG {loss_expression} FROM {1e6*P_off_times[0]:.6}u TO {1e6*P_off_times[1]:.6}u",
                f".meas X{i}_vds_pk MAX V(D{i},X{i}:hemt_s_noLs) FROM {1e6*Vds_pk_times[0]:.6}u TO {1e6*Vds_pk_times[1]:.6}u",
                f".meas X{i}_vgs_pk MAX V(G{i},X{i}:hemt_s_noLs) FROM {1e6*Vgs_pk_times[0]:.6}u TO {1e6*Vgs_pk_times[1]:.6}u",
                f".meas X{i}_MaxTj_on MAX V(Tj{i}) FROM {1e6*Tj_on_times[0]:.6}u TO {1e6*Tj_on_times[1]:.6}u",
                f".meas X{i}_MaxTj_off MAX V(Tj{i}) FROM {1e6*Tj_off_times[0]:.6}u TO {1e6*Tj_off_times[1]:.6}u",
            )

    def prepare_netlist_for_sim(self, new_cnfg: dict, netlist, sim_index: int):
        # Run all necessary steps to prepare netlist for simulation
        self.reset_and_trim_netlist(new_cnfg=new_cnfg, netlist=netlist, sim_index=sim_index)
        self.update_netlist_params(new_cnfg=new_cnfg, netlist=netlist)
        self.introduce_meas_commands(new_cnfg=new_cnfg, netlist=netlist)

    def define_selective_saving(self, wfms_to_save, netlist):
        # Remove existing .save commands
        netlist.remove_Xinstruction(search_pattern='.save')
        save_cmds = []
        for i in range(1, self.N_devices + 1):
            save_cmds.append(f".save V(G{i}) V(D{i}) V(S{i}) I(X{i}:R_drain) I(X{i}:L_drain) V(Tj{i}) V(X{i}:hemt_s_nols) I(x{i}:L_common_source) I(xg{i}:Lg_minusLcommonsrc)")
        netlist.add_instructions(*save_cmds)

# Useful for starting values of sim V3:

# doo = [lambda i: f"R_drain_{i}", lambda i: f"L_drain_{i}", lambda i: f"R_source_{i}", lambda i: f"L_source_{i}", 
#        lambda i: f"R_pcb_branch_{i}", lambda i: f"Lg_uncommon_{i}", lambda i: f"L_common_source_{i}", 
#        lambda i: f"R_driver_{i}", lambda i: f"R_g_{i}", lambda i: f"dev_{i}"]
# vals = ["1u", "1p", "1u", "1p", "1u", "1p", "1p", "1u", "1u", "0"]
# for func, val in zip(doo, vals):
#     for i in range(1, 11):
#         print(".param ", func(i)," = ", val)

if __name__ == "__main__":
    
    sim_config = LtSimConfiguration()
    a_dict = sim_config.get_hemt_parameters_from_lib(hemt_name='EPC2305')
    a_dict = sim_config.get_hemt_parameters_from_lib(hemt_name=['EPC2305', 'EPC2305_modified'])

    model_changes = {
    'EPC2305_mod1': {'A1': 1.2, 'k2': 2.3},
    'EPC2305_modified': {}
    }
    sim_config.generate_modif_lib(orig_hemt_name='EPC2305', model_changes=model_changes)
