import numpy as np
from itertools import product
from spicelib.editor import base_editor
import re
from pathlib import Path
import shutil

class LtSimConfiguration:
    def __init__(self, base_config: dict = None):
        if base_config:
            self.base_cnfg_dict = base_config.copy()
        else:
            self.base_cnfg_dict = {}
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
        raise ValueError("Not yet implemented")
        self.base_cnfg_dict['N_devices'] = N_devices

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
