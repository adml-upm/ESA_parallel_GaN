import numpy as np
from itertools import product
from spicelib.editor import base_editor
import re
from pathlib import Path

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
        self.hemt_lib_path = r"C:\Users\Alex\Documents\03 Projects\04 ESA parallel GaN\ESA_parallel_GaN\02 Simulations\00 ComonLibs\EPCGaNLibrary.lib"
        p = Path(self.hemt_lib_path)
        self.modif_hemt_lib_path = str(p.with_name(f"{p.stem}_modified{p.suffix}"))
        # self.out_file_path = r"C:\Users\Alex\Documents\03 Projects\04 ESA parallel GaN\ESA_parallel_GaN\02 Simulations\00 ComonLibs\EPCGaNLibrary_modified.lib"
        self.epc_model_param_mapping = {
            # 'si': 'si',
            # 'so': 'so',
            # 'sr': 'sr',
            # 'aWg': 'aWg',
            # 'Wg': 'Wg',
            'k_gm_factor': 'A1',  # ∝ transconductance gm scaling factor
            'Vgs_th': 'k2',  # Gate threshold voltage
            'Vgs_th_smoothing_factor': 'k3',  # Turn-on softness
            'Rds_on_base': 'rpara',  # Base Parasitic resistance
            # 'rpara_s_factor': 'rpara_s_factor', # Distributes Rdson to source and drain parasitics. Don't touch
            'aITc': 'aITc',  # Temperature coeff for gm, reduces Id per °C increase. (care for polarity)
            'arTc': 'arTc',  # Temperature coeff for Rds_on. (care for polarity)
            'Vgs_th_temp_coef': 'k2Tc',  # Temp coeff for Vth
            # 'Channel_modulation_0': 'x0_0',
            # 'channel_modulation_0_temp_coef': 'x0_0_TC',
            # 'Channel_modulation_1': 'x0_1',
            # 'channel_modulation_1_temp_coef': 'x0_1_TC',
            'Rgate_hemt': 'rg_value',
        }

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
        if param_name in self.base_cnfg_dict:
            if perturb_abs is not None and perturb_rel is not None:
                raise ValueError("Use either magnitude or percentage, not both")
            elif perturb_abs is None and perturb_rel is not None:
                perturb_abs = perturb_rel * abs(self.base_cnfg_dict[param_name])
            elif perturb_abs is not None and perturb_rel is None:
                pass
            else:
                perturb_abs = self.default_Kperturb * abs(self.base_cnfg_dict[param_name])
                
            self.perturb_params[param_name] = perturb_abs
            self.params_to_plot.append(param_name)
        else:
            raise ValueError(f"Parameter {param_name} not in base configuration dictionary.")
        
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
                yield config, {param_name: param_value}

    def single_param_perturbation_generator(self):
        """Generates configurations by perturbing one parameter at a time.
        returns: Yields a tuple of (configuration_dict, changed_params_dict) for each configuration."""
        for param_name, perturb_magnitude in self.perturb_params.items():
            if param_name in self.base_cnfg_dict.keys():
                config = self.base_cnfg_dict.copy()
                new_val = config[param_name] + perturb_magnitude
                config[param_name] = new_val
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

    def modify_hemt_parameters(self, hemt_name: str='EPC2305', hemt_param_changes: dict=None):
        """Modifies HEMT parameters in the base configuration.
        hemt_params: Dictionary of HEMT parameters to modify."""

        with open(self.hemt_lib_path, "r", encoding="utf-8") as f:
            file_text = f.read()

        # find the .subckt block for the requested hemt_name only (e.g. ".subckt EPC2305 gatein drainin sourcein ... .ends")
        device_block_re = re.compile(
            rf'^\s*\.subckt\s+({re.escape(hemt_name)})\s+gatein\s+drainin\s+sourcein\b(.*?^\s*\.ends\b)',
            re.DOTALL | re.MULTILINE | re.IGNORECASE)

        match = device_block_re.search(file_text)
        device_definition_block = match.group(0)
        if match.group(1) != hemt_name:
            raise ValueError(f"HEMT device {hemt_name} not found in the library file.")

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
    
        # start/end positions of the matched device block in the original file text
        modified_device_definition_block = device_definition_block  # we'll modify this copy
        # rename the device in the matched .subckt line by appending "_modified"
        new_name = f"{hemt_name}_modified"
        subckt_name_re = re.compile(r'(^\s*\.subckt\s+)'+re.escape(hemt_name)+r'(\b)', re.IGNORECASE | re.MULTILINE)
        modified_device_definition_block = subckt_name_re.sub(r'\1' + new_name + r'\2', modified_device_definition_block, count=1)

        # also replace the original block in the full file text so the file can be written as a full modified library if needed
        file_text = file_text[:match.start()] + modified_device_definition_block + file_text[match.end():]
        # apply each requested change: map friendly key -> model param name, then replace its value in the block
        for common_key in hemt_param_changes.keys() & self.epc_model_param_mapping.keys():
            param_spice_name = self.epc_model_param_mapping[common_key]
            new_value = hemt_param_changes[common_key]
            # update parsed params dictionary (if present) for bookkeeping
            if common_key in epc_model_params.keys():
                epc_model_params[common_key] = new_value

            # prepare replacement string for new_value
            if isinstance(new_value, (int, float)):
                new_val_str = repr(new_value)
            else:
                new_val_str = str(new_value)

            # find assignments like "param=VALUE" where VALUE is either a braced expression {...} or a single token
            pat = re.compile(r'(?<!\w)(' + re.escape(param_spice_name) + r')\s*=\s*(\{.*?\}|[^\s]+)', re.DOTALL | re.IGNORECASE)

            def _replace_assign(m):
                lhs = m.group(1)
                rhs = m.group(2)
                # braced expression: replace only the leading numeric coefficient, keep braces and any trailing variable/expression
                if rhs.startswith('{') and rhs.endswith('}'):
                    inner = rhs[1:-1]
                    # match a leading numeric coefficient (with optional sign and exponent) and optional remainder (e.g. *aWg)
                    m2 = re.match(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)$', inner)
                    if m2:
                        rest = m2.group(2) or ''
                        # remove extraneous whitespace around the remainder to preserve compact form like "*aWg"
                        rest = re.sub(r'\s+', '', rest)
                        # construct replacement preserving single braces
                        return lhs + '={' + new_val_str + rest + '}'
                    # if no leading numeric found, fall back to replacing entire RHS with the new value (preserve braces)
                    return lhs + '={' + new_val_str + '}'
                else:
                    # non-braced RHS: replace whole token
                    return f"{lhs}={new_val_str}"

            modified_device_definition_block = pat.sub(_replace_assign, modified_device_definition_block)

        # write the modified full file text to a static output file (overwrites existing)
        with open(self.modif_hemt_lib_path, "w", encoding="utf-8") as out_f:
            out_f.write(modified_device_definition_block)




