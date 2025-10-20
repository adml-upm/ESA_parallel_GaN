import numpy as np
from itertools import product
from spicelib.editor import base_editor

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
