import numpy as np

class LtSimConfiguration:
    def __init__(self, base_config: dict = None):
        if base_config:
            self.base_cnfg_dict = base_config.copy()
        else:
            self.base_cnfg_dict = {}
        self.config_ranges = {}

    def add_param_with_range(self, param_name: str, range_vals: list, m_points: int=5, log_space:bool =False):
        if not log_space:  # Default is linear space
            self.config_ranges[param_name] = np.linspace(range_vals[0], range_vals[1], m_points)
        else:
            self.config_ranges[param_name] = np.geomspace(range_vals[0], range_vals[1], m_points)

    def get_all_cnfgs_sequentially(self):
        for param_name, param_range in self.config_ranges.items():
            for param_value in param_range:
                config = self.base_cnfg_dict.copy()
                config[param_name] = param_value
                yield config, {param_name: param_value}

    def get_all_multiparam_cnfgs(self):
        from itertools import product
        keys, values = zip(*self.config_ranges.items())
        for v in product(*values):
            config = self.base_cnfg_dict.copy()
            config.update(dict(zip(keys, v)))
            yield config