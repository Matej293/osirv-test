import os
import yaml
import torch

class ConfigManager:
    def __init__(self, config_path="config/default_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_class_weights()
    
    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found.")
            return {}
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # resolve paths in the config
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.config_path)))
        self._resolve_paths(config, base_dir)
        return config
    
    def _resolve_paths(self, config, base_dir):
        """Resolve all paths in config dictionary that start with ./"""
        path_keys = {
            'data': ['csv_path', 'img_dir'],
            'model': ['pretrained_path', 'saved_path'],
            'logging': ['log_dir']
        }
        
        for section, keys in path_keys.items():
            if section in config:
                for key in keys:
                    if key in config[section] and config[section][key].startswith("./"):
                        rel_path = config[section][key][2:]  # removing the ./ prefix
                        config[section][key] = os.path.normpath(os.path.join(base_dir, rel_path))
    
    def _setup_class_weights(self):
        if 'data' in self.config and 'ssa_count' in self.config['data'] and 'hp_count' in self.config['data']:
            ssa_count = self.config['data']['ssa_count']
            hp_count = self.config['data']['hp_count']
            total_count = ssa_count + hp_count
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.class_weights = torch.tensor(
                [total_count / hp_count, total_count / ssa_count], 
                device=self.device
            )
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        
        if isinstance(value, str):
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return int(value)
            except (ValueError, TypeError):
                pass
                
        return value
    
    def update_from_args(self, args):
        arg_mapping = {
            'model_path': 'model.pretrained_path',
            'save_model_path': 'model.saved_path',
            'csv_path': 'data.csv_path',
            'img_dir': 'data.img_dir',
            'batch_size': 'data.batch_size',
            'epochs': 'training.epochs',
            'lr': 'training.learning_rate',
            'weight_decay': 'training.weight_decay',
            'ssa_threshold': 'training.ssa_threshold',
            'hp_threshold': 'training.hp_threshold',
        }
        
        for arg_name, config_path in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self._update_nested(self.config, config_path, getattr(args, arg_name))
        
        self._setup_class_weights()
    
    def _update_nested(self, config_dict, key_path, value):
        if value is None:
            return
            
        keys = key_path.split('.')
        d = config_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def update(self, update_dict):
        """Update configuration with the provided dictionary of values."""
        for key_path, value in update_dict.items():
            self._update_nested(self.config, key_path, value)
        
        self._setup_class_weights()
        return self
    
    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False)