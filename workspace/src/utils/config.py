"""Configuration management for election data analysis system."""

import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration settings for the election analysis system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, uses default.
        """
        if config_file is None:
            # Default config file relative to workspace root
            workspace_root = Path(__file__).parent.parent.parent
            self.config_file = workspace_root / "config" / "analysis_config.yaml"
        else:
            self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get data file paths from configuration."""
        data_config = self.config.get('data', {})
        
        # Resolve relative paths relative to workspace root
        workspace_root = Path(__file__).parent.parent.parent
        input_file = workspace_root / data_config.get('input_file', '')
        
        return {
            'input_file': str(input_file),
            'output_directory': str(workspace_root / self.config.get('output', {}).get('output_directory', 'reports/'))
        }
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns from configuration."""
        return self.config.get('data', {}).get('required_columns', [])
    
    def get_analysis_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters from configuration."""
        return self.config.get('analysis', {})
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings from configuration."""
        return self.config.get('visualization', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings from configuration."""
        return self.config.get('output', {})
