"""Data loading and validation module for election data analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from ..utils.config import ConfigManager
from ..utils.logger import setup_logger


class DataLoader:
    """Handles loading and initial validation of election data."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = setup_logger(__name__)
        self.data_paths = config.get_data_paths()
        self.required_columns = config.get_required_columns()
        
    def load_excel_data(self) -> pd.DataFrame:
        """
        Load election data from Excel file.
        
        Returns:
            DataFrame containing the election data
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file cannot be read or is empty
        """
        input_file = self.data_paths['input_file']
        
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        try:
            self.logger.info(f"Loading data from {input_file}")
            df = pd.read_excel(input_file)
            
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            raise ValueError(f"Could not read Excel file: {str(e)}")
    
    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns are present in the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all required columns are present
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(self.required_columns) - set(df.columns)
        
        if missing_columns:
            missing_str = ', '.join(missing_columns)
            raise ValueError(f"Missing required columns: {missing_str}")
        
        self.logger.info("All required columns are present")
        return True
    
    def get_assembly_constituencies(self, df: pd.DataFrame) -> List[str]:
        """
        Extract unique assembly constituency names from the data.
        
        Args:
            df: DataFrame containing election data
            
        Returns:
            List of unique assembly constituency names
        """
        if 'ac_name' not in df.columns:
            raise ValueError("Column 'ac_name' not found in data")
        
        ac_names = df['ac_name'].dropna().unique().tolist()
        ac_names = [str(name).strip() for name in ac_names if str(name).strip()]
        
        self.logger.info(f"Found {len(ac_names)} assembly constituencies")
        return sorted(ac_names)
    
    def get_parliamentary_constituencies(self, df: pd.DataFrame) -> List[str]:
        """
        Extract unique parliamentary constituency names from the data.
        
        Args:
            df: DataFrame containing election data
            
        Returns:
            List of unique parliamentary constituency names
        """
        if 'pc_name' not in df.columns:
            raise ValueError("Column 'pc_name' not found in data")
        
        pc_names = df['pc_name'].dropna().unique().tolist()
        pc_names = [str(name).strip() for name in pc_names if str(name).strip()]
        
        self.logger.info(f"Found {len(pc_names)} parliamentary constituencies")
        return sorted(pc_names)
    
    def load_constituency_data(self, df: pd.DataFrame, ac_name: str) -> pd.DataFrame:
        """
        Filter data for a specific assembly constituency.
        
        Args:
            df: Full DataFrame containing election data
            ac_name: Assembly constituency name to filter for
            
        Returns:
            DataFrame containing data only for the specified AC
            
        Raises:
            ValueError: If AC name not found in data
        """
        if 'ac_name' not in df.columns:
            raise ValueError("Column 'ac_name' not found in data")
        
        # Filter data for the specific AC
        ac_data = df[df['ac_name'] == ac_name].copy()
        
        if ac_data.empty:
            raise ValueError(f"No data found for assembly constituency: {ac_name}")
        
        self.logger.info(f"Loaded {len(ac_data)} records for AC: {ac_name}")
        return ac_data
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'assembly_constituencies': len(df['ac_name'].unique()) if 'ac_name' in df.columns else 0,
            'parliamentary_constituencies': len(df['pc_name'].unique()) if 'pc_name' in df.columns else 0,
            'total_booths': len(df['BOOTHNAME'].unique()) if 'BOOTHNAME' in df.columns else 0,
            'missing_values': df.isnull().sum().sum(),
            'columns': list(df.columns)
        }
        
        # Add vote-related statistics if available
        vote_columns = ['BJP', 'TMC', 'LF', 'ValidVotes', 'Voters2024']
        available_vote_cols = [col for col in vote_columns if col in df.columns]
        
        if available_vote_cols:
            summary['vote_statistics'] = {}
            for col in available_vote_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary['vote_statistics'][col] = {
                        'mean': float(df[col].mean()) if not df[col].isnull().all() else 0,
                        'median': float(df[col].median()) if not df[col].isnull().all() else 0,
                        'min': float(df[col].min()) if not df[col].isnull().all() else 0,
                        'max': float(df[col].max()) if not df[col].isnull().all() else 0
                    }
        
        return summary
