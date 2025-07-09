"""Helper utility functions for the election data analysis system."""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
from typing import List, Dict, Any
import logging

from .logger import setup_logger


def save_dataframe_to_excel(df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1'):
    """
    Save DataFrame to Excel file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        sheet_name: Excel sheet name
    """
    logger = setup_logger(__name__)
    
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Excel
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
        logger.info(f"DataFrame saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving DataFrame to Excel: {str(e)}")
        raise


def save_figure_to_file(fig: plt.Figure, file_path: str, format: str = "png", dpi: int = 300):
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        file_path: Output file path
        format: File format (png, pdf, svg, etc.)
        dpi: Resolution for raster formats
    """
    logger = setup_logger(__name__)
    
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        fig.savefig(file_path, format=format, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        raise


def create_directory_structure(base_path: str, subdirs: List[str] = None):
    """
    Create directory structure for the project.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectories to create
    """
    logger = setup_logger(__name__)
    
    try:
        # Create base directory
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories if specified
        if subdirs:
            for subdir in subdirs:
                subdir_path = Path(base_path) / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {subdir_path}")
        
        logger.info(f"Directory structure created at {base_path}")
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        raise


def validate_file_paths(paths: List[str]) -> bool:
    """
    Validate that all specified file paths exist.
    
    Args:
        paths: List of file paths to validate
        
    Returns:
        True if all paths exist, False otherwise
    """
    logger = setup_logger(__name__)
    
    missing_paths = []
    
    for path in paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        logger.warning(f"Missing file paths: {missing_paths}")
        return False
    
    logger.info("All file paths validated successfully")
    return True


def clean_constituency_name(name: str) -> str:
    """
    Clean and standardize constituency name for file naming.
    
    Args:
        name: Raw constituency name
        
    Returns:
        Cleaned constituency name suitable for file naming
    """
    if not isinstance(name, str):
        name = str(name)
    
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^\w\s-]', '', name)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    cleaned = cleaned.lower()
    
    return cleaned


def format_number(number: float, decimal_places: int = 2) -> str:
    """
    Format number for display with proper decimal places and comma separation.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if pd.isna(number):
        return "N/A"
    
    return f"{number:,.{decimal_places}f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (should be in percentage form, e.g., 45.6 for 45.6%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if pd.isna(value):
        return "N/A"
    
    return f"{value:.{decimal_places}f}%"


def create_summary_table(data: Dict[str, Any], title: str = "Summary") -> pd.DataFrame:
    """
    Create a summary table from a dictionary of statistics.
    
    Args:
        data: Dictionary containing summary statistics
        title: Title for the summary table
        
    Returns:
        DataFrame containing formatted summary table
    """
    summary_rows = []
    
    def flatten_dict(d, parent_key='', sep='_'):
        """Recursively flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Flatten the input dictionary
    flattened_data = flatten_dict(data)
    
    # Create summary rows
    for key, value in flattened_data.items():
        # Format the key for display
        display_key = key.replace('_', ' ').title()
        
        # Format the value based on its type
        if isinstance(value, float):
            if 'percentage' in key.lower() or 'share' in key.lower():
                display_value = format_percentage(value)
            else:
                display_value = format_number(value)
        elif isinstance(value, int):
            display_value = format_number(value, 0)
        else:
            display_value = str(value)
        
        summary_rows.append({
            'Metric': display_key,
            'Value': display_value
        })
    
    return pd.DataFrame(summary_rows)


def calculate_percentiles(series: pd.Series, percentiles: List[float] = None) -> Dict[str, float]:
    """
    Calculate percentiles for a pandas Series.
    
    Args:
        series: Pandas Series to analyze
        percentiles: List of percentiles to calculate (0-100)
        
    Returns:
        Dictionary mapping percentile names to values
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]
    
    result = {}
    
    for p in percentiles:
        percentile_value = series.quantile(p / 100)
        result[f'p{p}'] = float(percentile_value)
    
    return result


def get_top_n_records(df: pd.DataFrame, column: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """
    Get top N records from DataFrame based on a specific column.
    
    Args:
        df: DataFrame to analyze
        column: Column name to sort by
        n: Number of top records to return
        ascending: Sort order (False for top values, True for bottom values)
        
    Returns:
        DataFrame containing top N records
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    sorted_df = df.sort_values(column, ascending=ascending)
    return sorted_df.head(n).copy()


def safe_divide(numerator: pd.Series, denominator: pd.Series, default_value: float = 0.0) -> pd.Series:
    """
    Safely divide two pandas Series, handling division by zero.
    
    Args:
        numerator: Numerator Series
        denominator: Denominator Series
        default_value: Value to use when denominator is zero
        
    Returns:
        Series containing the division result
    """
    result = pd.Series(index=numerator.index, dtype=float)
    
    # Handle division by zero
    zero_mask = (denominator == 0) | pd.isna(denominator)
    non_zero_mask = ~zero_mask
    
    # Set default value for zero denominators
    result[zero_mask] = default_value
    
    # Perform division for non-zero denominators
    if non_zero_mask.any():
        result[non_zero_mask] = numerator[non_zero_mask] / denominator[non_zero_mask]
    
    return result


def merge_analysis_results(results_list: List[Dict[str, Any]], merge_key: str = 'ac_name') -> Dict[str, Any]:
    """
    Merge multiple analysis results into a single comprehensive result.
    
    Args:
        results_list: List of analysis result dictionaries
        merge_key: Key to use for merging (e.g., 'ac_name')
        
    Returns:
        Merged analysis results
    """
    if not results_list:
        return {}
    
    merged_results = {
        'merged_at': pd.Timestamp.now().isoformat(),
        'total_analyses': len(results_list),
        'constituencies': []
    }
    
    # Collect all constituency names
    for result in results_list:
        if merge_key in result:
            merged_results['constituencies'].append(result[merge_key])
    
    # Merge specific analysis sections
    section_names = ['electoral_analysis', 'demographic_analysis', 'opportunity_analysis']
    
    for section in section_names:
        merged_results[section] = {}
        
        for result in results_list:
            if section in result and merge_key in result:
                ac_name = result[merge_key]
                merged_results[section][ac_name] = result[section]
    
    return merged_results


def create_analysis_metadata(ac_name: str, analysis_date: str = None) -> Dict[str, Any]:
    """
    Create metadata for analysis results.
    
    Args:
        ac_name: Assembly constituency name
        analysis_date: Date of analysis (ISO format)
        
    Returns:
        Metadata dictionary
    """
    if analysis_date is None:
        analysis_date = pd.Timestamp.now().isoformat()
    
    return {
        'ac_name': ac_name,
        'analysis_date': analysis_date,
        'system_version': '1.0.0',
        'analysis_type': 'constituency_level'
    }
