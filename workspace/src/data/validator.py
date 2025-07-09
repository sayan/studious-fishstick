"""Data validation module for election data analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from ..utils.logger import setup_logger


class DataValidator:
    """Validates data quality and consistency for election data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data validator with DataFrame.
        
        Args:
            df: DataFrame to validate
        """
        self.df = df.copy()
        self.logger = setup_logger(__name__)
    
    def check_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.
        
        Returns:
            Dictionary containing missing value analysis
        """
        missing_analysis = {}
        
        # Overall missing value statistics
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        missing_analysis['overall'] = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': missing_percentage
        }
        
        # Column-wise missing values
        missing_by_column = self.df.isnull().sum()
        missing_analysis['by_column'] = {}
        
        for column, missing_count in missing_by_column.items():
            missing_pct = (missing_count / len(self.df)) * 100
            missing_analysis['by_column'][column] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }
        
        # Identify rows with high missing values
        missing_per_row = self.df.isnull().sum(axis=1)
        high_missing_threshold = len(self.df.columns) * 0.5  # 50% or more columns missing
        high_missing_rows = missing_per_row[missing_per_row >= high_missing_threshold].index.tolist()
        
        missing_analysis['high_missing_rows'] = {
            'count': len(high_missing_rows),
            'indices': high_missing_rows[:10]  # Show first 10
        }
        
        self.logger.info(f"Missing value analysis complete. Overall missing: {missing_percentage:.2f}%")
        return missing_analysis
    
    def validate_vote_percentages(self) -> Dict[str, Any]:
        """
        Validate vote counts and percentages for consistency.
        
        Returns:
            Dictionary containing vote validation results
        """
        validation_results = {}
        
        # Check if vote columns exist
        vote_columns = ['BJP', 'TMC', 'LF']
        available_vote_cols = [col for col in vote_columns if col in self.df.columns]
        
        if not available_vote_cols:
            validation_results['status'] = 'no_vote_columns'
            return validation_results
        
        validation_results['available_vote_columns'] = available_vote_cols
        
        # Check for negative values
        negative_values = {}
        for col in available_vote_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                negative_count = (self.df[col] < 0).sum()
                negative_values[col] = negative_count
        
        validation_results['negative_values'] = negative_values
        
        # Check vote totals vs ValidVotes (if available)
        if 'ValidVotes' in self.df.columns and len(available_vote_cols) >= 2:
            # Calculate sum of party votes
            party_vote_sum = self.df[available_vote_cols].sum(axis=1)
            valid_votes = self.df['ValidVotes']
            
            # Find discrepancies (allowing for small rounding errors)
            discrepancy_threshold = 10  # votes
            discrepancies = abs(party_vote_sum - valid_votes) > discrepancy_threshold
            
            validation_results['vote_total_validation'] = {
                'discrepancy_count': discrepancies.sum(),
                'discrepancy_percentage': (discrepancies.sum() / len(self.df)) * 100,
                'avg_discrepancy': abs(party_vote_sum - valid_votes).mean()
            }
        
        # Check for unrealistic vote shares (>100%)
        if 'ValidVotes' in self.df.columns:
            unrealistic_shares = {}
            for col in available_vote_cols:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    vote_share = (self.df[col] / self.df['ValidVotes']) * 100
                    unrealistic_count = (vote_share > 100).sum()
                    unrealistic_shares[col] = unrealistic_count
            
            validation_results['unrealistic_vote_shares'] = unrealistic_shares
        
        self.logger.info("Vote percentage validation complete")
        return validation_results
    
    def check_demographic_consistency(self) -> Dict[str, Any]:
        """
        Validate demographic data for consistency.
        
        Returns:
            Dictionary containing demographic validation results
        """
        validation_results = {}
        
        # Check age group consistency
        age_columns = ['P_20', 'P_20_30', 'P_30_40', 'P_40_50', 'P_50_60', 'Above_60']
        available_age_cols = [col for col in age_columns if col in self.df.columns]
        
        if len(available_age_cols) >= 2:
            # Check if age groups sum to reasonable totals
            age_sum = self.df[available_age_cols].sum(axis=1)
            validation_results['age_groups'] = {
                'available_columns': available_age_cols,
                'mean_total': age_sum.mean(),
                'median_total': age_sum.median(),
                'min_total': age_sum.min(),
                'max_total': age_sum.max()
            }
        
        # Check gender consistency
        gender_columns = ['MALE', 'FEMALE']
        available_gender_cols = [col for col in gender_columns if col in self.df.columns]
        
        if len(available_gender_cols) == 2:
            gender_sum = self.df[available_gender_cols].sum(axis=1)
            validation_results['gender'] = {
                'mean_total': gender_sum.mean(),
                'median_total': gender_sum.median(),
                'min_total': gender_sum.min(),
                'max_total': gender_sum.max()
            }
            
            # Check for unrealistic gender ratios
            if 'MALE' in self.df.columns and 'FEMALE' in self.df.columns:
                total_gender = self.df['MALE'] + self.df['FEMALE']
                male_ratio = self.df['MALE'] / total_gender
                extreme_ratios = ((male_ratio < 0.3) | (male_ratio > 0.7)).sum()
                validation_results['gender']['extreme_ratios'] = extreme_ratios
        
        # Check minority percentages
        if 'Minority' in self.df.columns:
            minority_stats = {
                'mean': self.df['Minority'].mean(),
                'median': self.df['Minority'].median(),
                'min': self.df['Minority'].min(),
                'max': self.df['Minority'].max(),
                'negative_values': (self.df['Minority'] < 0).sum(),
                'over_100_percent': (self.df['Minority'] > 100).sum()
            }
            validation_results['minority'] = minority_stats
        
        self.logger.info("Demographic consistency check complete")
        return validation_results
    
    def detect_outliers(self, columns: List[str] = None) -> Dict[str, List[int]]:
        """
        Detect statistical outliers in specified columns.
        
        Args:
            columns: List of columns to check for outliers. If None, checks all numeric columns.
            
        Returns:
            Dictionary mapping column names to lists of outlier row indices
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                # Use IQR method for outlier detection
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_indices = self.df.index[outlier_mask].tolist()
                
                outliers[col] = outlier_indices
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"Outlier detection complete. Found {total_outliers} outliers across {len(columns)} columns")
        
        return outliers
    
    def validate_data_types(self) -> Dict[str, Any]:
        """
        Validate that columns have appropriate data types.
        
        Returns:
            Dictionary containing data type validation results
        """
        validation_results = {}
        
        # Expected data types for specific columns
        expected_types = {
            'ac_name': 'object',
            'pc_name': 'object',
            'BOOTHNAME': 'object',
            'BJP': 'numeric',
            'TMC': 'numeric',
            'LF': 'numeric',
            'ValidVotes': 'numeric',
            'Voters2024': 'numeric',
            'MALE': 'numeric',
            'FEMALE': 'numeric',
            'Minority': 'numeric'
        }
        
        type_issues = {}
        
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                
                if expected_type == 'numeric':
                    is_valid = pd.api.types.is_numeric_dtype(self.df[col])
                elif expected_type == 'object':
                    is_valid = pd.api.types.is_object_dtype(self.df[col])
                else:
                    is_valid = expected_type in actual_type
                
                if not is_valid:
                    type_issues[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        validation_results['type_issues'] = type_issues
        validation_results['all_dtypes'] = dict(self.df.dtypes.astype(str))
        
        self.logger.info(f"Data type validation complete. Found {len(type_issues)} type issues")
        return validation_results
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Dictionary containing complete quality assessment
        """
        self.logger.info("Generating comprehensive data quality report")
        
        quality_report = {
            'dataset_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_values': self.check_missing_values(),
            'vote_validation': self.validate_vote_percentages(),
            'demographic_validation': self.check_demographic_consistency(),
            'data_types': self.validate_data_types(),
            'outliers': self.detect_outliers()
        }
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['overall_quality_score'] = quality_score
        
        self.logger.info(f"Data quality report generated. Overall quality score: {quality_score:.2f}/100")
        return quality_report
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """
        Calculate an overall data quality score (0-100).
        
        Args:
            quality_report: Complete quality report
            
        Returns:
            Quality score between 0 and 100
        """
        score = 100.0
        
        # Deduct points for missing values
        missing_pct = quality_report['missing_values']['overall']['missing_percentage']
        score -= min(missing_pct, 30)  # Max 30 point deduction for missing values
        
        # Deduct points for data type issues
        type_issues = len(quality_report['data_types']['type_issues'])
        score -= min(type_issues * 5, 20)  # 5 points per type issue, max 20
        
        # Deduct points for vote validation issues
        vote_validation = quality_report['vote_validation']
        if 'vote_total_validation' in vote_validation:
            discrepancy_pct = vote_validation['vote_total_validation']['discrepancy_percentage']
            score -= min(discrepancy_pct, 15)  # Max 15 point deduction
        
        return max(score, 0.0)
