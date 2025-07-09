"""Demographic analysis module for election data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from scipy.stats import pearsonr, spearmanr

from ..utils.logger import setup_logger


class DemographicAnalyzer:
    """Analyzes demographic patterns and their correlation with electoral performance."""
    
    def __init__(self, ac_data: pd.DataFrame, ac_name: str):
        """
        Initialize demographic analyzer for a specific assembly constituency.
        
        Args:
            ac_data: DataFrame containing data for the specific AC
            ac_name: Name of the assembly constituency
        """
        self.ac_data = ac_data.copy()
        self.ac_name = ac_name
        self.logger = setup_logger(__name__)
        
        # Cache for analysis results to avoid repeated calculations
        self._analysis_cache = {}
        
        # Define demographic columns
        self.age_columns = ['P_20', 'P_20_30', 'P_30_40', 'P_40_50', 'P_50_60', 'Above_60']
        self.gender_columns = ['MALE', 'FEMALE']
        self.available_age_columns = [col for col in self.age_columns if col in self.ac_data.columns]
        self.available_gender_columns = [col for col in self.gender_columns if col in self.ac_data.columns]
        
        # Calculate LF vote share for correlation analysis
        if 'LF' in self.ac_data.columns and 'ValidVotes' in self.ac_data.columns:
            self.ac_data['lf_vote_share'] = (self.ac_data['LF'] / self.ac_data['ValidVotes']) * 100
    
    def analyze_age_group_impact(self) -> Dict[str, Any]:
        """
        Analyze the impact of different age groups on LF performance.
        
        Returns:
            Dictionary containing age group analysis results
        """
        if not self.available_age_columns:
            return {'error': 'No age demographic columns found'}
        
        if 'lf_vote_share' not in self.ac_data.columns:
            return {'error': 'Cannot calculate LF vote share for correlation analysis'}
        
        age_analysis = {
            'available_age_groups': self.available_age_columns,
            'correlations': {},
            'age_group_statistics': {},
            'high_performing_demographics': {}
        }
        
        # Calculate correlations between age groups and LF performance
        for age_col in self.available_age_columns:
            if pd.api.types.is_numeric_dtype(self.ac_data[age_col]):
                # Calculate correlation
                correlation, p_value = pearsonr(
                    self.ac_data[age_col].fillna(0), 
                    self.ac_data['lf_vote_share'].fillna(0)
                )
                
                age_analysis['correlations'][age_col] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'significance': 'significant' if p_value < 0.05 else 'not_significant'
                }
                
                # Calculate basic statistics for this age group
                age_analysis['age_group_statistics'][age_col] = {
                    'mean': float(self.ac_data[age_col].mean()),
                    'median': float(self.ac_data[age_col].median()),
                    'std': float(self.ac_data[age_col].std()),
                    'min': float(self.ac_data[age_col].min()),
                    'max': float(self.ac_data[age_col].max())
                }
        
        # Identify age groups with strongest positive correlation
        significant_correlations = {
            col: data['correlation'] 
            for col, data in age_analysis['correlations'].items() 
            if data['significance'] == 'significant' and data['correlation'] > 0.1
        }
        
        if significant_correlations:
            best_age_group = max(significant_correlations.items(), key=lambda x: x[1])
            age_analysis['strongest_positive_correlation'] = {
                'age_group': best_age_group[0],
                'correlation': best_age_group[1]
            }
        
        # Analyze youth vote (combining young age groups)
        youth_columns = [col for col in ['P_20', 'P_20_30'] if col in self.available_age_columns]
        if youth_columns:
            youth_total = self.ac_data[youth_columns].sum(axis=1)
            if not youth_total.empty:
                youth_correlation, youth_p_value = pearsonr(
                    youth_total.fillna(0), 
                    self.ac_data['lf_vote_share'].fillna(0)
                )
                
                age_analysis['youth_vote_analysis'] = {
                    'correlation': float(youth_correlation),
                    'p_value': float(youth_p_value),
                    'significance': 'significant' if youth_p_value < 0.05 else 'not_significant',
                    'mean_youth_population': float(youth_total.mean())
                }
        
        # Analyze senior vote (age >50)
        senior_columns = [col for col in ['P_50_60', 'Above_60'] if col in self.available_age_columns]
        if senior_columns:
            senior_total = self.ac_data[senior_columns].sum(axis=1)
            if not senior_total.empty:
                senior_correlation, senior_p_value = pearsonr(
                    senior_total.fillna(0), 
                    self.ac_data['lf_vote_share'].fillna(0)
                )
                
                age_analysis['senior_vote_analysis'] = {
                    'correlation': float(senior_correlation),
                    'p_value': float(senior_p_value),
                    'significance': 'significant' if senior_p_value < 0.05 else 'not_significant',
                    'mean_senior_population': float(senior_total.mean())
                }
        
        self.logger.info(f"Age group impact analysis completed for {self.ac_name}")
        return age_analysis
    
    def analyze_gender_dynamics(self) -> Dict[str, Any]:
        """
        Analyze gender-based voting patterns and their correlation with LF performance.
        
        Returns:
            Dictionary containing gender analysis results
        """
        if len(self.available_gender_columns) < 2:
            return {'error': 'Insufficient gender data for analysis'}
        
        if 'lf_vote_share' not in self.ac_data.columns:
            return {'error': 'Cannot calculate LF vote share for correlation analysis'}
        
        gender_analysis = {
            'available_gender_columns': self.available_gender_columns,
            'correlations': {},
            'gender_statistics': {},
            'gender_gap_analysis': {}
        }
        
        # Calculate correlations for each gender
        for gender_col in self.available_gender_columns:
            if pd.api.types.is_numeric_dtype(self.ac_data[gender_col]):
                correlation, p_value = pearsonr(
                    self.ac_data[gender_col].fillna(0), 
                    self.ac_data['lf_vote_share'].fillna(0)
                )
                
                gender_analysis['correlations'][gender_col] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'significance': 'significant' if p_value < 0.05 else 'not_significant'
                }
                
                gender_analysis['gender_statistics'][gender_col] = {
                    'mean': float(self.ac_data[gender_col].mean()),
                    'median': float(self.ac_data[gender_col].median()),
                    'std': float(self.ac_data[gender_col].std())
                }
        
        # Calculate gender ratio and its correlation with LF performance
        if 'MALE' in self.ac_data.columns and 'FEMALE' in self.ac_data.columns:
            total_gender = self.ac_data['MALE'] + self.ac_data['FEMALE']
            # Avoid division by zero
            mask = total_gender > 0
            male_ratio = pd.Series(index=self.ac_data.index, dtype=float)
            male_ratio[mask] = self.ac_data.loc[mask, 'MALE'] / total_gender[mask]
            male_ratio[~mask] = 0.5  # Default to 50% if no data
            
            ratio_correlation, ratio_p_value = pearsonr(
                male_ratio.fillna(0.5), 
                self.ac_data['lf_vote_share'].fillna(0)
            )
            
            gender_analysis['gender_gap_analysis'] = {
                'male_ratio_correlation': float(ratio_correlation),
                'male_ratio_p_value': float(ratio_p_value),
                'male_ratio_significance': 'significant' if ratio_p_value < 0.05 else 'not_significant',
                'mean_male_ratio': float(male_ratio.mean()),
                'std_male_ratio': float(male_ratio.std())
            }
            
            # Identify booths with extreme gender ratios
            extreme_male_heavy = (male_ratio > 0.65).sum()
            extreme_female_heavy = (male_ratio < 0.35).sum()
            
            gender_analysis['gender_gap_analysis'].update({
                'extreme_male_heavy_booths': int(extreme_male_heavy),
                'extreme_female_heavy_booths': int(extreme_female_heavy),
                'extreme_ratio_percentage': float(((extreme_male_heavy + extreme_female_heavy) / len(self.ac_data)) * 100)
            })
        
        # Analyze which gender shows stronger correlation with LF
        correlations = gender_analysis['correlations']
        if len(correlations) >= 2:
            strongest_correlation = max(correlations.items(), key=lambda x: abs(x[1]['correlation']))
            gender_analysis['strongest_gender_correlation'] = {
                'gender': strongest_correlation[0],
                'correlation': strongest_correlation[1]['correlation']
            }
        
        self.logger.info(f"Gender dynamics analysis completed for {self.ac_name}")
        return gender_analysis
    
    def analyze_minority_patterns(self) -> Dict[str, Any]:
        """
        Analyze minority community voting patterns and correlation with LF performance.
        
        Returns:
            Dictionary containing minority analysis results
        """
        # Check cache first
        if 'minority_patterns' in self._analysis_cache:
            return self._analysis_cache['minority_patterns']
        
        if 'Minority' not in self.ac_data.columns:
            result = {'error': 'Minority column not found in data'}
            self._analysis_cache['minority_patterns'] = result
            return result
        
        if 'lf_vote_share' not in self.ac_data.columns:
            result = {'error': 'Cannot calculate LF vote share for correlation analysis'}
            self._analysis_cache['minority_patterns'] = result
            return result
        
        minority_analysis = {
            'minority_statistics': {},
            'correlation_analysis': {},
            'booth_categorization': {}
        }
        
        # Basic minority statistics
        minority_data = self.ac_data['Minority'].fillna(0)
        minority_analysis['minority_statistics'] = {
            'mean_minority_percentage': float(minority_data.mean()),
            'median_minority_percentage': float(minority_data.median()),
            'std_minority_percentage': float(minority_data.std()),
            'min_minority_percentage': float(minority_data.min()),
            'max_minority_percentage': float(minority_data.max()),
            'total_booths': len(self.ac_data)
        }
        
        # Correlation analysis
        correlation, p_value = pearsonr(
            minority_data, 
            self.ac_data['lf_vote_share'].fillna(0)
        )
        
        minority_analysis['correlation_analysis'] = {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'significance': 'significant' if p_value < 0.05 else 'not_significant',
            'interpretation': self._interpret_minority_correlation(correlation, p_value)
        }
        
        # Categorize booths by minority concentration
        minority_low = (minority_data <= 10).sum()
        minority_medium = ((minority_data > 10) & (minority_data <= 30)).sum()
        minority_high = (minority_data > 30).sum()
        
        minority_analysis['booth_categorization'] = {
            'low_minority_booths': int(minority_low),
            'medium_minority_booths': int(minority_medium),
            'high_minority_booths': int(minority_high),
            'low_minority_percentage': float((minority_low / len(self.ac_data)) * 100),
            'medium_minority_percentage': float((minority_medium / len(self.ac_data)) * 100),
            'high_minority_percentage': float((minority_high / len(self.ac_data)) * 100)
        }
        
        # Analyze LF performance by minority concentration
        performance_by_minority = {}
        
        for category, mask in [
            ('low_minority', minority_data <= 10),
            ('medium_minority', (minority_data > 10) & (minority_data <= 30)),
            ('high_minority', minority_data > 30)
        ]:
            if mask.sum() > 0:
                category_lf_performance = self.ac_data.loc[mask, 'lf_vote_share']
                performance_by_minority[category] = {
                    'booth_count': int(mask.sum()),
                    'mean_lf_vote_share': float(category_lf_performance.mean()),
                    'median_lf_vote_share': float(category_lf_performance.median()),
                    'std_lf_vote_share': float(category_lf_performance.std())
                }
        
        minority_analysis['performance_by_minority_concentration'] = performance_by_minority
        
        # Cache the result before returning
        self._analysis_cache['minority_patterns'] = minority_analysis
        
        self.logger.info(f"Minority patterns analysis completed for {self.ac_name}")
        return minority_analysis
    
    def calculate_demographic_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between all demographic variables and LF performance.
        
        Returns:
            DataFrame containing correlation matrix
        """
        if 'lf_vote_share' not in self.ac_data.columns:
            raise ValueError("Cannot calculate correlations without LF vote share")
        
        # Collect all demographic columns
        demographic_columns = []
        demographic_columns.extend(self.available_age_columns)
        demographic_columns.extend(self.available_gender_columns)
        
        if 'Minority' in self.ac_data.columns:
            demographic_columns.append('Minority')
        
        # Add LF vote share for correlation analysis
        analysis_columns = demographic_columns + ['lf_vote_share']
        
        # Select only numeric columns that exist in the data
        existing_columns = [col for col in analysis_columns if col in self.ac_data.columns]
        numeric_data = self.ac_data[existing_columns].select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric demographic data available for correlation analysis")
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Focus on correlations with LF vote share
        if 'lf_vote_share' in correlation_matrix.columns:
            lf_correlations = correlation_matrix['lf_vote_share'].drop('lf_vote_share')
            
            # Create a detailed correlation DataFrame
            correlation_details = pd.DataFrame({
                'variable': lf_correlations.index,
                'correlation': lf_correlations.values,
                'abs_correlation': abs(lf_correlations.values)
            })
            
            # Sort by absolute correlation value
            correlation_details = correlation_details.sort_values('abs_correlation', ascending=False)
            
            self.logger.info(f"Demographic correlation analysis completed for {self.ac_name}")
            return correlation_details
        else:
            # Return full correlation matrix if LF vote share not available
            return correlation_matrix
    
    def identify_demographic_opportunities(self) -> Dict[str, Any]:
        """
        Identify demographic segments with high potential for LF improvement.
        
        Returns:
            Dictionary containing demographic opportunities
        """
        if 'lf_vote_share' not in self.ac_data.columns:
            return {'error': 'Cannot identify opportunities without LF vote share data'}
        
        opportunities = {
            'high_potential_segments': [],
            'underperforming_segments': [],
            'target_demographics': {}
        }
        
        # Analyze age group opportunities
        age_analysis = self.analyze_age_group_impact()
        if 'correlations' in age_analysis:
            for age_group, corr_data in age_analysis['correlations'].items():
                if corr_data['significance'] == 'significant':
                    if corr_data['correlation'] > 0.2:
                        opportunities['high_potential_segments'].append({
                            'segment': age_group,
                            'type': 'age_group',
                            'correlation': corr_data['correlation'],
                            'potential': 'high'
                        })
                    elif corr_data['correlation'] < -0.2:
                        opportunities['underperforming_segments'].append({
                            'segment': age_group,
                            'type': 'age_group',
                            'correlation': corr_data['correlation'],
                            'issue': 'negative_correlation'
                        })
        
        # Analyze gender opportunities
        gender_analysis = self.analyze_gender_dynamics()
        if 'correlations' in gender_analysis:
            for gender, corr_data in gender_analysis['correlations'].items():
                if corr_data['significance'] == 'significant' and abs(corr_data['correlation']) > 0.15:
                    opportunities['target_demographics'][f'{gender}_focus'] = {
                        'correlation': corr_data['correlation'],
                        'recommendation': 'strong_positive' if corr_data['correlation'] > 0 else 'needs_attention'
                    }
        
        # Analyze minority community opportunities
        minority_analysis = self.analyze_minority_patterns()
        if 'correlation_analysis' in minority_analysis:
            corr_data = minority_analysis['correlation_analysis']
            if corr_data['significance'] == 'significant':
                opportunities['target_demographics']['minority_community'] = {
                    'correlation': corr_data['correlation'],
                    'interpretation': corr_data['interpretation']
                }
        
        # Identify specific booth-level opportunities
        opportunities['booth_level_opportunities'] = self._identify_booth_opportunities()
        
        self.logger.info(f"Demographic opportunities identified for {self.ac_name}")
        return opportunities
    
    def _interpret_minority_correlation(self, correlation: float, p_value: float) -> str:
        """
        Interpret the correlation between minority percentage and LF performance.
        
        Args:
            correlation: Correlation coefficient
            p_value: Statistical significance p-value
            
        Returns:
            Interpretation string
        """
        if p_value >= 0.05:
            return "No statistically significant relationship"
        
        if correlation > 0.3:
            return "Strong positive correlation - LF performs better in minority-majority areas"
        elif correlation > 0.1:
            return "Moderate positive correlation - some LF advantage in minority areas"
        elif correlation > -0.1:
            return "Weak correlation - minimal relationship"
        elif correlation > -0.3:
            return "Moderate negative correlation - LF underperforms in minority areas"
        else:
            return "Strong negative correlation - significant LF weakness in minority areas"
    
    def _identify_booth_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identify specific booths with favorable demographics but underperforming LF results.
        
        Returns:
            List of booth opportunity dictionaries
        """
        if 'lf_vote_share' not in self.ac_data.columns:
            return []
        
        opportunities = []
        
        # Define thresholds
        low_lf_threshold = self.ac_data['lf_vote_share'].quantile(0.33)  # Bottom third
        
        # Get minority analysis once outside the loop to avoid repeated calls (use cached version)
        minority_analysis = self.analyze_minority_patterns()
        minority_correlation_positive = (
            'correlation_analysis' in minority_analysis and 
            minority_analysis['correlation_analysis']['correlation'] > 0.1
        )
        
        for idx, row in self.ac_data.iterrows():
            if row['lf_vote_share'] < low_lf_threshold:
                # Check if demographics are favorable
                favorable_factors = []
                
                # Check age demographics (example: if youth vote correlates positively)
                youth_cols = [col for col in ['P_20', 'P_20_30'] if col in self.ac_data.columns]
                if youth_cols:
                    youth_total = sum(row[col] for col in youth_cols)
                    if youth_total > self.ac_data[youth_cols].sum(axis=1).median():
                        favorable_factors.append('high_youth_population')
                
                # Check minority percentage (if positively correlated)
                if 'Minority' in self.ac_data.columns and minority_correlation_positive:
                    if row['Minority'] > self.ac_data['Minority'].median():
                        favorable_factors.append('high_minority_population')
                
                if favorable_factors:
                    opportunities.append({
                        'booth_name': row['BOOTHNAME'],
                        'current_lf_vote_share': float(row['lf_vote_share']),
                        'favorable_factors': favorable_factors,
                        'potential_rating': 'high' if len(favorable_factors) >= 2 else 'medium'
                    })
        
        # Sort by potential and limit to top opportunities
        opportunities.sort(key=lambda x: (len(x['favorable_factors']), -x['current_lf_vote_share']), reverse=True)
        return opportunities[:10]  # Top 10 opportunities
