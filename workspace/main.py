"""Main execution pipeline for election data analysis system."""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import traceback

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.utils.helpers import create_directory_structure, create_analysis_metadata
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.analysis.electoral import ElectoralAnalyzer
from src.analysis.demographic import DemographicAnalyzer


class ElectionAnalysisPipeline:
    """Main pipeline for election data analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.logger = setup_logger(__name__, log_file="analysis.log")
        self.data_loader = DataLoader(self.config)
        
        # Create output directory structure
        output_dir = self.config.get_data_paths()['output_directory']
        create_directory_structure(output_dir, ['electoral', 'demographic', 'reports'])
        
        self.logger.info("Election Analysis Pipeline initialized")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis for all assembly constituencies.
        
        Returns:
            Dictionary containing complete analysis results
        """
        self.logger.info("Starting full analysis pipeline")
        
        try:
            # Load and validate data
            self.logger.info("Loading election data")
            df = self.data_loader.load_excel_data()
            self.data_loader.validate_required_columns(df)
            
            # Generate data quality report
            self.logger.info("Validating data quality")
            validator = DataValidator(df)
            quality_report = validator.generate_quality_report()
            
            # Log data quality score
            quality_score = quality_report['overall_quality_score']
            self.logger.info(f"Data quality score: {quality_score:.2f}/100")
            
            if quality_score < 70:
                self.logger.warning(f"Low data quality score: {quality_score:.2f}. Proceeding with caution.")
            
            # Get all assembly constituencies
            ac_names = self.data_loader.get_assembly_constituencies(df)
            self.logger.info(f"Found {len(ac_names)} assembly constituencies")
            
            # Process each constituency
            all_results = {
                'pipeline_metadata': {
                    'total_constituencies': len(ac_names),
                    'data_quality_score': quality_score,
                    'analysis_date': pd.Timestamp.now().isoformat()
                },
                'data_summary': self.data_loader.get_data_summary(df),
                'quality_report': quality_report,
                'constituency_results': {}
            }
            
            successful_analyses = 0
            failed_analyses = 0
            
            for i, ac_name in enumerate(ac_names, 1):
                self.logger.info(f"Processing constituency {i}/{len(ac_names)}: {ac_name}")
                
                try:
                    result = self.process_single_constituency(df, ac_name)
                    all_results['constituency_results'][ac_name] = result
                    successful_analyses += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {ac_name}: {str(e)}")
                    failed_analyses += 1
                    all_results['constituency_results'][ac_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Update metadata with success/failure counts
            all_results['pipeline_metadata'].update({
                'successful_analyses': successful_analyses,
                'failed_analyses': failed_analyses,
                'success_rate': (successful_analyses / len(ac_names)) * 100
            })
            
            self.logger.info(f"Full analysis completed. Success: {successful_analyses}, Failed: {failed_analyses}")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Critical error in full analysis pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def process_single_constituency(self, df: pd.DataFrame, ac_name: str) -> Dict[str, Any]:
        """
        Process analysis for a single assembly constituency.
        
        Args:
            df: Full dataset DataFrame
            ac_name: Assembly constituency name
            
        Returns:
            Dictionary containing analysis results for the constituency
        """
        try:
            # Load constituency-specific data
            ac_data = self.data_loader.load_constituency_data(df, ac_name)
            
            # Log constituency data for debugging
            self.logger.info(f"Loaded data for constituency: {ac_name}")
            self.logger.info(f"Constituency data shape: {ac_data.shape}")
            self.logger.info(f"Constituency data index range: {ac_data.index.min()} to {ac_data.index.max()}")
            self.logger.info(f"Constituency data head:\n{ac_data.head()}")
            
            # Create result structure
            result = {
                'metadata': create_analysis_metadata(ac_name),
                'basic_stats': {
                    'total_booths': len(ac_data),
                    'total_voters_2024': int(ac_data['Voters2024'].sum()) if 'Voters2024' in ac_data.columns else 0,
                    'total_valid_votes': int(ac_data['ValidVotes'].sum()) if 'ValidVotes' in ac_data.columns else 0
                }
            }
            
            # Calculate turnout if possible
            if 'Voters2024' in ac_data.columns and 'ValidVotes' in ac_data.columns:
                total_voters = ac_data['Voters2024'].sum()
                total_votes = ac_data['ValidVotes'].sum()
                if total_voters > 0:
                    result['basic_stats']['turnout_percentage'] = float((total_votes / total_voters) * 100)
            
            # Electoral analysis
            self.logger.info(f"Running electoral analysis for {ac_name}")
            electoral_analyzer = ElectoralAnalyzer(ac_data, ac_name)
            
            electoral_results = {
                'lf_performance': electoral_analyzer.calculate_lf_performance_metrics(),
                'competitive_landscape': electoral_analyzer.analyze_competitive_landscape(),
                'lf_strongholds': electoral_analyzer.identify_lf_strongholds().to_dict('records'),
                'margin_analysis': electoral_analyzer.calculate_margin_analysis().to_dict('records'),
                'three_way_competition': electoral_analyzer.assess_three_way_competition()
            }
            
            result['electoral_analysis'] = electoral_results
            
            # Demographic analysis
            self.logger.info(f"Running demographic analysis for {ac_name}")
            demographic_analyzer = DemographicAnalyzer(ac_data, ac_name)
            
            demographic_results: Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]] = {
                'age_group_impact': demographic_analyzer.analyze_age_group_impact(),
                'gender_dynamics': demographic_analyzer.analyze_gender_dynamics(),
                'minority_patterns': demographic_analyzer.analyze_minority_patterns(),
                'demographic_opportunities': demographic_analyzer.identify_demographic_opportunities()
            }
            
            # Try to calculate correlation matrix
            try:
                correlation_matrix = demographic_analyzer.calculate_demographic_correlations()
                demographic_results['correlation_matrix'] = correlation_matrix.to_dict('records')  # type: ignore
            except Exception as e:
                self.logger.warning(f"Could not calculate correlation matrix for {ac_name}: {str(e)}")
                demographic_results['correlation_matrix'] = []
            
            result['demographic_analysis'] = demographic_results
            
            # Generate summary insights
            result['summary_insights'] = self._generate_summary_insights(electoral_results, demographic_results)
            
            self.logger.info(f"Successfully completed analysis for {ac_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing constituency {ac_name}: {str(e)}")
            raise
    
    def _generate_summary_insights(self, electoral_results: Dict[str, Any], 
                                 demographic_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary insights from electoral and demographic analysis.
        
        Args:
            electoral_results: Electoral analysis results
            demographic_results: Demographic analysis results
            
        Returns:
            Dictionary containing summary insights
        """
        insights = {
            'key_findings': [],
            'opportunities': [],
            'challenges': [],
            'recommendations': []
        }
        
        # Analyze LF performance
        lf_performance = electoral_results.get('lf_performance', {})
        lf_vote_share = lf_performance.get('lf_vote_share_overall', 0)
        
        if lf_vote_share > 40:
            insights['key_findings'].append(f"Strong LF performance with {lf_vote_share:.1f}% vote share")
        elif lf_vote_share > 25:
            insights['key_findings'].append(f"Moderate LF performance with {lf_vote_share:.1f}% vote share")
        else:
            insights['key_findings'].append(f"Low LF performance with {lf_vote_share:.1f}% vote share")
            insights['challenges'].append("Below-average LF vote share requires strategic intervention")
        
        # Analyze competitive position
        if 'booths_lf_first' in lf_performance:
            lf_wins = lf_performance['booths_lf_first']
            total_booths = lf_performance.get('total_booths', 1)
            win_percentage = (lf_wins / total_booths) * 100
            
            if win_percentage > 30:
                insights['key_findings'].append(f"LF leads in {win_percentage:.1f}% of booths")
            else:
                insights['challenges'].append(f"LF leads in only {win_percentage:.1f}% of booths")
                insights['opportunities'].append("Focus on converting close second-place finishes")
        
        # Analyze demographic correlations
        age_analysis = demographic_results.get('age_group_impact', {})
        if 'strongest_positive_correlation' in age_analysis:
            strong_age_group = age_analysis['strongest_positive_correlation']['age_group']
            correlation = age_analysis['strongest_positive_correlation']['correlation']
            insights['key_findings'].append(f"Strong correlation with {strong_age_group} demographic (r={correlation:.2f})")
            insights['opportunities'].append(f"Target {strong_age_group} voters for campaign focus")
        
        # Analyze minority voting patterns
        minority_analysis = demographic_results.get('minority_patterns', {})
        if 'correlation_analysis' in minority_analysis:
            minority_corr = minority_analysis['correlation_analysis']
            if minority_corr.get('significance') == 'significant':
                correlation = minority_corr['correlation']
                if correlation > 0.2:
                    insights['opportunities'].append("Strong support among minority communities")
                elif correlation < -0.2:
                    insights['challenges'].append("Weak performance in minority-majority areas")
                    insights['recommendations'].append("Develop targeted minority outreach strategy")
        
        # Generic recommendations based on performance
        if lf_vote_share < 30:
            insights['recommendations'].extend([
                "Increase ground-level campaign presence",
                "Focus on booth-level voter mobilization",
                "Identify and address specific local issues"
            ])
        
        # Demographic opportunities
        demo_opportunities = demographic_results.get('demographic_opportunities', {})
        if demo_opportunities.get('high_potential_segments'):
            insights['opportunities'].append("Favorable demographic segments identified for targeted campaigning")
        
        return insights
    
    def run_quick_analysis(self, ac_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a quick analysis for testing purposes.
        
        Args:
            ac_name: Specific AC to analyze. If None, analyzes first AC found.
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Starting quick analysis")
        
        try:
            # Load data
            df = self.data_loader.load_excel_data()
            self.data_loader.validate_required_columns(df)
            
            # Get AC names and select one for analysis
            ac_names = self.data_loader.get_assembly_constituencies(df)
            
            if ac_name is None:
                ac_name = ac_names[0]  # Use first AC
            elif ac_name not in ac_names:
                raise ValueError(f"AC '{ac_name}' not found. Available ACs: {ac_names[:5]}...")
            
            self.logger.info(f"Running quick analysis for: {ac_name}")
            
            # Process single constituency
            result = self.process_single_constituency(df, ac_name)
            
            # Add summary for quick viewing
            result['quick_summary'] = {
                'ac_name': ac_name,
                'total_booths': result['basic_stats']['total_booths'],
                'lf_vote_share': result['electoral_analysis']['lf_performance'].get('lf_vote_share_overall', 0),
                'lf_strongholds': len(result['electoral_analysis']['lf_strongholds']),
                'top_3_insights': result['summary_insights']['key_findings'][:3]
            }
            
            self.logger.info("Quick analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in quick analysis: {str(e)}")
            raise


def main():
    """Main function for testing the pipeline."""
    try:
        # Initialize pipeline
        pipeline = ElectionAnalysisPipeline()
        
        # Run analysis specifically for Kasba constituency
        print("Running comprehensive analysis for Kasba constituency...")
        result = pipeline.run_quick_analysis(ac_name="Kasba")
        
        # Display detailed summary
        summary = result['quick_summary']
        print(f"\n{'='*60}")
        print(f"      KASBA CONSTITUENCY - COMPLETE ANALYSIS      ")
        print(f"{'='*60}")
        print(f"Assembly Constituency: {summary['ac_name']}")
        print(f"Total Booths: {summary['total_booths']}")
        print(f"LF Vote Share: {summary['lf_vote_share']:.1f}%")
        print(f"LF Strongholds: {summary['lf_strongholds']}")
        
        # Display electoral analysis details
        electoral = result['electoral_analysis']
        print(f"\n📊 ELECTORAL PERFORMANCE:")
        lf_perf = electoral['lf_performance']
        print(f"   • Booths where LF came 1st: {lf_perf.get('booths_lf_first', 0)}")
        print(f"   • Booths where LF came 2nd: {lf_perf.get('booths_lf_second', 0)}")
        print(f"   • Booths where LF came 3rd: {lf_perf.get('booths_lf_third', 0)}")
        
        # Display competitive landscape
        competitive = electoral['competitive_landscape']
        print(f"\n🏆 COMPETITIVE LANDSCAPE:")
        print(f"   • Most votes party: {competitive.get('most_votes_party', 'N/A')}")
        print(f"   • Average margin: {competitive.get('average_margin', 0):.1f}")
        print(f"   • Close contests (<5% margin): {competitive.get('close_contests', 0)}")
        
        # Display demographic insights
        demographic = result['demographic_analysis']
        print(f"\n👥 DEMOGRAPHIC ANALYSIS:")
        age_impact = demographic['age_group_impact']
        if 'strongest_positive_correlation' in age_impact:
            strong_demo = age_impact['strongest_positive_correlation']
            print(f"   • Strongest age group correlation: {strong_demo['age_group']} (r={strong_demo['correlation']:.3f})")
        
        gender_dynamics = demographic['gender_dynamics']
        if 'male_correlation' in gender_dynamics and 'female_correlation' in gender_dynamics:
            print(f"   • Male voter correlation: {gender_dynamics['male_correlation']:.3f}")
            print(f"   • Female voter correlation: {gender_dynamics['female_correlation']:.3f}")
        
        # Display key insights
        insights = result['summary_insights']
        print(f"\n💡 KEY FINDINGS:")
        for i, finding in enumerate(insights['key_findings'], 1):
            print(f"   {i}. {finding}")
        
        if insights['opportunities']:
            print(f"\n🎯 OPPORTUNITIES:")
            for i, opp in enumerate(insights['opportunities'], 1):
                print(f"   {i}. {opp}")
        
        if insights['challenges']:
            print(f"\n⚠️  CHALLENGES:")
            for i, challenge in enumerate(insights['challenges'], 1):
                print(f"   {i}. {challenge}")
        
        if insights['recommendations']:
            print(f"\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(insights['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n✅ Comprehensive analysis for Kasba completed successfully!")
        print(f"Full detailed results available in the returned dictionary.")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
