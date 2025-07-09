"""Electoral performance analysis module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

from ..utils.logger import setup_logger


class ElectoralAnalyzer:
    """Analyzes electoral performance patterns for assembly constituencies."""
    
    def __init__(self, ac_data: pd.DataFrame, ac_name: str):
        """
        Initialize electoral analyzer for a specific assembly constituency.
        
        Args:
            ac_data: DataFrame containing data for the specific AC
            ac_name: Name of the assembly constituency
        """
        self.ac_data = ac_data.copy()
        self.ac_name = ac_name
        self.logger = setup_logger(__name__)
        
        # Log DataFrame info for debugging
        self.logger.info(f"Initializing ElectoralAnalyzer for {ac_name}")
        self.logger.info(f"DataFrame shape: {self.ac_data.shape}")
        self.logger.info(f"DataFrame index: {self.ac_data.index[:5].tolist()}...")  # Show first 5 indices
        self.logger.info(f"DataFrame columns: {self.ac_data.columns.tolist()}")
        self.logger.info(f"DataFrame head:\n{self.ac_data.head()}")
        
        # Ensure we have required columns
        self.vote_columns = ['BJP', 'TMC', 'LF']
        self.available_vote_columns = [col for col in self.vote_columns if col in self.ac_data.columns]
        
        if len(self.available_vote_columns) < 2:
            raise ValueError(f"Insufficient vote columns available. Found: {self.available_vote_columns}")
    
    def calculate_lf_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive LF performance metrics.
        
        Returns:
            Dictionary containing LF performance statistics
        """
        if 'LF' not in self.ac_data.columns:
            raise ValueError("LF column not found in data")
        
        lf_votes = self.ac_data['LF']
        
        # Basic statistics
        performance_metrics = {
            'total_lf_votes': int(lf_votes.sum()),
            'mean_lf_votes_per_booth': float(lf_votes.mean()),
            'median_lf_votes_per_booth': float(lf_votes.median()),
            'std_lf_votes_per_booth': float(lf_votes.std()),
            'min_lf_votes': int(lf_votes.min()),
            'max_lf_votes': int(lf_votes.max()),
            'total_booths': len(self.ac_data)
        }
        
        # Calculate LF vote share if ValidVotes is available
        if 'ValidVotes' in self.ac_data.columns:
            total_valid_votes = self.ac_data['ValidVotes'].sum()
            performance_metrics['lf_vote_share_overall'] = float((lf_votes.sum() / total_valid_votes) * 100)
            
            # Booth-wise vote share statistics
            booth_vote_shares = (lf_votes / self.ac_data['ValidVotes']) * 100
            performance_metrics.update({
                'mean_lf_vote_share_per_booth': float(booth_vote_shares.mean()),
                'median_lf_vote_share_per_booth': float(booth_vote_shares.median()),
                'std_lf_vote_share_per_booth': float(booth_vote_shares.std()),
                'min_lf_vote_share': float(booth_vote_shares.min()),
                'max_lf_vote_share': float(booth_vote_shares.max())
            })
        
        # LF ranking analysis (position among parties)
        if len(self.available_vote_columns) >= 2:
            rankings = self._calculate_party_rankings()
            lf_positions = rankings['LF'] if 'LF' in rankings else []
            
            if lf_positions:
                performance_metrics.update({
                    'booths_lf_first': int(sum(1 for pos in lf_positions if pos == 1)),
                    'booths_lf_second': int(sum(1 for pos in lf_positions if pos == 2)),
                    'booths_lf_third': int(sum(1 for pos in lf_positions if pos == 3)),
                    'avg_lf_position': float(np.mean(lf_positions))
                })
        
        self.logger.info(f"LF performance metrics calculated for {self.ac_name}")
        return performance_metrics
    
    def analyze_competitive_landscape(self) -> Dict[str, Any]:
        """
        Analyze competitive dynamics between parties.
        
        Returns:
            Dictionary containing competitive analysis results
        """
        competitive_analysis = {}
        
        if len(self.available_vote_columns) < 2:
            competitive_analysis['error'] = "Insufficient parties for competitive analysis"
            return competitive_analysis
        
        # Head-to-head comparisons
        head_to_head = {}
        
        for i, party1 in enumerate(self.available_vote_columns):
            for party2 in self.available_vote_columns[i+1:]:
                comparison_key = f"{party1}_vs_{party2}"
                
                party1_votes = self.ac_data[party1]
                party2_votes = self.ac_data[party2]
                
                # Count wins
                party1_wins = (party1_votes > party2_votes).sum()
                party2_wins = (party2_votes > party1_votes).sum()
                ties = (party1_votes == party2_votes).sum()
                
                # Average margin
                margins = abs(party1_votes - party2_votes)
                avg_margin = margins.mean()
                
                head_to_head[comparison_key] = {
                    f'{party1}_wins': int(party1_wins),
                    f'{party2}_wins': int(party2_wins),
                    'ties': int(ties),
                    'avg_margin': float(avg_margin),
                    f'{party1}_win_percentage': float((party1_wins / len(self.ac_data)) * 100),
                    f'{party2}_win_percentage': float((party2_wins / len(self.ac_data)) * 100)
                }
        
        competitive_analysis['head_to_head'] = head_to_head
        
        # Overall competitive intensity
        if 'ValidVotes' in self.ac_data.columns:
            vote_shares = pd.DataFrame()
            for party in self.available_vote_columns:
                vote_shares[party] = (self.ac_data[party] / self.ac_data['ValidVotes']) * 100
            
            # Calculate Effective Number of Parties (ENP)
            enp_values = []
            for idx, row in vote_shares.iterrows():
                vote_share_squares = (row / 100) ** 2
                enp = 1 / vote_share_squares.sum() if vote_share_squares.sum() > 0 else 1
                enp_values.append(enp)
            
            competitive_analysis['competition_metrics'] = {
                'mean_effective_num_parties': float(np.mean(enp_values)),
                'median_effective_num_parties': float(np.median(enp_values)),
                'highly_competitive_booths': int(sum(1 for enp in enp_values if enp > 2.5))
            }
        
        self.logger.info(f"Competitive landscape analysis completed for {self.ac_name}")
        return competitive_analysis
    
    def identify_lf_strongholds(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify LF stronghold booths based on vote share.
        
        Args:
            top_n: Number of top booths to return
            
        Returns:
            DataFrame containing top LF performing booths
        """
        if 'LF' not in self.ac_data.columns:
            raise ValueError("LF column not found in data")
        
        strongholds_data = self.ac_data.copy()
        
        # Calculate LF vote share
        if 'ValidVotes' in strongholds_data.columns:
            strongholds_data['lf_vote_share'] = (strongholds_data['LF'] / strongholds_data['ValidVotes']) * 100
            sort_column = 'lf_vote_share'
        else:
            sort_column = 'LF'
        
        # Add LF ranking
        if len(self.available_vote_columns) >= 2:
            rankings = self._calculate_party_rankings()
            if 'LF' in rankings:
                strongholds_data['lf_rank'] = rankings['LF']
        
        # Sort by LF performance and get top booths
        strongholds = strongholds_data.nlargest(top_n, sort_column)
        
        # Select relevant columns for output
        output_columns = ['BOOTHNAME', 'LF']
        
        # Add booth identification columns if available
        booth_id_columns = ['BoothNo', 'Booth_3Digit', 'PSNo', 'SlNo']
        for col in booth_id_columns:
            if col in strongholds.columns and col not in output_columns:
                output_columns.append(col)
        
        if 'ValidVotes' in strongholds.columns:
            output_columns.extend(['ValidVotes', 'lf_vote_share'])
        if 'lf_rank' in strongholds.columns:
            output_columns.append('lf_rank')
        
        # Add other party votes for context
        for party in self.available_vote_columns:
            if party != 'LF' and party in strongholds.columns:
                output_columns.append(party)
        
        strongholds_output = strongholds[output_columns].round(2)
        
        self.logger.info(f"Identified top {len(strongholds_output)} LF strongholds in {self.ac_name}")
        return strongholds_output
    
    def calculate_margin_analysis(self) -> pd.DataFrame:
        """
        Calculate victory/defeat margins for each booth.
        
        Returns:
            DataFrame containing margin analysis for each booth
        """
        if len(self.available_vote_columns) < 2:
            raise ValueError("Need at least 2 parties for margin analysis")
        
        margin_data = self.ac_data.copy()
        
        # Include booth identification columns
        base_columns = ['BOOTHNAME'] + self.available_vote_columns
        booth_id_columns = ['BoothNo', 'Booth_3Digit', 'PSNo', 'SlNo']
        
        # Add available booth ID columns
        for col in booth_id_columns:
            if col in self.ac_data.columns and col not in base_columns:
                base_columns.append(col)
        
        # Add ValidVotes if available
        if 'ValidVotes' in self.ac_data.columns:
            base_columns.append('ValidVotes')
            
        margin_data = margin_data[base_columns]
        
        # For each booth, find winner and margin
        margins = []
        
        for idx, row in margin_data.iterrows():
            vote_counts = {party: row[party] for party in self.available_vote_columns}
            sorted_parties = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
            
            winner = sorted_parties[0][0]
            winner_votes = sorted_parties[0][1]
            
            if len(sorted_parties) > 1:
                runner_up = sorted_parties[1][0]
                runner_up_votes = sorted_parties[1][1]
                margin = winner_votes - runner_up_votes
                
                # Calculate margin percentage
                if 'ValidVotes' in self.ac_data.columns:
                    total_votes = row.get('ValidVotes', sum(vote_counts.values()))
                    margin_percentage = (margin / total_votes) * 100 if total_votes > 0 else 0
                else:
                    margin_percentage = None
            else:
                runner_up = None
                runner_up_votes = 0
                margin = winner_votes
                margin_percentage = None
            
            margin_dict = {
                'BOOTHNAME': row['BOOTHNAME'],
                'winner': winner,
                'winner_votes': winner_votes,
                'runner_up': runner_up,
                'runner_up_votes': runner_up_votes,
                'margin': margin,
                'margin_percentage': margin_percentage,
                'lf_position': self._get_party_position('LF', vote_counts)
            }
            
            # Add booth identification columns if available
            booth_id_columns = ['BoothNo', 'Booth_3Digit', 'PSNo', 'SlNo']
            for col in booth_id_columns:
                if col in row.index:
                    margin_dict[col] = row[col]
            
            margins.append(margin_dict)
        
        margin_df = pd.DataFrame(margins)
        
        self.logger.info(f"Margin analysis completed for {len(margin_df)} booths in {self.ac_name}")
        return margin_df
    
    def assess_three_way_competition(self) -> Dict[str, Any]:
        """
        Assess three-way competition dynamics if all three major parties are present.
        
        Returns:
            Dictionary containing three-way competition analysis
        """
        if len(self.available_vote_columns) < 3:
            return {'error': 'Need at least 3 parties for three-way analysis'}
        
        three_way_analysis = {}
        
        # Calculate vote share distribution
        if 'ValidVotes' in self.ac_data.columns:
            vote_shares = {}
            for party in self.available_vote_columns:
                vote_shares[party] = (self.ac_data[party] / self.ac_data['ValidVotes']) * 100
            
            # Find booths where all parties are competitive (each has >15% vote share)
            competitive_threshold = 15
            competitive_booths = 0
            
            for idx in self.ac_data.index:
                party_shares = [vote_shares[party].loc[idx] for party in self.available_vote_columns]
                if all(share >= competitive_threshold for share in party_shares):
                    competitive_booths += 1
            
            three_way_analysis['highly_competitive_booths'] = competitive_booths
            three_way_analysis['competitive_booth_percentage'] = (competitive_booths / len(self.ac_data)) * 100
            
            # Calculate fragmentation index (1 - Herfindahl index)
            fragmentation_scores = []
            for idx in self.ac_data.index:
                shares = [vote_shares[party].loc[idx] / 100 for party in self.available_vote_columns]
                herfindahl = sum(share ** 2 for share in shares)
                fragmentation = 1 - herfindahl
                fragmentation_scores.append(fragmentation)
            
            three_way_analysis['mean_fragmentation'] = float(np.mean(fragmentation_scores))
            three_way_analysis['median_fragmentation'] = float(np.median(fragmentation_scores))
        
        # Analyze vote patterns
        rankings = self._calculate_party_rankings()
        
        if len(rankings) >= 3:
            # Count different winning patterns
            winning_patterns = {}
            for party in self.available_vote_columns:
                if party in rankings:
                    wins = sum(1 for rank in rankings[party] if rank == 1)
                    winning_patterns[f'{party}_wins'] = wins
            
            three_way_analysis['winning_patterns'] = winning_patterns
        
        self.logger.info(f"Three-way competition analysis completed for {self.ac_name}")
        return three_way_analysis
    
    def _calculate_party_rankings(self) -> Dict[str, List[int]]:
        """
        Calculate rankings for each party in each booth.
        
        Returns:
            Dictionary mapping party names to lists of their rankings
        """
        rankings = {party: [] for party in self.available_vote_columns}
        
        for idx, row in self.ac_data.iterrows():
            vote_counts = [(party, row[party]) for party in self.available_vote_columns]
            vote_counts.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (party, votes) in enumerate(vote_counts, 1):
                rankings[party].append(rank)
        
        return rankings
    
    def _get_party_position(self, party: str, vote_counts: Dict[str, int]) -> int:
        """
        Get the position (rank) of a party in a booth.
        
        Args:
            party: Party name
            vote_counts: Dictionary of party vote counts
            
        Returns:
            Position of the party (1 = first, 2 = second, etc.)
        """
        if party not in vote_counts:
            return len(vote_counts) + 1
        
        sorted_parties = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        for position, (party_name, votes) in enumerate(sorted_parties, 1):
            if party_name == party:
                return position
        
        return len(vote_counts) + 1
