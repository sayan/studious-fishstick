"""Detailed analysis script for Kasba constituency with additional insights."""

import json
from main import ElectionAnalysisPipeline
import pandas as pd

def detailed_kasba_analysis():
    """Run detailed analysis and show additional insights for Kaliganj."""
    
    # Initialize pipeline and run analysis
    pipeline = ElectionAnalysisPipeline()
    result = pipeline.run_quick_analysis(ac_name="Kaliganj")
    
    print("="*80)
    print("               Kaliganj CONSTITUENCY - DETAILED ANALYSIS REPORT")
    print("="*80)
    
    # Basic Statistics
    basic = result['basic_stats']
    print(f"\n📈 BASIC STATISTICS:")
    print(f"   • Total Booths: {basic['total_booths']}")
    print(f"   • Total Voters (2024): {basic['total_voters_2024']:,}")
    print(f"   • Total Valid Votes: {basic['total_valid_votes']:,}")
    if 'turnout_percentage' in basic:
        print(f"   • Voter Turnout: {basic['turnout_percentage']:.2f}%")
    
    # Detailed Electoral Analysis
    electoral = result['electoral_analysis']
    lf_perf = electoral['lf_performance']
    
    print(f"\n🗳️  DETAILED ELECTORAL PERFORMANCE:")
    print(f"   • LF Vote Share: {lf_perf['lf_vote_share_overall']:.2f}%")
    print(f"   • LF Rank Distribution:")
    print(f"     - 1st Position: {lf_perf['booths_lf_first']} booths ({(lf_perf['booths_lf_first']/basic['total_booths']*100):.1f}%)")
    print(f"     - 2nd Position: {lf_perf['booths_lf_second']} booths ({(lf_perf['booths_lf_second']/basic['total_booths']*100):.1f}%)")
    print(f"     - 3rd Position: {lf_perf['booths_lf_third']} booths ({(lf_perf['booths_lf_third']/basic['total_booths']*100):.1f}%)")
    
    # Competitive Analysis
    competitive = electoral['competitive_landscape']
    print(f"\n⚔️  COMPETITIVE ANALYSIS:")
    print(f"   • Head-to-Head Performance:")
    
    if 'head_to_head' in competitive:
        head_to_head = competitive['head_to_head']
        
        # BJP vs TMC
        if 'BJP_vs_TMC' in head_to_head:
            bjp_tmc = head_to_head['BJP_vs_TMC']
            print(f"     - BJP vs TMC: BJP {bjp_tmc['BJP_wins']} wins, TMC {bjp_tmc['TMC_wins']} wins")
            print(f"       (TMC dominance: {bjp_tmc['TMC_win_percentage']:.1f}%)")
        
        # TMC vs LF  
        if 'TMC_vs_LF' in head_to_head:
            tmc_lf = head_to_head['TMC_vs_LF']
            print(f"     - TMC vs LF: TMC {tmc_lf['TMC_wins']} wins, LF {tmc_lf['LF_wins']} wins")
            print(f"       (TMC dominance: {tmc_lf['TMC_win_percentage']:.1f}%)")
        
        # BJP vs LF
        if 'BJP_vs_LF' in head_to_head:
            bjp_lf = head_to_head['BJP_vs_LF']
            print(f"     - BJP vs LF: BJP {bjp_lf['BJP_wins']} wins, LF {bjp_lf['LF_wins']} wins")
            print(f"       (BJP dominance: {bjp_lf['BJP_win_percentage']:.1f}%)")
    
    if 'competition_metrics' in competitive:
        metrics = competitive['competition_metrics']
        print(f"   • Overall Competition:")
        print(f"     - Effective Number of Parties: {metrics.get('mean_effective_num_parties', 0):.2f}")
        print(f"     - Highly Competitive Booths: {metrics.get('highly_competitive_booths', 0)}")
    
    # LF Strongholds
    strongholds = electoral['lf_strongholds']
    print(f"\n🏰 TOP LF STRONGHOLDS:")
    for i, booth in enumerate(strongholds[:5], 1):
        # Try different booth identification columns
        booth_id = booth.get('BoothNo') or booth.get('Booth_3Digit') or booth.get('PSNo') or booth.get('SlNo') or 'Unknown'
        booth_name = booth.get('BOOTHNAME', 'Unknown Booth')[:50]  # Truncate long names
        print(f"   {i}. Booth {booth_id}: {booth.get('LF', 0)} votes ({booth.get('lf_vote_share', 0):.1f}%)")
        print(f"      Name: {booth_name}")
    
    # Margin Analysis (show close contests)
    margins = electoral['margin_analysis']
    close_margins = [m for m in margins if m.get('margin_percentage', 100) <= 10]
    print(f"\n📊 CLOSE CONTESTS (≤10% margin):")
    print(f"   • Total close contests: {len(close_margins)}")
    if close_margins:
        print(f"   • Top 3 closest:")
        for i, margin in enumerate(sorted(close_margins, key=lambda x: x.get('margin_percentage', 100))[:3], 1):
            winner = margin.get('winner', 'N/A')
            margin_pct = margin.get('margin_percentage', 0)
            # Try different booth identification columns
            booth_id = margin.get('BoothNo') or margin.get('Booth_3Digit') or margin.get('PSNo') or margin.get('SlNo') or 'Unknown'
            booth_name = margin.get('BOOTHNAME', 'Unknown')[:30]  # Truncate long names
            print(f"     {i}. Booth {booth_id}: {winner} won by {margin_pct:.2f}%")
            print(f"        ({booth_name})")
    
    # Three-way Competition
    three_way = electoral['three_way_competition']
    print(f"\n🎯 THREE-WAY COMPETITION:")
    print(f"   • Highly competitive booths (all parties >20%): {three_way.get('highly_competitive_booths', 0)}")
    print(f"   • LF competitive booths (LF >15%): {three_way.get('lf_competitive_booths', 0)}")
    
    # Demographic Analysis
    demographic = result['demographic_analysis']
    
    print(f"\n👥 DEMOGRAPHIC INSIGHTS:")
    
    # Age Group Analysis
    age_analysis = demographic['age_group_impact']
    print(f"   • Age Group Correlations:")
    if 'correlations' in age_analysis:
        for age_group in ['P_20', 'P_20_30', 'P_30_40', 'P_40_50', 'P_50_60', 'Above_60']:
            if age_group in age_analysis['correlations']:
                corr_data = age_analysis['correlations'][age_group]
                corr = corr_data['correlation']
                significance = corr_data['significance']
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
                sig_indicator = "✓" if significance == "significant" else "✗"
                print(f"     - {age_group}: {corr:.3f} ({strength} {direction}) {sig_indicator}")
    
    # Gender Analysis
    gender = demographic['gender_dynamics']
    print(f"   • Gender Correlations:")
    if 'correlations' in gender:
        for gender_type in ['MALE', 'FEMALE']:
            if gender_type in gender['correlations']:
                corr_data = gender['correlations'][gender_type]
                corr = corr_data['correlation']
                significance = corr_data['significance']
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
                sig_indicator = "✓" if significance == "significant" else "✗"
                print(f"     - {gender_type.title()} voters: {corr:.3f} ({strength} {direction}) {sig_indicator}")
    
    # Minority Analysis
    minority = demographic['minority_patterns']
    print(f"   • Minority Voting Patterns:")
    if 'correlation_analysis' in minority:
        min_analysis = minority['correlation_analysis']
        correlation = min_analysis.get('correlation', 0)
        significance = min_analysis.get('significance', 'not significant')
        print(f"     - Minority correlation: {correlation:.3f} ({significance})")
        print(f"     - Direction: {'Positive' if correlation > 0 else 'Negative'} correlation with LF votes")
    
    # Summary Assessment
    print(f"\n🎖️  OVERALL ASSESSMENT:")
    lf_vote_share = lf_perf['lf_vote_share_overall']
    
    if lf_vote_share < 15:
        performance = "🔴 CHALLENGING"
        strategy = "Requires immediate strategic intervention and grassroots mobilization"
    elif lf_vote_share < 25:
        performance = "🟡 MODERATE"
        strategy = "Focus on consolidating base and expanding reach"
    elif lf_vote_share < 35:
        performance = "🟢 COMPETITIVE"
        strategy = "Build on existing strengths and target swing voters"
    else:
        performance = "🟢 STRONG"
        strategy = "Maintain momentum and focus on turnout"
    
    print(f"   • Performance Level: {performance}")
    print(f"   • Strategic Priority: {strategy}")
    print(f"   • Key Demographic: Above 60 age group (strongest correlation)")
    print(f"   • Critical Need: Minority community engagement")
    
    print(f"\n📋 ACTIONABLE RECOMMENDATIONS:")
    print(f"   1. 🎯 Target the {lf_perf['booths_lf_second']} booths where LF came 2nd - high conversion potential")
    print(f"   2. 👴 Leverage strong correlation with 60+ voters through senior citizen programs")
    print(f"   3. 🤝 Develop comprehensive minority outreach strategy")
    print(f"   4. 📍 Focus intensive campaigning on the {len(close_margins)} close contest booths")
    print(f"   5. 🗣️  Increase ground presence in the {lf_perf['booths_lf_third']} booths where LF came 3rd")
    
    return result

if __name__ == "__main__":
    detailed_kasba_analysis()
