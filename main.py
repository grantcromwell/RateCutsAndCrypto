#!/usr/bin/env python3
"""
Random Matrix Theory Analysis for Ethereum and Interest Rate Correlations

This script implements a comprehensive RMT analysis to identify correlations
between Ethereum price movements and US Federal Reserve interest rate decisions.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta

# Import our modules
from data.fetch_data import DataFetcher
from rmt.core import RandomMatrixTheory
from rmt.correlation import CorrelationAnalyzer
from rmt.signal_generator import SignalGenerator
from visualization.plots import RMTPlots

class RMTAnalysis:
    """Main RMT analysis class"""

    def __init__(self):
        self.assets = ['ETH-USD', 'SI=F', 'NET', '^IRX']
        self.fetcher = DataFetcher()
        self.correlation_analyzer = CorrelationAnalyzer(self.assets)
        self.signal_generator = SignalGenerator(self.assets)
        self.plots = RMTPlots(self.assets)

    def run_analysis(self, start_date=None, end_date=None, generate_plots=True):
        """
        Run the complete RMT analysis pipeline

        Args:
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            generate_plots: Whether to generate visualizations
        """
        print("="*60)
        print("Random Matrix Theory Analysis: Ethereum & Interest Rates")
        print("="*60)

        # 1. Fetch and process data
        print("\n1. Fetching and processing data...")
        returns_data = self.fetcher.load_processed_data()

        if start_date or end_date:
            # Filter data by date range
            start_date = pd.to_datetime(start_date) if start_date else returns_data.index.min()
            end_date = pd.to_datetime(end_date) if end_date else returns_data.index.max()
            returns_data = returns_data.loc[start_date:end_date]

        print(f"Data range: {returns_data.index.min().date()} to {returns_data.index.max().date()}")
        print(f"Number of observations: {len(returns_data)}")

        # 2. Get FOMC data
        print("\n2. Loading FOMC meeting data...")
        regime_info = self.fetcher.get_fomc_data()

        # 3. Regime-based correlation analysis
        print("\n3. Performing regime-based correlation analysis...")
        regime_results = self.correlation_analyzer.regime_correlation_analysis(
            returns_data, regime_info
        )

        # Print regime summary
        self.print_regime_summary(regime_results)

        # 4. Rolling correlation analysis
        print("\n4. Computing rolling correlations...")
        rolling_corr = self.correlation_analyzer.rolling_correlation_analysis(returns_data)

        # 5. Generate trading signals
        print("\n5. Generating trading signals...")
        signals_dict = self.signal_generator.generate_signals(
            regime_results, returns_data, regime_info
        )
        # Ensure signals_dict has required structure
        if 'regime_specific_signals' not in signals_dict:
            signals_dict['regime_specific_signals'] = {}
        signals_df = signals_dict['combined_signals']

        # 6. Performance analysis
        performance = signals_dict['performance']

        # 7. Generate visualizations
        plot_paths = []
        if generate_plots:
            print("\n6. Generating visualizations...")
            plot_paths = self.plots.create_all_plots(
                regime_results, rolling_corr, signals_df, performance, regime_info, returns_data
            )
            print(f"Plots saved to: {self.plots.save_dir}")

        # 8. Summary report
        print("\n7. Generating summary report...")
        report = self.generate_summary_report(
            returns_data, regime_results, rolling_corr, signals_dict, plot_paths
        )

        # Print key findings
        self.print_key_findings(report)

        # Save results
        self.save_results(regime_results, rolling_corr, signals_df, report)

        return {
            'returns_data': returns_data,
            'regime_results': regime_results,
            'rolling_corr': rolling_corr,
            'signals': signals_df,
            'performance': performance,
            'report': report,
            'plot_paths': plot_paths
        }

    def print_regime_summary(self, regime_results):
        """Print summary of regime analysis"""
        print("\nRegime Analysis Summary:")
        print("-" * 40)

        for regime, result in regime_results.items():
            if regime != 'comparison':
                corr_matrix = result['correlation_matrix']
                eth_rate_corr = corr_matrix[0, -1]  # ETH vs Rate
                n_obs = result['n_observations']
                rmt = result['rmt_analysis']['spectrum_test']

                print(f"\n{regime.upper()} Regime:")
                print(f"  - ETH-Rate correlation: {eth_rate_corr:.3f}")
                print(f"  - Observations: {n_obs}")
                print(f"  - Signal eigenvalues: {rmt['signal_eigenvalues']}")
                print(f"  - Largest eigenvalue: {rmt['lambda_max_empirical']:.3f}")

    def print_key_findings(self, report):
        """Print key findings from the analysis"""
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)

        # Current regime and signal
        if 'current_regime' in report:
            print(f"\nCurrent regime: {report['current_regime'].upper()}")
            if 'regime_signals' in report and report['current_regime'] in report['regime_signals']:
                signal = report['regime_signals'][report['current_regime']]['signal']
                print(f"Recommended action: {signal}")
            else:
                print(f"Recommended action: HOLD (regime data not available)")

        # ETH-rate correlation trends
        if 'correlation_trends' in report:
            trends = report['correlation_trends']
            print(f"\nETH-Rate correlation trends:")
            for regime, corr in trends.items():
                print(f"  - {regime}: {corr:.3f}")

        # Best performing signal
        if 'best_signal' in report:
            best = report['best_signal']
            print(f"\nBest performing signal: {best['signal']}")
            print(f"  - Annualized return: {best['annualized_return']:.2%}")
            print(f"  - Sharpe ratio: {best['sharpe_ratio']:.2f}")

    def generate_summary_report(self, returns_data, regime_results, rolling_corr,
                              signals_dict, plot_paths):
        """Generate a summary report"""
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_range': {
                'start': returns_data.index.min().strftime('%Y-%m-%d'),
                'end': returns_data.index.max().strftime('%Y-%m-%d'),
                'n_observations': len(returns_data)
            },
            'current_regime': signals_dict.get('current_regime', 'unknown'),
            'regime_signals': {},
            'correlation_trends': {},
            'signal_performance': {},
            'best_signal': None,
            'plot_paths': plot_paths
        }

        # Extract regime signals
        if 'regime_specific_signals' in signals_dict:
            for regime, signal_data in signals_dict['regime_specific_signals'].items():
                report['regime_signals'][regime] = {
                    'signal': signal_data['signal'],
                    'correlation': signal_data['correlation'],
                    'confidence': signal_data['confidence'],
                    'reasoning': signal_data['reasoning']
                }

        # Extract correlation trends
        if 'comparison' in regime_results:
            if 'eth_rate_correlations' in regime_results['comparison']:
                correlations = regime_results['comparison']['eth_rate_correlations']
                report['correlation_trends'] = correlations

        # Extract performance metrics
        if 'performance' in signals_dict:
            performance = signals_dict['performance']
            for signal, metrics in performance.items():
                report['signal_performance'][signal] = {
                    'annualized_return': metrics['annualized_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'volatility': metrics['volatility'],
                    'n_signals': metrics['n_signals']
                }

            # Find best performing signal
            if 'buy_and_hold' in performance:
                best_signal = 'buy_and_hold'
                best_sharpe = performance['buy_and_hold']['sharpe_ratio']

                for signal, metrics in performance.items():
                    if signal != 'buy_and_hold' and metrics['sharpe_ratio'] > best_sharpe:
                        best_signal = signal
                        best_sharpe = metrics['sharpe_ratio']

                if best_signal:
                    report['best_signal'] = {
                        'signal': best_signal,
                        'annualized_return': performance[best_signal]['annualized_return'],
                        'sharpe_ratio': performance[best_signal]['sharpe_ratio'],
                        'volatility': performance[best_signal]['volatility'],
                        'n_signals': performance[best_signal]['n_signals']
                    }

        return report

    def save_results(self, regime_results, rolling_corr, signals_df, report):
        """Save analysis results to files"""
        # Create results directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        # Save summary report
        report_file = results_dir / 'analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {report_file}")

        # Save signals data
        signals_file = results_dir / 'trading_signals.csv'
        signals_df.to_csv(signals_file)
        print(f"Trading signals saved to: {signals_file}")

        # Save rolling correlations
        corr_file = results_dir / 'rolling_correlations.csv'
        rolling_corr.to_csv(corr_file)
        print(f"Rolling correlations saved to: {corr_file}")

        # Save detailed regime results
        regime_file = results_dir / 'regime_results.json'
        # Convert numpy arrays to lists for JSON serialization
        serializable_regime = {}
        for regime, data in regime_results.items():
            if regime != 'comparison':
                serializable_regime[regime] = {
                    'n_observations': data['n_observations'],
                    'correlation_matrix': data['correlation_matrix'].tolist(),
                    'rmt_analysis': {
                        'eigenvalues': data['rmt_analysis']['eigenvalues'].tolist(),
                        'spectrum_test': data['rmt_analysis']['spectrum_test'],
                        'insights': data['rmt_analysis']['insights'],
                        'deflation_analysis': data['rmt_analysis']['deflation_analysis']
                    }
                }

        with open(regime_file, 'w') as f:
            json.dump(serializable_regime, f, indent=2, default=str)
        print(f"Regime results saved to: {regime_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Random Matrix Theory Analysis for Ethereum and Interest Rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis from 2020 to present
  python main.py --analyze

  # Run analysis with custom date range
  python main.py --analyze --start-date 2021-01-01 --end-date 2023-12-31

  # Fetch data only
  python main.py --fetch-data

  # Generate plots only (requires existing results)
  python main.py --plots-only

  # Validate data integrity
  python main.py --validate-data
        """
    )

    parser.add_argument('--analyze', action='store_true',
                       help='Run complete RMT analysis')
    parser.add_argument('--fetch-data', action='store_true',
                       help='Fetch financial data only')
    parser.add_argument('--plots-only', action='store_true',
                       help='Generate plots from existing results')
    parser.add_argument('--validate-data', action='store_true',
                       help='Validate data integrity')
    parser.add_argument('--start-date', type=str,
                       help='Analysis start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='Analysis end date (YYYY-MM-DD)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation during analysis')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()

    # Initialize analysis
    analysis = RMTAnalysis()

    # Handle different commands
    if args.fetch_data:
        print("Fetching financial data...")
        returns_data = analysis.fetcher.fetch_data()
        if returns_data is not None:
            print("Data fetch completed successfully!")
        else:
            print("Error fetching data!")
            sys.exit(1)

    elif args.validate_data:
        print("Validating data integrity...")
        returns_data = analysis.fetcher.load_processed_data()
        regime_info = analysis.fetcher.get_fomc_data()

        # Basic validation
        print(f"Data shape: {returns_data.shape}")
        print(f"Missing values: {returns_data.isnull().sum().sum()}")
        print(f"FOMC meetings: {len(regime_info)}")
        print(f"Date range: {returns_data.index.min()} to {returns_data.index.max()}")

        # Check for alignment
        aligned_dates = set(returns_data.index) & set(regime_info.index)
        print(f"Aligned FOMC dates: {len(aligned_dates)}")

    elif args.plots_only:
        # Check if results exist
        results_dir = Path('results')
        if not results_dir.exists():
            print("No existing results found. Run analysis first.")
            sys.exit(1)

        print("Generating plots from existing results...")
        # Load saved results and regenerate plots
        # This would require more complex implementation to load and reconstruct
        print("Plot generation from existing results not yet implemented.")
        sys.exit(1)

    elif args.analyze:
        # Run complete analysis
        results = analysis.run_analysis(
            start_date=args.start_date,
            end_date=args.end_date,
            generate_plots=not args.no_plots
        )

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {Path('results')}")
        if not args.no_plots:
            print(f"Plots saved to: {Path('plots')}")

    else:
        # Default: show help
        parser.print_help()

if __name__ == "__main__":
    main()