import numpy as np
import pandas as pd
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class CorrelationAnalyzer:
    """Correlation matrix analysis with regime-specific insights"""

    def __init__(self, assets: List[str]):
        """
        Initialize correlation analyzer

        Args:
            assets: List of asset names/tickers
        """
        self.assets = assets
        self.n_assets = len(assets)

    def compute_correlation_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """
        Compute correlation matrix from returns data

        Args:
            returns_data: DataFrame of asset returns (assets x time)

        Returns:
            Correlation matrix
        """
        # Drop any rows with NaN (should be handled in preprocessing)
        clean_returns = returns_data.dropna()

        # Compute correlation matrix
        correlation_matrix = clean_returns.corr().values

        return correlation_matrix

    def regime_correlation_analysis(self, returns_data: pd.DataFrame,
                                  regime_info: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations by interest rate regime

        Args:
            returns_data: Asset returns data
            regime_info: DataFrame with regime information

        Returns:
            Dictionary with regime-specific correlation results
        """
        results = {}

        # Get unique regimes
        regimes = ['hold', 'hike', 'cut']

        for regime in regimes:
            print(f"Analyzing {regime} regime...")

            # Get data for this regime
            regime_returns, regime_dates = self.get_regime_data(
                returns_data, regime_info, regime
            )

            if len(regime_returns) > 0:
                # Compute correlation matrix
                corr_matrix = self.compute_correlation_matrix(regime_returns)

                # Perform RMT analysis
                rmt_results = self.analyze_with_rmt(corr_matrix, regime)

                results[regime] = {
                    'correlation_matrix': corr_matrix,
                    'rmt_analysis': rmt_results,
                    'n_observations': len(regime_returns)
                }

        # Compare regimes
        results['comparison'] = self.compare_regime_correlations(results)

        return results

    def get_regime_data(self, returns_data: pd.DataFrame,
                       regime_info: pd.DataFrame, regime: str) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """
        Extract data for specific regime

        Args:
            returns_data: Asset returns data
            regime_info: DataFrame with regime information
            regime: Regime type ('hold', 'hike', 'cut')

        Returns:
            (regime_returns, regime_dates)
        """
        # Find regime periods
        regime_periods = regime_info[regime_info['decision'] == regime]

        if len(regime_periods) == 0:
            return pd.DataFrame(), pd.DatetimeIndex([])

        # Get all dates in regime periods
        regime_dates = []
        for idx, row in regime_periods.iterrows():
            start_date = max(row['regime_start'], returns_data.index[0])
            end_date = min(row['regime_end'], returns_data.index[-1])

            if start_date <= end_date:
                regime_dates.extend(pd.date_range(start_date, end_date))

        regime_dates = pd.DatetimeIndex(sorted(set(regime_dates)))

        # Get returns for these dates (handle missing dates gracefully)
        try:
            regime_returns = returns_data.loc[regime_dates]
        except KeyError:
            # Filter to only dates that exist in returns_data
            valid_dates = [d for d in regime_dates if d in returns_data.index]
            regime_returns = returns_data.loc[valid_dates] if valid_dates else pd.DataFrame()

        return regime_returns, regime_dates

    def analyze_with_rmt(self, correlation_matrix: np.ndarray, regime: str) -> Dict[str, Any]:
        """
        Perform RMT analysis on correlation matrix

        Args:
            correlation_matrix: Correlation matrix
            regime: Regime name

        Returns:
            RMT analysis results
        """
        from .core import RandomMatrixTheory

        # Number of assets and time points (approximate)
        N = correlation_matrix.shape[0]
        # Estimate T based on typical data length per regime
        T = 60  # Assume 60 days per regime for estimation

        rmt = RandomMatrixTheory(N, T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = eigh(correlation_matrix)

        # Sort eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Perform eigenvalue spectrum test
        spectrum_test = rmt.eigenvalue_spectrum_test(eigenvalues)

        # Get additional insights
        insights = rmt.get_eigenvalue_insights(eigenvalues, eigenvectors)
        deflation = rmt.deflation_analysis(correlation_matrix)

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'spectrum_test': spectrum_test,
            'insights': insights,
            'deflation_analysis': deflation,
            'regime': regime
        }

    def compare_regime_correlations(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare correlations across regimes

        Args:
            regime_results: Results from regime_correlation_analysis

        Returns:
            Comparison results
        """
        comparison = {}

        # Extract correlation matrices
        hold_corr = regime_results.get('hold', {}).get('correlation_matrix', None)
        hike_corr = regime_results.get('hike', {}).get('correlation_matrix', None)
        cut_corr = regime_results.get('cut', {}).get('correlation_matrix', None)

        # Compare ETH rate correlations
        if all([hold_corr is not None, hike_corr is not None, cut_corr is not None]):
            # ETH is first asset, rate is last asset (^IRX)
            eth_idx = 0
            rate_idx = -1

            hold_eth_rate = hold_corr[eth_idx, rate_idx]
            hike_eth_rate = hike_corr[eth_idx, rate_idx]
            cut_eth_rate = cut_corr[eth_idx, rate_idx]

            comparison['eth_rate_correlations'] = {
                'hold': hold_eth_rate,
                'hike': hike_eth_rate,
                'cut': cut_eth_rate
            }

            # Compare correlations
            comparison['eth_rate_max_diff'] = max(
                hold_eth_rate, hike_eth_rate, cut_eth_rate
            ) - min(
                hold_eth_rate, hike_eth_rate, cut_eth_rate
            )

        # Compare eigenvalue spectra
        comparison['eigenvalue_comparison'] = {}
        for regime, results in regime_results.items():
            if regime != 'comparison' and 'rmt_analysis' in results:
                comparison['eigenvalue_comparison'][regime] = {
                    'largest_eigenvalue': results['rmt_analysis']['spectrum_test']['lambda_max_empirical'],
                    'signal_ratio': results['rmt_analysis']['spectrum_test']['signal_ratio'],
                    'n_observations': results['n_observations']
                }

        return comparison

    def rolling_correlation_analysis(self, returns_data: pd.DataFrame,
                                   window: int = 60) -> pd.DataFrame:
        """
        Compute rolling correlation between ETH and interest rates

        Args:
            returns_data: Asset returns data
            window: Rolling window size in days

        Returns:
            DataFrame with rolling correlations
        """
        # Extract ETH and rate returns
        eth_returns = returns_data.iloc[:, 0]  # ETH is first asset
        rate_returns = returns_data.iloc[:, -1]  # Rate is last asset

        # Compute rolling correlation
        rolling_corr = pd.Series(
            eth_returns.rolling(window).corr(rate_returns),
            index=returns_data.index,
            name='rolling_correlation'
        )

        return rolling_corr.dropna()

    def plot_regime_correlations(self, regime_results: Dict[str, Any],
                               save_path: str = None):
        """
        Plot correlation heatmaps for different regimes

        Args:
            regime_results: Results from regime_correlation_analysis
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Correlation Matrices by Interest Rate Regime', fontsize=16)

        regimes = ['hold', 'hike', 'cut']
        titles = ['Hold', 'Hike', 'Cut']

        for i, (regime, title) in enumerate(zip(regimes, titles)):
            if regime in regime_results:
                corr_matrix = regime_results[regime]['correlation_matrix']

                # Create heatmap
                sns.heatmap(corr_matrix, ax=axes[i], annot=True, fmt='.2f',
                           cmap='RdBu_r', vmin=-1, vmax=1,
                           xticklabels=self.assets,
                           yticklabels=self.assets)

                axes[i].set_title(f'{title} Regime')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].tick_params(axis='y', rotation=0)

            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=axes[i].transAxes)
                axes[i].set_title(f'{title} Regime (No Data)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_eth_rate_correlation(self, rolling_corr: pd.DataFrame,
                                 regime_info: pd.DataFrame,
                                 save_path: str = None):
        """
        Plot rolling ETH-rate correlation with regime shading

        Args:
            rolling_corr: Series with rolling correlations
            regime_info: DataFrame with regime information
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot rolling correlation
        ax.plot(rolling_corr.index, rolling_corr, 'b-', linewidth=1,
                label='ETH-Rate Correlation')

        # Add regime shading
        regime_colors = {'hold': 'gray', 'hike': 'red', 'cut': 'green'}
        for regime, color in regime_colors.items():
            regime_periods = regime_info[regime_info['decision'] == regime]
            for idx, row in regime_periods.iterrows():
                start_date = max(row['regime_start'], rolling_corr.index[0])
                end_date = min(row['regime_end'], rolling_corr.index[-1])

                if start_date <= end_date:
                    ax.axvspan(start_date, end_date, alpha=0.3, color=color,
                             label=f'{regime.capitalize()} Regime' if regime == list(regime_colors.keys())[0] else '')

        # Add horizontal lines
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='High Correlation')
        ax.axhline(-0.5, color='red', linestyle='--', alpha=0.5, label='High Negative Correlation')

        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
        ax.set_title('Rolling Correlation: Ethereum vs Interest Rates (60-day window)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()