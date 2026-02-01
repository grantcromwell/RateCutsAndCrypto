import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Any, Optional
import matplotlib.dates as mdates

class RMTPlots:
    """Comprehensive visualization module for RMT analysis"""

    def __init__(self, assets: List[str], save_dir: str = "plots"):
        """
        Initialize RMT plots

        Args:
            assets: List of asset names
            save_dir: Directory to save plots
        """
        self.assets = assets
        self.eth_idx = 0
        self.rate_idx = -1
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_eigenvalue_spectrum(self, eigenvalues: np.ndarray,
                               lambda_min: float, lambda_max: float,
                               title: str = "Eigenvalue Spectrum",
                               save_path: Optional[str] = None) -> str:
        """
        Plot eigenvalue spectrum with Marchenko-Pastur bounds

        Args:
            eigenvalues: Array of eigenvalues
            lambda_min: Lower bound of Marchenko-Pastur distribution
            lambda_max: Upper bound of Marchenko-Pastur distribution
            title: Plot title
            save_path: Custom save path

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"eigenvalue_spectrum_{datetime.now().strftime('%Y%m%d')}.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16)

        # Plot 1: Histogram vs Marchenko-Pastur
        x = np.linspace(0, np.max(eigenvalues) * 1.1, 1000)
        # Simple MP approximation for plotting
        gamma_plus = (1 + 1)**2 if len(eigenvalues) / len(self.assets) > 1 else 4
        gamma_minus = max(0, (1 - 1)**2)

        # Create a simple PDF approximation
        pdf_mp = np.zeros_like(x)
        for i in range(len(x)):
            if gamma_minus <= x[i] <= gamma_plus:
                pdf_mp[i] = np.sqrt((gamma_plus - x[i]) * (x[i] - gamma_minus)) / (2 * np.pi * x[i])

        ax1.hist(eigenvalues, bins=20, density=True, alpha=0.7, color='blue',
                label='Empirical distribution')
        ax1.plot(x, pdf_mp, 'r-', linewidth=2, label='Marchenko-Pastur')
        ax1.axvline(lambda_min, color='red', linestyle='--',
                   label=f'λ_min = {lambda_min:.3f}')
        ax1.axvline(lambda_max, color='red', linestyle='--',
                   label=f'λ_max = {lambda_max:.3f}')
        ax1.set_xlabel('Eigenvalue')
        ax1.set_ylabel('Density')
        ax1.set_title('Eigenvalue Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Sorted eigenvalues
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        ax2.plot(sorted_eigenvalues, 'bo-', markersize=8, label='Empirical eigenvalues')
        ax2.axhline(lambda_max, color='red', linestyle='--',
                   label=f'λ_max = {lambda_max:.3f}')
        ax2.set_xlabel('Eigenvalue index (sorted)')
        ax2.set_ylabel('Eigenvalue')
        ax2.set_title('Eigenvalue Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_correlation_heatmaps(self, regime_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Plot correlation heatmaps for different regimes

        Args:
            regime_results: Results from correlation analysis
            save_path: Custom save path

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"correlation_heatmaps_{datetime.now().strftime('%Y%m%d')}.png")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Correlation Matrices by Interest Rate Regime', fontsize=16)

        regimes = ['hold', 'hike', 'cut']
        titles = ['Hold', 'Hike', 'Cut']
        colors = ['gray', 'red', 'green']

        for i, (regime, title, color) in enumerate(zip(regimes, titles, colors)):
            if regime in regime_results:
                corr_matrix = regime_results[regime]['correlation_matrix']

                # Create heatmap
                sns.heatmap(corr_matrix, ax=axes[i], annot=True, fmt='.2f',
                           cmap='RdBu_r', vmin=-1, vmax=1,
                           xticklabels=self.assets,
                           yticklabels=self.assets,
                           cbar_kws={'shrink': 0.8})

                # Highlight ETH-rate correlation
                eth_rate_value = corr_matrix[self.eth_idx, self.rate_idx]
                cell_text = axes[i].texts[self.eth_idx * len(self.assets) + self.rate_idx]
                cell_text.set_text(f'{eth_rate_value:.2f}*')
                cell_text.set_weight('bold')
                cell_text.set_color('red' if abs(eth_rate_value) > 0.5 else 'black')

                axes[i].set_title(f'{title} Regime\nETH-Rate: {eth_rate_value:.3f}')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].tick_params(axis='y', rotation=0)
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'{title} Regime (No Data)')

        # Add explanation for *
        fig.text(0.02, 0.02, '*Highlighted ETH-Rate correlation', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_rolling_correlation(self, rolling_corr: pd.Series,
                               regime_info: pd.DataFrame,
                               save_path: Optional[str] = None) -> str:
        """
        Plot rolling correlation with regime shading

        Args:
            rolling_corr: Series with rolling correlations
            regime_info: DataFrame with regime information
            save_path: Custom save path

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"rolling_correlation_{datetime.now().strftime('%Y%m%d')}.png")

        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot rolling correlation
        ax.plot(rolling_corr.index, rolling_corr, 'b-', linewidth=1.5,
                label='ETH-Rate Correlation')

        # Add regime shading
        regime_colors = {'hold': 'gray', 'hike': 'red', 'cut': 'green'}
        regime_alpha = 0.2

        # Create legend handles for regimes
        legend_elements = []

        for regime, color in regime_colors.items():
            regime_periods = regime_info[regime_info['decision'] == regime]
            for idx, row in regime_periods.iterrows():
                start_date = max(row['regime_start'], rolling_corr.index[0])
                end_date = min(row['regime_end'], rolling_corr.index[-1])

                if start_date <= end_date:
                    ax.axvspan(start_date, end_date, alpha=regime_alpha, color=color)
                    # Add to legend only once
                    if regime == list(regime_colors.keys())[0]:
                        legend_elements.append(plt.Rectangle((0,0),1,1, fc=color, alpha=regime_alpha,
                                                          label=f'{regime.capitalize()} Regime'))

        # Add horizontal lines
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='High Cor Threshold')
        ax.axhline(-0.5, color='red', linestyle='--', alpha=0.7, label='High Negative Cor Threshold')

        # Set up axes
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Rolling Correlation: Ethereum vs Interest Rates (60-day window)', fontsize=14)

        # Combine legend elements
        legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=1.5,
                                         label='ETH-Rate Correlation'))
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', alpha=0.5,
                                         label='Zero Correlation'))

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_signal_timeline(self, signals: pd.DataFrame,
                           returns_data: pd.DataFrame,
                           regime_info: pd.DataFrame,
                           save_path: Optional[str] = None) -> str:
        """
        Plot timeline of trading signals

        Args:
            signals: DataFrame with signals
            returns_data: Asset returns data
            regime_info: DataFrame with regime information
            save_path: Custom save path

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"signal_timeline_{datetime.now().strftime('%Y%m%d')}.png")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Ethereum Trading Signals Analysis', fontsize=16)

        # Plot 1: ETH Price
        eth_price = (1 + returns_data.iloc[:, 0]).cumprod()
        ax1.plot(eth_price.index, eth_price, 'k-', linewidth=2, label='ETH Price')

        # Plot signals
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'blue', 'NEUTRAL': 'gray'}
        signal_sizes = {'BUY': 100, 'SELL': 100, 'HOLD': 50, 'NEUTRAL': 30}

        for signal, color in signal_colors.items():
            mask = signals['combined_signal'] == signal
            if mask.sum() > 0:
                ax1.scatter(signals.index[mask], eth_price[mask],
                          color=color, s=signal_sizes[signal], alpha=0.7,
                          label=f'{signal} ({mask.sum()} signals)', edgecolor='black')

        # Add FOMC meetings
        for _, row in regime_info.iterrows():
            meeting_date = row.name
            if meeting_date in eth_price.index:
                ax1.axvline(meeting_date, color='orange', linestyle='--', alpha=0.7,
                           label='FOMC Meeting' if meeting_date == regime_info.index[0] else '')

        ax1.set_ylabel('ETH Price')
        ax1.set_title('Ethereum Price with Trading Signals')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Signals over time
        for signal, color in signal_colors.items():
            mask = signals['combined_signal'] == signal
            if mask.sum() > 0:
                # Create step plot
                signal_series = pd.Series(0, index=signals.index)
                signal_series[mask] = list(signal_colors.keys()).index(signal) + 1
                ax2.step(signal_series.index, signal_series,
                        where='post', color=color, linewidth=2, label=signal)

        ax2.set_ylabel('Signal Type')
        ax2.set_yticks(range(1, 5))
        ax2.set_yticklabels(list(signal_colors.keys()))
        ax2.set_title('Trading Signal Timeline')
        ax2.grid(True, alpha=0.3)

        # Plot 3: ETH vs Rate correlation heatmap (rolling)
        # Create a correlation matrix heatmap for signal periods
        correlation_matrix = np.corrcoef(returns_data.T)
        im = ax3.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

        # Add annotations
        for i in range(len(self.assets)):
            for j in range(len(self.assets)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")

        ax3.set_xticks(range(len(self.assets)))
        ax3.set_yticks(range(len(self.assets)))
        ax3.set_xticklabels(self.assets, rotation=45)
        ax3.set_yticklabels(self.assets)
        ax3.set_title('Asset Correlation Matrix (Full Period)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_regime_comparison(self, regime_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
        """
        Plot regime comparison metrics

        Args:
            regime_results: Results from correlation analysis
            save_path: Custom save path

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"regime_comparison_{datetime.now().strftime('%Y%m%d')}.png")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Regime Comparison Analysis', fontsize=16)

        regimes = ['hold', 'hike', 'cut']
        colors = ['gray', 'red', 'green']

        # Plot 1: ETH-Rate correlation by regime
        eth_rate_corrs = []
        for regime in regimes:
            if regime in regime_results:
                corr_matrix = regime_results[regime]['correlation_matrix']
                eth_rate_corr = corr_matrix[self.eth_idx, self.rate_idx]
                eth_rate_corrs.append(eth_rate_corr)
            else:
                eth_rate_corrs.append(0)

        bars1 = ax1.bar(regimes, eth_rate_corrs, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('ETH-Rate Correlation')
        ax1.set_title('ETH-Rate Correlation by Regime')
        ax1.set_ylim(-1, 1)

        # Add value labels
        for bar, corr in zip(bars1, eth_rate_corrs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top')

        # Plot 2: Eigenvalue ratios by regime
        eigenvalue_ratios = []
        for regime in regimes:
            if regime in regime_results and 'rmt_analysis' in regime_results[regime]:
                largest_eig = regime_results[regime]['rmt_analysis']['spectrum_test']['lambda_max_empirical']
                eigenvalue_ratios.append(largest_eig)
            else:
                eigenvalue_ratios.append(0)

        bars2 = ax2.bar(regimes, eigenvalue_ratios, color=colors, alpha=0.7)
        ax2.set_ylabel('Largest Eigenvalue')
        ax2.set_title('Dominant Eigenvalue by Regime')
        ax2.set_yscale('log')

        # Add value labels
        for bar, ratio in zip(bars2, eigenvalue_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.3f}', ha='center', va='bottom')

        # Plot 3: Participation ratio by regime
        participation_ratios = []
        for regime in regimes:
            if regime in regime_results and 'rmt_analysis' in regime_results[regime]:
                insights = regime_results[regime]['rmt_analysis']['insights']
                if 'max_participation' in insights:
                    participation_ratios.append(insights['max_participation'])
                else:
                    participation_ratios.append(0)
            else:
                participation_ratios.append(0)

        bars3 = ax3.bar(regimes, participation_ratios, color=colors, alpha=0.7)
        ax3.set_ylabel('Max Participation Ratio')
        ax3.set_title('Market Factor Participation by Regime')
        ax3.set_ylim(0, max(participation_ratios) * 1.1 if max(participation_ratios) > 0 else 1)

        # Add value labels
        for bar, pr in zip(bars3, participation_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pr:.3f}', ha='center', va='bottom')

        # Plot 4: Observation count by regime
        observation_counts = []
        for regime in regimes:
            if regime in regime_results:
                observation_counts.append(regime_results[regime]['n_observations'])
            else:
                observation_counts.append(0)

        bars4 = ax4.bar(regimes, observation_counts, color=colors, alpha=0.7)
        ax4.set_ylabel('Number of Observations')
        ax4.set_title('Data Coverage by Regime')

        # Add value labels
        for bar, count in zip(bars4, observation_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_signal_performance(self, performance: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
        """
        Plot signal performance metrics

        Args:
            performance: Performance analysis results
            save_path: Custom save path

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"signal_performance_{datetime.now().strftime('%Y%m%d')}.png")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Signal Performance Analysis', fontsize=16)

        # Extract metrics
        signals = list(performance.keys())
        returns = [perf['annualized_return'] for perf in performance.values()]
        sharpe_ratios = [perf['sharpe_ratio'] for perf in performance.values()]
        volatilities = [perf['volatility'] for perf in performance.values()]
        win_rates = [perf['win_rate'] for perf in performance.values()]

        # Plot 1: Annualized Returns
        colors = ['green' if r > 0 else 'red' for r in returns]
        bars1 = ax1.bar(signals, returns, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Annualized Return')
        ax1.set_title('Annualized Return by Signal')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, ret in zip(bars1, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.2%}', ha='center', va='bottom' if height > 0 else 'top')

        # Plot 2: Sharpe Ratios
        bars2 = ax2.bar(signals, sharpe_ratios, color=colors, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Returns (Sharpe Ratio)')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, sharpe in zip(bars2, sharpe_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sharpe:.2f}', ha='center', va='bottom' if height > 0 else 'top')

        # Plot 3: Volatility
        bars3 = ax3.bar(signals, volatilities, color=colors, alpha=0.7)
        ax3.set_ylabel('Annualized Volatility')
        ax3.set_title('Risk (Volatility) by Signal')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, vol in zip(bars3, volatilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{vol:.2%}', ha='center', va='bottom')

        # Plot 4: Win Rates
        bars4 = ax4.bar(signals, win_rates, color=colors, alpha=0.7)
        ax4.set_ylabel('Win Rate')
        ax4.set_title('Win Rate (Positive Returns) by Signal')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, win in zip(bars4, win_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{win:.2%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def create_all_plots(self, regime_results: Dict[str, Any],
                        rolling_corr: pd.Series,
                        signals: pd.DataFrame,
                        performance: Dict[str, Any],
                        regime_info: pd.DataFrame,
                        returns_data: pd.DataFrame) -> List[str]:
        """
        Create all plots and return their paths

        Args:
            regime_results: Results from correlation analysis
            rolling_corr: Series with rolling correlations
            signals: DataFrame with signals
            performance: Performance analysis results
            regime_info: DataFrame with regime information
            returns_data: Asset returns data

        Returns:
            List of plot file paths
        """
        plot_paths = []

        # Eigenvalue spectrum plots
        if 'rmt_analysis' in regime_results.get('hold', {}):
            eigenvalues = regime_results['hold']['rmt_analysis']['eigenvalues']
            lambda_min = regime_results['hold']['rmt_analysis']['spectrum_test']['lambda_min_theoretical']
            lambda_max = regime_results['hold']['rmt_analysis']['spectrum_test']['lambda_max_theoretical']
            plot_paths.append(self.plot_eigenvalue_spectrum(
                eigenvalues, lambda_min, lambda_max,
                "Eigenvalue Spectrum (Hold Regime)"
            ))

        # Correlation heatmaps
        plot_paths.append(self.plot_correlation_heatmaps(regime_results))

        # Rolling correlation
        plot_paths.append(self.plot_rolling_correlation(rolling_corr, regime_info))

        # Signal timeline
        plot_paths.append(self.plot_signal_timeline(signals, returns_data, regime_info))

        # Regime comparison
        plot_paths.append(self.plot_regime_comparison(regime_results))

        # Signal performance
        if performance:
            plot_paths.append(self.plot_signal_performance(performance))

        return plot_paths