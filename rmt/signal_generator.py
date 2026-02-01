import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class SignalGenerator:
    """Generate trading signals based on RMT correlation analysis"""

    def __init__(self, assets: List[str]):
        """
        Initialize signal generator

        Args:
            assets: List of asset names/tickers
        """
        self.assets = assets
        self.eth_idx = 0  # ETH is always first asset
        self.rate_idx = -1  # Rate is always last asset

    def generate_signals(self, regime_results: Dict[str, Any],
                         returns_data: pd.DataFrame,
                         regime_info: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate buy/sell signals based on RMT analysis

        Args:
            regime_results: Results from correlation analysis
            returns_data: Asset returns data
            regime_info: DataFrame with regime information

        Returns:
            Dictionary with trading signals and analysis
        """
        signals = {}

        # Current regime determination
        current_regime = self.determine_current_regime(regime_info)

        # Generate regime-specific signals
        signals['current_regime'] = current_regime
        signals['regime_signals'] = self.generate_regime_specific_signals(
            regime_results, current_regime
        )

        # Generate event-based signals around FOMC meetings
        signals['event_signals'] = self.generate_event_signals(
            returns_data, regime_info
        )

        # Generate rolling correlation signals
        signals['rolling_signals'] = self.generate_rolling_correlation_signals(
            returns_data
        )

        # Combine signals
        signals['combined_signals'] = self.combine_signals(signals)

        # Performance analysis
        signals['performance'] = self.analyze_signal_performance(
            signals, returns_data
        )

        return signals

    def determine_current_regime(self, regime_info: pd.DataFrame) -> str:
        """
        Determine current regime based on most recent FOMC decision

        Args:
            regime_info: DataFrame with regime information

        Returns:
            Current regime ('hold', 'hike', or 'cut')
        """
        if len(regime_info) == 0:
            return 'hold'

        # Get most recent regime
        most_recent = regime_info.iloc[-1]
        return most_recent['decision']

    def generate_regime_specific_signals(self, regime_results: Dict[str, Any],
                                       current_regime: str) -> Dict[str, Any]:
        """
        Generate signals based on current regime

        Args:
            regime_results: Results from correlation analysis
            current_regime: Current regime type

        Returns:
            Regime-specific signals
        """
        signals = {}

        if current_regime in regime_results:
            regime_data = regime_results[current_regime]
            rmt_analysis = regime_data['rmt_analysis']

            # Get ETH-rate correlation for this regime
            eth_rate_corr = regime_data['correlation_matrix'][self.eth_idx, self.rate_idx]

            # Signal logic based on correlation strength
            if abs(eth_rate_corr) > 0.5:
                # Strong correlation - follow rate trends
                if eth_rate_corr > 0:
                    signal = 'BUY' if current_regime == 'cut' else 'SELL'
                    reasoning = f"Strong positive correlation ({eth_rate_corr:.3f}): ETH moves with rates. Rates {current_regime}, so ETH {signal.lower()}."
                else:
                    signal = 'SELL' if current_regime == 'hike' else 'BUY'
                    reasoning = f"Strong negative correlation ({eth_rate_corr:.3f}): ETH moves against rates. Rates {current_regime}, so ETH {signal.lower()}."
            else:
                # Weak correlation - use eigenvalue analysis
                signal = 'HOLD'
                reasoning = f"Weak correlation ({eth_rate_corr:.3f}) in {current_regime} regime. Follow general market sentiment."

            # Additional check for signal eigenvalues
            if 'spectrum_test' in rmt_analysis:
                signal_ratio = rmt_analysis['spectrum_test']['signal_ratio']
                if signal_ratio > 0.2:
                    reasoning += f" High signal eigenvalue ratio ({signal_ratio:.3f}) indicates strong regime-specific pattern."

            signals[current_regime] = {
                'signal': signal,
                'correlation': eth_rate_corr,
                'reasoning': reasoning,
                'confidence': min(abs(eth_rate_corr) * 2, 1.0)  # Confidence from correlation strength
            }
        else:
            signals[current_regime] = {
                'signal': 'HOLD',
                'correlation': None,
                'reasoning': f"No data for {current_regime} regime",
                'confidence': 0.0
            }

        return signals

    def generate_event_signals(self, returns_data: pd.DataFrame,
                            regime_info: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate signals around FOMC meeting events

        Args:
            returns_data: Asset returns data
            regime_info: DataFrame with regime information

        Returns:
            Event-based signals
        """
        event_signals = {}

        # Analyze returns around FOMC meetings
        for i, (_, row) in enumerate(regime_info.iterrows()):
            meeting_date = row.name

            # Define event window (±10 days)
            start_date = meeting_date - timedelta(days=10)
            end_date = meeting_date + timedelta(days=10)

            # Get returns in window
            window_mask = (returns_data.index >= start_date) & (returns_data.index <= end_date)
            window_returns = returns_data.loc[window_mask]

            if len(window_returns) > 0:
                # Calculate cumulative ETH return
                eth_returns = window_returns.iloc[:, 0]
                cum_return = (1 + eth_returns).prod() - 1

                # Generate signal based on cumulative return
                if cum_return > 0.05:  # 5% gain
                    signal = 'STRONG_BUY'
                    reasoning = f"ETH gained {cum_return:.2%} around {meeting_date.date()} {row['decision']} decision"
                elif cum_return > 0.02:  # 2% gain
                    signal = 'BUY'
                    reasoning = f"ETH gained {cum_return:.2%} around {meeting_date.date()} {row['decision']} decision"
                elif cum_return < -0.05:  # 5% loss
                    signal = 'STRONG_SELL'
                    reasoning = f"ETH lost {abs(cum_return):.2%} around {meeting_date.date()} {row['decision']} decision"
                elif cum_return < -0.02:  # 2% loss
                    signal = 'SELL'
                    reasoning = f"ETH lost {abs(cum_return):.2%} around {meeting_date.date()} {row['decision']} decision"
                else:
                    signal = 'HOLD'
                    reasoning = f"ETH relatively unchanged ({cum_return:.2%}) around {meeting_date.date()} {row['decision']} decision"

                event_signals[str(meeting_date.date())] = {
                    'meeting_date': meeting_date.date(),
                    'decision': row['decision'],
                    'cumulative_return': cum_return,
                    'signal': signal,
                    'reasoning': reasoning,
                    'window_start': start_date.date(),
                    'window_end': end_date.date()
                }

        return event_signals

    def generate_rolling_correlation_signals(self, returns_data: pd.DataFrame,
                                         window: int = 60) -> pd.Series:
        """
        Generate signals based on rolling correlation

        Args:
            returns_data: Asset returns data
            window: Rolling window size

        Returns:
            Series of rolling signals
        """
        # Compute rolling correlation
        eth_returns = returns_data.iloc[:, 0]
        rate_returns = returns_data.iloc[:, -1]
        rolling_corr = eth_returns.rolling(window).corr(rate_returns)

        # Generate signals
        signals = pd.Series(index=rolling_corr.index, dtype=str)

        for idx in rolling_corr.index:
            corr = rolling_corr.loc[idx]
            if pd.notna(corr):
                if corr > 0.6:
                    signals.loc[idx] = 'BUY'
                elif corr < -0.6:
                    signals.loc[idx] = 'SELL'
                elif abs(corr) > 0.3:
                    signals.loc[idx] = 'HOLD'
                else:
                    signals.loc[idx] = 'NEUTRAL'

        return signals.dropna()

    def combine_signals(self, signals: Dict[str, Any]) -> pd.DataFrame:
        """
        Combine all signals into a unified signal series

        Args:
            signals: Dictionary containing all signal types

        Returns:
            DataFrame with combined signals
        """
        combined = pd.DataFrame(index=signals['rolling_signals'].index)

        # Add rolling signals
        combined['rolling_signal'] = signals['rolling_signals']

        # Add regime signals (forward-filled)
        regime_signal = 'HOLD'  # Default
        if 'regime_specific_signals' in signals and signals['current_regime'] in signals['regime_specific_signals']:
            regime_signal = signals['regime_specific_signals'][signals['current_regime']]['signal']
        combined['regime_signal'] = regime_signal

        # Add event signals (as vertical markers)
        event_signal_dates = []
        for date, event_data in signals['event_signals'].items():
            event_date = pd.to_datetime(event_data['meeting_date'])
            if event_date in combined.index:
                combined.loc[event_date, 'event_signal'] = event_data['signal']
                combined.loc[event_date, 'event_reasoning'] = event_data['reasoning']
                event_signal_dates.append(event_date)

        # Generate combined signal (majority vote)
        def majority_vote(row):
            votes = []
            if 'rolling_signal' in row and pd.notna(row['rolling_signal']):
                votes.append(row['rolling_signal'])
            if 'regime_signal' in row and pd.notna(row['regime_signal']):
                votes.append(row['regime_signal'])
            if 'event_signal' in row and pd.notna(row['event_signal']):
                votes.append(row['event_signal'])

            if len(votes) >= 2:
                # Majority vote
                vote_counts = pd.Series(votes).value_counts()
                if vote_counts.iloc[0] > len(votes) / 2:
                    return vote_counts.index[0]
                else:
                    return 'HOLD'
            elif len(votes) == 1:
                return votes[0]
            else:
                return 'NEUTRAL'

        combined['combined_signal'] = combined.apply(majority_vote, axis=1)

        return combined

    def analyze_signal_performance(self, signals: Dict[str, Any],
                                returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze historical performance of signals

        Args:
            signals: Dictionary with signals
            returns_data: Asset returns data

        Returns:
            Performance analysis results
        """
        performance = {}

        # Get ETH returns
        eth_returns = returns_data.iloc[:, 0]

        # Combine signals
        combined_signals = signals['combined_signals']

        # Calculate cumulative returns for each signal
        signal_returns = {}
        for signal in ['BUY', 'SELL', 'HOLD', 'NEUTRAL']:
            mask = combined_signals['combined_signal'] == signal
            if mask.sum() > 0:
                # Align indices
                aligned_returns = eth_returns.loc[mask.index[mask]]
                signal_returns[signal] = aligned_returns[mask[mask]]

        # Calculate performance metrics
        for signal, returns in signal_returns.items():
            if len(returns) > 0:
                cumulative = (1 + returns).prod() - 1
                annualized = (1 + cumulative) ** (252 / len(returns)) - 1
                volatility = returns.std() * np.sqrt(252)
                sharpe = annualized / volatility if volatility > 0 else 0

                performance[signal] = {
                    'n_signals': len(returns),
                    'cumulative_return': cumulative,
                    'annualized_return': annualized,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': self.calculate_max_drawdown(returns),
                    'win_rate': (returns > 0).mean()
                }

        # Compare to buy-and-hold
        buy_hold_cum = (1 + eth_returns).prod() - 1
        buy_hold_ann = (1 + buy_hold_cum) ** (252 / len(eth_returns)) - 1
        buy_hold_vol = eth_returns.std() * np.sqrt(252)
        buy_hold_sharpe = buy_hold_ann / buy_hold_vol if buy_hold_vol > 0 else 0

        performance['buy_and_hold'] = {
            'cumulative_return': buy_hold_cum,
            'annualized_return': buy_hold_ann,
            'volatility': buy_hold_vol,
            'sharpe_ratio': buy_hold_sharpe,
            'n_signals': len(eth_returns)
        }

        return performance

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown for a series of returns

        Args:
            returns: Series of returns

        Returns:
            Maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def plot_signals(self, signals: Dict[str, Any], returns_data: pd.DataFrame,
                    save_path: str = None):
        """
        Plot trading signals and performance

        Args:
            signals: Dictionary with signals
            returns_data: Asset returns data
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: ETH price with signals
        eth_price = (1 + returns_data.iloc[:, 0]).cumprod()
        ax1.plot(eth_price.index, eth_price, 'k-', label='ETH Price')

        # Plot signals
        combined_signals = signals['combined_signals']
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'blue', 'NEUTRAL': 'gray'}

        for signal, color in signal_colors.items():
            mask = combined_signals['combined_signal'] == signal
            if mask.sum() > 0:
                ax1.scatter(combined_signals.index[mask], eth_price[mask],
                          color=color, s=50, alpha=0.7, label=f'{signal} Signal')

        # Add FOMC meetings
        regime_info = signals['regime_info']
        for _, row in regime_info.iterrows():
            meeting_date = row.name
            if meeting_date in eth_price.index:
                ax1.axvline(meeting_date, color='orange', linestyle='--', alpha=0.5)

        ax1.set_ylabel('ETH Price')
        ax1.set_title('Ethereum Price with Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Performance comparison
        performance = signals['performance']
        signals_list = list(performance.keys())
        returns_list = [perf['annualized_return'] for perf in performance.values()]
        colors_list = [signal_colors.get(s, 'gray') for s in signals_list]

        bars = ax2.bar(signals_list, returns_list, color=colors_list, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Annualized Return')
        ax2.set_title('Signal Performance Comparison')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, return_val in zip(bars, returns_list):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{return_val:.2%}', ha='center', va='bottom' if height > 0 else 'top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()