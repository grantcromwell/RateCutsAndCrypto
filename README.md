# Random Matrix Theory Analysis: Ethereum & Interest Rates

An analysis system that uses Random Matrix Theory to identify correlations between Ethereum price movements and US Federal Reserve interest rate decisions 


- **Data Collection**: Historical price data for Ethereum, Silver, Cloudflare, and 13-week T-bills since 2020
- **Regime Analysis**: Separate correlation analysis for different Fed policy regimes
- **RMT Algorithms**: Marchenko-Pastur distribution, eigenvalue filtering, correlation matrix cleaning
- **Signal Generation**: Trading signals based on regime-specific correlations
- **Visual Analysis**: Comprehensive plots showing eigenvalue spectra, correlation heatmaps, and signal performance

## Features

- **Multi-Asset Analysis**: Tracks Ethereum alongside Silver and Cloudflare for context
- **FOMC Integration**: Incorporates Federal Reserve meeting dates and decisions
- **Regime-Specific Insights**: Different correlation patterns for hiking, cutting, and holding periods
- **Rolling Analysis**: Time-varying correlation analysis with configurable windows
- **Signal Generation**: Actionable trading signals with confidence metrics
- **Performance Metrics**: Sharpe ratios, win rates, and risk-adjusted returns

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd coris
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run the complete analysis:
```bash
python main.py --analyze
```

Analyze specific date range:
```bash
python main.py --analyze --start-date 2021-01-01 --end-date 2023-12-31
```

Fetch data only:
```bash
python main.py --fetch-data
```

Generate plots from existing results:
```bash
python main.py --plots-only
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--analyze` | Run complete RMT analysis |
| `--fetch-data` | Fetch financial data only |
| `--start-date YYYY-MM-DD` | Analysis start date |
| `--end-date YYYY-MM-DD` | Analysis end date |
| `--no-plots` | Skip plot generation |
| `--validate-data` | Check data integrity |
| `--output-dir DIR` | Custom output directory |

## Output Files

The analysis generates several files in the `results/` directory:

- `analysis_report.json`: Summary of findings and recommendations
- `trading_signals.csv`: Historical trading signals
- `rolling_correlations.csv`: Rolling correlation data
- `regime_results.json`: Detailed regime-specific analysis

Plots are saved in the `plots/` directory:
- `eigenvalue_spectrum_*.png`: Eigenvalue distribution analysis
- `correlation_heatmaps_*.png`: Regime-specific correlation matrices
- `rolling_correlation_*.png`: Time-varying correlations
- `signal_timeline_*.png`: Trading signal visualization
- `regime_comparison_*.png`: Regime comparison metrics
- `signal_performance_*.png`: Performance analysis

## Methodology

### 1. Data Pipeline
- **Source**: Yahoo Finance (yfinance)
- **Assets**: ETH-USD, SI=F (Silver), NET (Cloudflare), ^IRX (13-week T-bill)
- **Frequency**: Daily returns
- **Period**: January 2020 to present

### 2. Random Matrix Theory Implementation
- **Correlation Matrix**: C = (1/T) * R @ R.T
- **Marchenko-Pastur Distribution**: Theoretical bounds for random eigenvalues
- **Eigenvalue Filtering**: Separate signal from noise eigenvalues
- **Correlation Cleaning**: Reconstruct "clean" correlation matrices

### 3. Regime Analysis
- **FOMC Meetings**: 8 meetings/year with decision type (hold/cut/hike)
- **Regime Windows**: Periods between meetings as rate regimes
- **Regime-Specific Correlations**: Separate analysis per regime type

### 4. Signal Generation
- **Regime Signals**: Based on current regime and historical patterns
- **Event Signals**: Analysis around FOMC meetings (±10 days)
- **Rolling Signals**: Based on rolling correlation thresholds
- **Combined Signals**: Majority vote from all signal types

### 5. Performance Metrics
- **Returns**: Cumulative and annualized returns
- **Risk**: Volatility and maximum drawdown
- **Risk-Adjusted**: Sharpe ratio
- **Win Rate**: Percentage of positive returns

## Key Findings

The analysis typically reveals:

1. **Hiking Cycles (2022-2023)**:
   - ETH shows negative correlation with rates
   - High sensitivity eigenvalue in first component
   - Sell signals when correlation turns positive

2. **Cutting Cycles (2020, 2024)**:
   - ETH positive correlation with rate cuts
   - Buy signals when rates peak before cutting

3. **Hold Periods**:
   - Lower correlation, driven by crypto-specific factors
   - Eigenvalue spectrum more dispersed
   - Follow general market sentiment

## Project Structure

```
coris/
├── data/
│   ├── fetch_data.py          # Data fetching and processing
│   ├── fomo_dates.json        # FOMC meeting database
│   ├── processed/             # Cached processed data
│   └── raw_data.csv           # Raw downloaded data
├── rmt/
│   ├── core.py                # RMT algorithms
│   ├── correlation.py         # Correlation analysis
│   └── signal_generator.py    # Trading signals
├── visualization/
│   └── plots.py               # All plotting functions
├── main.py                    # Main analysis pipeline
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── results/                   # Analysis outputs (generated)
```

## Mathematical Background

### Random Matrix Theory

For a correlation matrix C constructed from N assets over T time points:

```
C = (1/T) * R @ R.T
```

where R is the returns matrix (N × T).

The Marchenko-Pastur distribution gives the theoretical eigenvalue bounds:

```
λ_min,max = (1 ± √(T/N))²
```

Eigenvalues outside these bounds represent "signal" while those within represent "noise".

### Signal Generation Logic

```python
def generate_signal(correlation, regime):
    if abs(correlation) > 0.5:
        if correlation > 0:
            return "BUY" if regime == "cut" else "SELL"
        else:
            return "SELL" if regime == "hike" else "BUY"
    else:
        return "HOLD"
```

# RateCutsAndCrypto
# RateCutsAndCrypto
