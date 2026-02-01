import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from datetime import datetime, timedelta

class DataFetcher:
    """Fetch and process financial data for RMT analysis"""

    def __init__(self):
        # Asset tickers and names
        self.tickers = {
            'ETH-USD': 'Ethereum',
            'SI=F': 'Silver Futures',
            'NET': 'Cloudflare',
            '^IRX': '13-Week T-Bill Rate'
        }

        self.start_date = '2020-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')

        # Paths
        self.raw_dir = Path(__file__).parent
        self.processed_dir = self.raw_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)

    def fetch_data(self):
        """Download historical data from Yahoo Finance"""
        print(f"Fetching data from {self.start_date} to {self.end_date}")

        try:
            # Download data
            data = yf.download(
                list(self.tickers.keys()),
                start=self.start_date,
                end=self.end_date,
                progress=True
            )

            # Save raw data
            raw_file = self.raw_dir / 'raw_data.csv'
            data.to_csv(raw_file)
            print(f"Raw data saved to {raw_file}")

            # Process data
            processed_data = self.process_data(data)

            # Save processed data
            processed_file = self.processed_dir / 'returns.csv'
            processed_data.to_csv(processed_file)
            print(f"Processed returns saved to {processed_file}")

            return processed_data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def process_data(self, data):
        """Process raw price data into returns"""
        # Keep only Adj Close prices
        try:
            price_data = data['Adj Close']
        except KeyError:
            # If Adj Close not available, use Close
            price_data = data['Close']

        # Handle missing values
        price_data = price_data.ffill()  # Forward fill
        price_data = price_data.bfill()  # Backward fill if needed

        # Calculate log returns
        returns = np.log(price_data / price_data.shift(1))
        returns = returns.dropna()

        # Add metadata
        returns.attrs = {
            'start_date': returns.index.min().strftime('%Y-%m-%d') if not pd.isna(returns.index.min()) else 'N/A',
            'end_date': returns.index.max().strftime('%Y-%m-%d') if not pd.isna(returns.index.max()) else 'N/A',
            'tickers': self.tickers
        }

        return returns

    def load_processed_data(self):
        """Load processed returns data if available"""
        processed_file = self.processed_dir / 'returns.csv'

        if processed_file.exists():
            print(f"Loading processed data from {processed_file}")
            data = pd.read_csv(processed_file, index_col=0, parse_dates=True)
            return data
        else:
            print("Processed data not found, fetching new data...")
            return self.fetch_data()

    def get_fomc_data(self):
        """Load FOMC meeting dates and decisions"""
        fomo_file = Path(__file__).parent / 'fomo_dates.json'

        with open(fomo_file, 'r') as f:
            fomo_data = json.load(f)

        # Convert to DataFrame
        fomo_df = pd.DataFrame.from_dict(fomo_data, orient='index')
        fomo_df.index = pd.to_datetime(fomo_df.index, errors='coerce')

        # Add regime columns
        fomo_df = self.add_regime_columns(fomo_df)

        return fomo_df

    def add_regime_columns(self, fomo_df):
        """Add regime information for periods between meetings"""
        # Sort by date
        fomo_df = fomo_df.sort_index()

        # Add start and end of regimes
        regime_periods = []
        for i in range(len(fomo_df)):
            if i == 0:
                # First period: start of data to first meeting
                start_date = pd.to_datetime(self.start_date)
            else:
                start_date = fomo_df.index[i-1] + timedelta(days=1)

            end_date = fomo_df.index[i]

            regime_periods.append({
                'start': start_date,
                'end': end_date,
                'decision': fomo_df.iloc[i]['decision'],
                'rate': fomo_df.iloc[i]['rate']
            })

        # Create regime DataFrame
        regime_df = pd.DataFrame(regime_periods)

        # Mark regime periods in original data
        fomo_df['regime_start'] = [pd.NaT] * len(fomo_df)
        fomo_df['regime_end'] = [pd.NaT] * len(fomo_df)

        for i, period in enumerate(regime_periods):
            mask = (fomo_df.index >= period['start']) & (fomo_df.index <= period['end'])
            fomo_df.loc[mask, 'regime_start'] = period['start']
            fomo_df.loc[mask, 'regime_end'] = period['end']

        return fomo_df

if __name__ == "__main__":
    fetcher = DataFetcher()

    # Fetch data
    returns_data = fetcher.fetch_data()

    # Get FOMC data
    fomo_data = fetcher.get_fomc_data()

    print("Data fetching completed!")