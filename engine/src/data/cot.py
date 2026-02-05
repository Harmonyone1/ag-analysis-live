"""
Commitment of Traders (COT) Data Fetcher

Fetches weekly COT data from CFTC to understand institutional positioning.
- Commercial traders (hedgers) - often contrarian signal
- Non-commercial (speculators/funds) - trend following
- Non-reportable (retail) - contrarian signal
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd


@dataclass
class COTData:
    """COT positioning data for a futures contract."""
    symbol: str
    report_date: datetime

    # Non-Commercial (Speculators/Funds)
    spec_long: int
    spec_short: int
    spec_net: int

    # Commercial (Hedgers)
    comm_long: int
    comm_short: int
    comm_net: int

    # Non-Reportable (Small traders/Retail)
    nonrep_long: int
    nonrep_short: int
    nonrep_net: int

    # Open Interest
    open_interest: int

    @property
    def spec_net_pct(self) -> float:
        """Net speculator position as percentage of open interest."""
        return (self.spec_net / self.open_interest * 100) if self.open_interest > 0 else 0

    @property
    def bias(self) -> str:
        """
        Determine bias from COT data.

        We follow speculators (non-commercial) as they tend to be
        trend followers with better success rates.
        """
        if self.spec_net_pct > 20:
            return "bullish"
        elif self.spec_net_pct < -20:
            return "bearish"
        else:
            return "neutral"

    @property
    def extreme_positioning(self) -> Optional[str]:
        """Check if positioning is at extreme levels."""
        if self.spec_net_pct > 50:
            return "extreme_long"
        elif self.spec_net_pct < -50:
            return "extreme_short"
        return None


class COTFetcher:
    """
    Fetches COT data from CFTC.

    COT data is released weekly (Friday) with data as of Tuesday.
    Use this for longer-term directional bias.
    """

    # CFTC contract codes for forex futures
    CONTRACT_CODES = {
        "EURUSD": "099741",  # Euro FX
        "GBPUSD": "096742",  # British Pound
        "USDJPY": "097741",  # Japanese Yen (inverted)
        "USDCHF": "092741",  # Swiss Franc (inverted)
        "AUDUSD": "232741",  # Australian Dollar
        "USDCAD": "090741",  # Canadian Dollar (inverted)
        "NZDUSD": "112741",  # New Zealand Dollar
        "XAUUSD": "088691",  # Gold
    }

    def __init__(self):
        """Initialize COT fetcher."""
        self.cache: Dict[str, COTData] = {}
        self.base_url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

    def fetch_cot_data(self, symbol: str) -> Optional[COTData]:
        """
        Fetch latest COT data for a symbol.

        Args:
            symbol: Currency pair or commodity symbol

        Returns:
            COTData object or None if not available
        """
        contract_code = self.CONTRACT_CODES.get(symbol.upper())
        if not contract_code:
            return None

        try:
            # Query CFTC API
            params = {
                "cftc_contract_market_code": contract_code,
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": 1
            }

            response = requests.get(self.base_url, params=params, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()

            if not data:
                return None

            row = data[0]

            cot = COTData(
                symbol=symbol,
                report_date=datetime.strptime(
                    row.get("report_date_as_yyyy_mm_dd", ""),
                    "%Y-%m-%d"
                ),
                spec_long=int(row.get("noncomm_positions_long_all", 0)),
                spec_short=int(row.get("noncomm_positions_short_all", 0)),
                spec_net=int(row.get("noncomm_positions_long_all", 0)) -
                         int(row.get("noncomm_positions_short_all", 0)),
                comm_long=int(row.get("comm_positions_long_all", 0)),
                comm_short=int(row.get("comm_positions_short_all", 0)),
                comm_net=int(row.get("comm_positions_long_all", 0)) -
                         int(row.get("comm_positions_short_all", 0)),
                nonrep_long=int(row.get("nonrept_positions_long_all", 0)),
                nonrep_short=int(row.get("nonrept_positions_short_all", 0)),
                nonrep_net=int(row.get("nonrept_positions_long_all", 0)) -
                           int(row.get("nonrept_positions_short_all", 0)),
                open_interest=int(row.get("open_interest_all", 1))
            )

            self.cache[symbol] = cot
            return cot

        except Exception as e:
            print(f"Error fetching COT data: {e}")
            return self.cache.get(symbol)

    def get_cot_history(
        self,
        symbol: str,
        weeks: int = 52
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical COT data for analysis.

        Args:
            symbol: Currency pair
            weeks: Number of weeks of history

        Returns:
            DataFrame with COT history
        """
        contract_code = self.CONTRACT_CODES.get(symbol.upper())
        if not contract_code:
            return None

        try:
            params = {
                "cftc_contract_market_code": contract_code,
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": weeks
            }

            response = requests.get(self.base_url, params=params, timeout=15)

            if response.status_code != 200:
                return None

            data = response.json()

            if not data:
                return None

            records = []
            for row in data:
                records.append({
                    "date": row.get("report_date_as_yyyy_mm_dd"),
                    "spec_long": int(row.get("noncomm_positions_long_all", 0)),
                    "spec_short": int(row.get("noncomm_positions_short_all", 0)),
                    "comm_long": int(row.get("comm_positions_long_all", 0)),
                    "comm_short": int(row.get("comm_positions_short_all", 0)),
                    "open_interest": int(row.get("open_interest_all", 1)),
                })

            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df['spec_net'] = df['spec_long'] - df['spec_short']
            df['comm_net'] = df['comm_long'] - df['comm_short']
            df['spec_net_pct'] = df['spec_net'] / df['open_interest'] * 100

            return df.sort_values('date')

        except Exception as e:
            print(f"Error fetching COT history: {e}")
            return None

    def get_cot_extremes(self, symbol: str, weeks: int = 52) -> Optional[dict]:
        """
        Calculate COT positioning extremes for context.

        Args:
            symbol: Currency pair
            weeks: Lookback period

        Returns:
            Dictionary with min/max/percentile info
        """
        history = self.get_cot_history(symbol, weeks)
        if history is None or len(history) < 10:
            return None

        current = history.iloc[-1]

        return {
            "current_spec_net": current['spec_net'],
            "current_spec_net_pct": current['spec_net_pct'],
            "max_spec_net": history['spec_net'].max(),
            "min_spec_net": history['spec_net'].min(),
            "percentile": (
                (history['spec_net'] <= current['spec_net']).sum() /
                len(history) * 100
            ),
            "is_extreme_long": current['spec_net_pct'] > history['spec_net_pct'].quantile(0.9),
            "is_extreme_short": current['spec_net_pct'] < history['spec_net_pct'].quantile(0.1),
        }
