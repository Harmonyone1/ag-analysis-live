"""
Retail Sentiment Data Fetcher

Fetches retail positioning data from various sources:
- Myfxbook Community Outlook
- DailyFX IG Client Sentiment

Used as a contrarian indicator (fade retail).
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
import requests


@dataclass
class RetailSentiment:
    """Retail sentiment data for a currency pair."""
    symbol: str
    long_percentage: float
    short_percentage: float
    long_short_ratio: float
    source: str
    timestamp: datetime

    @property
    def bias(self) -> str:
        """
        Get contrarian bias (opposite of retail).

        If retail is >60% long, bias is bearish (fade longs).
        If retail is >60% short, bias is bullish (fade shorts).
        """
        if self.long_percentage > 60:
            return "bearish"  # Fade retail longs
        elif self.short_percentage > 60:
            return "bullish"  # Fade retail shorts
        else:
            return "neutral"

    @property
    def strength(self) -> str:
        """Get the strength of the sentiment signal."""
        extreme = max(self.long_percentage, self.short_percentage)
        if extreme > 75:
            return "extreme"
        elif extreme > 65:
            return "strong"
        elif extreme > 55:
            return "moderate"
        else:
            return "weak"


class SentimentFetcher:
    """
    Fetches retail sentiment data from multiple sources.

    Use retail sentiment as a contrarian indicator - when retail
    is heavily positioned one way, consider trading the opposite.
    """

    def __init__(self):
        """Initialize the sentiment fetcher."""
        self.cache: Dict[str, RetailSentiment] = {}
        self.cache_duration = 300  # 5 minutes

    def get_myfxbook_sentiment(self, symbol: str) -> Optional[RetailSentiment]:
        """
        Fetch sentiment from Myfxbook Community Outlook.

        Args:
            symbol: Currency pair (e.g., "EURUSD")

        Returns:
            RetailSentiment object or None if fetch fails
        """
        try:
            # Myfxbook community outlook URL
            url = "https://www.myfxbook.com/community/outlook"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            # Parse the HTML to extract sentiment data
            # This is a simplified example - actual parsing would be more complex
            content = response.text

            # Try to find the symbol in the response
            symbol_clean = symbol.upper().replace("/", "")

            # Look for pattern like "EURUSD" followed by percentages
            pattern = rf'{symbol_clean}.*?(\d+\.?\d*)%.*?(\d+\.?\d*)%'
            match = re.search(pattern, content, re.DOTALL)

            if match:
                long_pct = float(match.group(1))
                short_pct = float(match.group(2))

                return RetailSentiment(
                    symbol=symbol,
                    long_percentage=long_pct,
                    short_percentage=short_pct,
                    long_short_ratio=long_pct / short_pct if short_pct > 0 else 0,
                    source="myfxbook",
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error fetching Myfxbook sentiment: {e}")

        return None

    def get_dailyfx_sentiment(self, symbol: str) -> Optional[RetailSentiment]:
        """
        Fetch sentiment from DailyFX (IG Client Sentiment).

        Args:
            symbol: Currency pair (e.g., "EURUSD")

        Returns:
            RetailSentiment object or None if fetch fails
        """
        try:
            # DailyFX API endpoint
            symbol_map = {
                "EURUSD": "EUR/USD",
                "GBPUSD": "GBP/USD",
                "USDJPY": "USD/JPY",
                "USDCHF": "USD/CHF",
                "AUDUSD": "AUD/USD",
                "USDCAD": "USD/CAD",
                "NZDUSD": "NZD/USD",
                "XAUUSD": "Gold",
            }

            pair = symbol_map.get(symbol.upper(), symbol)

            url = f"https://www.dailyfx.com/sentiment-report"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            # Parse response for sentiment data
            content = response.text

            # Look for sentiment data pattern
            pattern = rf'{pair}.*?(\d+\.?\d*)%\s*(?:long|Long).*?(\d+\.?\d*)%\s*(?:short|Short)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

            if match:
                long_pct = float(match.group(1))
                short_pct = float(match.group(2))

                return RetailSentiment(
                    symbol=symbol,
                    long_percentage=long_pct,
                    short_percentage=short_pct,
                    long_short_ratio=long_pct / short_pct if short_pct > 0 else 0,
                    source="dailyfx",
                    timestamp=datetime.now()
                )

        except Exception as e:
            print(f"Error fetching DailyFX sentiment: {e}")

        return None

    def get_sentiment(self, symbol: str) -> Optional[RetailSentiment]:
        """
        Get sentiment from any available source.

        Args:
            symbol: Currency pair

        Returns:
            RetailSentiment object or None
        """
        # Try Myfxbook first
        sentiment = self.get_myfxbook_sentiment(symbol)
        if sentiment:
            self.cache[symbol] = sentiment
            return sentiment

        # Try DailyFX
        sentiment = self.get_dailyfx_sentiment(symbol)
        if sentiment:
            self.cache[symbol] = sentiment
            return sentiment

        # Return cached if available
        return self.cache.get(symbol)

    def get_all_sentiment(self, symbols: list) -> Dict[str, RetailSentiment]:
        """
        Get sentiment for multiple symbols.

        Args:
            symbols: List of currency pairs

        Returns:
            Dictionary mapping symbol to RetailSentiment
        """
        results = {}
        for symbol in symbols:
            sentiment = self.get_sentiment(symbol)
            if sentiment:
                results[symbol] = sentiment
        return results
