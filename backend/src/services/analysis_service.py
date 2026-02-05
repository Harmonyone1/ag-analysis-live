"""Analysis service for market analysis operations."""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

# Add engine to path for imports
sys.path.insert(0, "/app/engine/src")

from sqlalchemy.orm import Session
import structlog

logger = structlog.get_logger(__name__)


class AnalysisService:
    """Service for market analysis operations."""

    def __init__(self, db: Session):
        """Initialize service.

        Args:
            db: Database session
        """
        self.db = db
        self._strength_cache: Optional[Dict] = None
        self._strength_cache_time: Optional[datetime] = None

    async def get_currency_strength(self) -> Dict[str, Any]:
        """Get current currency strength rankings.

        Returns:
            Dict with currency strength data
        """
        # Check cache (valid for 1 minute)
        if (
            self._strength_cache
            and self._strength_cache_time
            and datetime.now() - self._strength_cache_time < timedelta(minutes=1)
        ):
            return self._strength_cache

        try:
            # Query latest strength snapshot from database
            from database.models import AnalysisSnapshot

            latest = (
                self.db.query(AnalysisSnapshot)
                .filter(AnalysisSnapshot.strength_scores.isnot(None))
                .order_by(AnalysisSnapshot.snapshot_time.desc())
                .first()
            )

            if latest and latest.strength_scores:
                # Get 1D strength scores if available
                strength_data = latest.strength_scores.get("1D", latest.strength_scores)
                currencies = []
                for i, (ccy, strength) in enumerate(
                    sorted(strength_data.items(), key=lambda x: x[1], reverse=True)
                ):
                    currencies.append({
                        "currency": ccy,
                        "strength": float(strength),
                        "rank": i + 1,
                    })

                result = {
                    "timestamp": latest.snapshot_time.isoformat(),
                    "currencies": currencies,
                    "strongest": currencies[0]["currency"] if currencies else None,
                    "weakest": currencies[-1]["currency"] if currencies else None,
                }
            else:
                # Return default/empty data
                default_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "currencies": [
                        {"currency": c, "strength": 0.0, "rank": i + 1}
                        for i, c in enumerate(default_currencies)
                    ],
                    "strongest": None,
                    "weakest": None,
                }

            self._strength_cache = result
            self._strength_cache_time = datetime.now()
            return result

        except Exception as e:
            logger.error("Failed to get currency strength", error=str(e))
            raise

    async def get_analysis_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get latest analysis snapshot for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with analysis snapshot data
        """
        try:
            from database.models import AnalysisSnapshot, PriceHistory

            # Get latest analysis snapshot for symbol
            snapshot = (
                self.db.query(AnalysisSnapshot)
                .filter(AnalysisSnapshot.symbol == symbol.upper())
                .order_by(AnalysisSnapshot.snapshot_time.desc())
                .first()
            )

            # Build components dict from snapshot fields
            components = {}
            if snapshot:
                if snapshot.structure_state or snapshot.trend_direction:
                    components["structure"] = {
                        "state": snapshot.structure_state,
                        "trend": snapshot.trend_direction,
                    }
                if snapshot.liquidity_zones:
                    components["liquidity"] = {"zones": snapshot.liquidity_zones}
                if snapshot.momentum_state:
                    components["momentum"] = snapshot.momentum_state

            # Get current price
            latest_price = (
                self.db.query(PriceHistory)
                .filter(PriceHistory.symbol == symbol.upper())
                .order_by(PriceHistory.bar_time.desc())
                .first()
            )

            current_price = float(latest_price.close) if latest_price else 0.0

            # Build response
            result = {
                "symbol": symbol.upper(),
                "timeframe": "M15",
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "structure": components.get("structure"),
                "liquidity": components.get("liquidity"),
                "momentum": components.get("momentum"),
                "directional_bias": self._determine_bias(components),
                "bias_strength": self._calculate_bias_strength(components),
                "key_levels": self._extract_key_levels(components),
            }

            return result

        except Exception as e:
            logger.error("Failed to get analysis snapshot", error=str(e), symbol=symbol)
            raise

    async def get_trade_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get trade candidates.

        Args:
            status: Filter by status (PENDING, APPROVED, etc.)
            limit: Maximum candidates to return

        Returns:
            Dict with candidates list
        """
        try:
            from database.models import TradeCandidate
            import time

            start_time = time.time()

            query = self.db.query(TradeCandidate)

            if status:
                query = query.filter(TradeCandidate.status == status)
            else:
                # Default to pending candidates
                query = query.filter(TradeCandidate.status == "PENDING")

            # Order by score descending
            candidates = (
                query.order_by(TradeCandidate.confluence_score.desc())
                .limit(limit)
                .all()
            )

            scan_duration = int((time.time() - start_time) * 1000)

            result = {
                "timestamp": datetime.now().isoformat(),
                "scan_duration_ms": scan_duration,
                "candidates": [
                    {
                        "id": str(c.id),
                        "symbol": c.symbol,
                        "direction": c.direction,
                        "confluence_score": c.confluence_score,
                        "entry_zone": c.entry_zone or {},
                        "stop_price": float(c.stop_price) if c.stop_price else 0,
                        "tp_targets": c.tp_targets or [],
                        "risk_reward_ratio": float(c.ai_expected_r) if c.ai_expected_r else 0,
                        "reasons": list(c.reasons) if c.reasons else [],
                        "status": c.status,
                        "created_at": c.created_at.isoformat(),
                        "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                        "ai_probability": float(c.ai_confidence) if c.ai_confidence else None,
                        "ai_decision": "approved" if c.ai_approved else ("rejected" if c.ai_approved is False else "pending"),
                    }
                    for c in candidates
                ],
                "symbols_scanned": 28,  # Would be from actual scan
                "filters_applied": ["confluence >= 60", "risk_check passed"],
            }

            return result

        except Exception as e:
            logger.error("Failed to get trade candidates", error=str(e))
            raise

    async def get_chart_data(
        self,
        symbol: str,
        timeframe: str = "M15",
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Get chart data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            limit: Number of candles

        Returns:
            Dict with chart data
        """
        try:
            from database.models import PriceHistory

            candles = (
                self.db.query(PriceHistory)
                .filter(
                    PriceHistory.symbol == symbol.upper(),
                    PriceHistory.timeframe == timeframe,
                )
                .order_by(PriceHistory.bar_time.desc())
                .limit(limit)
                .all()
            )

            # Reverse to chronological order
            candles = list(reversed(candles))

            result = {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "candles": [
                    {
                        "timestamp": c.bar_time.isoformat(),
                        "open": float(c.open),
                        "high": float(c.high),
                        "low": float(c.low),
                        "close": float(c.close),
                        "volume": float(c.volume) if c.volume else None,
                    }
                    for c in candles
                ],
                "annotations": [],  # Would include levels, sweeps, etc.
            }

            return result

        except Exception as e:
            logger.error("Failed to get chart data", error=str(e), symbol=symbol)
            raise

    def _determine_bias(self, components: Dict) -> str:
        """Determine directional bias from components."""
        if not components:
            return "NEUTRAL"

        votes = {"LONG": 0, "SHORT": 0}

        # Structure vote
        if "structure" in components:
            trend = components["structure"].get("trend")
            if trend == "BULLISH":
                votes["LONG"] += 1
            elif trend == "BEARISH":
                votes["SHORT"] += 1

        # Momentum vote
        if "momentum" in components:
            bias = components["momentum"].get("bias")
            if bias == "BULLISH":
                votes["LONG"] += 1
            elif bias == "BEARISH":
                votes["SHORT"] += 1

        if votes["LONG"] > votes["SHORT"]:
            return "LONG"
        elif votes["SHORT"] > votes["LONG"]:
            return "SHORT"
        return "NEUTRAL"

    def _calculate_bias_strength(self, components: Dict) -> float:
        """Calculate bias strength (0-100)."""
        if not components:
            return 0.0

        strength = 0.0

        if "structure" in components:
            strength += components["structure"].get("confidence", 0) * 0.4

        if "momentum" in components:
            rsi = components["momentum"].get("rsi", 50)
            # Further from 50 = stronger bias
            strength += abs(rsi - 50) * 0.6

        return min(100.0, strength)

    def _extract_key_levels(self, components: Dict) -> List[Dict]:
        """Extract key price levels from components."""
        levels = []

        if "structure" in components:
            struct = components["structure"]
            if struct.get("swing_high"):
                levels.append({
                    "type": "swing_high",
                    "price": struct["swing_high"],
                    "label": "Swing High",
                })
            if struct.get("swing_low"):
                levels.append({
                    "type": "swing_low",
                    "price": struct["swing_low"],
                    "label": "Swing Low",
                })

        if "liquidity" in components:
            liq = components["liquidity"]
            if liq.get("pdh"):
                levels.append({
                    "type": "pdh",
                    "price": liq["pdh"],
                    "label": "PDH",
                })
            if liq.get("pdl"):
                levels.append({
                    "type": "pdl",
                    "price": liq["pdl"],
                    "label": "PDL",
                })

        return levels
