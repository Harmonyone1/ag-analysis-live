"""TradeLocker Broker Adapter.

Implements the BrokerAdapter interface for TradeLocker API.
Uses the official tradelocker Python package.
"""

import logging
import warnings
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

# Suppress noisy "Missing type specification" warnings from tradelocker_api
warnings.filterwarnings("ignore", message="Missing type specification")

import pandas as pd
from tradelocker import TLAPI
from tradelocker.exceptions import TLAPIException, TLAPIOrderException

from .broker import (
    AccountInfo,
    BrokerAdapter,
    BrokerError,
    Candle,
    Instrument,
    Order,
    OrderRequest,
    Position,
    Quote,
)

logger = logging.getLogger(__name__)

# Timeframe mapping: internal -> TradeLocker resolution
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
    "1M": "1M",
    # Aliases
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
    "W1": "1W",
}


class TradeLockerAdapter(BrokerAdapter):
    """TradeLocker implementation of BrokerAdapter.

    Example usage:
        adapter = TradeLockerAdapter(
            environment="https://demo.tradelocker.com",
            email="user@example.com",
            password="password",
            server="server_name",
            acc_num=12345
        )
        adapter.connect()
        account = adapter.get_account()
    """

    def __init__(
        self,
        environment: str,
        email: str,
        password: str,
        server: str,
        acc_num: int = 0,
        account_id: int = 0,
        log_level: str = "info",
    ):
        """Initialize TradeLocker adapter.

        Args:
            environment: TradeLocker API URL (e.g., https://demo.tradelocker.com)
            email: TradeLocker account email
            password: TradeLocker account password
            server: TradeLocker server name
            acc_num: Account number to use
            account_id: Account ID (alternative to acc_num)
            log_level: Logging level for TradeLocker API
        """
        self._environment = environment
        self._email = email
        self._password = password
        self._server = server
        self._acc_num = acc_num
        self._account_id = account_id
        self._log_level = log_level
        self._api: Optional[TLAPI] = None
        self._instruments_cache: Optional[pd.DataFrame] = None

    def connect(self) -> bool:
        """Establish connection to TradeLocker."""
        try:
            self._api = TLAPI(
                environment=self._environment,
                username=self._email,
                password=self._password,
                server=self._server,
                acc_num=self._acc_num,
                account_id=self._account_id,
                log_level=self._log_level,
            )
            # Verify connection by fetching account info
            self._api.get_all_accounts()
            logger.info(
                f"Connected to TradeLocker: {self._api.account_name} "
                f"(acc_num: {self._api.acc_num})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TradeLocker: {e}")
            self._api = None
            return False

    def disconnect(self) -> None:
        """Disconnect from TradeLocker."""
        self._api = None
        self._instruments_cache = None
        logger.info("Disconnected from TradeLocker")

    def is_connected(self) -> bool:
        """Check if connected to TradeLocker."""
        if self._api is None:
            return False
        try:
            # Verify token is still valid
            self._api.get_access_token()
            return True
        except Exception:
            return False

    def _ensure_connected(self) -> None:
        """Ensure we have a valid connection."""
        if not self.is_connected():
            raise BrokerError("Not connected to TradeLocker")

    def get_account(self) -> AccountInfo:
        """Get account information."""
        self._ensure_connected()
        try:
            state = self._api.get_account_state()
            accounts = self._api.get_all_accounts()
            current = accounts[accounts["accNum"] == self._api.acc_num].iloc[0]

            return AccountInfo(
                account_id=self._api.account_id,
                account_name=str(current.get("name", "")),
                balance=Decimal(str(state.get("balance", 0))),
                equity=Decimal(str(state.get("equity", 0))),
                margin_used=Decimal(str(state.get("usedMargin", 0))),
                margin_available=Decimal(str(state.get("freeMargin", 0))),
                currency=str(state.get("currency", "USD")),
            )
        except Exception as e:
            raise BrokerError(f"Failed to get account info: {e}")

    def list_instruments(self) -> List[Instrument]:
        """Get all available instruments."""
        self._ensure_connected()
        try:
            if self._instruments_cache is None:
                self._instruments_cache = self._api.get_all_instruments()

            instruments = []
            for _, row in self._instruments_cache.iterrows():
                # Determine asset class from symbol
                symbol = str(row.get("name", ""))
                asset_class = self._determine_asset_class(symbol)

                instruments.append(
                    Instrument(
                        instrument_id=int(row.get("tradableInstrumentId", 0)),
                        symbol=symbol,
                        name=str(row.get("description", symbol)),
                        asset_class=asset_class,
                        pip_size=Decimal(str(row.get("pipSize", 0.0001))),
                        tick_size=Decimal(str(row.get("tickSize", 0.00001))),
                        contract_size=Decimal(str(row.get("lotSize", 100000))),
                        min_lot=Decimal(str(row.get("minOrderSize", 0.01))),
                        max_lot=Decimal(str(row.get("maxOrderSize", 100))),
                        base_currency=symbol[:3] if len(symbol) >= 6 else None,
                        quote_currency=symbol[3:6] if len(symbol) >= 6 else None,
                    )
                )
            return instruments
        except Exception as e:
            raise BrokerError(f"Failed to list instruments: {e}")

    def _determine_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol name."""
        symbol_upper = symbol.upper()

        # Common FX pairs
        fx_currencies = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
        if len(symbol) == 6:
            if symbol[:3] in fx_currencies and symbol[3:] in fx_currencies:
                return "FX"

        # Indices
        indices = ["US30", "US500", "US100", "GER40", "UK100", "FRA40", "ESP35", "JPN225"]
        if any(idx in symbol_upper for idx in indices):
            return "INDEX"

        # Commodities
        commodities = ["XAUUSD", "XAGUSD", "GOLD", "SILVER", "OIL", "BRENT", "WTI"]
        if any(comm in symbol_upper for comm in commodities):
            return "COMMODITY"

        # Crypto
        crypto = ["BTC", "ETH", "LTC", "XRP"]
        if any(c in symbol_upper for c in crypto):
            return "CRYPTO"

        return "OTHER"

    def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol."""
        instruments = self.list_instruments()
        for inst in instruments:
            if inst.symbol.upper() == symbol.upper():
                return inst
        return None

    def _get_instrument_id(self, symbol: str) -> int:
        """Get instrument ID from symbol."""
        self._ensure_connected()
        try:
            return self._api.get_instrument_id_from_symbol_name(symbol)
        except ValueError:
            raise BrokerError(f"Instrument not found: {symbol}")

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500
    ) -> List[Candle]:
        """Get historical candle data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M15, H1, etc)
            start: Optional start datetime
            end: Optional end datetime
            limit: Number of bars to fetch (default 500)

        Returns:
            List of Candle objects
        """
        self._ensure_connected()
        try:
            instrument_id = self._get_instrument_id(symbol)
            resolution = TIMEFRAME_MAP.get(timeframe, timeframe)

            # Convert timestamps to milliseconds
            start_ts = int(start.timestamp() * 1000) if start else 0
            end_ts = int(end.timestamp() * 1000) if end else 0

            # Build lookback based on timeframe and limit
            # TradeLocker API has a limit of 30000 rows, so we need to be careful
            lookback = ""
            if not start and not end:
                # Calculate approximate days needed based on timeframe
                tf_minutes = {
                    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                    "1h": 60, "1H": 60, "4h": 240, "4H": 240,
                    "1D": 1440, "1W": 10080
                }
                minutes_per_bar = tf_minutes.get(resolution, 15)
                total_minutes = limit * minutes_per_bar
                days_needed = max(1, int(total_minutes / 1440) + 1)  # Round up
                # Cap at 20 days to stay under API limit
                days_needed = min(days_needed, 300)
                lookback = f"{days_needed}D"

            df = self._api.get_price_history(
                instrument_id=instrument_id,
                resolution=resolution,
                lookback_period=lookback,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
            )

            candles = []
            for _, row in df.iterrows():
                candles.append(
                    Candle(
                        timestamp=datetime.fromtimestamp(row["t"] / 1000),
                        open=Decimal(str(row["o"])),
                        high=Decimal(str(row["h"])),
                        low=Decimal(str(row["l"])),
                        close=Decimal(str(row["c"])),
                        volume=Decimal(str(row.get("v", 0))),
                    )
                )
            # Sort and limit the result
            candles = sorted(candles, key=lambda c: c.timestamp)
            return candles[-limit:] if len(candles) > limit else candles
        except Exception as e:
            raise BrokerError(f"Failed to get candles for {symbol}: {e}")

    def get_quote(self, symbol: str) -> Quote:
        """Get current price quote."""
        self._ensure_connected()
        try:
            instrument_id = self._get_instrument_id(symbol)
            quotes = self._api.get_quotes(instrument_id)

            bid = Decimal(str(quotes.get("bp", 0)))
            ask = Decimal(str(quotes.get("ap", 0)))

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread=ask - bid,
                timestamp=datetime.now(),
            )
        except Exception as e:
            raise BrokerError(f"Failed to get quote for {symbol}: {e}")

    def place_order(self, request: OrderRequest) -> Order:
        """Place a new order."""
        self._ensure_connected()
        try:
            # Determine validity based on order type
            validity = "IOC" if request.order_type == "market" else "GTC"

            order_id = self._api.create_order(
                instrument_id=request.instrument_id,
                quantity=float(request.quantity),
                side=request.side,
                price=float(request.price) if request.price else None,
                type_=request.order_type,
                validity=validity,
                stop_loss=float(request.stop_loss) if request.stop_loss else None,
                stop_loss_type=request.stop_loss_type,
                take_profit=float(request.take_profit) if request.take_profit else None,
                take_profit_type=request.take_profit_type,
                stop_price=float(request.stop_price) if request.stop_price else None,
                strategy_id=request.strategy_id,
            )

            if order_id is None:
                raise BrokerError("Order placement returned None")

            logger.info(f"Order placed: {order_id} for {request.symbol}")

            return Order(
                order_id=order_id,
                instrument_id=request.instrument_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                filled_quantity=Decimal("0"),
                price=request.price,
                order_type=request.order_type,
                status="pending",
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except TLAPIOrderException as e:
            raise BrokerError(f"Order rejected: {e}")
        except TLAPIException as e:
            raise BrokerError(f"Order failed: {e}")
        except Exception as e:
            raise BrokerError(f"Failed to place order: {e}")

    def modify_order(self, order_id: int, modifications: dict) -> Order:
        """Modify an existing order."""
        self._ensure_connected()
        try:
            success = self._api.modify_order(order_id, modifications)
            if not success:
                raise BrokerError(f"Failed to modify order {order_id}")

            # Return updated order
            orders = self.list_orders(include_history=True)
            for order in orders:
                if order.order_id == order_id:
                    return order
            raise BrokerError(f"Order {order_id} not found after modification")
        except Exception as e:
            raise BrokerError(f"Failed to modify order: {e}")

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order."""
        self._ensure_connected()
        try:
            return self._api.delete_order(order_id)
        except Exception as e:
            raise BrokerError(f"Failed to cancel order: {e}")

    def list_orders(self, include_history: bool = False) -> List[Order]:
        """Get all orders."""
        self._ensure_connected()
        try:
            df = self._api.get_all_orders(history=include_history)
            orders = []

            for _, row in df.iterrows():
                orders.append(
                    Order(
                        order_id=int(row.get("id", 0)),
                        instrument_id=int(row.get("tradableInstrumentId", 0)),
                        symbol=self._api.get_symbol_name_from_instrument_id(
                            int(row.get("tradableInstrumentId", 0))
                        ),
                        side=str(row.get("side", "")),
                        quantity=Decimal(str(row.get("qty", 0))),
                        filled_quantity=Decimal(str(row.get("filledQty", 0))),
                        price=Decimal(str(row.get("price", 0))) if row.get("price") else None,
                        order_type=str(row.get("type", "")),
                        status=str(row.get("status", "")),
                        stop_loss=Decimal(str(row.get("stopLoss", 0))) if row.get("stopLoss") else None,
                        take_profit=Decimal(str(row.get("takeProfit", 0))) if row.get("takeProfit") else None,
                        created_at=datetime.now(),  # TL doesn't provide this
                        updated_at=datetime.now(),
                    )
                )
            return orders
        except Exception as e:
            raise BrokerError(f"Failed to list orders: {e}")

    def list_positions(self) -> List[Position]:
        """Get all open positions."""
        self._ensure_connected()
        try:
            df = self._api.get_all_positions()
            positions = []

            # Fetch open orders to resolve SL/TP prices from linked order IDs
            sl_tp_prices = {}
            try:
                orders_df = self._api.get_all_orders(history=False)
                if orders_df is not None and len(orders_df) > 0:
                    for _, orow in orders_df.iterrows():
                        oid = int(orow.get("id", 0))
                        otype = str(orow.get("type", "")).lower()
                        if otype == "stop":
                            price = Decimal(str(orow.get("stopPrice", 0)))
                            if price > 0:
                                sl_tp_prices[oid] = price
                        elif otype == "limit":
                            price = Decimal(str(orow.get("price", 0)))
                            if price > 0:
                                sl_tp_prices[oid] = price
            except Exception:
                pass  # Proceed without SL/TP if orders fetch fails

            for _, row in df.iterrows():
                instrument_id = int(row.get("tradableInstrumentId", 0))
                symbol = self._api.get_symbol_name_from_instrument_id(instrument_id)

                # Get current price for unrealized PnL
                try:
                    quote = self.get_quote(symbol)
                    current_price = quote.bid if row.get("side") == "buy" else quote.ask
                except Exception:
                    current_price = Decimal(str(row.get("avgPrice", 0)))

                # Resolve SL/TP from linked order IDs
                sl_id = int(row.get("stopLossId", 0))
                tp_id = int(row.get("takeProfitId", 0))
                stop_loss = sl_tp_prices.get(sl_id)
                take_profit = sl_tp_prices.get(tp_id)

                # Parse open time from epoch ms
                open_date_ms = int(row.get("openDate", 0))
                open_time = datetime.fromtimestamp(open_date_ms / 1000) if open_date_ms else datetime.now()

                positions.append(
                    Position(
                        position_id=int(row.get("id", 0)),
                        instrument_id=instrument_id,
                        symbol=symbol,
                        side=str(row.get("side", "")),
                        quantity=Decimal(str(row.get("qty", 0))),
                        avg_price=Decimal(str(row.get("avgPrice", 0))),
                        current_price=current_price,
                        unrealized_pnl=Decimal(str(row.get("unrealizedPl", 0))),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        open_time=open_time,
                    )
                )
            return positions
        except Exception as e:
            raise BrokerError(f"Failed to list positions: {e}")

    def close_position(
        self,
        position_id: int,
        quantity: Optional[Decimal] = None
    ) -> bool:
        """Close a position."""
        self._ensure_connected()
        try:
            close_qty = float(quantity) if quantity else 0
            return self._api.close_position(
                position_id=position_id,
                close_quantity=close_qty
            )
        except Exception as e:
            raise BrokerError(f"Failed to close position: {e}")

    def modify_position(
        self,
        position_id: int,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> bool:
        """Modify position SL/TP."""
        self._ensure_connected()
        try:
            modifications = {}
            if stop_loss is not None:
                modifications["stopLoss"] = float(stop_loss)
            if take_profit is not None:
                modifications["takeProfit"] = float(take_profit)

            return self._api.modify_position(position_id, modifications)
        except Exception as e:
            raise BrokerError(f"Failed to modify position: {e}")

    def get_positions(self) -> List[Position]:
        """Alias for list_positions for compatibility."""
        return self.list_positions()
