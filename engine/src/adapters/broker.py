"""Abstract Broker Adapter Interface.

All broker integrations must implement this interface to ensure
the trading engine remains decoupled from specific broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from decimal import Decimal


@dataclass
class AccountInfo:
    """Account information from broker."""
    account_id: int
    account_name: str
    balance: Decimal
    equity: Decimal
    margin_used: Decimal
    margin_available: Decimal
    currency: str


@dataclass
class Instrument:
    """Tradeable instrument details."""
    instrument_id: int
    symbol: str
    name: str
    asset_class: str  # FX, INDEX, COMMODITY, CRYPTO
    pip_size: Decimal
    tick_size: Decimal
    contract_size: Decimal
    min_lot: Decimal
    max_lot: Decimal
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class Quote:
    """Current price quote."""
    symbol: str
    bid: Decimal
    ask: Decimal
    spread: Decimal
    timestamp: datetime


@dataclass
class OrderRequest:
    """Order placement request."""
    symbol: str
    instrument_id: int
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    order_type: str  # 'market', 'limit', 'stop'
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    stop_loss_type: Optional[str] = None  # 'absolute' or 'offset'
    take_profit: Optional[Decimal] = None
    take_profit_type: Optional[str] = None  # 'absolute' or 'offset'
    validity: str = 'GTC'
    strategy_id: Optional[str] = None


@dataclass
class Order:
    """Order details."""
    order_id: int
    instrument_id: int
    symbol: str
    side: str
    quantity: Decimal
    filled_quantity: Decimal
    price: Optional[Decimal]
    order_type: str
    status: str
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    created_at: datetime
    updated_at: datetime


@dataclass
class Position:
    """Open position details."""
    position_id: int
    instrument_id: int
    symbol: str
    side: str
    quantity: Decimal
    avg_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    open_time: datetime


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters.

    The trading engine interacts ONLY through this interface,
    ensuring broker implementation details are isolated.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with current account state.
        """
        pass

    @abstractmethod
    def list_instruments(self) -> List[Instrument]:
        """Get all available instruments.

        Returns:
            List of tradeable instruments.
        """
        pass

    @abstractmethod
    def get_instrument(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol.

        Args:
            symbol: Instrument symbol (e.g., 'EURUSD')

        Returns:
            Instrument details or None if not found.
        """
        pass

    @abstractmethod
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
            symbol: Instrument symbol
            timeframe: Candle timeframe (e.g., '15m', '1h', '4h', '1d')
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of candles

        Returns:
            List of candles sorted by timestamp ascending.
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Get current price quote.

        Args:
            symbol: Instrument symbol

        Returns:
            Current bid/ask quote.
        """
        pass

    @abstractmethod
    def place_order(self, request: OrderRequest) -> Order:
        """Place a new order.

        Args:
            request: Order request details

        Returns:
            Created order.

        Raises:
            BrokerError: If order placement fails.
        """
        pass

    @abstractmethod
    def modify_order(self, order_id: int, modifications: dict) -> Order:
        """Modify an existing order.

        Args:
            order_id: Order to modify
            modifications: Fields to update

        Returns:
            Updated order.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled successfully.
        """
        pass

    @abstractmethod
    def list_orders(self, include_history: bool = False) -> List[Order]:
        """Get all orders.

        Args:
            include_history: Include filled/cancelled orders

        Returns:
            List of orders.
        """
        pass

    @abstractmethod
    def list_positions(self) -> List[Position]:
        """Get all open positions.

        Returns:
            List of open positions.
        """
        pass

    @abstractmethod
    def close_position(
        self,
        position_id: int,
        quantity: Optional[Decimal] = None
    ) -> bool:
        """Close a position.

        Args:
            position_id: Position to close
            quantity: Partial close quantity (None = full close)

        Returns:
            True if close order placed successfully.
        """
        pass

    @abstractmethod
    def modify_position(
        self,
        position_id: int,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> bool:
        """Modify position SL/TP.

        Args:
            position_id: Position to modify
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            True if modified successfully.
        """
        pass


class BrokerError(Exception):
    """Exception raised for broker-related errors."""
    pass
