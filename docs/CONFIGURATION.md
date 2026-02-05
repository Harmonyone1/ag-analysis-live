# Configuration Guide

## Environment Variables

### TradeLocker Connection

| Variable | Description | Example |
|----------|-------------|---------|
| `TL_EMAIL` | TradeLocker account email | `user@example.com` |
| `TL_PASSWORD` | TradeLocker password | `your-password` |
| `TL_SERVER` | Broker server name | `HEROFX` |
| `TL_ENVIRONMENT` | API endpoint | `https://live.tradelocker.com` |
| `TL_ACC_NUM` | Account number to use | `1` |

### Database

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |

### Bot Mode

| Variable | Description | Values |
|----------|-------------|--------|
| `BOT_MODE` | Trading mode | `live`, `paper`, `dry_run` |
| `TRADING_ENABLED` | Enable trade execution | `true`, `false` |

## Bot Configuration (config.py)

### Risk Settings

```python
risk:
  max_risk_per_trade: 0.01      # 1% per trade
  max_daily_loss: 0.03          # 3% daily loss limit
  max_positions: 5              # Max concurrent positions
  max_per_currency: 2           # Max positions per base currency
```

### AI Gate Thresholds

```python
ai:
  min_confluence_score: 65      # Minimum score to consider
  min_probability: 0.55         # P(win) threshold
  min_expected_r: 0.02          # E[R] threshold
  model_version: "v3"           # Model version to use
```

### Execution Settings

```python
execution:
  default_lot_size: 0.10        # Default position size
  min_stop_pips: 20             # Minimum stop distance
  min_rr_ratio: 1.5             # Minimum risk:reward
  cooldown_hours: 4             # Hours to wait after loss
```

### Session Settings

The bot can be configured to trade only during specific sessions:

```python
sessions:
  enabled: false                # Session filtering (disabled - AI handles this)
  london_start: 7               # UTC hour
  london_end: 16
  ny_start: 12
  ny_end: 21
```

**Note:** Session filtering is currently disabled because the AI model includes time features (hour_sin/hour_cos) that make it session-aware.

## Modifying Symbols

To add or remove trading symbols, edit `engine/src/main.py`:

```python
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY",
    # Add more symbols here
]
```

**Warning:** Only add symbols the model was trained on. Adding crypto or indices without retraining will likely produce poor results.

## Model Files

Located in `models/` directory:

| File | Description |
|------|-------------|
| `gate_model_v3.0.0-*.pkl` | Main AI gate model |
| `classifier_v1.0.0-*.pkl` | Win/loss classifier |
| `regressor_v1.0.0-*.pkl` | R-multiple regressor |
| `calibrator_v1.0.0-*.pkl` | Probability calibrator |

The bot loads the latest model version on startup.

## Docker Compose Configuration

`docker-compose.yml` settings:

```yaml
services:
  engine:
    environment:
      - BOT_MODE=live
      - TRADING_ENABLED=true
    restart: unless-stopped

  backend:
    ports:
      - "8000:8000"

  db:
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

## Adjusting Trade Frequency

The bot scans every ~150 seconds by default. To change:

In `engine/src/main.py`, modify:
```python
await asyncio.sleep(150)  # Seconds between scans
```

## Logging Verbosity

Set log level in `engine/src/main.py`:

```python
logging.basicConfig(level=logging.INFO)  # or DEBUG for more detail
```
