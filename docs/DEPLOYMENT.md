# Deployment Guide

## Prerequisites

- Docker & Docker Compose
- TradeLocker account with API access
- Server with minimum 2GB RAM

## Server Setup (DigitalOcean Droplet)

### 1. Create Droplet
- Ubuntu 22.04 LTS
- 2GB RAM / 1 vCPU minimum
- Enable monitoring

### 2. Install Docker
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### 3. Clone Repository
```bash
git clone https://github.com/Harmonyone1/ag-analysis-live.git
cd ag-analysis-live
```

### 4. Configure Environment
```bash
cp .env.example .env
nano .env
```

Required environment variables:
```env
# TradeLocker Credentials
TL_EMAIL=your-email@example.com
TL_PASSWORD=your-password
TL_SERVER=HEROFX
TL_ENVIRONMENT=https://live.tradelocker.com
TL_ACC_NUM=1

# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/ag_analyzer

# Bot Configuration
BOT_MODE=live
TRADING_ENABLED=true
```

### 5. Deploy with Docker Compose
```bash
docker-compose up -d
```

This starts:
- `ag_analyzer_engine` - Main trading bot
- `ag_analyzer_backend` - REST API (port 8000)
- `ag_analyzer_db` - PostgreSQL database

### 6. Verify Deployment
```bash
# Check containers are running
docker ps

# View bot logs
docker logs -f ag_analyzer_engine

# Check API health
curl http://localhost:8000/health
```

## Updating the Bot

```bash
cd ag-analysis-live
git pull origin master

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

## Monitoring

### View Real-time Logs
```bash
docker logs -f ag_analyzer_engine 2>&1 | grep -E "\[EXEC\]|\[AI_GATE\]"
```

### Check Account Status
```bash
docker exec ag_analyzer_engine python3 -c "
from tradelocker import TLAPI
import os
tl = TLAPI(
    environment=os.environ.get('TL_ENVIRONMENT'),
    username=os.environ.get('TL_EMAIL'),
    password=os.environ.get('TL_PASSWORD'),
    server=os.environ.get('TL_SERVER'),
    acc_num=int(os.environ.get('TL_ACC_NUM', 1))
)
state = tl.get_account_state()
print(f'Balance: \${state[\"balance\"]:.2f}')
print(f'Today P&L: \${state[\"todayNet\"]:.2f}')
print(f'Open Positions: {state[\"positionsCount\"]}')
"
```

## Stopping the Bot

```bash
# Graceful stop
docker-compose down

# Force stop (if needed)
docker kill ag_analyzer_engine
```

## Troubleshooting

### Bot not trading
1. Check logs for `[EXEC]` messages
2. Verify `BOT_MODE=live` and `TRADING_ENABLED=true`
3. Confirm TradeLocker credentials are correct
4. Check if AI gate is rejecting all candidates (P(win) < 55%)

### Connection errors
1. Verify TradeLocker API is accessible
2. Check server firewall rules
3. Ensure TL_ENVIRONMENT URL is correct

### Database issues
```bash
# Reset database
docker-compose down -v
docker-compose up -d
```
