# AG ANALYZER

AI-driven trading analyzer and trading bot built for TradeLocker.

## Overview

A production-grade system that:
1. Ingests market data (FX + indices)
2. Runs multi-factor analysis (strength, structure, liquidity, momentum)
3. Produces ranked trade candidates with explainable reason codes
4. Uses AI as a decision gate + weight tuner
5. Executes trades automatically on TradeLocker
6. Logs, audits, monitors, and learns from outcomes

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Next.js   │────▶│   FastAPI   │────▶│   Engine    │
│     UI      │◀────│   Backend   │◀────│  (Python)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Supabase   │     │ TradeLocker │
                    │  (Postgres) │     │     API     │
                    └─────────────┘     └─────────────┘
```

## Components

- **engine/** - Python trading engine (analysis, scoring, AI gate, execution)
- **backend/** - FastAPI REST API + WebSocket server
- **ui/** - Next.js trader dashboard
- **supabase/** - Database migrations and configuration

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- TradeLocker account

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Harmonyone1/AG-ANALYZER.git
cd AG-ANALYZER
```

2. Copy environment template:
```bash
cp .env.example .env
```

3. Configure your environment variables in `.env`

4. Start with Docker Compose:
```bash
docker-compose up -d
```

5. Run database migrations:
```bash
cd supabase && supabase db push
```

## Environment Variables

See `.env.example` for all required configuration options.

## License

Private - All rights reserved
