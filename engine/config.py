"""
Trading API Configuration
Load credentials from environment variables for security.
"""
import os
from pathlib import Path
from tradelocker import TLAPI

# Load .env file from project root
def _load_env():
    # Try to find .env relative to this file
    try:
        config_dir = Path(__file__).resolve().parent
        env_path = config_dir.parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            return True
    except Exception as e:
        pass
    return False

_load_env()

def get_api():
    """Get authenticated TradeLocker API instance."""
    return TLAPI(
        environment=os.getenv('TL_ENVIRONMENT', 'https://live.tradelocker.com'),
        username=os.getenv('TL_USERNAME'),
        password=os.getenv('TL_PASSWORD'),
        server=os.getenv('TL_SERVER', 'HEROFX'),
        account_id=int(os.getenv('TL_ACCOUNT_ID', '0')),
    )

def get_api_config():
    """Return config dict for manual API initialization."""
    return {
        'environment': os.getenv('TL_ENVIRONMENT', 'https://live.tradelocker.com'),
        'username': os.getenv('TL_USERNAME'),
        'password': os.getenv('TL_PASSWORD'),
        'server': os.getenv('TL_SERVER', 'HEROFX'),
        'account_id': int(os.getenv('TL_ACCOUNT_ID', '0')),
    }
