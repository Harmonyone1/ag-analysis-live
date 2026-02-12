"""Add trade management columns to positions table (idempotent)."""
import sys

sys.path.insert(0, ".")
from src.config import load_config
from src.database import init_db

config = load_config()

# Build database URL from config
db_url = (
    f"postgresql://{config.database.user}:{config.database.password}"
    f"@{config.database.host}:{config.database.port}/{config.database.name}"
)
db = init_db(db_url)

ALTER_STATEMENTS = [
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS original_stop_loss NUMERIC(18,8)",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS original_take_profit NUMERIC(18,8)",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS break_even_moved BOOLEAN DEFAULT FALSE",
    "ALTER TABLE positions ADD COLUMN IF NOT EXISTS trailing_active BOOLEAN DEFAULT FALSE",
]

from sqlalchemy import text

with db.session() as session:
    for stmt in ALTER_STATEMENTS:
        session.execute(text(stmt))
        print(f"  OK: {stmt}")
    session.commit()

print("\nMigration complete. Columns in positions table:")
with db.session() as session:
    result = session.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='positions' ORDER BY ordinal_position"
    ))
    for row in result:
        print(f"  - {row[0]}")
