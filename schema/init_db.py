import asyncio
import asyncpg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.db_config import db_config

async def init_database():
    """Initialize database schema."""
    try:
        # Connect to database
        conn = await asyncpg.connect(
            user=db_config.DB_USER,
            password=db_config.DB_PASSWORD,
            database=db_config.DB_DATABASE,
            host=db_config.DB_HOST,
            port=db_config.DB_PORT,
        )

        # Read schema file
        schema_path = Path(__file__).parent / "001_init_schema.sql"
        with open(schema_path, 'r') as f:
            schema = f.read()

        # Execute schema
        await conn.execute(schema)
        print("✓ Database schema initialized successfully")

        await conn.close()

    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(init_database())
