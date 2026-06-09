import asyncpg
from contextlib import asynccontextmanager


class DBService:
    def __init__(self, db_config):
        self.db_config = db_config
        self.pool = None

    async def init_pool(self):
        """Initialize postgres connection pool."""
        self.pool = await asyncpg.create_pool(
            user=self.db_config.DB_USER,
            password=self.db_config.DB_PASSWORD,
            database=self.db_config.DB_DATABASE,
            host=self.db_config.DB_HOST,
            port=self.db_config.DB_PORT,
            min_size=5,
            max_size=20,
            command_timeout=60,
        )

    async def close_pool(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool."""
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)

    async def query(self, sql_query: str, *args):
        """Fetch multiple rows."""
        async with self.get_connection() as conn:
            return await conn.fetch(sql_query, *args)

    async def query_one(self, sql_query: str, *args):
        """Fetch single row."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(sql_query, *args)

    async def query_val(self, sql_query: str, *args):
        """Fetch single value."""
        async with self.get_connection() as conn:
            return await conn.fetchval(sql_query, *args)

    async def execute(self, sql_query: str, *args):
        """Execute query without returning rows."""
        async with self.get_connection() as conn:
            return await conn.execute(sql_query, *args)

    async def execute_many(self, sql_query: str, args_list: list):
        """Execute multiple queries."""
        async with self.get_connection() as conn:
            return await conn.executemany(sql_query, args_list)
