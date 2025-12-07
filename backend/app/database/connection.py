import asyncpg
from backend.app.config import settings


_db_pool = None


async def get_db():
    """
    Lazily create and return the asyncpg connection pool.
    """
    global _db_pool

    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=1,
            max_size=5
        )

    return _db_pool
