import redis
import os
import logging

REDIS_URL=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
client=redis.from_url(REDIS_URL,decode_responses=True)

logging.info(client.ping())