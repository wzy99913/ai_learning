import json
from typing import Any

import redis.asyncio as redis

from app.core.config import Settings


class ChatHistoryStore:
    """使用 Redis 保存多轮聊天记录。"""

    def __init__(self, settings: Settings):
        self._redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self._ttl = settings.chat_history_ttl
        self._key_prefix = "ai_learning:chat_history:"

    def _build_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """根据 session_id 读取聊天记录。"""
        key = self._build_key(session_id)
        data = await self._redis.get(key)
        if not data:
            return []
        return json.loads(data)

    async def save(
        self,
        session_id: str,
        history: list[dict[str, Any]],
    ) -> None:
        """保存聊天记录，并设置过期时间。"""
        key = self._build_key(session_id)
        await self._redis.setex(
            key,
            self._ttl,
            json.dumps(history, ensure_ascii=False),
        )

    async def clear(self, session_id: str) -> None:
        """删除某个会话的聊天记录。"""
        key = self._build_key(session_id)
        await self._redis.delete(key)

    async def close(self) -> None:
        """关闭 Redis 连接。"""
        await self._redis.aclose()
