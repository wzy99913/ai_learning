"""Multi-turn agent that can choose the knowledge-base retrieval tool."""

import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings
from app.services.chat_history import ChatHistoryStore
from app.services.rag_service import RAGService


class AgentService:
    def __init__(
        self,
        settings: Settings,
        rag_service: RAGService,
        history_store: ChatHistoryStore,
    ):
        self._settings = settings
        self._rag_service = rag_service
        self._history_store = history_store

    async def chat(self, message: str, session_id: str) -> str:
        """Answer a normal question or use a tool when knowledge is needed."""
        if not message.strip():
            raise ValueError("消息不能为空")

        history = await self._history_store.load(session_id)
        messages: list[dict[str, Any]] = [self._system_message()]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        client = self._create_client()
        for _ in range(5):
            response = await client.chat.completions.create(
                model=self._settings.deepseek_model,
                messages=messages,
                tools=self._tool_schemas(),
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                answer = assistant_message.content or "模型没有返回有效回答。"
                await self._remember(session_id, history, message, answer)
                return answer

            messages.append(assistant_message.model_dump(exclude_none=True))
            for tool_call in assistant_message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result = self._execute_tool(tool_call.function.name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return "Agent 达到最大工具调用次数。"

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "calculate":
            return self._calculate(**arguments)
        if name == "count_text":
            return len(arguments["text"])
        if name == "search_knowledge_base":
            chunks = self._rag_service.retrieve(arguments["question"], top_k=3)
            return [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "distance": chunk.distance,
                }
                for chunk in chunks
            ]
        return "未知工具"

    @staticmethod
    def _calculate(a: float, b: float, operation: str) -> float | str:
        operations = {
            "add": lambda: a + b,
            "subtract": lambda: a - b,
            "multiply": lambda: a * b,
            "divide": lambda: "除数不能为零" if b == 0 else a / b,
        }
        if operation not in operations:
            return "不支持的运算"
        return operations[operation]()

    def _create_client(self) -> AsyncOpenAI:
        if not self._settings.deepseek_api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        return AsyncOpenAI(
            api_key=self._settings.deepseek_api_key,
            base_url=self._settings.deepseek_base_url,
        )

    async def _remember(
        self,
        session_id: str,
        history: list[dict[str, Any]],
        question: str,
        answer: str,
    ) -> None:
        updated_history = [
            *history,
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        await self._history_store.save(session_id, updated_history[-20:])

    @staticmethod
    def _system_message() -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                "你是校园智能助手。涉及南京理工大学学生手册、考试、宿舍、"
                "奖学金或校园制度的问题，必须调用 search_knowledge_base。"
                "普通聊天无需查询。使用检索结果时不得编造，并在答案末尾列出来源。"
            ),
        }

    @staticmethod
    def _tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "计算两个数字，支持加减乘除。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                        },
                        "required": ["a", "b", "operation"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "count_text",
                    "description": "统计一段文本的字符数量。",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "查询南理工校园知识库，用于学生手册和校内制度问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            },
        ]
