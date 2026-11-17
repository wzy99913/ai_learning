"""RAG question-answering workflow built on the vector-store adapter."""

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.core.config import Settings
from app.rag.embedding import embed_query
from app.rag.vector_store import ChromaVectorStore, RetrievedChunk


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk]


class RAGService:
    def __init__(self, settings: Settings, vector_store: ChromaVectorStore):
        self._settings = settings
        self._vector_store = vector_store

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Vectorize a question and retrieve relevant chunks."""
        if not question.strip():
            raise ValueError("问题不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")

        return self._vector_store.search(embed_query(question), top_k)

    async def answer(self, question: str, top_k: int = 3) -> AnswerResult:
        """Retrieve context and ask DeepSeek for a source-grounded answer."""
        chunks = self.retrieve(question, top_k)
        prompt = self._build_prompt(question, chunks)
        client = self._create_client()

        response = await client.chat.completions.create(
            model=self._settings.deepseek_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的南京理工大学校园知识库问答助手。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content or "模型没有返回有效回答。"
        return AnswerResult(answer=answer, sources=chunks)

    def _create_client(self) -> AsyncOpenAI:
        if not self._settings.deepseek_api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")

        return AsyncOpenAI(
            api_key=self._settings.deepseek_api_key,
            base_url=self._settings.deepseek_base_url,
        )

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        if chunks:
            context = "\n\n".join(
                "[资料 {number}]\n来源：{source}\nChunk 编号：{chunk_index}\n内容：{text}".format(
                    number=index,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                )
                for index, chunk in enumerate(chunks, start=1)
            )
        else:
            context = "（知识库中没有检索到相关资料）"

        return f"""请严格根据以下资料回答问题，不要编造。
如果资料不足，请明确回答“根据当前知识库无法确定”。
回答末尾写出“来源：文档名，Chunk 编号”。

资料：
{context}

问题：{question}"""
