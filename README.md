# AI Learning：RAG + Agent 学习项目

这是一个面向 AI 应用开发/后端实习的学习项目。它支持上传 PDF 或 TXT 建立知识库，基于 ChromaDB 的 RAG 问答，以及可按需调用知识库工具的多轮 Agent。

## 项目能力

- `POST /knowledge/upload`：上传 PDF/TXT，自动完成解析、切分、向量化和入库。
- `POST /knowledge/query`：固定 RAG 流程，返回带来源的知识库回答。
- `POST /chat`：支持普通聊天、计算器、字符统计和校园知识库检索的 Agent。
- `GET /health`：服务健康检查。

## 目录和职责

```text
ai_learning/
├── main.py                         # 组合依赖、创建 FastAPI 应用
├── app/
│   ├── core/config.py               # 读取 .env，集中管理路径、模型和集合配置
│   ├── api/
│   │   ├── schemas.py               # HTTP 请求/响应数据结构
│   │   └── routes.py                # HTTP 路由、参数校验、异常转 HTTP 状态码
│   ├── rag/
│   │   ├── document_loader.py       # PDF/TXT → 文本
│   │   ├── text_splitter.py         # 文本 → Chunk
│   │   ├── embedding.py             # 文本/问题 → 向量（按需加载模型）
│   │   └── vector_store.py          # ChromaDB 的写入、替换、检索
│   └── services/
│       ├── ingestion_service.py     # 读取 → 切分 → 向量化 → 入库
│       ├── rag_service.py           # 检索 → Prompt → DeepSeek 回答
│       └── agent_service.py         # 多轮对话和工具调用
├── knowledge_base/uploads/          # 上传成功后保存的原始文件（不提交 Git）
├── .env                             # 本机密钥与配置（不提交 Git）
├── .env.example                     # 无密钥的配置模板
└── requirements.txt                 # 运行依赖
```

## 调用关系

```text
上传接口
  routes.py → IngestionService → document_loader → text_splitter
            → embedding → ChromaVectorStore

知识库问答接口
  routes.py → RAGService → embedding → ChromaVectorStore → DeepSeek

Agent 对话接口
  routes.py → AgentService
              └─ 需要校园资料时 → RAGService.retrieve()
                                    → embedding → ChromaVectorStore
```

依赖方向始终从外层指向内层：`api → services → rag`。`rag` 不依赖 FastAPI，便于单独测试和复用。

## 环境配置

项目复用旧项目的虚拟环境：

```powershell
E:\AI学习\LLM_api_learning\.venv\Scripts\python.exe --version
```

Python 版本为 3.11。现有虚拟环境已经具备本项目所需依赖；若在新机器重新安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

创建本地配置：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。默认的 `CHROMA_DB_PATH` 指向旧项目已有的 ChromaDB，因此不会重新初始化原有 `njust_knowledge` 集合。

聊天记录使用 Redis 持久化。首次运行前启动项目专用容器：

```powershell
docker run -d `
  --name ai-learning-redis `
  -p 6379:6379 `
  -v ai-learning-redis-data:/data `
  redis:7-alpine `
  redis-server --appendonly yes
```

以后只需执行 `docker start ai-learning-redis`。可用下面的命令确认服务状态：

```powershell
docker exec ai-learning-redis redis-cli ping
```

预期返回 `PONG`。`.env` 中的 `REDIS_URL` 默认为 `redis://localhost:6379/0`，聊天记录默认保留 86400 秒。

## 启动

从项目根目录执行：

```powershell
E:\AI学习\LLM_api_learning\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

打开 Swagger：<http://127.0.0.1:8000/docs>

## 测试建议

1. 访问 `GET /health`，应返回 `status: ok`。
2. 向 `POST /knowledge/upload` 上传 PDF 或 TXT。
3. 用 `POST /knowledge/query` 测试固定 RAG 问答。
4. 用 `POST /chat` 先测试普通聊天，再测试校园制度问题与同一个 `session_id` 的追问。

## 关键设计说明

- 同名文件再次上传时，`ChromaVectorStore.replace_document()` 只替换该来源的 Chunk，避免重复入库；不会清空整个集合。
- Embedding 模型采用延迟加载，服务启动时不会立刻加载模型。
- API Key 只从 `.env` 读取，代码和 `.env.example` 不包含真实密钥。
- `knowledge_base/uploads/` 与 `.env` 已被 `.gitignore` 排除。
