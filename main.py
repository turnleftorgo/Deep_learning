import asyncio
import fastapi
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# 初始化 FastAPI 应用
app = FastAPI(title="Medical Pathology RAG Retrieval API", version="1.0.0")

# ==========================================
# 1. 异步 Mock 函数 (模拟耗时的 I/O 操作)
# ==========================================
async def mock_embed_query(text: str) -> List[float]:
    """模拟调用 Embedding 模型（如 Sentence-Transformer）"""
    await asyncio.sleep(0.1)  # 必须用 asyncio.sleep 模拟异步 I/O，千万别用 time.sleep
    return [0.1] * 768        # 假设返回 768 维向量

async def mock_vector_search(embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    """模拟查询向量数据库（如 FAISS/Milvus）"""
    await asyncio.sleep(0.2) 
    # 模拟生成得分递减的数据，故意造一些低于 0.3 的分数来测试兜底逻辑
    return [
        {
            "chunk_id": i, 
            "text": f"模拟病理报告片段_{i} (相关查询: ...)", 
            "score": round(0.9 - (i * 0.15), 2)
        }
        for i in range(top_k)
    ]

# ==========================================
# 2. Pydantic 数据模型 (严格的边界校验)
# ==========================================
class RetrieveRequest(BaseModel):
    # 强制必填，并添加描述
    query: str = Field(..., description="用户的自然语言医学查询")
    # 设定默认值为5，限制范围 1~20，防止恶意请求打爆数据库
    top_k: int = Field(default=5, ge=1, le=20, description="返回的最大片段数量")

    @field_validator('query')
    def query_must_not_be_empty(cls, value: str):
        """核心防御：防止纯空格或空字符串绕过请求"""
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("查询字符串不能为空或纯空格")
        return cleaned_value

class RetrieveResponse(BaseModel):
    status: int
    message: str
    data: List[Dict[str, Any]]

# ==========================================
# 3. 核心路由 (业务逻辑流转)
# ==========================================
@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(request: RetrieveRequest):
    try:
        # Step 1: 文本向量化 (必须加 await)
        embedding = await mock_embed_query(request.query)

        # Step 2: 向量数据库检索
        raw_results = await mock_vector_search(embedding, request.top_k)

        # Step 3: 业务后处理 - 阈值过滤 (拦截幻觉源头)
        # 只保留相似度大于等于 0.3 的有效片段
        THRESHOLD = 0.3
        valid_chunks = [item for item in raw_results if item["score"] >= THRESHOLD]

        # Step 4: 业务兜底逻辑
        if not valid_chunks:
            # 状态码依然是 200 (接口调用成功)，但业务上提示查无此物
            return RetrieveResponse(
                status=200,
                message="未检索到高度相关的病理报告片段，请尝试更换医学术语或提供更具体的查询。",
                data=[]
            )

        # Step 5: 正常返回
        return RetrieveResponse(
            status=200,
            message="检索成功",
            data=valid_chunks
        )

    except Exception as e:
        # 全局异常捕获，防止底层报错直接抛给前端，造成信息泄露
        # 实际开发中这里应该接入 Logger 记录真实的 e 信息
        raise HTTPException(status_code=500, detail=f"检索服务内部处理异常: {str(e)}")

# ==========================================
# 启动说明：
# 终端运行: uvicorn main:app --reload
# ==========================================