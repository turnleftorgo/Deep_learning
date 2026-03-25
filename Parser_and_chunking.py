import os
# 如果没有安装，需要 pip install langchain langchain-community pymupdf
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_medical_pdfs_to_chunks(pdf_dir_path: str):
    """
    将指定目录下的所有 PDF 批量解析并切分为文本块
    """
    print(f"📂 开始扫描目录: {pdf_dir_path} 中的 PDF 文件...")

    # ==========================================
    # Step 1: 批量加载 PDF (Parsing)
    # ==========================================
    # DirectoryLoader 是 LangChain 处理文件夹的利器
    # 我们指定只读取 .pdf 文件，并使用速度最快的 PyMuPDFLoader 作为底层解析器
    loader = DirectoryLoader(
        pdf_dir_path,
        glob="**/*.pdf",          # 支持读取子文件夹下的 PDF
        loader_cls=PyMuPDFLoader,
        show_progress=True        # 在终端显示进度条，处理 10 个以上文件时非常有用
    )
    
    # 执行加载：这里会将 10 个 PDF 变成一个 Document 对象列表
    # 通常每一个 Document 对应 PDF 的一页
    documents = loader.load()
    print(f"✅ 成功加载了 {len(documents)} 页文档。")

    # ==========================================
    # Step 2: 定义切分策略 (Chunking Strategy)
    # ==========================================
    # 医疗文书专业词汇多，切分策略极其关键
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # 每个 Chunk 约 500 字，适合送给大模型做上下文
        chunk_overlap=50,     # 黄金参数：前后保留 50 字重叠，防止“鳞状/细胞癌”被生硬切断
        separators=["\n\n", "\n", "。", "；", " ", ""] # 优先按段落和中文句号切分，保全句子完整性
    )

    # ==========================================
    # Step 3: 执行切分
    # ==========================================
    chunks = text_splitter.split_documents(documents)
    print(f"🔪 10 个 PDF 最终被切分成了 {len(chunks)} 个文本块 (Chunks)。\n")

    # ==========================================
    # 🕵️‍♂️ 面试高光：展示对 Metadata 的控制力
    # ==========================================
    if chunks:
        sample_chunk = chunks[0]
        print("【抽取一个 Chunk 进行校验】")
        # Metadata 里自动保留了这是从哪个 PDF、第几页切出来的，这对于 RAG 溯源至关重要
        print(f"📄 来源文件: {sample_chunk.metadata.get('source')}")
        print(f"📑 所在页码: 第 {sample_chunk.metadata.get('page') + 1} 页")
        print(f"📝 文本内容: {sample_chunk.page_content[:100]}...\n")

    return chunks

# ---------------------------------------------------------
# 假装运行测试
# if __name__ == "__main__":
#     # 假设当前目录下有一个 pathology_reports 文件夹装了 10 个 PDF
#     all_chunks = process_medical_pdfs_to_chunks("./pathology_reports")
# ---------------------------------------------------------

#uvicorn main:app --reload --port 8001