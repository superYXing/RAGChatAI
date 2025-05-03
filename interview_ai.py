import os
import glob
import time
import random
import hashlib
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from rag_evaluator import RAGEvaluator

class InterviewAI:
    def __init__(self, data_dir, persist_directory="Chroma", youtube_urls=None):
        self.data_dir = data_dir
        self.persist_directory = persist_directory
        self.chain = None
        self.chat_history = []
        self.embeddings = OpenAIEmbeddings(
            chunk_size=50,  # 每次API调用的最大文本数
            request_timeout=60,  # 增加超时时间
            max_retries=3  # 内部重试机制
        )
        self.youtube_urls = youtube_urls or []  # 存储YouTube视频链接
        self.initialize_chain()
        self.evaluator = RAGEvaluator(self)  # 创建评估器

    def get_files_hash(self):
        """计算文件夹中所有文件的哈希值，用于判断文档是否更新"""
        files = []
        files.extend(glob.glob(os.path.join(self.data_dir, "**/*.pdf"), recursive=True))
        files.extend(glob.glob(os.path.join(self.data_dir, "**/*.doc*"), recursive=True))

        files.sort()  # 确保文件顺序一致

        if not files:
            return ""

        hasher = hashlib.md5()
        for file_path in files:
            mtime = os.path.getmtime(file_path)
            file_info = f"{file_path}:{mtime}"
            hasher.update(file_info.encode())

        return hasher.hexdigest()

    def load_documents(self):
        """从指定目录加载所有PDF和DOC文件，以及YouTube视频字幕"""
        documents = []

        # 确保目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建了目录: {self.data_dir}")

        # 加载所有PDF文件
        pdf_files = glob.glob(os.path.join(self.data_dir, "**/*.pdf"), recursive=True)
        for pdf_file in tqdm(pdf_files, desc="加载PDF文件"):
            try:
                print(f"加载PDF: {pdf_file}")
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                # 过滤掉page_content为None的文档
                valid_docs = []
                for doc in docs:
                    if doc.page_content is None:
                        print(f"警告: 文档 {pdf_file} 中发现page_content为None，已跳过")
                        continue
                    if not isinstance(doc.page_content, str) or len(doc.page_content.strip()) == 0:
                        print(f"警告: 文档 {pdf_file} 中发现空内容，已跳过")
                        continue
                    # 添加源文件信息到元数据
                    doc.metadata["source"] = os.path.basename(pdf_file)
                    doc.metadata["file_path"] = pdf_file
                    valid_docs.append(doc)
                documents.extend(valid_docs)
                print(f"成功加载PDF: {pdf_file}，有效文档数: {len(valid_docs)}/{len(docs)}")
            except Exception as e:
                print(f"加载PDF失败 {pdf_file}: {e}")

        # 加载所有DOC/DOCX文件
        docx_files = glob.glob(os.path.join(self.data_dir, "**/*.doc*"), recursive=True)
        for doc_file in tqdm(docx_files, desc="加载DOCX文件"):
            try:
                print(f"加载DOCX: {doc_file}")
                loader = UnstructuredWordDocumentLoader(doc_file)
                docs = loader.load()
                # 过滤掉page_content为None的文档
                valid_docs = []
                for doc in docs:
                    if doc.page_content is None:
                        print(f"警告: 文档 {doc_file} 中发现page_content为None，已跳过")
                        continue
                    if not isinstance(doc.page_content, str) or len(doc.page_content.strip()) == 0:
                        print(f"警告: 文档 {doc_file} 中发现空内容，已跳过")
                        continue
                    # 添加源文件信息到元数据
                    doc.metadata["source"] = os.path.basename(doc_file)
                    doc.metadata["file_path"] = doc_file
                    valid_docs.append(doc)
                documents.extend(valid_docs)
                print(f"成功加载DOCX: {doc_file}，有效文档数: {len(valid_docs)}/{len(docs)}")
            except Exception as e:
                print(f"加载DOCX失败 {doc_file}: {e}")

        # 加载YouTube视频字幕
        # if self.youtube_urls:
        #     print("开始加载YouTube视频字幕...")
        #     for url in self.youtube_urls:
        #         try:
        #             # 一个Youtube的视频对应一个document
        #             youtube_docs = YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
        #             for doc in youtube_docs:
        #                 # 给doc添加额外的元数据：视频发布的年份
        #                 doc.metadata['publish_year'] = int(
        #                     datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
        #                 doc.metadata["source"] = f"Youtube: {doc.metadata['title']}"
        #             documents.extend(youtube_docs)
        #             print(f"成功加载YouTube视频: {url}")
        #         except Exception as e:
        #             print(f"加载YouTube视频失败 {url}: {e}")

        if not documents:
            print("未找到任何有效文档，请将PDF、DOC文件放入指定目录或提供有效的YouTube链接")
            return []

        print(f"总共加载了{len(documents)}个有效文档")
        return documents

    def should_rebuild_vectorstore(self):
        """判断是否需要重建向量存储"""
        # 检查持久化目录是否存在
        if not os.path.exists(self.persist_directory):
            return True

        # 读取上次构建时的文件哈希值
        hash_file = os.path.join(self.persist_directory, "files_hash.txt")
        if not os.path.exists(hash_file):
            return True

        # 对比当前文件哈希值和上次保存的哈希值
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()

        current_hash = self.get_files_hash()

        return saved_hash != current_hash

    def initialize_chain(self):
        # 判断是否需要重建向量存储
        rebuild = self.should_rebuild_vectorstore()

        if rebuild:
            print("需要重建向量存储...")
            self.build_vectorstore()
        else:
            print("加载已有的向量存储...")
            self.load_vectorstore()

    def build_vectorstore(self):
        """构建并持久化向量存储"""
        # 加载文档
        documents = self.load_documents()
        if not documents:
            return

        # 验证所有文档的page_content都是有效的字符串
        valid_documents = []
        for doc in documents:
            try:
                # 确保page_content是非空字符串
                if doc.page_content is None:
                    continue
                if not isinstance(doc.page_content, str) or len(doc.page_content.strip()) == 0:
                    continue
                valid_documents.append(doc)
            except Exception as e:
                print(f"验证文档时出错: {e}")
                continue

        print(f"验证后的有效文档数: {len(valid_documents)}/{len(documents)}")
        documents = valid_documents

        if not documents:
            print("没有有效的文档可以处理")
            return

        # 文本分割
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            print(f"文本分割完成，共{len(splits)}个片段")
        except Exception as e:
            print(f"文本分割失败: {e}")
            # 尝试一个文档一个文档地分割，跳过有问题的文档
            splits = []
            for doc in tqdm(documents, desc="逐个分割文档"):
                try:
                    doc_splits = text_splitter.split_documents([doc])
                    splits.extend(doc_splits)
                except Exception as doc_e:
                    print(f"分割文档失败，已跳过: {doc_e}")
            print(f"文本分割完成(使用备选方法)，共{len(splits)}个片段")

        if not splits:
            print("文本分割后没有可用片段")
            return

        # 验证所有分割后的文档
        valid_splits = []
        for split in splits:
            try:
                # 确保page_content是非空字符串
                if split.page_content is None:
                    continue
                if not isinstance(split.page_content, str) or len(split.page_content.strip()) == 0:
                    continue
                valid_splits.append(split)
            except Exception as e:
                print(f"验证文本片段时出错: {e}")
                continue

        print(f"验证后的有效文本片段数: {len(valid_splits)}/{len(splits)}")
        splits = valid_splits

        # 批量处理文档，避免ConnectionError
        max_retries = 5
        retry_count = 0
        batch_size = 50  # 每批处理的文档数

        while retry_count < max_retries:
            try:
                # 创建向量存储 (使用Chroma)
                print("创建向量存储中，这可能需要一些时间...")

                # 分批处理文档
                total_batches = (len(splits) + batch_size - 1) // batch_size
                processed_docs = []

                for i in tqdm(range(0, len(splits), batch_size), desc="处理文档批次", total=total_batches):
                    batch = splits[i:i+batch_size]
                    processed_docs.extend(batch)

                    # 添加随机延迟，避免API调用过于频繁
                    if i + batch_size < len(splits):
                        time.sleep(random.uniform(1, 3))

                # 使用处理后的文档创建向量存储并持久化
                vectorstore = Chroma.from_documents(
                    documents=processed_docs,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )

                # 保存文件哈希值
                current_hash = self.get_files_hash()
                with open(os.path.join(self.persist_directory, "files_hash.txt"), "w") as f:
                    f.write(current_hash)

                # 保存构建时间
                with open(os.path.join(self.persist_directory, "build_time.txt"), "w") as f:
                    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                print("向量存储创建完成并持久化到磁盘")

                # 创建对话链
                self.create_chain(vectorstore)
                break

            except Exception as e:
                retry_count += 1
                wait_time = 2 ** retry_count  # 指数退避
                print(f"创建向量存储失败 (尝试 {retry_count}/{max_retries}): {e}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

        if retry_count == max_retries:
            print("创建向量存储失败，已达到最大重试次数")

    def load_vectorstore(self):
        """加载已持久化的向量存储"""
        try:
            # 从持久化目录加载向量存储
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            # 读取构建时间
            build_time_file = os.path.join(self.persist_directory, "build_time.txt")
            build_time = "未知"
            if os.path.exists(build_time_file):
                with open(build_time_file, "r") as f:
                    build_time = f.read().strip()

            print(f"成功加载向量存储，构建时间: {build_time}")

            # 创建对话链
            self.create_chain(vectorstore)

        except Exception as e:
            print(f"加载向量存储失败: {e}")
            print("将尝试重新构建向量存储...")
            self.build_vectorstore()

    def create_chain(self, vectorstore):
        """基于向量存储创建对话链"""
        try:
            # 创建对话链
            llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                request_timeout=60  # 增加超时时间
            )

            # 创建高级检索器
            # 1. 配置多种检索策略
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",  # 基于相似度阈值的检索
                search_kwargs={
                    "k": 8,  # 增加初始检索数量
                    "score_threshold": 0.5,  # 相似度阈值，低于此值的文档将被过滤
                    "filter": None  # 可以根据需要添加过滤条件
                }
            )

            # 创建更高级的检索链
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                # 添加更精确的问题重写步骤，使检索更准确
                rephrase_question=True,
                # 优化检索得分计算
                get_chat_history=lambda h: " ".join([f"Human: {q}\nAI: {a}" for q, a in h[-3:]]),
                # 使用最近的3次对话作为上下文，减少干扰
                max_tokens_limit=4000
            )
            print("对话链创建完成")
        except Exception as e:
            print(f"创建对话链失败: {e}")

    def respond(self, question):
        if not self.chain:
            return "系统未正确初始化，可能是由于文档加载或处理失败"

        try:
            # 第一次提示词，宽泛检索
            prompt1 = f"""你是一个AI求职者，文档内容是你的知识库，除此之外，你一无所知。
            请以专业、友好的回答。
            请根据问题初步检索相关文档。
            
            当前问题：{question}
            """

            # 第一次检索
            result1 = self.chain({"question": prompt1, "chat_history": self.chat_history})
            source_documents = result1.get("source_documents", [])

            # 第二次提示词，结合第一次检索结果进行更精确检索
            source_info = "\n".join([doc.page_content for doc in source_documents])
            prompt2 = f"""你是一个AI求职者，文档内容是你的知识库，除此之外，你一无所知。
            请以专业、友好的回答。
            如果问题超出文档范围，请礼貌地说明。
            
            请注意以下要点：
            1. 仅使用检索到的文档中的事实信息回答
            2. 如果文档中没有足够信息，明确说明而不是猜测
            3. 保持答案的简洁性和相关性
            4. 如果问题有多个方面，分别回答每个方面
            
            以下是初步检索到的相关文档内容：
            {source_info}
            
            当前问题：{question}
            """

            # 第二次检索
            result2 = self.chain({"question": prompt2, "chat_history": self.chat_history})
            answer = result2["answer"]

            # 过滤和排序文档引用
            if "source_documents" in result2 and result2["source_documents"]:
                # 按相关性分数排序（如果可用）
                sorted_docs = sorted(
                    result2["source_documents"],
                    key=lambda doc: doc.metadata.get("score", 0) if isinstance(doc.metadata.get("score"), (int, float)) else 0,
                    reverse=True
                )

                # 提取最相关的来源
                sources = set()
                content_sources = []  # 用于存储文档内容和来源的对

                for doc in sorted_docs[:5]:  # 限制为最相关的5个文档
                    if "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if source not in sources:
                            sources.add(source)
                            # 存储文档片段和来源
                            content_snippet = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                            content_sources.append((content_snippet, source))

                # 添加来源信息到回答中
                if sources:
                    answer += f"\n\n参考来源: {', '.join(sources)}"

                    # 可选：添加每个来源的相关内容片段
                    answer += "\n\n引用片段:"
                    for i, (content, source) in enumerate(content_sources[:3], 1):
                        answer += f"\n{i}. [{source}] {content}"

            # 更新对话历史
            self.chat_history.append((question, answer))

            return answer
        except Exception as e:
            print(f"回答问题时出错: {e}")
            return f"处理您的问题时出现错误: {str(e)}"

    def save_chat_history(self, filename="chat_history.txt"):
        """保存聊天历史到文件"""
        if not self.chat_history:
            return "没有聊天记录可保存"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"聊天记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for i, (question, answer) in enumerate(self.chat_history):
                    f.write(f"问题 {i+1}: {question}\n")
                    f.write(f"回答 {i+1}: {answer}\n\n")
            return f"聊天记录已保存到: {filename}"
        except Exception as e:
            return f"保存聊天记录失败: {str(e)}"