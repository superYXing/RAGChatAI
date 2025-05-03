class MMRRetriever:
    """实现最大边际相关性搜索的检索器包装类"""

    def __init__(self, vectorstore, k=5, fetch_k=15, lambda_mult=0.7):
        """
        初始化MMR检索器

        参数:
        - vectorstore: 向量存储
        - k: 返回的文档数量
        - fetch_k: 初始检索的文档数量
        - lambda_mult: 多样性权重参数(0-1)，越高越倾向相关性，越低越倾向多样性
        """
        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    def get_relevant_documents(self, query):
        """获取相关文档"""
        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult
        )
        return docs