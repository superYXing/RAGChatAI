import time
import random
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, rag_system, test_questions=None):
        self.rag_system = rag_system
        self.test_questions = test_questions or []
        self.results = []

    def add_test_question(self, question, expected_source=None, reference_answer=None):
        """添加测试问题"""
        self.test_questions.append({
            "question": question,
            "expected_source": expected_source,
            "reference_answer": reference_answer
        })

    def load_test_questions(self, file_path):
        """从JSON文件加载测试问题"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.test_questions = json.load(f)
            return f"成功加载{len(self.test_questions)}个测试问题"
        except Exception as e:
            return f"加载测试问题失败: {str(e)}"

    def save_test_questions(self, file_path="test_questions.json"):
        """保存测试问题到JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_questions, f, ensure_ascii=False, indent=2)
            return f"测试问题已保存到: {file_path}"
        except Exception as e:
            return f"保存测试问题失败: {str(e)}"

    def run_evaluation(self, max_samples=None):
        """运行评估"""
        self.results = []

        samples = self.test_questions
        if max_samples and max_samples < len(samples):
            samples = random.sample(self.test_questions, max_samples)

        for i, test in enumerate(tqdm(samples, desc="评估进度")):
            question = test["question"]

            start_time = time.time()
            response = self.rag_system.respond(question)
            latency = time.time() - start_time

            # 从响应中提取来源
            sources = []
            if "参考来源:" in response:
                try:
                    sources_str = response.split("参考来源:")[1].strip()
                    sources = [s.strip() for s in sources_str.split(',')]
                except:
                    pass

            # 计算来源准确率
            source_accuracy = 0
            if test.get("expected_source") and sources:
                # 简单计算: 预期来源在找到的来源列表中的比例
                expected = set([s.strip() for s in test["expected_source"].split(',')])
                found = set(sources)
                if expected and found:
                    source_accuracy = len(expected.intersection(found)) / len(expected)

            self.results.append({
                "question": question,
                "response": response,
                "latency_seconds": latency,
                "sources": sources,
                "expected_source": test.get("expected_source"),
                "source_accuracy": source_accuracy,
                "reference_answer": test.get("reference_answer"),
            })

        return self.generate_report()

    def generate_report(self):
        """生成评估报告"""
        if not self.results:
            return "没有可用的评估结果"

        # 计算基本指标
        avg_latency = sum(r["latency_seconds"] for r in self.results) / len(self.results)
        avg_source_accuracy = sum(r["source_accuracy"] for r in self.results) / len(self.results)

        # 计算回答长度
        response_lengths = [len(r["response"]) for r in self.results]
        avg_length = sum(response_lengths) / len(response_lengths)

        report = f"""
        # RAG系统评估报告
        
        ## 基本指标
        - 评估问题数: {len(self.results)}
        - 平均响应时间: {avg_latency:.2f} 秒
        - 平均来源准确率: {avg_source_accuracy:.2%}
        - 平均回答长度: {avg_length:.0f} 字符
        
        ## 详细结果
        """

        # 保存结果到CSV
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False, encoding='utf-8')
            report += f"\n详细结果已保存到CSV文件\n"
        except:
            report += f"\n保存详细结果失败\n"

        return report

    def manual_evaluate(self, question):
        """手动评估单个问题"""
        if not self.rag_system:
            return "RAG系统未初始化", 0

        start_time = time.time()
        response = self.rag_system.respond(question)
        latency = time.time() - start_time

        return response, latency