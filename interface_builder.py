import os
import glob
from datetime import datetime
import gradio as gr
from interview_ai import InterviewAI

def create_interface(data_dir, persist_directory="Chroma"):
    with gr.Blocks() as demo:
        gr.Markdown("# AI面试官系统")

        with gr.Tabs() as tabs:
            with gr.TabItem("聊天界面"):
                gr.Markdown("基于知识库的智能面试系统")

                # 创建状态显示
                with gr.Row():
                    status_msg = gr.Textbox(value="正在加载知识库，请稍候...", label="状态")

                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(height=500)
                        msg = gr.Textbox(label="请输入问题", placeholder="请输入您的问题...", show_label=True)

                        with gr.Row():
                            submit = gr.Button("发送", variant="primary")
                            clear = gr.Button("清空对话")
                            export = gr.Button("导出对话")

                    with gr.Column(scale=1):
                        gr.Markdown("### 系统信息")
                        chroma_info = gr.Textbox(label="知识库信息", interactive=False)
                        file_count = gr.Textbox(label="文件数量", interactive=False)

                        rebuild_btn = gr.Button("重建知识库")

            # 添加性能测评标签页
            with gr.TabItem("性能测评"):
                gr.Markdown("### RAG系统性能测评")

                with gr.Row():
                    with gr.Column():
                        with gr.Group(visible=True):
                            gr.Markdown("#### 手动测评")
                            test_question = gr.Textbox(label="测试问题", placeholder="输入一个测试问题...")
                            expected_source = gr.Textbox(label="预期来源", placeholder="文件1.pdf,文件2.pdf")
                            reference_answer = gr.Textbox(label="参考答案", placeholder="标准答案...", lines=3)

                            with gr.Row():
                                add_question_btn = gr.Button("添加到测试集")
                                test_question_btn = gr.Button("测试此问题")

                            single_result = gr.Textbox(label="测试结果", lines=8)
                            test_latency = gr.Number(label="响应时间(秒)")

                    with gr.Column():
                        with gr.Group(visible=True):
                            gr.Markdown("#### 批量测评")
                            max_samples = gr.Slider(1, 100, 10, step=1, label="最大测试样本数")

                            with gr.Row():
                                load_tests_btn = gr.Button("加载测试集")
                                save_tests_btn = gr.Button("保存测试集")
                                run_eval_btn = gr.Button("运行评估", variant="primary")

                            eval_result = gr.Textbox(label="评估结果", lines=10)
                            test_file = gr.File(label="测试集文件", file_types=[".json"])

        # 延迟加载AI
        interview_ai = None

        def load_ai():
            nonlocal interview_ai
            try:
                # 如果不存在Chroma目录，则创建
                if not os.path.exists(persist_directory):
                    os.makedirs(persist_directory)

                interview_ai = InterviewAI(data_dir=data_dir, persist_directory=persist_directory,youtube_urls=["https://www.youtube.com/watch?v=HAn9vnJy6S4"])

                # 获取知识库信息
                info = "未知"
                build_time_file = os.path.join(persist_directory, "build_time.txt")
                if os.path.exists(build_time_file):
                    with open(build_time_file, "r") as f:
                        info = f"上次构建时间: {f.read().strip()}"

                # 获取文件计数
                pdf_count = len(glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True))
                doc_count = len(glob.glob(os.path.join(data_dir, "**/*.doc*"), recursive=True))
                file_info = f"PDF: {pdf_count}个, DOC/DOCX: {doc_count}个"

                return "知识库加载完成，可以开始对话", info, file_info
            except Exception as e:
                return f"知识库加载失败: {str(e)}", "加载失败", "无法获取文件信息"

        def chat(question, history):
            if not question:
                return history, ""

            if not interview_ai or not interview_ai.chain:
                return history + [["系统提示", "知识库尚未完成加载，请稍候..."]], ""

            response = interview_ai.respond(question)
            return history + [[question, response]], ""

        def clear_history():
            if interview_ai:
                interview_ai.chat_history = []
            return [], "对话已清空"

        def export_history():
            if not interview_ai or not interview_ai.chat_history:
                return "没有聊天记录可导出"

            filename = f"聊天记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            result = interview_ai.save_chat_history(filename)
            return result

        def rebuild_knowledge_base():
            nonlocal interview_ai
            try:
                # 删除持久化目录中的内容
                import shutil
                if os.path.exists(persist_directory):
                    shutil.rmtree(persist_directory)
                    os.makedirs(persist_directory)

                # 重新加载AI
                interview_ai = InterviewAI(data_dir=data_dir, persist_directory=persist_directory,youtube_urls=["https://www.youtube.com/watch?v=HAn9vnJy6S4"])


                # 获取知识库信息
                info = "未知"
                build_time_file = os.path.join(persist_directory, "build_time.txt")
                if os.path.exists(build_time_file):
                    with open(build_time_file, "r") as f:
                        info = f"上次构建时间: {f.read().strip()}"

                # 获取文件计数
                pdf_count = len(glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True))
                doc_count = len(glob.glob(os.path.join(data_dir, "**/*.doc*"), recursive=True))
                file_info = f"PDF: {pdf_count}个, DOC/DOCX: {doc_count}个"

                return "知识库已重建，可以开始对话", info, file_info
            except Exception as e:
                return f"知识库重建失败: {str(e)}", "重建失败", "无法获取文件信息"

        # 测评功能函数
        def add_test_question(question, expected_source, reference_answer):
            if not interview_ai:
                return "系统尚未初始化，请稍候"

            if not question:
                return "请输入测试问题"

            interview_ai.evaluator.add_test_question(
                question=question,
                expected_source=expected_source,
                reference_answer=reference_answer
            )
            return f"已添加测试问题：{question[:30]}..."

        def test_single_question(question):
            if not interview_ai:
                return "系统尚未初始化，请稍候", 0

            if not question:
                return "请输入测试问题", 0

            result, latency = interview_ai.evaluator.manual_evaluate(question)
            return result, latency

        def run_evaluation(max_samples):
            if not interview_ai:
                return "系统尚未初始化，请稍候"

            if not interview_ai.evaluator.test_questions:
                return "测试集为空，请先添加测试问题"

            result = interview_ai.evaluator.run_evaluation(max_samples=int(max_samples))
            return result

        def load_test_questions(file):
            if not interview_ai:
                return "系统尚未初始化，请稍候"

            if not file:
                return "请选择测试集文件"

            result = interview_ai.evaluator.load_test_questions(file.name)
            return result

        def save_test_questions():
            if not interview_ai:
                return "系统尚未初始化，请稍候"

            if not interview_ai.evaluator.test_questions:
                return "测试集为空，请先添加测试问题"

            filename = f"测试集_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result = interview_ai.evaluator.save_test_questions(filename)
            return result

        # 界面逻辑
        demo.load(load_ai, inputs=None, outputs=[status_msg, chroma_info, file_count])

        # 聊天界面逻辑
        submit.click(chat, [msg, chatbot], [chatbot, msg])
        msg.submit(chat, [msg, chatbot], [chatbot, msg])
        clear.click(clear_history, None, [chatbot, status_msg])
        export.click(export_history, None, status_msg)
        rebuild_btn.click(rebuild_knowledge_base, None, [status_msg, chroma_info, file_count])

        # 测评界面逻辑
        add_question_btn.click(
            add_test_question,
            [test_question, expected_source, reference_answer],
            single_result
        )

        test_question_btn.click(
            test_single_question,
            [test_question],
            [single_result, test_latency]
        )

        run_eval_btn.click(
            run_evaluation,
            [max_samples],
            eval_result
        )

        load_tests_btn.click(
            load_test_questions,
            [test_file],
            eval_result
        )

        save_tests_btn.click(
            save_test_questions,
            None,
            eval_result
        )

    return demo