import os

from interface_builder import create_interface

if __name__ == "__main__":
    # 设置环境变量
    os.environ['OPENAI_API_KEY'] = 'sk-y2akJICa6eI0lIZVZkeIULUyEO0lmg8j14pa21u2860JwdAB'
    os.environ['OPENAI_API_BASE'] = "https://yunwu.ai/v1"
    # 指定知识库目录和向量存储持久化目录
    data_dir = "Data"  # 知识库目录
    persist_directory = "Chroma"  # 向量存储持久化目录

    print(f"初始化AI面试官系统")
    print(f"知识库目录: {os.path.abspath(data_dir)}")
    print(f"向量存储目录: {os.path.abspath(persist_directory)}")

    # 创建并启动界面
    demo = create_interface(data_dir, persist_directory)
    demo.launch(share=True)