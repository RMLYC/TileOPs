import os
import google.generativeai as genai
from github import Github

# 在 review_bot.py 顶部引入


# 在 main() 或初始化 model 之前加入
def debug_models():
    print("正在列出可用模型...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"列出模型失败: {e}")


# 在代码开始处调用
debug_models()
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# 1. 配置 API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("GITHUB_REPOSITORY")
PR_NUMBER = int(os.getenv("PR_NUMBER"))

genai.configure(api_key=GEMINI_API_KEY)

# 2. 初始化 Gemini 模型
# 使用 flash 模型速度快且便宜，适合简单 Review
# 如果代码量极大，建议使用 gemini-1.5-pro
model = genai.GenerativeModel('gemini-2.0-flash-exp')


def get_pr_diff():
    """获取 Pull Request 的代码变更"""
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    pr = repo.get_pull(PR_NUMBER)

    # 获取 diff 字符串
    # 注意：实际生产中可能需要过滤掉 .lock 文件或自动生成的文件
    files = pr.get_files()
    diff_content = ""
    for file in files:
        if file.status in ["added", "modified"]:
            diff_content += f"File: {file.filename}\nPatch:\n{file.patch}\n\n"
    return pr, diff_content


def analyze_code(diff_text):
    """发送给 Gemini 进行分析"""
    prompt = f"""
    你是一个资深的代码审查专家。请审查以下 GitHub Pull Request 的代码变更 (Diff)。
    
    关注点：
    1. 潜在的 Bug 或 逻辑错误。
    2. 安全漏洞。
    3. 代码风格改进建议。
    4. 如果代码看起来没问题，请给予简短的肯定。
    
    请用 Markdown 格式输出建议。
    
    代码变更如下：
    {diff_text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 分析出错: {str(e)}"


def main():
    print("正在获取 PR Diff...")
    pr, diff_text = get_pr_diff()

    if not diff_text:
        print("没有检测到代码变更。")
        return

    print("正在请求 Gemini 进行审查...")
    review_comment = analyze_code(diff_text)

    print("正在提交评论到 GitHub...")
    # 在 PR 的时间线上发布评论
    pr.create_issue_comment(f"## 🤖 Gemini Code Review\n\n{review_comment}")
    print("完成！")


if __name__ == "__main__":
    main()
