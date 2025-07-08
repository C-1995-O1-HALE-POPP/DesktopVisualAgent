import os
from openai import OpenAI

from utils.imageProcessing import encode_image_to_base64

API_KEY = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DESCRIBE_PROMPT = """
你是一个网页理解专家，任务是分析用户提供的桌面截图，描述该页面的整体功能和结构布局。

你需要回答：
1. 当前页面的主要功能或用途；
2. 页面中包含的关键元素（如按钮、输入框、标题等）；
3. 每个元素的文字内容、类型（按钮/文本/输入框等）及其在页面中的大致位置（如“顶部居中”、“下方靠右”等）；
4. 哪些区域是用户交互区域，哪些是信息展示区域；
5. 页面是否属于某个多步骤流程，如果是，请判断当前是第几步。

请用自然语言完整描述分析结果，逐段列出你的观察和推理。
"""

def ask_question_about_image(image_path: str, question: str) -> str | None:
    base64_img = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "你是一个视觉问答助手，能回答用户提出的关于图像的问题。"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                    {"type": "text", "text": question}
                ]
            }
        ]
    )
    return response.choices[0].message.content

def describe_screen_caption(image_path: str) -> str | None:
    base64_img = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": DESCRIBE_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                    {"type": "text", "text": "请分析这个页面的结构和功能。"}
                ]
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # description = describe_screen_caption("test.png")
    # print("🧠 页面结构分析结果：\n")
    # print(description)
    answer = ask_question_about_image("test.png", "分析一下我的朋友圈")
    print("🧠 问题回答结果：\n"
          f"{answer}")
