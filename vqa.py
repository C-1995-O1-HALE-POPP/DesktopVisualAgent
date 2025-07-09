import os
import json
from loguru import logger
from openai import OpenAI

from utils.imageProcessing import encode_image_to_base64

API_KEY = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DESCRIBE_PROMPT = """
你是一个网页理解专家，任务是分析用户提供的桌面截图，描述该页面的整体功能和结构布局。

用户正在完成一个任务，需要你帮助他们理解当前页面的主要功能和各个元素的作用。

请根据用户提供的截图，回答以下问题：

    1. 当前页面的主要功能或用途；

    2. 页面中包含的关键元素（如按钮、输入框、标题等）；

    3. 每个元素的文字内容、类型（按钮/文本/输入框等）及其在页面中的大致位置（如“顶部居中”、“下方靠右”等）；

    4. 哪些区域是用户交互区域，哪些是信息展示区域；

    5. 页面是否属于某个多步骤流程，如果是，请判断当前是第几步。

    6. 请注意分析图形元素的作用，如图标、图片等。

请用自然语言完整描述分析结果，逐段列出你的观察和推理。
"""

DESC_TO_STATE_PROMPT = '''
你是一个结构信息提取助手，用户将提供页面结构的自然语言描述，请将其转化为一个 JSON 格式的页面状态对象 PageState。

结构格式如下：
{
  "page_type": "页面的功能",
  "step": 步骤编号，如果没有可填 null,
  "elements": [
    {
      "label": "文字内容",
      "type": "按钮/输入框/文本/图片等",
      "position": "相对位置，如顶部居中、右下角",
      "role": "interactive 或 informational"
    }
  ]
}

请确保输出是一个合法 JSON，不要添加多余文字。

'''

OPERATION_INFERENCE_PROMPT = '''

你是一个网页操作助手。用户会向你提供一个JSON数组，包含：

    1）page_state：页面的结构状态；
    
    2）user_target：用户的任务目标；
    
    3）user_history：用户的操作历史。

你的任务是根据当前页面的结构状态和用户提供的信息，仔细思考，给出显式的思考过程，推断决定下一步用户应执行的操作：

    - 页面状态包括页面类型、页面中所有元素的文字内容、类型、位置，以及是否可交互。

    - 任务目标是一个简短的描述，表明用户通过 **整个操作流程** 希望完成的操作或目标。

    - 历史操作是一个列表，包含用户在当前页面上已经执行过的操作、用户做出当前操作的原因、用户观察到的页面布局和元素状态。

你需要综合判断：

    - 用户目前所处的页面类型和流程步骤；

    - 页面中有哪些交互元素可操作；

    - 历史中是否已经对某些元素执行过操作；

    - 哪个操作最可能是“下一步”。

你可以进行的操作：

1. "CLICK": 点击某个按钮或链接；

    - 如果用户需要点击一个按钮或者可交互元素，你需要标注出按钮：1）大致位置，以及 2）文字内容（如果有）或者图标含义。

    - 如果用户需要点击一个链接，你需要标注出链接：1）大致位置和 2）文字内容。

    - 交互参数格式如下：

    ```json
    {
        "target": "元素的 label 或者图标含义",
        "pos": 大致位置,    
    }
    ```
2. "TYPE": 输入文本到输入框；

    - 如果用户需要在输入框中输入文本，你需要标注出输入框：1）大致位置，2）文字内容或者label的图标含义（如果有），3）输入的文本内容。

    - 交互参数格式如下：

    ```json 
    {
        "target": "输入框的 label 或者图标含义",
        "pos": 大致位置,
        "text": "要输入的文本内容"
    }
    ```

3. "SCROLL": 滚动页面；

    - 如果用户需要滚动页面，你需要标注出滚动的方向（如向上、向下、向左、向右）。

    - 交互参数格式如下：
    ```json
    {
        "direction": "向上" / "向下" / "向左" / "向右"
    }
    ```

4. "SUCCESS": 表示当前页面已经完成任务；

    - 如果你认为用户的 **任务目标** （而不是当前页面的任务）已经得到充分的完成，你需要返回 "SUCCESS"。

    - 交互参数为：none

5. "FAIL": 表示当前页面无法完成任务，需要返回上一步；

    - 如果你认为当前页面无法完成用户的任务目标，或者用户的操作已经失败，你需要返回 "FAIL"。

    - 交互参数为：none

6. "ASK_USER": 询问用户，当你需要更多的信息来做出决定。

    - 如果你认为当前页面的状态无法推断出下一步操作，或者用户的任务目标不明确，你可以使用这个功能来向用户提问。
    
    - 如果你需要用户确认某个操作是否正确，你也可以使用 "ASK_USER"。

    - 请不要询问用户关于页面元素的具体位置或文字内容，因为你已经有了页面的结构化状态。

    - 交互参数为：

    ```json
        {   
            "question": "你想要问用户的问题"
        }
    ```

请返回一个 JSON 对象，格式如下：

{
    "reasoning": "你的思考过程，描述为什么选择这个操作",
    "action": "CLICK" / "TYPE" / "SUCCESS" / "FAIL" / "SCROLL" / "ASK_USER", # 操作类型
    "params": { 
            # 操作参数
        ... # 根据操作类型不同，参数内容也不同，但是必须是一个有效的 JSON 对象
    },
}

请不要添加任何额外的文字或解释，只返回 JSON 内容。
'''


def ask_question_about_image(image_path: str, question: str) -> str:
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
    return response.choices[0].message.content # type: ignore

def describe_screen_caption(image_path: str) -> str:
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
    return response.choices[0].message.content # type: ignore

def parse_page_state_from_description(description: str) -> dict:
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": DESC_TO_STATE_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": description}]}
        ]
    )

    try:
        content = response.choices[0].message.content
        page_state = json.loads(content) # type: ignore
        return page_state
    except Exception as e:
        print("解析失败：", e)
        return {}

def decide_next_action(page_state, target, history=[]):
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": OPERATION_INFERENCE_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps({"page_state": page_state, "user_target": target, "user_history": history}, ensure_ascii=False)}
                ]
            }
        ]
    )
    
    try:
        content = response.choices[0].message.content
        action = json.loads(content) # type: ignore
        return action
    except Exception as e:
        print("解析失败：", e)
        return {}
    
if __name__ == "__main__":
    # description = describe_screen_caption("test.png")
    # logger.success("🧠 页面结构分析结果：\n")
    # print(description)

    # answer = ask_question_about_image("test.png", "分析一下我的朋友圈")
    # print("🧠 问题回答结果：\n"
    #       f"{answer}")

    description = describe_screen_caption("grouding_lowres.png") 
    logger.success("🧠 页面结构分析结果：\n")
    logger.info(description)
    page_state = parse_page_state_from_description(description)
    logger.success("🧠 页面状态结构化结果：\n")
    logger.info(json.dumps(page_state, indent=2, ensure_ascii=False))
    target = "帮我搜索洛天依演唱会的回放视频"
    history = []
    todo = decide_next_action(page_state, target, history)
    logger.success("🧠 决定的下一步操作：\n")
    logger.info(json.dumps(todo, indent=2, ensure_ascii=False))
    todo["result"] = 
