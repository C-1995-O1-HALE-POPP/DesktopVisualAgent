import os
import json
from loguru import logger

from utils.imageProcessing import encode_image_to_base64
from utils import client, INPUT_IMAGE_PATH, MAX_RETRY, VL_MODEL, CHAT_MODEL
from utils.tool import load_json_from_llm
PIC_TO_JSON_PROMPT = """我需要你作为一名前端无障碍与用户体验专家，对提供的网页截图进行分析。请仔细观察页面，找出主要的可交互或可视信息元素，并将分析结果以结构化JSON格式呈现。

具体要求如下：

1. 元素识别范围：
   - 只关注用户会直接交互或注意的重要元素
   - 排除背景、装饰性等次要元素
   - 判断标准：元素面积较大、视觉特征明显、在用户操作流程中起关键作用

2. 每个元素需要包含以下信息：
   - 简短易懂的名称(label)
   - 元素类型(type)
   - 在页面中的相对位置(position)
   - 是可交互元素还是纯展示信息(role)
   - 如果是文字元素，提供完整文字内容(content)
   - 如果是图片/图标，描述其含义或用途(alt)

3. 注意事项：
   - 对于被截断的文字，尽量推断完整含义
   - 图片语义不确定时，可标注"(推测)"
   - 确保JSON格式规范完整

请将分析结果用```json包裹输出，包含页面类型(page_type)和所有识别到的元素(elements数组)。每个元素对象应包含label、type、position、role、content和alt字段。

示例格式(请勿直接复制)：
```json
{
  "page_type": "页面类型描述",
  "step": null,
  "elements": [
    {
      "label": "元素名称",
      "type": "元素类型",
      "position": "位置描述",
      "role": "interactive/informational",
      "content": "文字内容/null",
      "alt": "图片描述/null"
    }
  ]
}
```

请一次性输出完整的JSON分析结果，确保包含尽可能多的页面元素信息。
"""

DESCRIBE_PROMPT = """
你是一位网页理解专家，任务是分析用户提供的桌面截图，全面、细致地描述该页面的结构与页面上的 **主要**内容。

请根据图像回答以下问题：

1. 当前页面的主要功能或用途，例如是否是搜索页面、视频播放页、个人中心、设置页等；

2. 对于页面中包含的关键元素你应该列出：
    - 元素的文字内容（如“搜索”、“首页推荐”、“综合排序”）；
    - 元素的类型（按钮 / 文本 / 输入框 / 标签 / 列表项 / 图标 等）；
    - 元素在页面中的相对位置（如“顶部左侧”、“页面正中”、“底部右侧”等）；

3. 对页面进行结构化描述，包括：
    - 每个结果项中包含的字段（如标题、作者、封面图、播放量等）；
    - 排列方式（如垂直列表、九宫格、横向滑动卡片等）；
    - 文本或图像内容（如推荐的视频标题、具体的用户信息等）；
    - 区分哪些区域是用户可交互区域（如按钮、输入框、链接），哪些是信息展示区域（如正文、标签、说明文字）；
    - 4分析图形元素的作用，如图标按钮、封面图、功能图示等，说明它们可能对应的操作或信息。

5. 如果页面属于某个多步骤流程（如注册、填写表单、支付等），请判断当前是第几步，并说明依据；


请尽可能细致地结合文字、布局与结构进行推理，帮助用户获得对当前页面的准确理解。
输出应为自然语言段落，逐段分析你观察到的内容和推理过程。请一次性地输出整个结果，即使牺牲长度。
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

    - 你需要特别注意用户的历史操作，不要执行重复操作。

你需要综合判断：

    - 用户目前所处的页面类型和流程步骤；

    - 页面中有哪些交互元素可操作；

    - 历史中是否已经对某些元素执行过操作；

    - 哪个操作最可能是“下一步”。

你可以进行的操作：

1. "CLICK": 点击某个按钮或链接；

    - 如果用户需要点击一个按钮或者可交互元素，你需要标注出按钮：1）大致位置，以及 2）文字内容（如果有）或者图标含义。

    - 如果用户需要点击一个链接，你需要标注出链接：1）大致位置和 2）文字内容。

    - 注意：如果你想要用户输入内容，你可以直接调用下方的"TYPE"，而不是先点击后输入。

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

    - 如果你认为用户的 **整体任务目标** （而不是当前页面的任务）已经得到充分的完成，你需要返回 "SUCCESS"。

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
```json
    {
        "reasoning": "你的思考过程，描述为什么选择这个操作",
        "action": "CLICK" / "TYPE" / "SUCCESS" / "FAIL" / "SCROLL" / "ASK_USER", # 操作类型
        "params": { 
            这里填写操作参数：操作类型不同，参数内容也不同，但是必须是一个有效的 JSON 对象
        },
    }
```
请不要添加任何额外的文字或解释，只返回 JSON 内容，确保JSON的格式有效。
'''


def ask_question_about_image(image_path: str, question: str) -> str:
    """向图像提问，使用视觉问答模型回答问题。"""
    base64_img = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model=VL_MODEL,
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

def describe_screen_caption(image_path: str = INPUT_IMAGE_PATH) -> str:
    """描述屏幕截图的结构和功能。"""
    base64_img = encode_image_to_base64(image_path)
    response = client.chat.completions.create(
        model=VL_MODEL,
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
    """将页面描述转换为结构化的页面状态对象。"""
    for i in range(MAX_RETRY):
        try:
            # 使用 Qwen Turbo 模型解析页面状态
            logger.info("正在解析页面状态...")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": DESC_TO_STATE_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": description}]}
                ]
            )
            content = response.choices[0].message.content
            page_state = load_json_from_llm(content) # type: ignore
            return page_state # type: ignore
        except Exception as e:
            logger.error("解析失败：", e)
    raise RuntimeError("所有尝试均失败，请检查输入描述。")

def parse_image_state_to_json(image_path: str = INPUT_IMAGE_PATH):
    """将图像状态解析为结构化的 JSON 对象。"""
    for i in range(MAX_RETRY):
        try:
            logger.info("正在解析图像状态...")
            base64_img = encode_image_to_base64(image_path)
            response = client.chat.completions.create(
                model=VL_MODEL,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": PIC_TO_JSON_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                        ]
                    }
                ]
            )
            content = response.choices[0].message.content
            logger.info(f"模型原始回答：\n{content}")
            page_state = load_json_from_llm(content) # type: ignore
            return page_state
        except Exception as e:
            logger.error(f"第 {i+1} 次尝试失败: {e}")
            if i < MAX_RETRY - 1:
                logger.info("正在重试...")
    raise RuntimeError("所有尝试均失败，请检查输入图像。")

def decide_next_action(page_state, target, history=[]):
    """根据页面状态、用户目标和历史操作，决定下一步操作。"""
    for i in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
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
            content = response.choices[0].message.content
            action = load_json_from_llm(content) # type: ignore

            if action["action"] not in ["CLICK", "TYPE", "SUCCESS", "FAIL", "SCROLL", "ASK_USER"]: # type: ignore
                raise ValueError("操作类型不合法，请检查模型输出。")
            if action["action"] == "TYPE": # type: ignore
                if "params" not in action or "target" not in action["params"] or "pos" not in action["params"] or "text" not in action["params"]: # type: ignore
                    raise ValueError("TYPE 操作缺少必要参数（target, pos, text）")
            elif action["action"] == "CLICK": # type: ignore
                if "params" not in action or "target" not in action["params"] or "pos" not in action["params"]: # type: ignore
                    raise ValueError("CLICK 操作缺少必要参数（target, pos）")
            elif action["action"] == "SCROLL": # type: ignore
                if "params" not in action or "direction" not in action["params"]: # type: ignore
                    raise ValueError("SCROLL 操作缺少必要参数（direction）")
                if action["params"]["direction"] not in ["向上", "向下", "向左", "向右"]: # type: ignore
                    raise ValueError("SCROLL 操作的方向参数无效，请选择：向上、向下、向左或向右")
            elif action["action"] == "ASK_USER": # type: ignore
                if "params" not in action or "question" not in action["params"]: # type: ignore
                    raise ValueError("ASK_USER 操作缺少必要参数（question）")
            return action
        except Exception as e:
            print("解析失败：", e)
            print("正在重试...")

    raise RuntimeError("所有尝试均失败，请检查输入数据。")