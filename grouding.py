import json
import os
import argparse
from openai import OpenAI
from loguru import logger

from utils.imageProcessing import draw_box_on_image, get_resolution, encode_image_to_base64
# ======= 配置区 =======
API_KEY = os.getenv("DASHSCOPE_API_KEY")
INPUT_IMAGE_PATH = "output_screenshot.png"
OUTPUT_IMAGE_PATH = "output_screenshot_with_box.png"
# ======================

SYSTEM_PROMPT_UI = '''你是一个视觉助手，可以定位图像中的 UI 元素并返回坐标。

用户会向你提供屏幕的分辨率。

请根据用户提供的图像和指令，返回一个页面上可以用来实现用户指令的交互元素。要求：

    - 交互元素包括按钮、输入框、下拉菜单、标签、可点击的文字等。

    - 你需要标注的边界框坐标、类型及其文字内容。

    - 尽可能精确、小范围地框选出符合用户指令的元素。用户需要使用这个坐标范围进行点击或输入等后续操作。

    - 如果你无法完全确定某个元素是否符合指令的要求，也请尽可能找到文字相似、语义相近的区域并框选出来，作为“可能目标”返回。

    - 即使你不确定，也请勇于给出你认为最接近用户意图的内容，帮助用户完成任务。

请注意：

    - 用户提供指令中涉及到的的元素名称不一定与页面上的元素名称完全一致。
    
    - 请仔细的思考并且区分指令的内容。比如，如果用户打算往搜索框输入字符，你要标注出搜索框的位置；相反，如果用户打算点击搜索按钮，你要标注出搜索按钮的位置。

你需要输出一个 JSON 元素，元素内容包含：

    - "box"：该 UI 元素的边界框坐标，格式为 [x1, y1, x2, y2]（左上角、右下角，单位：像素）

    - "label"：该元素上的文字内容（如按钮文字）

    - "type"：元素的类型，如按钮、输入框等

    - "screen"：图像的分辨率，例如 [800, 600]

输出示例如下：

```json
    {
        "box": [300, 400, 420, 460],
        "label": "确认提交",
        "type": "按钮",
        "screen": [800, 600]
    }
```

请确保json包裹在三重反引号内，并且没有额外的文本或解释。只返回json内容，不要添加任何其他信息。
'''

client = OpenAI(
api_key=API_KEY,
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def send_grounding_request(base64_image, prompt):
    response = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_UI}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
    )
    return response


def parse_box_from_response(response):
    """提取三重反引号中的 JSON 块并解析坐标和分辨率"""
    try:
        content = response.choices[0].message.content
        logger.info(f"模型原始回答：\n{content}")
        json_text = content.split("```json")[-1].split("```")[0].strip()
        data = json.loads(json_text)
        for item in data:
            if "box" in item and "screen" in item:
                box = item["box"]
                screen = item["screen"]
                logger.info(f"提取到的坐标框: {box}, 分辨率: {screen}")
                break
        return data
    except Exception as e:
        logger.error(f"解析 response 失败: {e}")
        return None



def do_instruction_from_todo(todo: dict, history):
    action = todo.get("action")
    params = todo.get("params", {})

    # 所有支持的动作类型
    supported_actions = {"CLICK", "TYPE", "SCROLL", "SUCCESS", "FAIL", "ASK_USER"}
    if action not in supported_actions:
        raise ValueError(f"[错误] 不支持的操作类型: {action}，请让 LLM 重新分析")

    # 对每种操作类型进行参数校验
    if action == "TYPE":
        if not all(k in params for k in ("target", "pos", "text")):
            raise ValueError("[TYPE] 缺少必要参数（target, pos, text）")
        prompt = f"请找出页面中用于输入“{params['target']}”相关内容的输入框，位于{params['pos']}，我将输入“{params['text']}”。"
        data = grounding(prompt)
    elif action == "CLICK":
        if not all(k in params for k in ("target", "pos")):
            raise ValueError("[CLICK] 缺少必要参数（target, pos）")
        prompt = f"请找出页面中标注为“{params['target']}”的按钮或可点击区域，位于{params['pos']}，我准备点击它。"
        data = grounding(prompt)
    elif action == "SCROLL":
        if "direction" not in params:
            raise ValueError("[SCROLL] 缺少必要参数（direction）")
        
    elif action == "ASK_USER":
        if "question" not in params:
            raise ValueError("[ASK_USER] 缺少必要参数（question）")
        
    elif action == "SUCCESS":
        return ""

    elif action == "FAIL":
        return ""
    
    return history

INPUT_IMAGE_PATH = "output_screenshot.png"
OUTPUT_IMAGE_PATH = "output_screenshot_with_box.png"

def grounding(prompt):
    base64_img = encode_image_to_base64(INPUT_IMAGE_PATH)
    os.system(f"cp -f {INPUT_IMAGE_PATH} {INPUT_IMAGE_PATH}")

    rsolution_prompt = f"图像分辨率为 {get_resolution(INPUT_IMAGE_PATH)}"
    response = send_grounding_request(base64_img, rsolution_prompt + prompt)
    data = parse_box_from_response(response)
    if data:
        box = item.get("box")  # type: ignore
        screen = item.get("screen")  # type: ignore
        label = item.get("label", "未知元素")  # type: ignore
        if box and screen:
            draw_box_on_image(OUTPUT_IMAGE_PATH, box, screen, label, OUTPUT_IMAGE_PATH)
        else:
            logger.error("未能成功提取边界框或分辨率。")
    else:
        logger.error("未能从响应中解析到有效数据。")
    return data

def main():
    parser = argparse.ArgumentParser(description="使用 Qwen 模型进行图像边界框标注")
    parser.add_argument("--input", type=str, default=INPUT_IMAGE_PATH, help="输入图像路径")
    parser.add_argument("--output", type=str, default=OUTPUT_IMAGE_PATH, help="输出图像路径")
    parser.add_argument("--inst", type=str, default="帮我下载文件", help="用户指令")
    args = parser.parse_args()
    input_path, output_path, instruction = args.input, args.output, args.inst

    base64_img = encode_image_to_base64(input_path)
    os.system(f"cp -f {input_path} {output_path}")

    rsolution_prompt = f"图像分辨率为 {get_resolution(input_path)}"
    response = send_grounding_request(base64_img, rsolution_prompt + instruction)
    data = parse_box_from_response(response)

    if data:
        box = data.get("box") # type: ignore
        screen = data.get("screen") # type: ignore
        label = data.get("label", "未知元素") # type: ignore
        if box and screen:
            draw_box_on_image(output_path, box, screen, label, output_path)
        else:
            logger.error("未能成功提取边界框或分辨率。")

if __name__ == "__main__":
    main()
