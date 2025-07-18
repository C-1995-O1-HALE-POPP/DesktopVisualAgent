import json
import os
import sys
from openai import OpenAI
from loguru import logger
from pathlib import Path

from utils.imageProcessing import (
    draw_box_on_image, get_resolution, encode_image_to_base64, get_date_time)
from utils import client, INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, MAX_RETRY, VL_MODEL, CHAT_MODEL

SYSTEM_PROMPT_UI = '''你是一个视觉助手，可以定位图像中的 UI 元素并返回坐标。

用户会向你提供屏幕的分辨率。

请根据用户提供的图像和指令，返回一个页面上可以用来实现用户指令的交互元素。要求：

    - 交互元素包括按钮、输入框、下拉菜单、标签、可点击的文字等。

    - 你需要标注的边界框坐标、类型及其文字内容。

    - 尽可能精确、小范围地框选出符合用户指令的元素。用户需要使用这个坐标范围进行点击或输入等后续操作。

    - 如果你无法完全确定某个元素是否符合指令的要求，也请尽可能找到文字相似、语义相近的区域并框选出来，作为“可能目标”返回。

    - 即使你不确定，也请勇于给出你认为最接近用户意图的内容，帮助用户完成任务。

请注意：

    - 即使是页面上的非文字图表也可能具有重要的交互功能，比如退出界面的叉号。你需要仔细观察理解这些非文字元素。

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

def send_grounding_request(base64_image, prompt):
    response = client.chat.completions.create(
        model=VL_MODEL,
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
        if isinstance(data, list):
            data = data[0]
        if "box" in data and "screen" in data:
            box = data["box"]
            screen = data["screen"]
            logger.info(f"提取到的坐标框: {box}, 分辨率: {screen}")
        return data
    except Exception as e:
        logger.error(f"解析 response 失败: {e}")
        return None
    
from utils import RECORD_IMAGE_PATH

def grounding(prompt, input_image_path=INPUT_IMAGE_PATH, output_image_path=OUTPUT_IMAGE_PATH):
    base64_img = encode_image_to_base64(input_image_path)
    os.system(f"cp -f {input_image_path} {output_image_path}")

    rsolution_prompt = f"图像分辨率为 {get_resolution(input_image_path)}"
    for i in range(MAX_RETRY):
        try:
            response = send_grounding_request(base64_img, rsolution_prompt + prompt)
            box_data = parse_box_from_response(response)
            if box_data:
                box = box_data.get("box")  # type: ignore
                screen = box_data.get("screen")  # type: ignore
                label = box_data.get("label", "未知元素")  # type: ignore
                if box and screen:
                    draw_box_on_image(output_image_path, box, screen, label, output_image_path)
                    if RECORD_IMAGE_PATH:
                        if not os.path.exists(RECORD_IMAGE_PATH):
                            os.makedirs(RECORD_IMAGE_PATH)
                        # 备份图像
                        backup_image_path = Path(f"{get_date_time()}_{output_image_path}")
                        backup_image_path = Path.joinpath(Path(RECORD_IMAGE_PATH), Path(backup_image_path))
                        os.system(f"cp -f {output_image_path} {backup_image_path}")
                        logger.success(f"已保存结果图像：{backup_image_path}")
                else:
                    raise ValueError("未能成功提取边界框或分辨率。")
            else:
                raise ValueError("未能从响应中解析到有效数据。")
            return box_data
        except Exception as e:
            logger.error(f"第 {i+1} 次尝试失败: {e}")
            if i < MAX_RETRY - 1:
                logger.info("正在重试...")

    raise RuntimeError("所有尝试均失败，请检查输入图像和提示内容。")


