import os
import argparse
from loguru import logger

from utils.imageProcessing import draw_box_on_image, get_resolution, encode_image_to_base64
from utils.grounding import send_grounding_request, parse_box_from_response


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen 模型进行图像边界框标注")
    parser.add_argument("--input", type=str, default="test.png", help="输入图像路径")
    parser.add_argument("--output", type=str, default="test_demo.png", help="输出图像路径")
    parser.add_argument("--inst", type=str, default="请找出页面中用于输入“搜索框”相关内容的输入框，位于正上方，我将输入“洛天依演唱会”。", help="用户指令")
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