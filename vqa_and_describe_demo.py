import argparse
import json
from loguru import logger

from utils.llm import (
    describe_screen_caption,
    ask_question_about_image,
    parse_page_state_from_description,
    decide_next_action,
)

def main():
    parser = argparse.ArgumentParser(description="vqa和画面解释demo")
    parser.add_argument(
        "--image_path",
        type=str,
        default="test.png",
        help="输入图像的路径，默认为 test.png"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="我的主页上推荐了哪些视频",
        help="要问的问题，默认为 '我的主页上推荐了哪些视频'"
    )
    parser.add_argument(
        "--inst",
        type=str,
        default="帮我搜索洛天依演唱会的回放视频",
        help="用户指令，默认为 '帮我搜索洛天依演唱会的回放视频'"
    )
    args = parser.parse_args()


    description = describe_screen_caption(args.image_path)
    logger.success("页面结构分析结果：\n")
    logger.info(description)

    answer = ask_question_about_image(args.image_path, args.question)
    logger.success("问题回答结果：\n")
    logger.info(answer)

    page_state = parse_page_state_from_description(description)
    logger.success("页面状态结构化结果：\n")
    logger.info(json.dumps(page_state, indent=2, ensure_ascii=False))

    todo = decide_next_action(page_state, args.inst)
    logger.success("🧠 决定的下一步操作：\n")
    logger.info(json.dumps(todo, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()