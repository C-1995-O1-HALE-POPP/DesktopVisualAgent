import os
import json
from loguru import logger

from utils.grounding import grounding
from utils.llm import (
    describe_screen_caption,
    parse_page_state_from_description,
    decide_next_action,
)
    
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
        box_data = grounding(prompt)["box"]
    elif action == "CLICK":
        if not all(k in params for k in ("target", "pos")):
            raise ValueError("[CLICK] 缺少必要参数（target, pos）")
        prompt = f"请找出页面中标注为“{params['target']}”的按钮或可点击区域，位于{params['pos']}，我准备点击它。"
        box_data = grounding(prompt)["box"]
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
