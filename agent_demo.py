import os
import json
import argparse
import sys
from loguru import logger

from utils.grounding import grounding
from utils.llm import (
    describe_screen_caption,
    parse_page_state_from_description,
    decide_next_action,
    parse_image_state_to_json
)
from utils.webBrowser import webBrowserOperator
from utils import browser
MAX_RETRY = 3
running = True

def do_instruction_from_todo(todo: dict):
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
        
        box_data = grounding(prompt)
        box = box_data["box"]
        if not box or len(box) != 4:
            raise ValueError(f"[CLICK] 未找到标注为“{params['target']}”的按钮或区域，请让 LLM 重新分析")
        browser.execute({"type": "TYPE"}, box, params["text"])
        return f"输入内容到 {params['target']}：{params['text']}"
    
    elif action == "CLICK":
        if not all(k in params for k in ("target", "pos")):
            raise ValueError("[CLICK] 缺少必要参数（target, pos）")
        prompt = f"请找出页面中标注为“{params['target']}”的按钮或可点击区域，位于{params['pos']}，我准备点击它。"

        box_data = grounding(prompt)
        box = box_data["box"]
        if not box or len(box) != 4:
            raise ValueError(f"[CLICK] 未找到标注为“{params['target']}”的按钮或区域，请让 LLM 重新分析")
        browser.execute({"type": "CLICK"}, box)
        return f"点击 {params['target']} 按钮或区域"
    
    elif action == "SCROLL":
        if "direction" not in params:
            raise ValueError("[SCROLL] 缺少必要参数（direction）")
        if params["direction"] not in {"向上", "向下", "向左", "向右"}:
            raise ValueError("[SCROLL] 方向参数无效，请选择：向上、向下、向左或向右")

        browser.execute({"type": "SCROLL"}, {"direction": params["direction"]})
        return f"向{params['direction']}滚动页面"
    
    elif action == "ASK_USER":
        if "question" not in params:
            raise ValueError("[ASK_USER] 缺少必要参数（question）")
        
        response = ""
        while not response.strip():
            response = ask_user_for_plain_answer(params["question"])
        return f"用户回答: {response}"
        
    elif action == "SUCCESS":
        response = ask_user_for_decision("确认操作成功？")
        if response:
            global running
            running = False
            logger.success("用户确认操作成功，代理将退出。")
            return "用户确认操作成功，代理将退出。"

    elif action == "FAIL":
        browser.back()
        return ("操作失败，返回上一步。")
    
def ask_user_for_plain_answer(question: str):
    """向用户询问问题，并返回用户的回答"""
    logger.info(f"需要用户确认: {question}")
    response = input("请输入您的回答：")
    return response

def ask_user_for_decision(question: str):
    """向用户询问问题，并返回用户的决策"""
    logger.info(f"🧠 需要用户决策: {question}")
    response = input("请输入您的决策（YES/NO）：")
    if response.strip().upper() == "YES":
        return True
    elif response.strip().upper() == "NO":
        return False
    else:
        logger.error("无效的输入，请输入 YES 或 NO")
        return ask_user_for_decision(question)

def agent_start(url: str, instruction: str = "帮我搜索洛天依演唱会的回放视频"):
    """启动代理，执行一系列操作"""
    logger.info("🧠 启动浏览器代理...")
    browser.start(url)

    history = []

    global running
    while running:

        logger.info("\n\n1. 截图当前页面...")
        browser.screen_shot()

        logger.info("\n\n2. 分析页面结构...")
        # description = describe_screen_caption()
        # logger.success("页面结构分析结果：\n")
        # logger.info(description)
        # logger.info("\n\n3. 解析页面状态...")
        # page_state = parse_page_state_from_description(description)
        # logger.success("页面状态结构化结果：\n")
        # logger.info(json.dumps(page_state, indent=2, ensure_ascii=False))
        page_state = parse_image_state_to_json()

        operation = decide_next_action(page_state, instruction, history)
        logger.success("\n\n4. 决定的下一步操作：\n")
        logger.info(json.dumps(operation, indent=2, ensure_ascii=False))

        result = do_instruction_from_todo(operation)
        logger.success(f"\n\n5. 操作结果：{result}")
        operation["result"] = result
        history.append(operation)

        logger.info("\n\n 6. 等待下一步操作...")
        logger.info("当前历史操作记录：")
        logger.info(json.dumps(history, indent=2, ensure_ascii=False))
        browser.wait(sleep_sec = 10)
        
    logger.info("🧠 代理执行完毕，关闭浏览器...")

def main():
    args = argparse.ArgumentParser(description="启动浏览器代理执行任务")
    args.add_argument("--url", type=str, default="https://www.bilibili.com",
                      help="要访问的初始网址，默认为 B 站首页")
    args.add_argument("--instruction", type=str, default="帮我搜索洛天依演唱会的回放视频",
                      help="代理执行的任务指令，默认为搜索洛天依演唱会的回放视频")
    parsed_args = args.parse_args()
    logger.info(f"启动代理，访问网址: {parsed_args.url}")
    logger.info(f"执行任务指令: {parsed_args.instruction}")
    agent_start(parsed_args.url, parsed_args.instruction)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    main()
    browser.close()
    logger.info("🧠 代理已成功关闭。")