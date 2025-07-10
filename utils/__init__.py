import os
from loguru import logger
from openai import OpenAI


# ======= 配置区 =======
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
API_URL = os.getenv("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)

INPUT_IMAGE_PATH = "output_screenshot.png"
OUTPUT_IMAGE_PATH = "output_screenshot_with_box.png"
RECORD_IMAGE_PATH = "log_image"
MAX_RETRY = 5
VL_MODEL = os.getenv("VL_MODEL", "qwen2.5-vl-32b-instruct")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5-32b-instruct")
# ======================
from utils.webBrowser import webBrowserOperator
browser = webBrowserOperator()