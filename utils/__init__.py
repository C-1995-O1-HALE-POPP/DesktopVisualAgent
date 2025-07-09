import os
from loguru import logger
from openai import OpenAI
# ======= 配置区 =======
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)

INPUT_IMAGE_PATH = "output_screenshot.png"
OUTPUT_IMAGE_PATH = "output_screenshot_with_box.png"
RECORD_IMAGE_PATH = "log_image"
# ======================