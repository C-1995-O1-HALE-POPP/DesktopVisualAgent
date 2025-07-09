
from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO
from loguru import logger
from pathlib import Path

import os
import base64
import time

from utils.imageProcessing import get_date_time
from utils import INPUT_IMAGE_PATH, RECORD_IMAGE_PATH
class BrowserAgent:
    def __init__(self, headless=False, resolution=(1280, 720)):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context(
            viewport={"width": resolution[0], "height": resolution[1]},
            screen={"width": resolution[0], "height": resolution[1]}
        )
        self.page = self.context.new_page()
        self.context.on("page", self._on_new_page)
        logger.success(f"浏览器已启动，分辨率设置为: {resolution[0]}x{resolution[1]}")
    def _on_new_page(self, new_page):
        try:
            logger.info("监听到新页面打开，等待加载中...")
            new_page.wait_for_load_state("load", timeout=30000)

            if new_page.is_closed():
                logger.warning("⚠️ 新页面已关闭，放弃切换")
                return

            self.page = new_page
            logger.info("成功切换到新页面")
        except Exception as e:
            logger.error(f"切换到新页面失败: {e}")
    def goto(self, url: str):
        self.page.goto(url)

    def capture_screenshot(self):
        """截图并保存到本地（PNG）"""
        img_bytes = self.page.screenshot(full_page=False)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(INPUT_IMAGE_PATH, format="PNG")
        logger.success(f"已保存页面截图到 {INPUT_IMAGE_PATH}")
        if RECORD_IMAGE_PATH:
            if not os.path.exists(RECORD_IMAGE_PATH):
                os.makedirs(RECORD_IMAGE_PATH)
            # 备份图像
            backup_image_path = Path(f"{get_date_time()}_{INPUT_IMAGE_PATH}")
            backup_image_path = Path.joinpath(Path(RECORD_IMAGE_PATH), Path(backup_image_path))
            os.system(f"cp -f {INPUT_IMAGE_PATH} {backup_image_path}")
            logger.success(f"已备份结果图像：{backup_image_path}")

    def click_box(self, box):
        x = (box[0] + box[2]) // 2
        y = (box[1] + box[3]) // 2
        logger.info(f"→ 点击坐标: ({x}, {y})")

        original_pages = self.context.pages
        original_page_count = len(original_pages)

        # 先点击
        self.page.mouse.click(x, y)
        logger.info("点击完成，等待是否出现新页面...")

        # 等待短时间，检测是否新页面被打开
        self.page.wait_for_timeout(5000)  # 1 秒缓冲时间

        new_pages = self.context.pages
        if len(new_pages) > original_page_count:
            # 尝试找到新打开的页面
            for p in new_pages:
                if p not in original_pages:
                    try:
                        p.wait_for_load_state("load", timeout=30000)
                        if p.is_closed():
                            logger.warning("⚠️ 新页面已关闭，保留原页面")
                        else:
                            self.page = p
                            logger.success("✅ 成功切换到新页面")
                    except Exception as e:
                        logger.error(f"❌ 新页面加载失败，保持当前页面。错误: {e}")
                    break
        else:
            logger.info("ℹ️ 没有新页面打开，继续使用当前页面")

    def type_box(self, box, text: str):
        """点击输入框并输入文本"""
        x = (box[0] + box[2]) // 2
        y = (box[1] + box[3]) // 2
        logger.info(f"→ 输入坐标: ({x}, {y}) 文字: {text}")
        self.page.mouse.click(x, y)
        self.page.keyboard.type(text, delay=50)

    def scroll(self, direction="向下", amount=300):
        dx, dy = 0, 0
        if direction == "向下": dy = amount
        elif direction == "向上": dy = -amount
        elif direction == "向右": dx = amount
        elif direction == "向左": dx = -amount
        else:
            logger.warning("⚠️ 未知滚动方向")
            return
        logger.info(f"→ 滚动页面 ({dx}, {dy})")
        self.page.mouse.wheel(dx, dy)
    
    def back(self):
        """后退到上一个页面"""
        self.page.go_back()

    def close(self):
        self.browser.close()
        self.playwright.stop()
    
    def wait_for_load(self, timeout=30000):
        """等待页面加载完成"""
        self.page.wait_for_load_state("networkidle", timeout=timeout)
        logger.info("→ 页面加载完成")

class webBrowserOperator:
    def __init__(self, url):
        self.agent = BrowserAgent(headless=False)
        self.agent.goto(url)

    def screen_shot(self):
        self.agent.capture_screenshot()

    def execute(self, operation, data, text = ""):
        if operation["type"] == "CLICK":
            self.agent.click_box(data["box"])
        elif operation["type"] == "TYPE":
            self.agent.type_box(data["box"], text)
        elif operation["type"] == "SCROLL":
            self.agent.scroll(operation["direction"])
        else:
            raise ValueError(f"Unknown operation type: {operation['type']}")
    
    def wait(self, timeout=30000):
        """等待页面加载完成"""
        time.sleep(5)  # 等待5秒，确保操作稳定
        logger.info("→ 等待页面加载...")
        self.agent.wait_for_load(timeout)
        

import time

def test_web_browser_operator():
    # 示例网址：B站首页（或你可以替换为任何支持的网页）
    url = "https://www.bilibili.com"
    operator = webBrowserOperator(url)

    # 等待页面加载
    operator.wait()

    # 1. 截图测试
    operator.screen_shot()
    logger.success("已保存页面截图到 output_screenshot.png")

    # 2. 测试输入（模拟点击搜索框并输入）
    type_op = {
        "type": "TYPE"
    }
    operator.execute(type_op, {"box": [580, 16, 837, 49]}, text="洛天依演唱会")
    logger.success("已输入关键词")
    operator.wait()

    # 3. 测试点击
    click_op = {
        "type": "CLICK"
    }
    dummy_box = [806, 25, 819, 40]
    operator.execute(click_op, {"box": dummy_box})
    logger.success("已模拟点击")

    operator.wait()

    # 4. 测试滚动
    scroll_op = {
        "type": "SCROLL",
        "direction": "向下"
    }
    operator.execute(scroll_op, {})
    logger.success("已执行滚动操作")

    # 等待观察效果
    operator.wait()

    # 关闭浏览器
    operator.agent.close()
    logger.success("✅ 浏览器已关闭")

if __name__ == "__main__":
    test_web_browser_operator()