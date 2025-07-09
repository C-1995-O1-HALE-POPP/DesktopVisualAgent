
from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO
import base64
import time
OUTPUT_IMAGE_PATH = "output_screenshot.png"
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
    def _on_new_page(self, new_page):
        try:
            print("📄 监听到新页面打开，等待加载中...")
            new_page.wait_for_load_state("load", timeout=10000)

            if new_page.is_closed():
                print("⚠️ 新页面已关闭，放弃切换")
                return

            self.page = new_page
            print("✅ 成功切换到新页面")
        except Exception as e:
            print(f"❌ 切换到新页面失败: {e}")
    def goto(self, url: str):
        self.page.goto(url)

    def capture_screenshot(self):
        """截图并保存到本地（PNG）"""
        img_bytes = self.page.screenshot(full_page=False)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.save(OUTPUT_IMAGE_PATH, format="PNG")

    def click_box(self, box):
        x = (box[0] + box[2]) // 2
        y = (box[1] + box[3]) // 2
        print(f"→ 点击坐标: ({x}, {y})")

        with self.context.expect_page() as new_page_info:
            self.page.mouse.click(x, y)
        new_page = new_page_info.value

        try:
            new_page.wait_for_load_state("load", timeout=10000)
            if new_page.is_closed():
                print("⚠️ 捕获的新页面已关闭，保留当前页面不变")
            else:
                self.page = new_page
                print("✅ 成功切换到新页面")
        except Exception as e:
            print(f"⚠️ 新页面未能加载完成，保持当前页面。错误: {e}")

    def type_box(self, box, text: str):
        """点击输入框并输入文本"""
        x = (box[0] + box[2]) // 2
        y = (box[1] + box[3]) // 2
        print(f"→ 输入坐标: ({x}, {y}) 文字: {text}")
        self.page.mouse.click(x, y)
        self.page.keyboard.type(text, delay=50)

    def scroll(self, direction="向下", amount=300):
        dx, dy = 0, 0
        if direction == "向下": dy = amount
        elif direction == "向上": dy = -amount
        elif direction == "向右": dx = amount
        elif direction == "向左": dx = -amount
        else:
            print("⚠️ 未知滚动方向")
            return
        print(f"→ 滚动页面 ({dx}, {dy})")
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
        print("→ 页面加载完成")

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
        time.sleep(1)  # 等待1秒，确保操作稳定
        print("→ 等待页面加载...")
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
    print("✅ 已保存页面截图到 output_screenshot.png")

    # 2. 测试输入（模拟点击搜索框并输入）
    type_op = {
        "type": "TYPE"
    }
    operator.execute(type_op, {"box": [580, 16, 837, 49]}, text="洛天依演唱会")
    print("✅ 已输入关键词")
    operator.wait()

    # 3. 测试点击某区域（例如大致中部区域）
    click_op = {
        "type": "CLICK"
    }
    dummy_box = [600, 300, 800, 400]  # 示例中部 box
    operator.execute(click_op, {"box": dummy_box})
    print("✅ 已模拟点击")

    operator.wait()

    # 4. 测试滚动
    scroll_op = {
        "type": "SCROLL",
        "direction": "向下"
    }
    operator.execute(scroll_op, {})
    print("✅ 已执行滚动操作")

    # 等待观察效果
    operator.wait()

    # 关闭浏览器
    operator.agent.close()
    print("✅ 浏览器已关闭")

if __name__ == "__main__":
    test_web_browser_operator()