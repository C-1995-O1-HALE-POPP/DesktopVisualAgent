from PIL import Image, ImageDraw, ImageFont
import base64
import os
from loguru import logger

def draw_box_on_image(image_path, box, screen_resolution, label, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    actual_width, actual_height = img.size
    screen_width, screen_height = screen_resolution

    # 按比例缩放坐标
    x_scale = actual_width / screen_width
    y_scale = actual_height / screen_height

    scaled_box = [
        int(box[0] * x_scale),
        int(box[1] * y_scale),
        int(box[2] * x_scale),
        int(box[3] * y_scale),
    ]

    logger.info(f"绘制缩放后的坐标框: {scaled_box}")

    draw.rectangle(scaled_box, outline="red", width=4)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    draw.text((scaled_box[0], scaled_box[1] - 20), label, fill="red", font=font)
    img.save(output_path)
    logger.success(f"已保存结果图像：{output_path}")

def get_resolution(image_path):
    """获取图像的分辨率"""
    try:
        img = Image.open(image_path)
        return img.size  # 返回 (width, height)
    except Exception as e:
        logger.error(f"获取图像分辨率失败: {e}")
        return None
    
def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def get_date_time():
    """获取当前日期时间：yyyy-MM-dd@HH:mm:ss"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d@%H:%M:%S")