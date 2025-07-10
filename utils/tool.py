import json
import re
from typing import Any, Union, List, Dict

JsonData = Union[Dict[str, Any], List[Any]]

def load_json_from_llm(output: str) -> JsonData:
    """
    通吃“裸 JSON / 单行反引号 / 三重反引号代码块”的解析器，
    兼容对象（dict）和数组（list）。

    Parameters
    ----------
    output : str
        LLM 原始输出文本

    Returns
    -------
    JsonData
        Python dict 或 list

    Raises
    ------
    ValueError
        无法解析时抛出，异常里保留原始信息方便调试
    """
    if not isinstance(output, str):
        raise TypeError("output must be a str")

    text = output.strip()

    # --- 三重反引号代码块 ---
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()

    # --- 单行反引号 ---
    if text.startswith("`") and text.endswith("`") and text.count("`") == 2:
        text = text[1:-1].strip()

    # --- 直接尝试解析 ---
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：截取最外层 {} 或 [] 再试一次
        match = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # 最终失败
        raise ValueError(f"无法从给定文本解析 JSON：{text[:100]}...")
