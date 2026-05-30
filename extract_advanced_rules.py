import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests

# =========================
# 1. 基本配置
# =========================

INPUT_FILE = r"D:\Desktop\bool\raw_rules_advanced.jsonl"
OUTPUT_FILE = r"D:\Desktop\bool\extract_advanced_results.jsonl"
ERROR_FILE = r"D:\Desktop\bool\extract_advanced_errors.jsonl"

BASE_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-v4-pro"   # 可改为 deepseek-v4-flash

MAX_RETRIES = 3
RETRY_SLEEP = 3
REQUEST_TIMEOUT = 120


# =========================
# 2. 提示词
# =========================

SYSTEM_PROMPT = """
你是一个“自然语言规则结构化抽取器”。
你的任务是把一条中文自然语言风险规则，转换成标准 JSON。

当前阶段处理“布尔逻辑 + 时间/过程逻辑”的规则抽取，采用“三阶段抽取”方式：

第一阶段：抽取基本事件
- 把规则中的每个最小条件拆成独立事件，编号为 E1, E2, E3...
- 每个事件必须是一个可以单独成立的最小条件单元
- 不要把多个并列条件合并成一个事件
- 输出结论不要放入 events，而是单独放入 output

第二阶段：抽取布尔逻辑
- 根据自然语言中的“并且、或者、不是、没有”等表达，将事件组合成布尔表达式
- 当前允许使用以下逻辑符号：
  AND
  OR
  NOT
  ()
- 如果规则中没有明显 OR 或 NOT，也要给出最基本的 AND 结构
- 如果原文表达“没有某事件”，优先将该事件写成肯定式事件文本，并在 booleanLogic 中用 NOT 表示

第三阶段：抽取时间/过程逻辑
当前阶段允许识别以下五类逻辑：

1. 顺序逻辑：
   SEQUENCE(E1, E2)

2. 时间窗口逻辑：
   WITHIN(E2, 30天)

3. 持续逻辑：
   DURATION(E1, 3天)

4. 超时逻辑：
   TIMEOUT(E1, E2, 30天)

5. 因果链逻辑：
   CAUSAL_CHAIN(E1, E2, E3, E4)

如果规则中没有明确时间/过程逻辑，则输出空数组：[]

输出要求：
你只能输出一个 JSON 对象，不要输出任何解释文字，不要输出 markdown。
JSON 字段固定为：
- ruleId
- rawText
- events
- booleanLogic
- temporalLogic
- output

字段说明如下：

1. events
是数组，每个元素格式为：
{"id":"E1","text":"事件文本"}

2. booleanLogic
是由 E1, E2, E3... 组成的布尔表达式，例如：
"E1 AND E2"
"E1 OR E2"
"E1 AND NOT E2"
"(E1 OR E2) AND E3"
"NOT (E2 OR E3)"

3. temporalLogic
是数组，内部每个元素是一个时间/过程逻辑表达式，例如：
[
  "SEQUENCE(E1, E2)",
  "WITHIN(E2, 30天)",
  "DURATION(E1, 3天)",
  "TIMEOUT(E1, E2, 30天)",
  "CAUSAL_CHAIN(E1, E2, E3, E4)"
]

4. output
是规则最终输出的风险结论，尽量保留原文表述，但不保留“判定为”“触发”“生成”“判断为”这类动作词，只保留风险结论本体。

规则：
- 如果规则存在歧义，优先保持最保守抽取，不要自行补充隐含条件
- 不要虚构不存在的事件
- 不要增加额外字段
- 否定应尽量体现在 booleanLogic 中，而不是写进事件文本里
- 如果原文没有明确说明事件之间的严格先后关系，不要随意补充 SEQUENCE
- 如果原文明确存在“连续X天”“持续X天”，优先使用 DURATION
- 如果原文明确存在“在X天内没有发生某事件”，优先使用 TIMEOUT
- 如果原文明确表现出“原因—过程—结果”的链路，优先使用 CAUSAL_CHAIN
"""

USER_PROMPT_TEMPLATE = """
请对下面这条规则进行抽取。

输入 JSON：
{input_json}

请直接输出最终 JSON 对象。
"""

# 备用提示词：当主提示词无法稳定输出 JSON 时，使用更短更硬的约束
FALLBACK_USER_PROMPT_TEMPLATE = """
请严格抽取下面这条规则，只输出一个合法 JSON 对象，不要输出任何其他内容。

要求：
1. 输出字段只能有：
   ruleId, rawText, events, booleanLogic, temporalLogic, output
2. 如果涉及“没有A或者B”，优先写成：
   booleanLogic = "E1 AND NOT (E2 OR E3)"
3. 如果涉及“原因—过程—结果”链路，优先在 temporalLogic 中使用：
   "CAUSAL_CHAIN(...)"
4. 如果同时存在多个时间条件，分别写成多个 WITHIN(...)
5. 如果存在“连续X天”“持续X天”，优先写成 DURATION(...)
6. 如果存在“X天内没有发生某事件”，优先写成 TIMEOUT(...)
7. 如果没有把握，不要补充额外逻辑
8. 结果必须是完整 JSON，不能是空字符串，不能带 markdown

输入 JSON：
{input_json}
"""


# =========================
# 3. 工具函数
# =========================

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def normalize_output_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(则)?判定为", "", text)
    text = re.sub(r"^(则)?触发", "", text)
    text = re.sub(r"^(则)?生成", "", text)
    text = re.sub(r"^(则)?判断为", "", text)
    return text.strip()


def is_valid_result(obj: Dict[str, Any]) -> bool:
    required_top_keys = {"ruleId", "rawText", "events", "booleanLogic", "temporalLogic", "output"}
    if not isinstance(obj, dict):
        return False
    if not required_top_keys.issubset(set(obj.keys())):
        return False

    if not isinstance(obj["ruleId"], str):
        return False
    if not isinstance(obj["rawText"], str):
        return False
    if not isinstance(obj["events"], list):
        return False
    if not isinstance(obj["booleanLogic"], str):
        return False
    if not isinstance(obj["temporalLogic"], list):
        return False
    if not isinstance(obj["output"], str):
        return False

    for e in obj["events"]:
        if not isinstance(e, dict):
            return False
        if "id" not in e or "text" not in e:
            return False
        if not isinstance(e["id"], str) or not isinstance(e["text"], str):
            return False

    for t in obj["temporalLogic"]:
        if not isinstance(t, str):
            return False

    return True


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    if not text:
        return None

    # 先直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 再从文本中提取最外层 JSON
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def send_request(user_prompt: str, api_key: str) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 1800,
    }

    resp = requests.post(
        BASE_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    obj = extract_json_from_text(content)
    if obj is None:
        raise ValueError(f"模型返回无法解析为 JSON：{content}")

    if not is_valid_result(obj):
        raise ValueError(f"模型返回 JSON 结构不合法：{json.dumps(obj, ensure_ascii=False)}")

    obj["output"] = normalize_output_text(obj["output"])
    return obj


def call_deepseek(input_obj: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    normal_prompt = USER_PROMPT_TEMPLATE.format(
        input_json=json.dumps(input_obj, ensure_ascii=False, indent=2)
    )

    fallback_prompt = FALLBACK_USER_PROMPT_TEMPLATE.format(
        input_json=json.dumps(input_obj, ensure_ascii=False, indent=2)
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        # 第一次尝试：正常提示词
        try:
            return send_request(normal_prompt, api_key)
        except Exception as e:
            last_error = f"normal_error={e}"

        # 第二次尝试：fallback 提示词
        try:
            return send_request(fallback_prompt, api_key)
        except Exception as e2:
            last_error = f"{last_error}; fallback_error={e2}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP)
        else:
            raise RuntimeError(last_error)


# =========================
# 4. 主流程
# =========================

def main() -> None:
    ensure_parent_dir(OUTPUT_FILE)
    ensure_parent_dir(ERROR_FILE)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if not api_key:
        raise ValueError("未读取到 DEEPSEEK_API_KEY，请先设置环境变量。")

    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError:
        raise ValueError("DEEPSEEK_API_KEY 中包含中文或非法字符，请重新设置。")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    success_count = 0
    error_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout, \
         open(ERROR_FILE, "w", encoding="utf-8") as ferr:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw_obj = json.loads(line)

                if not isinstance(raw_obj, dict):
                    raise ValueError("输入行不是 JSON 对象")

                if "ruleId" not in raw_obj or "rawText" not in raw_obj:
                    raise ValueError("输入缺少 ruleId 或 rawText")

                input_obj = {
                    "ruleId": raw_obj["ruleId"],
                    "rawText": raw_obj["rawText"],
                }

                result = call_deepseek(input_obj, api_key)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                success_count += 1
                print(f"[OK] line={line_no}, ruleId={raw_obj['ruleId']}")

            except Exception as e:
                error_record = {
                    "lineNo": line_no,
                    "rawLine": line,
                    "error": str(e),
                }
                ferr.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                error_count += 1
                print(f"[ERR] line={line_no}, error={e}")

    print("=" * 60)
    print(f"处理完成：成功 {success_count} 条，失败 {error_count} 条")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"错误文件：{ERROR_FILE}")


if __name__ == "__main__":
    main()