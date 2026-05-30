import argparse
import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests

# =========================
# 1. 基本配置
# =========================

INPUT_FILE = "/Users/yangjunlong/bool/bool/raw_rules.jsonl"
OUTPUT_FILE = "/Users/yangjunlong/bool/bool/extract_results.jsonl"
ERROR_FILE = "/Users/yangjunlong/bool/bool/extract_errors.jsonl"

BASE_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-v4-pro"   # 可改成 deepseek-v4-flash

MAX_RETRIES = 3
RETRY_SLEEP = 3
REQUEST_TIMEOUT = 120
MAX_TOKENS = 4000


# =========================
# 2. 提示词
# =========================

SYSTEM_PROMPT = """
你是一个“自然语言规则结构化抽取器”。
你的任务是把一条中文自然语言风险规则，转换成标准 JSON。

当前阶段只处理“布尔逻辑规则抽取”，采用“两阶段抽取”方式：

第一阶段：抽取基本事件
- 把规则中的每个最小条件拆成独立事件，编号为 E1, E2, E3...
- 每个事件必须是一个可以单独成立的最小条件单元
- 不要把多个并列条件合并成一个事件
- 输出结论不要放入 events，而是单独放入 output

第二阶段：抽取布尔逻辑
- 根据自然语言中的“并且、或者、不是”等表达，将事件组合成布尔表达式
- 当前只允许使用以下逻辑符号：
  AND
  OR
  NOT
  ()
- 当前阶段不要处理时序逻辑、窗口逻辑、结构流转逻辑、上下文约束逻辑

输出要求：
你只能输出一个 JSON 对象，不要输出任何解释文字，不要输出 markdown。
JSON 字段固定为：
- ruleId
- rawText
- events
- logic
- output

其中：
1. events 是数组，每个元素格式为：
   {"id":"E1","text":"事件文本"}

2. logic 是由 E1, E2, E3... 组成的布尔表达式，例如：
   "E1 AND E2"
   "E1 OR E2"
   "E1 AND NOT E2"
   "(E1 OR E2) AND E3"

3. output 是规则最终输出的风险结论，尽量保留原文表述

规则：
- 如果规则存在歧义，优先保持最保守抽取，不要自行补充隐含条件
- 不要虚构不存在的事件
- 不要增加额外字段
- 否定必须体现在 logic 中（用 NOT），不要写进事件文本里。事件文本必须用肯定式表述。例如"企业没有采购行为"应拆为：事件"企业有采购行为" + logic 中用 NOT E1
"""

USER_PROMPT_TEMPLATE = """
请对下面这条规则进行抽取。

输入 JSON：
{input_json}

请直接输出最终 JSON 对象。
"""


# =========================
# 3. 工具函数
# =========================

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def is_valid_result(obj: Dict[str, Any]) -> bool:
    required_top_keys = {"ruleId", "rawText", "events", "logic", "output"}
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
    if not isinstance(obj["logic"], str):
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

    return True


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

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


def call_deepseek(input_obj: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    user_prompt = USER_PROMPT_TEMPLATE.format(
        input_json=json.dumps(input_obj, ensure_ascii=False, indent=2)
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": MAX_TOKENS,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
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
                raise ValueError(
                    f"模型返回 JSON 结构不合法：{json.dumps(obj, ensure_ascii=False)}"
                )

            return obj

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
            else:
                raise RuntimeError(last_error)


# =========================
# 4. 主流程
# =========================

def load_completed_ids() -> set:
    """读取已有结果文件，返回已完成的 ruleId 集合。"""
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "ruleId" in obj:
                        completed.add(obj["ruleId"])
                except Exception:
                    pass
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="布尔逻辑规则抽取")
    parser.add_argument("--full", action="store_true", help="全部重新执行（清除已有结果和错误记录）")
    args = parser.parse_args()

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

    completed_ids: set = set()
    success_count = 0
    error_count = 0

    if args.full:
        print("=== 全量重新执行模式 ===")
        # 清空结果和错误文件
        open(OUTPUT_FILE, "w", encoding="utf-8").close()
        open(ERROR_FILE, "w", encoding="utf-8").close()
    else:
        completed_ids = load_completed_ids()
        if completed_ids:
            print(f"=== 增量执行模式：已有 {len(completed_ids)} 条成功结果将跳过 ===")
        else:
            print("=== 增量执行模式：无已有结果，将全量执行 ===")
        print("将处理：失败重试 + 新增规则")
        # 清空错误文件，失败的要重跑
        open(ERROR_FILE, "w", encoding="utf-8").close()

    out_mode = "w" if args.full else "a"

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, out_mode, encoding="utf-8") as fout, \
         open(ERROR_FILE, "a", encoding="utf-8") as ferr:

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

                rule_id = raw_obj["ruleId"]

                # 跳过已成功的
                if rule_id in completed_ids:
                    print(f"[SKIP] line={line_no}, ruleId={rule_id} (已完成)")
                    continue

                input_obj = {
                    "ruleId": rule_id,
                    "rawText": raw_obj["rawText"],
                }

                result = call_deepseek(input_obj, api_key)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                success_count += 1
                print(f"[OK] line={line_no}, ruleId={rule_id}")

            except Exception as e:
                error_record = {
                    "lineNo": line_no,
                    "rawLine": line,
                    "error": str(e),
                }
                ferr.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                ferr.flush()
                error_count += 1
                print(f"[ERR] line={line_no}, error={e}")

    print("=" * 60)
    print(f"本次新增：成功 {success_count} 条，失败 {error_count} 条")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"错误文件：{ERROR_FILE}")


if __name__ == "__main__":
    main()