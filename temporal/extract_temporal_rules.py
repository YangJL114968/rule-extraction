import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

import requests

# =========================
# 1. 基本配置
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "test_rules.jsonl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "extract_results.jsonl")
ERROR_FILE = os.path.join(SCRIPT_DIR, "extract_errors.jsonl")

BASE_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-v4-pro"

MAX_RETRIES = 3
RETRY_SLEEP = 3
REQUEST_TIMEOUT = 120
MAX_TOKENS = 4000

# 合法的时序约束类型
VALID_TEMPORAL_TYPES: Set[str] = {
    "SEQUENCE", "WINDOW", "CONTINUOUS", "TIMEOUT", "CAUSAL_CHAIN",
}


# =========================
# 2. 提示词
# =========================

SYSTEM_PROMPT = """
你是一个"自然语言规则结构化抽取器"。
你的任务是把一条中文自然语言风险规则，转换成标准 JSON。

当前阶段在布尔逻辑抽取的基础上，增加"时序 / 时间约束"抽取，采用"两阶段抽取"方式：

第一阶段：抽取基本事件
- 把规则中的每个最小条件拆成独立事件，编号为 E1, E2, E3...
- 每个事件必须是一个可以单独成立的最小条件单元
- 不要把多个并列条件合并成一个事件
- 每个事件文本都必须包含明确、完整的主语。即使原文在并列分句或后续分句中省略了主语，也必须根据原文补全主语。
- 优先使用原文中实际执行动作或具有状态的主体，例如"企业"、"企业账户"、"贷款资金"、"第三方账户"、"收款方"。不要笼统地省略为"存在..."、"出现..."、"发生..."。
- 不要为了挂载时间约束而虚构"整个流程完成"、"出现上述任一资金流向"等汇总事件。时间约束应尽量引用原文中真实存在的最小事件。
- 输出结论不要放入 events，而是单独放入 output
- 否定必须通过 logic 中的 NOT 体现，不要写进事件文本里。事件文本必须用肯定式表述。
  例如"企业没有采购行为"应拆为：事件"企业有采购行为" + logic 中用 NOT E1
- 主语补全示例：
  原文："企业收到贷款后，短期内转入第三方账户"
  正确事件："企业收到贷款"、"贷款资金转入第三方账户"
  错误事件："收到贷款"、"转入第三方账户"
  原文："企业账户先发生资金分拆转出，再发生多级账户流转"
  正确事件："企业账户发生资金分拆转出"、"企业账户发生多级账户流转"
  错误事件："资金分拆转出"、"发生多级账户流转"

第二阶段：抽取布尔逻辑 + 时序约束
- 根据自然语言中的"并且、或者、不是"等表达，将事件组合成布尔表达式
- 只允许使用以下逻辑符号：AND, OR, NOT, ()
- 同时识别并抽取时序/时间约束
- 时序约束必须严格限定为 README 中定义的5类：顺序、窗口、持续、超时、因果链。不要自行扩展新的时序约束类型。

  1. SEQUENCE — 顺序逻辑：A先发生，B后发生。
     格式：{"type":"SEQUENCE","events":["E2","E3","E4"]}
     其中 events 数组按发生先后顺序排列，至少包含2个事件

  2. WINDOW — 窗口逻辑：多个事件必须在规定时间内发生。
     格式：{"type":"WINDOW","events":["E1","E2","E3"],"value":"30天"}
     events 是时间窗口覆盖的多个事件，至少包含2个事件；value 保留原文中的时间表述

  3. CONTINUOUS — 持续逻辑：某种异常状态持续一段时间才触发。
     格式：{"type":"CONTINUOUS","events":["E1","E2"],"value":"连续七日"}
     events 是持续存在的一个或多个异常事件；value 保留原文中的持续时间表述

  4. TIMEOUT — 超时逻辑：某事件发生后，规定时间内没有后续合理事件。
     格式：{"type":"TIMEOUT","trigger":"E1","missingEvents":["E2","E3"],"value":"30天"}
     trigger 是先发生的触发事件；missingEvents 是规定时间内没有发生的后续合理事件，必须以肯定式事件文本写入 events，并在 logic 中通过 NOT 表达缺失；value 保留原文中的超时时间表述

  5. CAUSAL_CHAIN — 因果链逻辑：形成"原因—过程—结果"链条。
     格式：{"type":"CAUSAL_CHAIN","events":["E1","E2","E3","E4"]}
     events 按照原因、过程、结果的链条顺序排列，至少包含3个事件

- "夜间"、"节假日"、"频繁"、"短期内"等描述本身不是独立的时序约束类型。只有当原文满足上述5类定义时，才写入 temporalConstraints。
- 结构流转逻辑和上下文约束逻辑不属于当前 temporalConstraints，不要混入时序约束数组。

输出要求：
你只能输出一个 JSON 对象，不要输出任何解释文字，不要输出 markdown。
JSON 字段固定为：
- ruleId
- rawText
- events
- logic
- temporalConstraints
- output

其中：
1. events 是数组，每个元素格式为：{"id":"E1","text":"事件文本"}
2. logic 是由 E1,E2,E3... 组成的布尔表达式
3. temporalConstraints 是时序约束数组，如果规则中没有时序内容则为空数组 []
   每个约束对象必须包含 "type" 字段，取值必须是上述5种之一
4. output 是规则最终输出的风险结论，尽量保留原文表述

规则：
- 如果规则存在歧义，优先保持最保守抽取，不要自行补充隐含条件
- 不要虚构不存在的事件
- 不要虚构不存在的时序约束
- 如果没有明确的时序触发词，temporalConstraints 设为 []
- 事件文本必须用肯定式表述，不要出现"没有"、"不是"等否定词在事件文本中
- events 中每个事件文本必须带有明确主语，不能省略主语，不能只写动作、状态或宾语
- 并列分句、顺承分句和省略句中的共享主语必须逐个补全；不同事件的真实主语不同时，不要强行统一为"企业"
- 不要增加额外字段
- events 中的事件 id 必须是 "E1","E2","E3"... 连续递增，不能跳号
- temporalConstraints 中引用的 events / trigger / missingEvents 必须对应 events 中存在的 id
- 同一段原文可以同时满足多种 README 时序逻辑，例如一个因果链也可能要求在30天窗口内完成；此时分别保留对应约束
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


def validate_temporal_constraints(
    constraints: Any, event_ids: Set[str]
) -> bool:
    """校验 temporalConstraints 字段的结构合法性。"""
    if not isinstance(constraints, list):
        return False

    for c in constraints:
        if not isinstance(c, dict):
            return False
        if "type" not in c or c["type"] not in VALID_TEMPORAL_TYPES:
            return False

        t = c["type"]

        if t in ("SEQUENCE", "CAUSAL_CHAIN"):
            if "events" not in c or not isinstance(c["events"], list):
                return False
            min_events = 2 if t == "SEQUENCE" else 3
            if len(c["events"]) < min_events:
                return False
            for eid in c["events"]:
                if not isinstance(eid, str) or eid not in event_ids:
                    return False

        elif t in ("WINDOW", "CONTINUOUS"):
            if "events" not in c or not isinstance(c["events"], list):
                return False
            min_events = 2 if t == "WINDOW" else 1
            if len(c["events"]) < min_events:
                return False
            for eid in c["events"]:
                if not isinstance(eid, str) or eid not in event_ids:
                    return False
            if "value" not in c or not isinstance(c["value"], str) or not c["value"].strip():
                return False

        elif t == "TIMEOUT":
            if "trigger" not in c:
                return False
            if not isinstance(c["trigger"], str) or c["trigger"] not in event_ids:
                return False
            if "value" not in c or not isinstance(c["value"], str) or not c["value"].strip():
                return False
            if "missingEvents" not in c or not isinstance(c["missingEvents"], list):
                return False
            if not c["missingEvents"]:
                return False
            for eid in c["missingEvents"]:
                if not isinstance(eid, str) or eid not in event_ids:
                    return False

    return True


def is_valid_result(obj: Dict[str, Any]) -> bool:
    required_top_keys = {"ruleId", "rawText", "events", "logic", "temporalConstraints", "output"}
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

    event_ids: Set[str] = set()
    for e in obj["events"]:
        if not isinstance(e, dict):
            return False
        if "id" not in e or "text" not in e:
            return False
        if not isinstance(e["id"], str) or not isinstance(e["text"], str):
            return False
        event_ids.add(e["id"])

    # 验证 temporalConstraints
    if not validate_temporal_constraints(obj["temporalConstraints"], event_ids):
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

def load_completed_ids() -> Set[str]:
    """读取已有结果文件，返回已完成的 ruleId 集合。"""
    completed: Set[str] = set()
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
    parser = argparse.ArgumentParser(description="时序逻辑规则抽取")
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

    completed_ids: Set[str] = set()
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
