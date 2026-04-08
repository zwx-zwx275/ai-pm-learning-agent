import os, requests, re, json, threading, time
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# ── 环境变量 ──────────────────────────────────────────────────────────────────
FEISHU_APP_ID     = os.environ.get("FEISHU_APP_ID")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET")
FEISHU_APP_TOKEN  = os.environ.get("FEISHU_APP_TOKEN")
TABLE_ID_FAST     = os.environ.get("TABLE_ID_FAST")
TABLE_ID_DEEP     = os.environ.get("TABLE_ID_DEEP")
TABLE_ID_HOLD     = os.environ.get("TABLE_ID_HOLD")
TABLE_ID_GRAPH    = os.environ.get("TABLE_ID_GRAPH")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# ── 用户背景 ──────────────────────────────────────────────────────────────────
USER_PROFILE = """
用户背景：
- 身份：从金融（银行）数据分析转行的 AI 产品经理，孩子不到1岁的妈妈
- 目标行业：旅游、金融、教育
- 学习阶段：AI PM 入门到进阶，正在学习如何用 AI 做产品
- 实战方向：希望能边学边做，逐步构建完整的 AI 产品项目
- 碎片化时间学习，需要高效筛选和消化 AI 相关知识
"""

ANALYSIS_PROMPT_TEMPLATE = """你是一位 AI 产品经理学习助手。请分析以下内容，帮助用户判断这个知识对她的价值，严格返回 JSON，禁止输出任何 JSON 以外的内容。

{user_profile}

内容来源：{source}
内容：
{content}

返回 JSON 结构：
{{
  "type": "只能是以下三种，判断标准如下：FAST（资讯/新闻/行业动态/即使以后也不能直接动手操作的内容，了解一下就够）/ ACTION（有具体方法论或工具，用户现在就具备上手条件）/ HOLD（有明确实操价值，但用户当前缺少某个具体的前置条件，等条件具备后能直接用上）。重要原则：没有实操性的内容一律归 FAST，不要因为'以后可能有用'就归 HOLD。HOLD 必须能回答'等什么条件具备后可以动手'这个问题。",
  "reason": "1-2句话解释分类理由，帮助用户自己做最终判断，语气直接",
  "content": {{
    "title": "简洁标题，15字以内",
    "summary": "### 核心是什么\\n- 一句话说明这个知识点是干什么的\\n### 对你有什么用\\n- 结合用户背景说明具体价值\\n### 关键要点\\n- 2-3个最重要的知识点",
    "main_cat": "能力大类，从以下选择：质量治理 / 产品设计 / AI工程 / 数据分析 / 增长运营 / 行业知识",
    "sub_dir": "子方向，能力大类下的具体细分，如：Prompt优化 / 评估体系 / 数据质量 / 用户研究 / Agent设计等",
    "action_task": "仅 ACTION 类填写，其他填空字符串。格式：\\n【目标】...\\n【预计时长】...\\n【步骤】\\n1. ...\\n2. ...\\n【验收标准】...",
    "activate_when": "仅 HOLD 类填写，其他填空字符串。描述什么情况下这个知识会变得有用",
    "related_project": "这个知识可以用在哪个实战项目里，如：旅游规划助手 / 金融风控助手，无明确关联则填空字符串",
    "graph_next": "该子方向下一步建议补充什么类型的内容"
  }}
}}"""

# ── 内存缓存 ──────────────────────────────────────────────────────────────────
session_cache: dict = {}
SESSION_TTL = 3600


def _expire_sessions():
    now = time.time()
    dead = [k for k, v in session_cache.items() if now - v.get("ts", 0) > SESSION_TTL]
    for k in dead:
        del session_cache[k]


# ── 飞书基础工具 ──────────────────────────────────────────────────────────────
def get_token() -> str:
    res = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
        timeout=10,
    )
    return res.json().get("tenant_access_token", "")


def send_reply(message_id: str, text: str):
    token = get_token()
    requests.post(
        f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": json.dumps({"text": text}), "msg_type": "text"},
        timeout=10,
    )


def send_message(chat_id: str, text: str):
    """主动发消息给指定 chat"""
    token = get_token()
    requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        headers={"Authorization": f"Bearer {token}"},
        params={"receive_id_type": "chat_id"},
        json={
            "receive_id": chat_id,
            "content": json.dumps({"text": text}),
            "msg_type": "text",
        },
        timeout=10,
    )


def write_record(table_id: str, fields: dict) -> bool:
    token = get_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_APP_TOKEN}/tables/{table_id}/records"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}

    res = requests.post(url, headers=headers, json={"fields": fields}, timeout=10)
    if res.json().get("code") == 0:
        return True

    fallback = {k: (v["link"] if isinstance(v, dict) and "link" in v else v) for k, v in fields.items()}
    res2 = requests.post(url, headers=headers, json={"fields": fallback}, timeout=10)
    if res2.json().get("code") == 0:
        return True

    print(f"❌ 入库失败: {res2.text}")
    return False


def search_graph_record(main_cat: str, sub_dir: str):
    token = get_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_APP_TOKEN}/tables/{TABLE_ID_GRAPH}/records/search"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "filter": {
            "conjunction": "and",
            "conditions": [
                {"field_name": "能力大类", "operator": "is", "value": [main_cat]},
                {"field_name": "子方向",   "operator": "is", "value": [sub_dir]},
            ]
        }
    }
    try:
        res = requests.post(url, headers=headers, json=body, timeout=10)
        items = res.json().get("data", {}).get("items", [])
        return items[0] if items else None
    except Exception:
        return None


def update_graph_record(record_id: str, fields: dict):
    token = get_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_APP_TOKEN}/tables/{TABLE_ID_GRAPH}/records/{record_id}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    requests.put(url, headers=headers, json={"fields": fields}, timeout=10)


def get_all_records(table_id: str) -> list:
    token = get_token()
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_APP_TOKEN}/tables/{table_id}/records"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.get(url, headers=headers, params={"page_size": 100}, timeout=10)
        return res.json().get("data", {}).get("items", [])
    except Exception:
        return []


# ── 知识图谱更新 ──────────────────────────────────────────────────────────────
def update_knowledge_graph(item: dict, kind: str):
    if not TABLE_ID_GRAPH:
        return
    try:
        main_cat = str(item.get("main_cat", "")).strip()
        sub_dir  = str(item.get("sub_dir", "")).strip()
        if not main_cat or not sub_dir:
            return

        existing = search_graph_record(main_cat, sub_dir)
        today = time.strftime("%Y-%m-%d")

        if existing:
            record_id = existing["record_id"]
            old_fields = existing.get("fields", {})
            old_count  = int(old_fields.get("内容数字", 0))
            new_count  = old_count + 1
            old_level  = str(old_fields.get("掌握程度", "空白"))
            new_level  = old_level
            if old_level == "空白":
                new_level = "了解中"
            elif old_level == "了解中" and kind == "ACTION":
                new_level = "有实践"
            update_graph_record(record_id, {
                "内容数字": new_count,
                "掌握程度": new_level,
                "最近更新": today,
            })
        else:
            init_level = "了解中" if kind == "ACTION" else "空白"
            write_record(TABLE_ID_GRAPH, {
                "能力大类": main_cat,
                "子方向":   sub_dir,
                "掌握程度": init_level,
                "内容数字": 1,
                "关联项目": str(item.get("related_project", "")).strip(),
                "下一步":   str(item.get("graph_next", "继续收集相关内容")).strip(),
                "最近更新": today,
            })
    except Exception as e:
        print(f"⚠️ 知识图谱更新失败: {e}")


# ── 入库 ──────────────────────────────────────────────────────────────────────
def commit_to_bitable(chat_id: str, message_id: str, override_kind: str = ""):
    _expire_sessions()
    session = session_cache.get(chat_id)
    if not session or "last_analysis" not in session:
        send_reply(message_id, "❌ 没有找到待入库的内容，请重新发送链接。")
        return

    analysis = session["last_analysis"]
    item     = analysis.get("content", {})
    kind     = override_kind.upper() if override_kind else str(analysis.get("type", "FAST")).upper()
    raw_url  = str(session.get("url") or "").strip()
    url_field = {"link": raw_url, "text": "查看原文"} if raw_url else None

    if kind == "FAST":
        table_id, table_name = TABLE_ID_FAST, "快读资讯库"
        fields = {
            "标题":       str(item.get("title", "未命名")).strip(),
            "高价值总结": str(item.get("summary", "")).strip(),
            "能力大类":   str(item.get("main_cat", "")).strip(),
            "子方向":     str(item.get("sub_dir", "")).strip(),
        }
    elif kind == "ACTION":
        table_id, table_name = TABLE_ID_DEEP, "上手实战库"
        fields = {
            "标题":       str(item.get("title", "未命名")).strip(),
            "高价值总结": str(item.get("summary", "")).strip(),
            "能力大类":   str(item.get("main_cat", "")).strip(),
            "子方向":     str(item.get("sub_dir", "")).strip(),
            "实战任务":   str(item.get("action_task", "")).strip(),
            "内化状态":   "待学习",
        }
    else:
        table_id, table_name = TABLE_ID_HOLD, "待激活库"
        fields = {
            "标题":     str(item.get("title", "未命名")).strip(),
            "内容摘要": str(item.get("summary", "")).strip(),
            "能力大类": str(item.get("main_cat", "")).strip(),
            "子方向":   str(item.get("sub_dir", "")).strip(),
            "激活条件": str(item.get("activate_when", "")).strip(),
        }

    if url_field:
        fields["原文链接"] = url_field

    ok = write_record(table_id, fields)
    if ok:
        send_reply(message_id, f"✅ 已入库 →「{table_name}」\n继续发链接，或回复「整合」查看知识整合报告。")
        session_cache.pop(chat_id, None)
        threading.Thread(target=update_knowledge_graph, args=(item, kind), daemon=True).start()
    else:
        send_reply(message_id, "❌ 入库失败，请检查飞书列名是否与代码一致。")


# ── AI 分析核心（共用）────────────────────────────────────────────────────────
def run_ai_analysis(chat_id: str, message_id: str, source_url: str, raw_content: str):
    """拿到内容后统一走这里做 AI 分析，无论来源是 Jina、OCR 还是用户粘贴"""
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        user_profile=USER_PROFILE,
        source=source_url,
        content=raw_content[:5000],
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=60,
        )
        res_data = json.loads(response.choices[0].message.content)
    except Exception as e:
        send_reply(message_id, f"❌ AI 分析失败: {e}")
        return

    item = res_data.get("content", {})
    if not item and "title" in res_data:
        item = res_data

    _expire_sessions()
    session_cache[chat_id] = {
        "url":           source_url,
        "raw_content":   raw_content,
        "last_analysis": {"type": res_data.get("type", "FAST"), "content": item},
        "ts":            time.time(),
    }

    kind = str(res_data.get("type", "FAST")).upper()
    kind_labels = {"FAST": "⚡ 快读资讯库", "ACTION": "🔥 上手实战库", "HOLD": "📦 待激活库"}
    kind_label  = kind_labels.get(kind, "⚡ 快读资讯库")

    lines = [
        f"📁 建议分流：{kind_label}",
        f"💬 {res_data.get('reason', '')}",
        "",
        f"📌 {item.get('title', '解析失败')}",
        "",
        item.get("summary", "摘要生成失败"),
        "",
        f"📂 {item.get('main_cat', '')} › {item.get('sub_dir', '')}",
    ]

    if kind == "ACTION" and item.get("action_task"):
        lines += ["", f"🛠️ 实战任务：\n{item.get('action_task')}"]
    elif kind == "HOLD" and item.get("activate_when"):
        lines += ["", f"⏰ 激活条件：{item.get('activate_when')}"]

    if item.get("related_project"):
        lines.append(f"🔗 关联项目：{item.get('related_project')}")

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━",
        "✅ 回复 1 → 按建议入库",
        "📝 回复「快读」「上手」「存档」→ 覆盖分类后入库",
    ]

    send_reply(message_id, "\n".join(lines))
    print("✅ 分析完成，等待用户确认")


# ── 链接抓取 ──────────────────────────────────────────────────────────────────
def ai_analyze(chat_id: str, message_id: str, target_url: str):
    print(f"🧠 开始分析: {target_url}")
    send_reply(message_id, "⏳ 正在抓取并分析，请稍候……")

    try:
        res = requests.get(f"https://r.jina.ai/{target_url}", timeout=20)
        raw_content = res.text[:6000]

        # 检测是否是拦截/错误页面而非真实内容
        # 逻辑：包含拦截关键词 AND 内容很短，才判定为被拦截
        # 真实文章就算包含"安全"等词，内容也远超2000字
        BLOCK_SIGNALS = [
            "451", "blocked", "安全", "拦截", "验证码",
            "access denied", "forbidden", "error"
        ]
        is_blocked = (
            any(s in raw_content.lower() for s in BLOCK_SIGNALS)
            and len(raw_content) < 2000
        )

        if not raw_content.strip() or len(raw_content) < 100 or is_blocked:
            raise ValueError("内容被拦截或无效")

    except Exception as e:
        print(f"⚠️ 链接抓取失败: {e}，等待用户粘贴内容")
        _expire_sessions()
        session_cache[chat_id] = {
            "url": target_url,
            "waiting_for_text": True,
            "ts": time.time(),
        }
        send_reply(
            message_id,
            "📋 无法自动读取链接内容（可能是小红书、公众号等需要登录的平台）\n\n"
            "请把文章正文复制粘贴发给我，我来帮你分析 👇",
        )
        return

    run_ai_analysis(chat_id, message_id, target_url, raw_content)


# ── 整合报告 ──────────────────────────────────────────────────────────────────
def generate_digest(message_id: str):
    send_reply(message_id, "⏳ 正在生成整合报告，请稍候……")

    try:
        fast_records   = get_all_records(TABLE_ID_FAST)
        action_records = get_all_records(TABLE_ID_DEEP)
        hold_records   = get_all_records(TABLE_ID_HOLD)
        graph_records  = get_all_records(TABLE_ID_GRAPH)
    except Exception as e:
        send_reply(message_id, f"❌ 读取表格失败: {e}")
        return

    def fmt_records(records, title_key="标题", cat_key="能力大类", sub_key="子方向"):
        out = []
        for r in records[-15:]:
            f = r.get("fields", {})
            out.append(f"- 【{f.get(cat_key,'')}›{f.get(sub_key,'')}】{f.get(title_key,'无标题')}")
        return "\n".join(out) if out else "（暂无内容）"

    def fmt_graph(records):
        out = []
        for r in records:
            f = r.get("fields", {})
            out.append(
                f"- {f.get('能力大类','')}›{f.get('子方向','')} "
                f"[{f.get('掌握程度','空白')}] {f.get('内容数字',0)}条"
                f" 关联:{f.get('关联项目','无')}"
            )
        return "\n".join(out) if out else "（暂无内容）"

    prompt = f"""你是一位 AI PM 学习教练。请基于以下知识库数据，生成一份整合报告。

{USER_PROFILE}

【快读资讯库】
{fmt_records(fast_records)}

【上手实战库】
{fmt_records(action_records)}

【待激活库】
{fmt_records(hold_records, title_key="标题", cat_key="能力大类", sub_key="子方向")}

【知识图谱现状】
{fmt_graph(graph_records)}

请生成整合报告，包含：
1. 📚 最近学了什么（按能力大类归组，简洁概括）
2. 🔗 知识串联发现（哪些内容可以组合用在同一个项目，要具体说出项目名和涉及的知识点）
3. 💡 待激活库唤醒（结合新内容，有没有之前存档的知识现在可以激活了，如果有请点名）
4. 🗺️ 技能地图（哪些子方向有积累，哪些是空白需要补）
5. 🎯 本周推荐实战（具体到明天就能开始做的一个任务）

语气像了解她背景的学习搭档，简洁直接。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            timeout=90,
        )
        report = response.choices[0].message.content
        send_reply(message_id, f"📊 知识整合报告\n{'━'*18}\n{report}")
    except Exception as e:
        send_reply(message_id, f"❌ 生成报告失败: {e}")


# ── Webhook 入口 ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return "AI PM Learning Agent 2.0 OK", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json or {}

    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    msg   = event.get("message", {})

    try:
        content = json.loads(msg.get("content", "{}"))
        text    = content.get("text", "").strip()
    except Exception:
        return "OK"

    chat_id = msg.get("chat_id", "")
    msg_id  = msg.get("message_id", "")

    if not chat_id or not msg_id or not text:
        return "OK"

    print(f"📨 收到: {text[:80]}")

    # ── 系统指令（优先级最高）────────────────────────────────────────────────
    if text == "1":
        threading.Thread(target=commit_to_bitable, args=(chat_id, msg_id), daemon=True).start()
        return "OK"
    elif text in ("快读", "fast"):
        threading.Thread(target=commit_to_bitable, args=(chat_id, msg_id, "FAST"), daemon=True).start()
        return "OK"
    elif text in ("上手", "action"):
        threading.Thread(target=commit_to_bitable, args=(chat_id, msg_id, "ACTION"), daemon=True).start()
        return "OK"
    elif text in ("存档", "hold"):
        threading.Thread(target=commit_to_bitable, args=(chat_id, msg_id, "HOLD"), daemon=True).start()
        return "OK"
    elif text in ("整合", "digest", "报告"):
        threading.Thread(target=generate_digest, args=(msg_id,), daemon=True).start()
        return "OK"
    elif text in ("帮助", "help", "?", "？"):
        send_reply(msg_id, (
            "📖 AI PM 学习助手 v2.0\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "发链接 → AI 分析并给分类建议\n"
            "  • 能自动读取：直接分析\n"
            "  • 无法读取（小红书/公众号）：提示你粘贴正文\n"
            "粘贴正文 → 继续上一个链接的分析\n"
            "回复 1 → 按建议入库\n"
            "回复「快读」→ 入快读资讯库\n"
            "回复「上手」→ 入上手实战库\n"
            "回复「存档」→ 入待激活库\n"
            "回复「整合」→ 生成知识整合报告\n"
            "━━━━━━━━━━━━━━━━━━"
        ))
        return "OK"

    # ── 检查是否在等待用户粘贴内容 ───────────────────────────────────────────
    _expire_sessions()
    session = session_cache.get(chat_id, {})
    if session.get("waiting_for_text"):
        source_url = session.get("url", "")
        print(f"📝 收到粘贴内容，来源链接: {source_url[:60]}")
        threading.Thread(
            target=run_ai_analysis,
            args=(chat_id, msg_id, source_url, text),
            daemon=True,
        ).start()
        return "OK"

    # ── 链接处理 ──────────────────────────────────────────────────────────────
    match = re.search(r'https?://[^\s<>"]+', text)
    if match:
        threading.Thread(
            target=ai_analyze,
            args=(chat_id, msg_id, match.group()),
            daemon=True,
        ).start()
    else:
        send_reply(msg_id, "🤔 发送链接开始分析，或回复「帮助」查看指南。")

    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)