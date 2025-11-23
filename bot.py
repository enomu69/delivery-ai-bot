import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============= CONFIG =============

TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

TIMEZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"  # full NLP


# logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Per-chat config (default summary mode etc) - in-memory only
CHAT_CONFIG: Dict[int, Dict[str, Any]] = {}
DEFAULT_SUMMARY_MODE = "full"  # or "medium" / "compact"


def get_chat_config(chat_id: int) -> Dict[str, Any]:
    if chat_id not in CHAT_CONFIG:
        CHAT_CONFIG[chat_id] = {
            "summary_mode": DEFAULT_SUMMARY_MODE,
        }
    return CHAT_CONFIG[chat_id]


# ============= SUPABASE HELPERS =============

def supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def supabase_insert_orders(
    chat_id: int,
    orders: List[Dict[str, Any]],
    msg_time_utc: datetime,
):
    """
    Insert new orders as status 'pending'.
    Each order: {"location": str, "amount": float}
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error("Supabase config missing; cannot insert orders.")
        return

    url = f"{SUPABASE_URL}/rest/v1/orders"

    # start of calendar week (Monday 00:00) in UTC
    local = msg_time_utc.astimezone(TIMEZONE)
    days_since_monday = local.weekday()
    monday_local = (local - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_start_utc = monday_local.astimezone(UTC)

    rows = []
    for o in orders:
        rows.append(
            {
                "chat_id": str(chat_id),
                "location": o["location"],
                "amount": float(o["amount"]),
                "order_timestamp": msg_time_utc.isoformat(),
                "week_start": week_start_utc.isoformat(),
                "status": "pending",
            }
        )

    try:
        params = {"select": "id,location,amount,status"}
        headers = supabase_headers()
        headers["Prefer"] = "return=representation"
        resp = requests.post(
            url,
            params=params,
            headers=headers,
            data=json.dumps(rows),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Inserted {len(data)} orders into Supabase for chat {chat_id}")
    except Exception as e:
        logger.error(f"Supabase insert error: {e}")


def supabase_fetch_orders(
    chat_id: int,
    status: Optional[str] = None,
    start_utc: Optional[datetime] = None,
    end_utc: Optional[datetime] = None,
    limit: Optional[int] = None,
    order_desc: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch orders for a chat with optional filters.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error("Supabase config missing; cannot fetch orders.")
        return []

    url = f"{SUPABASE_URL}/rest/v1/orders"

    params_list = [
        ("select", "id,chat_id,location,amount,order_timestamp,status"),
        ("chat_id", f"eq.{chat_id}"),
    ]

    if status:
        params_list.append(("status", f"eq.{status}"))
    if start_utc:
        params_list.append(("order_timestamp", f"gte.{start_utc.isoformat()}"))
    if end_utc:
        params_list.append(("order_timestamp", f"lt.{end_utc.isoformat()}"))
    if limit:
        params_list.append(("limit", str(limit)))
    order_dir = "desc" if order_desc else "asc"
    params_list.append(("order", f"order_timestamp.{order_dir}"))

    params = urlencode(params_list, doseq=True)

    try:
        resp = requests.get(
            url,
            headers=supabase_headers(),
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        logger.error(f"Supabase fetch error: {e}")
        return []


def supabase_fetch_pending(chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch latest pending orders (most recent first).
    """
    return supabase_fetch_orders(
        chat_id=chat_id,
        status="pending",
        start_utc=None,
        end_utc=None,
        limit=limit,
        order_desc=True,
    )


def supabase_update_status(order_ids: List[int], new_status: str):
    """
    Update status for the given order IDs.
    """
    if not order_ids:
        return
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.error("Supabase config missing; cannot update status.")
        return

    url = f"{SUPABASE_URL}/rest/v1/orders"
    headers = supabase_headers()
    headers["Prefer"] = "return=representation"

    for oid in order_ids:
        params = {
            "id": f"eq.{oid}",
            "select": "id,location,amount,status",
        }
        try:
            resp = requests.patch(
                url,
                params=params,
                headers=headers,
                data=json.dumps({"status": new_status}),
                timeout=20,
            )
            resp.raise_for_status()
            logger.info(f"Updated order {oid} to status={new_status}")
        except Exception as e:
            logger.error(f"Supabase status update error for id={oid}: {e}")


# ============= OPENAI FULL NLP ANALYZER =============

def call_openai_analyzer(
    message_text: str,
    pending_orders: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Ask OpenAI to decide:
    - is this message accepting/rejecting existing orders?
    - or creating new orders?
    - or irrelevant?

    Returns a dict:
    {
      "action": "confirm" | "reject" | "orders" | "ignore",
      "accepted_indices": [1,2],
      "rejected_indices": [3],
      "new_orders": [{"location": "...", "amount": 35.0}]
    }
    where indices refer to the numbered pending_orders context we provide.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set.")
        return None

    # Build pending orders context (indexed)
    if pending_orders:
        lines = []
        for idx, o in enumerate(pending_orders, start=1):
            loc = o.get("location", "")
            amt = float(o.get("amount", 0) or 0)
            ts_str = o.get("order_timestamp") or ""
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts_local = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_local = ts_str
            lines.append(f"{idx}) {loc}, ${amt:.2f}, time={ts_local}")
        pending_text = "\n".join(lines)
    else:
        pending_text = "(none)"

    system_prompt = """
You are an assistant for a weed delivery dispatch system.

There is ONE driver per chat. The dispatcher posts offers as "pending orders".
The driver later sends messages which may ACCEPT or REJECT some of those pending orders.
Sometimes, the dispatcher is also sending NEW orders in the same chat.

You are given:
- A list of CURRENT PENDING ORDERS, each with an index starting at 1.
- A NEW chat message from this driver/dispatcher.

Your job:
1) Decide what the message is doing:
   - ACCEPTING one or more pending orders
   - REJECTING one or more pending orders
   - DESCRIBING NEW orders (for this driver)
   - Or IRRELEVANT (small talk, questions, etc.)

2) If the message accepts orders:
   - Use "action": "confirm".
   - Put the indices of accepted pending orders in "accepted_indices".
   - If the message clearly accepts ALL pending orders ("I'll take both", "I'll take everything", "I'll take them", "send it", etc.), put ALL indices.
   - Treat positive emojis like 👍 ❤️ 🙌 ✔️ ✅ 💯 as accepting the relevant pending orders.

3) If the message rejects orders:
   - Use "action": "reject" (or also use 'confirm' if it both accepts and rejects different ones).
   - Put indices of rejected pending orders in "rejected_indices".
   - Treat negative emojis like 👎 ❌ 🚫 ⛔ as rejecting the relevant pending orders.

4) If the message describes NEW orders:
   - Use "action": "orders".
   - Extract each order into an object {"location": string, "amount": float}.
   - Ignore time windows, emojis, or extra commentary.
   - If no valid orders are described, leave "new_orders" as an empty list.

5) If the message does not clearly refer to any pending orders and does not clearly describe new orders:
   - Use "action": "ignore".
   - Keep all lists empty.

IMPORTANT:
- "accepted_indices" and "rejected_indices" must reference the numbered pending orders (1,2,3,...).
- The message may both accept and reject different orders at once (e.g., accept index 1, reject index 2).
- Always include ALL keys: "action", "accepted_indices", "rejected_indices", "new_orders" in the JSON.

Return ONLY JSON, no explanations.
"""

    user_prompt = f"""
CURRENT PENDING ORDERS (for this driver/chat):
{pending_text}

NEW MESSAGE:
\"\"\"{message_text}\"\"\"

Now output JSON like:
{{
  "action": "confirm" | "reject" | "orders" | "ignore",
  "accepted_indices": [1,2],
  "rejected_indices": [3],
  "new_orders": [
    {{"location": "Example", "amount": 35.0}}
  ]
}}
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI analyzer error: {e}")
        return None

    # Strip code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            content = content.split("\n", 1)[1].strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Analyzer JSON parse error: {e} | content={content[:200]}")
        return None

    action = parsed.get("action", "ignore")
    accepted_indices = parsed.get("accepted_indices", []) or []
    rejected_indices = parsed.get("rejected_indices", []) or []
    new_orders = parsed.get("new_orders", []) or []

    def to_int_list(x):
        res = []
        for v in x:
            try:
                iv = int(v)
                res.append(iv)
            except Exception:
                continue
        return res

    accepted_indices = to_int_list(accepted_indices)
    rejected_indices = to_int_list(rejected_indices)

    cleaned_new_orders: List[Dict[str, Any]] = []
    for item in new_orders:
        try:
            loc = str(item.get("location", "")).strip()
            amt = float(item.get("amount", 0))
            if loc and amt > 0:
                cleaned_new_orders.append({"location": loc, "amount": amt})
        except Exception:
            continue

    return {
        "action": str(action),
        "accepted_indices": accepted_indices,
        "rejected_indices": rejected_indices,
        "new_orders": cleaned_new_orders,
    }


# ============= DATE / SUMMARY HELPERS =============

def start_of_today_utc() -> datetime:
    now_local = datetime.now(tz=TIMEZONE)
    today_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return today_local.astimezone(UTC)


def start_of_week_utc() -> datetime:
    now_local = datetime.now(tz=TIMEZONE)
    days_since_monday = now_local.weekday()
    monday_local = (now_local - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return monday_local.astimezone(UTC)


def days_ago_utc(days: int) -> datetime:
    now_utc = datetime.now(tz=UTC)
    return now_utc - timedelta(days=days)


def summarize_orders(
    orders: List[Dict[str, Any]],
    label: str,
    summary_mode: str = "full",
) -> str:
    """
    Build a human-readable summary.
    - full: per-day totals + each order
    - medium: per-day totals only
    - compact: just overall stats
    """
    if not orders:
        return f"{label}\n\nNo accepted orders in this period."

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    total_amount = 0.0

    for o in orders:
        ts_str = o.get("order_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            local_date = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d (%a)")
        except Exception:
            local_date = "Unknown date"
        grouped.setdefault(local_date, []).append(o)
        total_amount += float(o["amount"])

    lines: List[str] = [label, ""]
    num_days = len(grouped)

    # day-level
    for date_key in sorted(grouped.keys()):
        day_orders = grouped[date_key]
        day_total = sum(float(o["amount"]) for o in day_orders)

        if summary_mode in ("full", "medium"):
            lines.append(f"{date_key}: ${day_total:.2f}")

        if summary_mode == "full":
            for o in sorted(day_orders, key=lambda x: x["order_timestamp"]):
                lines.append(f"  - {o['location']}: ${float(o['amount']):.2f}")
            lines.append("")

    if summary_mode == "compact":
        lines = [label, ""]

    # overall stats
    lines.append(f"TOTAL: ${total_amount:.2f}")
    avg_per_day = total_amount / num_days if num_days > 0 else 0.0
    lines.append(f"Avg per active day: ${avg_per_day:.2f}")

    # simple location breakdown
    per_location: Dict[str, float] = {}
    for o in orders:
        loc = str(o.get("location", "Unknown"))
        amt = float(o.get("amount", 0))
        per_location[loc] = per_location.get(loc, 0.0) + amt

    if summary_mode in ("full", "medium"):
        lines.append("")
        lines.append("By location:")
        for loc, amt in sorted(per_location.items(), key=lambda x: -x[1]):
            lines.append(f"- {loc}: ${amt:.2f}")

    return "\n".join(lines)


def get_mode_from_args(chat_id: int, args: List[str]) -> str:
    """
    If args[0] is "full|medium|compact", use that.
    Otherwise use this chat's stored default mode.
    """
    cfg = get_chat_config(chat_id)
    if args:
        candidate = args[0].lower()
        if candidate in ("full", "medium", "compact"):
            return candidate
    return cfg["summary_mode"]


# ============= COMMAND HANDLERS =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I'm your delivery totals bot.\n\n"
        "How I work (NLP mode):\n"
        "• Dispatcher posts offers in the chat (locations + $ amounts) → I save them as *pending* orders.\n"
        "• When the driver confirms (\"send it\", \"coming\", \"I'll take it\", 👍, etc.), "
        "I mark those pending orders as *accepted*.\n"
        "• When the driver rejects (\"can't\", \"nah\", 👎, etc.), I mark them as *rejected*.\n"
        "• Only *accepted* orders count toward your totals & stats.\n\n"
        "Commands:\n"
        "/today – accepted orders today\n"
        "/totals – accepted orders this calendar week (since Monday)\n"
        "/last7days – accepted orders in last 7 days\n"
        "/last14days – last 14 days\n"
        "/last30days – last 30 days\n"
        "/last60days – last 60 days\n"
        "/last90days – last 90 days\n"
        "/last180days – last 180 days\n"
        "/last365days – last 365 days\n"
        "/alltime – all accepted orders ever\n"
        "/statsrange YYYY-MM-DD YYYY-MM-DD [full|medium|compact] – custom date range\n"
        "/pending – show recent pending orders\n"
        "/mode full|medium|compact – set how detailed summaries are\n"
        "/help – show this again\n\n"
        "Tip: you can also do /last30days compact or /totals full to override the mode just for that command."
    )
    if update.message:
        await update.message.reply_text(msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    args = context.args
    cfg = get_chat_config(chat_id)

    if not args:
        await message.reply_text(
            f"Current summary mode: {cfg['summary_mode']}\n"
            "Options: full, medium, compact\n"
            "Example: /mode compact"
        )
        return

    choice = args[0].lower()
    if choice not in ("full", "medium", "compact"):
        await message.reply_text("Invalid mode. Use one of: full, medium, compact.")
        return

    cfg["summary_mode"] = choice
    await message.reply_text(f"Summary mode set to: {choice}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main message handler:
    - Always calls OpenAI analyzer with current pending orders.
    - Analyzer decides:
      • action: confirm / reject / orders / ignore
      • which pending indices accepted/rejected
      • any new orders in this message
    """
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    text = message.text

    msg_time = message.date
    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=UTC)
    msg_time_utc = msg_time.astimezone(UTC)

    # Fetch recent pending orders for this chat (most recent first)
    pending = supabase_fetch_pending(chat_id, limit=10)

    analysis = call_openai_analyzer(text, pending)
    if analysis is None:
        return

    action = analysis["action"]
    accepted_indices = analysis["accepted_indices"]
    rejected_indices = analysis["rejected_indices"]
    new_orders = analysis["new_orders"]

    # Map indices -> Supabase IDs using 'pending' list
    def indices_to_ids(indices: List[int]) -> List[int]:
        ids: List[int] = []
        for idx in indices:
            if 1 <= idx <= len(pending):
                oid = pending[idx - 1].get("id")
                if oid is not None:
                    try:
                        ids.append(int(oid))
                    except Exception:
                        continue
        return ids

    accepted_ids = indices_to_ids(accepted_indices)
    rejected_ids = indices_to_ids(rejected_indices)

    # Insert newly described orders as pending
    if new_orders:
        supabase_insert_orders(chat_id, new_orders, msg_time_utc)

    # Update statuses based on acceptance/rejection
    if accepted_ids:
        supabase_update_status(accepted_ids, "accepted")
    if rejected_ids:
        supabase_update_status(rejected_ids, "rejected")

    # Build feedback message
    parts = []

    if new_orders:
        lines = [f"Stored {len(new_orders)} pending order(s):"]
        for o in new_orders:
            lines.append(f"- {o['location']}: ${float(o['amount']):.2f}")
        parts.append("\n".join(lines))

    if accepted_ids:
        accepted_details = [
            o for o in pending if int(o.get("id", -1)) in accepted_ids
        ]
        if accepted_details:
            lines = ["Marked as *accepted*:"]
            for o in accepted_details:
                lines.append(f"- {o['location']} ${float(o['amount']):.2f}")
            parts.append("\n".join(lines))

    if rejected_ids:
        rejected_details = [
            o for o in pending if int(o.get("id", -1)) in rejected_ids
        ]
        if rejected_details:
            lines = ["Marked as *rejected*:"]
            for o in rejected_details:
                lines.append(f"- {o['location']} ${float(o['amount']):.2f}")
            parts.append("\n".join(lines))

    if not parts:
        # analyzer decided ignore / nothing to do
        return

    await message.reply_text("\n\n".join(parts), parse_mode="Markdown")


async def generic_range_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    label: str,
    start_utc: datetime,
    end_utc: Optional[datetime] = None,
    summary_mode: Optional[str] = None,
):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    if end_utc is None:
        end_utc = datetime.now(tz=UTC)

    if summary_mode is None:
        summary_mode = get_chat_config(chat_id)["summary_mode"]

    orders = supabase_fetch_orders(
        chat_id=chat_id,
        status="accepted",
        start_utc=start_utc,
        end_utc=end_utc,
        limit=None,
        order_desc=False,
    )

    text = summarize_orders(orders, label, summary_mode=summary_mode)
    await message.reply_text(text)


# ---- Range commands ----

async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = start_of_today_utc()
    await generic_range_command(
        update,
        context,
        "TODAY (accepted orders)",
        start,
        summary_mode=mode,
    )


async def totals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = start_of_week_utc()
    await generic_range_command(
        update,
        context,
        "THIS WEEK (accepted orders since Monday)",
        start,
        summary_mode=mode,
    )


async def last7days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(7)
    await generic_range_command(
        update,
        context,
        "LAST 7 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def last14days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(14)
    await generic_range_command(
        update,
        context,
        "LAST 14 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def last30days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(30)
    await generic_range_command(
        update,
        context,
        "LAST 30 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def last60days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(60)
    await generic_range_command(
        update,
        context,
        "LAST 60 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def last90days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(90)
    await generic_range_command(
        update,
        context,
        "LAST 90 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def last180days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(180)
    await generic_range_command(
        update,
        context,
        "LAST 180 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def last365days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = days_ago_utc(365)
    await generic_range_command(
        update,
        context,
        "LAST 365 DAYS (accepted orders)",
        start,
        summary_mode=mode,
    )


async def alltime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    mode = get_mode_from_args(chat_id, context.args)
    start = datetime(2000, 1, 1, tzinfo=UTC)
    await generic_range_command(
        update,
        context,
        "ALL-TIME (accepted orders)",
        start,
        summary_mode=mode,
    )


async def statsrange_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /statsrange YYYY-MM-DD YYYY-MM-DD [full|medium|compact]
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    args = context.args

    if len(args) < 2:
        await message.reply_text(
            "Usage: /statsrange YYYY-MM-DD YYYY-MM-DD [full|medium|compact]\n"
            "Example: /statsrange 2025-01-01 2025-01-15 full"
        )
        return

    start_str, end_str = args[0], args[1]
    mode = get_chat_config(chat_id)["summary_mode"]
    if len(args) >= 3:
        maybe_mode = args[2].lower()
        if maybe_mode in ("full", "medium", "compact"):
            mode = maybe_mode

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        await message.reply_text(
            "Dates must be in YYYY-MM-DD format, e.g.\n"
            "/statsrange 2025-01-01 2025-01-15"
        )
        return

    if end_date < start_date:
        await message.reply_text("End date must be on or after start date.")
        return

    # end is exclusive -> add 1 day
    end_utc = end_date + timedelta(days=1)
    label = f"STATS {start_str} to {end_str} (accepted orders)"
    await generic_range_command(
        update,
        context,
        label,
        start_date,
        end_utc,
        summary_mode=mode,
    )


async def pending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    pending = supabase_fetch_pending(chat_id, limit=10)
    if not pending:
        await message.reply_text("No pending orders for this chat.")
        return

    lines = ["Recent pending orders:"]
    for o in pending:
        ts_str = o.get("order_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            local_ts = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M")
        except Exception:
            local_ts = ts_str
        lines.append(
            f"- #{o['id']} {o['location']} ${float(o['amount']):.2f} ({local_ts})"
        )

    await message.reply_text("\n".join(lines))


# ============= MAIN =============

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is not set.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is not set.")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))

    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("totals", totals_command))
    application.add_handler(CommandHandler("last7days", last7days_command))
    application.add_handler(CommandHandler("last14days", last14days_command))
    application.add_handler(CommandHandler("last30days", last30days_command))
    application.add_handler(CommandHandler("last60days", last60days_command))
    application.add_handler(CommandHandler("last90days", last90days_command))
    application.add_handler(CommandHandler("last180days", last180days_command))
    application.add_handler(CommandHandler("last365days", last365days_command))
    application.add_handler(CommandHandler("alltime", alltime_command))
    application.add_handler(CommandHandler("statsrange", statsrange_command))
    application.add_handler(CommandHandler("pending", pending_command))

    # messages (non-commands)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
