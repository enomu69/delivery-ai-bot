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
    return supabase_fetch_orders(
        chat_id=chat_id,
        status="pending",
        limit=limit,
        order_desc=True,
    )


def supabase_update_status(order_ids: List[int], new_status: str):
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
    Uses OpenAI to classify the message.
    - Identify acceptance/rejection of pending orders
    - Detect new orders
    - Or ignore irrelevant messages
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set.")
        return None

    # Build pending orders context (indexed for NLP)
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
Sometimes the dispatcher also sends NEW orders.

You must:
- Identify confirmations (send it, coming, yes, I'll take it, thumbs up emoji, etc.)
- Identify rejections (no, can't, nah, not going, thumbs down emoji, etc.)
- Identify newly described orders (locations + $)
- Or return "ignore".

Return ONLY JSON.

"""

    user_prompt = f"""
CURRENT PENDING ORDERS:
{pending_text}

NEW MESSAGE:
\"\"\"{message_text}\"\"\"

Return json:
{{
  "action": "confirm" | "reject" | "orders" | "ignore",
  "accepted_indices": [...],
  "rejected_indices": [...],
  "new_orders": [{{"location": "...", "amount": 00.0}}]
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
        content = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI analyzer error: {e}")
        return None

    # Strip possible code fences
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            content = content.split("\n", 1)[1].strip()

    try:
        data = json.loads(content)
    except Exception as e:
        logger.error(f"JSON parse error: {e} | content={content[:200]}")
        return None

    # clean numeric lists
    def to_int_list(x):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except:
                pass
        return out

    accepted = to_int_list(data.get("accepted_indices", []))
    rejected = to_int_list(data.get("rejected_indices", []))

    cleaned_new = []
    for item in data.get("new_orders", []):
        try:
            loc = str(item.get("location", "")).strip()
            amt = float(item.get("amount", 0))
            if loc and amt > 0:
                cleaned_new.append({"location": loc, "amount": amt})
        except:
            continue

    return {
        "action": data.get("action", "ignore"),
        "accepted_indices": accepted,
        "rejected_indices": rejected,
        "new_orders": cleaned_new,
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
    return datetime.now(tz=UTC) - timedelta(days=days)


def summarize_orders(
    orders: List[Dict[str, Any]],
    label: str,
    summary_mode: str = "full",
) -> str:
    if not orders:
        return f"{label}\n\nNo accepted orders in this period."

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    total_amount = 0.0

    for o in orders:
        ts_str = o.get("order_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            local_date = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d (%a)")
        except:
            local_date = "Unknown date"

        grouped.setdefault(local_date, []).append(o)
        total_amount += float(o["amount"])

    lines = [label, ""]
    num_days = len(grouped)

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

    lines.append(f"TOTAL: ${total_amount:.2f}")
    avg_per_day = total_amount / num_days if num_days else 0.0
    lines.append(f"Avg per active day: ${avg_per_day:.2f}")

    per_location = {}
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


# ============= COMMAND HANDLERS =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I'm your delivery totals bot.\n\n"
        "How I work:\n"
        "• Dispatcher posts offers → I save them as *pending*.\n"
        "• Driver confirms (\"send it\", 👍, \"coming\", etc.) → I mark as *accepted*.\n"
        "• Driver rejects (\"nah\", 👎, \"can't\", etc.) → I mark as *rejected*.\n\n"
        "Commands:\n"
        "/today\n/totals\n/last7days\n/last30days\n/alltime\n"
        "/pending\n/statsrange\n/mode\n"
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
        )
        return

    choice = args[0].lower()
    if choice not in ("full", "medium", "compact"):
        await message.reply_text("Invalid mode.")
        return

    cfg["summary_mode"] = choice
    await message.reply_text(f"Summary mode set to: {choice}")


# ============= MAIN MESSAGE HANDLER =============

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main NLP message handler.
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

    pending = supabase_fetch_pending(chat_id, limit=10)
    analysis = call_openai_analyzer(text, pending)
    if analysis is None:
        return

    action = analysis["action"]
    accepted_indices = analysis["accepted_indices"]
    rejected_indices = analysis["rejected_indices"]
    new_orders = analysis["new_orders"]

    # index → supabase ID
    def indices_to_ids(indices: List[int]) -> List[int]:
        ids = []
        for idx in indices:
            if 1 <= idx <= len(pending):
                oid = pending[idx - 1].get("id")
                if oid:
                    ids.append(int(oid))
        return ids

    accepted_ids = indices_to_ids(accepted_indices)
    rejected_ids = indices_to_ids(rejected_indices)

    # Store new pending orders
    if new_orders:
        supabase_insert_orders(chat_id, new_orders, msg_time_utc)

    # Update statuses
    if accepted_ids:
        supabase_update_status(accepted_ids, "accepted")
    if rejected_ids:
        supabase_update_status(rejected_ids, "rejected")

    # ---- SILENT MODE: bot never replies for confirmations or new orders ----
    return


# ============= RANGE COMMANDS =============

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
        order_desc=False,
    )

    text = summarize_orders(orders, label, summary_mode)
    await message.reply_text(text)


# ---- Commands ----

async def today_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "TODAY", start_of_today_utc(), summary_mode=mode)


async def totals_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "THIS WEEK", start_of_week_utc(), summary_mode=mode)


async def last7days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 7 DAYS", days_ago_utc(7), summary_mode=mode)


async def last30days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 30 DAYS", days_ago_utc(30), summary_mode=mode)


async def last14days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 14 DAYS", days_ago_utc(14), summary_mode=mode)


async def last60days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 60 DAYS", days_ago_utc(60), summary_mode=mode)


async def last90days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 90 DAYS", days_ago_utc(90), summary_mode=mode)


async def last180days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 180 DAYS", days_ago_utc(180), summary_mode=mode)


async def last365days_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    await generic_range_command(update, context, "LAST 365 DAYS", days_ago_utc(365), summary_mode=mode)


async def alltime_command(update, context):
    chat_id = update.message.chat_id
    mode = get_chat_config(chat_id)["summary_mode"]
    start = datetime(2000, 1, 1, tzinfo=UTC)
    await generic_range_command(update, context, "ALL TIME", start, summary_mode=mode)


async def statsrange_command(update, context):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    args = context.args

    if len(args) < 2:
        await message.reply_text("Usage: /statsrange YYYY-MM-DD YYYY-MM-DD")
        return

    start_str, end_str = args[0], args[1]
    mode = get_chat_config(chat_id)["summary_mode"]

    if len(args) >= 3:
        if args[2].lower() in ("full", "medium", "compact"):
            mode = args[2].lower()

    try:
        start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=UTC)
    except:
        await message.reply_text("Invalid date format.")
        return

    if end < start:
        await message.reply_text("End date must be after start date.")
        return

    end_exclusive = end + timedelta(days=1)
    await generic_range_command(
        update, context,
        f"STATS {start_str} to {end_str}",
        start,
        end_exclusive,
        summary_mode=mode,
    )


async def pending_command(update, context):
    chat_id = update.message.chat_id

    pending = supabase_fetch_pending(chat_id, limit=10)
    if not pending:
        await update.message.reply_text("No pending orders.")
        return

    lines = ["Recent pending:"]
    for o in pending:
        ts_str = o.get("order_timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            local_ts = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M")
        except:
            local_ts = ts_str

        lines.append(
            f"- #{o['id']} {o['location']} ${float(o['amount']):.2f} ({local_ts})"
        )

    await update.message.reply_text("\n".join(lines))


# ============= MAIN =============

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN not set")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase config missing")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("mode", mode_command))

    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("totals", totals_command))
    app.add_handler(CommandHandler("last7days", last7days_command))
    app.add_handler(CommandHandler("last14days", last14days_command))
    app.add_handler(CommandHandler("last30days", last30days_command))
    app.add_handler(CommandHandler("last60days", last60days_command))
    app.add_handler(CommandHandler("last90days", last90days_command))
    app.add_handler(CommandHandler("last180days", last180days_command))
    app.add_handler(CommandHandler("last365days", last365days_command))
    app.add_handler(CommandHandler("alltime", alltime_command))
    app.add_handler(CommandHandler("statsrange", statsrange_command))
    app.add_handler(CommandHandler("pending", pending_command))

    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
