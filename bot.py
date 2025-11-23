import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Tuple
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

# timezone for reporting (your business timezone)
TIMEZONE = ZoneInfo("America/New_York")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

# logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Per-chat config (in memory only; resets on restart)
CHAT_CONFIG: Dict[int, Dict[str, Any]] = {}
DEFAULT_SUMMARY_MODE = "full"  # full | medium | compact


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

    rows = []
    # start of calendar week (Monday 00:00) in UTC
    week_start_local = msg_time_utc.astimezone(TIMEZONE)
    days_since_monday = week_start_local.weekday()
    monday_local = (week_start_local - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    week_start_utc = monday_local.astimezone(ZoneInfo("UTC"))

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
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Inserted {len(data)} orders to Supabase for chat {chat_id}")
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
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        logger.error(f"Supabase fetch error: {e}")
        return []


def supabase_fetch_pending(chat_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    return supabase_fetch_orders(
        chat_id=chat_id,
        status="pending",
        start_utc=None,
        end_utc=None,
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
                timeout=15,
            )
            resp.raise_for_status()
            logger.info(f"Updated order {oid} to status={new_status}")
        except Exception as e:
            logger.error(f"Supabase status update error for id={oid}: {e}")


# ============= OPENAI ORDER EXTRACTION =============

def call_openai_chat(prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract structured delivery orders (location + amount) "
                    "and respond ONLY with JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.error(f"Error calling OpenAI HTTP API: {e}")
        return None


async def extract_orders_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI to extract zero or more orders from a dispatcher message.
    """
    # quick cheap heuristic: if no digit, probably no amount -> skip OpenAI
    if not any(ch.isdigit() for ch in text):
        return []

    prompt = f"""
You are a precise data extraction assistant for a weed delivery dispatch team.

A dispatcher posts messy natural-language messages describing one or more delivery orders.
Each order has:
- a LOCATION (like "Manassas", "Alexandria", "Upper Marlboro", "Deale MD", "DC", etc.)
- an AMOUNT in dollars (e.g. $35 means 35.0)

Your job:
- Read the message carefully.
- Find ALL delivery orders mentioned.
- For each, return an object with "location" and "amount".
- Ignore time windows, emojis, extra commentary, and anything not clearly an order.
- If a location is mentioned with no clear dollar amount, skip that location.
- If the message does NOT contain any valid orders, return an empty list.

Output STRICTLY valid JSON in this EXACT format (no extra text, no comments):

[
  {{"location": "ExampleLocation", "amount": 35.0}}
]

Now extract orders from this message:

\"\"\"{text}\"\"\"
"""

    content = call_openai_chat(prompt)
    if content is None:
        return []

    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            content = content.split("\n", 1)[1].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from OpenAI response: {e} | content={content[:200]}")
        return []

    orders: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            try:
                loc = str(item.get("location", "")).strip()
                amt = float(item.get("amount", 0))
                if loc and amt > 0:
                    orders.append({"location": loc, "amount": amt})
            except Exception:
                continue
    return orders


# ============= INTENT + CONFIRMATION / REJECTION =============

ACCEPT_EMOJIS = {"👍", "❤️", "♥️", "🙏", "🙌", "🔥", "✔️", "✅", "😎", "👌", "💯", "😁", "🙂", "🤝", "🚀"}
REJECT_EMOJIS = {"👎", "❌", "🚫", "⛔", "😕", "😒", "🤦"}


ACCEPT_PHRASES = [
    "send it",
    "send them",
    "send",
    "coming",
    "on the way",
    "otw",
    "omw",
    "headed there",
    "be there soon",
    "i'll take",
    "ill take",
    "i will take",
    "i can take",
    "i can do",
    "i'll do",
    "ill do",
    "run it",
    "lock me in",
    "got it",
    "got them",
    "i got it",
    "i got them",
    "i'll grab",
    "ill grab",
    "i'll scoop",
    "ill scoop",
    "i'll go",
    "ill go",
    "i'll pick it up",
    "ill pick it up",
    "i'm down",
    "im down",
    "sure",
    "ok",
    "okay",
    "bet",
    "say less",
    "i'll take both",
    "ill take both",
    "i'll take them",
    "ill take them",
    "i'll take it",
    "ill take it",
    "i'll do it",
    "ill do it",
]

REJECT_PHRASES = [
    "can't",
    "cant",
    "cannot",
    "won't",
    "wont",
    "no ",
    " no",
    "nah",
    "not taking",
    "don't want",
    "dont want",
    "skip it",
    "skip that",
    "pass",
    "reject",
    "drop it",
    "can't do",
    "cant do",
    "can't take",
    "cant take",
]


def classify_intent(text: str) -> Optional[str]:
    """
    Return 'accept', 'reject', or None.
    """
    t = text.strip()
    if not t:
        return None

    # emoji-only messages
    if len(t) <= 4 and all(ch in ACCEPT_EMOJIS or ch in REJECT_EMOJIS for ch in t):
        if any(ch in REJECT_EMOJIS for ch in t):
            return "reject"
        if any(ch in ACCEPT_EMOJIS for ch in t):
            return "accept"

    lower = t.lower()

    # check rejection first (so "can't take it" doesn't get treated as accept)
    for phrase in REJECT_PHRASES:
        if phrase in lower:
            return "reject"

    for phrase in ACCEPT_PHRASES:
        if phrase in lower:
            return "accept"

    return None


def match_orders_for_message(
    message_text: str,
    pending_orders: List[Dict[str, Any]],
) -> List[int]:
    """
    Decide which pending orders a confirmation/rejection applies to.
    Strategy:
      - if only one pending -> that one
      - else, if message mentions a location name -> those that match
      - else, if message says "both"/"all" -> all
      - else -> most recent only
    """
    if not pending_orders:
        return []

    if len(pending_orders) == 1:
        return [pending_orders[0]["id"]]

    txt = message_text.lower()

    # try location-based matching
    matched_ids: List[int] = []
    for o in pending_orders:
        loc = str(o.get("location", "")).lower()
        if loc and loc in txt:
            matched_ids.append(o["id"])

    if matched_ids:
        return matched_ids

    # phrases for multiple
    if "both" in txt or "all" in txt or "2 orders" in txt or "two orders" in txt:
        return [o["id"] for o in pending_orders]

    # default: most recent pending (first in list because we sorted desc)
    return [pending_orders[0]["id"]]


# ============= COMMAND HELPERS =============

def start_of_today_utc() -> datetime:
    now_local = datetime.now(tz=TIMEZONE)
    today_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return today_local.astimezone(ZoneInfo("UTC"))


def start_of_week_utc() -> datetime:
    now_local = datetime.now(tz=TIMEZONE)
    days_since_monday = now_local.weekday()
    monday_local = (now_local - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return monday_local.astimezone(ZoneInfo("UTC"))


def days_ago_utc(days: int) -> datetime:
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    return now_utc - timedelta(days=days)


def summarize_orders(
    orders: List[Dict[str, Any]],
    label: str,
    summary_mode: str = "full",
) -> str:
    if not orders:
        return f"{label}\n\nNo accepted orders in this period."

    # convert timestamps to local & group by date
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    total_amount = 0.0

    for o in orders:
        ts = datetime.fromisoformat(o["order_timestamp"].replace("Z", "+00:00"))
        local_date = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d (%a)")
        grouped.setdefault(local_date, []).append(o)
        total_amount += float(o["amount"])

    lines: List[str] = [label, ""]
    num_days = len(grouped)

    for date_key in sorted(grouped.keys()):
        day_orders = grouped[date_key]
        day_total = sum(float(o["amount"]) for o in day_orders)

        if summary_mode in ("full", "medium"):
            lines.append(f"{date_key}: ${day_total:.2f}")

        if summary_mode == "full":
            # show individual orders in time order
            for o in sorted(
                day_orders,
                key=lambda x: x["order_timestamp"],
            ):
                lines.append(f"  - {o['location']}: ${float(o['amount']):.2f}")
            lines.append("")

    if summary_mode == "compact":
        lines = [label, ""]

    lines.append(f"TOTAL: ${total_amount:.2f}")
    avg_per_day = total_amount / num_days if num_days > 0 else 0.0
    lines.append(f"Avg per active day: ${avg_per_day:.2f}")

    return "\n".join(lines)


# ============= HANDLERS =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I'm your delivery totals bot.\n\n"
        "How it works:\n"
        "• Dispatcher posts messages with orders (locations + $ amounts).\n"
        "• I extract those orders and save them as *pending*.\n"
        "• When the driver confirms (text like 'send it', 'coming', 'I'll take it' or emoji like 👍❤️✔️), "
        "I mark those orders as *accepted*.\n"
        "• Rejections like 'can't', 'nah', or 👎 mark them as *rejected*.\n"
        "• Only *accepted* orders count toward totals & stats.\n\n"
        "Commands:\n"
        "/today – accepted orders today\n"
        "/totals – this calendar week (since Monday)\n"
        "/last7days – last 7 days\n"
        "/last30days – last 30 days\n"
        "/alltime – all accepted orders ever\n"
        "/pending – show recent pending orders\n"
        "/mode full|medium|compact – change summary detail\n"
        "/help – show this again"
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
    if not args:
        cfg = get_chat_config(chat_id)
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

    cfg = get_chat_config(chat_id)
    cfg["summary_mode"] = choice
    await message.reply_text(f"Summary mode set to: {choice}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle any non-command text message.
    1) Try to interpret as confirmation/rejection for pending orders.
    2) If not, try to extract new orders from dispatcher text.
    """
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    text = message.text

    # 1) intent classification
    intent = classify_intent(text)
    if intent in ("accept", "reject"):
        pending = supabase_fetch_pending(chat_id, limit=5)
        if not pending:
            # nothing to apply this to
            return

        order_ids = match_orders_for_message(text, pending)
        if not order_ids:
            return

        new_status = "accepted" if intent == "accept" else "rejected"
        supabase_update_status(order_ids, new_status)

        # small confirmation message
        affected = [o for o in pending if o["id"] in order_ids]
        parts = [f"{a['location']} ${float(a['amount']):.2f}" for a in affected]
        joined = ", ".join(parts)
        await message.reply_text(
            f"Marked as *{new_status}*: {joined}", parse_mode="Markdown"
        )
        return

    # 2) Maybe it's a dispatcher message with new orders
    msg_time = message.date
    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=ZoneInfo("UTC"))
    msg_time_utc = msg_time.astimezone(ZoneInfo("UTC"))

    orders = await extract_orders_from_text(text)
    if not orders:
        return

    # Insert as pending
    supabase_insert_orders(chat_id, orders, msg_time_utc)
    lines = [f"Stored {len(orders)} pending order(s):"]
    for o in orders:
        lines.append(f"- {o['location']}: ${float(o['amount']):.2f}")
    await message.reply_text("\n".join(lines))


async def generic_range_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    label: str,
    start_utc: datetime,
    end_utc: Optional[datetime] = None,
):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    cfg = get_chat_config(chat_id)

    if end_utc is None:
        end_utc = datetime.now(tz=ZoneInfo("UTC"))

    orders = supabase_fetch_orders(
        chat_id=chat_id,
        status="accepted",
        start_utc=start_utc,
        end_utc=end_utc,
        limit=None,
        order_desc=False,
    )

    text = summarize_orders(orders, label, summary_mode=cfg["summary_mode"])
    await message.reply_text(text)


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start = start_of_today_utc()
    await generic_range_command(update, context, "TODAY (accepted orders)", start)


async def totals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start = start_of_week_utc()
    await generic_range_command(
        update, context, "THIS WEEK (since Monday, accepted orders)", start
    )


async def last7_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start = days_ago_utc(7)
    await generic_range_command(
        update, context, "LAST 7 DAYS (accepted orders)", start
    )


async def last30_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start = days_ago_utc(30)
    await generic_range_command(
        update, context, "LAST 30 DAYS (accepted orders)", start
    )


async def alltime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # effectively from far past
    start = datetime(2000, 1, 1, tzinfo=ZoneInfo("UTC"))
    await generic_range_command(update, context, "ALL-TIME (accepted orders)", start)


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
        ts = datetime.fromisoformat(o["order_timestamp"].replace("Z", "+00:00"))
        local_ts = ts.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M")
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
    application.add_handler(CommandHandler("last7days", last7_command))
    application.add_handler(CommandHandler("last30days", last30_command))
    application.add_handler(CommandHandler("alltime", alltime_command))
    application.add_handler(CommandHandler("pending", pending_command))

    # messages (non-commands)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
