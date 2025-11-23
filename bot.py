import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional

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

# timezone for weekly reset logic (your business timezone)
TIMEZONE = ZoneInfo("America/New_York")

# max messages we realistically store in memory per chat (safety cap)
MAX_ORDERS_MEMORY = 1000

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============= STATE =============

# In-memory state (no external DB)
# CHAT_STATE[chat_id] = {
#   "orders": [ { "timestamp": datetime, "location": str, "amount": float, "weekday": str } ],
#   "manual_active": bool,
#   "manual_start": datetime | None,
# }
CHAT_STATE: Dict[int, Dict[str, Any]] = {}


def get_chat_state(chat_id: int) -> Dict[str, Any]:
    """Return (and initialize) state for this chat."""
    if chat_id not in CHAT_STATE:
        CHAT_STATE[chat_id] = {
            "orders": [],
            "manual_active": False,
            "manual_start": None,
        }
    return CHAT_STATE[chat_id]


# ============= OPENAI VIA HTTP =============

def call_openai_chat(prompt: str) -> Optional[str]:
    """
    Call OpenAI's chat completions endpoint using plain HTTP.
    Returns the message content string, or None on error.
    """
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
                "content": "You extract structured delivery orders (location + amount) and respond ONLY with JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
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
    Use OpenAI to extract zero or more orders from a single message.

    Returns a list of objects like:
    [
      {"location": "Manassas", "amount": 35.0},
      {"location": "Alexandria", "amount": 30.0}
    ]
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
- Ignore time windows (like "11-12", "3-4"), emojis, extra commentary, and anything not clearly an order.
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

    # Sometimes models wrap JSON in code fences – strip if needed
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


# ============= HELPERS =============

def last_monday_start(now_utc: datetime) -> datetime:
    """
    Get the datetime for Monday 00:00 of the current week in local TIMEZONE,
    then convert back to UTC.
    """
    local_now = now_utc.astimezone(TIMEZONE)
    days_since_monday = local_now.weekday()  # Monday=0
    monday_local = (local_now - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return monday_local.astimezone(ZoneInfo("UTC"))


def prune_old_orders(chat_state: Dict[str, Any]):
    """
    Keep the orders list from growing forever.
    Strategy: if > MAX_ORDERS_MEMORY, drop the oldest ones.
    """
    orders = chat_state["orders"]
    if len(orders) > MAX_ORDERS_MEMORY:
        chat_state["orders"] = orders[-MAX_ORDERS_MEMORY:]


def filter_orders_in_range(
    chat_state: Dict[str, Any],
    start: datetime,
    end: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Return all orders between [start, end], sorted oldest -> newest.
    """
    if end is None:
        end = datetime.now(tz=ZoneInfo("UTC"))

    result = []
    for order in chat_state["orders"]:
        ts = order["timestamp"]
        if start <= ts <= end:
            result.append(order)

    # sort by timestamp oldest -> newest
    result.sort(key=lambda o: o["timestamp"])
    return result


def group_orders_by_weekday(orders: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group orders by weekday name (Monday, Tuesday, ...).
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for order in orders:
        day = order.get("weekday")
        if not day:
            # fallback if missing
            day = order["timestamp"].astimezone(TIMEZONE).strftime("%A")
        grouped.setdefault(day, []).append(order)
    return grouped


def format_orders_grouped_by_weekday(orders: List[Dict[str, Any]], label: str) -> str:
    """
    Format a detailed list of orders grouped by weekday, oldest -> newest,
    with a weekly total at the bottom.
    """
    if not orders:
        return f"{label} – No orders found in this period."

    grouped = group_orders_by_weekday(orders)
    total = sum(o["amount"] for o in orders)

    lines: List[str] = [label, ""]
    for day in WEEKDAY_ORDER:
        day_orders = grouped.get(day, [])
        if not day_orders:
            continue
        lines.append(day.upper())
        for o in day_orders:
            lines.append(f"• {o['location']} — ${o['amount']:.2f}")
        lines.append("")  # blank line between days

    lines.append(f"TOTAL FOR PERIOD: ${total:.2f}")
    return "\n".join(lines).rstrip()


def format_revenue_by_day(orders: List[Dict[str, Any]], label: str) -> str:
    """
    Format daily totals (per weekday) and weekly total.
    """
    if not orders:
        return f"{label} – No orders found in this period."

    grouped = group_orders_by_weekday(orders)
    day_totals: Dict[str, float] = {}
    for day, day_orders in grouped.items():
        day_totals[day] = sum(o["amount"] for o in day_orders)

    grand_total = sum(day_totals.values())

    lines: List[str] = [label]
    for day in WEEKDAY_ORDER:
        if day in day_totals:
            lines.append(f"{day}: ${day_totals[day]:.2f}")
    lines.append("")
    lines.append(f"TOTAL FOR PERIOD: ${grand_total:.2f}")
    return "\n".join(lines)


def compute_stats(orders: List[Dict[str, Any]]) -> str:
    """
    Compute simple stats for a set of orders.
    """
    if not orders:
        return "No orders found in this period."

    amounts = [o["amount"] for o in orders]
    total = sum(amounts)
    count = len(amounts)
    avg = total / count if count > 0 else 0.0
    max_amt = max(amounts)
    min_amt = min(amounts)

    return (
        f"STATS FOR PERIOD\n"
        f"Total Orders: {count}\n"
        f"Total Amount: ${total:.2f}\n"
        f"Average Order: ${avg:.2f}\n"
        f"Largest Order: ${max_amt:.2f}\n"
        f"Smallest Order: ${min_amt:.2f}"
    )


def get_week_period(chat_state: Dict[str, Any]):
    """
    Return (start, end, label) for the current period:
    - Manual week if active
    - Otherwise calendar week since Monday 00:00 (business timezone)
    """
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    if chat_state["manual_active"] and chat_state["manual_start"] is not None:
        start = chat_state["manual_start"]
        label = "MANUAL WEEK TOTALS"
    else:
        start = last_monday_start(now_utc)
        label = "CALENDAR WEEK TOTALS (since Monday 00:00)"
    return start, now_utc, label


def get_today_bounds():
    now_local = datetime.now(tz=TIMEZONE)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(ZoneInfo("UTC")), end_local.astimezone(ZoneInfo("UTC")), now_local.strftime("%A")


def get_yesterday_bounds():
    now_local = datetime.now(tz=TIMEZONE)
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_local = today_start_local - timedelta(days=1)
    end_local = today_start_local
    day_name = start_local.strftime("%A")
    return start_local.astimezone(ZoneInfo("UTC")), end_local.astimezone(ZoneInfo("UTC")), day_name


# ============= HANDLERS =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I'm your delivery totals bot.\n\n"
        "I watch this chat for dispatch messages like:\n"
        "• I have Manassas for $35 for 3-4 Alex\n"
        "• I have largo $27 1-2 can go early and Alexandria $30 2-3\n"
        "• Adding Woodbridge to route for $35\n\n"
        "Then I extract the location + amount and keep a running tally.\n\n"
        "Core commands:\n"
        "/totals – detailed list for current week/manual period grouped by weekday\n"
        "/week – same as /totals\n"
        "/today – today’s orders only\n"
        "/yesterday – yesterday’s orders only\n"
        "/revenue – daily totals for the current week/manual period\n"
        "/stats – summary stats for the current week/manual period\n"
        "/orders – full detailed log for the current week/manual period\n"
        "/startweek – start a manual payout week (overrides auto calendar week)\n"
        "/endweek – end the current manual week and show totals\n"
        "/clear – clear all stored data for this chat\n"
        "/help – show this message again\n\n"
        "Note: totals are in-memory only. If the bot is restarted or redeployed, counters reset. "
        "Each Telegram chat (driver group) is tracked separately."
    )
    if update.message:
        await update.message.reply_text(msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle any non-command text message.
    We call OpenAI to extract orders and store them in memory for this chat.
    """
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    text = message.text
    chat_state = get_chat_state(chat_id)

    msg_time = message.date
    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=ZoneInfo("UTC"))

    orders = await extract_orders_from_text(text)
    if not orders:
        return

    for o in orders:
        # Determine weekday in your business timezone
        weekday = msg_time.astimezone(TIMEZONE).strftime("%A")

        chat_state["orders"].append(
            {
                "timestamp": msg_time,
                "location": o["location"],
                "amount": o["amount"],
                "weekday": weekday,
            }
        )

    prune_old_orders(chat_state)
    logger.info(
        f"Chat {chat_id}: stored {len(orders)} orders from message {message.message_id}"
    )


async def totals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /totals – show detailed orders for:
    - If manual week active: from manual_start .. now
    - Else: from last Monday 00:00 (business timezone) .. now
    Grouped by weekday, oldest -> newest.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, label = get_week_period(chat_state)
    orders = filter_orders_in_range(chat_state, start, end)
    text = format_orders_grouped_by_weekday(orders, label)
    await message.reply_text(text)


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # alias of /totals
    await totals_command(update, context)


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /today – show today’s orders grouped by weekday (single day) oldest -> newest.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, day_name = get_today_bounds()
    orders = filter_orders_in_range(chat_state, start, end)
    label = f"TODAY ({day_name})"
    text = format_orders_grouped_by_weekday(orders, label)
    await message.reply_text(text)


async def yesterday_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /yesterday – show yesterday’s orders grouped by weekday (single day) oldest -> newest.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, day_name = get_yesterday_bounds()
    orders = filter_orders_in_range(chat_state, start, end)
    label = f"YESTERDAY ({day_name})"
    text = format_orders_grouped_by_weekday(orders, label)
    await message.reply_text(text)


async def revenue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /revenue – daily totals for the current week/manual period.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, label = get_week_period(chat_state)
    orders = filter_orders_in_range(chat_state, start, end)
    text = format_revenue_by_day(orders, f"REVENUE BY DAY – {label}")
    await message.reply_text(text)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stats – summary stats for the current week/manual period.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, label = get_week_period(chat_state)
    orders = filter_orders_in_range(chat_state, start, end)
    stats_text = compute_stats(orders)
    text = f"{label}\n\n{stats_text}"
    await message.reply_text(text)


async def orders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /orders – full detailed log for current week/manual period (same as /totals, but explicit).
    """
    await totals_command(update, context)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /clear – clear all stored data for this chat.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    CHAT_STATE[chat_id] = {
        "orders": [],
        "manual_active": False,
        "manual_start": None,
    }

    await message.reply_text(
        "All stored orders and week settings for this chat have been cleared."
    )


async def startweek_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /startweek – clear prior manual state and start a new manual payout week
    from 'now' (UTC).
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    chat_state["manual_active"] = True
    chat_state["manual_start"] = now_utc

    await message.reply_text(
        "Manual week started from *now*.\n"
        "I'll track orders from this point until you run /endweek.",
        parse_mode="Markdown",
    )


async def endweek_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /endweek – finalize the current manual week, show totals,
    and return to automatic calendar-week mode.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    if not chat_state["manual_active"] or chat_state["manual_start"] is None:
        await message.reply_text(
            "No manual week is currently active. Use /startweek to begin one."
        )
        return

    start = chat_state["manual_start"]
    now_utc = datetime.now(tz=ZoneInfo("UTC"))

    orders = filter_orders_in_range(chat_state, start, now_utc)
    text = format_orders_grouped_by_weekday(orders, "MANUAL WEEK FINAL TOTALS")

    # reset manual mode
    chat_state["manual_active"] = False
    chat_state["manual_start"] = None

    await message.reply_text(text)


# ============= MAIN =============

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is not set.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("totals", totals_command))
    application.add_handler(CommandHandler("week", week_command))
    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("yesterday", yesterday_command))
    application.add_handler(CommandHandler("revenue", revenue_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("orders", orders_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("startweek", startweek_command))
    application.add_handler(CommandHandler("endweek", endweek_command))

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
