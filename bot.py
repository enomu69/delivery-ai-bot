import os
import json
import logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Tuple

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

SUPABASE_ORDERS_URL = (
    f"{SUPABASE_URL.rstrip('/')}/rest/v1/orders" if SUPABASE_URL else None
)

# Column name in Supabase for the order timestamp
SUPABASE_ORDER_TS_COLUMN = "order_timestamp"

TIMEZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MAX_ROWS_PER_QUERY = 10000

# logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============= STATE =============

# In-memory state only for manual week flags per chat.
# Orders themselves live in Supabase so they survive restarts.
#
# CHAT_STATE[chat_id] = {
#   "manual_active": bool,
#   "manual_start": datetime | None,
# }
CHAT_STATE: Dict[int, Dict[str, Any]] = {}


def get_chat_state(chat_id: int) -> Dict[str, Any]:
    if chat_id not in CHAT_STATE:
        CHAT_STATE[chat_id] = {
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


# ============= SUPABASE HELPERS =============

def supabase_headers() -> Dict[str, str]:
    if not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_KEY is not set.")
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=minimal",
    }


def save_order_to_supabase(
    chat_id: int,
    msg_time: datetime,
    location: str,
    amount: float,
    week_start_utc: datetime,
):
    """
    Insert a single order row into Supabase.
    """
    if not SUPABASE_ORDERS_URL:
        logger.error("SUPABASE_URL is not set.")
        return

    try:
        payload = {
            "chat_id": str(chat_id),
            "location": location,
            "amount": amount,
            "week_start": week_start_utc.astimezone(UTC).isoformat(),
            SUPABASE_ORDER_TS_COLUMN: msg_time.astimezone(UTC).isoformat(),
        }

        resp = requests.post(
            SUPABASE_ORDERS_URL,
            headers=supabase_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Error inserting order into Supabase: {e}")


def fetch_orders_from_supabase(
    chat_id: int,
    start_utc: datetime,
    end_utc: datetime,
) -> List[Dict[str, Any]]:
    """
    Fetch orders for a given chat between [start_utc, end_utc], sorted oldest -> newest.
    """
    if not SUPABASE_ORDERS_URL:
        logger.error("SUPABASE_URL is not set.")
        return []

    params = [
        ("chat_id", f"eq.{chat_id}"),
        (SUPABASE_ORDER_TS_COLUMN, f"gte.{start_utc.astimezone(UTC).isoformat()}"),
        (SUPABASE_ORDER_TS_COLUMN, f"lte.{end_utc.astimezone(UTC).isoformat()}"),
        ("order", f"{SUPABASE_ORDER_TS_COLUMN}.asc"),
        ("limit", str(MAX_ROWS_PER_QUERY)),
    ]

    try:
        resp = requests.get(
            SUPABASE_ORDERS_URL,
            headers=supabase_headers(),
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        orders: List[Dict[str, Any]] = []
        for row in data:
            try:
                ts_str = row.get(SUPABASE_ORDER_TS_COLUMN)
                ts = (
                    datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts_str
                    else None
                )
                if ts is None:
                    continue
                orders.append(
                    {
                        "timestamp": ts,
                        "location": row.get("location", ""),
                        "amount": float(row.get("amount", 0) or 0),
                    }
                )
            except Exception:
                continue
        return orders
    except Exception as e:
        logger.error(f"Error fetching orders from Supabase: {e}")
        return []


def fetch_all_orders(chat_id: int) -> List[Dict[str, Any]]:
    """
    Fetch ALL orders for this chat.
    """
    if not SUPABASE_ORDERS_URL:
        logger.error("SUPABASE_URL is not set.")
        return []

    params = [
        ("chat_id", f"eq.{chat_id}"),
        ("order", f"{SUPABASE_ORDER_TS_COLUMN}.asc"),
        ("limit", str(MAX_ROWS_PER_QUERY)),
    ]

    try:
        resp = requests.get(
            SUPABASE_ORDERS_URL,
            headers=supabase_headers(),
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        orders: List[Dict[str, Any]] = []
        for row in data:
            try:
                ts_str = row.get(SUPABASE_ORDER_TS_COLUMN)
                ts = (
                    datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts_str
                    else None
                )
                if ts is None:
                    continue
                orders.append(
                    {
                        "timestamp": ts,
                        "location": row.get("location", ""),
                        "amount": float(row.get("amount", 0) or 0),
                    }
                )
            except Exception:
                continue
        return orders
    except Exception as e:
        logger.error(f"Error fetching all orders from Supabase: {e}")
        return []


def clear_orders_for_chat(chat_id: int) -> None:
    if not SUPABASE_ORDERS_URL:
        return
    try:
        params = [("chat_id", f"eq.{chat_id}")]
        resp = requests.delete(
            SUPABASE_ORDERS_URL,
            headers=supabase_headers(),
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Error clearing orders for chat {chat_id}: {e}")


# ============= DATE HELPERS =============

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
    return monday_local.astimezone(UTC)


def get_week_period(chat_state: Dict[str, Any]) -> Tuple[datetime, datetime, str]:
    """
    Return (start_utc, end_utc, label) for the current period:
    Hybrid logic:
    - If manual week active: from manual_start .. now
    - Else: calendar week since Monday 00:00 (business timezone) .. now
    """
    now_utc = datetime.now(tz=UTC)
    if chat_state["manual_active"] and chat_state["manual_start"] is not None:
        start = chat_state["manual_start"]
        label = "MANUAL WEEK TOTALS"
    else:
        start = last_monday_start(now_utc)
        label = "CALENDAR WEEK TOTALS (since Monday 00:00)"
    return start, now_utc, label


def get_today_bounds() -> Tuple[datetime, datetime, str]:
    now_local = datetime.now(tz=TIMEZONE)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC), now_local.strftime("%A")


def get_yesterday_bounds() -> Tuple[datetime, datetime, str]:
    now_local = datetime.now(tz=TIMEZONE)
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_local = today_start_local - timedelta(days=1)
    end_local = today_start_local
    day_name = start_local.strftime("%A")
    return start_local.astimezone(UTC), end_local.astimezone(UTC), day_name


def get_last_n_days_bounds(n: int) -> Tuple[datetime, datetime]:
    """
    Inclusive of today.
    Example: n=7 → today + previous 6 full days.
    """
    now_local = datetime.now(tz=TIMEZONE)
    today_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_local = today_start_local - timedelta(days=n - 1)
    start_utc = start_local.astimezone(UTC)
    end_utc = now_local.astimezone(UTC)
    return start_utc, end_utc


# ============= STATS & FORMATTING HELPERS =============

def group_orders_by_weekday(orders: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for order in orders:
        ts_local = order["timestamp"].astimezone(TIMEZONE)
        day = ts_local.strftime("%A")
        grouped.setdefault(day, []).append(order)
    return grouped


def format_orders_grouped_by_weekday_per_order(
    orders: List[Dict[str, Any]],
    label: str,
) -> str:
    """
    Used for /totals-style weekly view: lists each order grouped by weekday.
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


def compute_basic_stats(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    amounts = [o["amount"] for o in orders]
    total = sum(amounts)
    count = len(amounts)
    avg = total / count if count else 0.0
    max_amt = max(amounts) if amounts else 0.0
    min_amt = min(amounts) if amounts else 0.0
    return {
        "total_orders": count,
        "total_amount": total,
        "avg_order": avg,
        "max_order": max_amt,
        "min_order": min_amt,
    }


def compute_daily_totals(orders: List[Dict[str, Any]]) -> Dict[date, float]:
    """
    Aggregate revenue by calendar date (local TZ).
    """
    by_day: Dict[date, float] = {}
    for o in orders:
        d = o["timestamp"].astimezone(TIMEZONE).date()
        by_day[d] = by_day.get(d, 0.0) + o["amount"]
    return by_day


def compute_location_stats(orders: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Returns (counts_by_location, revenue_by_location)
    """
    counts: Dict[str, int] = {}
    revenue: Dict[str, float] = {}
    for o in orders:
        loc = o["location"] or "Unknown"
        counts[loc] = counts.get(loc, 0) + 1
        revenue[loc] = revenue.get(loc, 0.0) + o["amount"]
    return counts, revenue


def summarize_orders_range(
    orders: List[Dict[str, Any]],
    label: str,
    detail_level: str = "medium",
) -> str:
    """
    Generic summary for any date range.
    detail_level: "compact" | "medium" | "full"
    """
    if not orders:
        return f"{label}\n\nNo orders found in this period."

    stats = compute_basic_stats(orders)
    daily_totals = compute_daily_totals(orders)
    counts_by_loc, revenue_by_loc = compute_location_stats(orders)

    days_count = len(daily_totals)
    avg_per_day = stats["total_amount"] / days_count if days_count else 0.0

    # Busiest / slowest day by revenue
    busiest_day = None
    slowest_day = None
    if daily_totals:
        busiest_day = max(daily_totals.items(), key=lambda x: x[1])
        slowest_day = min(daily_totals.items(), key=lambda x: x[1])

    def fmt_day(d: date, amount: float) -> str:
        weekday = d.strftime("%A")
        return f"{weekday} {d.isoformat()} – ${amount:.2f}"

    # Top locations by count
    sorted_locs = sorted(counts_by_loc.items(), key=lambda x: x[1], reverse=True)
    total_orders = stats["total_orders"]

    lines: List[str] = [label, ""]

    # Shared core info
    lines.append(f"Total Orders: {stats['total_orders']}")
    lines.append(f"Total Revenue: ${stats['total_amount']:.2f}")
    lines.append(f"Average Order: ${stats['avg_order']:.2f}")
    lines.append(f"Average Revenue Per Day: ${avg_per_day:.2f}")
    lines.append(f"Largest Order: ${stats['max_order']:.2f}")
    lines.append(f"Smallest Order: ${stats['min_order']:.2f}")

    if busiest_day:
        lines.append(
            f"Busiest Day: {fmt_day(busiest_day[0], busiest_day[1])}"
        )
    if slowest_day and slowest_day != busiest_day:
        lines.append(
            f"Slowest Day: {fmt_day(slowest_day[0], slowest_day[1])}"
        )

    # Compact: stop here
    if detail_level == "compact":
        return "\n".join(lines)

    # Medium & Full: add top locations
    lines.append("")
    lines.append("Top Locations:")
    top_n = 3 if detail_level == "medium" else len(sorted_locs)
    for loc, cnt in sorted_locs[:top_n]:
        pct = (cnt / total_orders) * 100 if total_orders else 0.0
        lines.append(f"• {loc} – {cnt} orders ({pct:.1f}%)")

    # Medium: add short revenue-by-day (sorted high → low)
    if detail_level == "medium":
        lines.append("")
        lines.append("Revenue by Day (top 7):")
        for d, amt in sorted(daily_totals.items(), key=lambda x: x[1], reverse=True)[:7]:
            lines.append(f"• {fmt_day(d, amt)}")
        return "\n".join(lines)

    # Full: full revenue-by-day + weekday totals
    lines.append("")
    lines.append("Revenue by Day (all):")
    for d in sorted(daily_totals.keys()):
        amt = daily_totals[d]
        lines.append(f"• {fmt_day(d, amt)}")

    # Weekday totals
    weekday_totals: Dict[str, float] = {}
    for o in orders:
        wd = o["timestamp"].astimezone(TIMEZONE).strftime("%A")
        weekday_totals[wd] = weekday_totals.get(wd, 0.0) + o["amount"]

    lines.append("")
    lines.append("Totals by Weekday:")
    for wd in WEEKDAY_ORDER:
        if wd in weekday_totals:
            lines.append(f"{wd}: ${weekday_totals[wd]:.2f}")

    return "\n".join(lines)


def parse_detail_level(args: List[str]) -> str:
    """
    Look for 'compact' or 'full' in args; default 'medium'.
    """
    lowered = [a.lower() for a in args]
    if "compact" in lowered:
        return "compact"
    if "full" in lowered:
        return "full"
    return "medium"


# ============= COMMAND HANDLERS =============

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
        "/today – today’s orders\n"
        "/yesterday – yesterday’s orders\n"
        "/revenue – daily totals for current period\n"
        "/stats – summary stats for current period\n"
        "/orders – alias for /totals\n"
        "/startweek – start a manual payout week (overrides auto calendar week)\n"
        "/endweek – end the current manual week and show totals\n"
        "/clear – clear all stored data for this chat\n\n"
        "Advanced stats (include optional 'compact' or 'full'):\n"
        "/last7days, /last14days, /last30days, /last60days, /last90days,\n"
        "/last180days, /last365days, /alltime\n"
        "Example: /last30days full\n\n"
        "Custom range:\n"
        "/statsrange YYYY-MM-DD YYYY-MM-DD [compact|full]\n"
        "Example: /statsrange 2024-11-01 2024-11-15 compact\n\n"
        "Note: totals are stored in Supabase, so they survive restarts. "
        "Each Telegram chat (driver group) is tracked separately."
    )
    if update.message:
        await update.message.reply_text(msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle any non-command text message.
    We call OpenAI to extract orders and store them in Supabase for this chat.
    """
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    text = message.text
    chat_state = get_chat_state(chat_id)

    msg_time = message.date
    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=UTC)

    orders = await extract_orders_from_text(text)
    if not orders:
        return

    # Decide which week this order belongs to (hybrid: manual or calendar week).
    now_utc = datetime.now(tz=UTC)
    if chat_state["manual_active"] and chat_state["manual_start"] is not None:
        week_start_utc = chat_state["manual_start"]
    else:
        week_start_utc = last_monday_start(now_utc)

    for o in orders:
        save_order_to_supabase(
            chat_id=chat_id,
            msg_time=msg_time,
            location=o["location"],
            amount=o["amount"],
            week_start_utc=week_start_utc,
        )

    logger.info(
        f"Chat {chat_id}: stored {len(orders)} orders from message {message.message_id}"
    )


async def totals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /totals – show detailed orders for the current hybrid week period.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, label = get_week_period(chat_state)
    orders = fetch_orders_from_supabase(chat_id, start, end)
    text = format_orders_grouped_by_weekday_per_order(orders, label)
    await message.reply_text(text)


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await totals_command(update, context)


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /today – show today’s orders.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    start, end, day_name = get_today_bounds()
    orders = fetch_orders_from_supabase(chat_id, start, end)
    label = f"TODAY ({day_name})"
    text = format_orders_grouped_by_weekday_per_order(orders, label)
    await message.reply_text(text)


async def yesterday_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /yesterday – show yesterday’s orders.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    start, end, day_name = get_yesterday_bounds()
    orders = fetch_orders_from_supabase(chat_id, start, end)
    label = f"YESTERDAY ({day_name})"
    text = format_orders_grouped_by_weekday_per_order(orders, label)
    await message.reply_text(text)


async def revenue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /revenue – daily totals for the current hybrid week period (medium summary).
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, label = get_week_period(chat_state)
    orders = fetch_orders_from_supabase(chat_id, start, end)
    text = summarize_orders_range(orders, f"REVENUE – {label}", detail_level="medium")
    await message.reply_text(text)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stats – summary stats for the current hybrid week period (medium summary).
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    start, end, label = get_week_period(chat_state)
    orders = fetch_orders_from_supabase(chat_id, start, end)
    text = summarize_orders_range(orders, label, detail_level="medium")
    await message.reply_text(text)


async def orders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /orders – alias for /totals.
    """
    await totals_command(update, context)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /clear – clear all stored data for this chat (in Supabase + in-memory state).
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    clear_orders_for_chat(chat_id)
    CHAT_STATE[chat_id] = {"manual_active": False, "manual_start": None}
    await message.reply_text("All stored orders and week settings for this chat have been cleared.")


async def startweek_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /startweek – start a new manual payout week from now (UTC).
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    now_utc = datetime.now(tz=UTC)
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
    now_utc = datetime.now(tz=UTC)

    orders = fetch_orders_from_supabase(chat_id, start, now_utc)
    text = format_orders_grouped_by_weekday_per_order(orders, "MANUAL WEEK FINAL TOTALS")

    chat_state["manual_active"] = False
    chat_state["manual_start"] = None

    await message.reply_text(text)


# ====== RANGE STATS COMMANDS (last7days, alltime, statsrange, etc.) ======

async def handle_last_ndays(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    n_days: int,
):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    detail_level = parse_detail_level(context.args)
    start, end = get_last_n_days_bounds(n_days)
    orders = fetch_orders_from_supabase(chat_id, start, end)
    label = f"LAST {n_days} DAYS (including today)"
    text = summarize_orders_range(orders, label, detail_level)
    await message.reply_text(text)


async def last7days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 7)


async def last14days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 14)


async def last30days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 30)


async def last60days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 60)


async def last90days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 90)


async def last180days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 180)


async def last365days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await handle_last_ndays(update, context, 365)


async def alltime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    detail_level = parse_detail_level(context.args)

    orders = fetch_all_orders(chat_id)
    label = "ALL-TIME TOTALS FOR THIS CHAT"
    text = summarize_orders_range(orders, label, detail_level)
    await message.reply_text(text)


async def statsrange_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /statsrange YYYY-MM-DD YYYY-MM-DD [compact|full]
    """
    message = update.message
    if not message:
        return

    args = context.args
    if len(args) < 2:
        await message.reply_text(
            "Usage: /statsrange YYYY-MM-DD YYYY-MM-DD [compact|full]"
        )
        return

    date_str1 = args[0]
    date_str2 = args[1]
    detail_level = parse_detail_level(args[2:]) if len(args) > 2 else "medium"

    try:
        d1 = datetime.strptime(date_str1, "%Y-%m-%d").date()
        d2 = datetime.strptime(date_str2, "%Y-%m-%d").date()
    except ValueError:
        await message.reply_text("Dates must be in YYYY-MM-DD format.")
        return

    if d2 < d1:
        d1, d2 = d2, d1

    start_local = datetime(d1.year, d1.month, d1.day, 0, 0, 0, tzinfo=TIMEZONE)
    # end inclusive: end of d2 (23:59:59)
    end_local = datetime(d2.year, d2.month, d2.day, 23, 59, 59, tzinfo=TIMEZONE)

    start_utc = start_local.astimezone(UTC)
    end_utc = end_local.astimezone(UTC)

    chat_id = message.chat_id
    orders = fetch_orders_from_supabase(chat_id, start_utc, end_utc)
    label = f"CUSTOM STATS ({d1.isoformat()} → {d2.isoformat()})"
    text = summarize_orders_range(orders, label, detail_level)
    await message.reply_text(text)


# ============= MAIN =============

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is not set.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables are not set.")

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Core commands
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

    # Range/stat commands
    application.add_handler(CommandHandler("last7days", last7days_command))
    application.add_handler(CommandHandler("last14days", last14days_command))
    application.add_handler(CommandHandler("last30days", last30days_command))
    application.add_handler(CommandHandler("last60days", last60days_command))
    application.add_handler(CommandHandler("last90days", last90days_command))
    application.add_handler(CommandHandler("last180days", last180days_command))
    application.add_handler(CommandHandler("last365days", last365days_command))
    application.add_handler(CommandHandler("alltime", alltime_command))
    application.add_handler(CommandHandler("statsrange", statsrange_command))

    # Message handler for orders
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
