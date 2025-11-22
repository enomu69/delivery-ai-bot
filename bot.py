import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# ============= CONFIG =============

TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# timezone for weekly reset logic (your business timezone)
TIMEZONE = ZoneInfo("America/New_York")

# max messages we realistically store in memory per chat (safety cap)
MAX_ORDERS_MEMORY = 1000

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============= STATE =============

# In-memory state (no external DB)
# Structure:
# CHAT_STATE[chat_id] = {
#   "orders": [ { "timestamp": datetime, "location": str, "amount": float } ],
#   "manual_active": bool,
#   "manual_start": datetime | None,
# }
CHAT_STATE = {}


def get_chat_state(chat_id: int) -> dict:
    """Return (and initialize) state for this chat."""
    if chat_id not in CHAT_STATE:
        CHAT_STATE[chat_id] = {
            "orders": [],
            "manual_active": False,
            "manual_start": None,
        }
    return CHAT_STATE[chat_id]


# ============= OPENAI PARSER =============

async def extract_orders_from_text(text: str) -> list[dict]:
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured delivery orders (location + amount) and respond ONLY with JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()

        # Sometimes models wrap JSON in code fences – strip if needed
        if content.startswith("```"):
            content = content.strip("`")
            # after stripping backticks, remove possible language hint like json\n
            if "\n" in content:
                content = content.split("\n", 1)[1].strip()

        data = json.loads(content)

        orders = []
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

    except Exception as e:
        logger.error(f"Error calling OpenAI: {e}")
        return []


# ============= HELPERS =============

def last_monday_start(now_utc: datetime) -> datetime:
    """
    Get the datetime for Monday 00:00 of the current week in local TIMEZONE,
    then convert back to UTC.
    """
    # convert to local tz
    local_now = now_utc.astimezone(TIMEZONE)
    # Monday is 0
    days_since_monday = local_now.weekday()  # 0-6
    monday_local = (local_now - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # back to UTC
    return monday_local.astimezone(ZoneInfo("UTC"))


def prune_old_orders(chat_state: dict):
    """
    Keep the orders list from growing forever.
    Strategy: if > MAX_ORDERS_MEMORY, drop the oldest ones.
    """
    orders = chat_state["orders"]
    if len(orders) > MAX_ORDERS_MEMORY:
        # keep the newest ones
        to_keep = orders[-MAX_ORDERS_MEMORY:]
        chat_state["orders"] = to_keep


def compute_totals(chat_state: dict, start: datetime, end: datetime | None = None):
    """
    Sum amounts per location between [start, end].
    end can be None = now.
    """
    if end is None:
        end = datetime.now(tz=ZoneInfo("UTC"))

    per_location = {}
    grand_total = 0.0

    for order in chat_state["orders"]:
        ts = order["timestamp"]
        if ts < start or ts > end:
            continue
        loc = order["location"]
        amt = order["amount"]
        per_location[loc] = per_location.get(loc, 0.0) + amt
        grand_total += amt

    return per_location, grand_total


def format_totals_message(per_location: dict, grand_total: float, label: str) -> str:
    if not per_location:
        return f"{label} – No orders found in this period."

    lines = [f"{label}"]
    for loc, amt in sorted(per_location.items()):
        lines.append(f"{loc}: ${amt:.2f}")
    lines.append("")
    lines.append(f"TOTAL: ${grand_total:.2f}")
    return "\n".join(lines)


# ============= HANDLERS =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I'm your delivery totals bot.\n\n"
        "I watch this chat for dispatch messages like:\n"
        "• I have Manassas for $35 for 3-4 Alex\n"
        "• I have largo $27 1-2 can go early and Alexandria $30 2-3\n"
        "• Adding Woodbridge to route for $35\n\n"
        "Then I extract the location + amount and keep a running tally.\n\n"
        "Commands:\n"
        "/totals – show current period totals (manual week if active, otherwise this calendar week)\n"
        "/week – same as /totals\n"
        "/startweek – start a manual payout week (overrides auto calendar week)\n"
        "/endweek – end the current manual week and show totals\n"
        "/help – show this message again\n\n"
        "Note: totals are in-memory only. If the bot is restarted or redeployed, counters reset."
    )
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

    # Telegram gives UTC-aware datetime
    msg_time = message.date
    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=ZoneInfo("UTC"))

    # extract with OpenAI
    orders = await extract_orders_from_text(text)
    if not orders:
        return

    # store orders in memory
    for o in orders:
        chat_state["orders"].append(
            {
                "timestamp": msg_time,
                "location": o["location"],
                "amount": o["amount"],
            }
        )

    prune_old_orders(chat_state)

    logger.info(
        f"Chat {chat_id}: stored {len(orders)} orders from message {message.message_id}"
    )


async def totals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /totals – show totals for:
    - If manual week active: from manual_start .. now
    - Else: from last Monday 00:00 (business timezone) .. now
    """
    message = update.message
    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    now_utc = datetime.now(tz=ZoneInfo("UTC"))

    if chat_state["manual_active"] and chat_state["manual_start"] is not None:
        start = chat_state["manual_start"]
        label = "MANUAL WEEK TOTALS"
    else:
        start = last_monday_start(now_utc)
        label = "CALENDAR WEEK TOTALS (since Monday 00:00)"

    per_location, grand_total = compute_totals(chat_state, start=start, end=now_utc)
    text = format_totals_message(per_location, grand_total, label)
    await message.reply_text(text)


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # alias of /totals
    await totals_command(update, context)


async def startweek_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /startweek – clear prior manual state and start a new manual payout week
    from 'now' (UTC).
    """
    message = update.message
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
    chat_id = message.chat_id
    chat_state = get_chat_state(chat_id)

    if not chat_state["manual_active"] or chat_state["manual_start"] is None:
        await message.reply_text(
            "No manual week is currently active. Use /startweek to begin one."
        )
        return

    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    start = chat_state["manual_start"]

    per_location, grand_total = compute_totals(chat_state, start=start, end=now_utc)
    text = format_totals_message(per_location, grand_total, "MANUAL WEEK FINAL TOTALS")

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

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("totals", totals_command))
    application.add_handler(CommandHandler("week", week_command))
    application.add_handler(CommandHandler("startweek", startweek_command))
    application.add_handler(CommandHandler("endweek", endweek_command))

    # Any other text -> attempt extraction
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
