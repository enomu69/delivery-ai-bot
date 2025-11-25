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

VERBOSE = False  # set True if you want the bot to chat about pending/accepted orders


def get_chat_config(chat_id: int) -> Dict[str, Any]:
    if chat_id not in CHAT_CONFIG:
        CHAT_CONFIG[chat_id] = {
            "summary_mode": DEFAULT_SUMMARY_MODE,
        }
    return CHAT_CONFIG[chat_id]


# ============= SUPABASE HELPERS =============

def supabase_headers() -> Dict[str, str]:
    if not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_KEY not set")
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


def supabase_postgrest_url(path: str) -> str:
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL not set")
    base = SUPABASE_URL
    if not base.endswith("/"):
        base += "/"
    return base + "rest/v1/" + path.lstrip("/")


def supabase_fetch_orders(
    chat_id: int,
    status: Optional[str] = None,
    start_utc: Optional[datetime] = None,
    end_utc: Optional[datetime] = None,
    limit: Optional[int] = None,
    order_desc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch from delivery_orders table with optional filters.
    """
    url = supabase_postgrest_url("delivery_orders")
    params = {
        "chat_id": f"eq.{chat_id}",
        "select": "*",
    }

    if status:
        params["status"] = f"eq.{status}"

    filters = []
    if start_utc:
        filters.append(f"order_timestamp=gte.{start_utc.isoformat()}")
    if end_utc:
        filters.append(f"order_timestamp=lt.{end_utc.isoformat()}")

    if filters:
        params["and"] = ",".join(filters)

    order_dir = "desc" if order_desc else "asc"
    params["order"] = f"order_timestamp.{order_dir}"

    if limit is not None:
        params["limit"] = str(limit)

    qs = urlencode(params, doseq=True)
    full_url = f"{url}?{qs}"

    resp = requests.get(full_url, headers=supabase_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def supabase_fetch_pending(chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch pending orders for a chat, oldest first within the period.
    """
    url = supabase_postgrest_url("delivery_orders")
    params = {
        "chat_id": f"eq.{chat_id}",
        "status": "eq.pending",
        "order": "created_at.asc",
        "limit": str(limit),
        "select": "*",
    }
    qs = urlencode(params, doseq=True)
    full_url = f"{url}?{qs}"

    resp = requests.get(full_url, headers=supabase_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def supabase_insert_orders(
    chat_id: int,
    orders: List[Dict[str, Any]],
    week_start_utc: datetime,
):
    """
    Insert new pending orders.
    Each order dict needs: location, amount (float), order_timestamp (datetime)
    """
    if not orders:
        return

    url = supabase_postgrest_url("delivery_orders")

    payload = []
    for o in orders:
        payload.append(
            {
                "chat_id": str(chat_id),
                "location": o["location"],
                "amount": float(o["amount"]),
                "order_timestamp": o["order_timestamp"].astimezone(UTC).isoformat(),
                "week_start": week_start_utc.astimezone(UTC).isoformat(),
                "status": "pending",
            }
        )

    resp = requests.post(url, headers=supabase_headers(), data=json.dumps(payload), timeout=15)
    resp.raise_for_status()


def supabase_update_status(order_ids: List[int], new_status: str):
    """
    Bulk update status for given IDs.
    """
    if not order_ids:
        return

    url = supabase_postgrest_url("delivery_orders")
    # ids in (1,2,3)
    in_list = ",".join(str(i) for i in order_ids)
    params = {
        "id": f"in.({in_list})",
    }
    qs = urlencode(params, doseq=True)
    full_url = f"{url}?{qs}"

    body = {"status": new_status}

    resp = requests.patch(full_url, headers=supabase_headers(), data=json.dumps(body), timeout=15)
    resp.raise_for_status()


# ============= OPENAI HELPERS =============

def call_openai_chat(messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
        "messages": messages,
    }

    try:
        resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.error(f"Error calling OpenAI HTTP API: {e}")
        return None


def call_openai_analyzer(
    text: str,
    pending_orders: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to:
    - Extract any new orders (location, amount)
    - Decide if text is "confirmation" / "rejection" / ignore
    - Map confirmations/rejections to indices of pending_orders (1-based)

    Returns dict:
    {
      "action": "orders" | "confirm" | "reject" | "mixed" | "ignore",
      "accepted_indices": [1,2],
      "rejected_indices": [],
      "new_orders": [
         {"location": "Annandale", "amount": 35.0, "order_timestamp": datetime}
      ]
    }
    """
    # Build a compact description of pending orders for the model
    pending_lines = []
    for idx, o in enumerate(pending_orders, start=1):
        loc = o.get("location", "")
        amt = o.get("amount", 0)
        pending_lines.append(f"{idx}. {loc} ${float(amt):.2f}")

    pending_text = "\n".join(pending_lines) if pending_lines else "None."

    system_prompt = """
You are an assistant for a weed delivery dispatch team.

You will be given:
- A list of CURRENT PENDING ORDERS for one driver (each has index, location, amount).
- A NEW MESSAGE from the chat.

Your job:
1. Detect any NEW DELIVERY ORDERS described in the message.
   - Each order must have a LOCATION (like "Annandale", "College Park MD") and a DOLLAR AMOUNT.
   - Ignore time windows, emojis, extra commentary.
   - Do NOT guess amounts. If the location is mentioned but no dollar amount, ignore it.
2. Detect whether the message is CONFIRMING or REJECTING any of the pending orders.
   - Confirmation examples: "I'll take it", "I'll grab both", "send it", "coming", "I'll do Annandale", any clear acceptance.
   - Rejection examples: "can't do that", "skip that one", "not taking Manassas", etc.
   - Map confirmations/rejections to the appropriate pending indices (1-based).
   - If you are unsure, leave that order untouched (neither accepted nor rejected).
3. Classify the overall ACTION:
   - "orders" if the message mainly describes new orders.
   - "confirm" if it mainly confirms existing pending orders.
   - "reject" if it mainly rejects existing pending orders.
   - "mixed" if both new orders and confirmations/rejections appear.
   - "ignore" if it is just small-talk, questions, etc. with no impact on orders.

You MUST respond with STRICT JSON and NOTHING ELSE.
The JSON format MUST be:

{
  "action": "orders",
  "accepted_indices": [1, 2],
  "rejected_indices": [],
  "new_orders": [
    {
      "location": "Annandale",
      "amount": 35.0
    }
  ]
}

Notes:
- accepted_indices and rejected_indices refer to the numbering of CURRENT PENDING ORDERS.
- new_orders is a list; can be empty if none.
- If there is any ambiguity, be conservative (prefer "ignore" or no indices).
- If the message is just "ok", "thanks", etc., treat as "ignore".
"""

    user_prompt = f"""
CURRENT PENDING ORDERS:
{pending_text}

NEW MESSAGE:
\"\"\"{text}\"\"\"

Remember: output ONLY valid JSON, nothing else.
"""

    content = call_openai_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    if content is None:
        return None

    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            content = content.split("\n", 1)[1].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Analyzer JSON parse error: {e} | content={content[:200]}")
        return None

    action = data.get("action", "ignore")
    accepted_indices = data.get("accepted_indices", [])
    rejected_indices = data.get("rejected_indices", [])
    new_orders_raw = data.get("new_orders", [])

    new_orders: List[Dict[str, Any]] = []
    now_utc = datetime.now(tz=UTC)

    for item in new_orders_raw:
        try:
            loc = str(item.get("location", "")).strip()
            amt = float(item.get("amount", 0))
            if loc and amt > 0:
                new_orders.append(
                    {
                        "location": loc,
                        "amount": amt,
                        "order_timestamp": now_utc,
                    }
                )
        except Exception:
            continue

    return {
        "action": action,
        "accepted_indices": [int(i) for i in accepted_indices if isinstance(i, int)],
        "rejected_indices": [int(i) for i in rejected_indices if isinstance(i, int)],
        "new_orders": new_orders,
    }


# ============= HELPERS (DATES, SUMMARY) =============

def start_of_local_day(dt_utc: datetime) -> datetime:
    local = dt_utc.astimezone(TIMEZONE)
    local_midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(UTC)


def last_monday_start(now_utc: datetime) -> datetime:
    local_now = now_utc.astimezone(TIMEZONE)
    days_since_monday = local_now.weekday()
    monday_local = (local_now - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return monday_local.astimezone(UTC)


def parse_date_yyyy_mm_dd(s: str) -> Optional[datetime]:
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return TIMEZONE.localize(dt).astimezone(UTC) if hasattr(TIMEZONE, "localize") else dt.replace(tzinfo=TIMEZONE).astimezone(UTC)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y/%m/%d")
            return dt.replace(tzinfo=TIMEZONE).astimezone(UTC)
        except Exception:
            return None


def summarize_orders(
    orders: List[Dict[str, Any]],
    label: str,
    summary_mode: str = "full",
) -> str:
    """
    Summarize orders grouped by day of week and location.
    summary_mode: "full", "medium", "compact"
    """
    if not orders:
        return f"{label}\nNo accepted orders in this period."

    # Group by day (local) then location
    grouped: Dict[str, Dict[str, float]] = {}
    total_amount = 0.0
    total_orders = 0

    for o in orders:
        ts_str = o.get("order_timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(UTC)
            except Exception:
                ts = datetime.now(tz=UTC)
        else:
            ts = datetime.now(tz=UTC)

        local = ts.astimezone(TIMEZONE)
        day_label = local.strftime("%A")  # Monday, Tuesday...

        loc = o.get("location", "Unknown")
        amt = float(o.get("amount", 0))

        grouped.setdefault(day_label, {})
        grouped[day_label][loc] = grouped[day_label].get(loc, 0.0) + amt

        total_amount += amt
        total_orders += 1

    lines: List[str] = [label]

    # Order days by actual chronological order Mon..Sun
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sorted_days = sorted(grouped.keys(), key=lambda d: day_order.index(d) if d in day_order else 99)

    for day in sorted_days:
        per_loc = grouped[day]
        if summary_mode == "compact":
            # Just show day total
            day_total = sum(per_loc.values())
            lines.append(f"{day}: ${day_total:.2f}")
        else:
            lines.append(f"{day}:")
            # sort by location name
            for loc, amt in sorted(per_loc.items(), key=lambda x: x[0].lower()):
                lines.append(f"  - {loc}: ${amt:.2f}")

    avg_per_day = total_amount / max(len(sorted_days), 1)
    avg_per_order = total_amount / max(total_orders, 1)

    if summary_mode in ("full", "medium"):
        lines.append("")
        lines.append(f"TOTAL orders: {total_orders}")
        lines.append(f"TOTAL amount: ${total_amount:.2f}")
        lines.append(f"Avg per day: ${avg_per_day:.2f}")
        lines.append(f"Avg per order: ${avg_per_order:.2f}")

    if summary_mode == "full":
        # Could add extra stats later if you want
        pass

    return "\n".join(lines)


# ============= COMMAND HANDLERS =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I'm your delivery totals bot.\n\n"
        "I watch this chat for dispatch messages like:\n"
        "• I have Manassas for $35 for 3-4 Alex\n"
        "• I have Largo $27 1-2 can go early and Alexandria $30 2-3\n"
        "• Adding Woodbridge to route for $35\n\n"
        "I store the orders in Supabase and keep track of which ones are accepted or rejected "
        "based on follow-up messages (\"I'll take Annandale\", \"skip College Park\", etc.).\n\n"
        "Main commands:\n"
        "/totals – this week’s accepted orders (Mon–Sun)\n"
        "/week – same as /totals\n"
        "/today – today only\n"
        "/yesterday – yesterday only\n"
        "/last7days – last 7 days\n"
        "/last14days – last 14 days\n"
        "/last30days – last 30 days\n"
        "/last60days – last 60 days\n"
        "/last90days – last 90 days\n"
        "/range YYYY-MM-DD YYYY-MM-DD – custom date range\n"
        "/pending – show current pending orders\n"
        "/mode <full|medium|compact> – change how detailed the summaries are\n\n"
        "You can run commands in any driver chat; each Telegram chat is kept separate.\n"
        "Note: summaries only include *accepted* orders.\n"
        "Orders are stored in Supabase so they survive restarts."
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
    cfg = get_chat_config(chat_id)

    parts = message.text.strip().split()
    if len(parts) < 2:
        msg = (
            f"Current summary mode: {cfg['summary_mode']}\n"
            "Usage: /mode full | medium | compact\n"
            "Tip: you can also do /last30days compact or /totals full to override the mode just for that command."
        )
        if update.message:
            await update.message.reply_text(msg)
        return

    choice = parts[1].lower()
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

    msg_time_utc = message.date
    if msg_time_utc.tzinfo is None:
        msg_time_utc = msg_time_utc.replace(tzinfo=UTC)
    else:
        msg_time_utc = msg_time_utc.astimezone(UTC)

    # Determine week_start for this message
    week_start_utc = last_monday_start(msg_time_utc)

    # Fetch current pending orders for this chat
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
        supabase_insert_orders(chat_id, new_orders, week_start_utc)

    # Update statuses based on acceptance/rejection
    if accepted_ids:
        supabase_update_status(accepted_ids, "accepted")
    if rejected_ids:
        supabase_update_status(rejected_ids, "rejected")

    # Build feedback message (only if VERBOSE)
    if VERBOSE:
        parts: List[str] = []

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

        if parts:
            await message.reply_text("\n\n".join(parts), parse_mode="Markdown")
    else:
        # In silent mode we just log high-level actions and do not send chat messages
        if new_orders:
            logger.info(
                "Chat %s: stored %d pending orders (silent mode)",
                chat_id,
                len(new_orders),
            )
        if accepted_ids:
            logger.info(
                "Chat %s: accepted %d pending orders (silent mode)",
                chat_id,
                len(accepted_ids),
            )
        if rejected_ids:
            logger.info(
                "Chat %s: rejected %d pending orders (silent mode)",
                chat_id,
                len(rejected_ids),
            )


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
    cfg = get_chat_config(chat_id)

    if end_utc is None:
        end_utc = datetime.now(tz=UTC)

    if summary_mode is None:
        summary_mode = cfg["summary_mode"]

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
    now_utc = datetime.now(tz=UTC)
    start = start_of_local_day(now_utc)
    label = "TODAY – accepted orders"
    await generic_range_command(update, context, label, start, now_utc)


async def yesterday_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    now_utc = datetime.now(tz=UTC)
    today_start = start_of_local_day(now_utc)
    yesterday_start = today_start - timedelta(days=1)
    label = "YESTERDAY – accepted orders"
    await generic_range_command(update, context, label, yesterday_start, today_start)


async def totals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /totals – calendar week (Mon–Sun) in local timezone.
    """
    message = update.message
    if not message:
        return
    chat_id = message.chat_id
    now_utc = datetime.now(tz=UTC)
    start = last_monday_start(now_utc)
    label = "THIS WEEK (Mon–Sun) – accepted orders"
    await generic_range_command(update, context, label, start, now_utc)


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await totals_command(update, context)


async def last7days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    now_utc = datetime.now(tz=UTC)
    start = now_utc - timedelta(days=7)
    label = "LAST 7 DAYS – accepted orders"

    parts = message.text.strip().split()
    summary_mode = None
    if len(parts) > 1:
        summary_mode = parts[1].lower()

    await generic_range_command(update, context, label, start, now_utc, summary_mode)


async def last14days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    now_utc = datetime.now(tz=UTC)
    start = now_utc - timedelta(days=14)
    label = "LAST 14 DAYS – accepted orders"

    parts = message.text.strip().split()
    summary_mode = None
    if len(parts) > 1:
        summary_mode = parts[1].lower()

    await generic_range_command(update, context, label, start, now_utc, summary_mode)


async def last30days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    now_utc = datetime.now(tz=UTC)
    start = now_utc - timedelta(days=30)
    label = "LAST 30 DAYS – accepted orders"

    parts = message.text.strip().split()
    summary_mode = None
    if len(parts) > 1:
        summary_mode = parts[1].lower()

    await generic_range_command(update, context, label, start, now_utc, summary_mode)


async def last60days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    now_utc = datetime.now(tz=UTC)
    start = now_utc - timedelta(days=60)
    label = "LAST 60 DAYS – accepted orders"

    parts = message.text.strip().split()
    summary_mode = None
    if len(parts) > 1:
        summary_mode = parts[1].lower()

    await generic_range_command(update, context, label, start, now_utc, summary_mode)


async def last90days_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message:
        return
    now_utc = datetime.now(tz=UTC)
    start = now_utc - timedelta(days=90)
    label = "LAST 90 DAYS – accepted orders"

    parts = message.text.strip().split()
    summary_mode = None
    if len(parts) > 1:
        summary_mode = parts[1].lower()

    await generic_range_command(update, context, label, start, now_utc, summary_mode)


async def range_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /range YYYY-MM-DD YYYY-MM-DD [mode]
    """
    message = update.message
    if not message:
        return

    parts = message.text.strip().split()
    if len(parts) < 3:
        await message.reply_text(
            "Usage: /range YYYY-MM-DD YYYY-MM-DD [full|medium|compact]"
        )
        return

    start_s = parts[1]
    end_s = parts[2]
    mode = parts[3].lower() if len(parts) > 3 else None

    start_utc = parse_date_yyyy_mm_dd(start_s)
    end_utc = parse_date_yyyy_mm_dd(end_s)
    if not start_utc or not end_utc:
        await message.reply_text("Could not parse dates. Use YYYY-MM-DD.")
        return

    if end_utc < start_utc:
        await message.reply_text("End date must be on or after start date.")
        return

    label = f"RANGE {start_s}..{end_s} – accepted orders"
    await generic_range_command(update, context, label, start_utc, end_utc, mode)


async def pending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /pending – show current pending orders for this chat.
    """
    message = update.message
    if not message:
        return

    chat_id = message.chat_id
    pending = supabase_fetch_pending(chat_id, limit=20)
    if not pending:
        await message.reply_text("No pending orders for this chat.")
        return

    lines = ["Current pending orders:"]
    for idx, o in enumerate(pending, start=1):
        loc = o.get("location", "Unknown")
        amt = float(o.get("amount", 0))
        lines.append(f"{idx}. {loc}: ${amt:.2f}")

    await message.reply_text("\n".join(lines))


# ============= MAIN =============

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN environment variable is not set.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase env vars not set.")

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))

    application.add_handler(CommandHandler("totals", totals_command))
    application.add_handler(CommandHandler("week", week_command))
    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("yesterday", yesterday_command))

    application.add_handler(CommandHandler("last7days", last7days_command))
    application.add_handler(CommandHandler("last14days", last14days_command))
    application.add_handler(CommandHandler("last30days", last30days_command))
    application.add_handler(CommandHandler("last60days", last60days_command))
    application.add_handler(CommandHandler("last90days", last90days_command))
    application.add_handler(CommandHandler("range", range_command))

    application.add_handler(CommandHandler("pending", pending_command))

    # All non-command text goes through the NLP analyzer
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
