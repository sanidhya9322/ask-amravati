import os
import json
import sqlite3
import pytz
from datetime import date, datetime, timedelta
from typing import Tuple, List, Dict, Any

from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Try optional AI clients
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from groq import Groq
except Exception:
    Groq = None

# ====================== CONFIG ======================
DAILY_LIMIT = 15
SUB_DURATION_DAYS = 30
PRICE_PER_MONTH = 20
DB_FILE = "users.db"
UPI_ID = "9322264040-2@ybl"
LOCAL_JSON = "amravati_data.json"
ADMIN_IDS = [7122746194]  # replace with your admin telegram id(s)
PAGE_SIZE = 3  # number of items per page for pagination
CONTEXT_EXPIRY_MINUTES = 15
PAGINATION_STATE_TTL_MINUTES = 15
# ====================================================

# Load env
load_dotenv(dotenv_path=".env")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

if not TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not found in .env")

# Init AI clients (if available)
openai_client = OpenAI(api_key=OPENAI_KEY) if (OPENAI_KEY and OpenAI) else None
groq_client = Groq(api_key=GROQ_KEY) if (GROQ_KEY and Groq) else None

# ================== LOCAL DATA LAYER ==================
def load_local_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"⚠️ Local JSON not found at {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return {}

AMRAVATI: Dict[str, Any] = load_local_json(LOCAL_JSON)

# Simple per-user in-memory context (last_category + timestamp)
USER_CONTEXT: Dict[int, Dict[str, Any]] = {}  # e.g. {userid: {"last_category":"clothes", "ts": datetime}}

# Pagination state per user
PAGINATION_STATE: Dict[int, Dict[str, Any]] = {}  
# e.g. {userid: {"category": "clothes", "results": [...], "offset":0, "ts": datetime}}

# Category keywords mapping (used to detect user intent)
CATEGORY_KEYWORDS = {
    "clothes": ["clothes", "dress", "shirt", "pant", "kapde", "shopping", "buy clothes", "shop", "kurta"],
    "food_places": ["food", "eat", "restaurant", "hotel", "mithai", "snack", "dinner", "breakfast", "lunch"],
    "markets.clothes": ["market", "bazaar", "kapda", "clothes market", "shopping market"],
    "colleges": ["college", "university", "engineering", "institute", "campus"],
    "schools": ["school", "cbse", "primary", "secondary"],
    "hospitals": ["hospital", "clinic", "doctor", "emergency", "ambulance"],
    "influencers": ["influencer", "instagram", "youtube", "blogger"],
    "politicians": ["mp", "mla", "politician", "leader"],
    "tourist_places": ["tourist", "visit", "place", "park", "lake", "temple", "sightseeing"],
    "festivals": ["festival", "diwali", "holi", "ganesh", "celebration"],
    "events": ["event", "yatra", "fair"],
    "shopping_malls": ["mall", "big bazaar", "d-mart", "shopping mall"],
    "transport": ["bus", "railway", "station", "airport", "train"],
    "emergency_numbers": ["police", "fire", "ambulance", "emergency"],
    "fun_facts": ["fact", "fun fact", "interesting"],
    "local_slang": ["slang", "phrase", "kiti", "bhau"]
}

def get_google_maps_link(lat: float, lon: float) -> str:
    try:
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    except Exception:
        return ""

def detect_category(user_text: str) -> str:
    text = (user_text or "").lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for k in keywords:
            if k in text:
                return cat
    return ""

def format_entry(entry: dict, category_label: str, emoji: str = "📍") -> str:
    """Format a JSON entry into a neat Markdown string with emojis + map link."""
    name = entry.get("name") or entry.get("title") or "Unnamed"
    area = entry.get("area") or ""
    speciality = entry.get("speciality") or entry.get("type") or ""
    timing = entry.get("timing") or entry.get("hours") or ""
    contact = entry.get("contact") or ""
    description = entry.get("description") or ""

    lat = entry.get("latitude")
    lon = entry.get("longitude")

    lines = [f"{emoji} *{name}*"]
    if area:
        lines.append(f"📍 Area: {area}")
    if speciality:
        lines.append(f"✨ Speciality: {speciality}")
    if timing:
        lines.append(f"⏰ Timing: {timing}")
    if contact:
        lines.append(f"📞 Contact: {contact}")
    if description:
        lines.append(f"ℹ️ {description}")
    if lat and lon:
        lines.append(f"[View on Map]({get_google_maps_link(lat, lon)})")

    return "\n".join(lines)

def get_all_local_results(query: str) -> Tuple[List[str], str]:
    """
    Return ALL formatted local results for query (no pagination).
    Returns (list_of_formatted_strings, detected_category)
    """
    cat = detect_category(query)
    if not cat:
        return [], ""

    results: List[str] = []
    print(f"🔍 Detected category: {cat}")  # debug log

    try:
        if cat in ("markets.clothes", "clothes"):
            clothes_list = AMRAVATI.get("markets", {}).get("clothes", [])
            for entry in clothes_list:
                results.append(format_entry(entry, "Clothes Market", "🏬"))
            return results, "clothes"

        if cat == "food_places" or cat == "food":
            fp = AMRAVATI.get("food_places", [])
            for entry in fp:
                results.append(format_entry(entry, "Food Place", "🍴"))
            return results, "food_places"

        if cat == "hospitals":
            h = AMRAVATI.get("hospitals", [])
            for entry in h:
                results.append(format_entry(entry, "Hospital", "🏥"))
            return results, "hospitals"

        if cat == "colleges":
            coll = AMRAVATI.get("colleges", [])
            for entry in coll:
                results.append(format_entry(entry, "College", "🎓"))
            return results, "colleges"

        if cat == "schools":
            s = AMRAVATI.get("schools", [])
            for entry in s:
                results.append(format_entry(entry, "School", "🏫"))
            return results, "schools"

        if cat == "tourist_places":
            t = AMRAVATI.get("tourist_places", [])
            for entry in t:
                if isinstance(entry, str):
                    results.append(f"🗺️ *{entry}*")
                else:
                    results.append(format_entry(entry, "Tourist Place", "🗺️"))
            return results, "tourist_places"

        if cat == "festivals":
            fests = AMRAVATI.get("festivals", [])
            for entry in fests:
                if isinstance(entry, str):
                    results.append(f"🎉 *{entry}*")
                else:
                    results.append(format_entry(entry, "Festival", "🎉"))
            return results, "festivals"

        if cat == "events":
            ev = AMRAVATI.get("events", [])
            for entry in ev:
                if isinstance(entry, str):
                    results.append(f"📅 *{entry}*")
                else:
                    results.append(format_entry(entry, "Event", "📅"))
            return results, "events"

        if cat == "shopping_malls":
            malls = AMRAVATI.get("shopping_malls", [])
            for entry in malls:
                results.append(format_entry(entry, "Mall", "🏬"))
            return results, "shopping_malls"

        if cat == "influencers":
            infs = AMRAVATI.get("influencers", [])
            for entry in infs:
                results.append(format_entry(entry, "Influencer", "⭐"))
            return results, "influencers"

        if cat == "politicians":
            p = AMRAVATI.get("politicians", [])
            for entry in p:
                results.append(format_entry(entry, "Politician", "🗳️"))
            return results, "politicians"

        if cat == "emergency_numbers":
            em = AMRAVATI.get("emergency_numbers", {})
            for k, v in em.items():
                results.append(f"🚨 *{k.title()}*: `{v}`")
            return results, "emergency_numbers"

        if cat == "fun_facts":
            ff = AMRAVATI.get("fun_facts", [])
            for entry in ff:
                results.append(f"💡 *{entry}*")
            return results, "fun_facts"

        if cat == "local_slang":
            ls = AMRAVATI.get("local_slang", [])
            for s in ls:
                results.append(f"🗣️ *{s}*")
            return results, "local_slang"

    except Exception as e:
        print("⚠️ get_all_local_results error:", e)

    return [], cat

# =================== DATABASE LAYER ===================
def _connect():
    return sqlite3.connect(DB_FILE)

def migrate_db():
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            daily_count INTEGER DEFAULT 0,
            is_subscribed INTEGER DEFAULT 0,
            last_reset TEXT,
            expiry_date TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_user(user_id: int):
    today_str = str(date.today())
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT user_id, daily_count, is_subscribed, last_reset, expiry_date FROM users WHERE user_id=?",
              (user_id,))
    row = c.fetchone()
    if not row:
        c.execute(
            "INSERT INTO users (user_id, daily_count, is_subscribed, last_reset, expiry_date) VALUES (?, 0, 0, ?, NULL)",
            (user_id, today_str)
        )
        conn.commit()
        row = (user_id, 0, 0, today_str, None)
    if row[3] != today_str:
        c.execute("UPDATE users SET daily_count=0, last_reset=? WHERE user_id=?", (today_str, user_id))
        conn.commit()
        row = (row[0], 0, row[2], today_str, row[4])
    is_sub = int(row[2]) == 1
    expiry = row[4]
    if is_sub and expiry:
        try:
            if datetime.strptime(expiry, "%Y-%m-%d").date() < date.today():
                c.execute("UPDATE users SET is_subscribed=0 WHERE user_id=?", (user_id,))
                conn.commit()
                row = (row[0], row[1], 0, row[3], row[4])
        except Exception:
            pass
    conn.close()
    return row

def update_count(user_id: int):
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET daily_count = daily_count + 1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def set_subscription(user_id: int, months: int = 1):
    expiry = date.today() + timedelta(days=SUB_DURATION_DAYS * months)
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET is_subscribed=1, expiry_date=? WHERE user_id=?", (str(expiry), user_id))
    conn.commit()
    conn.close()
    return expiry

def revoke_subscription(user_id: int):
    conn = _connect()
    c = conn.cursor()
    c.execute("UPDATE users SET is_subscribed=0, expiry_date=NULL WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

# ====================== COMMANDS ======================
WELCOME = (
    "🔥 Ask Amravati Premium = Unlimited मजा! 🔥\n\n"
    f"💎 Free users: फक्त {DAILY_LIMIT} msgs/day\n"
    "⚡ Premium users: Unlimited + Fast answers\n\n"
    "👉 Already 500+ Amravati लोक Premium वर गेले 🎉\n"
    "तू पण मागे पडणार का?\n\n"
    f"फक्त ₹{PRICE_PER_MONTH}/month = 1 cutting chai पेक्षा कमी ☕\n\n"
    "🚀 Upgrade आता करा → /subscribe"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME, parse_mode="Markdown")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = get_user(update.message.from_user.id)
    remaining = max(0, DAILY_LIMIT - int(u[1]))
    is_sub = "✅ Active" if int(u[2]) == 1 else "❌ Not active"
    expiry = u[4] if u[4] else "—"
    msg = (
        f"📊 *Your Status*\n"
        f"• Subscription: {is_sub}\n"
        f"• Expiry: {expiry}\n"
        f"• Free messages left today: {remaining}\n\n"
        "Need more? Upgrade with /subscribe"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    msg = (
        f"💎 *Ask Amravati Premium* — ₹{PRICE_PER_MONTH}/month\n"
        f"Pay via UPI: `{UPI_ID}`\n"
        "Payment note: *AMRAVATI PREMIUM*\n\n"
        "After payment, send screenshot to admin.\n"
        "We’ll activate within minutes.\n\n"
        f"Your Telegram ID (for activation): `{user_id}`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id not in ADMIN_IDS:
        return await update.message.reply_text("❌ You are not authorized.")
    if not context.args:
        return await update.message.reply_text("Usage: /approve <user_id>")
    try:
        uid = int(context.args[0])
        expiry = set_subscription(uid, 1)
        await update.message.reply_text(f"✅ User {uid} upgraded until {expiry}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def revoke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id not in ADMIN_IDS:
        return await update.message.reply_text("❌ You are not authorized.")
    if not context.args:
        return await update.message.reply_text("Usage: /revoke <user_id>")
    try:
        uid = int(context.args[0])
        revoke_subscription(uid)
        await update.message.reply_text(f"🚫 User {uid} subscription revoked")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# ================ CORE CHAT HANDLER ==================
SYSTEM_PROMPT = (
   "You are Ask Amravati Bot 🤖 — a hyperlocal digital assistant for Amravati, Maharashtra.\n"
   "- Personality: friendly, warm, concise, local.\n"
   "- Local First → answer like a local guide.\n"
   "- Tone → Marathi/English mix.\n"
   "- Keep answers short & engaging (2–8 sentences).\n"
   "- Use 1–2 emojis naturally.\n"
   "- Honesty → If unsure, say politely: 'माझ्याकडे ह्या विषयी माहिती नाही'.\n"
)

# helper: expire user context after N minutes
CONTEXT_EXPIRY_MINUTES = 15

def set_user_context(user_id: int, category: str):
    USER_CONTEXT[user_id] = {"last_category": category, "ts": datetime.utcnow()}

def get_user_context(user_id: int) -> str:
    ctx = USER_CONTEXT.get(user_id)
    if not ctx:
        return ""
    ts = ctx.get("ts")
    if not ts:
        return ctx.get("last_category", "")
    # expire old context
    if datetime.utcnow() - ts > timedelta(minutes=CONTEXT_EXPIRY_MINUTES):
        del USER_CONTEXT[user_id]
        return ""
    return ctx.get("last_category", "")

async def send_page_for_user(user_id: int, chat_id: int, msg_id: int, offset: int):
    """
    Helper to edit existing message (when user taps 'more') or send new one.
    This will be called from the callback handler.
    """
    state = PAGINATION_STATE.get(user_id)
    if not state:
        return None
    results: List[str] = state.get("results", [])
    category = state.get("category", "")
    # calculate slice
    page = results[offset: offset + PAGE_SIZE]
    has_more = offset + PAGE_SIZE < len(results)

    text = "📍 *Local picks for you:*\n\n" + "\n\n".join(page)
    keyboard = None
    if has_more:
        next_offset = offset + PAGE_SIZE
        callback_data = f"more:{user_id}:{next_offset}"
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("👉 Show more results", callback_data=callback_data)]])

    # edit original message if msg_id provided; otherwise send new message
    return {"text": text, "reply_markup": keyboard, "parse_mode": "Markdown", "disable_web_page_preview": False}

async def handle_show_more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
    await query.answer()  # acknowledge button press quickly

    data = query.data  # expected format: "more:{user_id}:{offset}"
    try:
        prefix, uid_str, offset_str = data.split(":")
        uid = int(uid_str)
        offset = int(offset_str)
    except Exception:
        await query.edit_message_text("Sorry, couldn't load more results.")
        return

    # Only allow the same user to paginate their own results
    clicker_id = query.from_user.id
    if clicker_id != uid:
        await query.answer("This button isn't for you.", show_alert=True)
        return

    # Check state
    state = PAGINATION_STATE.get(uid)
    if not state:
        await query.edit_message_text("Session expired. Please search again.")
        return

    # Clear expired states
    ts = state.get("ts")
    if ts and (datetime.utcnow() - ts > timedelta(minutes=PAGINATION_STATE_TTL_MINUTES)):
        del PAGINATION_STATE[uid]
        await query.edit_message_text("Session expired. Please search again.")
        return

    # Build next page and edit message
    results = state.get("results", [])
    page = results[offset: offset + PAGE_SIZE]
    has_more = offset + PAGE_SIZE < len(results)

    text = "📍 *Local picks for you:*\n\n" + "\n\n".join(page)
    keyboard = None
    if has_more:
        next_offset = offset + PAGE_SIZE
        callback_data = f"more:{uid}:{next_offset}"
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("👉 Show more results", callback_data=callback_data)]])

    try:
        await query.edit_message_text(text=text, reply_markup=keyboard, parse_mode="Markdown", disable_web_page_preview=False)
        # update state timestamp
        state["ts"] = datetime.utcnow()
        state["offset"] = offset
        PAGINATION_STATE[uid] = state
        print(f"[PAGINATION] User {uid} requested more (offset={offset})")
    except Exception as e:
        print("⚠️ Failed to edit message for pagination:", e)
        # fallback: send a new message
        try:
            await context.bot.send_message(chat_id=query.message.chat_id, text=text, reply_markup=keyboard, parse_mode="Markdown", disable_web_page_preview=False)
        except Exception as e2:
            print("⚠️ Also failed to send new message:", e2)

# ====================== CHAT HANDLER ======================
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    text = (update.message.text or "").strip()
    if not text:
        return

    user = get_user(user_id)
    daily_count = int(user[1])
    is_subscribed = int(user[2]) == 1

    # ----- Check daily limit -----
    if not is_subscribed and daily_count >= DAILY_LIMIT:
        await update.message.reply_text(
            f"⚠️ Free limit reached ({DAILY_LIMIT}/day).\n💎 Upgrade for ₹{PRICE_PER_MONTH}/month → /subscribe"
        )
        print(f"⚠️ User {user_id} hit daily free limit")
        return

    # ----- Local JSON first (with pagination) -----
    results_all, category = get_all_local_results(text)
    if results_all:
        # Save pagination state for this user
        PAGINATION_STATE[user_id] = {
            "category": category,
            "results": results_all,
            "offset": 0,
            "ts": datetime.utcnow()
        }
        # build first page
        first_page = results_all[:PAGE_SIZE]
        has_more = len(results_all) > PAGE_SIZE
        text_message = "📍 *Local picks for you:*\n\n" + "\n\n".join(first_page)

        keyboard = None
        if has_more:
            callback_data = f"more:{user_id}:{PAGE_SIZE}"
            keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("👉 Show more results", callback_data=callback_data)]])

        update_count(user_id)  # count only on first reply
        set_user_context(user_id, category)
        print(f"📦 Answer from Local JSON for user {user_id} (Category: {category}) — total results: {len(results_all)}")
        await update.message.reply_text(text_message, reply_markup=keyboard, parse_mode="Markdown", disable_web_page_preview=False)
        return

    # ----- AI fallback (OpenAI -> Groq) -----
    last_category = get_user_context(user_id)
    system_msg = SYSTEM_PROMPT
    if last_category:
        system_msg += f"\n⚡ CONTEXT: User was last interested in '{last_category}'. Keep answers relevant to that if user intent is ambiguous."

    reply = None
    if openai_client:
        try:
            print(f"🤖 Calling OpenAI for user {user_id}...")
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": text},
                ],
                max_tokens=350,
                temperature=0.6,
            )
            reply = (resp.choices[0].message.content or "").strip()
            print(f"✅ Answer from OpenAI for user {user_id}")
        except Exception as e:
            print(f"⚠️ OpenAI error for user {user_id}: {e}")
            reply = None

    if reply is None and groq_client:
        try:
            print(f"⚡ Calling Groq fallback for user {user_id}...")
            resp = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": text},
                ],
                max_tokens=350,
                temperature=0.6,
            )
            reply = (resp.choices[0].message.content or "").strip()
            print(f"✅ Answer from Groq for user {user_id}")
        except Exception as e:
            print(f"⚠️ Groq error for user {user_id}: {e}")
            reply = None

    if not reply:
        reply = "😕 Sorry, I couldn’t fetch an answer right now. Try again later."
        print(f"❌ No AI answer for user {user_id}")

    update_count(user_id)
    await update.message.reply_text(reply)

# ====================== BOOT =========================
def main():
    migrate_db()
    kolkata_tz = pytz.timezone("Asia/Kolkata")
    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )
    app.job_queue.scheduler.configure(timezone=kolkata_tz)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("approve", approve))
    app.add_handler(CommandHandler("revoke", revoke))
    # message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    # callback handler for pagination buttons
    app.add_handler(CallbackQueryHandler(handle_show_more, pattern=r"^more:"))

    print("🤖 Ask Amravati running — Local JSON → OpenAI → Groq | Limits + Subscription | Pagination enabled")
    app.run_polling()

if __name__ == "__main__":
    main()