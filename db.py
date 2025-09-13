import sqlite3

def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            daily_count INTEGER DEFAULT 0,
            is_subscribed INTEGER DEFAULT 0,
            last_reset DATE
        )
    """)
    conn.commit()
    conn.close()
