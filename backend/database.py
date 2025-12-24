# backend/database.py
import sqlite3
import json
from datetime import datetime

DB_NAME = "chat_memory.db"

def init_db():
    """Initializes the SQLite database for chat history."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_message(session_id, role, content):
    """Adds a single message to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", 
                   (session_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(session_id, limit=10):
    """Retrieves the last N messages for context."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", 
                   (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    # Return in reverse order (oldest to newest) for LLM context
    return [{"role": r[0], "content": r[1]} for r in rows][::-1]