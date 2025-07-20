import os
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Journaling Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLite database file
DATABASE_FILE = "journal.db"

# Pydantic models
class MessageRequest(BaseModel):
    content: str
    user_id: str

def get_db_connection():
    """Get SQLite database connection"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row  # Enable row access by column name
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def init_database():
    """Initialize SQLite database and tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_id 
            ON messages(user_id)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ SQLite database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("🚀 Starting up with SQLite...")
    success = init_database()
    if success:
        logger.info("✅ Ready to serve requests")
    else:
        logger.warning("⚠️ Database initialization failed, but continuing...")

@app.get("/health")
async def health_check():
    """Health check with SQLite database test"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "sqlite_connected",
            "database_file": DATABASE_FILE,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "sqlite_disconnected", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/init")
async def initialize_database():
    """Manual database initialization endpoint"""
    try:
        success = init_database()
        if success:
            return {"status": "success", "message": "SQLite database initialized"}
        else:
            raise HTTPException(status_code=503, detail="Database initialization failed")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database initialization failed: {str(e)}")

@app.post("/api/save")
async def save_message(message: MessageRequest):
    """Save a journal entry to SQLite"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert message and get the ID
        cursor.execute(
            "INSERT INTO messages (content, user_id, timestamp) VALUES (?, ?, ?)",
            (message.content, message.user_id, datetime.utcnow().isoformat())
        )
        
        message_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"💾 Saved message {message_id} for user {message.user_id}")
        
        return {
            "status": "success",
            "message_id": message_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Save message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

@app.get("/api/messages/{user_id}")
async def get_messages(user_id: str, limit: int = 100, offset: int = 0):
    """Get all messages for a user from SQLite"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get messages for user with pagination
        cursor.execute(
            """SELECT id, content, user_id, timestamp 
               FROM messages 
               WHERE user_id = ? 
               ORDER BY timestamp DESC 
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset)
        )
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert rows to dictionaries
        messages = []
        for row in rows:
            messages.append({
                "id": row["id"],
                "content": row["content"],
                "user_id": row["user_id"],
                "timestamp": row["timestamp"]
            })
        
        logger.info(f"📖 Retrieved {len(messages)} messages for user {user_id}")
        
        return {
            "status": "success",
            "messages": messages,
            "count": len(messages)
        }
        
    except Exception as e:
        logger.error(f"Get messages failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Journaling Backend (SQLite)",
        "version": "1.0.0",
        "database": "SQLite",
        "endpoints": [
            "/health",
            "/init", 
            "/api/save",
            "/api/messages/{user_id}"
        ],
        "status": "ready"
    }

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total message count
        cursor.execute("SELECT COUNT(*) as total FROM messages")
        total_messages = cursor.fetchone()["total"]
        
        # Get unique user count
        cursor.execute("SELECT COUNT(DISTINCT user_id) as users FROM messages")
        unique_users = cursor.fetchone()["users"]
        
        cursor.close()
        conn.close()
        
        return {
            "total_messages": total_messages,
            "unique_users": unique_users,
            "database_file": DATABASE_FILE,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)