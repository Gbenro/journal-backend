import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import logging

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

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Pydantic models
class MessageRequest(BaseModel):
    content: str
    user_id: str

class MessageResponse(BaseModel):
    id: int
    content: str
    user_id: str
    timestamp: datetime

def get_db_connection():
    """Get database connection with error handling"""
    try:
        # Add SSL requirement for Railway
        if "sslmode" not in DATABASE_URL:
            conn_string = DATABASE_URL + "?sslmode=require"
        else:
            conn_string = DATABASE_URL
            
        conn = psycopg2.connect(
            conn_string,
            cursor_factory=RealDictCursor,
            connect_timeout=30
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

def init_database():
    """Initialize database tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()

@app.get("/health")
async def health_check():
    """Simple health check"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )

@app.post("/api/save")
async def save_message(message: MessageRequest):
    """Save a journal entry"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO messages (content, user_id) VALUES (%s, %s) RETURNING id, timestamp",
            (message.content, message.user_id)
        )
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "message_id": result["id"],
            "timestamp": result["timestamp"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Save message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

@app.get("/api/messages/{user_id}")
async def get_messages(user_id: str):
    """Get all messages for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, content, user_id, timestamp FROM messages WHERE user_id = %s ORDER BY timestamp DESC",
            (user_id,)
        )
        
        messages = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "messages": [dict(msg) for msg in messages]
        }
        
    except Exception as e:
        logger.error(f"Get messages failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)