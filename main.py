import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import logging
import time

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

# Log the DATABASE_URL (cleaned for security)
if DATABASE_URL:
    # Remove password for logging
    safe_url = DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL
    logger.info(f"🔍 Database host: {safe_url}")
else:
    logger.error("❌ No DATABASE_URL found!")

# Pydantic models
class MessageRequest(BaseModel):
    content: str
    user_id: str

def get_db_connection():
    """Get database connection with error handling"""
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not configured")
    
    try:
        # Clean the URL and add SSL
        conn_string = DATABASE_URL.strip()
        if "sslmode" not in conn_string:
            conn_string += "?sslmode=require"
            
        conn = psycopg2.connect(
            conn_string,
            cursor_factory=RealDictCursor,
            connect_timeout=30
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def init_database():
    """Initialize database tables - with retries"""
    max_retries = 5
    for attempt in range(max_retries):
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
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error("❌ All database initialization attempts failed")
                return False

# Remove the @app.on_event("startup") - don't fail startup on DB issues
@app.on_event("startup")
async def startup_event():
    """Try to initialize database, but don't fail if it doesn't work"""
    logger.info("🚀 Starting up...")
    # Don't block startup on database issues
    # init_database()  # We'll initialize on first request instead

@app.get("/health")
async def health_check():
    """Health check with database test"""
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
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/init")
async def initialize_database():
    """Manual database initialization endpoint"""
    try:
        success = init_database()
        if success:
            return {"status": "success", "message": "Database initialized"}
        else:
            raise HTTPException(status_code=503, detail="Database initialization failed")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database initialization failed: {str(e)}")

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