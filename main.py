import os
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Journal Backend API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/journal")
# Clean up DATABASE_URL by removing whitespace/newlines
DATABASE_URL = DATABASE_URL.strip()
# Handle Railway's postgres:// URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with connection pooling disabled for Railway
engine = create_engine(DATABASE_URL, poolclass=NullPool)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {e}")

# Pydantic models
class MessageCreate(BaseModel):
    content: str
    user_id: str

class MessageResponse(BaseModel):
    id: int
    content: str
    user_id: str
    timestamp: datetime
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class HealthResponse(BaseModel):
    status: str
    service: str
    database: str
    timestamp: datetime

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and database connectivity"""
    try:
        db = SessionLocal()
        # Test database connection
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "service": "backend",
            "database": "connected",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail={
            "status": "unhealthy",
            "service": "backend",
            "database": "disconnected",
            "error": str(e)
        })

@app.post("/api/save", response_model=dict)
async def save_message(message: MessageCreate):
    """Save a new message to the database"""
    db = SessionLocal()
    try:
        db_message = Message(
            content=message.content,
            user_id=message.user_id
        )
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        logger.info(f"Message saved for user {message.user_id}")
        
        return {
            "success": True,
            "message": "Entry saved successfully",
            "id": db_message.id
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")
    finally:
        db.close()

@app.get("/api/messages/{user_id}", response_model=List[MessageResponse])
async def get_user_messages(user_id: str, limit: Optional[int] = 100, offset: Optional[int] = 0):
    """Retrieve all messages for a specific user"""
    db = SessionLocal()
    try:
        messages = db.query(Message).filter(
            Message.user_id == user_id
        ).order_by(
            Message.timestamp.desc()
        ).limit(limit).offset(offset).all()
        
        logger.info(f"Retrieved {len(messages)} messages for user {user_id}")
        
        return messages
    except Exception as e:
        logger.error(f"Error retrieving messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")
    finally:
        db.close()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Journal Backend API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/save",
            "/api/messages/{user_id}"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)