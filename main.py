import os
import sqlite3
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import logging
from typing import List, Optional, Dict, Any
import json

# Configure logging - Last updated: 2025-07-21 for Railway deployment
# This ensures proper logging across development and production environments
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Journaling Backend with Tags")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent storage configuration - Critical for Railway deployments
# This configuration ensures data persists across container restarts
def is_railway_environment():
    """Check if running on Railway - determines storage strategy"""
    return bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_DEPLOYMENT_ID"))

def ensure_data_directory():
    """Ensure data directory exists for persistent storage"""
    if is_railway_environment():
        data_dir = "/app/data"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Data directory ensured: {data_dir}")
    else:
        # For local development, ensure current directory is writable
        logger.info("🏠 Local development mode - using current directory")

def get_database_path():
    """Get appropriate database path for environment"""
    if is_railway_environment():
        return "/app/data/journal.db"
    else:
        return "journal.db"

def get_storage_info():
    """Get storage configuration info"""
    if is_railway_environment():
        return {
            "platform": "Railway",
            "storage": "Persistent Volume",
            "path": "/app/data/journal.db",
            "persistent": True
        }
    else:
        return {
            "platform": "Local",
            "storage": "Local File",
            "path": "journal.db",
            "persistent": False
        }

# Get database file path
DATABASE_FILE = get_database_path()

# Predefined tags data
PREDEFINED_TAGS = [
    # Emotions
    {"name": "gratitude", "category": "emotion", "color": "#4CAF50", "description": "Thankful moments"},
    {"name": "joy", "category": "emotion", "color": "#FFC107", "description": "Happy experiences"},
    {"name": "stress", "category": "emotion", "color": "#F44336", "description": "Stressful situations"},
    {"name": "reflection", "category": "emotion", "color": "#9C27B0", "description": "Deep thinking"},
    {"name": "accomplishment", "category": "emotion", "color": "#2196F3", "description": "Achievements"},
    
    # Life Areas
    {"name": "work", "category": "life", "color": "#607D8B", "description": "Professional activities"},
    {"name": "family", "category": "life", "color": "#E91E63", "description": "Family interactions"},
    {"name": "friends", "category": "life", "color": "#FF9800", "description": "Social connections"},
    {"name": "health", "category": "life", "color": "#4CAF50", "description": "Physical/mental health"},
    {"name": "learning", "category": "life", "color": "#3F51B5", "description": "Educational experiences"},
    
    # Activities
    {"name": "coding", "category": "activity", "color": "#795548", "description": "Programming work"},
    {"name": "reading", "category": "activity", "color": "#009688", "description": "Reading activities"},
    {"name": "exercise", "category": "activity", "color": "#FF5722", "description": "Physical activities"},
    {"name": "travel", "category": "activity", "color": "#CDDC39", "description": "Travel experiences"},
    {"name": "breakthrough", "category": "activity", "color": "#FFD700", "description": "Major insights"},
    
    # Goals
    {"name": "goal-setting", "category": "goal", "color": "#8BC34A", "description": "Setting objectives"},
    {"name": "progress", "category": "goal", "color": "#00BCD4", "description": "Making progress"},
    {"name": "challenge", "category": "goal", "color": "#FF6F00", "description": "Facing difficulties"},
    {"name": "milestone", "category": "goal", "color": "#6A1B9A", "description": "Important achievements"}
]

# Auto-tagging patterns
AUTO_TAG_PATTERNS = {
    "work": ["job", "office", "boss", "colleague", "project", "meeting", "deadline", "career", "professional", "business"],
    "coding": ["code", "programming", "bug", "deploy", "api", "database", "frontend", "backend", "python", "javascript", "development"],
    "family": ["mom", "dad", "sister", "brother", "parents", "family", "home", "mother", "father", "sibling"],
    "friends": ["friend", "friends", "buddy", "pal", "companion", "social", "hangout", "party"],
    "gratitude": ["thankful", "grateful", "appreciate", "blessed", "fortunate", "thank", "appreciate"],
    "stress": ["stressed", "overwhelmed", "anxious", "pressure", "worried", "tense", "anxiety"],
    "accomplishment": ["achieved", "completed", "success", "proud", "accomplished", "finished", "done"],
    "breakthrough": ["breakthrough", "insight", "discovered", "realized", "epiphany", "eureka", "revelation"],
    "learning": ["learned", "study", "course", "book", "education", "skill", "training", "knowledge"],
    "health": ["exercise", "workout", "doctor", "medicine", "sleep", "tired", "fitness", "wellness"],
    "goal-setting": ["goal", "plan", "objective", "target", "aim", "resolution", "planning"],
    "joy": ["happy", "excited", "delighted", "joyful", "cheerful", "elated", "wonderful"],
    "reflection": ["thinking", "pondering", "contemplating", "reflecting", "considering", "meditation"],
    "travel": ["trip", "vacation", "journey", "visit", "explore", "adventure", "destination"],
    "reading": ["book", "novel", "article", "reading", "literature", "story", "chapter"],
    "exercise": ["gym", "running", "cycling", "sports", "fitness", "training", "workout"]
}

# Pydantic models
class MessageRequest(BaseModel):
    content: str
    user_id: str
    manual_tags: Optional[List[str]] = []
    auto_tag: Optional[bool] = True

class TagCreate(BaseModel):
    name: str
    category: Optional[str] = "custom"
    color: Optional[str] = "#808080"
    description: Optional[str] = ""

class TagSuggestionRequest(BaseModel):
    content: str
    limit: Optional[int] = 5

# Auto-tagging engine - Intelligent content analysis for automatic tag suggestions
# Updated 2025-07-21: Enhanced keyword matching and confidence scoring
class AutoTagger:
    def __init__(self):
        pass
    
    def suggest_tags(self, content: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Suggest tags based on content analysis"""
        content_lower = content.lower()
        suggestions = []
        
        # Keyword matching with confidence scoring
        for tag_name, keywords in AUTO_TAG_PATTERNS.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches > 0:
                confidence = min(0.95, 0.6 + (matches * 0.1))
                suggestions.append({
                    "name": tag_name,
                    "confidence": confidence,
                    "source": "keyword_match",
                    "matches": matches
                })
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:limit]
    
    def apply_auto_tags(self, content: str, existing_tags: List[str]) -> List[Dict[str, Any]]:
        """Apply auto-tags that don't conflict with manual tags"""
        suggestions = self.suggest_tags(content, limit=10)
        auto_tags = []
        
        for suggestion in suggestions:
            if suggestion["name"] not in existing_tags and suggestion["confidence"] > 0.7:
                auto_tags.append({
                    "name": suggestion["name"],
                    "confidence": suggestion["confidence"],
                    "source": "auto"
                })
        
        return auto_tags[:3]  # Limit to 3 auto tags

def get_db_connection():
    """Get SQLite database connection with persistent path"""
    ensure_data_directory()
    db_path = get_database_path()
    
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    except Exception as e:
        logger.error(f"Database connection failed at {db_path}: {e}")
        raise

def init_database():
    """Initialize SQLite database with enhanced schema and persistent storage"""
    ensure_data_directory()
    db_path = get_database_path()
    
    logger.info(f"🔧 Initializing database at: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
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
        
        # Create tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(50) UNIQUE NOT NULL,
                category VARCHAR(30),
                is_predefined BOOLEAN DEFAULT FALSE,
                color VARCHAR(7),
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create entry_tags relationship table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entry_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence FLOAT DEFAULT 1.0,
                is_auto_tagged BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id),
                UNIQUE(message_id, tag_id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_timestamp ON messages(user_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_message_id ON entry_tags(message_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_tag_id ON entry_tags(tag_id)")
        
        # Insert predefined tags if they don't exist
        for tag_data in PREDEFINED_TAGS:
            cursor.execute("""
                INSERT OR IGNORE INTO tags (name, category, is_predefined, color, description)
                VALUES (?, ?, TRUE, ?, ?)
            """, (tag_data["name"], tag_data["category"], tag_data["color"], tag_data["description"]))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Enhanced SQLite database with tags initialized successfully at: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed at {db_path}: {e}")
        return False

def get_or_create_tag(conn: sqlite3.Connection, tag_name: str) -> int:
    """Get existing tag ID or create new tag"""
    cursor = conn.cursor()
    
    # Try to get existing tag
    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
    result = cursor.fetchone()
    
    if result:
        return result["id"]
    
    # Create new custom tag
    cursor.execute("""
        INSERT INTO tags (name, category, is_predefined, color, description)
        VALUES (?, 'custom', FALSE, '#808080', ?)
    """, (tag_name, f"Custom tag: {tag_name}"))
    
    return cursor.lastrowid

def apply_tags_to_entry(conn: sqlite3.Connection, message_id: int, tags_data: List[Dict[str, Any]]):
    """Apply tags to a journal entry"""
    cursor = conn.cursor()
    
    for tag_info in tags_data:
        tag_id = get_or_create_tag(conn, tag_info["name"])
        
        # Insert entry-tag relationship
        cursor.execute("""
            INSERT OR IGNORE INTO entry_tags (message_id, tag_id, confidence, is_auto_tagged)
            VALUES (?, ?, ?, ?)
        """, (message_id, tag_id, tag_info.get("confidence", 1.0), tag_info.get("source") == "auto"))

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup with comprehensive logging
    Railway deployment test - 2025-07-21 - Verifying persistent volume mount"""
    logger.info("🚀 Starting Mirror Scribe Backend with Intelligent Tags...")
    logger.info(f"📁 Environment: {'Railway' if is_railway_environment() else 'Local'}")
    logger.info(f"💾 Database path: {get_database_path()}")
    logger.info(f"📂 Data directory: {'/app/data' if is_railway_environment() else 'local'}")
    
    # Initialize database
    success = init_database()
    
    if success:
        # Check existing data
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM tags")
            tag_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM entry_tags")
            tag_applications = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"📝 Existing journal entries: {message_count}")
            logger.info(f"🏷️ Available tags: {tag_count}")
            logger.info(f"🔗 Tag applications: {tag_applications}")
        except Exception as e:
            logger.warning(f"⚠️ Could not check existing data: {e}")
        
        logger.info("✅ Mirror Scribe Backend ready with persistent storage!")
    else:
        logger.warning("⚠️ Database initialization failed, but continuing...")

@app.get("/health")
async def health_check():
    """Enhanced health check with persistent storage info"""
    try:
        db_path = get_database_path()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Test database connectivity
        cursor.execute("SELECT 1")
        
        # Get comprehensive stats
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tags")
        tag_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM entry_tags")
        tag_applications = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Check if database file exists and get size
        db_exists = os.path.exists(db_path)
        db_size = os.path.getsize(db_path) if db_exists else 0
        
        return {
            "status": "healthy",
            "database": "sqlite_connected",
            "database_path": db_path,
            "database_exists": db_exists,
            "database_size_bytes": db_size,
            "storage_info": get_storage_info(),
            "persistent_storage": is_railway_environment(),
            "data_directory_exists": os.path.exists("/app/data") if is_railway_environment() else True,
            "features": ["tagging", "auto_tagging", "search", "persistent_storage"],
            "stats": {
                "total_entries": message_count,
                "total_tags": tag_count,
                "tag_applications": tag_applications
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "sqlite_disconnected", 
            "error": str(e),
            "database_path": get_database_path(),
            "storage_info": get_storage_info(),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/init")
async def initialize_database():
    """Manual database initialization endpoint"""
    try:
        success = init_database()
        if success:
            return {"status": "success", "message": "Enhanced SQLite database with tags initialized"}
        else:
            raise HTTPException(status_code=503, detail="Database initialization failed")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database initialization failed: {str(e)}")

@app.get("/storage-info")
async def get_storage_information():
    """Get comprehensive storage configuration details"""
    try:
        storage_info = get_storage_info()
        db_path = get_database_path()
        
        # Check if database file exists
        db_exists = os.path.exists(db_path)
        
        # Get file size if exists
        db_size = os.path.getsize(db_path) if db_exists else 0
        
        # Check data directory
        data_dir_exists = True
        if is_railway_environment():
            data_dir_exists = os.path.exists("/app/data")
        
        # Get database stats if available
        stats = {"total_entries": 0, "total_tags": 0, "tag_applications": 0}
        if db_exists:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM messages")
                stats["total_entries"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM tags")  
                stats["total_tags"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM entry_tags")
                stats["tag_applications"] = cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass  # Stats are optional
        
        return {
            "storage_config": storage_info,
            "database_path": db_path,
            "database_exists": db_exists,
            "database_size_bytes": db_size,
            "database_size_mb": round(db_size / (1024 * 1024), 2),
            "data_directory_exists": data_dir_exists,
            "environment": {
                "is_railway": is_railway_environment(),
                "railway_env": os.getenv("RAILWAY_ENVIRONMENT"),
                "railway_project": os.getenv("RAILWAY_PROJECT_ID"),
                "railway_deployment": os.getenv("RAILWAY_DEPLOYMENT_ID")
            },
            "persistence_test": {
                "description": "Database will persist across Railway deployments",
                "volume_mount": "/app/data" if is_railway_environment() else "N/A",
                "backup_recommended": True
            },
            "database_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Storage info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage info: {str(e)}")

# Tag management endpoints
@app.get("/api/tags")
async def get_all_tags():
    """Get all available tags"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, category, is_predefined, color, description, created_at
            FROM tags
            ORDER BY category, name
        """)
        
        tags = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "tags": tags,
            "count": len(tags)
        }
        
    except Exception as e:
        logger.error(f"Get tags failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")

@app.get("/api/tags/categories")
async def get_tags_by_category():
    """Get tags grouped by category"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT category, name, color, description
            FROM tags
            ORDER BY category, name
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Group by category
        categories = {}
        for row in rows:
            category = row["category"] or "uncategorized"
            if category not in categories:
                categories[category] = []
            
            categories[category].append({
                "name": row["name"],
                "color": row["color"],
                "description": row["description"]
            })
        
        return {
            "status": "success",
            "categories": categories
        }
        
    except Exception as e:
        logger.error(f"Get tags by category failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tags by category: {str(e)}")

@app.post("/api/tags")
async def create_tag(tag: TagCreate):
    """Create a new custom tag"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tags (name, category, is_predefined, color, description)
            VALUES (?, ?, FALSE, ?, ?)
        """, (tag.name, tag.category, tag.color, tag.description))
        
        tag_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✨ Created new tag: {tag.name}")
        
        return {
            "status": "success",
            "message": f"Tag '{tag.name}' created successfully",
            "tag_id": tag_id
        }
        
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"Tag '{tag.name}' already exists")
    except Exception as e:
        logger.error(f"Create tag failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create tag: {str(e)}")

@app.post("/api/tags/suggestions")
async def suggest_tags(request: TagSuggestionRequest):
    """Get tag suggestions for given content"""
    try:
        tagger = AutoTagger()
        suggestions = tagger.suggest_tags(request.content, request.limit)
        
        return {
            "status": "success",
            "content": request.content,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Tag suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag suggestions: {str(e)}")

@app.post("/api/save")
async def save_message(message: MessageRequest):
    """Save a journal entry with enhanced tag support"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute(
            "INSERT INTO messages (content, user_id, timestamp) VALUES (?, ?, ?)",
            (message.content, message.user_id, datetime.utcnow().isoformat())
        )
        
        message_id = cursor.lastrowid
        
        # Prepare tags data
        applied_tags = []
        
        # Add manual tags
        for tag_name in message.manual_tags:
            applied_tags.append({
                "name": tag_name,
                "confidence": 1.0,
                "source": "manual"
            })
        
        # Add auto tags if enabled
        if message.auto_tag:
            tagger = AutoTagger()
            auto_tags = tagger.apply_auto_tags(message.content, message.manual_tags)
            applied_tags.extend(auto_tags)
        
        # Apply all tags to the entry
        apply_tags_to_entry(conn, message_id, applied_tags)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"💾 Saved message {message_id} with {len(applied_tags)} tags for user {message.user_id}")
        
        return {
            "status": "success",
            "message_id": message_id,
            "timestamp": datetime.utcnow().isoformat(),
            "applied_tags": applied_tags,
            "tag_count": len(applied_tags)
        }
        
    except Exception as e:
        logger.error(f"Save message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

@app.get("/api/messages/{user_id}")
async def get_messages(user_id: str, limit: int = 100, offset: int = 0, tags: Optional[str] = None):
    """Get messages for a user with optional tag filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if tags:
            # Filter by tags
            tag_list = [tag.strip() for tag in tags.split(",")]
            placeholders = ",".join("?" * len(tag_list))
            
            cursor.execute(f"""
                SELECT DISTINCT m.id, m.content, m.user_id, m.timestamp
                FROM messages m
                JOIN entry_tags et ON m.id = et.message_id
                JOIN tags t ON et.tag_id = t.id
                WHERE m.user_id = ? AND t.name IN ({placeholders})
                ORDER BY m.timestamp DESC
                LIMIT ? OFFSET ?
            """, [user_id] + tag_list + [limit, offset])
        else:
            # Get all messages
            cursor.execute(
                """SELECT id, content, user_id, timestamp 
                   FROM messages 
                   WHERE user_id = ? 
                   ORDER BY timestamp DESC 
                   LIMIT ? OFFSET ?""",
                (user_id, limit, offset)
            )
        
        rows = cursor.fetchall()
        
        # Get tags for each message
        messages = []
        for row in rows:
            message = dict(row)
            
            # Get tags for this message
            cursor.execute("""
                SELECT t.name, t.color, t.category, et.confidence, et.is_auto_tagged
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                WHERE et.message_id = ?
            """, (message["id"],))
            
            tags_data = cursor.fetchall()
            message["tags"] = [dict(tag) for tag in tags_data]
            messages.append(message)
        
        cursor.close()
        conn.close()
        
        logger.info(f"📖 Retrieved {len(messages)} messages for user {user_id}")
        
        return {
            "status": "success",
            "messages": messages,
            "count": len(messages),
            "filtered_by_tags": tags.split(",") if tags else None
        }
        
    except Exception as e:
        logger.error(f"Get messages failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.get("/api/messages/{user_id}/tags/{tag_names}")
async def get_messages_by_tags(user_id: str, tag_names: str, limit: int = 100, offset: int = 0):
    """Get messages filtered by specific tags"""
    return await get_messages(user_id, limit, offset, tag_names)

@app.get("/")
async def root():
    """Root endpoint with enhanced API information"""
    return {
        "service": "Mirror Scribe Backend with Intelligent Tags & Persistent Storage",
        "version": "2.1.0",
        "database": "SQLite with Persistent Volume",
        "storage": get_storage_info(),
        "features": [
            "intelligent_tagging",
            "auto_tag_suggestions", 
            "manual_tags",
            "tag_filtering",
            "category_organization",
            "keyword_matching",
            "persistent_storage",
            "railway_optimized"
        ],
        "endpoints": [
            "/health",
            "/storage-info",
            "/init", 
            "/api/save",
            "/api/messages/{user_id}",
            "/api/messages/{user_id}/tags/{tag_names}",
            "/api/tags",
            "/api/tags/categories",
            "/api/tags/suggestions",
            "/stats"
        ],
        "persistence": {
            "description": "All data persists across Railway deployments",
            "volume_mount": "/app/data" if is_railway_environment() else "N/A",
            "database_path": get_database_path()
        },
        "status": "ready"
    }

@app.get("/stats")
async def get_stats():
    """Get enhanced database statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get message count
        cursor.execute("SELECT COUNT(*) as total FROM messages")
        total_messages = cursor.fetchone()["total"]
        
        # Get unique user count
        cursor.execute("SELECT COUNT(DISTINCT user_id) as users FROM messages")
        unique_users = cursor.fetchone()["users"]
        
        # Get tag statistics
        cursor.execute("SELECT COUNT(*) as total FROM tags")
        total_tags = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(*) as predefined FROM tags WHERE is_predefined = TRUE")
        predefined_tags = cursor.fetchone()["predefined"]
        
        cursor.execute("SELECT COUNT(*) as custom FROM tags WHERE is_predefined = FALSE")
        custom_tags = cursor.fetchone()["custom"]
        
        # Get tagging statistics
        cursor.execute("SELECT COUNT(*) as total FROM entry_tags")
        total_tag_applications = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(*) as auto FROM entry_tags WHERE is_auto_tagged = TRUE")
        auto_tagged = cursor.fetchone()["auto"]
        
        cursor.execute("SELECT COUNT(*) as manual FROM entry_tags WHERE is_auto_tagged = FALSE")
        manual_tagged = cursor.fetchone()["manual"]
        
        cursor.close()
        conn.close()
        
        return {
            "messages": {
                "total": total_messages,
                "unique_users": unique_users
            },
            "tags": {
                "total": total_tags,
                "predefined": predefined_tags,
                "custom": custom_tags
            },
            "tagging": {
                "total_applications": total_tag_applications,
                "auto_tagged": auto_tagged,
                "manual_tagged": manual_tagged
            },
            "database_file": DATABASE_FILE,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting enhanced tagging server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)