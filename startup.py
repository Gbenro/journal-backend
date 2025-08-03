#!/usr/bin/env python3
"""
Startup Script for Journal Backend
Ensures database schema is correct before starting the FastAPI server.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from schema_migration import check_database_schema, fix_database_schema
from timestamp_synchronization import create_timestamp_tables
from temporal_awareness import create_temporal_tables

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_database_ready():
    """Ensure the database is ready before starting the server"""
    
    # Determine database path based on environment
    if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"):
        db_path = "/app/data/journal.db"
        # Ensure data directory exists
        os.makedirs("/app/data", exist_ok=True)
    else:
        db_path = os.getenv('DATABASE_PATH', 'journal.db')
    
    # Check if database exists
    if not os.path.exists(db_path):
        logger.info(f"📁 Database does not exist at {db_path}, will be created on first request")
        return True
    
    logger.info(f"🔍 Checking database schema at: {db_path}")
    
    # Check current schema
    schema_info = check_database_schema(db_path)
    
    if not schema_info.get("exists", False):
        logger.error(f"❌ Database check failed: {schema_info.get('error', 'Unknown error')}")
        return False
    
    # First ensure timestamp and temporal tables exist
    try:
        logger.info("🕐 Ensuring timestamp tables and columns exist...")
        create_timestamp_tables(db_path)
        logger.info("✅ Timestamp tables verified/created")
    except Exception as e:
        logger.warning(f"⚠️ Timestamp table creation warning: {e}")
        
    try:
        logger.info("🧠 Ensuring temporal awareness tables exist...")
        create_temporal_tables(db_path)
        logger.info("✅ Temporal tables verified/created")
    except Exception as e:
        logger.warning(f"⚠️ Temporal table creation warning: {e}")
    
    if not schema_info.get("needs_migration", False):
        logger.info("✅ Database schema is up to date")
        return True
    
    logger.info("🔧 Database needs migration, fixing schema...")
    logger.info(f"  - Missing columns: {schema_info.get('missing_columns', [])}")
    logger.info(f"  - Messages needing timestamp migration: {schema_info.get('total_messages', 0) - schema_info.get('properly_timestamped', 0)}")
    
    # Perform migration
    result = fix_database_schema(db_path)
    
    if result['success']:
        logger.info("🎉 Database migration completed successfully!")
        logger.info(f"  - Columns added: {result['columns_added']}")
        logger.info(f"  - Entries migrated: {result['entries_migrated']}")
        return True
    else:
        logger.error(f"💥 Migration failed: {result['error']}")
        return False

def start_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    
    # Get server configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    # Railway uses PORT environment variable
    port = int(os.getenv('PORT', '8080'))
    
    # Start the server using uvicorn
    cmd = [
        'uvicorn', 
        'main:app',
        '--host', host,
        '--port', str(port)
    ]
    
    # Add reload flag only in development
    if os.getenv('ENVIRONMENT') == 'development':
        cmd.append('--reload')
    
    logger.info(f"📡 Server command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
        sys.exit(0)

def main():
    """Main startup function"""
    logger.info("🎯 Journal Backend Startup")
    
    # Ensure database is ready
    if not ensure_database_ready():
        logger.error("❌ Database preparation failed, cannot start server")
        sys.exit(1)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()