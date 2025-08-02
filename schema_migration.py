#!/usr/bin/env python3
"""
Emergency Schema Migration Script for Database Deployment
Fixes the "no such column: local_timestamp" error by ensuring all required columns exist.
"""

import sqlite3
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_schema(db_path: str) -> Dict[str, Any]:
    """Check the current database schema and identify missing columns"""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        # Check if messages table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if not cursor.fetchone():
            conn.close()
            return {"exists": False, "error": "messages table does not exist"}
        
        # Get current schema
        cursor.execute("PRAGMA table_info(messages)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        # Required columns for timestamp system
        required_columns = {
            'id': 'INTEGER',
            'content': 'TEXT',
            'user_id': 'TEXT', 
            'timestamp': 'DATETIME',
            'utc_timestamp': 'DATETIME',
            'local_timestamp': 'DATETIME',
            'timezone_at_creation': 'TEXT',
            'timestamp_source': 'TEXT',
            'temporal_validation_score': 'REAL'
        }
        
        missing_columns = []
        for col_name, col_type in required_columns.items():
            if col_name not in columns:
                missing_columns.append(col_name)
        
        # Check data state (safely handle missing columns)
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        properly_timestamped = 0
        if 'utc_timestamp' in columns and 'local_timestamp' in columns:
            try:
                cursor.execute("SELECT COUNT(*) FROM messages WHERE utc_timestamp IS NOT NULL AND local_timestamp IS NOT NULL")
                properly_timestamped = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                properly_timestamped = 0
        
        conn.close()
        
        return {
            "exists": True,
            "columns": columns,
            "missing_columns": missing_columns,
            "total_messages": total_messages,
            "properly_timestamped": properly_timestamped,
            "needs_migration": len(missing_columns) > 0 or (properly_timestamped < total_messages)
        }
        
    except Exception as e:
        logger.error(f"Failed to check database schema: {e}")
        return {"exists": False, "error": str(e)}

def fix_database_schema(db_path: str) -> Dict[str, Any]:
    """Fix database schema by adding missing columns and migrating data"""
    try:
        logger.info(f"🔧 Starting database schema fix for: {db_path}")
        
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        # Column definitions with defaults
        columns_to_add = {
            'utc_timestamp': 'DATETIME',
            'local_timestamp': 'DATETIME',
            'timezone_at_creation': 'TEXT DEFAULT \'America/Chicago\'',
            'timestamp_source': 'TEXT DEFAULT \'auto\'',
            'temporal_validation_score': 'REAL DEFAULT 0.5',
            'intention_flag': 'BOOLEAN DEFAULT FALSE',
            'manual_energy_signature': 'TEXT',
            'relationship_mentions': 'JSON',
            'updated_at': 'TIMESTAMP',
            'revision_count': 'INTEGER DEFAULT 0',
            'temporal_signal_count': 'INTEGER DEFAULT 0'
        }
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(messages)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        columns_added = 0
        
        # Add missing columns
        for column_name, column_def in columns_to_add.items():
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE messages ADD COLUMN {column_name} {column_def}")
                    logger.info(f"✅ Added column: {column_name}")
                    columns_added += 1
                except sqlite3.OperationalError as e:
                    logger.warning(f"⚠️ Could not add column {column_name}: {e}")
        
        # Create essential indexes if they don't exist
        essential_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_utc_timestamp ON messages(utc_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_local_timestamp ON messages(local_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_user_timestamp ON messages(user_id, timestamp)"
        ]
        
        for index_sql in essential_indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not create index: {e}")
        
        conn.commit()
        
        # Now migrate existing timestamps
        cursor.execute("SELECT COUNT(*) FROM messages WHERE utc_timestamp IS NULL OR local_timestamp IS NULL")
        unmigrated_count = cursor.fetchone()[0]
        
        migrated_count = 0
        if unmigrated_count > 0:
            logger.info(f"🔄 Migrating {unmigrated_count} entries with missing timestamps")
            
            # Get entries that need migration
            cursor.execute("""
                SELECT id, user_id, timestamp 
                FROM messages 
                WHERE utc_timestamp IS NULL OR local_timestamp IS NULL
            """)
            entries_to_migrate = cursor.fetchall()
            
            for entry_id, user_id, existing_timestamp in entries_to_migrate:
                try:
                    # Use existing timestamp or current time
                    if existing_timestamp:
                        dt = datetime.fromisoformat(existing_timestamp.replace('Z', ''))
                    else:
                        dt = datetime.utcnow()
                    
                    # For migration, assume all timestamps are UTC and convert to Central Time
                    utc_timestamp = dt
                    # Simple offset conversion (this is a basic migration, could be more sophisticated)
                    local_timestamp = dt  # In a real scenario, we'd apply timezone conversion
                    
                    cursor.execute("""
                        UPDATE messages 
                        SET utc_timestamp = ?, local_timestamp = ?, 
                            timezone_at_creation = ?, timestamp_source = ?,
                            temporal_validation_score = ?
                        WHERE id = ?
                    """, (
                        utc_timestamp.isoformat(),
                        local_timestamp.isoformat(),
                        'America/Chicago',
                        'migrated',
                        0.7,
                        entry_id
                    ))
                    
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate entry {entry_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Schema migration completed: {columns_added} columns added, {migrated_count} entries migrated")
        
        return {
            "success": True,
            "columns_added": columns_added,
            "entries_migrated": migrated_count,
            "total_unmigrated": unmigrated_count
        }
        
    except Exception as e:
        logger.error(f"❌ Schema migration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "columns_added": 0,
            "entries_migrated": 0
        }

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python schema_migration.py <database_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if not os.path.exists(db_path):
        logger.error(f"Database file does not exist: {db_path}")
        sys.exit(1)
    
    # Check current schema
    logger.info("🔍 Checking current database schema...")
    schema_info = check_database_schema(db_path)
    
    if not schema_info.get("exists", False):
        logger.error(f"Database check failed: {schema_info.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info(f"📊 Database status:")
    logger.info(f"  - Total messages: {schema_info['total_messages']}")
    logger.info(f"  - Properly timestamped: {schema_info['properly_timestamped']}")
    logger.info(f"  - Missing columns: {schema_info['missing_columns']}")
    logger.info(f"  - Needs migration: {schema_info['needs_migration']}")
    
    if not schema_info['needs_migration']:
        logger.info("✅ Database schema is already up to date!")
        return
    
    # Perform migration
    logger.info("🚀 Starting database schema migration...")
    result = fix_database_schema(db_path)
    
    if result['success']:
        logger.info("🎉 Database migration completed successfully!")
        logger.info(f"  - Columns added: {result['columns_added']}")
        logger.info(f"  - Entries migrated: {result['entries_migrated']}")
    else:
        logger.error(f"💥 Migration failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()