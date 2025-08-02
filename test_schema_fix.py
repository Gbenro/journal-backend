#!/usr/bin/env python3
"""
Test the database schema fix to ensure it resolves the "no such column" error.
"""

import sqlite3
import tempfile
import os
import logging
from schema_migration import check_database_schema, fix_database_schema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_schema_fix():
    """Test the complete schema fix process"""
    
    # Create a temporary database with the old schema (mimicking the production issue)
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        logger.info(f"🧪 Testing schema fix with temporary database: {db_path}")
        
        # Create the problematic old schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the old messages table (without timestamp columns)
        cursor.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert some test data
        cursor.execute("""
            INSERT INTO messages (content, user_id, timestamp) 
            VALUES ('Test message 1', 'user123', '2024-01-01 10:00:00')
        """)
        
        cursor.execute("""
            INSERT INTO messages (content, user_id, timestamp) 
            VALUES ('Test message 2', 'user123', '2024-01-02 15:30:00')
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Created test database with old schema and sample data")
        
        # Test that the query that was failing will indeed fail
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, local_timestamp FROM messages WHERE user_id = 'user123'")
            conn.close()
            logger.error("❌ Expected query to fail but it succeeded - test setup is wrong")
            return False
            
        except sqlite3.OperationalError as e:
            if "no such column: local_timestamp" in str(e):
                logger.info("✅ Confirmed: Query fails with 'no such column: local_timestamp' as expected")
            else:
                logger.error(f"❌ Unexpected error: {e}")
                return False
        
        # Check schema before fix
        schema_info = check_database_schema(db_path)
        logger.info(f"📊 Schema before fix: {schema_info}")
        
        if not schema_info['needs_migration']:
            logger.error("❌ Schema check incorrectly says no migration needed")
            return False
        
        # Apply the fix
        logger.info("🔧 Applying schema fix...")
        result = fix_database_schema(db_path)
        
        if not result['success']:
            logger.error(f"❌ Schema fix failed: {result['error']}")
            return False
        
        logger.info(f"✅ Schema fix succeeded: {result}")
        
        # Test that the previously failing query now works
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, local_timestamp, utc_timestamp, timezone_at_creation FROM messages WHERE user_id = 'user123'")
            results = cursor.fetchall()
            conn.close()
            
            logger.info(f"✅ Query now succeeds! Found {len(results)} messages")
            
            # Verify that timestamp fields are populated
            for row in results:
                msg_id, content, local_ts, utc_ts, timezone = row
                logger.info(f"  Message {msg_id}: local_ts={local_ts}, utc_ts={utc_ts}, tz={timezone}")
                
                if local_ts is None or utc_ts is None:
                    logger.error(f"❌ Message {msg_id} still has NULL timestamps")
                    return False
            
            logger.info("✅ All messages have proper timestamps")
            
        except sqlite3.OperationalError as e:
            logger.error(f"❌ Query still fails after fix: {e}")
            return False
        
        # Check schema after fix
        schema_info_after = check_database_schema(db_path)
        logger.info(f"📊 Schema after fix: {schema_info_after}")
        
        if schema_info_after['needs_migration']:
            logger.error("❌ Schema still needs migration after fix")
            return False
        
        logger.info("🎉 All tests passed! Schema fix is working correctly.")
        return True
        
    finally:
        # Clean up temporary file
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    success = test_schema_fix()
    if success:
        print("✅ Test passed: Schema fix resolves the database issue")
        exit(0)
    else:
        print("❌ Test failed: Schema fix does not work properly")
        exit(1)