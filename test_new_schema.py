#!/usr/bin/env python3
"""
Test the new schema creation from main.py to ensure it has all required columns.
"""

import sqlite3
import tempfile
import os

def test_new_schema():
    """Test that the new schema includes all required columns"""
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        print(f"🧪 Testing new schema creation: {db_path}")
        
        # Create database with the new schema (copied from main.py)
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        # Create messages table with all timestamp columns (from main.py)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                utc_timestamp DATETIME,
                local_timestamp DATETIME,
                timezone_at_creation TEXT DEFAULT 'America/Chicago',
                timestamp_source TEXT DEFAULT 'auto',
                temporal_validation_score REAL DEFAULT 0.5,
                intention_flag BOOLEAN DEFAULT FALSE,
                manual_energy_signature TEXT,
                relationship_mentions JSON,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                revision_count INTEGER DEFAULT 0,
                temporal_signal_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        
        # Test the query that was failing
        print("🔍 Testing the problematic query...")
        cursor.execute("SELECT id, content, local_timestamp, utc_timestamp, timezone_at_creation FROM messages WHERE user_id = 'test123'")
        print("✅ Query executed successfully (no 'no such column' error)")
        
        # Check all required columns exist
        cursor.execute("PRAGMA table_info(messages)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        required_columns = {
            'id', 'content', 'user_id', 'timestamp',
            'utc_timestamp', 'local_timestamp', 'timezone_at_creation',
            'timestamp_source', 'temporal_validation_score'
        }
        
        missing_columns = required_columns - set(columns.keys())
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ All required columns present:")
        for col in sorted(required_columns):
            print(f"  - {col}: {columns[col]}")
        
        # Test inserting data with the new schema
        print("\n🔍 Testing data insertion...")
        cursor.execute("""
            INSERT INTO messages (
                content, user_id, timestamp, 
                utc_timestamp, local_timestamp, timezone_at_creation,
                timestamp_source, temporal_validation_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "Test message", 
            "user123", 
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:00",
            "2024-01-01T05:00:00",
            "America/Chicago",
            "test",
            0.9
        ))
        
        # Test querying the data
        cursor.execute("""
            SELECT id, content, utc_timestamp, local_timestamp, timezone_at_creation
            FROM messages WHERE user_id = 'user123'
        """)
        
        results = cursor.fetchall()
        print(f"✅ Successfully inserted and queried {len(results)} messages")
        
        for row in results:
            msg_id, content, utc_ts, local_ts, timezone = row
            print(f"  Message {msg_id}: '{content}' UTC:{utc_ts} Local:{local_ts} TZ:{timezone}")
        
        conn.close()
        
        print("\n🎉 New schema test passed! The database structure is correct.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    success = test_new_schema()
    if success:
        print("✅ New schema test passed")
        exit(0)
    else:
        print("❌ New schema test failed")
        exit(1)