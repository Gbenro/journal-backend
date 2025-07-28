#!/usr/bin/env python3
"""
Comprehensive Test Suite for Timestamp Synchronization System
Tests all components including TimezoneManager, TemporalValidator, and TimestampSynchronizer.
"""

import os
import sqlite3
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from timestamp_synchronization import (
    TimezoneManager, TemporalValidator, TimestampSynchronizer, TimestampSource,
    ValidationSeverity, create_timestamp_tables, migrate_existing_timestamps
)
from timestamp_migration import TimestampMigrationManager

class TestTimestampSynchronization:
    """Test suite for timestamp synchronization system"""
    
    def setup_method(self):
        """Set up test database for each test"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize database with basic schema
        self._create_basic_schema()
        
        # Initialize managers
        self.timezone_manager = TimezoneManager()
        self.validator = TemporalValidator(self.db_path)
        self.synchronizer = TimestampSynchronizer(self.db_path)
    
    def teardown_method(self):
        """Clean up after each test"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def _create_basic_schema(self):
        """Create basic database schema for testing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create basic messages table
        cursor.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create temporal_state table for NLP integration
        cursor.execute("""
            CREATE TABLE temporal_state (
                user_id TEXT PRIMARY KEY,
                timezone TEXT DEFAULT 'America/Chicago',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create entry_tags table
        cursor.execute("""
            CREATE TABLE entry_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                tag TEXT,
                FOREIGN KEY (entry_id) REFERENCES messages(id)
            )
        """)
        
        # Create tags table
        cursor.execute("""
            CREATE TABLE tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def test_timezone_manager_validation(self):
        """Test timezone validation functionality"""
        print("🧪 Testing TimezoneManager validation...")
        
        # Valid timezones
        assert self.timezone_manager.validate_timezone("America/Chicago")
        assert self.timezone_manager.validate_timezone("Europe/London")
        assert self.timezone_manager.validate_timezone("UTC")
        
        # Invalid timezones
        assert not self.timezone_manager.validate_timezone("Invalid/Timezone")
        assert not self.timezone_manager.validate_timezone("")
        assert not self.timezone_manager.validate_timezone("America/NotReal")
        
        print("✅ Timezone validation tests passed")
    
    def test_timezone_conversion(self):
        """Test timezone conversion functionality"""
        print("🧪 Testing timezone conversion...")
        
        # Test UTC to Chicago conversion
        utc_time = datetime(2024, 1, 15, 18, 30, 0)  # 6:30 PM UTC
        chicago_time = self.timezone_manager.convert_to_timezone(
            utc_time.replace(tzinfo=timezone.utc), "America/Chicago"
        )
        
        # Should be 12:30 PM CST (UTC-6 in January)
        assert chicago_time.hour == 12
        assert chicago_time.minute == 30
        
        # Test get_local_and_utc
        local_dt, utc_dt = self.timezone_manager.get_local_and_utc(
            "2024-01-15T12:30:00", "America/Chicago"
        )
        
        assert local_dt.hour == 12
        assert utc_dt.hour == 18  # 6 hours ahead in UTC
        
        print("✅ Timezone conversion tests passed")
    
    def test_timestamp_tables_creation(self):
        """Test database schema creation"""
        print("🧪 Testing timestamp tables creation...")
        
        # Create timestamp tables
        create_timestamp_tables(self.db_path)
        
        # Verify tables and columns exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check users table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert cursor.fetchone() is not None
        
        # Check timestamp_audit_log table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='timestamp_audit_log'")
        assert cursor.fetchone() is not None
        
        # Check new columns in messages table
        cursor.execute("PRAGMA table_info(messages)")
        columns = [row[1] for row in cursor.fetchall()]
        
        assert "utc_timestamp" in columns
        assert "local_timestamp" in columns
        assert "timezone_at_creation" in columns
        assert "timestamp_source" in columns
        assert "temporal_validation_score" in columns
        
        conn.close()
        print("✅ Timestamp tables creation tests passed")
    
    def test_user_timezone_management(self):
        """Test user timezone preference management"""
        print("🧪 Testing user timezone management...")
        
        # Create timestamp tables first
        create_timestamp_tables(self.db_path)
        
        user_id = "test_user_123"
        timezone = "America/Los_Angeles"
        
        # Set user timezone
        success = self.timezone_manager.set_user_timezone(self.db_path, user_id, timezone)
        assert success
        
        # Get user timezone
        retrieved_timezone = self.timezone_manager.get_user_timezone(self.db_path, user_id)
        assert retrieved_timezone == timezone
        
        # Test invalid timezone (should fail)
        success = self.timezone_manager.set_user_timezone(self.db_path, user_id, "Invalid/Zone")
        assert not success
        
        print("✅ User timezone management tests passed")
    
    def test_timestamp_info_creation(self):
        """Test TimestampInfo creation"""
        print("🧪 Testing timestamp info creation...")
        
        # Create timestamp tables
        create_timestamp_tables(self.db_path)
        
        # Test with client timestamp
        user_id = "test_user_456"
        content = "Started my day with coffee this morning"
        client_timestamp = "2024-01-15T08:30:00"
        client_timezone = "America/New_York"
        
        # Set user timezone first
        self.timezone_manager.set_user_timezone(self.db_path, user_id, client_timezone)
        
        timestamp_info = self.synchronizer.create_timestamp_info(
            content=content,
            user_id=user_id,
            client_timestamp=client_timestamp,
            client_timezone=client_timezone,
            timestamp_source=TimestampSource.CLIENT_PROVIDED
        )
        
        assert timestamp_info.timezone_name == client_timezone
        assert timestamp_info.timestamp_source == TimestampSource.CLIENT_PROVIDED
        assert timestamp_info.local_timestamp.hour == 8
        assert timestamp_info.local_timestamp.minute == 30
        assert timestamp_info.validation_score >= 0.0
        assert len(timestamp_info.validation_notes) > 0
        
        print("✅ Timestamp info creation tests passed")
    
    def test_migration_functionality(self):
        """Test database migration functionality"""
        print("🧪 Testing migration functionality...")
        
        # Insert some test data with old schema
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        test_entries = [
            ("user1", "Morning journal entry", "2024-01-15T08:00:00"),
            ("user2", "Evening reflection", "2024-01-15T22:30:00"),
            ("user1", "Afternoon thoughts", "2024-01-15T15:15:00")
        ]
        
        for user_id, content, timestamp in test_entries:
            cursor.execute(
                "INSERT INTO messages (user_id, content, timestamp) VALUES (?, ?, ?)",
                (user_id, content, timestamp)
            )
        
        conn.commit()
        conn.close()
        
        # Run migration
        create_timestamp_tables(self.db_path)
        migration_result = migrate_existing_timestamps(self.db_path, "America/Chicago")
        
        assert migration_result["success"]
        assert migration_result["migrated_count"] == 3
        assert len(migration_result["errors"]) == 0
        
        # Verify migration results
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT utc_timestamp, local_timestamp, timezone_at_creation FROM messages")
        results = cursor.fetchall()
        
        for utc_ts, local_ts, timezone in results:
            assert utc_ts is not None
            assert local_ts is not None
            assert timezone == "America/Chicago"
        
        conn.close()
        print("✅ Migration functionality tests passed")
    
    def test_timestamp_override(self):
        """Test manual timestamp override functionality"""
        print("🧪 Testing timestamp override...")
        
        # Create timestamp tables and insert test entry
        create_timestamp_tables(self.db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (user_id, content, utc_timestamp, local_timestamp, timezone_at_creation)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_user", "Test entry", "2024-01-15T12:00:00", "2024-01-15T06:00:00", "America/Chicago"))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Override timestamp
        new_timestamp = "2024-01-15T14:30:00"
        timezone_name = "America/Chicago"
        reason = "Corrected timestamp per user request"
        
        success = self.synchronizer.override_timestamp(
            entry_id, new_timestamp, timezone_name, "test_user", reason
        )
        
        assert success
        
        # Verify the override
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT utc_timestamp, local_timestamp, timestamp_source FROM messages WHERE id = ?", (entry_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert "14:30:00" in result[1]  # local timestamp should be updated
        assert result[2] == "manual_override"
        
        # Check audit log
        cursor.execute("SELECT reason FROM timestamp_audit_log WHERE entry_id = ?", (entry_id,))
        audit_result = cursor.fetchone()
        assert audit_result[0] == reason
        
        conn.close()
        print("✅ Timestamp override tests passed")
    
    def test_full_migration_manager(self):
        """Test the complete migration manager functionality"""
        print("🧪 Testing full migration manager...")
        
        # Insert test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        test_entries = [
            ("user1", "Good morning! Starting the day", "2024-01-15T07:30:00"),
            ("user2", "Lunch break thoughts", "2024-01-15T12:15:00"),
            ("user1", "End of day reflection", "2024-01-15T21:45:00")
        ]
        
        for user_id, content, timestamp in test_entries:
            cursor.execute(
                "INSERT INTO messages (user_id, content, timestamp) VALUES (?, ?, ?)",
                (user_id, content, timestamp)
            )
        
        conn.commit()
        conn.close()
        
        # Create migration manager
        migrator = TimestampMigrationManager(self.db_path)
        
        # Verify database integrity
        integrity_check = migrator.verify_database_integrity()
        assert integrity_check["integrity_ok"]
        assert integrity_check["statistics"]["total_messages"] == 3
        
        # Analyze existing timestamps
        analysis = migrator.analyze_existing_timestamps()
        assert analysis["total_entries"] == 3
        assert "timestamp_formats" in analysis
        assert "user_timezones" in analysis
        
        # Perform migration
        result = migrator.perform_migration("America/New_York")
        
        assert result["success"]
        assert result["migration_result"]["migrated_count"] == 3
        assert result["validation"]["completion_rate"] >= 0.95
        
        # Verify backup was created
        assert os.path.exists(result["backup_created"])
        
        # Clean up backup
        migrator.cleanup_backup()
        
        print("✅ Full migration manager tests passed")
    
    def test_temporal_validation(self):
        """Test temporal validation functionality"""
        print("🧪 Testing temporal validation...")
        
        # Create timestamp tables
        create_timestamp_tables(self.db_path)
        
        user_id = "validation_test_user"
        timezone_name = "America/Chicago"
        
        # Set user timezone
        self.timezone_manager.set_user_timezone(self.db_path, user_id, timezone_name)
        
        # Test morning entry validation
        morning_content = "Started my day with meditation this morning"
        morning_timestamp = datetime(2024, 1, 15, 7, 30, 0)  # 7:30 AM
        
        validation_result = self.validator.validate_entry_timestamp(
            morning_content, morning_timestamp, timezone_name, user_id
        )
        
        assert validation_result.severity in [ValidationSeverity.OK, ValidationSeverity.WARNING]
        assert 0.0 <= validation_result.confidence <= 1.0
        assert len(validation_result.message) > 0
        
        # Test suspicious late night entry
        night_content = "Just a random thought"
        night_timestamp = datetime(2024, 1, 15, 3, 0, 0)  # 3:00 AM
        
        validation_result = self.validator.validate_entry_timestamp(
            night_content, night_timestamp, timezone_name, user_id
        )
        
        # Should detect this as potentially suspicious
        assert validation_result.severity in [ValidationSeverity.WARNING, ValidationSeverity.SUSPICIOUS]
        
        print("✅ Temporal validation tests passed")

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 Starting Comprehensive Timestamp Synchronization Tests")
    print("=" * 60)
    
    test_suite = TestTimestampSynchronization()
    
    # List of all test methods
    test_methods = [
        'test_timezone_manager_validation',
        'test_timezone_conversion', 
        'test_timestamp_tables_creation',
        'test_user_timezone_management',
        'test_timestamp_info_creation',
        'test_migration_functionality',
        'test_timestamp_override',
        'test_full_migration_manager',
        'test_temporal_validation'
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_method_name in test_methods:
        try:
            print(f"\n📋 Running {test_method_name}...")
            test_suite.setup_method()
            
            test_method = getattr(test_suite, test_method_name)
            test_method()
            
            passed_tests += 1
            print(f"✅ {test_method_name} PASSED")
            
        except Exception as e:
            failed_tests += 1
            print(f"❌ {test_method_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                test_suite.teardown_method()
            except:
                pass
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results Summary:")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ❌ Failed: {failed_tests}")
    print(f"   📊 Success Rate: {passed_tests/(passed_tests + failed_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Timestamp synchronization system is ready for production.")
        return True
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Please review and fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)