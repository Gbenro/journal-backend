#!/usr/bin/env python3
"""
Comprehensive Database Migration Script for Timestamp Synchronization
Safely migrates existing data to the new dual-timestamp system with full validation.
"""

import os
import sqlite3
import shutil
from datetime import datetime
import logging
import json
from typing import Dict, Any, List, Tuple
from timestamp_synchronization import (
    TimezoneManager, TimestampSynchronizer, create_timestamp_tables, 
    migrate_existing_timestamps, TimestampSource
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimestampMigrationManager:
    """Manages comprehensive database migration for timestamp synchronization"""
    
    def __init__(self, db_path: str, backup_path: str = None):
        self.db_path = db_path
        self.backup_path = backup_path or f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.timezone_manager = TimezoneManager()
        
    def create_backup(self) -> bool:
        """Create a backup of the database before migration"""
        try:
            shutil.copy2(self.db_path, self.backup_path)
            logger.info(f"✅ Database backup created: {self.backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False
    
    def verify_database_integrity(self) -> Dict[str, Any]:
        """Verify database integrity before migration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if critical tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['messages', 'tags', 'entry_tags']
            missing_tables = [table for table in required_tables if table not in tables]
            
            # Get statistics
            stats = {}
            if 'messages' in tables:
                cursor.execute("SELECT COUNT(*) FROM messages")
                stats['total_messages'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM messages")
                stats['unique_users'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE timestamp IS NOT NULL")
                result = cursor.fetchone()
                stats['date_range'] = result if result[0] and result[1] else None
            
            conn.close()
            
            return {
                "integrity_ok": len(missing_tables) == 0,
                "missing_tables": missing_tables,
                "existing_tables": tables,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return {"integrity_ok": False, "error": str(e)}
    
    def analyze_existing_timestamps(self) -> Dict[str, Any]:
        """Analyze existing timestamp data to plan migration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze timestamp formats and patterns
            cursor.execute("""
                SELECT timestamp, user_id
                FROM messages 
                WHERE timestamp IS NOT NULL
                ORDER BY timestamp
            """)
            
            entries = cursor.fetchall()
            analysis = {
                "total_entries": len(entries),
                "timestamp_formats": {},
                "user_timezones": {},
                "temporal_patterns": {},
                "potential_issues": []
            }
            
            for timestamp_str, user_id in entries:
                try:
                    # Try to parse timestamp
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # Analyze format
                    if 'T' in timestamp_str:
                        fmt = "ISO_FORMAT"
                    elif ' ' in timestamp_str:
                        fmt = "DATETIME_FORMAT"
                    else:
                        fmt = "UNKNOWN_FORMAT"
                    
                    analysis["timestamp_formats"][fmt] = analysis["timestamp_formats"].get(fmt, 0) + 1
                    
                    # Analyze hour patterns for timezone detection
                    hour = dt.hour
                    if user_id not in analysis["temporal_patterns"]:
                        analysis["temporal_patterns"][user_id] = {"hours": [], "weekdays": []}
                    
                    analysis["temporal_patterns"][user_id]["hours"].append(hour)
                    analysis["temporal_patterns"][user_id]["weekdays"].append(dt.weekday())
                    
                except Exception as e:
                    analysis["potential_issues"].append(f"Invalid timestamp: {timestamp_str} - {str(e)}")
            
            # Suggest timezones based on patterns
            for user_id, patterns in analysis["temporal_patterns"].items():
                hours = patterns["hours"]
                if hours:
                    avg_hour = sum(hours) / len(hours)
                    
                    # Simple timezone suggestion based on average journaling hour
                    if 6 <= avg_hour <= 22:  # Normal waking hours
                        suggested_tz = "America/Chicago"  # Default
                    elif 22 <= avg_hour or avg_hour <= 6:  # Night owl or early bird
                        suggested_tz = "America/Los_Angeles"  # West coast pattern
                    else:
                        suggested_tz = "America/New_York"  # East coast pattern
                    
                    analysis["user_timezones"][user_id] = {
                        "suggested": suggested_tz,
                        "avg_hour": avg_hour,
                        "entries": len(hours)
                    }
            
            conn.close()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Timestamp analysis failed: {e}")
            return {"error": str(e)}
    
    def perform_migration(self, default_timezone: str = "America/Chicago", 
                         user_timezone_overrides: Dict[str, str] = None) -> Dict[str, Any]:
        """Perform the complete migration with validation"""
        
        logger.info("🚀 Starting comprehensive timestamp migration...")
        
        # Step 1: Create backup
        if not self.create_backup():
            return {"success": False, "error": "Failed to create backup"}
        
        # Step 2: Verify database integrity
        integrity_check = self.verify_database_integrity()
        if not integrity_check["integrity_ok"]:
            return {"success": False, "error": "Database integrity check failed", "details": integrity_check}
        
        # Step 3: Analyze existing data
        analysis = self.analyze_existing_timestamps()
        if "error" in analysis:
            return {"success": False, "error": "Timestamp analysis failed", "details": analysis}
        
        logger.info(f"📊 Analysis complete: {analysis['total_entries']} entries to migrate")
        
        try:
            # Step 4: Create new tables and columns
            create_timestamp_tables(self.db_path)
            logger.info("✅ New timestamp tables and columns created")
            
            # Step 5: Migrate user timezone preferences
            self._migrate_user_timezones(user_timezone_overrides or {}, default_timezone)
            
            # Step 6: Migrate timestamp data
            migration_result = migrate_existing_timestamps(self.db_path, default_timezone)
            
            if not migration_result["success"]:
                return {"success": False, "error": "Timestamp migration failed", "details": migration_result}
            
            logger.info(f"✅ Migrated {migration_result['migrated_count']} entries")
            
            # Step 7: Validate migration
            validation_result = self._validate_migration()
            
            # Step 8: Create indexes for performance
            self._optimize_indexes()
            
            logger.info("🎉 Migration completed successfully!")
            
            return {
                "success": True,
                "backup_created": self.backup_path,
                "analysis": analysis,
                "migration_result": migration_result,
                "validation": validation_result,
                "default_timezone": default_timezone
            }
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _migrate_user_timezones(self, user_overrides: Dict[str, str], default_timezone: str):
        """Migrate user timezone preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all unique users
        cursor.execute("SELECT DISTINCT user_id FROM messages")
        users = [row[0] for row in cursor.fetchall()]
        
        for user_id in users:
            # Use override if provided, otherwise default
            timezone = user_overrides.get(user_id, default_timezone)
            
            # Validate timezone
            if not self.timezone_manager.validate_timezone(timezone):
                timezone = default_timezone
            
            # Insert or update user timezone
            cursor.execute("""
                INSERT OR REPLACE INTO users (user_id, timezone, updated_at)
                VALUES (?, ?, ?)
            """, (user_id, timezone, datetime.utcnow().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Migrated timezone preferences for {len(users)} users")
    
    def _validate_migration(self) -> Dict[str, Any]:
        """Validate that migration completed successfully"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check that all entries have both timestamps
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN utc_timestamp IS NOT NULL THEN 1 ELSE 0 END) as has_utc,
                       SUM(CASE WHEN local_timestamp IS NOT NULL THEN 1 ELSE 0 END) as has_local,
                       SUM(CASE WHEN timezone_at_creation IS NOT NULL THEN 1 ELSE 0 END) as has_timezone
                FROM messages
            """)
            
            result = cursor.fetchone()
            total, has_utc, has_local, has_timezone = result
            
            # Check for consistency
            cursor.execute("""
                SELECT COUNT(*) FROM messages 
                WHERE utc_timestamp IS NOT NULL 
                AND local_timestamp IS NOT NULL
                AND timezone_at_creation IS NOT NULL
            """)
            
            complete_entries = cursor.fetchone()[0]
            
            # Sample validation - check a few entries for correctness
            cursor.execute("""
                SELECT id, utc_timestamp, local_timestamp, timezone_at_creation
                FROM messages 
                WHERE utc_timestamp IS NOT NULL 
                LIMIT 10
            """)
            
            sample_entries = cursor.fetchall()
            validation_errors = []
            
            for entry_id, utc_str, local_str, timezone_name in sample_entries:
                try:
                    utc_dt = datetime.fromisoformat(utc_str)
                    local_dt = datetime.fromisoformat(local_str)
                    
                    # Verify conversion accuracy
                    expected_local = self.timezone_manager.convert_to_timezone(
                        utc_dt.replace(tzinfo=datetime.timezone.utc), timezone_name
                    )
                    
                    # Allow small differences due to timezone rules
                    time_diff = abs((expected_local.replace(tzinfo=None) - local_dt).total_seconds())
                    if time_diff > 3600:  # More than 1 hour difference
                        validation_errors.append(f"Entry {entry_id}: timestamp conversion error")
                
                except Exception as e:
                    validation_errors.append(f"Entry {entry_id}: validation error - {str(e)}")
            
            conn.close()
            
            validation_result = {
                "total_entries": total,
                "complete_migrations": complete_entries,
                "has_utc_timestamp": has_utc,
                "has_local_timestamp": has_local,
                "has_timezone": has_timezone,
                "completion_rate": complete_entries / total if total > 0 else 0,
                "validation_errors": validation_errors,
                "sample_validated": len(sample_entries)
            }
            
            if validation_result["completion_rate"] >= 0.95:  # 95% completion threshold
                logger.info(f"✅ Migration validation passed: {complete_entries}/{total} entries complete")
            else:
                logger.warning(f"⚠️ Migration validation concerns: {complete_entries}/{total} entries complete")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return {"error": str(e)}
    
    def _optimize_indexes(self):
        """Create optimized indexes for timestamp queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop old indexes that might conflict
        old_indexes = [
            "idx_messages_timestamp",
            "idx_messages_user_timestamp"
        ]
        
        for index_name in old_indexes:
            try:
                cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            except:
                pass
        
        # Create new optimized indexes
        new_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_utc_timestamp ON messages(utc_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_local_timestamp ON messages(local_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_user_local_time ON messages(user_id, local_timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_user_utc_time ON messages(user_id, utc_timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timezone ON messages(timezone_at_creation)",
            "CREATE INDEX IF NOT EXISTS idx_messages_validation_score ON messages(temporal_validation_score)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp_audit_entry ON timestamp_audit_log(entry_id)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp_audit_user ON timestamp_audit_log(user_id, changed_at DESC)"
        ]
        
        for index_sql in new_indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database indexes optimized for timestamp queries")
    
    def rollback_migration(self) -> bool:
        """Rollback migration by restoring from backup"""
        try:
            if not os.path.exists(self.backup_path):
                logger.error("❌ Backup file not found - cannot rollback")
                return False
            
            # Replace current database with backup
            shutil.copy2(self.backup_path, self.db_path)
            logger.info(f"✅ Migration rolled back - database restored from {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def cleanup_backup(self) -> bool:
        """Remove backup file after successful migration"""
        try:
            if os.path.exists(self.backup_path):
                os.remove(self.backup_path)
                logger.info(f"✅ Backup file removed: {self.backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove backup: {e}")
            return False

def run_migration_interactive():
    """Interactive migration script for command-line usage"""
    print("🕐 Journal Backend Timestamp Migration")
    print("=" * 50)
    
    # Get database path
    db_path = input("Enter database path (or press Enter for 'journal.db'): ").strip()
    if not db_path:
        db_path = "journal.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    # Get default timezone
    default_tz = input("Enter default timezone (or press Enter for 'America/Chicago'): ").strip()
    if not default_tz:
        default_tz = "America/Chicago"
    
    # Validate timezone
    tz_manager = TimezoneManager()
    if not tz_manager.validate_timezone(default_tz):
        print(f"❌ Invalid timezone: {default_tz}")
        suggestions = tz_manager.get_timezone_suggestions()
        print(f"Suggestions: {', '.join(suggestions[:5])}")
        return
    
    # Create migration manager
    migrator = TimestampMigrationManager(db_path)
    
    # Show analysis
    print("\n📊 Analyzing existing data...")
    analysis = migrator.analyze_existing_timestamps()
    
    if "error" in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return
    
    print(f"Total entries: {analysis['total_entries']}")
    print(f"Timestamp formats: {analysis['timestamp_formats']}")
    print(f"Potential issues: {len(analysis['potential_issues'])}")
    
    if analysis['potential_issues']:
        print("Issues found:")
        for issue in analysis['potential_issues'][:5]:
            print(f"  - {issue}")
    
    # Confirm migration
    confirm = input("\nProceed with migration? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Migration cancelled.")
        return
    
    # Run migration
    print("\n🚀 Starting migration...")
    result = migrator.perform_migration(default_tz)
    
    if result["success"]:
        print("🎉 Migration completed successfully!")
        print(f"Backup created: {result['backup_created']}")
        print(f"Entries migrated: {result['migration_result']['migrated_count']}")
        
        # Ask about cleanup
        cleanup = input("\nRemove backup file? (y/N): ").strip().lower()
        if cleanup == 'y':
            migrator.cleanup_backup()
    else:
        print(f"❌ Migration failed: {result['error']}")
        if 'details' in result:
            print(f"Details: {result['details']}")

if __name__ == "__main__":
    run_migration_interactive()