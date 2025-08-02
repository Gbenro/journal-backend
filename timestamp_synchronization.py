"""
Comprehensive Timestamp Synchronization System
Provides intelligent timezone management, timestamp validation, and temporal intelligence.
"""

import sqlite3
import pytz
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import re
# Delayed import to avoid circular dependency
# from temporal_awareness import TemporalSignalDetector, SignalType

logger = logging.getLogger(__name__)

class TimestampSource(Enum):
    AUTO = "auto"
    MANUAL_OVERRIDE = "manual_override"
    CORRECTED = "corrected"
    MIGRATED = "migrated"
    CLIENT_PROVIDED = "client_provided"

class ValidationSeverity(Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SUSPICIOUS = "suspicious"

@dataclass
class TimestampInfo:
    utc_timestamp: datetime
    local_timestamp: datetime
    timezone_name: str
    timestamp_source: TimestampSource
    validation_score: float
    validation_notes: List[str]

@dataclass
class ValidationResult:
    severity: ValidationSeverity
    confidence: float
    message: str
    suggested_correction: Optional[Dict] = None
    metadata: Optional[Dict] = None

class TimezoneManager:
    """Sophisticated timezone management with pytz integration"""
    
    def __init__(self):
        self.default_timezone = "America/Chicago"
        self.common_timezones = [
            "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
            "America/Phoenix", "America/Anchorage", "America/Honolulu",
            "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Rome",
            "Asia/Tokyo", "Asia/Shanghai", "Asia/Hong_Kong", "Asia/Kolkata",
            "Australia/Sydney", "Australia/Melbourne", "Pacific/Auckland",
            "America/Toronto", "America/Vancouver", "America/Sao_Paulo",
            "UTC", "GMT"
        ]
        
    def validate_timezone(self, timezone_str: str) -> bool:
        """Validate if timezone string is valid"""
        try:
            pytz.timezone(timezone_str)
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            return False
    
    def get_timezone_suggestions(self, partial_name: str = None, country_code: str = None) -> List[str]:
        """Get timezone suggestions based on partial name or country"""
        if partial_name:
            partial_lower = partial_name.lower()
            suggestions = [tz for tz in self.common_timezones 
                          if partial_lower in tz.lower()]
            
            # Also search in all pytz timezones if no common matches
            if not suggestions:
                all_timezones = list(pytz.all_timezones)
                suggestions = [tz for tz in all_timezones 
                              if partial_lower in tz.lower()][:10]
            
            return suggestions
        
        return self.common_timezones[:10]
    
    def detect_timezone_from_offset(self, offset_minutes: int) -> List[str]:
        """Detect possible timezones from UTC offset in minutes"""
        target_offset = timedelta(minutes=offset_minutes)
        now = datetime.now(timezone.utc)
        
        possible_timezones = []
        
        for tz_name in self.common_timezones:
            try:
                tz = pytz.timezone(tz_name)
                tz_offset = tz.utcoffset(now)
                
                if tz_offset == target_offset:
                    possible_timezones.append(tz_name)
            except:
                continue
        
        return possible_timezones
    
    def convert_to_timezone(self, dt: datetime, target_timezone: str, 
                           source_timezone: str = None) -> datetime:
        """Convert datetime to target timezone"""
        try:
            target_tz = pytz.timezone(target_timezone)
            
            # If datetime is naive, assume it's in source timezone or UTC
            if dt.tzinfo is None:
                if source_timezone:
                    source_tz = pytz.timezone(source_timezone)
                    dt = source_tz.localize(dt)
                else:
                    dt = pytz.UTC.localize(dt)
            
            # Convert to target timezone
            return dt.astimezone(target_tz)
            
        except Exception as e:
            logger.error(f"Timezone conversion failed: {e}")
            return dt
    
    def get_local_and_utc(self, dt: Union[datetime, str], timezone_name: str) -> Tuple[datetime, datetime]:
        """Get both local and UTC versions of a datetime"""
        try:
            # Parse string datetime if needed
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            
            # Handle timezone-aware datetime
            if dt.tzinfo is not None:
                utc_dt = dt.astimezone(pytz.UTC).replace(tzinfo=None)
                local_dt = dt.astimezone(pytz.timezone(timezone_name)).replace(tzinfo=None)
            else:
                # Assume naive datetime is in the specified timezone
                local_tz = pytz.timezone(timezone_name)
                local_dt = dt
                aware_dt = local_tz.localize(dt)
                utc_dt = aware_dt.astimezone(pytz.UTC).replace(tzinfo=None)
            
            return local_dt, utc_dt
            
        except Exception as e:
            logger.error(f"Failed to get local and UTC timestamps: {e}")
            # Fallback: assume input is UTC
            utc_dt = dt if isinstance(dt, datetime) else datetime.fromisoformat(dt)
            local_dt = self.convert_to_timezone(
                pytz.UTC.localize(utc_dt), timezone_name
            ).replace(tzinfo=None)
            return local_dt, utc_dt
    
    def get_user_timezone(self, db_path: str, user_id: str) -> str:
        """Get user's timezone preference from database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Try to get from users table first
            cursor.execute("SELECT timezone FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result:
                conn.close()
                return result[0] if self.validate_timezone(result[0]) else self.default_timezone
            
            # Fallback to temporal_state table
            cursor.execute("SELECT timezone FROM temporal_state WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return result[0] if self.validate_timezone(result[0]) else self.default_timezone
            
            return self.default_timezone
            
        except Exception as e:
            logger.error(f"Failed to get user timezone: {e}")
            return self.default_timezone
    
    def set_user_timezone(self, db_path: str, user_id: str, timezone_name: str) -> bool:
        """Set user's timezone preference"""
        if not self.validate_timezone(timezone_name):
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Update or insert into users table
            cursor.execute("""
                INSERT OR REPLACE INTO users (user_id, timezone, updated_at)
                VALUES (?, ?, ?)
            """, (user_id, timezone_name, datetime.utcnow().isoformat()))
            
            # Also update temporal_state table if exists
            cursor.execute("""
                UPDATE temporal_state SET timezone = ?, updated_at = ?
                WHERE user_id = ?
            """, (timezone_name, datetime.utcnow().isoformat(), user_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set user timezone: {e}")
            return False

class TemporalValidator:
    """Advanced timestamp validation with temporal intelligence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.timezone_manager = TimezoneManager()
        self.signal_detector = TemporalSignalDetector()
        
    def validate_entry_timestamp(self, content: str, timestamp: datetime, 
                                timezone_name: str, user_id: str) -> ValidationResult:
        """Comprehensive timestamp validation for journal entries"""
        
        # Detect temporal signals in content
        signals = self.signal_detector.detect_signals(content, timestamp)
        
        # Get user's typical patterns
        user_patterns = self._get_user_patterns(user_id)
        
        # Validate against temporal signals
        signal_validation = self._validate_against_signals(signals, timestamp, timezone_name)
        
        # Validate against user patterns
        pattern_validation = self._validate_against_patterns(timestamp, timezone_name, user_patterns)
        
        # Check for suspicious timing patterns
        anomaly_validation = self._detect_anomalies(timestamp, timezone_name, signals)
        
        # Combine validation results
        return self._combine_validations([signal_validation, pattern_validation, anomaly_validation])
    
    def _validate_against_signals(self, signals: List, timestamp: datetime, 
                                 timezone_name: str) -> ValidationResult:
        """Validate timestamp against detected temporal signals"""
        if not signals:
            return ValidationResult(
                severity=ValidationSeverity.OK,
                confidence=0.5,
                message="No temporal signals detected to validate against"
            )
        
        # Convert to local time for validation
        local_time = self.timezone_manager.convert_to_timezone(
            pytz.UTC.localize(timestamp), timezone_name
        )
        
        hour = local_time.hour
        weekday = local_time.weekday()
        day_of_month = local_time.day
        
        validation_issues = []
        confidence_adjustments = []
        
        for signal in signals:
            signal_type = signal.signal_type
            base_confidence = signal.confidence
            
            # Validate day start signals
            if signal_type == SignalType.DAY_START:
                if hour >= 18:  # Evening
                    validation_issues.append(f"Day start signal detected at {hour}:00 (evening)")
                    confidence_adjustments.append(-0.3)
                elif 5 <= hour <= 11:  # Morning
                    confidence_adjustments.append(0.2)
                    
            # Validate day end signals
            elif signal_type == SignalType.DAY_END:
                if hour <= 12:  # Morning/noon
                    validation_issues.append(f"Day end signal detected at {hour}:00 (morning)")
                    confidence_adjustments.append(-0.3)
                elif 18 <= hour <= 23:  # Evening
                    confidence_adjustments.append(0.2)
            
            # Validate week start signals
            elif signal_type == SignalType.WEEK_START:
                if weekday == 0:  # Monday
                    confidence_adjustments.append(0.3)
                elif weekday >= 4:  # Friday or later
                    validation_issues.append(f"Week start signal on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]}")
                    confidence_adjustments.append(-0.2)
            
            # Validate month start signals
            elif signal_type == SignalType.MONTH_START:
                if day_of_month <= 3:
                    confidence_adjustments.append(0.2)
                elif day_of_month >= 15:
                    validation_issues.append(f"Month start signal on day {day_of_month}")
                    confidence_adjustments.append(-0.2)
        
        # Calculate overall confidence
        avg_confidence_adj = sum(confidence_adjustments) / len(confidence_adjustments) if confidence_adjustments else 0
        final_confidence = max(0.0, min(1.0, 0.7 + avg_confidence_adj))
        
        if validation_issues:
            return ValidationResult(
                severity=ValidationSeverity.WARNING,
                confidence=final_confidence,
                message=f"Temporal signal timing inconsistencies: {'; '.join(validation_issues)}",
                metadata={"signal_count": len(signals), "issues": validation_issues}
            )
        else:
            return ValidationResult(
                severity=ValidationSeverity.OK,
                confidence=final_confidence,
                message=f"Temporal signals align well with timestamp ({len(signals)} signals detected)"
            )
    
    def _validate_against_patterns(self, timestamp: datetime, timezone_name: str, 
                                  user_patterns: Dict) -> ValidationResult:
        """Validate against user's historical patterns"""
        local_time = self.timezone_manager.convert_to_timezone(
            pytz.UTC.localize(timestamp), timezone_name
        )
        
        hour = local_time.hour
        weekday = local_time.weekday()
        
        typical_hours = user_patterns.get('typical_hours', [])
        typical_weekdays = user_patterns.get('typical_weekdays', [])
        
        issues = []
        confidence = 0.8
        
        # Check against typical hours
        if typical_hours:
            if hour not in typical_hours:
                closest_hour = min(typical_hours, key=lambda x: abs(x - hour))
                hours_diff = abs(hour - closest_hour)
                
                if hours_diff > 6:  # More than 6 hours from typical
                    issues.append(f"Entry at {hour}:00, but you typically journal around {closest_hour}:00")
                    confidence -= 0.3
                elif hours_diff > 3:  # 3-6 hours from typical
                    issues.append(f"Entry timing unusual (typically around {closest_hour}:00)")
                    confidence -= 0.1
        
        # Check against typical weekdays
        if typical_weekdays and weekday not in typical_weekdays:
            typical_day_names = [['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d] for d in typical_weekdays]
            current_day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]
            issues.append(f"Entry on {current_day_name}, but you typically journal on {', '.join(typical_day_names)}")
            confidence -= 0.1
        
        if issues:
            return ValidationResult(
                severity=ValidationSeverity.WARNING,
                confidence=max(0.0, confidence),
                message=f"Pattern deviations detected: {'; '.join(issues)}",
                metadata={"user_patterns": user_patterns}
            )
        else:
            return ValidationResult(
                severity=ValidationSeverity.OK,
                confidence=min(1.0, confidence + 0.1),
                message="Timestamp aligns with your typical journaling patterns"
            )
    
    def _detect_anomalies(self, timestamp: datetime, timezone_name: str, 
                         signals: List) -> ValidationResult:
        """Detect suspicious timestamp patterns"""
        local_time = self.timezone_manager.convert_to_timezone(
            pytz.UTC.localize(timestamp), timezone_name
        )
        
        hour = local_time.hour
        minute = local_time.minute
        
        anomalies = []
        
        # Check for exact hour/minute patterns (might indicate fake timestamps)
        if minute == 0:
            anomalies.append("Entry at exactly the hour (00 minutes)")
        
        # Check for unusual late night/early morning entries
        if 2 <= hour <= 5 and not any(s.signal_type in [SignalType.DAY_END, SignalType.TRANSITION] for s in signals):
            anomalies.append(f"Entry at {hour}:00 (unusual hour without night-related signals)")
        
        # Check for weekend vs weekday pattern breaks
        is_weekend = local_time.weekday() >= 5
        if is_weekend and 6 <= hour <= 9 and any(s.signal_type == SignalType.DAY_START for s in signals):
            # Early weekend morning is actually normal for many people
            pass
        elif is_weekend and 22 <= hour <= 23 and not signals:
            anomalies.append("Late weekend entry without temporal context")
        
        if anomalies:
            return ValidationResult(
                severity=ValidationSeverity.SUSPICIOUS,
                confidence=0.3,
                message=f"Timestamp anomalies detected: {'; '.join(anomalies)}",
                metadata={"anomalies": anomalies, "local_time": local_time.isoformat()}
            )
        else:
            return ValidationResult(
                severity=ValidationSeverity.OK,
                confidence=0.8,
                message="No timestamp anomalies detected"
            )
    
    def _get_user_patterns(self, user_id: str) -> Dict:
        """Get user's historical journaling patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get entries from last 30 days
            thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
            
            cursor.execute("""
                SELECT local_timestamp FROM messages 
                WHERE user_id = ? AND utc_timestamp > ?
                ORDER BY utc_timestamp DESC
                LIMIT 100
            """, (user_id, thirty_days_ago))
            
            entries = cursor.fetchall()
            conn.close()
            
            if not entries:
                return {}
            
            # Analyze patterns
            hours = []
            weekdays = []
            
            for entry in entries:
                try:
                    dt = datetime.fromisoformat(entry[0])
                    hours.append(dt.hour)
                    weekdays.append(dt.weekday())
                except:
                    continue
            
            # Find typical hours (most common hours)
            if hours:
                hour_counts = {}
                for hour in hours:
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                
                # Get hours that appear in at least 20% of entries
                min_count = max(1, len(hours) * 0.2)
                typical_hours = [hour for hour, count in hour_counts.items() if count >= min_count]
            else:
                typical_hours = []
            
            # Find typical weekdays
            if weekdays:
                weekday_counts = {}
                for day in weekdays:
                    weekday_counts[day] = weekday_counts.get(day, 0) + 1
                
                # Get weekdays that appear in at least 15% of entries
                min_count = max(1, len(weekdays) * 0.15)
                typical_weekdays = [day for day, count in weekday_counts.items() if count >= min_count]
            else:
                typical_weekdays = []
            
            return {
                "typical_hours": typical_hours,
                "typical_weekdays": typical_weekdays,
                "total_entries": len(entries),
                "analysis_period_days": 30
            }
            
        except Exception as e:
            logger.error(f"Failed to get user patterns: {e}")
            return {}
    
    def _combine_validations(self, validations: List[ValidationResult]) -> ValidationResult:
        """Combine multiple validation results into a single result"""
        if not validations:
            return ValidationResult(
                severity=ValidationSeverity.OK,
                confidence=0.5,
                message="No validation performed"
            )
        
        # Determine overall severity
        severities = [v.severity for v in validations]
        if ValidationSeverity.ERROR in severities:
            overall_severity = ValidationSeverity.ERROR
        elif ValidationSeverity.SUSPICIOUS in severities:
            overall_severity = ValidationSeverity.SUSPICIOUS
        elif ValidationSeverity.WARNING in severities:
            overall_severity = ValidationSeverity.WARNING
        else:
            overall_severity = ValidationSeverity.OK
        
        # Calculate average confidence
        confidences = [v.confidence for v in validations]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Combine messages
        messages = [v.message for v in validations if v.message]
        combined_message = "; ".join(messages)
        
        return ValidationResult(
            severity=overall_severity,
            confidence=avg_confidence,
            message=combined_message,
            metadata={"individual_validations": [asdict(v) for v in validations]}
        )

class TimestampSynchronizer:
    """Main class for timestamp synchronization operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.timezone_manager = TimezoneManager()
        self.validator = TemporalValidator(db_path)
    
    def create_timestamp_info(self, content: str, user_id: str, 
                            client_timestamp: Optional[str] = None,
                            client_timezone: Optional[str] = None,
                            timestamp_source: TimestampSource = TimestampSource.AUTO) -> TimestampInfo:
        """Create comprehensive timestamp information for an entry"""
        
        # Get user's timezone
        user_timezone = client_timezone or self.timezone_manager.get_user_timezone(self.db_path, user_id)
        
        # Determine the timestamp to use
        if client_timestamp:
            try:
                base_timestamp = datetime.fromisoformat(client_timestamp.replace('Z', '+00:00'))
                if base_timestamp.tzinfo is None:
                    # Assume client timestamp is in user's timezone
                    local_dt = base_timestamp
                    _, utc_dt = self.timezone_manager.get_local_and_utc(base_timestamp, user_timezone)
                else:
                    utc_dt = base_timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
                    local_dt = base_timestamp.astimezone(pytz.timezone(user_timezone)).replace(tzinfo=None)
                
                timestamp_source = TimestampSource.CLIENT_PROVIDED
            except Exception as e:
                logger.warning(f"Failed to parse client timestamp: {e}")
                utc_dt = datetime.utcnow()
                local_dt = self.timezone_manager.convert_to_timezone(
                    pytz.UTC.localize(utc_dt), user_timezone
                ).replace(tzinfo=None)
                timestamp_source = TimestampSource.AUTO
        else:
            # Use current time
            utc_dt = datetime.utcnow()
            local_dt = self.timezone_manager.convert_to_timezone(
                pytz.UTC.localize(utc_dt), user_timezone
            ).replace(tzinfo=None)
        
        # Validate the timestamp
        validation = self.validator.validate_entry_timestamp(content, utc_dt, user_timezone, user_id)
        
        return TimestampInfo(
            utc_timestamp=utc_dt,
            local_timestamp=local_dt,
            timezone_name=user_timezone,
            timestamp_source=timestamp_source,
            validation_score=validation.confidence,
            validation_notes=[validation.message] if validation.message else []
        )
    
    def override_timestamp(self, entry_id: int, new_timestamp: str, 
                          timezone_name: str, user_id: str, reason: str = "") -> bool:
        """Override an entry's timestamp with manual correction"""
        try:
            # Validate new timestamp
            new_dt = datetime.fromisoformat(new_timestamp)
            local_dt, utc_dt = self.timezone_manager.get_local_and_utc(new_dt, timezone_name)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Log the change for audit trail
            cursor.execute("""
                INSERT INTO timestamp_audit_log 
                (entry_id, user_id, old_utc_timestamp, old_local_timestamp, old_timezone,
                 new_utc_timestamp, new_local_timestamp, new_timezone, reason, changed_at)
                SELECT id, user_id, utc_timestamp, local_timestamp, timezone_at_creation,
                       ?, ?, ?, ?, ?
                FROM messages WHERE id = ?
            """, (utc_dt.isoformat(), local_dt.isoformat(), timezone_name, 
                  reason, datetime.utcnow().isoformat(), entry_id))
            
            # Update the entry
            cursor.execute("""
                UPDATE messages 
                SET utc_timestamp = ?, local_timestamp = ?, 
                    timezone_at_creation = ?, timestamp_source = ?,
                    temporal_validation_score = ?
                WHERE id = ? AND user_id = ?
            """, (utc_dt.isoformat(), local_dt.isoformat(), timezone_name,
                  TimestampSource.MANUAL_OVERRIDE.value, 1.0, entry_id, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Timestamp overridden for entry {entry_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to override timestamp: {e}")
            return False
    
    def bulk_correct_timestamps(self, user_id: str, timezone_correction: str, 
                               date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Bulk correct timestamps for timezone changes or corrections"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query with optional date range
            base_query = "SELECT id, utc_timestamp, timezone_at_creation FROM messages WHERE user_id = ?"
            params = [user_id]
            
            if date_range:
                base_query += " AND utc_timestamp BETWEEN ? AND ?"
                params.extend(date_range)
            
            cursor.execute(base_query, params)
            entries = cursor.fetchall()
            
            corrections_made = 0
            errors = []
            
            for entry_id, utc_timestamp_str, old_timezone in entries:
                try:
                    # Parse existing timestamp
                    utc_dt = datetime.fromisoformat(utc_timestamp_str)
                    
                    # Calculate new local timestamp with corrected timezone
                    new_local_dt = self.timezone_manager.convert_to_timezone(
                        pytz.UTC.localize(utc_dt), timezone_correction
                    ).replace(tzinfo=None)
                    
                    # Update entry
                    cursor.execute("""
                        UPDATE messages 
                        SET local_timestamp = ?, timezone_at_creation = ?, 
                            timestamp_source = ?
                        WHERE id = ?
                    """, (new_local_dt.isoformat(), timezone_correction,
                          TimestampSource.CORRECTED.value, entry_id))
                    
                    corrections_made += 1
                    
                except Exception as e:
                    errors.append(f"Entry {entry_id}: {str(e)}")
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "corrections_made": corrections_made,
                "total_entries": len(entries),
                "errors": errors,
                "new_timezone": timezone_correction
            }
            
        except Exception as e:
            logger.error(f"Bulk timestamp correction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "corrections_made": 0
            }

def create_timestamp_tables(db_path: str) -> None:
    """Create timestamp synchronization tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            timezone TEXT DEFAULT 'America/Chicago',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create timestamp audit log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timestamp_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            old_utc_timestamp DATETIME,
            old_local_timestamp DATETIME,
            old_timezone TEXT,
            new_utc_timestamp DATETIME,
            new_local_timestamp DATETIME,
            new_timezone TEXT,
            reason TEXT,
            changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entry_id) REFERENCES messages(id) ON DELETE CASCADE
        )
    """)
    
    # Add columns to messages table if they don't exist
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN local_timestamp DATETIME")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN utc_timestamp DATETIME")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN timestamp_source TEXT DEFAULT 'auto'")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN timezone_at_creation TEXT DEFAULT 'America/Chicago'")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN temporal_validation_score REAL DEFAULT 0.5")
    except sqlite3.OperationalError:
        pass
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_utc_timestamp ON messages(utc_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_local_timestamp ON messages(local_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_local_time ON messages(user_id, local_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timezone ON messages(timezone_at_creation)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_audit_entry ON timestamp_audit_log(entry_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_audit_user ON timestamp_audit_log(user_id)")
    
    conn.commit()
    conn.close()
    
    logger.info("✅ Timestamp synchronization tables created successfully")

def migrate_existing_timestamps(db_path: str, default_timezone: str = "America/Chicago") -> Dict[str, Any]:
    """Migrate existing timestamps to the new dual-timestamp system"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timezone_manager = TimezoneManager()
        
        # Get all messages that need migration (missing utc_timestamp or local_timestamp)
        cursor.execute("""
            SELECT id, user_id, timestamp, timezone_at_creation
            FROM messages 
            WHERE utc_timestamp IS NULL OR local_timestamp IS NULL
        """)
        
        entries = cursor.fetchall()
        migrated_count = 0
        errors = []
        
        for entry_id, user_id, old_timestamp, existing_timezone in entries:
            try:
                # Parse existing timestamp
                if old_timestamp:
                    dt = datetime.fromisoformat(old_timestamp)
                else:
                    dt = datetime.utcnow()
                
                # Determine timezone to use
                user_timezone = existing_timezone or timezone_manager.get_user_timezone(db_path, user_id)
                if not timezone_manager.validate_timezone(user_timezone):
                    user_timezone = default_timezone
                
                # Get both local and UTC timestamps
                local_dt, utc_dt = timezone_manager.get_local_and_utc(dt, user_timezone)
                
                # Update the entry
                cursor.execute("""
                    UPDATE messages 
                    SET utc_timestamp = ?, local_timestamp = ?, 
                        timezone_at_creation = ?, timestamp_source = ?,
                        temporal_validation_score = ?
                    WHERE id = ?
                """, (utc_dt.isoformat(), local_dt.isoformat(), user_timezone,
                      TimestampSource.MIGRATED.value, 0.7, entry_id))
                
                migrated_count += 1
                
            except Exception as e:
                errors.append(f"Entry {entry_id}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "migrated_count": migrated_count,
            "total_entries": len(entries),
            "errors": errors,
            "default_timezone": default_timezone
        }
        
    except Exception as e:
        logger.error(f"Timestamp migration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "migrated_count": 0
        }