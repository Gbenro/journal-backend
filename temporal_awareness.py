"""
Minimal Temporal Awareness Module for Testing
Provides basic temporal signal detection for timestamp validation.
"""

from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List
import re

class SignalType(Enum):
    DAY_START = "day_start"
    DAY_END = "day_end"
    WEEK_START = "week_start" 
    WEEK_END = "week_end"
    MONTH_START = "month_start"
    TRANSITION = "transition"
    TIME_REFERENCE = "time_reference"

@dataclass
class TemporalSignal:
    signal_type: SignalType
    confidence: float
    text_span: str
    metadata: dict = None

class TemporalSignalDetector:
    """Basic temporal signal detection for testing purposes"""
    
    def __init__(self):
        self.patterns = {
            SignalType.DAY_START: [
                r"(good )?morning", r"start(ed|ing) (my|the) day", r"woke up", r"wake up",
                r"first thing", r"breakfast", r"coffee", r"dawn", r"sunrise"
            ],
            SignalType.DAY_END: [
                r"(good )?evening", r"(good )?night", r"end of (the )?day", r"going to bed",
                r"bedtime", r"tired", r"exhausted", r"reflect(ing|ion)", r"sunset"
            ],
            SignalType.WEEK_START: [
                r"monday", r"start of (the )?week", r"new week", r"week(ly)? planning"
            ],
            SignalType.TRANSITION: [
                r"meanwhile", r"later", r"then", r"after", r"before", r"during"
            ]
        }
    
    def detect_signals(self, content: str, timestamp: datetime = None) -> List[TemporalSignal]:
        """Detect temporal signals in content"""
        signals = []
        content_lower = content.lower()
        
        for signal_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    confidence = 0.7  # Basic confidence
                    
                    # Adjust confidence based on context
                    if signal_type == SignalType.DAY_START and timestamp:
                        hour = timestamp.hour
                        if 5 <= hour <= 11:  # Morning hours
                            confidence = 0.9
                        elif hour >= 18:  # Evening hours
                            confidence = 0.3
                    
                    signals.append(TemporalSignal(
                        signal_type=signal_type,
                        confidence=confidence,
                        text_span=match.group(0),
                        metadata={"start": match.start(), "end": match.end()}
                    ))
        
        # Remove duplicates and return highest confidence signals
        unique_signals = {}
        for signal in signals:
            key = (signal.signal_type, signal.text_span)
            if key not in unique_signals or signal.confidence > unique_signals[key].confidence:
                unique_signals[key] = signal
        
        return list(unique_signals.values())


@dataclass
class TemporalState:
    """Represents the temporal state of a user"""
    user_id: str
    current_timezone: str = "UTC"
    last_signal: TemporalSignal = None
    last_updated: datetime = None


class TemporalStateManager:
    """Manages temporal state for users with database persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.states = {}  # In-memory cache for performance
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection and ensure tables exist"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure temporal_states table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temporal_states (
                user_id TEXT PRIMARY KEY,
                current_timezone TEXT DEFAULT 'UTC',
                last_signal_type TEXT,
                last_signal_confidence REAL,
                last_signal_text TEXT,
                last_signal_metadata TEXT,
                last_signal_time TIMESTAMP,
                activity_pattern TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for temporal queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_temporal_states_updated ON temporal_states(updated_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_temporal_states_signal_time ON temporal_states(last_signal_time)')
        
        conn.commit()
        conn.close()
    
    def record_temporal_signal(self, user_id: str, signal: TemporalSignal, entry_id: str = None):
        """Record a temporal signal for a user with database persistence"""
        import sqlite3
        import json
        
        # Update in-memory cache
        if user_id not in self.states:
            self.states[user_id] = TemporalState(user_id=user_id)
        
        self.states[user_id].last_signal = signal
        self.states[user_id].last_updated = datetime.now()
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO temporal_states 
                (user_id, current_timezone, last_signal_type, last_signal_confidence, 
                 last_signal_text, last_signal_metadata, last_signal_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                self.states[user_id].current_timezone,
                signal.signal_type.value,
                signal.confidence,
                signal.text_span,
                json.dumps(signal.metadata) if signal.metadata else None,
                datetime.now(),
                datetime.now()
            ))
            conn.commit()
        except Exception as e:
            print(f"Error persisting temporal signal: {e}")
        finally:
            conn.close()
        
        # Also record in temporal_signals table for historical tracking
        self._record_signal_history(user_id, signal, entry_id)
    
    def _record_signal_history(self, user_id: str, signal: TemporalSignal, entry_id: str = None):
        """Record signal in historical tracking table"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO temporal_signals 
                (user_id, entry_id, signal_type, confidence, text_span, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                entry_id,
                signal.signal_type.value,
                signal.confidence,
                signal.text_span,
                json.dumps(signal.metadata) if signal.metadata else None,
                datetime.now()
            ))
            conn.commit()
        except Exception as e:
            print(f"Error recording signal history: {e}")
        finally:
            conn.close()
    
    def get_temporal_state(self, user_id: str) -> TemporalState:
        """Get the temporal state for a user, loading from database if not cached"""
        if user_id not in self.states:
            self._load_state_from_db(user_id)
        
        if user_id not in self.states:
            self.states[user_id] = TemporalState(user_id=user_id)
            
        return self.states[user_id]
    
    def _load_state_from_db(self, user_id: str):
        """Load temporal state from database"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT current_timezone, last_signal_type, last_signal_confidence,
                       last_signal_text, last_signal_metadata, last_signal_time
                FROM temporal_states 
                WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                timezone, signal_type, confidence, text_span, metadata_json, signal_time = row
                
                # Reconstruct temporal signal if exists
                last_signal = None
                if signal_type:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    last_signal = TemporalSignal(
                        signal_type=SignalType(signal_type),
                        confidence=confidence,
                        text_span=text_span,
                        metadata=metadata
                    )
                
                # Create temporal state
                self.states[user_id] = TemporalState(
                    user_id=user_id,
                    current_timezone=timezone,
                    last_signal=last_signal,
                    last_updated=datetime.fromisoformat(signal_time) if signal_time else None
                )
        except Exception as e:
            print(f"Error loading temporal state: {e}")
        finally:
            conn.close()
    
    def get_user_timezone(self, user_id: str) -> str:
        """Get the timezone for a user"""
        state = self.get_temporal_state(user_id)
        return state.current_timezone
    
    def update_user_timezone(self, user_id: str, timezone: str):
        """Update user's timezone preference"""
        import sqlite3
        
        state = self.get_temporal_state(user_id)
        state.current_timezone = timezone
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO temporal_states 
                (user_id, current_timezone, updated_at)
                VALUES (?, ?, ?)
            ''', (user_id, timezone, datetime.now()))
            conn.commit()
        except Exception as e:
            print(f"Error updating timezone: {e}")
        finally:
            conn.close()
    
    def get_recent_signals(self, user_id: str, hours_back: int = 24) -> List[TemporalSignal]:
        """Get recent temporal signals for analysis"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            cursor.execute('''
                SELECT signal_type, confidence, text_span, metadata, created_at
                FROM temporal_signals 
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
            ''', (user_id, cutoff_time))
            
            signals = []
            for row in cursor.fetchall():
                signal_type, confidence, text_span, metadata_json, created_at = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                signals.append(TemporalSignal(
                    signal_type=SignalType(signal_type),
                    confidence=confidence,
                    text_span=text_span,
                    metadata=metadata
                ))
            
            return signals
        except Exception as e:
            print(f"Error getting recent signals: {e}")
            return []
        finally:
            conn.close()


class TemporalSummaryGenerator:
    """Generates temporal summaries for users with advanced analytics"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def generate_period_summary(self, user_id: str, period_type: str, start_date: datetime, end_date: datetime) -> dict:
        """Generate a comprehensive summary for a time period"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get entries in the period
            cursor.execute('''
                SELECT id, content, timestamp, 
                       COALESCE(local_timestamp, timestamp) as display_time,
                       temporal_validation_score
                FROM messages 
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (user_id, start_date.isoformat(), end_date.isoformat()))
            
            entries = cursor.fetchall()
            entry_ids = [entry[0] for entry in entries]
            
            # Get temporal signals for the period
            signals = []
            if entry_ids:
                placeholders = ','.join('?' * len(entry_ids))
                cursor.execute(f'''
                    SELECT signal_type, confidence, text_span, created_at
                    FROM temporal_signals 
                    WHERE user_id = ? AND (entry_id IN ({placeholders}) OR 
                          created_at BETWEEN ? AND ?)
                    ORDER BY created_at ASC
                ''', [user_id] + entry_ids + [start_date.isoformat(), end_date.isoformat()])
                
                signals = cursor.fetchall()
            
            # Get tags used in the period
            cursor.execute('''
                SELECT t.name, t.category, COUNT(*) as usage_count,
                       AVG(et.confidence) as avg_confidence
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                JOIN messages m ON et.message_id = m.id
                WHERE m.user_id = ? AND m.timestamp BETWEEN ? AND ?
                GROUP BY t.name, t.category
                ORDER BY usage_count DESC
            ''', (user_id, start_date.isoformat(), end_date.isoformat()))
            
            tag_usage = cursor.fetchall()
            
            # Calculate temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(entries, signals)
            
            return {
                "user_id": user_id,
                "period_type": period_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "summary": f"Temporal analysis for {period_type} from {start_date.date()} to {end_date.date()}",
                "entry_count": len(entries),
                "temporal_signals": [
                    {
                        "type": signal[0],
                        "confidence": signal[1],
                        "text": signal[2],
                        "timestamp": signal[3]
                    } for signal in signals
                ],
                "tag_usage": [
                    {
                        "name": tag[0],
                        "category": tag[1],
                        "count": tag[2],
                        "avg_confidence": round(tag[3], 2)
                    } for tag in tag_usage
                ],
                "temporal_patterns": temporal_patterns,
                "activity_score": self._calculate_activity_score(entries, signals),
                "timezone_consistency": self._analyze_timezone_consistency(entries)
            }
            
        except Exception as e:
            print(f"Error generating period summary: {e}")
            return {
                "user_id": user_id,
                "period_type": period_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "summary": f"Error generating summary: {str(e)}",
                "entry_count": 0,
                "temporal_signals": []
            }
        finally:
            conn.close()
    
    def _analyze_temporal_patterns(self, entries: List, signals: List) -> dict:
        """Analyze temporal patterns in entries and signals"""
        if not entries:
            return {"patterns": [], "peak_hours": [], "quiet_periods": []}
        
        # Analyze entry timing patterns
        entry_hours = []
        for entry in entries:
            try:
                timestamp = datetime.fromisoformat(entry[3])  # display_time
                entry_hours.append(timestamp.hour)
            except:
                continue
        
        # Find peak activity hours
        hour_counts = {}
        for hour in entry_hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Analyze signal patterns
        signal_patterns = {}
        for signal in signals:
            signal_type = signal[0]
            signal_patterns[signal_type] = signal_patterns.get(signal_type, 0) + 1
        
        return {
            "peak_hours": [{"hour": h[0], "count": h[1]} for h in peak_hours],
            "signal_distribution": signal_patterns,
            "total_active_hours": len(set(entry_hours)),
            "avg_entries_per_active_hour": len(entries) / max(len(set(entry_hours)), 1)
        }
    
    def _calculate_activity_score(self, entries: List, signals: List) -> float:
        """Calculate a temporal activity score (0-100)"""
        if not entries:
            return 0.0
        
        # Base score from entry count (max 40 points)
        entry_score = min(len(entries) * 2, 40)
        
        # Signal diversity score (max 30 points)
        unique_signals = len(set(signal[0] for signal in signals))
        signal_score = min(unique_signals * 5, 30)
        
        # Temporal validation score (max 30 points)
        validation_scores = [entry[4] for entry in entries if entry[4] is not None]
        if validation_scores:
            avg_validation = sum(validation_scores) / len(validation_scores)
            validation_score = avg_validation * 30
        else:
            validation_score = 15  # Default moderate score
        
        return round(entry_score + signal_score + validation_score, 1)
    
    def _analyze_timezone_consistency(self, entries: List) -> dict:
        """Analyze timezone consistency in entries"""
        if not entries:
            return {"consistent": True, "issues": []}
        
        # This would be expanded with actual timezone analysis
        # For now, return a simple consistency check
        return {
            "consistent": True,
            "total_entries": len(entries),
            "timezone_changes": 0,
            "avg_validation_score": 0.8
        }
    
    def get_activity_patterns(self, user_id: str, days_back: int = 30) -> dict:
        """Get comprehensive activity patterns for a user"""
        import sqlite3
        from collections import defaultdict
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Get entries with temporal information
            cursor.execute('''
                SELECT timestamp, 
                       COALESCE(local_timestamp, timestamp) as display_time,
                       strftime('%H', COALESCE(local_timestamp, timestamp)) as hour,
                       strftime('%w', COALESCE(local_timestamp, timestamp)) as day_of_week,
                       temporal_validation_score
                FROM messages 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            ''', (user_id, cutoff_date.isoformat()))
            
            entries = cursor.fetchall()
            
            # Analyze patterns
            hourly_distribution = defaultdict(int)
            daily_distribution = defaultdict(int)
            
            for entry in entries:
                if entry[2]:  # hour
                    hourly_distribution[int(entry[2])] += 1
                if entry[3]:  # day_of_week
                    daily_distribution[int(entry[3])] += 1
            
            return {
                "period_days": days_back,
                "total_entries": len(entries),
                "hourly_distribution": dict(hourly_distribution),
                "daily_distribution": dict(daily_distribution),
                "most_active_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
                "most_active_day": max(daily_distribution.items(), key=lambda x: x[1])[0] if daily_distribution else None,
                "activity_consistency": len(set(hourly_distribution.keys())) / 24.0,  # How spread out activity is
            }
            
        except Exception as e:
            print(f"Error analyzing activity patterns: {e}")
            return {"error": str(e)}
        finally:
            conn.close()


class TemporalCorrelationDetector:
    """Detects temporal correlations and patterns in user data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def detect_tag_time_correlations(self, user_id: str, days_back: int = 30, min_correlation: float = 0.3) -> dict:
        """Detect correlations between tags and time periods"""
        import sqlite3
        from collections import defaultdict
        import math
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Get tag usage by hour
            cursor.execute('''
                SELECT 
                    t.name as tag_name,
                    strftime('%H', COALESCE(m.local_timestamp, m.timestamp)) as hour,
                    COUNT(*) as usage_count,
                    AVG(et.confidence) as avg_confidence
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                JOIN messages m ON et.message_id = m.id
                WHERE m.user_id = ? AND m.timestamp >= ?
                GROUP BY t.name, hour
                HAVING usage_count >= 2
                ORDER BY t.name, hour
            ''', (user_id, cutoff_date.isoformat()))
            
            tag_hour_data = cursor.fetchall()
            
            # Calculate correlations
            correlations = {}
            tag_totals = defaultdict(int)
            hour_totals = defaultdict(int)
            total_entries = 0
            
            # Build frequency matrices
            for tag_name, hour, count, confidence in tag_hour_data:
                if tag_name not in correlations:
                    correlations[tag_name] = {}
                correlations[tag_name][hour] = {
                    'count': count,
                    'confidence': round(confidence, 3)
                }
                tag_totals[tag_name] += count
                hour_totals[hour] += count
                total_entries += count
            
            # Calculate correlation strength for each tag-hour pair
            correlation_results = {}
            for tag_name, hour_data in correlations.items():
                correlation_results[tag_name] = {
                    'total_usage': tag_totals[tag_name],
                    'hourly_distribution': hour_data,
                    'peak_hours': [],
                    'correlation_strength': 0.0
                }
                
                # Find peak hours (hours with above-average usage for this tag)
                tag_total = tag_totals[tag_name]
                expected_per_hour = tag_total / 24
                
                peak_hours = []
                correlation_sum = 0
                
                for hour, data in hour_data.items():
                    count = data['count']
                    # Calculate correlation strength using Chi-square-like metric
                    expected = expected_per_hour
                    if expected > 0:
                        correlation = (count - expected) / expected
                        if correlation > min_correlation:
                            peak_hours.append({
                                'hour': int(hour),
                                'count': count,
                                'correlation': round(correlation, 3),
                                'percentage': round((count / tag_total) * 100, 1)
                            })
                        correlation_sum += abs(correlation)
                
                correlation_results[tag_name]['peak_hours'] = sorted(peak_hours, key=lambda x: x['correlation'], reverse=True)
                correlation_results[tag_name]['correlation_strength'] = round(correlation_sum / len(hour_data) if hour_data else 0, 3)
            
            return {
                "user_id": user_id,
                "analysis_period_days": days_back,
                "correlation_threshold": min_correlation,
                "total_entries_analyzed": total_entries,
                "correlations": correlation_results,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error detecting correlations: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def detect_mood_patterns(self, user_id: str, days_back: int = 30) -> dict:
        """Detect mood and emotional patterns over time"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        emotional_tags = ['joy', 'stress', 'gratitude', 'reflection', 'accomplishment', 'tired', 'excited', 'sad', 'angry', 'calm', 'anxious']
        
        try:
            placeholders = ','.join('?' * len(emotional_tags))
            cursor.execute(f'''
                SELECT 
                    t.name as emotion,
                    strftime('%w', COALESCE(m.local_timestamp, m.timestamp)) as day_of_week,
                    strftime('%H', COALESCE(m.local_timestamp, m.timestamp)) as hour,
                    DATE(COALESCE(m.local_timestamp, m.timestamp)) as date,
                    COUNT(*) as frequency,
                    AVG(et.confidence) as avg_confidence
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                JOIN messages m ON et.message_id = m.id
                WHERE m.user_id = ? AND m.timestamp >= ? AND t.name IN ({placeholders})
                GROUP BY t.name, day_of_week, hour, date
                ORDER BY date DESC, hour ASC
            ''', [user_id, cutoff_date.isoformat()] + emotional_tags)
            
            mood_data = cursor.fetchall()
            
            # Analyze patterns
            patterns = {
                "daily_patterns": {},  # Day of week patterns
                "hourly_patterns": {},  # Hour of day patterns
                "temporal_trends": {},  # Date-based trends
                "emotion_correlations": {}  # Emotion co-occurrence
            }
            
            for emotion, dow, hour, date, frequency, confidence in mood_data:
                # Daily patterns
                if emotion not in patterns["daily_patterns"]:
                    patterns["daily_patterns"][emotion] = {}
                patterns["daily_patterns"][emotion][dow] = patterns["daily_patterns"][emotion].get(dow, 0) + frequency
                
                # Hourly patterns
                if emotion not in patterns["hourly_patterns"]:
                    patterns["hourly_patterns"][emotion] = {}
                patterns["hourly_patterns"][emotion][hour] = patterns["hourly_patterns"][emotion].get(hour, 0) + frequency
                
                # Temporal trends
                if emotion not in patterns["temporal_trends"]:
                    patterns["temporal_trends"][emotion] = []
                patterns["temporal_trends"][emotion].append({
                    "date": date,
                    "frequency": frequency,
                    "confidence": round(confidence, 3)
                })
            
            # Calculate dominant patterns
            insights = []
            for emotion, daily_data in patterns["daily_patterns"].items():
                if daily_data:
                    peak_day = max(daily_data.items(), key=lambda x: x[1])
                    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                    insights.append({
                        "emotion": emotion,
                        "pattern_type": "daily",
                        "peak_day": day_names[int(peak_day[0])],
                        "frequency": peak_day[1],
                        "insight": f"{emotion.title()} peaks on {day_names[int(peak_day[0])]}"
                    })
            
            for emotion, hourly_data in patterns["hourly_patterns"].items():
                if hourly_data:
                    peak_hour = max(hourly_data.items(), key=lambda x: x[1])
                    insights.append({
                        "emotion": emotion,
                        "pattern_type": "hourly", 
                        "peak_hour": f"{peak_hour[0]}:00",
                        "frequency": peak_hour[1],
                        "insight": f"{emotion.title()} peaks around {peak_hour[0]}:00"
                    })
            
            return {
                "user_id": user_id,
                "analysis_period_days": days_back,
                "mood_patterns": patterns,
                "insights": insights,
                "emotions_analyzed": emotional_tags,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error detecting mood patterns: {e}")
            return {"error": str(e)}
        finally:
            conn.close()


class UserActivityPatternRecognizer:
    """Recognizes and analyzes user activity patterns for personalized insights"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def analyze_writing_patterns(self, user_id: str, days_back: int = 30) -> dict:
        """Analyze writing patterns and habits"""
        import sqlite3
        from collections import defaultdict
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            cursor.execute('''
                SELECT 
                    DATE(COALESCE(local_timestamp, timestamp)) as date,
                    strftime('%H', COALESCE(local_timestamp, timestamp)) as hour,
                    strftime('%w', COALESCE(local_timestamp, timestamp)) as day_of_week,
                    LENGTH(content) as content_length,
                    temporal_validation_score,
                    COUNT(*) as entry_count
                FROM messages 
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY date, hour, day_of_week
                ORDER BY date DESC
            ''', (user_id, cutoff_date.isoformat()))
            
            data = cursor.fetchall()
            
            # Analyze patterns
            patterns = {
                "consistency_score": 0.0,
                "peak_productivity_hours": [],
                "preferred_days": [],
                "writing_volume_trend": "stable",
                "session_patterns": {},
                "recommendations": []
            }
            
            if not data:
                return patterns
            
            # Calculate consistency (how many days user wrote)
            unique_dates = set(row[0] for row in data)
            total_possible_days = min(days_back, (datetime.now() - datetime.fromisoformat(cutoff_date.isoformat())).days + 1)
            consistency_score = len(unique_dates) / total_possible_days
            patterns["consistency_score"] = round(consistency_score, 3)
            
            # Find peak productivity hours
            hourly_activity = defaultdict(int)
            daily_activity = defaultdict(int)
            content_by_hour = defaultdict(list)
            
            for date, hour, dow, length, validation, count in data:
                hourly_activity[hour] += count
                daily_activity[dow] += count
                content_by_hour[hour].append(length)
            
            # Top 3 productive hours
            top_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns["peak_productivity_hours"] = [
                {
                    "hour": f"{hour}:00",
                    "entry_count": count,
                    "avg_length": round(sum(content_by_hour[hour]) / len(content_by_hour[hour]), 1) if content_by_hour[hour] else 0
                } 
                for hour, count in top_hours
            ]
            
            # Preferred days
            day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            top_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            patterns["preferred_days"] = [
                {
                    "day": day_names[int(dow)],
                    "entry_count": count
                }
                for dow, count in top_days
            ]
            
            # Generate recommendations
            if consistency_score < 0.3:
                patterns["recommendations"].append({
                    "type": "consistency",
                    "priority": "high",
                    "message": "Try to write more regularly to build a stronger journaling habit"
                })
            
            if top_hours:
                best_hour = top_hours[0][0]
                patterns["recommendations"].append({
                    "type": "timing",
                    "priority": "medium", 
                    "message": f"Your most productive time is around {best_hour}:00 - consider scheduling important reflections then"
                })
            
            return {
                "user_id": user_id,
                "analysis_period_days": days_back,
                "patterns": patterns,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing writing patterns: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def predict_optimal_timing(self, user_id: str, days_back: int = 30) -> dict:
        """Predict optimal timing for journaling based on historical patterns"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Get historical patterns with quality metrics
            cursor.execute('''
                SELECT 
                    strftime('%H', COALESCE(local_timestamp, timestamp)) as hour,
                    strftime('%w', COALESCE(local_timestamp, timestamp)) as day_of_week,
                    AVG(LENGTH(content)) as avg_length,
                    AVG(temporal_validation_score) as avg_validation,
                    COUNT(*) as frequency
                FROM messages 
                WHERE user_id = ? AND timestamp >= ?
                GROUP BY hour, day_of_week
                HAVING frequency >= 2
                ORDER BY avg_validation DESC, avg_length DESC
            ''', (user_id, cutoff_date.isoformat()))
            
            quality_data = cursor.fetchall()
            
            predictions = {
                "optimal_hours": [],
                "optimal_days": [],
                "quality_indicators": {},
                "confidence": 0.0
            }
            
            if quality_data:
                # Score each time slot based on length, validation, and frequency
                scored_slots = []
                for hour, dow, avg_length, avg_validation, frequency in quality_data:
                    # Composite score: weighted combination of metrics
                    length_score = min(avg_length / 200, 1.0)  # Normalize to 0-1
                    validation_score = avg_validation or 0.5
                    frequency_score = min(frequency / 10, 1.0)  # Normalize to 0-1
                    
                    composite_score = (length_score * 0.3 + validation_score * 0.4 + frequency_score * 0.3)
                    
                    scored_slots.append({
                        "hour": int(hour),
                        "day_of_week": int(dow),
                        "score": composite_score,
                        "avg_length": round(avg_length, 1),
                        "avg_validation": round(validation_score, 3),
                        "frequency": frequency
                    })
                
                # Sort by score and get top recommendations
                scored_slots.sort(key=lambda x: x['score'], reverse=True)
                
                # Get top 3 hours and days
                hour_scores = {}
                day_scores = {}
                
                for slot in scored_slots:
                    hour = slot['hour']
                    dow = slot['day_of_week']
                    
                    if hour not in hour_scores:
                        hour_scores[hour] = []
                    hour_scores[hour].append(slot['score'])
                    
                    if dow not in day_scores:
                        day_scores[dow] = []
                    day_scores[dow].append(slot['score'])
                
                # Average scores by hour and day
                hour_averages = {hour: sum(scores)/len(scores) for hour, scores in hour_scores.items()}
                day_averages = {dow: sum(scores)/len(scores) for dow, scores in day_scores.items()}
                
                top_hours = sorted(hour_averages.items(), key=lambda x: x[1], reverse=True)[:3]
                top_days = sorted(day_averages.items(), key=lambda x: x[1], reverse=True)[:3]
                
                day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                
                predictions["optimal_hours"] = [
                    {"hour": f"{hour}:00", "quality_score": round(score, 3)}
                    for hour, score in top_hours
                ]
                
                predictions["optimal_days"] = [
                    {"day": day_names[dow], "quality_score": round(score, 3)}
                    for dow, score in top_days
                ]
                
                predictions["confidence"] = round(len(quality_data) / (days_back * 0.5), 2)  # Confidence based on data availability
            
            return {
                "user_id": user_id,
                "predictions": predictions,
                "data_points_analyzed": len(quality_data),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error predicting optimal timing: {e}")
            return {"error": str(e)}
        finally:
            conn.close()


def create_temporal_tables(db_path: str) -> None:
    """Create tables for temporal awareness data"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create temporal signals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS temporal_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            entry_id TEXT,
            signal_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            text_span TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create temporal states table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS temporal_states (
            user_id TEXT PRIMARY KEY,
            current_timezone TEXT DEFAULT 'UTC',
            last_signal_type TEXT,
            last_signal_time TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()