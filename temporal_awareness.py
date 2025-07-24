"""
Comprehensive Temporal Awareness System for Journaling Backend
Provides intelligent temporal signal detection, boundary management, and contextual insights.
"""

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Any
import pytz
import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    DAY_START = "day_start"
    DAY_END = "day_end"
    WEEK_START = "week_start"
    WEEK_END = "week_end"
    MONTH_START = "month_start"
    MONTH_END = "month_end"
    YEAR_START = "year_start"
    YEAR_END = "year_end"
    PERIOD_REFLECTION = "period_reflection"
    TRANSITION = "transition"

@dataclass
class TemporalSignal:
    signal_type: SignalType
    confidence: float
    detected_text: str
    signal_timestamp: datetime
    entry_id: Optional[int] = None
    metadata: Optional[Dict] = None

@dataclass
class TemporalState:
    user_id: str
    last_day_start: Optional[datetime] = None
    last_day_end: Optional[datetime] = None
    last_week_start: Optional[datetime] = None
    last_week_end: Optional[datetime] = None
    last_month_start: Optional[datetime] = None
    last_month_end: Optional[datetime] = None
    last_year_start: Optional[datetime] = None
    last_year_end: Optional[datetime] = None
    timezone: str = "America/Chicago"
    updated_at: Optional[datetime] = None

class TemporalSignalDetector:
    """NLP-based temporal signal detection with confidence scoring"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[SignalType, List[Dict]]:
        """Initialize temporal language patterns with confidence weights"""
        return {
            SignalType.DAY_START: [
                {"pattern": r"starting?\s+(?:my|the|this)\s+day", "confidence": 0.9},
                {"pattern": r"(?:good\s+)?morning(?:\s+thoughts?|\s+reflection)?", "confidence": 0.85},
                {"pattern": r"waking?\s+up(?:\s+to)?", "confidence": 0.8},
                {"pattern": r"begin(?:ning)?\s+(?:today|this\s+day)", "confidence": 0.9},
                {"pattern": r"first\s+thing\s+(?:this\s+)?morning", "confidence": 0.85},
                {"pattern": r"as\s+(?:the\s+)?day\s+begins?", "confidence": 0.8},
                {"pattern": r"dawn(?:ing)?|sunrise", "confidence": 0.75},
                {"pattern": r"today\s+I?\s*(?:will|am|feel|want)", "confidence": 0.7},
            ],
            
            SignalType.DAY_END: [
                {"pattern": r"ending?\s+(?:my|the|this)\s+day", "confidence": 0.9},
                {"pattern": r"(?:good\s+)?night(?:\s+thoughts?|\s+reflection)?", "confidence": 0.85},
                {"pattern": r"going\s+to\s+(?:bed|sleep)", "confidence": 0.8},
                {"pattern": r"reflect(?:ing)?\s+on\s+(?:today|this\s+day)", "confidence": 0.9},
                {"pattern": r"(?:day|today)\s+(?:is\s+)?(?:coming\s+to\s+an?\s+)?end", "confidence": 0.85},
                {"pattern": r"as\s+(?:the\s+)?day\s+(?:comes\s+to\s+an?\s+)?end", "confidence": 0.8},
                {"pattern": r"sunset|dusk|evening", "confidence": 0.75},
                {"pattern": r"looking\s+back\s+on\s+today", "confidence": 0.8},
            ],
            
            SignalType.WEEK_START: [
                {"pattern": r"starting?\s+(?:a\s+new\s+|this\s+)?week", "confidence": 0.9},
                {"pattern": r"monday(?:\s+morning)?(?:\s+thoughts?)?", "confidence": 0.8},
                {"pattern": r"begin(?:ning)?\s+(?:the\s+)?week", "confidence": 0.85},
                {"pattern": r"(?:new\s+)?week\s+ahead", "confidence": 0.8},
                {"pattern": r"weekly?\s+(?:planning|goals?|intentions?)", "confidence": 0.75},
                {"pattern": r"fresh\s+start.*week", "confidence": 0.7},
            ],
            
            SignalType.WEEK_END: [
                {"pattern": r"ending?\s+(?:the\s+|this\s+)?week", "confidence": 0.9},
                {"pattern": r"week(?:end)?\s+reflection", "confidence": 0.85},
                {"pattern": r"(?:this\s+)?week\s+(?:is\s+)?(?:coming\s+to\s+an?\s+)?end", "confidence": 0.8},
                {"pattern": r"sunday(?:\s+evening)?(?:\s+thoughts?)?", "confidence": 0.75},
                {"pattern": r"looking\s+back\s+on\s+(?:this\s+)?week", "confidence": 0.8},
                {"pattern": r"weekly?\s+(?:review|summary)", "confidence": 0.75},
            ],
            
            SignalType.MONTH_START: [
                {"pattern": r"starting?\s+(?:a\s+new\s+|this\s+)?month", "confidence": 0.9},
                {"pattern": r"first\s+of\s+(?:the\s+month|[A-Z][a-z]+)", "confidence": 0.85},
                {"pattern": r"begin(?:ning)?\s+(?:the\s+)?month", "confidence": 0.8},
                {"pattern": r"(?:new\s+)?month\s+ahead", "confidence": 0.75},
                {"pattern": r"monthly?\s+(?:planning|goals?|intentions?)", "confidence": 0.7},
            ],
            
            SignalType.MONTH_END: [
                {"pattern": r"ending?\s+(?:the\s+|this\s+)?month", "confidence": 0.9},
                {"pattern": r"month(?:ly)?\s+reflection", "confidence": 0.85},
                {"pattern": r"(?:this\s+)?month\s+(?:is\s+)?(?:coming\s+to\s+an?\s+)?end", "confidence": 0.8},
                {"pattern": r"last\s+day\s+of\s+(?:the\s+month|[A-Z][a-z]+)", "confidence": 0.8},
                {"pattern": r"looking\s+back\s+on\s+(?:this\s+)?month", "confidence": 0.75},
                {"pattern": r"monthly?\s+(?:review|summary)", "confidence": 0.7},
            ],
            
            SignalType.YEAR_START: [
                {"pattern": r"starting?\s+(?:a\s+new\s+|this\s+)?year", "confidence": 0.95},
                {"pattern": r"new\s+year(?:'s)?(?:\s+(?:day|eve|resolution))?", "confidence": 0.9},
                {"pattern": r"january\s+1st?", "confidence": 0.85},
                {"pattern": r"begin(?:ning)?\s+(?:the\s+)?year", "confidence": 0.8},
                {"pattern": r"annual\s+(?:planning|goals?|resolutions?)", "confidence": 0.75},
            ],
            
            SignalType.YEAR_END: [
                {"pattern": r"ending?\s+(?:the\s+|this\s+)?year", "confidence": 0.95},
                {"pattern": r"year(?:ly)?\s+reflection", "confidence": 0.9},
                {"pattern": r"december\s+31st?", "confidence": 0.85},
                {"pattern": r"(?:this\s+)?year\s+(?:is\s+)?(?:coming\s+to\s+an?\s+)?end", "confidence": 0.8},
                {"pattern": r"looking\s+back\s+on\s+(?:this\s+)?year", "confidence": 0.8},
                {"pattern": r"annual\s+(?:review|summary)", "confidence": 0.75},
            ],
            
            SignalType.PERIOD_REFLECTION: [
                {"pattern": r"reflect(?:ing|ion)\s+on", "confidence": 0.8},
                {"pattern": r"looking\s+back", "confidence": 0.7},
                {"pattern": r"(?:time\s+to\s+)?(?:pause|stop)\s+and\s+(?:think|reflect)", "confidence": 0.75},
                {"pattern": r"taking\s+stock", "confidence": 0.7},
                {"pattern": r"where\s+(?:I\s+)?(?:am|stand)\s+(?:right\s+)?now", "confidence": 0.65},
            ],
            
            SignalType.TRANSITION: [
                {"pattern": r"transition(?:ing)?", "confidence": 0.8},
                {"pattern": r"moving\s+(?:on|forward|ahead)", "confidence": 0.7},
                {"pattern": r"(?:big|major|significant)\s+(?:change|shift)", "confidence": 0.75},
                {"pattern": r"new\s+(?:chapter|phase|stage)", "confidence": 0.7},
                {"pattern": r"turning\s+point", "confidence": 0.8},
                {"pattern": r"crossroads", "confidence": 0.75},
            ]
        }
    
    def detect_signals(self, content: str, entry_timestamp: datetime = None) -> List[TemporalSignal]:
        """Detect temporal signals in text content"""
        if entry_timestamp is None:
            entry_timestamp = datetime.now()
            
        signals = []
        content_lower = content.lower()
        
        for signal_type, patterns in self.patterns.items():
            for pattern_data in patterns:
                pattern = pattern_data["pattern"]
                base_confidence = pattern_data["confidence"]
                
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 20)
                    context = content[start:end].strip()
                    
                    # Adjust confidence based on context and timing
                    confidence = self._adjust_confidence(
                        base_confidence, 
                        signal_type, 
                        context, 
                        entry_timestamp
                    )
                    
                    if confidence > 0.5:  # Threshold for signal detection
                        signals.append(TemporalSignal(
                            signal_type=signal_type,
                            confidence=confidence,
                            detected_text=match.group(),
                            signal_timestamp=entry_timestamp,
                            metadata={
                                "context": context,
                                "match_position": (match.start(), match.end()),
                                "pattern_used": pattern
                            }
                        ))
        
        # Remove duplicate signals of the same type, keeping highest confidence
        unique_signals = {}
        for signal in signals:
            key = signal.signal_type
            if key not in unique_signals or signal.confidence > unique_signals[key].confidence:
                unique_signals[key] = signal
        
        return list(unique_signals.values())
    
    def _adjust_confidence(self, base_confidence: float, signal_type: SignalType, 
                          context: str, timestamp: datetime) -> float:
        """Adjust confidence based on temporal context and timing"""
        confidence = base_confidence
        
        # Time-based adjustments
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        day_of_month = timestamp.day
        month = timestamp.month
        
        # Boost confidence for time-appropriate signals
        if signal_type in [SignalType.DAY_START] and 5 <= hour <= 11:
            confidence += 0.1
        elif signal_type in [SignalType.DAY_END] and 18 <= hour <= 23:
            confidence += 0.1
        elif signal_type in [SignalType.WEEK_START] and weekday == 0:  # Monday
            confidence += 0.15
        elif signal_type in [SignalType.WEEK_END] and weekday == 6:  # Sunday
            confidence += 0.15
        elif signal_type in [SignalType.MONTH_START] and day_of_month <= 3:
            confidence += 0.2
        elif signal_type in [SignalType.MONTH_END] and day_of_month >= 28:
            confidence += 0.2
        elif signal_type in [SignalType.YEAR_START] and month == 1 and day_of_month <= 7:
            confidence += 0.25
        elif signal_type in [SignalType.YEAR_END] and month == 12 and day_of_month >= 25:
            confidence += 0.25
        
        # Context-based adjustments
        context_lower = context.lower()
        
        # Boost for emotional/reflective language
        emotional_indicators = ["feel", "emotion", "heart", "soul", "spirit", "grateful", "blessed"]
        if any(word in context_lower for word in emotional_indicators):
            confidence += 0.05
        
        # Boost for planning/intention language
        planning_indicators = ["plan", "goal", "intention", "hope", "want", "will", "going to"]
        if any(word in context_lower for word in planning_indicators):
            confidence += 0.05
        
        # Reduce for uncertain language
        uncertain_indicators = ["maybe", "perhaps", "might", "possibly", "unsure"]
        if any(word in context_lower for word in uncertain_indicators):
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))

class TemporalStateManager:
    """Manages temporal state and boundary tracking for users"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def get_user_timezone(self, user_id: str) -> str:
        """Get user's timezone preference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timezone FROM temporal_state 
            WHERE user_id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else "America/Chicago"
    
    def get_temporal_state(self, user_id: str) -> TemporalState:
        """Get current temporal state for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, last_day_start, last_day_end, last_week_start, last_week_end,
                   last_month_start, last_month_end, last_year_start, last_year_end,
                   timezone, updated_at
            FROM temporal_state 
            WHERE user_id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return TemporalState(
                user_id=result[0],
                last_day_start=datetime.fromisoformat(result[1]) if result[1] else None,
                last_day_end=datetime.fromisoformat(result[2]) if result[2] else None,
                last_week_start=datetime.fromisoformat(result[3]) if result[3] else None,
                last_week_end=datetime.fromisoformat(result[4]) if result[4] else None,
                last_month_start=datetime.fromisoformat(result[5]) if result[5] else None,
                last_month_end=datetime.fromisoformat(result[6]) if result[6] else None,
                last_year_start=datetime.fromisoformat(result[7]) if result[7] else None,
                last_year_end=datetime.fromisoformat(result[8]) if result[8] else None,
                timezone=result[9],
                updated_at=datetime.fromisoformat(result[10]) if result[10] else None,
            )
        else:
            # Create default state for new user
            return TemporalState(user_id=user_id)
    
    def update_temporal_state(self, user_id: str, signal: TemporalSignal) -> None:
        """Update temporal state based on detected signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current state
        state = self.get_temporal_state(user_id)
        
        # Update based on signal type
        update_fields = {}
        
        if signal.signal_type == SignalType.DAY_START:
            update_fields['last_day_start'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.DAY_END:
            update_fields['last_day_end'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.WEEK_START:
            update_fields['last_week_start'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.WEEK_END:
            update_fields['last_week_end'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.MONTH_START:
            update_fields['last_month_start'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.MONTH_END:
            update_fields['last_month_end'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.YEAR_START:
            update_fields['last_year_start'] = signal.signal_timestamp.isoformat()
        elif signal.signal_type == SignalType.YEAR_END:
            update_fields['last_year_end'] = signal.signal_timestamp.isoformat()
        
        if update_fields:
            # Check if user exists in temporal_state
            cursor.execute("SELECT user_id FROM temporal_state WHERE user_id = ?", (user_id,))
            if cursor.fetchone():
                # Update existing record
                set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
                set_clause += ", updated_at = ?"
                values = list(update_fields.values()) + [datetime.now().isoformat(), user_id]
                
                cursor.execute(f"""
                    UPDATE temporal_state 
                    SET {set_clause}
                    WHERE user_id = ?
                """, values)
            else:
                # Insert new record
                update_fields['user_id'] = user_id
                update_fields['timezone'] = state.timezone
                update_fields['updated_at'] = datetime.now().isoformat()
                
                columns = ", ".join(update_fields.keys())
                placeholders = ", ".join(["?" for _ in update_fields])
                
                cursor.execute(f"""
                    INSERT INTO temporal_state ({columns})
                    VALUES ({placeholders})
                """, list(update_fields.values()))
        
        conn.commit()
        conn.close()
    
    def record_temporal_signal(self, user_id: str, signal: TemporalSignal, entry_id: int = None) -> None:
        """Record a temporal signal in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO temporal_signals 
            (user_id, signal_type, signal_timestamp, entry_id, timezone, confidence, detected_text, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            signal.signal_type.value,
            signal.signal_timestamp.isoformat(),
            entry_id,
            self.get_user_timezone(user_id),
            signal.confidence,
            signal.detected_text,
            json.dumps(signal.metadata) if signal.metadata else None,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Update temporal state
        self.update_temporal_state(user_id, signal)

class TemporalSummaryGenerator:
    """Generates intelligent period summaries with temporal awareness"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.state_manager = TemporalStateManager(db_path)
    
    def generate_period_summary(self, user_id: str, period_type: str, 
                              start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate a comprehensive summary for a temporal period"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get entries for the period
        cursor.execute("""
            SELECT m.*, et.tag_id, t.name as tag_name, t.category as tag_category
            FROM messages m
            LEFT JOIN entry_tags et ON m.id = et.message_id
            LEFT JOIN tags t ON et.tag_id = t.id
            WHERE m.user_id = ? AND m.timestamp BETWEEN ? AND ?
            ORDER BY m.timestamp
        """, (user_id, start_date.isoformat(), end_date.isoformat()))
        
        entries_data = cursor.fetchall()
        
        # Get temporal signals for the period
        cursor.execute("""
            SELECT signal_type, confidence, detected_text, signal_timestamp, metadata
            FROM temporal_signals
            WHERE user_id = ? AND signal_timestamp BETWEEN ? AND ?
            ORDER BY signal_timestamp
        """, (user_id, start_date.isoformat(), end_date.isoformat()))
        
        signals_data = cursor.fetchall()
        conn.close()
        
        # Process entries and group by message
        entries_by_id = {}
        for row in entries_data:
            entry_id = row['id']
            if entry_id not in entries_by_id:
                entries_by_id[entry_id] = {
                    'id': entry_id,
                    'content': row['content'],
                    'timestamp': row['timestamp'],
                    'tags': []
                }
            if row['tag_name']:
                entries_by_id[entry_id]['tags'].append({
                    'name': row['tag_name'],
                    'category': row['tag_category']
                })
        
        entries = list(entries_by_id.values())
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(signals_data, period_type)
        
        # Generate theme analysis
        theme_analysis = self._analyze_themes(entries)
        
        # Generate emotional journey
        emotional_journey = self._analyze_emotional_journey(entries)
        
        # Generate growth insights
        growth_insights = self._generate_growth_insights(entries, temporal_analysis)
        
        # Create sacred summary
        sacred_summary = self._generate_sacred_summary(
            period_type, entries, temporal_analysis, theme_analysis, emotional_journey
        )
        
        return {
            "period_type": period_type,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "entry_count": len(entries),
            "temporal_signals": len(signals_data),
            "temporal_analysis": temporal_analysis,
            "theme_analysis": theme_analysis,
            "emotional_journey": emotional_journey,
            "growth_insights": growth_insights,
            "sacred_summary": sacred_summary,
            "wisdom_insights": self._generate_wisdom_insights(entries, temporal_analysis)
        }
    
    def _analyze_temporal_patterns(self, signals_data: List, period_type: str) -> Dict[str, Any]:
        """Analyze temporal signal patterns"""
        signal_counts = {}
        signal_confidences = {}
        
        for signal in signals_data:
            signal_type = signal['signal_type']
            confidence = signal['confidence']
            
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            if signal_type not in signal_confidences:
                signal_confidences[signal_type] = []
            signal_confidences[signal_type].append(confidence)
        
        # Calculate average confidences
        avg_confidences = {
            signal_type: sum(confidences) / len(confidences)
            for signal_type, confidences in signal_confidences.items()
        }
        
        return {
            "signal_counts": signal_counts,
            "average_confidences": avg_confidences,
            "dominant_signals": sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "temporal_awareness_score": sum(avg_confidences.values()) / len(avg_confidences) if avg_confidences else 0
        }
    
    def _analyze_themes(self, entries: List[Dict]) -> Dict[str, Any]:
        """Analyze dominant themes across entries"""
        tag_counts = {}
        tag_categories = {}
        
        for entry in entries:
            for tag in entry['tags']:
                tag_name = tag['name']
                tag_category = tag['category']
                
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
                tag_categories[tag_category] = tag_categories.get(tag_category, 0) + 1
        
        dominant_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        dominant_categories = sorted(tag_categories.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "tag_counts": tag_counts,
            "category_counts": tag_categories,
            "dominant_tags": dominant_tags,
            "dominant_categories": dominant_categories,
            "theme_diversity": len(tag_counts),
            "category_balance": len(tag_categories)
        }
    
    def _analyze_emotional_journey(self, entries: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns and journey"""
        emotional_tags = ['joy', 'gratitude', 'stress', 'reflection', 'accomplishment']
        emotional_timeline = []
        emotional_counts = {}
        
        for entry in entries:
            entry_emotions = []
            for tag in entry['tags']:
                if tag['name'] in emotional_tags:
                    entry_emotions.append(tag['name'])
                    emotional_counts[tag['name']] = emotional_counts.get(tag['name'], 0) + 1
            
            if entry_emotions:
                emotional_timeline.append({
                    'timestamp': entry['timestamp'],
                    'emotions': entry_emotions,
                    'dominant_emotion': entry_emotions[0] if entry_emotions else None
                })
        
        return {
            "emotional_counts": emotional_counts,
            "emotional_timeline": emotional_timeline,
            "dominant_emotions": sorted(emotional_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "emotional_range": len(emotional_counts),
            "emotional_entries": len(emotional_timeline)
        }
    
    def _generate_growth_insights(self, entries: List[Dict], temporal_analysis: Dict) -> List[str]:
        """Generate insights about growth and patterns"""
        insights = []
        
        # Temporal awareness insight
        temporal_score = temporal_analysis.get('temporal_awareness_score', 0)
        if temporal_score > 0.7:
            insights.append("You demonstrate strong temporal awareness, marking transitions and boundaries clearly.")
        elif temporal_score > 0.4:
            insights.append("Your temporal awareness is developing - consider noting more transitions and milestones.")
        else:
            insights.append("Cultivating greater awareness of temporal boundaries could enhance your reflection practice.")
        
        # Entry frequency insight
        if len(entries) > 0:
            avg_content_length = sum(len(entry['content']) for entry in entries) / len(entries)
            if avg_content_length > 200:
                insights.append("Your entries show depth and thoughtfulness in expression.")
            else:
                insights.append("Consider expanding your reflections for deeper insights.")
        
        return insights
    
    def _generate_sacred_summary(self, period_type: str, entries: List[Dict], 
                                temporal_analysis: Dict, theme_analysis: Dict, 
                                emotional_journey: Dict) -> str:
        """Generate a sacred, poetic summary of the period"""
        entry_count = len(entries)
        dominant_themes = theme_analysis.get('dominant_tags', [])
        dominant_emotions = emotional_journey.get('dominant_emotions', [])
        temporal_signals = temporal_analysis.get('signal_counts', {})
        
        # Build sacred narrative
        summary_parts = []
        
        # Opening with temporal context
        if period_type == "daily":
            summary_parts.append("In the sacred container of this day")
        elif period_type == "weekly":
            summary_parts.append("Through the flowing rhythm of this week")
        elif period_type == "monthly":
            summary_parts.append("Within the lunar cycle of this month")
        else:
            summary_parts.append(f"In the {period_type} period of reflection")
        
        # Entry count and engagement
        if entry_count > 0:
            if entry_count == 1:
                summary_parts.append("you offered one sacred reflection to the universe")
            else:
                summary_parts.append(f"you wove {entry_count} threads of consciousness into the tapestry of time")
        
        # Dominant themes
        if dominant_themes:
            theme_names = [theme[0] for theme in dominant_themes[:3]]
            if len(theme_names) == 1:
                summary_parts.append(f"The theme of '{theme_names[0]}' illuminated your path")
            elif len(theme_names) == 2:
                summary_parts.append(f"The themes of '{theme_names[0]}' and '{theme_names[1]}' danced through your awareness")
            else:
                summary_parts.append(f"The themes of '{theme_names[0]}', '{theme_names[1]}', and '{theme_names[2]}' formed a trinity of exploration")
        
        # Emotional essence
        if dominant_emotions:
            emotion_names = [emotion[0] for emotion in dominant_emotions[:2]]
            if len(emotion_names) == 1:
                summary_parts.append(f"Your heart resonated most deeply with {emotion_names[0]}")
            else:
                summary_parts.append(f"Your emotional landscape flowed between {emotion_names[0]} and {emotion_names[1]}")
        
        # Temporal awareness
        signal_count = sum(temporal_signals.values())
        if signal_count > 0:
            summary_parts.append(f"marking {signal_count} sacred boundaries and transitions with conscious awareness")
        
        # Closing blessing
        summary_parts.append("May these reflections guide your continuing journey of growth and wisdom.")
        
        return ". ".join(summary_parts) + "."
    
    def _generate_wisdom_insights(self, entries: List[Dict], temporal_analysis: Dict) -> List[str]:
        """Generate wisdom insights from the period"""
        insights = []
        
        # Pattern recognition insights
        if temporal_analysis.get('temporal_awareness_score', 0) > 0.6:
            insights.append("Your conscious marking of temporal boundaries shows developing wisdom in the art of presence.")
        
        # Reflection depth insights
        if len(entries) > 0:
            total_content = sum(len(entry['content']) for entry in entries)
            if total_content > 1000:
                insights.append("The depth of your reflections reveals a soul committed to growth and understanding.")
        
        return insights

def create_temporal_tables(db_path: str) -> None:
    """Create temporal awareness tables in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create temporal_signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporal_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            signal_timestamp DATETIME NOT NULL,
            entry_id INTEGER,
            timezone TEXT DEFAULT 'America/Chicago',
            confidence FLOAT DEFAULT 0.0,
            detected_text TEXT,
            metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entry_id) REFERENCES messages(id) ON DELETE CASCADE
        )
    """)
    
    # Create temporal_state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporal_state (
            user_id TEXT PRIMARY KEY,
            last_day_start DATETIME,
            last_day_end DATETIME,
            last_week_start DATETIME,
            last_week_end DATETIME,
            last_month_start DATETIME,
            last_month_end DATETIME,
            last_year_start DATETIME,
            last_year_end DATETIME,
            timezone TEXT DEFAULT 'America/Chicago',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add temporal_signals JSON field to messages table if it doesn't exist
    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN temporal_signals JSON")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_user_id ON temporal_signals(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_type ON temporal_signals(signal_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_timestamp ON temporal_signals(signal_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_user_timestamp ON temporal_signals(user_id, signal_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_state_user_id ON temporal_state(user_id)")
    
    conn.commit()
    conn.close()
    
    logger.info("✅ Temporal awareness tables created successfully")