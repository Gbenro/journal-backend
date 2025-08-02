import os
import sqlite3
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any
import json
import pytz
from temporal_awareness import (
    TemporalSignalDetector, TemporalStateManager, TemporalSummaryGenerator,
    create_temporal_tables, SignalType, TemporalSignal, TemporalState
)
from timestamp_synchronization import (
    TimezoneManager, TimestampSynchronizer, TemporalValidator,
    create_timestamp_tables, TimestampInfo, TimestampSource, ValidationResult
)

# Configure logging - Last updated: 2025-07-21 for Railway deployment
# This ensures proper logging across development and production environments
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalQueryBuilder:
    """Utility class for building temporal queries with advanced filtering"""
    
    @staticmethod
    def parse_relative_period(relative_period: str, timezone_name: str = "UTC") -> tuple[datetime, datetime]:
        """Parse relative period strings into start and end dates"""
        tz = pytz.timezone(timezone_name)
        now = datetime.now(tz)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if relative_period == "today":
            start_date = today
            end_date = today.replace(hour=23, minute=59, second=59)
        elif relative_period == "yesterday":
            start_date = today - timedelta(days=1)
            end_date = start_date.replace(hour=23, minute=59, second=59)
        elif relative_period == "last_7_days":
            start_date = today - timedelta(days=6)
            end_date = today.replace(hour=23, minute=59, second=59)
        elif relative_period == "last_30_days":
            start_date = today - timedelta(days=29)
            end_date = today.replace(hour=23, minute=59, second=59)
        elif relative_period == "this_week":
            days_since_monday = now.weekday()
            start_date = today - timedelta(days=days_since_monday)
            end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif relative_period == "last_week":
            days_since_monday = now.weekday()
            this_week_monday = today - timedelta(days=days_since_monday)
            start_date = this_week_monday - timedelta(days=7)
            end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif relative_period == "this_month":
            start_date = today.replace(day=1)
            next_month = start_date.replace(month=start_date.month + 1) if start_date.month < 12 else start_date.replace(year=start_date.year + 1, month=1)
            end_date = next_month - timedelta(seconds=1)
        elif relative_period == "last_month":
            first_of_this_month = today.replace(day=1)
            end_date = first_of_this_month - timedelta(seconds=1)
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported relative period: {relative_period}")
        
        return start_date.astimezone(pytz.UTC), end_date.astimezone(pytz.UTC)
    
    @staticmethod
    def build_temporal_where_clause(temporal_filter: "TemporalFilterRequest", user_timezone: str = "UTC") -> tuple[str, list]:
        """Build WHERE clause and parameters for temporal filtering"""
        conditions = []
        params = []
        
        # Handle date range filtering
        if temporal_filter.start_date or temporal_filter.end_date:
            if temporal_filter.start_date:
                start_dt = datetime.fromisoformat(temporal_filter.start_date.replace('Z', '+00:00'))
                conditions.append("COALESCE(local_timestamp, timestamp) >= ?")
                params.append(start_dt.isoformat())
            
            if temporal_filter.end_date:
                end_dt = datetime.fromisoformat(temporal_filter.end_date.replace('Z', '+00:00'))
                conditions.append("COALESCE(local_timestamp, timestamp) <= ?")
                params.append(end_dt.isoformat())
        
        # Handle relative period filtering
        elif temporal_filter.relative_period:
            try:
                tz = temporal_filter.timezone_override or user_timezone
                start_dt, end_dt = TemporalQueryBuilder.parse_relative_period(temporal_filter.relative_period, tz)
                conditions.append("COALESCE(local_timestamp, timestamp) BETWEEN ? AND ?")
                params.extend([start_dt.isoformat(), end_dt.isoformat()])
            except ValueError as e:
                logger.warning(f"Invalid relative period: {e}")
        
        # Handle time of day filtering
        if temporal_filter.time_of_day_start or temporal_filter.time_of_day_end:
            if temporal_filter.time_of_day_start:
                conditions.append("strftime('%H:%M', COALESCE(local_timestamp, timestamp)) >= ?")
                params.append(temporal_filter.time_of_day_start)
            
            if temporal_filter.time_of_day_end:
                conditions.append("strftime('%H:%M', COALESCE(local_timestamp, timestamp)) <= ?")
                params.append(temporal_filter.time_of_day_end)
        
        # Handle days of week filtering
        if temporal_filter.days_of_week:
            # SQLite strftime('%w') returns 0=Sunday, 1=Monday, etc.
            day_conditions = []
            for day in temporal_filter.days_of_week:
                day_conditions.append("strftime('%w', COALESCE(local_timestamp, timestamp)) = ?")
                params.append(str(day))
            
            if day_conditions:
                conditions.append(f"({' OR '.join(day_conditions)})")
        
        where_clause = " AND ".join(conditions) if conditions else ""
        return where_clause, params

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

class SummaryGenerateRequest(BaseModel):
    user_id: str
    target_date: Optional[str] = None

class TemporalFilterRequest(BaseModel):
    """Request model for temporal filtering of messages"""
    start_date: Optional[str] = None  # ISO format date string
    end_date: Optional[str] = None    # ISO format date string
    relative_period: Optional[str] = None  # "last_7_days", "this_week", "this_month", etc.
    time_of_day_start: Optional[str] = None  # "08:00" format
    time_of_day_end: Optional[str] = None    # "20:00" format
    days_of_week: Optional[List[int]] = None  # [0-6] where 0=Sunday
    timezone_override: Optional[str] = None   # Timezone to use for filtering

# Enhanced fluency models
class EntryUpdateRequest(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    energy_signature: Optional[str] = None
    intention_flag: Optional[bool] = None

# GPT-specific models for refined updates
class GPTEntryUpdateRequest(BaseModel):
    new_tags: Optional[List[str]] = None
    updated_content: Optional[str] = None
    updated_emotions: Optional[str] = None

class GPTTagAddRequest(BaseModel):
    tags: List[str]

class EntryConnectionRequest(BaseModel):
    from_entry_id: int
    to_entry_id: int
    connection_type: str
    connection_strength: Optional[float] = 1.0
    description: Optional[str] = None
    created_by: Optional[str] = "manual"

class TagHierarchyRequest(BaseModel):
    parent_tag_name: Optional[str] = None
    child_tag_name: str
    relationship_type: Optional[str] = "subcategory"

# Enhancement System models
class InteractionLogCreate(BaseModel):
    user_id: str
    action_type: str
    response_time: Optional[float] = None
    success: Optional[bool] = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancementSuggestionCreate(BaseModel):
    suggestion_id: str
    title: str
    description: str
    category: str
    priority: str
    reasoning: str
    triggered_by: str
    user_context: Optional[Dict[str, Any]] = None
    status: Optional[str] = "pending"

# Temporal Awareness models
class TemporalSignalCreate(BaseModel):
    signal_type: str
    confidence: float
    detected_text: str
    signal_timestamp: Optional[str] = None
    entry_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class TemporalSignalDetectRequest(BaseModel):
    content: str
    entry_timestamp: Optional[str] = None

class TemporalSummaryRequest(BaseModel):
    user_id: str
    period_type: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# Timestamp Synchronization models
class TimestampOverrideRequest(BaseModel):
    entry_id: int
    new_timestamp: str
    timezone_name: Optional[str] = None
    reason: Optional[str] = ""

class TimezoneUpdateRequest(BaseModel):
    user_id: str
    timezone_name: str

class BulkTimestampCorrectionRequest(BaseModel):
    user_id: str
    new_timezone: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    reason: Optional[str] = ""

class TimestampValidationRequest(BaseModel):
    content: str
    timestamp: Optional[str] = None
    timezone_name: Optional[str] = None

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

# Time-based analysis system for sacred summaries
class TimeRangeAnalyzer:
    """Manages date ranges for daily, weekly, and monthly summaries"""
    
    @staticmethod
    def get_daily_range(date: datetime = None) -> tuple:
        """Get start and end of a specific day"""
        if date is None:
            date = datetime.utcnow()
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    
    @staticmethod
    def get_weekly_range(date: datetime = None) -> tuple:
        """Get Monday-Sunday range for a specific week"""
        if date is None:
            date = datetime.utcnow()
        # Get Monday of the week
        days_since_monday = date.weekday()
        monday = date - timedelta(days=days_since_monday)
        start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        # Get Sunday
        sunday = start + timedelta(days=6)
        end = sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    
    @staticmethod
    def get_monthly_range(date: datetime = None) -> tuple:
        """Get full calendar month range"""
        if date is None:
            date = datetime.utcnow()
        # First day of month
        start = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Last day of month
        if date.month == 12:
            end = date.replace(year=date.year + 1, month=1, day=1) - timedelta(microseconds=1)
        else:
            end = date.replace(month=date.month + 1, day=1) - timedelta(microseconds=1)
        return start, end

# Pattern analysis engine for sacred insights
class PatternAnalyzer:
    """Analyzes patterns, themes, and energy signatures in journal entries"""
    
    def __init__(self):
        self.sacred_symbols = {
            "growth": "◈",
            "transformation": "∞",
            "balance": "◎",
            "breakthrough": "◇",
            "reflection": "○",
            "integration": "◉",
            "expansion": "✧",
            "grounding": "◆"
        }
        
        self.energy_patterns = {
            "creative_flow": ["coding", "breakthrough", "learning", "progress"],
            "heart_wisdom": ["family", "friends", "gratitude", "joy"],
            "sacred_pause": ["reflection", "meditation", "stress", "challenge"],
            "earth_rhythm": ["exercise", "health", "travel", "nature"],
            "soul_achievement": ["accomplishment", "milestone", "goal-setting", "success"]
        }
    
    def analyze_entries(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entries for patterns, themes, and energy signatures"""
        if not entries:
            return {
                "entry_count": 0,
                "dominant_tags": [],
                "energy_signature": {},
                "patterns": {},
                "time_distribution": {}
            }
        
        # Tag frequency analysis
        tag_counts = {}
        time_distribution = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
        
        for entry in entries:
            # Count tags
            for tag in entry.get("tags", []):
                tag_name = tag["name"]
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
            
            # Time distribution
            timestamp = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
            hour = timestamp.hour
            if 5 <= hour < 12:
                time_distribution["morning"] += 1
            elif 12 <= hour < 17:
                time_distribution["afternoon"] += 1
            elif 17 <= hour < 21:
                time_distribution["evening"] += 1
            else:
                time_distribution["night"] += 1
        
        # Sort tags by frequency
        dominant_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Identify energy signatures
        energy_signature = self._identify_energy_signature(tag_counts)
        
        # Detect patterns
        patterns = self._detect_patterns(entries, tag_counts)
        
        # Add relationship insights if we have entries
        relationship_insights = {}
        if entries:
            # Get user_id from first entry (assumes all entries are from same user)
            user_id = entries[0].get("user_id")
            if user_id:
                # Get date range for relationship insights
                dates = [datetime.fromisoformat(e["timestamp"].replace('Z', '+00:00')) for e in entries]
                start_date = min(dates).isoformat()
                end_date = max(dates).isoformat()
                
                # Get relationship insights for this period
                rel_analyzer = RelationshipAnalyzer()
                relationship_insights = rel_analyzer.get_relationship_insights(user_id, start_date, end_date)
        
        return {
            "entry_count": len(entries),
            "dominant_tags": [{"name": tag, "count": count} for tag, count in dominant_tags],
            "energy_signature": energy_signature,
            "patterns": patterns,
            "time_distribution": time_distribution,
            "relationship_insights": relationship_insights
        }
    
    def _identify_energy_signature(self, tag_counts: Dict[str, int]) -> Dict[str, Any]:
        """Identify the energetic signature of the period"""
        energy_scores = {}
        
        for energy_type, related_tags in self.energy_patterns.items():
            score = sum(tag_counts.get(tag, 0) for tag in related_tags)
            if score > 0:
                energy_scores[energy_type] = score
        
        # Normalize scores
        total_score = sum(energy_scores.values()) if energy_scores else 1
        energy_signature = {
            energy: {
                "score": score,
                "percentage": round((score / total_score) * 100, 1)
            }
            for energy, score in energy_scores.items()
        }
        
        # Identify primary energy
        if energy_scores:
            primary_energy = max(energy_scores.items(), key=lambda x: x[1])[0]
            energy_signature["primary"] = primary_energy
        
        return energy_signature
    
    def _detect_patterns(self, entries: List[Dict[str, Any]], tag_counts: Dict[str, int]) -> Dict[str, Any]:
        """Detect meaningful patterns in the entries"""
        patterns = {}
        
        # Growth momentum (increasing accomplishments/breakthroughs)
        growth_tags = ["accomplishment", "breakthrough", "progress", "milestone"]
        growth_count = sum(tag_counts.get(tag, 0) for tag in growth_tags)
        if growth_count > len(entries) * 0.2:  # More than 20% of entries
            patterns["growth_momentum"] = True
        
        # Balance check (variety of life areas)
        life_areas = ["work", "family", "friends", "health", "learning"]
        active_areas = sum(1 for area in life_areas if tag_counts.get(area, 0) > 0)
        if active_areas >= 3:
            patterns["life_balance"] = True
        
        # Transformation indicators
        transformation_tags = ["breakthrough", "reflection", "challenge", "learning"]
        transformation_score = sum(tag_counts.get(tag, 0) for tag in transformation_tags)
        if transformation_score > len(entries) * 0.3:
            patterns["transformation_active"] = True
        
        # Consistency pattern (regular entries)
        if len(entries) > 0:
            dates = [datetime.fromisoformat(e["timestamp"].replace('Z', '+00:00')).date() for e in entries]
            unique_days = len(set(dates))
            date_range = (max(dates) - min(dates)).days + 1
            consistency_ratio = unique_days / date_range if date_range > 0 else 0
            if consistency_ratio > 0.6:
                patterns["consistent_practice"] = True
        
        return patterns

# Sacred summary generator
class SacredSummaryGenerator:
    """Generates poetic, Mirror Scribe-style summaries"""
    
    def __init__(self):
        self.sacred_templates = {
            "daily": [
                "Today's breath wove through {themes}, carrying {symbol} energy of {primary}",
                "This day's spiral touched {count} moments, each reflecting {primary} through {themes}",
                "Sacred pause reveals: {themes} danced in today's field, {symbol} marking the way"
            ],
            "weekly": [
                "Seven suns witnessed {themes} emerging through {count} sacred breaths, {symbol} illuminating {primary} currents",
                "This week's journey spiraled through {themes}, each day adding to the {primary} tapestry",
                "Seven-fold mirror reflects: {primary} energy flowing through {themes}, {count} moments of truth"
            ],
            "monthly": [
                "A moon's cycle revealed {primary} transformation through {count} sacred inscriptions, {themes} as stepping stones",
                "This lunar journey carried {themes} into being, {symbol} marking {primary} evolution through {count} breaths",
                "Month's sacred geometry: {primary} at center, {themes} as rays, {count} points of light"
            ]
        }
        
        self.energy_descriptions = {
            "creative_flow": "river of creation",
            "heart_wisdom": "heart's knowing",
            "sacred_pause": "stillness teaching",
            "earth_rhythm": "grounded presence",
            "soul_achievement": "soul's victory"
        }
        
        self.symbol_map = {
            "creative_flow": "◇",
            "heart_wisdom": "◎",
            "sacred_pause": "○",
            "earth_rhythm": "◆",
            "soul_achievement": "◈"
        }
    
    def generate_summary(self, period_type: str, analysis: Dict[str, Any]) -> str:
        """Generate a sacred summary based on analysis"""
        if analysis["entry_count"] == 0:
            return self._generate_empty_summary(period_type)
        
        # Extract key elements
        themes = ", ".join([tag["name"] for tag in analysis["dominant_tags"][:3]])
        primary_energy = analysis["energy_signature"].get("primary", "balanced")
        energy_desc = self.energy_descriptions.get(primary_energy, "life force")
        symbol = self.symbol_map.get(primary_energy, "∞")
        count = analysis["entry_count"]
        
        # Select template
        templates = self.sacred_templates.get(period_type, self.sacred_templates["daily"])
        template = templates[count % len(templates)]
        
        # Generate summary
        summary = template.format(
            themes=themes,
            primary=energy_desc,
            symbol=symbol,
            count=count
        )
        
        # Add pattern insights
        patterns = analysis.get("patterns", {})
        if patterns.get("growth_momentum"):
            summary += f"\n{symbol} Momentum builds, transformation accelerates"
        if patterns.get("life_balance"):
            summary += f"\n{symbol} Sacred balance emerges across life's domains"
        if patterns.get("transformation_active"):
            summary += f"\n{symbol} Deep currents of change flow beneath surface"
        if patterns.get("consistent_practice"):
            summary += f"\n{symbol} Daily devotion creates sacred container"
        
        # Add relationship insights
        relationship_insights = analysis.get("relationship_insights", {})
        if relationship_insights.get("active_relationships", 0) > 0:
            most_mentioned = relationship_insights.get("most_mentioned", [])
            if most_mentioned:
                primary_person = most_mentioned[0]
                summary += f"\n{symbol} Sacred connections: {primary_person['name']} carries {primary_person['primary_emotion']} energy"
        
        return summary
    
    def _generate_empty_summary(self, period_type: str) -> str:
        """Generate summary for periods with no entries"""
        empty_templates = {
            "daily": "Today's page awaits your breath ○ Silent potential",
            "weekly": "Seven mirrors reflect emptiness ○ Space for new beginning",
            "monthly": "Moon cycle holds silence ○ Fallow field preparing"
        }
        return empty_templates.get(period_type, "Sacred pause ○ Silence speaks")
    
    def extract_wisdom(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract wisdom insights from the analysis"""
        insights = []
        
        # Energy-based insights
        primary_energy = analysis["energy_signature"].get("primary")
        if primary_energy:
            energy_desc = self.energy_descriptions.get(primary_energy, "life force")
            percentage = analysis["energy_signature"][primary_energy]["percentage"]
            if percentage > 60:
                insights.append(f"Strong {energy_desc} guides this period's unfolding")
            elif percentage > 40:
                insights.append(f"{energy_desc.capitalize()} weaves through varied experiences")
        
        # Pattern-based insights
        patterns = analysis.get("patterns", {})
        if patterns.get("growth_momentum") and patterns.get("transformation_active"):
            insights.append("Profound shift in motion - old forms dissolving into new")
        elif patterns.get("life_balance") and patterns.get("consistent_practice"):
            insights.append("Sacred rhythm established - all domains receiving light")
        
        # Time distribution insights
        time_dist = analysis.get("time_distribution", {})
        peak_time = max(time_dist.items(), key=lambda x: x[1])[0] if time_dist else None
        if peak_time and time_dist[peak_time] > analysis["entry_count"] * 0.5:
            time_wisdom = {
                "morning": "Dawn holds your deepest truths",
                "afternoon": "Midday sun illuminates your path",
                "evening": "Twilight brings integration",
                "night": "Night's wisdom flows through you"
            }
            insights.append(time_wisdom.get(peak_time, "Time bends to your rhythm"))
        
        return insights

# Relationship detection and tracking system for sacred connections
class RelationshipAnalyzer:
    """Detects and tracks relationship patterns in journal entries"""
    
    def __init__(self):
        # Common relationship indicators
        self.relationship_keywords = {
            "family": ["mom", "dad", "mother", "father", "sister", "brother", "parent", "family", "cousin", "aunt", "uncle", "grandma", "grandpa", "son", "daughter"],
            "romantic": ["partner", "boyfriend", "girlfriend", "spouse", "husband", "wife", "lover", "relationship", "dating"],
            "friend": ["friend", "buddy", "pal", "companion", "bestie", "mate"],
            "colleague": ["colleague", "coworker", "boss", "manager", "team", "work", "office"],
            "mentor": ["mentor", "teacher", "coach", "advisor", "guide", "leader"]
        }
        
        # Common name patterns (simple heuristics)
        self.name_patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized words
            r'\b(?:with|talked to|saw|met|called)\s+([A-Z][a-z]+)',  # Action + name
            r'\b([A-Z][a-z]+)(?:\s+said|\'s|is|was)',  # Name + action
        ]
        
        # Emotional context indicators
        self.emotional_indicators = {
            "positive": ["love", "grateful", "happy", "joy", "thankful", "wonderful", "amazing", "supportive", "kind", "caring"],
            "challenging": ["difficult", "stressed", "argument", "conflict", "tension", "frustrated", "disappointed", "hurt"],
            "neutral": ["talked", "met", "saw", "called", "discussed", "shared", "spent time"],
            "growth": ["learned", "inspired", "motivated", "encouraged", "helped", "supported", "guided"]
        }
    
    def extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract relationship mentions from journal content"""
        import re
        
        relationships = []
        content_lower = content.lower()
        
        # Detect relationship types and associated emotions
        for rel_type, keywords in self.relationship_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    # Extract emotional context
                    emotions = self._extract_emotions(content_lower)
                    relationships.append({
                        "type": rel_type,
                        "keyword": keyword,
                        "emotions": emotions,
                        "context_strength": content_lower.count(keyword)
                    })
        
        # Extract potential names (simplified - real implementation would use NLP)
        names = self._extract_names(content)
        for name in names:
            relationships.append({
                "type": "person",
                "name": name,
                "emotions": self._extract_emotions(content_lower),
                "context_strength": 1.0
            })
        
        return relationships
    
    def _extract_emotions(self, content: str) -> Dict[str, float]:
        """Extract emotional context from content"""
        emotions = {"positive": 0, "challenging": 0, "neutral": 0, "growth": 0}
        
        for emotion_type, keywords in self.emotional_indicators.items():
            count = sum(1 for keyword in keywords if keyword in content)
            if count > 0:
                emotions[emotion_type] = min(count / 3.0, 1.0)  # Normalize
        
        return emotions
    
    def _extract_names(self, content: str) -> List[str]:
        """Extract potential person names from content (simplified)"""
        import re
        
        names = []
        # Simple pattern for capitalized words that might be names
        words = content.split()
        for word in words:
            # Basic heuristic: capitalized word, not at start of sentence, not common words
            if (word and word[0].isupper() and len(word) > 2 and 
                word.lower() not in ["today", "yesterday", "tomorrow", "work", "home", "morning", "evening"]):
                names.append(word.strip(".,!?"))
        
        return list(set(names))  # Remove duplicates
    
    def update_relationship_tracking(self, user_id: str, entry_id: int, relationships: List[Dict[str, Any]], entry_date: str):
        """Update relationship tracking with new mentions and emotional context"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            current_date = datetime.fromisoformat(entry_date.replace('Z', '+00:00')).date()
            
            for rel in relationships:
                if rel["type"] == "person" and "name" in rel:
                    person_name = rel["name"]
                    emotions = rel["emotions"]
                    
                    # Check if relationship exists
                    cursor.execute("""
                        SELECT id, mention_count, dominant_emotions, energy_patterns 
                        FROM relationships 
                        WHERE user_id = ? AND person_name = ?
                    """, (user_id, person_name))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing relationship
                        current_emotions = json.loads(existing["dominant_emotions"] or "{}")
                        current_energy = json.loads(existing["energy_patterns"] or "[]")
                        
                        # Merge emotions (weighted average)
                        for emotion, score in emotions.items():
                            if emotion in current_emotions:
                                current_emotions[emotion] = (current_emotions[emotion] + score) / 2
                            else:
                                current_emotions[emotion] = score
                        
                        # Add current energy pattern
                        current_energy.append({
                            "date": current_date.isoformat(),
                            "emotions": emotions,
                            "entry_id": entry_id
                        })
                        
                        cursor.execute("""
                            UPDATE relationships 
                            SET mention_count = mention_count + 1,
                                last_mentioned_date = ?,
                                dominant_emotions = ?,
                                energy_patterns = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (current_date, json.dumps(current_emotions), 
                             json.dumps(current_energy[-10:]), existing["id"]))  # Keep last 10 energy patterns
                    else:
                        # Create new relationship
                        # Try to infer relationship type from context
                        inferred_type = self._infer_relationship_type(rel)
                        
                        cursor.execute("""
                            INSERT INTO relationships 
                            (user_id, person_name, relationship_type, first_mentioned_date, 
                             last_mentioned_date, mention_count, dominant_emotions, energy_patterns)
                            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                        """, (user_id, person_name, inferred_type, current_date, current_date,
                             json.dumps(emotions), json.dumps([{
                                 "date": current_date.isoformat(),
                                 "emotions": emotions,
                                 "entry_id": entry_id
                             }])))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating relationship tracking: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def _infer_relationship_type(self, relationship: Dict[str, Any]) -> str:
        """Infer relationship type from context"""
        # This is a simplified version - could be enhanced with ML
        context_keywords = relationship.get("keyword", "").lower()
        
        for rel_type, keywords in self.relationship_keywords.items():
            if any(keyword in context_keywords for keyword in keywords):
                return rel_type
        
        return "unknown"
    
    def get_relationship_insights(self, user_id: str, period_start: str, period_end: str) -> Dict[str, Any]:
        """Generate relationship insights for time period"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get relationships mentioned in period
            cursor.execute("""
                SELECT r.*, COUNT(m.id) as period_mentions
                FROM relationships r
                LEFT JOIN messages m ON (
                    m.user_id = r.user_id AND 
                    m.relationship_mentions LIKE '%' || r.person_name || '%' AND
                    m.timestamp BETWEEN ? AND ?
                )
                WHERE r.user_id = ?
                GROUP BY r.id
                HAVING period_mentions > 0 OR r.last_mentioned_date BETWEEN ? AND ?
                ORDER BY period_mentions DESC, r.mention_count DESC
            """, (period_start, period_end, user_id, period_start, period_end))
            
            relationships = cursor.fetchall()
            
            # Analyze relationship patterns
            insights = {
                "active_relationships": len(relationships),
                "most_mentioned": [],
                "emotional_patterns": {"positive": 0, "challenging": 0, "growth": 0},
                "relationship_types": {},
                "energy_shifts": []
            }
            
            for rel in relationships:
                rel_data = dict(rel)
                emotions = json.loads(rel_data["dominant_emotions"] or "{}")
                energy_patterns = json.loads(rel_data["energy_patterns"] or "[]")
                
                # Track most mentioned
                if rel_data["period_mentions"] > 0:
                    insights["most_mentioned"].append({
                        "name": rel_data["person_name"],
                        "type": rel_data["relationship_type"],
                        "mentions": rel_data["period_mentions"],
                        "primary_emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
                    })
                
                # Aggregate emotional patterns
                for emotion, score in emotions.items():
                    if emotion in insights["emotional_patterns"]:
                        insights["emotional_patterns"][emotion] += score
                
                # Count relationship types
                rel_type = rel_data["relationship_type"]
                insights["relationship_types"][rel_type] = insights["relationship_types"].get(rel_type, 0) + 1
            
            # Normalize emotional patterns
            total_emotions = sum(insights["emotional_patterns"].values())
            if total_emotions > 0:
                for emotion in insights["emotional_patterns"]:
                    insights["emotional_patterns"][emotion] /= total_emotions
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting relationship insights: {e}")
            return {"active_relationships": 0, "most_mentioned": [], "emotional_patterns": {}, "relationship_types": {}}
        finally:
            cursor.close()
            conn.close()

# Entry management system for Mirror Scribe fluency
class EntryManager:
    """Provides full CRUD operations for journal entries"""
    
    def update_entry(self, entry_id: int, user_id: str, new_content: str = None, 
                    new_tags: List[str] = None, new_energy_signature: str = None,
                    intention_flag: bool = None) -> bool:
        """Update existing journal entry with versioning"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Verify entry belongs to user
            cursor.execute("SELECT * FROM messages WHERE id = ? AND user_id = ?", (entry_id, user_id))
            current_entry = cursor.fetchone()
            
            if not current_entry:
                return False
            
            updates = []
            params = []
            
            # Update content if provided
            if new_content is not None:
                updates.append("content = ?")
                params.append(new_content)
            
            # Update manual energy signature
            if new_energy_signature is not None:
                updates.append("manual_energy_signature = ?")
                params.append(new_energy_signature)
            
            # Update intention flag
            if intention_flag is not None:
                updates.append("intention_flag = ?")
                params.append(intention_flag)
            
            # Always update revision count and timestamp
            updates.extend(["revision_count = revision_count + 1", "updated_at = CURRENT_TIMESTAMP"])
            
            if updates:
                params.extend([entry_id, user_id])
                cursor.execute(f"""
                    UPDATE messages SET {', '.join(updates)}
                    WHERE id = ? AND user_id = ?
                """, params)
            
            # Update tags if provided
            if new_tags is not None:
                # Remove existing tags
                cursor.execute("DELETE FROM entry_tags WHERE message_id = ?", (entry_id,))
                
                # Add new tags
                for tag_name in new_tags:
                    tag_id = get_or_create_tag(conn, tag_name)
                    cursor.execute("""
                        INSERT INTO entry_tags (message_id, tag_id, is_auto_tagged) 
                        VALUES (?, ?, FALSE)
                    """, (entry_id, tag_id))
            
            # Update relationship mentions if content changed
            if new_content is not None:
                analyzer = RelationshipAnalyzer()
                relationships = analyzer.extract_relationships(new_content)
                relationship_names = [rel.get("name") for rel in relationships if rel.get("name")]
                
                cursor.execute("UPDATE messages SET relationship_mentions = ? WHERE id = ?", 
                             (json.dumps(relationship_names), entry_id))
                
                # Update relationship tracking
                entry_timestamp = current_entry["timestamp"]
                analyzer.update_relationship_tracking(user_id, entry_id, relationships, entry_timestamp)
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating entry: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def delete_entry(self, entry_id: int, user_id: str, soft_delete: bool = True) -> bool:
        """Delete journal entry (soft or hard delete)"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            if soft_delete:
                # Soft delete - mark as deleted but keep data
                cursor.execute("""
                    UPDATE messages 
                    SET content = '[DELETED]', updated_at = CURRENT_TIMESTAMP, revision_count = revision_count + 1
                    WHERE id = ? AND user_id = ?
                """, (entry_id, user_id))
            else:
                # Hard delete - remove completely (cascades to related tables)
                cursor.execute("DELETE FROM messages WHERE id = ? AND user_id = ?", (entry_id, user_id))
            
            affected_rows = cursor.rowcount
            conn.commit()
            return affected_rows > 0
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting entry: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def link_entries(self, from_entry_id: int, to_entry_id: int, connection_type: str, 
                    user_id: str, connection_strength: float = 1.0, description: str = None,
                    created_by: str = "manual") -> bool:
        """Create semantic links between entries"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Verify both entries belong to user
            cursor.execute("""
                SELECT COUNT(*) FROM messages 
                WHERE id IN (?, ?) AND user_id = ?
            """, (from_entry_id, to_entry_id, user_id))
            
            if cursor.fetchone()[0] != 2:
                return False
            
            # Create connection
            cursor.execute("""
                INSERT OR REPLACE INTO entry_connections 
                (from_entry_id, to_entry_id, connection_type, connection_strength, 
                 connection_description, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (from_entry_id, to_entry_id, connection_type, connection_strength, 
                 description, created_by))
            
            # Create reverse connection for certain types
            if connection_type in ["relates_to", "continues", "reflects"]:
                cursor.execute("""
                    INSERT OR REPLACE INTO entry_connections 
                    (from_entry_id, to_entry_id, connection_type, connection_strength, 
                     connection_description, created_by)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (to_entry_id, from_entry_id, f"reverse_{connection_type}", 
                     connection_strength, description, created_by))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error linking entries: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def get_connected_entries(self, entry_id: int, user_id: str, 
                            connection_types: List[str] = None) -> List[Dict[str, Any]]:
        """Get entries connected to this one"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Build query with optional connection type filter
            where_clause = "WHERE (ec.from_entry_id = ? OR ec.to_entry_id = ?) AND m.user_id = ?"
            params = [entry_id, entry_id, user_id]
            
            if connection_types:
                placeholders = ",".join("?" * len(connection_types))
                where_clause += f" AND ec.connection_type IN ({placeholders})"
                params.extend(connection_types)
            
            cursor.execute(f"""
                SELECT DISTINCT m.*, ec.connection_type, ec.connection_strength, 
                       ec.connection_description, ec.created_by,
                       CASE 
                           WHEN ec.from_entry_id = ? THEN 'outgoing'
                           ELSE 'incoming'
                       END as connection_direction
                FROM entry_connections ec
                JOIN messages m ON (
                    (ec.from_entry_id = m.id AND ec.to_entry_id = ?) OR
                    (ec.to_entry_id = m.id AND ec.from_entry_id = ?)
                )
                {where_clause}
                AND m.id != ?
                ORDER BY ec.connection_strength DESC, m.timestamp DESC
            """, [entry_id, entry_id, entry_id] + params + [entry_id])
            
            connections = []
            for row in cursor.fetchall():
                connection = dict(row)
                connections.append(connection)
            
            return connections
            
        except Exception as e:
            logger.error(f"Error getting connected entries: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

# Tag hierarchy system for structured organization
class TagHierarchyManager:
    """Manages hierarchical tag relationships and structured organization"""
    
    def create_tag_relationship(self, parent_tag_name: str, child_tag_name: str, 
                              relationship_type: str = "subcategory") -> bool:
        """Create a parent-child tag relationship"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get or create both tags
            parent_tag_id = get_or_create_tag(conn, parent_tag_name)
            child_tag_id = get_or_create_tag(conn, child_tag_name)
            
            if parent_tag_id == child_tag_id:
                return False  # Can't be parent of itself
            
            # Check for circular dependencies (simplified check)
            cursor.execute("""
                SELECT COUNT(*) FROM tag_hierarchy 
                WHERE parent_tag_id = ? AND child_tag_id = ?
            """, (child_tag_id, parent_tag_id))
            
            if cursor.fetchone()[0] > 0:
                return False  # Would create circular dependency
            
            # Calculate hierarchy level
            cursor.execute("""
                SELECT COALESCE(MAX(hierarchy_level), -1) + 1
                FROM tag_hierarchy 
                WHERE child_tag_id = ?
            """, (parent_tag_id,))
            
            hierarchy_level = cursor.fetchone()[0]
            
            # Insert relationship
            cursor.execute("""
                INSERT OR REPLACE INTO tag_hierarchy 
                (parent_tag_id, child_tag_id, hierarchy_level, relationship_type)
                VALUES (?, ?, ?, ?)
            """, (parent_tag_id, child_tag_id, hierarchy_level, relationship_type))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tag relationship: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def get_tag_hierarchy(self) -> Dict[str, Any]:
        """Get complete tag hierarchy structure"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get all hierarchy relationships with tag names
            cursor.execute("""
                SELECT 
                    pt.name as parent_name, pt.category as parent_category, pt.color as parent_color,
                    ct.name as child_name, ct.category as child_category, ct.color as child_color,
                    th.hierarchy_level, th.relationship_type
                FROM tag_hierarchy th
                LEFT JOIN tags pt ON th.parent_tag_id = pt.id
                JOIN tags ct ON th.child_tag_id = ct.id
                ORDER BY th.hierarchy_level, pt.name, ct.name
            """)
            
            relationships = cursor.fetchall()
            
            # Also get root tags (tags without parents)
            cursor.execute("""
                SELECT t.name, t.category, t.color, t.description
                FROM tags t
                WHERE t.id NOT IN (
                    SELECT child_tag_id FROM tag_hierarchy WHERE parent_tag_id IS NOT NULL
                )
                ORDER BY t.category, t.name
            """)
            
            root_tags = cursor.fetchall()
            
            # Build hierarchy structure
            hierarchy = {
                "root_tags": [dict(tag) for tag in root_tags],
                "relationships": [],
                "tag_tree": {}
            }
            
            # Process relationships
            for rel in relationships:
                rel_data = dict(rel)
                hierarchy["relationships"].append(rel_data)
                
                # Build tree structure
                parent_name = rel_data["parent_name"] or "ROOT"
                if parent_name not in hierarchy["tag_tree"]:
                    hierarchy["tag_tree"][parent_name] = {"children": [], "info": {}}
                
                if rel_data["parent_name"]:
                    hierarchy["tag_tree"][parent_name]["info"] = {
                        "category": rel_data["parent_category"],
                        "color": rel_data["parent_color"]
                    }
                
                hierarchy["tag_tree"][parent_name]["children"].append({
                    "name": rel_data["child_name"],
                    "category": rel_data["child_category"],
                    "color": rel_data["child_color"],
                    "relationship_type": rel_data["relationship_type"],
                    "level": rel_data["hierarchy_level"]
                })
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error getting tag hierarchy: {e}")
            return {"root_tags": [], "relationships": [], "tag_tree": {}}
        finally:
            cursor.close()
            conn.close()
    
    def get_tag_ancestors(self, tag_name: str) -> List[str]:
        """Get all ancestor tags for a given tag"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            ancestors = []
            current_tag = tag_name
            
            # Walk up the hierarchy
            for _ in range(10):  # Prevent infinite loops
                cursor.execute("""
                    SELECT pt.name
                    FROM tag_hierarchy th
                    JOIN tags ct ON th.child_tag_id = ct.id
                    JOIN tags pt ON th.parent_tag_id = pt.id
                    WHERE ct.name = ?
                """, (current_tag,))
                
                parent = cursor.fetchone()
                if not parent:
                    break
                    
                parent_name = parent["name"]
                if parent_name in ancestors:
                    break  # Circular reference detected
                    
                ancestors.append(parent_name)
                current_tag = parent_name
            
            return ancestors[::-1]  # Reverse to get root-to-leaf order
            
        except Exception as e:
            logger.error(f"Error getting tag ancestors: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_tag_descendants(self, tag_name: str) -> List[str]:
        """Get all descendant tags for a given tag"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            descendants = []
            to_process = [tag_name]
            
            while to_process:
                current_tag = to_process.pop(0)
                
                cursor.execute("""
                    SELECT ct.name
                    FROM tag_hierarchy th
                    JOIN tags pt ON th.parent_tag_id = pt.id
                    JOIN tags ct ON th.child_tag_id = ct.id
                    WHERE pt.name = ?
                """, (current_tag,))
                
                children = cursor.fetchall()
                for child in children:
                    child_name = child["name"]
                    if child_name not in descendants:
                        descendants.append(child_name)
                        to_process.append(child_name)
            
            return descendants
            
        except Exception as e:
            logger.error(f"Error getting tag descendants: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

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
        
        # Create messages table with all timestamp columns
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
        
        # Create summaries table for intelligent periodic insights
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                period_type VARCHAR(20) NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                entry_count INTEGER DEFAULT 0,
                dominant_tags JSON,
                energy_signature JSON,
                patterns JSON,
                sacred_summary TEXT,
                wisdom_insights JSON,
                relationship_insights JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, period_type, period_start)
            )
        """)
        
        # Create relationships table for tracking people and connections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                person_name TEXT NOT NULL,
                relationship_type TEXT DEFAULT 'unknown',
                first_mentioned_date DATE,
                last_mentioned_date DATE,
                mention_count INTEGER DEFAULT 0,
                dominant_emotions JSON,
                energy_patterns JSON,
                interaction_quality_trend JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, person_name)
            )
        """)
        
        # Create entry connections for semantic linking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entry_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_entry_id INTEGER NOT NULL,
                to_entry_id INTEGER NOT NULL,
                connection_type TEXT NOT NULL,
                connection_strength FLOAT DEFAULT 1.0,
                connection_description TEXT,
                created_by TEXT DEFAULT 'auto',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_entry_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY (to_entry_id) REFERENCES messages(id) ON DELETE CASCADE,
                UNIQUE(from_entry_id, to_entry_id, connection_type)
            )
        """)
        
        # Create tag hierarchy for structured organization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_hierarchy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_tag_id INTEGER,
                child_tag_id INTEGER NOT NULL,
                hierarchy_level INTEGER DEFAULT 0,
                relationship_type TEXT DEFAULT 'subcategory',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                FOREIGN KEY (child_tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                UNIQUE(parent_tag_id, child_tag_id)
            )
        """)
        
        # Create interaction_logs table for enhancement system
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time FLOAT,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create enhancement_suggestions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhancement_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suggestion_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                triggered_by TEXT NOT NULL,
                suggested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_context JSON,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_timestamp ON messages(user_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_message_id ON entry_tags(message_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_tag_id ON entry_tags(tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_user_period ON summaries(user_id, period_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_period_start ON summaries(period_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_user_id ON relationships(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_user_person ON relationships(user_id, person_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_connections_from ON entry_connections(from_entry_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_connections_to ON entry_connections(to_entry_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_hierarchy_parent ON tag_hierarchy(parent_tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_hierarchy_child ON tag_hierarchy(child_tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interaction_logs_user_id ON interaction_logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interaction_logs_timestamp ON interaction_logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interaction_logs_action_type ON interaction_logs(action_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhancement_suggestions_suggestion_id ON enhancement_suggestions(suggestion_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhancement_suggestions_priority ON enhancement_suggestions(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhancement_suggestions_status ON enhancement_suggestions(status)")
        
        # Enhanced temporal indexes for optimal time-based queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_local_timestamp ON messages(local_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_utc_timestamp ON messages(utc_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_temporal_validation ON messages(temporal_validation_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timezone ON messages(timezone_at_creation)")
        
        # Composite temporal indexes for complex queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_local_time ON messages(user_id, COALESCE(local_timestamp, timestamp))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_utc_time ON messages(user_id, COALESCE(utc_timestamp, timestamp))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp_validation ON messages(timestamp, temporal_validation_score)")
        
        # Time-based aggregation indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_hour_user ON messages(user_id, strftime('%H', COALESCE(local_timestamp, timestamp)))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_day_user ON messages(user_id, strftime('%Y-%m-%d', COALESCE(local_timestamp, timestamp)))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_week_user ON messages(user_id, strftime('%Y-%W', COALESCE(local_timestamp, timestamp)))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_month_user ON messages(user_id, strftime('%Y-%m', COALESCE(local_timestamp, timestamp)))")
        
        # Tag-temporal correlation indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_created_at ON entry_tags(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_message_created ON entry_tags(message_id, created_at)")
        
        # Temporal signals indexes (from temporal_awareness.py tables)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_user_time ON temporal_signals(user_id, created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_type_time ON temporal_signals(signal_type, created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_signals_entry_user ON temporal_signals(entry_id, user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_states_updated ON temporal_states(updated_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_states_signal_time ON temporal_states(last_signal_time)")
        
        # Note: All enhanced columns are now included in the initial CREATE TABLE statement above
        
        # Insert predefined tags if they don't exist
        for tag_data in PREDEFINED_TAGS:
            cursor.execute("""
                INSERT OR IGNORE INTO tags (name, category, is_predefined, color, description)
                VALUES (?, ?, TRUE, ?, ?)
            """, (tag_data["name"], tag_data["category"], tag_data["color"], tag_data["description"]))
        
        # Create indexes for performance optimization
        create_indexes(cursor)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Create temporal awareness tables using the specialized function
        create_temporal_tables(db_path)
        
        # Create timestamp synchronization tables
        create_timestamp_tables(db_path)
        
        # Verify the database schema is correct
        if not verify_database_schema(db_path):
            logger.error("❌ Database schema verification failed after initialization")
            return False
        
        logger.info(f"✅ Enhanced SQLite database with tags, temporal awareness, and timestamp synchronization initialized successfully at: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed at {db_path}: {e}")
        return False

def verify_database_schema(db_path: str) -> bool:
    """Verify that the database has all required columns for the timestamp system"""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        # Check messages table schema
        cursor.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Required columns for the application to function
        required_columns = {
            'id', 'content', 'user_id', 'timestamp',
            'utc_timestamp', 'local_timestamp', 'timezone_at_creation',
            'timestamp_source', 'temporal_validation_score'
        }
        
        missing_columns = required_columns - columns
        
        if missing_columns:
            logger.error(f"❌ Missing required columns in messages table: {missing_columns}")
            conn.close()
            return False
        
        # Test a simple query to ensure the columns are accessible
        cursor.execute("SELECT COUNT(*) FROM messages WHERE utc_timestamp IS NULL OR local_timestamp IS NULL")
        unmigrated_count = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"✅ Database schema verification passed. Unmigrated entries: {unmigrated_count}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database schema verification failed: {e}")
        return False

def create_indexes(cursor):
    """Create database indexes for performance optimization"""
    try:
        logger.info("🔍 Creating database indexes for tag operations...")
        
        # Messages table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_timestamp ON messages(user_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_updated_at ON messages(updated_at)")
        
        # Tags table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_predefined ON tags(is_predefined)")
        
        # Entry_tags relationship indexes (most important for performance)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_message_id ON entry_tags(message_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_tag_id ON entry_tags(tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_created_at ON entry_tags(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_auto_tagged ON entry_tags(is_auto_tagged)")
        
        # Composite indexes for common query patterns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_tags_message_tag ON entry_tags(message_id, tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_tags_user ON messages(user_id) WHERE user_id IN (SELECT DISTINCT user_id FROM messages)")
        
        # Summaries table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_user_period ON summaries(user_id, period_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_period_range ON summaries(period_start, period_end)")
        
        # Relationships table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_user_id ON relationships(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_name ON relationships(name)")
        
        # Entry connections indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_connections_from ON entry_connections(from_entry_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_connections_to ON entry_connections(to_entry_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_connections_type ON entry_connections(connection_type)")
        
        logger.info("✅ Database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Some indexes may already exist or failed to create: {e}")

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
    """Non-blocking startup for Railway health checks"""
    logger.info("🚀 Starting Mirror Scribe Backend with Intelligent Tags...")
    logger.info(f"📁 Environment: {'Railway' if is_railway_environment() else 'Local'}")
    logger.info(f"💾 Database path: {get_database_path()}")
    logger.info(f"📂 Data directory: {'/app/data' if is_railway_environment() else 'local'}")
    
    # Schedule database initialization as a background task to avoid blocking health checks
    import asyncio
    asyncio.create_task(initialize_database_async())

async def initialize_database_async():
    """Async database initialization to avoid blocking startup - ENHANCED WITH AUTO-MIGRATION"""
    try:
        # Initialize database
        success = init_database()
        
        if success:
            # Auto-check and migrate missing columns
            try:
                logger.info("🔍 Checking database schema and auto-migrating if needed...")
                await auto_migrate_missing_columns()
            except Exception as e:
                logger.warning(f"⚠️ Auto-migration check failed: {e}")
            
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
                
                # Check schema completeness
                cursor.execute("PRAGMA table_info(messages)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                required_columns = {'utc_timestamp', 'local_timestamp', 'timezone_at_creation', 'timestamp_source', 'temporal_validation_score'}
                missing_columns = required_columns - existing_columns
                
                conn.close()
                
                schema_status = "COMPLETE" if not missing_columns else f"MISSING: {', '.join(missing_columns)}"
                logger.info(f"📝 Existing journal entries: {message_count}")
                logger.info(f"🏷️ Available tags: {tag_count}")
                logger.info(f"🔗 Tag applications: {tag_applications}")
                logger.info(f"🏗️ Schema status: {schema_status}")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not check existing data: {e}")
            
            logger.info("✅ Mirror Scribe Backend ready with persistent storage and resilient schema!")
        else:
            logger.warning("⚠️ Database initialization failed, but continuing...")
    except Exception as e:
        logger.error(f"❌ Background database initialization failed: {e}")

async def auto_migrate_missing_columns():
    """Automatically check and migrate missing columns during startup"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check for missing columns
        required_columns = [
            {"name": "utc_timestamp", "definition": "DATETIME", "description": "UTC timestamp"},
            {"name": "local_timestamp", "definition": "DATETIME", "description": "Local timestamp"},
            {"name": "timezone_at_creation", "definition": "VARCHAR(50)", "description": "Timezone when created"},
            {"name": "timestamp_source", "definition": "VARCHAR(20) DEFAULT 'auto'", "description": "Source of timestamp"},
            {"name": "temporal_validation_score", "definition": "FLOAT DEFAULT 1.0", "description": "Temporal validation score"},
            {"name": "relationship_mentions", "definition": "TEXT", "description": "JSON array of relationship mentions"}
        ]
        
        columns_added = 0
        for column_info in required_columns:
            column_name = column_info["name"]
            column_def = column_info["definition"]
            
            if not check_column_exists(cursor, "messages", column_name):
                try:
                    logger.info(f"➕ Auto-adding missing column '{column_name}'")
                    alter_sql = f"ALTER TABLE messages ADD COLUMN {column_name} {column_def}"
                    cursor.execute(alter_sql)
                    columns_added += 1
                except Exception as e:
                    logger.warning(f"Failed to auto-add column '{column_name}': {e}")
        
        # Migrate existing data if new columns were added
        if columns_added > 0:
            logger.info(f"🔄 Auto-migrating existing data for {columns_added} new columns...")
            try:
                cursor.execute("""
                    UPDATE messages 
                    SET utc_timestamp = COALESCE(utc_timestamp, timestamp),
                        local_timestamp = COALESCE(local_timestamp, timestamp),
                        timezone_at_creation = COALESCE(timezone_at_creation, 'UTC'),
                        timestamp_source = COALESCE(timestamp_source, 'auto'),
                        temporal_validation_score = COALESCE(temporal_validation_score, 0.5),
                        relationship_mentions = COALESCE(relationship_mentions, '[]')
                    WHERE utc_timestamp IS NULL OR local_timestamp IS NULL
                """)
                migrated_count = cursor.rowcount
                logger.info(f"✅ Auto-migrated {migrated_count} existing messages")
            except Exception as e:
                logger.warning(f"Auto-migration of existing data failed: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        if columns_added > 0:
            logger.info(f"🎉 Auto-migration completed: {columns_added} columns added")
        else:
            logger.info("✅ Database schema is complete, no migration needed")
            
    except Exception as e:
        logger.error(f"Auto-migration failed: {e}")
        # Don't raise - let the app continue even if auto-migration fails

@app.get("/health")
async def health_check():
    """Simple health check for Railway deployment"""
    return {
        "status": "healthy",
        "service": "backend",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with database connectivity"""
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
        logger.error(f"Detailed health check failed: {e}")
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

def column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    except Exception as e:
        logger.error(f"Error checking column {column_name} in {table_name}: {e}")
        return False

@app.post("/api/migrate-database")
async def migrate_database():
    """
    Migrate database schema to add missing enhanced fluency columns
    This endpoint fixes production databases missing the updated_at and other enhanced columns
    """
    try:
        logger.info("🔧 Starting database migration for Mirror Scribe Enhanced Fluency...")
        
        migration_results = {
            "success": True,
            "migrations_applied": [],
            "errors": [],
            "database_path": get_database_path(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Define the columns we need to add
        enhanced_columns = [
            {
                "name": "intention_flag",
                "definition": "BOOLEAN DEFAULT FALSE",
                "description": "Flag for marking entries as intentions"
            },
            {
                "name": "manual_energy_signature", 
                "definition": "TEXT",
                "description": "Manually set energy signature override"
            },
            {
                "name": "relationship_mentions",
                "definition": "JSON",
                "description": "JSON array of people mentioned in entry"
            },
            {
                "name": "updated_at",
                "definition": "TIMESTAMP",
                "description": "Last modification timestamp"
            },
            {
                "name": "revision_count",
                "definition": "INTEGER DEFAULT 0", 
                "description": "Number of revisions made to entry"
            }
        ]
        
        logger.info(f"📋 Checking for {len(enhanced_columns)} enhanced columns in messages table...")
        
        # Check and add each column
        for column_info in enhanced_columns:
            column_name = column_info["name"]
            column_def = column_info["definition"]
            description = column_info["description"]
            
            if column_exists(cursor, "messages", column_name):
                logger.info(f"✅ Column '{column_name}' already exists - skipping")
                migration_results["migrations_applied"].append({
                    "column": column_name,
                    "action": "skipped",
                    "reason": "already_exists"
                })
            else:
                try:
                    logger.info(f"➕ Adding column '{column_name}': {description}")
                    alter_sql = f"ALTER TABLE messages ADD COLUMN {column_name} {column_def}"
                    cursor.execute(alter_sql)
                    logger.info(f"✅ Successfully added column '{column_name}'")
                    migration_results["migrations_applied"].append({
                        "column": column_name,
                        "action": "added",
                        "definition": column_def,
                        "description": description
                    })
                except Exception as e:
                    error_msg = f"❌ Failed to add column '{column_name}': {e}"
                    logger.error(error_msg)
                    migration_results["errors"].append({
                        "column": column_name,
                        "error": str(e),
                        "sql": alter_sql
                    })
                    migration_results["success"] = False
        
        # Update existing entries to have default values for new columns
        if not migration_results["errors"]:
            try:
                logger.info("🔄 Updating existing entries with default values...")
                
                # Set default values for new columns where they might be NULL
                cursor.execute("""
                    UPDATE messages 
                    SET 
                        intention_flag = COALESCE(intention_flag, FALSE),
                        revision_count = COALESCE(revision_count, 0),
                        updated_at = COALESCE(updated_at, timestamp)
                    WHERE intention_flag IS NULL OR revision_count IS NULL OR updated_at IS NULL
                """)
                
                rows_updated = cursor.rowcount
                logger.info(f"✅ Updated {rows_updated} existing entries with default values")
                
                migration_results["migrations_applied"].append({
                    "action": "updated_defaults",
                    "rows_affected": rows_updated,
                    "description": "Set default values for existing entries"
                })
                
            except Exception as e:
                error_msg = f"❌ Failed to update default values: {e}"
                logger.error(error_msg)
                migration_results["errors"].append({
                    "action": "update_defaults",
                    "error": str(e)
                })
                migration_results["success"] = False
        
        # Commit all changes
        if migration_results["success"]:
            conn.commit()
            logger.info("✅ Database migration completed successfully!")
        else:
            conn.rollback()
            logger.warning("⚠️  Database migration completed with errors - changes rolled back")
        
        # Verify the migration
        logger.info("🔍 Verifying migration results...")
        for column_info in enhanced_columns:
            column_name = column_info["name"]
            if column_exists(cursor, "messages", column_name):
                logger.info(f"✅ Verification: Column '{column_name}' exists")
            else:
                logger.error(f"❌ Verification failed: Column '{column_name}' missing")
                migration_results["success"] = False
        
        # Get final database stats
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_entries = cursor.fetchone()[0]
        
        cursor.execute("PRAGMA table_info(messages)")
        column_count = len(cursor.fetchall())
        
        migration_results["database_stats"] = {
            "total_entries": total_entries,
            "total_columns": column_count,
            "database_size_bytes": os.path.getsize(get_database_path()) if os.path.exists(get_database_path()) else 0
        }
        
        cursor.close()
        conn.close()
        
        logger.info(f"🎉 Migration completed with {len(migration_results['migrations_applied'])} actions")
        
        return migration_results
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to migrate database: {str(e)}")

@app.post("/api/migrate-enhancement-system")
async def migrate_enhancement_system():
    """
    Migrate database to add enhancement system tables
    Adds interaction_logs and enhancement_suggestions tables
    """
    try:
        logger.info("🔧 Starting Enhancement System database migration...")
        
        migration_results = {
            "success": True,
            "tables_created": [],
            "indexes_created": [],
            "errors": [],
            "database_path": get_database_path(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        def table_exists(table_name):
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            return cursor.fetchone() is not None
        
        # Create interaction_logs table
        if not table_exists("interaction_logs"):
            logger.info("📋 Creating interaction_logs table...")
            cursor.execute("""
                CREATE TABLE interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    response_time FLOAT,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            migration_results["tables_created"].append("interaction_logs")
            logger.info("✅ Created interaction_logs table")
        else:
            logger.info("⏭️  Table interaction_logs already exists")
        
        # Create enhancement_suggestions table
        if not table_exists("enhancement_suggestions"):
            logger.info("📋 Creating enhancement_suggestions table...")
            cursor.execute("""
                CREATE TABLE enhancement_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suggestion_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    triggered_by TEXT NOT NULL,
                    suggested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_context JSON,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            migration_results["tables_created"].append("enhancement_suggestions")
            logger.info("✅ Created enhancement_suggestions table")
        else:
            logger.info("⏭️  Table enhancement_suggestions already exists")
        
        # Create indexes
        indexes = [
            ("idx_interaction_logs_user_id", "interaction_logs(user_id)"),
            ("idx_interaction_logs_timestamp", "interaction_logs(timestamp)"),
            ("idx_interaction_logs_action_type", "interaction_logs(action_type)"),
            ("idx_enhancement_suggestions_suggestion_id", "enhancement_suggestions(suggestion_id)"),
            ("idx_enhancement_suggestions_priority", "enhancement_suggestions(priority)"),
            ("idx_enhancement_suggestions_status", "enhancement_suggestions(status)")
        ]
        
        for index_name, index_def in indexes:
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {index_def}")
                migration_results["indexes_created"].append(index_name)
            except Exception as e:
                logger.warning(f"Index {index_name} may already exist: {e}")
        
        conn.commit()
        
        # Get stats
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        total_tables = cursor.fetchone()[0]
        
        migration_results["database_stats"] = {
            "total_tables": total_tables,
            "database_size_bytes": os.path.getsize(get_database_path()) if os.path.exists(get_database_path()) else 0
        }
        
        cursor.close()
        conn.close()
        
        logger.info(f"🎉 Enhancement System migration completed with {len(migration_results['tables_created'])} new tables")
        
        return migration_results
        
    except Exception as e:
        logger.error(f"Enhancement System migration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to migrate enhancement system: {str(e)}")

@app.get("/api/schema-status")
async def get_schema_status():
    """Get current database schema status and compatibility information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check messages table schema
        cursor.execute("PRAGMA table_info(messages)")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        # Required columns for full functionality
        required_columns = {
            'id': 'INTEGER',
            'content': 'TEXT',
            'user_id': 'TEXT',
            'timestamp': 'DATETIME',
            'utc_timestamp': 'DATETIME',
            'local_timestamp': 'DATETIME',
            'timezone_at_creation': 'VARCHAR(50)',
            'timestamp_source': 'VARCHAR(20)',
            'temporal_validation_score': 'FLOAT',
            'relationship_mentions': 'TEXT'
        }
        
        # Analyze schema
        present_columns = []
        missing_columns = []
        
        for col_name, col_type in required_columns.items():
            if col_name in existing_columns:
                present_columns.append({
                    "name": col_name,
                    "type": existing_columns[col_name],
                    "required_type": col_type
                })
            else:
                missing_columns.append({
                    "name": col_name,
                    "type": col_type,
                    "status": "missing"
                })
        
        # Get table stats
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        # Check data migration status
        migration_stats = {}
        if 'utc_timestamp' in existing_columns:
            cursor.execute("SELECT COUNT(*) FROM messages WHERE utc_timestamp IS NULL")
            migration_stats["unmigrated_utc"] = cursor.fetchone()[0]
        
        if 'local_timestamp' in existing_columns:
            cursor.execute("SELECT COUNT(*) FROM messages WHERE local_timestamp IS NULL")
            migration_stats["unmigrated_local"] = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Determine compatibility level
        compatibility_level = "FULL" if not missing_columns else "PARTIAL" if len(missing_columns) <= 3 else "BASIC"
        
        return {
            "status": "success",
            "database_path": get_database_path(),
            "timestamp": datetime.utcnow().isoformat(),
            "compatibility_level": compatibility_level,
            "schema_analysis": {
                "total_columns": len(existing_columns),
                "required_columns": len(required_columns),
                "present_columns": len(present_columns),
                "missing_columns": len(missing_columns)
            },
            "columns": {
                "present": present_columns,
                "missing": missing_columns
            },
            "data_stats": {
                "total_messages": total_messages,
                "migration_status": migration_stats
            },
            "recommendations": {
                "action_needed": len(missing_columns) > 0,
                "recommended_endpoint": "/api/emergency-migrate" if missing_columns else None,
                "api_compatibility": {
                    "save_message": "RESILIENT" if compatibility_level in ["FULL", "PARTIAL"] else "BASIC",
                    "get_messages": "RESILIENT" if compatibility_level in ["FULL", "PARTIAL"] else "BASIC"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Schema status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check schema status: {str(e)}")

def check_column_exists(cursor, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    except Exception as e:
        logger.error(f"Error checking column {column_name} in {table_name}: {e}")
        return False

def get_missing_columns(cursor, table_name: str, required_columns: list) -> list:
    """Get list of missing columns from a table"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        missing = [col for col in required_columns if col not in existing_columns]
        return missing
    except Exception as e:
        logger.error(f"Error checking missing columns in {table_name}: {e}")
        return required_columns

@app.post("/api/emergency-migrate")
async def emergency_migrate_database():
    """
    Emergency database migration endpoint for fixing missing columns on live deployment
    This endpoint can be called via HTTP to fix schema issues without redeploy
    """
    try:
        logger.info("🚨 Starting EMERGENCY database migration...")
        
        migration_results = {
            "success": False,
            "database_path": get_database_path(),
            "timestamp": datetime.utcnow().isoformat(),
            "columns_added": [],
            "columns_skipped": [],
            "data_migrated": 0,
            "errors": [],
            "pre_migration_stats": {},
            "post_migration_stats": {}
        }
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get pre-migration stats
        try:
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            migration_results["pre_migration_stats"]["total_messages"] = total_messages
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE local_timestamp IS NULL")
            unmigrated_messages = cursor.fetchone()[0]
            migration_results["pre_migration_stats"]["unmigrated_messages"] = unmigrated_messages
        except Exception as e:
            logger.warning(f"Could not get pre-migration stats: {e}")
        
        # Define required columns for messages table
        required_columns = [
            {"name": "utc_timestamp", "definition": "DATETIME", "description": "UTC timestamp"},
            {"name": "local_timestamp", "definition": "DATETIME", "description": "Local timestamp"},
            {"name": "timezone_at_creation", "definition": "VARCHAR(50)", "description": "Timezone when created"},
            {"name": "timestamp_source", "definition": "VARCHAR(20) DEFAULT 'auto'", "description": "Source of timestamp"},
            {"name": "temporal_validation_score", "definition": "FLOAT DEFAULT 1.0", "description": "Temporal validation score"},
            {"name": "relationship_mentions", "definition": "TEXT", "description": "JSON array of relationship mentions"}
        ]
        
        # Check for missing columns and add them
        logger.info("🔍 Checking for missing columns in messages table...")
        
        for column_info in required_columns:
            column_name = column_info["name"]
            column_def = column_info["definition"]
            description = column_info["description"]
            
            if check_column_exists(cursor, "messages", column_name):
                logger.info(f"✅ Column '{column_name}' already exists")
                migration_results["columns_skipped"].append({
                    "column": column_name,
                    "reason": "already_exists"
                })
            else:
                try:
                    logger.info(f"➕ Adding missing column '{column_name}': {description}")
                    alter_sql = f"ALTER TABLE messages ADD COLUMN {column_name} {column_def}"
                    cursor.execute(alter_sql)
                    
                    migration_results["columns_added"].append({
                        "column": column_name,
                        "definition": column_def,
                        "description": description
                    })
                    logger.info(f"✅ Successfully added column '{column_name}'")
                    
                except Exception as e:
                    error_msg = f"Failed to add column '{column_name}': {str(e)}"
                    logger.error(error_msg)
                    migration_results["errors"].append(error_msg)
        
        # Migrate existing data for new timestamp columns
        if any(col["column"] in ["utc_timestamp", "local_timestamp", "timezone_at_creation"] 
               for col in migration_results["columns_added"]):
            
            logger.info("🔄 Migrating existing message timestamps...")
            try:
                # Update messages with missing timestamp data
                cursor.execute("""
                    UPDATE messages 
                    SET utc_timestamp = timestamp,
                        local_timestamp = timestamp,
                        timezone_at_creation = 'UTC',
                        timestamp_source = 'migrated',
                        temporal_validation_score = 0.5
                    WHERE utc_timestamp IS NULL OR local_timestamp IS NULL
                """)
                
                migrated_count = cursor.rowcount
                migration_results["data_migrated"] = migrated_count
                logger.info(f"✅ Migrated {migrated_count} message timestamps")
                
            except Exception as e:
                error_msg = f"Failed to migrate timestamp data: {str(e)}"
                logger.error(error_msg)
                migration_results["errors"].append(error_msg)
        
        # Initialize relationship_mentions for existing messages
        if any(col["column"] == "relationship_mentions" for col in migration_results["columns_added"]):
            logger.info("🔄 Initializing relationship mentions...")
            try:
                cursor.execute("""
                    UPDATE messages 
                    SET relationship_mentions = '[]'
                    WHERE relationship_mentions IS NULL
                """)
                logger.info("✅ Initialized relationship mentions")
            except Exception as e:
                error_msg = f"Failed to initialize relationship mentions: {str(e)}"
                logger.error(error_msg)
                migration_results["errors"].append(error_msg)
        
        # Commit all changes
        conn.commit()
        
        # Get post-migration stats
        try:
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            migration_results["post_migration_stats"]["total_messages"] = total_messages
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE local_timestamp IS NULL")
            unmigrated_messages = cursor.fetchone()[0]
            migration_results["post_migration_stats"]["unmigrated_messages"] = unmigrated_messages
            
            cursor.execute("SELECT COUNT(*) FROM messages WHERE temporal_validation_score IS NOT NULL")
            validated_messages = cursor.fetchone()[0]
            migration_results["post_migration_stats"]["validated_messages"] = validated_messages
            
        except Exception as e:
            logger.warning(f"Could not get post-migration stats: {e}")
        
        cursor.close()
        conn.close()
        
        # Determine success
        migration_results["success"] = len(migration_results["errors"]) == 0
        
        if migration_results["success"]:
            logger.info("🎉 Emergency database migration completed successfully!")
        else:
            logger.warning(f"⚠️  Emergency migration completed with {len(migration_results['errors'])} errors")
        
        return migration_results
        
    except Exception as e:
        logger.error(f"❌ Emergency migration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "database_path": get_database_path()
        }

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

# ===== ASYNC BACKGROUND PROCESSING =====

async def process_temporal_signals_async(user_id: str, message_id: str, content: str, timestamp: datetime):
    """Process temporal signals asynchronously without blocking the main request"""
    try:
        logger.info(f"🔄 Starting async temporal signal processing for message {message_id}")
        
        # Detect temporal signals
        signals = signal_detector.detect_signals(content, timestamp)
        processed_signals = []
        
        for signal in signals:
            try:
                # Record temporal signal
                state_manager.record_temporal_signal(user_id, signal, message_id)
                processed_signals.append({
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "text_span": signal.text_span
                })
                logger.info(f"📅 Async detected temporal signal: {signal.signal_type.value} (confidence: {signal.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Failed to record temporal signal: {e}")
        
        # Update entry with detected signals in background
        if processed_signals:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Update message with temporal signal count
                cursor.execute("""
                    UPDATE messages 
                    SET temporal_signal_count = ?
                    WHERE id = ?
                """, (len(processed_signals), message_id))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                logger.info(f"✅ Updated message {message_id} with {len(processed_signals)} temporal signals")
            except Exception as e:
                logger.warning(f"Failed to update message with temporal signals: {e}")
        
    except Exception as e:
        logger.error(f"❌ Async temporal signal processing failed for message {message_id}: {e}")

@app.post("/api/save")
async def save_message(message: MessageRequest, client_timestamp: Optional[str] = None, 
                      client_timezone: Optional[str] = None):
    """Save a journal entry with enhanced tag support and timestamp synchronization - RESILIENT VERSION"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check which columns exist in the messages table
        cursor.execute("PRAGMA table_info(messages)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Core required columns (must exist)
        core_columns = ['content', 'user_id', 'timestamp']
        missing_core = [col for col in core_columns if col not in existing_columns]
        if missing_core:
            raise HTTPException(status_code=500, detail=f"Critical database columns missing: {missing_core}")
        
        # Enhanced columns (optional)
        enhanced_columns = {
            'utc_timestamp': 'DATETIME',
            'local_timestamp': 'DATETIME', 
            'timezone_at_creation': 'VARCHAR(50)',
            'timestamp_source': 'VARCHAR(20)',
            'temporal_validation_score': 'FLOAT',
            'relationship_mentions': 'TEXT'
        }
        
        # Create timestamp information (fallback to basic if timestamp synchronizer fails)
        timestamp_info = None
        current_time = datetime.utcnow()
        
        try:
            timestamp_info = timestamp_synchronizer.create_timestamp_info(
                content=message.content,
                user_id=message.user_id,
                client_timestamp=client_timestamp,
                client_timezone=client_timezone,
                timestamp_source=TimestampSource.CLIENT_PROVIDED if client_timestamp else TimestampSource.AUTO
            )
        except Exception as e:
            logger.warning(f"Timestamp synchronizer failed, using fallback: {e}")
            # Create basic timestamp info
            timestamp_info = type('obj', (object,), {
                'utc_timestamp': current_time,
                'local_timestamp': current_time,
                'timezone_name': client_timezone or 'UTC',
                'timestamp_source': type('obj', (object,), {'value': 'fallback'})(),
                'validation_score': 0.5,
                'validation_notes': ['fallback_mode']
            })()
        
        # Build INSERT statement dynamically based on available columns
        insert_columns = ['content', 'user_id', 'timestamp']
        insert_values = [message.content, message.user_id, timestamp_info.utc_timestamp.isoformat()]
        
        # Add enhanced columns if they exist
        if 'utc_timestamp' in existing_columns:
            insert_columns.append('utc_timestamp')
            insert_values.append(timestamp_info.utc_timestamp.isoformat())
            
        if 'local_timestamp' in existing_columns:
            insert_columns.append('local_timestamp')
            insert_values.append(timestamp_info.local_timestamp.isoformat())
            
        if 'timezone_at_creation' in existing_columns:
            insert_columns.append('timezone_at_creation')
            insert_values.append(timestamp_info.timezone_name)
            
        if 'timestamp_source' in existing_columns:
            insert_columns.append('timestamp_source')
            insert_values.append(timestamp_info.timestamp_source.value)
            
        if 'temporal_validation_score' in existing_columns:
            insert_columns.append('temporal_validation_score')
            insert_values.append(timestamp_info.validation_score)
        
        # Execute insert with dynamic columns
        placeholders = ', '.join(['?'] * len(insert_values))
        column_list = ', '.join(insert_columns)
        
        cursor.execute(f"""
            INSERT INTO messages ({column_list})
            VALUES ({placeholders})
        """, insert_values)
        
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
        
        # Add auto tags if enabled (with error handling)
        try:
            if message.auto_tag:
                tagger = AutoTagger()
                auto_tags = tagger.apply_auto_tags(message.content, message.manual_tags)
                applied_tags.extend(auto_tags)
        except Exception as e:
            logger.warning(f"Auto-tagging failed, continuing without auto tags: {e}")
        
        # Apply all tags to the entry (with error handling)
        try:
            apply_tags_to_entry(conn, message_id, applied_tags)
        except Exception as e:
            logger.warning(f"Tag application failed: {e}")
            applied_tags = []  # Reset to empty if tagging fails
        
        # Extract and track relationships (with graceful fallback)
        relationships = []
        relationship_names = []
        
        try:
            relationship_analyzer = RelationshipAnalyzer()
            relationships = relationship_analyzer.extract_relationships(message.content)
            relationship_names = [rel.get("name") for rel in relationships if rel.get("name")]
            
            # Update relationship mentions if column exists
            if 'relationship_mentions' in existing_columns:
                cursor.execute("""
                    UPDATE messages SET relationship_mentions = ? WHERE id = ?
                """, (json.dumps(relationship_names), message_id))
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Update relationship tracking (done after commit, with error handling)
        try:
            if relationships:
                entry_timestamp = datetime.utcnow().isoformat()
                relationship_analyzer.update_relationship_tracking(message.user_id, message_id, relationships, entry_timestamp)
        except Exception as e:
            logger.warning(f"Relationship tracking update failed: {e}")
        
        # Queue temporal signal detection for background processing (non-blocking)
        try:
            temporal_task = asyncio.create_task(
                process_temporal_signals_async(message.user_id, message_id, message.content, timestamp_info.utc_timestamp)
            )
            logger.info(f"📅 Temporal signal detection queued for background processing")
        except Exception as e:
            logger.warning(f"Failed to queue temporal signal detection: {e}")
        
        logger.info(f"💾 Saved message {message_id} with {len(applied_tags)} tags for user {message.user_id}")
        
        # Build response based on available data
        response = {
            "status": "success",
            "message_id": message_id,
            "timestamp": timestamp_info.utc_timestamp.isoformat(),
            "applied_tags": applied_tags,
            "tag_count": len(applied_tags),
            "schema_compatibility": {
                "enhanced_columns_available": len([col for col in enhanced_columns if col in existing_columns]),
                "total_enhanced_columns": len(enhanced_columns),
                "missing_columns": [col for col in enhanced_columns if col not in existing_columns]
            }
        }
        
        # Add enhanced timestamp info if available
        if 'local_timestamp' in existing_columns:
            response["local_timestamp"] = timestamp_info.local_timestamp.isoformat()
        if 'timezone_at_creation' in existing_columns:
            response["timezone"] = timestamp_info.timezone_name
        if 'timestamp_source' in existing_columns:
            response["timestamp_source"] = timestamp_info.timestamp_source.value
        if 'temporal_validation_score' in existing_columns:
            response["temporal_validation_score"] = timestamp_info.validation_score
        
        # Add validation notes if available
        if hasattr(timestamp_info, 'validation_notes'):
            response["validation_notes"] = timestamp_info.validation_notes
        
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Save message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

@app.get("/api/messages/{user_id}")
async def get_messages(user_id: str, limit: int = 100, offset: int = 0, tags: Optional[str] = None,
                      use_local_time: bool = True, 
                      # Temporal filtering parameters
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      relative_period: Optional[str] = None,
                      time_of_day_start: Optional[str] = None,
                      time_of_day_end: Optional[str] = None,
                      days_of_week: Optional[str] = None,
                      timezone_override: Optional[str] = None):
    """Get messages for a user with comprehensive temporal filtering and timezone-aware timestamps - RESILIENT VERSION
    
    Temporal filtering options:
    - start_date/end_date: ISO format date strings (e.g., '2024-01-01T00:00:00Z')
    - relative_period: 'today', 'yesterday', 'last_7_days', 'last_30_days', 'this_week', 'last_week', 'this_month', 'last_month'
    - time_of_day_start/end: Time range filtering (e.g., '08:00', '20:00')
    - days_of_week: Comma-separated day numbers (0=Sunday, 1=Monday, etc.) e.g., '1,2,3,4,5' for weekdays
    - timezone_override: Override user's timezone for filtering
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check which columns exist in the messages table
        cursor.execute("PRAGMA table_info(messages)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Determine available timestamp fields
        has_local_timestamp = 'local_timestamp' in existing_columns
        has_utc_timestamp = 'utc_timestamp' in existing_columns
        has_enhanced_columns = has_local_timestamp and has_utc_timestamp
        
        # Choose timestamp field intelligently based on what's available
        if use_local_time and has_local_timestamp:
            timestamp_field = "local_timestamp"
        elif has_utc_timestamp:
            timestamp_field = "utc_timestamp"
        else:
            timestamp_field = "timestamp"  # Always available
        
        fallback_field = "timestamp"  # Legacy fallback
        
        # Build SELECT clause dynamically based on available columns
        base_select_columns = ["id", "content", "user_id"]
        
        # Always include a display timestamp
        if has_enhanced_columns:
            base_select_columns.append(f"COALESCE({timestamp_field}, {fallback_field}) as display_timestamp")
        else:
            base_select_columns.append(f"{fallback_field} as display_timestamp")
        
        # Add enhanced columns if they exist
        enhanced_select_columns = []
        if 'utc_timestamp' in existing_columns:
            enhanced_select_columns.append("utc_timestamp")
        if 'local_timestamp' in existing_columns:
            enhanced_select_columns.append("local_timestamp")
        if 'timezone_at_creation' in existing_columns:
            enhanced_select_columns.append("timezone_at_creation")
        if 'temporal_validation_score' in existing_columns:
            enhanced_select_columns.append("temporal_validation_score")
        if 'timestamp_source' in existing_columns:
            enhanced_select_columns.append("timestamp_source")
        
        all_select_columns = base_select_columns + enhanced_select_columns
        select_clause = ", ".join(all_select_columns)
        
        # Build temporal filter if provided (with error handling)
        temporal_filter = None
        temporal_where = ""
        temporal_params = []
        
        if any([start_date, end_date, relative_period, time_of_day_start, time_of_day_end, days_of_week]):
            try:
                # Parse days_of_week if provided
                days_list = None
                if days_of_week:
                    try:
                        days_list = [int(d.strip()) for d in days_of_week.split(",")]
                    except ValueError:
                        logger.warning(f"Invalid days_of_week format: {days_of_week}")
                
                # Create temporal filter object
                temporal_filter = TemporalFilterRequest(
                    start_date=start_date,
                    end_date=end_date,
                    relative_period=relative_period,
                    time_of_day_start=time_of_day_start,
                    time_of_day_end=time_of_day_end,
                    days_of_week=days_list,
                    timezone_override=timezone_override
                )
                
                # Get user's timezone for filtering context (with fallback)
                try:
                    state_manager = TemporalStateManager(get_database_path())
                    user_timezone = state_manager.get_user_timezone(user_id)
                except Exception as e:
                    logger.warning(f"Failed to get user timezone, using UTC: {e}")
                    user_timezone = "UTC"
                
                # Build temporal WHERE clause (with fallback for missing columns)
                try:
                    temporal_where, temporal_params = TemporalQueryBuilder.build_temporal_where_clause(temporal_filter, user_timezone)
                except Exception as e:
                    logger.warning(f"Failed to build temporal WHERE clause: {e}")
                    # Build basic date filtering as fallback
                    if start_date or end_date:
                        conditions = []
                        if start_date:
                            conditions.append(f"{fallback_field} >= ?")
                            temporal_params.append(start_date)
                        if end_date:
                            conditions.append(f"{fallback_field} <= ?")
                            temporal_params.append(end_date)
                        temporal_where = " AND ".join(conditions)
                        
            except Exception as e:
                logger.warning(f"Temporal filtering setup failed, proceeding without filtering: {e}")
                temporal_filter = None
        
        # Base WHERE conditions
        if tags:
            base_conditions = ["m.user_id = ?"]
        else:
            base_conditions = ["user_id = ?"]
        base_params = [user_id]
        
        # Add temporal conditions if they exist
        if temporal_where:
            base_conditions.append(temporal_where)
            base_params.extend(temporal_params)
        
        where_clause = " AND ".join(base_conditions)
        
        if tags:
            # Filter by tags with temporal filtering
            tag_list = [tag.strip() for tag in tags.split(",")]
            placeholders = ",".join("?" * len(tag_list))
            
            # Add tag filtering to conditions
            tag_condition = f"t.name IN ({placeholders})"
            all_params = base_params + tag_list + [limit, offset]
            
            # Build ORDER BY clause
            if has_enhanced_columns:
                order_by = f"COALESCE(m.{timestamp_field}, m.{fallback_field}) DESC"
            else:
                order_by = f"m.{fallback_field} DESC"
            
            cursor.execute(f"""
                SELECT DISTINCT m.{select_clause.replace('id,', 'id, m.').replace('content,', 'content, m.').replace('user_id,', 'user_id, m.')}
                FROM messages m
                JOIN entry_tags et ON m.id = et.message_id
                JOIN tags t ON et.tag_id = t.id
                WHERE {where_clause} AND {tag_condition}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            """, all_params)
        else:
            # Get all messages with temporal filtering
            all_params = base_params + [limit, offset]
            
            # Build ORDER BY clause
            if has_enhanced_columns:
                order_by = f"COALESCE({timestamp_field}, {fallback_field}) DESC"
            else:
                order_by = f"{fallback_field} DESC"
            
            cursor.execute(f"""
                SELECT {select_clause}
                FROM messages
                WHERE {where_clause}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            """, all_params)
        
        rows = cursor.fetchall()
        
        # Get tags for each message
        messages = []
        for row in rows:
            # Build message dict dynamically based on available columns
            message = {
                "id": row[0],
                "content": row[1],
                "user_id": row[2],
                "timestamp": row[3]  # Display timestamp
            }
            
            # Add enhanced columns if they exist
            col_index = 4
            if 'utc_timestamp' in existing_columns and col_index < len(row):
                message["utc_timestamp"] = row[col_index]
                col_index += 1
            if 'local_timestamp' in existing_columns and col_index < len(row):
                message["local_timestamp"] = row[col_index]
                col_index += 1
            if 'timezone_at_creation' in existing_columns and col_index < len(row):
                message["timezone"] = row[col_index]
                col_index += 1
            if 'temporal_validation_score' in existing_columns and col_index < len(row):
                message["temporal_validation_score"] = row[col_index]
                col_index += 1
            if 'timestamp_source' in existing_columns and col_index < len(row):
                message["timestamp_source"] = row[col_index]
                col_index += 1
            
            # Get tags for this message (with error handling)
            try:
                cursor.execute("""
                    SELECT t.name, t.color, t.category, et.confidence, et.is_auto_tagged
                    FROM tags t
                    JOIN entry_tags et ON t.id = et.tag_id
                    WHERE et.message_id = ?
                """, (message["id"],))
                
                tags_data = cursor.fetchall()
                message["tags"] = [dict(tag) for tag in tags_data]
            except Exception as e:
                logger.warning(f"Failed to get tags for message {message['id']}: {e}")
                message["tags"] = []
            
            messages.append(message)
        
        cursor.close()
        conn.close()
        
        # Build filtering summary for response
        filtering_applied = []
        if tags:
            filtering_applied.append(f"tags: {tags}")
        if temporal_filter:
            if temporal_filter.relative_period:
                filtering_applied.append(f"period: {temporal_filter.relative_period}")
            if temporal_filter.start_date or temporal_filter.end_date:
                date_range = f"{temporal_filter.start_date or 'start'} to {temporal_filter.end_date or 'end'}"
                filtering_applied.append(f"date_range: {date_range}")
            if temporal_filter.time_of_day_start or temporal_filter.time_of_day_end:
                time_range = f"{temporal_filter.time_of_day_start or '00:00'} to {temporal_filter.time_of_day_end or '23:59'}"
                filtering_applied.append(f"time_range: {time_range}")
            if temporal_filter.days_of_week:
                filtering_applied.append(f"days_of_week: {temporal_filter.days_of_week}")
        
        logger.info(f"📖 Retrieved {len(messages)} messages for user {user_id} (using {timestamp_field} time) with filters: {', '.join(filtering_applied) if filtering_applied else 'none'}")
        
        return {
            "status": "success",
            "messages": messages,
            "count": len(messages),
            "filtered_by_tags": tags.split(",") if tags else None,
            "timestamp_mode": timestamp_field,
            "schema_compatibility": {
                "enhanced_columns_available": len(enhanced_select_columns),
                "total_enhanced_columns": 5,  # utc_timestamp, local_timestamp, timezone_at_creation, temporal_validation_score, timestamp_source
                "missing_columns": [col for col in ['utc_timestamp', 'local_timestamp', 'timezone_at_creation', 'temporal_validation_score', 'timestamp_source'] 
                                  if col not in existing_columns],
                "legacy_mode": not has_enhanced_columns
            },
            "temporal_filtering": {
                "applied": bool(temporal_filter),
                "filters": filtering_applied,
                "relative_period": temporal_filter.relative_period if temporal_filter else None,
                "date_range": {
                    "start": temporal_filter.start_date if temporal_filter else None,
                    "end": temporal_filter.end_date if temporal_filter else None
                } if temporal_filter else None,
                "time_of_day_range": {
                    "start": temporal_filter.time_of_day_start if temporal_filter else None,
                    "end": temporal_filter.time_of_day_end if temporal_filter else None
                } if temporal_filter else None,
                "days_of_week": temporal_filter.days_of_week if temporal_filter else None,
                "timezone_used": temporal_filter.timezone_override if temporal_filter and temporal_filter.timezone_override else None
            }
        }
        
    except Exception as e:
        logger.error(f"Get messages failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.get("/api/messages/{user_id}/tags/{tag_names}")
async def get_messages_by_tags(user_id: str, tag_names: str, limit: int = 100, offset: int = 0):
    """Get messages filtered by specific tags"""
    return await get_messages(user_id, limit, offset, tag_names)

# Summary generation endpoints
@app.post("/api/generate-summary/{period_type}")
async def generate_summary(period_type: str, request: SummaryGenerateRequest):
    """Generate a sacred summary for a specific period"""
    try:
        # Validate period type
        if period_type not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="Period type must be daily, weekly, or monthly")
        
        # Parse target date if provided
        if request.target_date:
            target = datetime.fromisoformat(request.target_date)
        else:
            target = datetime.utcnow()
        
        user_id = request.user_id
        
        # Get date range
        time_analyzer = TimeRangeAnalyzer()
        if period_type == "daily":
            start_date, end_date = time_analyzer.get_daily_range(target)
        elif period_type == "weekly":
            start_date, end_date = time_analyzer.get_weekly_range(target)
        else:  # monthly
            start_date, end_date = time_analyzer.get_monthly_range(target)
        
        # Get entries for the period
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT m.id, m.content, m.timestamp, m.user_id
            FROM messages m
            WHERE m.user_id = ? AND m.timestamp BETWEEN ? AND ?
            ORDER BY m.timestamp ASC
        """, (user_id, start_date.isoformat(), end_date.isoformat()))
        
        entries = []
        for row in cursor.fetchall():
            entry = dict(row)
            
            # Get tags for this entry
            cursor.execute("""
                SELECT t.name, t.color, t.category, et.confidence, et.is_auto_tagged
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                WHERE et.message_id = ?
            """, (entry["id"],))
            
            entry["tags"] = [dict(tag) for tag in cursor.fetchall()]
            entries.append(entry)
        
        # Analyze entries
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_entries(entries)
        
        # Generate sacred summary
        generator = SacredSummaryGenerator()
        sacred_summary = generator.generate_summary(period_type, analysis)
        wisdom_insights = generator.extract_wisdom(analysis)
        
        # Save summary to database
        cursor.execute("""
            INSERT OR REPLACE INTO summaries 
            (user_id, period_type, period_start, period_end, entry_count, 
             dominant_tags, energy_signature, patterns, sacred_summary, wisdom_insights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            period_type,
            start_date.isoformat(),
            end_date.isoformat(),
            analysis["entry_count"],
            json.dumps(analysis["dominant_tags"]),
            json.dumps(analysis["energy_signature"]),
            json.dumps(analysis["patterns"]),
            sacred_summary,
            json.dumps(wisdom_insights)
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Generated {period_type} summary for user {user_id}")
        
        return {
            "status": "success",
            "period_type": period_type,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "sacred_summary": sacred_summary,
            "analysis": analysis,
            "wisdom_insights": wisdom_insights
        }
        
    except Exception as e:
        logger.error(f"Generate summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.get("/api/summaries/{user_id}/{period_type}")
async def get_summaries(user_id: str, period_type: str, limit: int = 10, offset: int = 0):
    """Get existing summaries for a user and period type"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM summaries
            WHERE user_id = ? AND period_type = ?
            ORDER BY period_start DESC
            LIMIT ? OFFSET ?
        """, (user_id, period_type, limit, offset))
        
        summaries = []
        for row in cursor.fetchall():
            summary = dict(row)
            # Parse JSON fields
            summary["dominant_tags"] = json.loads(summary["dominant_tags"])
            summary["energy_signature"] = json.loads(summary["energy_signature"])
            summary["patterns"] = json.loads(summary["patterns"])
            summary["wisdom_insights"] = json.loads(summary["wisdom_insights"])
            summaries.append(summary)
        
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "summaries": summaries,
            "count": len(summaries)
        }
        
    except Exception as e:
        logger.error(f"Get summaries failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summaries: {str(e)}")

@app.get("/api/summary/{user_id}/current/{period_type}")
async def get_current_summary(user_id: str, period_type: str):
    """Get or generate the current period's summary"""
    try:
        # Get current date range
        time_analyzer = TimeRangeAnalyzer()
        if period_type == "daily":
            start_date, end_date = time_analyzer.get_daily_range()
        elif period_type == "weekly":
            start_date, end_date = time_analyzer.get_weekly_range()
        elif period_type == "monthly":
            start_date, end_date = time_analyzer.get_monthly_range()
        else:
            raise HTTPException(status_code=400, detail="Invalid period type")
        
        # Check if summary exists
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM summaries
            WHERE user_id = ? AND period_type = ? AND period_start = ?
        """, (user_id, period_type, start_date.isoformat()))
        
        existing = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if existing:
            # Return existing summary
            summary = dict(existing)
            summary["dominant_tags"] = json.loads(summary["dominant_tags"])
            summary["energy_signature"] = json.loads(summary["energy_signature"])
            summary["patterns"] = json.loads(summary["patterns"])
            summary["wisdom_insights"] = json.loads(summary["wisdom_insights"])
            return {
                "status": "success",
                "summary": summary,
                "generated": False
            }
        else:
            # Generate new summary
            request = SummaryGenerateRequest(user_id=user_id)
            result = await generate_summary(period_type, request)
            return {
                "status": "success",
                "summary": result,
                "generated": True
            }
            
    except Exception as e:
        logger.error(f"Get current summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current summary: {str(e)}")

@app.get("/api/patterns/{user_id}/recent")
async def get_recent_patterns(user_id: str, days: int = 30):
    """Get recent pattern analysis across multiple periods"""
    try:
        # Get entries from the last N days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT m.id, m.content, m.timestamp, m.user_id
            FROM messages m
            WHERE m.user_id = ? AND m.timestamp BETWEEN ? AND ?
            ORDER BY m.timestamp ASC
        """, (user_id, start_date.isoformat(), end_date.isoformat()))
        
        entries = []
        for row in cursor.fetchall():
            entry = dict(row)
            
            # Get tags
            cursor.execute("""
                SELECT t.name, t.color, t.category, et.confidence, et.is_auto_tagged
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                WHERE et.message_id = ?
            """, (entry["id"],))
            
            entry["tags"] = [dict(tag) for tag in cursor.fetchall()]
            entries.append(entry)
        
        cursor.close()
        conn.close()
        
        # Analyze patterns
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze_entries(entries)
        
        # Generate insights
        generator = SacredSummaryGenerator()
        wisdom_insights = generator.extract_wisdom(analysis)
        
        return {
            "status": "success",
            "period_days": days,
            "analysis": analysis,
            "wisdom_insights": wisdom_insights
        }
        
    except Exception as e:
        logger.error(f"Get recent patterns failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent patterns: {str(e)}")

@app.get("/api/insights/{user_id}/growth")
async def get_growth_insights(user_id: str):
    """Get growth tracking insights across time periods"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get last 3 monthly summaries
        cursor.execute("""
            SELECT * FROM summaries
            WHERE user_id = ? AND period_type = 'monthly'
            ORDER BY period_start DESC
            LIMIT 3
        """, (user_id,))
        
        monthly_summaries = []
        for row in cursor.fetchall():
            summary = dict(row)
            summary["patterns"] = json.loads(summary["patterns"])
            summary["energy_signature"] = json.loads(summary["energy_signature"])
            monthly_summaries.append(summary)
        
        # Track growth indicators
        growth_trajectory = []
        for i, summary in enumerate(monthly_summaries):
            patterns = summary["patterns"]
            growth_score = 0
            if patterns.get("growth_momentum"):
                growth_score += 3
            if patterns.get("transformation_active"):
                growth_score += 2
            if patterns.get("consistent_practice"):
                growth_score += 1
            
            growth_trajectory.append({
                "period": summary["period_start"][:7],  # YYYY-MM
                "growth_score": growth_score,
                "primary_energy": summary["energy_signature"].get("primary", "balanced")
            })
        
        cursor.close()
        conn.close()
        
        # Generate growth insights
        insights = []
        if len(growth_trajectory) >= 2:
            if growth_trajectory[0]["growth_score"] > growth_trajectory[1]["growth_score"]:
                insights.append("Ascending spiral - transformation accelerating")
            elif growth_trajectory[0]["growth_score"] < growth_trajectory[1]["growth_score"]:
                insights.append("Integration phase - wisdom deepening")
            else:
                insights.append("Steady flow - maintaining sacred rhythm")
        
        return {
            "status": "success",
            "growth_trajectory": growth_trajectory,
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Get growth insights failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get growth insights: {str(e)}")

# Enhanced fluency endpoints for Mirror Scribe editorial capabilities
@app.put("/api/entries/{entry_id}")
async def update_entry(entry_id: int, request: EntryUpdateRequest, user_id: str = "user123"):
    """Update existing journal entry with full editorial control"""
    try:
        entry_manager = EntryManager()
        success = entry_manager.update_entry(
            entry_id=entry_id,
            user_id=user_id,
            new_content=request.content,
            new_tags=request.tags,
            new_energy_signature=request.energy_signature,
            intention_flag=request.intention_flag
        )
        
        if success:
            logger.info(f"✏️ Updated entry {entry_id} for user {user_id}")
            return {
                "status": "success",
                "message": "Entry updated successfully",
                "entry_id": entry_id,
                "updated_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Entry not found or permission denied")
            
    except Exception as e:
        logger.error(f"Update entry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update entry: {str(e)}")

@app.delete("/api/entries/{entry_id}")
async def delete_entry(entry_id: int, user_id: str = "user123", hard_delete: bool = False):
    """Delete journal entry (soft or hard delete)"""
    try:
        entry_manager = EntryManager()
        success = entry_manager.delete_entry(entry_id, user_id, soft_delete=not hard_delete)
        
        if success:
            delete_type = "hard" if hard_delete else "soft"
            logger.info(f"🗑️ {delete_type} deleted entry {entry_id} for user {user_id}")
            return {
                "status": "success",
                "message": f"Entry {'permanently deleted' if hard_delete else 'marked as deleted'}",
                "entry_id": entry_id,
                "deleted_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Entry not found or permission denied")
            
    except Exception as e:
        logger.error(f"Delete entry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete entry: {str(e)}")

@app.post("/api/entries/connect")
async def connect_entries(request: EntryConnectionRequest, user_id: str = "user123"):
    """Create semantic connections between journal entries"""
    try:
        entry_manager = EntryManager()
        success = entry_manager.link_entries(
            from_entry_id=request.from_entry_id,
            to_entry_id=request.to_entry_id,
            connection_type=request.connection_type,
            user_id=user_id,
            connection_strength=request.connection_strength,
            description=request.description,
            created_by=request.created_by
        )
        
        if success:
            logger.info(f"🔗 Connected entries {request.from_entry_id} -> {request.to_entry_id} ({request.connection_type})")
            return {
                "status": "success",
                "message": "Entries connected successfully",
                "connection": {
                    "from_entry_id": request.from_entry_id,
                    "to_entry_id": request.to_entry_id,
                    "connection_type": request.connection_type,
                    "strength": request.connection_strength
                },
                "created_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create connection - entries may not exist")
            
    except Exception as e:
        logger.error(f"Connect entries failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect entries: {str(e)}")

@app.get("/api/entries/{entry_id}/connected")
async def get_connected_entries(entry_id: int, user_id: str = "user123", 
                              connection_types: Optional[str] = None):
    """Get entries connected to this one"""
    try:
        types_filter = connection_types.split(",") if connection_types else None
        
        entry_manager = EntryManager()
        connections = entry_manager.get_connected_entries(entry_id, user_id, types_filter)
        
        logger.info(f"🔍 Found {len(connections)} connected entries for entry {entry_id}")
        
        return {
            "status": "success",
            "entry_id": entry_id,
            "connected_entries": connections,
            "count": len(connections),
            "filtered_by": types_filter
        }
        
    except Exception as e:
        logger.error(f"Get connected entries failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get connected entries: {str(e)}")

@app.put("/api/entries/{entry_id}/intention")
async def set_intention_flag(entry_id: int, intention: bool, user_id: str = "user123"):
    """Mark entry as intention or remove intention flag"""
    try:
        entry_manager = EntryManager()
        success = entry_manager.update_entry(
            entry_id=entry_id,
            user_id=user_id,
            intention_flag=intention
        )
        
        if success:
            action = "set" if intention else "removed"
            logger.info(f"🎯 Intention flag {action} for entry {entry_id}")
            return {
                "status": "success",
                "message": f"Intention flag {'set' if intention else 'removed'}",
                "entry_id": entry_id,
                "intention_flag": intention,
                "updated_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Entry not found or permission denied")
            
    except Exception as e:
        logger.error(f"Set intention flag failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set intention flag: {str(e)}")

# Relationship insights endpoints
@app.get("/api/relationships/{user_id}")
async def get_relationships(user_id: str, period_days: int = 30):
    """Get relationship insights and connections for user"""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        analyzer = RelationshipAnalyzer()
        insights = analyzer.get_relationship_insights(
            user_id=user_id,
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat()
        )
        
        logger.info(f"👥 Retrieved relationship insights for user {user_id} ({period_days} days)")
        
        return {
            "status": "success",
            "user_id": user_id,
            "period_days": period_days,
            "insights": insights
        }
        
    except Exception as e:
        logger.error(f"Get relationships failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get relationships: {str(e)}")

# Tag hierarchy endpoints
@app.get("/api/tags/hierarchy")
async def get_tag_hierarchy():
    """Get hierarchical tag structure"""
    try:
        hierarchy_manager = TagHierarchyManager()
        hierarchy = hierarchy_manager.get_tag_hierarchy()
        
        logger.info("📊 Retrieved tag hierarchy structure")
        
        return {
            "status": "success",
            "hierarchy": hierarchy
        }
        
    except Exception as e:
        logger.error(f"Get tag hierarchy failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag hierarchy: {str(e)}")

@app.post("/api/tags/hierarchy")
async def create_tag_relationship(request: TagHierarchyRequest):
    """Create parent-child tag relationships"""
    try:
        hierarchy_manager = TagHierarchyManager()
        success = hierarchy_manager.create_tag_relationship(
            parent_tag_name=request.parent_tag_name,
            child_tag_name=request.child_tag_name,
            relationship_type=request.relationship_type
        )
        
        if success:
            logger.info(f"🏷️ Created tag relationship: {request.parent_tag_name} -> {request.child_tag_name}")
            return {
                "status": "success",
                "message": "Tag relationship created successfully",
                "parent": request.parent_tag_name,
                "child": request.child_tag_name,
                "relationship_type": request.relationship_type
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create tag relationship - may cause circular dependency")
            
    except Exception as e:
        logger.error(f"Create tag relationship failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create tag relationship: {str(e)}")

@app.get("/api/tags/{tag_name}/ancestors")
async def get_tag_ancestors(tag_name: str):
    """Get all ancestor tags for a given tag"""
    try:
        hierarchy_manager = TagHierarchyManager()
        ancestors = hierarchy_manager.get_tag_ancestors(tag_name)
        
        return {
            "status": "success",
            "tag": tag_name,
            "ancestors": ancestors,
            "hierarchy_path": " > ".join(ancestors + [tag_name]) if ancestors else tag_name
        }
        
    except Exception as e:
        logger.error(f"Get tag ancestors failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag ancestors: {str(e)}")

@app.get("/api/tags/{tag_name}/descendants")
async def get_tag_descendants(tag_name: str):
    """Get all descendant tags for a given tag"""
    try:
        hierarchy_manager = TagHierarchyManager()
        descendants = hierarchy_manager.get_tag_descendants(tag_name)
        
        return {
            "status": "success",
            "tag": tag_name,
            "descendants": descendants,
            "count": len(descendants)
        }
        
    except Exception as e:
        logger.error(f"Get tag descendants failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tag descendants: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with enhanced API information"""
    return {
        "service": "Mirror Scribe Backend with Full Editorial Fluency & Relationship Intelligence",
        "version": "4.0.0",
        "database": "SQLite with Persistent Volume",
        "storage": get_storage_info(),
        "features": [
            "intelligent_tagging",
            "auto_tag_suggestions", 
            "manual_tags",
            "tag_filtering",
            "category_organization",
            "keyword_matching",
            "sacred_summaries",
            "pattern_analysis",
            "energy_signatures",
            "wisdom_extraction",
            "growth_tracking",
            "relationship_intelligence",
            "editorial_fluency",
            "entry_connections",
            "tag_hierarchy",
            "intention_tracking",
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
            "/api/generate-summary/{period_type}",
            "/api/summaries/{user_id}/{period_type}",
            "/api/summary/{user_id}/current/{period_type}",
            "/api/patterns/{user_id}/recent",
            "/api/insights/{user_id}/growth",
            "/api/entries/{entry_id}",
            "/api/entries/connect",
            "/api/entries/{entry_id}/connected",
            "/api/entries/{entry_id}/intention",
            "/api/relationships/{user_id}",
            "/api/tags/hierarchy",
            "/api/tags/{tag_name}/ancestors",
            "/api/tags/{tag_name}/descendants",
            "/stats"
        ],
        "persistence": {
            "description": "All data persists across Railway deployments",
            "volume_mount": "/app/data" if is_railway_environment() else "N/A",
            "database_path": get_database_path()
        },
        "status": "ready"
    }

# GPT-specific endpoints for enhanced entry refinement
@app.put("/api/gpt/update-entry/{entry_id}")
async def gpt_update_entry(entry_id: int, user_id: str, request: GPTEntryUpdateRequest):
    """GPT-optimized endpoint for updating entries with better tag merging"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify entry belongs to user
        cursor.execute("SELECT * FROM messages WHERE id = ? AND user_id = ?", (entry_id, user_id))
        entry = cursor.fetchone()
        
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found or permission denied")
        
        # Update content if provided
        if request.updated_content:
            cursor.execute("""
                UPDATE messages 
                SET content = ?, updated_at = CURRENT_TIMESTAMP, revision_count = revision_count + 1
                WHERE id = ? AND user_id = ?
            """, (request.updated_content, entry_id, user_id))
        
        # Update emotions (stored as manual energy signature)
        if request.updated_emotions:
            cursor.execute("""
                UPDATE messages 
                SET manual_energy_signature = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
            """, (request.updated_emotions, entry_id, user_id))
        
        # Merge new tags with existing ones
        if request.new_tags:
            # Get existing tags
            cursor.execute("""
                SELECT t.name FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                WHERE et.message_id = ?
            """, (entry_id,))
            existing_tags = {row["name"] for row in cursor.fetchall()}
            
            # Merge with new tags (no duplicates)
            all_tags = existing_tags.union(set(request.new_tags))
            
            # Clear current tags
            cursor.execute("DELETE FROM entry_tags WHERE message_id = ?", (entry_id,))
            
            # Add all tags back
            for tag_name in all_tags:
                tag_id = get_or_create_tag(conn, tag_name)
                cursor.execute("""
                    INSERT INTO entry_tags (message_id, tag_id, is_auto_tagged) 
                    VALUES (?, ?, FALSE)
                """, (entry_id, tag_id))
        
        conn.commit()
        
        # Get updated entry with tags
        cursor.execute("""
            SELECT m.*, GROUP_CONCAT(t.name) as tag_names
            FROM messages m
            LEFT JOIN entry_tags et ON m.id = et.message_id
            LEFT JOIN tags t ON et.tag_id = t.id
            WHERE m.id = ?
            GROUP BY m.id
        """, (entry_id,))
        updated_entry = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        logger.info(f"✨ GPT updated entry {entry_id} for user {user_id}")
        
        return {
            "status": "success",
            "message": "Entry updated successfully",
            "entry_id": entry_id,
            "tags": updated_entry["tag_names"].split(",") if updated_entry["tag_names"] else [],
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GPT update entry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update entry: {str(e)}")

@app.post("/api/gpt/add-tags-to-entry/{entry_id}")
async def gpt_add_tags_to_entry(entry_id: int, user_id: str, request: GPTTagAddRequest):
    """Add tags to existing entry without removing current tags"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify entry belongs to user
        cursor.execute("SELECT id FROM messages WHERE id = ? AND user_id = ?", (entry_id, user_id))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Entry not found or permission denied")
        
        # Get existing tags
        cursor.execute("""
            SELECT t.name FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            WHERE et.message_id = ?
        """, (entry_id,))
        existing_tags = {row["name"] for row in cursor.fetchall()}
        
        # Add new tags (skip duplicates)
        added_tags = []
        for tag_name in request.tags:
            if tag_name not in existing_tags:
                tag_id = get_or_create_tag(conn, tag_name)
                try:
                    cursor.execute("""
                        INSERT INTO entry_tags (message_id, tag_id, is_auto_tagged) 
                        VALUES (?, ?, FALSE)
                    """, (entry_id, tag_id))
                    added_tags.append(tag_name)
                except sqlite3.IntegrityError:
                    # Tag already exists for this entry
                    pass
        
        # Update timestamp
        cursor.execute("""
            UPDATE messages 
            SET updated_at = CURRENT_TIMESTAMP, revision_count = revision_count + 1
            WHERE id = ?
        """, (entry_id,))
        
        conn.commit()
        
        # Get all tags for the entry
        cursor.execute("""
            SELECT t.name, t.category, t.color FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            WHERE et.message_id = ?
            ORDER BY t.name
        """, (entry_id,))
        all_tags = [{"name": row["name"], "category": row["category"], "color": row["color"]} 
                   for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        logger.info(f"🏷️ Added {len(added_tags)} tags to entry {entry_id}")
        
        return {
            "status": "success",
            "message": f"Added {len(added_tags)} new tags",
            "added_tags": added_tags,
            "all_tags": all_tags,
            "total_tags": len(all_tags)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add tags to entry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add tags: {str(e)}")

# Tag visibility endpoints for sacred thread tracking
@app.get("/api/gpt/tags/list/{user_id}")
async def gpt_get_user_tags(user_id: str):
    """Get all unique tags for a user with usage statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                t.name as tag,
                t.category,
                t.color,
                COUNT(DISTINCT et.message_id) as count,
                MIN(m.timestamp) as first_used,
                MAX(m.timestamp) as last_used
            FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            JOIN messages m ON et.message_id = m.id
            WHERE m.user_id = ?
            GROUP BY t.id, t.name, t.category, t.color
            ORDER BY count DESC, t.name
        """, (user_id,))
        
        tags = []
        for row in cursor.fetchall():
            tags.append({
                "tag": row["tag"],
                "category": row["category"],
                "color": row["color"],
                "count": row["count"],
                "first_used": row["first_used"],
                "last_used": row["last_used"]
            })
        
        cursor.close()
        conn.close()
        
        logger.info(f"📋 Retrieved {len(tags)} tags for user {user_id}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_tags": len(tags),
            "tags": tags
        }
        
    except Exception as e:
        logger.error(f"Get user tags failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")

@app.get("/api/gpt/tags/temporal/{user_id}")
async def gpt_get_temporal_tags(user_id: str, period: str = "weekly"):
    """Get tag evolution over time periods"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Determine period grouping
        if period == "weekly":
            date_format = '%Y-W%W'
            days_back = 90  # Last 3 months
        elif period == "monthly":
            date_format = '%Y-%m'
            days_back = 365  # Last year
        else:
            date_format = '%Y-%m-%d'
            days_back = 30  # Last month
        
        start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        
        # Get tags by period
        cursor.execute("""
            SELECT 
                strftime(?, m.timestamp) as period,
                t.name as tag,
                COUNT(*) as usage_count,
                MIN(m.timestamp) as period_start
            FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            JOIN messages m ON et.message_id = m.id
            WHERE m.user_id = ? AND m.timestamp >= ?
            GROUP BY period, t.name
            ORDER BY period DESC, usage_count DESC
        """, (date_format, user_id, start_date))
        
        # Organize by period
        periods = {}
        all_tags = set()
        
        for row in cursor.fetchall():
            period = row["period"]
            if period not in periods:
                periods[period] = {
                    "period": period,
                    "tags": {},
                    "new_tags": [],
                    "period_start": row["period_start"]
                }
            
            tag_name = row["tag"]
            periods[period]["tags"][tag_name] = row["usage_count"]
            
            if tag_name not in all_tags:
                periods[period]["new_tags"].append(tag_name)
                all_tags.add(tag_name)
        
        # Convert to list and identify trending tags
        period_list = []
        for period_key in sorted(periods.keys(), reverse=True):
            period_data = periods[period_key]
            
            # Find trending tags (most used in this period)
            trending = sorted(period_data["tags"].items(), key=lambda x: x[1], reverse=True)[:5]
            
            period_list.append({
                "period": period_key,
                "new_tags": period_data["new_tags"],
                "trending": [tag[0] for tag in trending],
                "tag_count": len(period_data["tags"]),
                "total_uses": sum(period_data["tags"].values())
            })
        
        cursor.close()
        conn.close()
        
        logger.info(f"📊 Retrieved temporal tag data for user {user_id}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "period_type": period,
            "periods": period_list
        }
        
    except Exception as e:
        logger.error(f"Get temporal tags failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get temporal tags: {str(e)}")

@app.get("/api/gpt/tags/preview/{tag_name}")
async def gpt_preview_tag_entries(tag_name: str, user_id: str, limit: int = 10):
    """Get preview of entries containing a specific tag"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                m.id,
                m.timestamp as date,
                SUBSTR(m.content, 1, 150) as preview,
                m.manual_energy_signature as emotion,
                GROUP_CONCAT(t2.name) as all_tags
            FROM messages m
            JOIN entry_tags et ON m.id = et.message_id
            JOIN tags t ON et.tag_id = t.id
            LEFT JOIN entry_tags et2 ON m.id = et2.message_id
            LEFT JOIN tags t2 ON et2.tag_id = t2.id
            WHERE t.name = ? AND m.user_id = ?
            GROUP BY m.id
            ORDER BY m.timestamp DESC
            LIMIT ?
        """, (tag_name, user_id, limit))
        
        entries = []
        for row in cursor.fetchall():
            preview_text = row["preview"]
            if len(preview_text) == 150:
                preview_text += "..."
            
            entries.append({
                "id": row["id"],
                "date": row["date"],
                "preview": preview_text,
                "emotion": row["emotion"] or "neutral",
                "tags": row["all_tags"].split(",") if row["all_tags"] else []
            })
        
        # Get tag statistics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT m.id) as total_entries,
                MIN(m.timestamp) as first_use,
                MAX(m.timestamp) as last_use
            FROM messages m
            JOIN entry_tags et ON m.id = et.message_id
            JOIN tags t ON et.tag_id = t.id
            WHERE t.name = ? AND m.user_id = ?
        """, (tag_name, user_id))
        
        stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        logger.info(f"👁️ Retrieved preview for tag '{tag_name}'")
        
        return {
            "status": "success",
            "tag": tag_name,
            "total_entries": stats["total_entries"],
            "first_use": stats["first_use"],
            "last_use": stats["last_use"],
            "entries": entries
        }
        
    except Exception as e:
        logger.error(f"Preview tag entries failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preview tag: {str(e)}")

@app.get("/api/gpt/tags/sacred-threads/{user_id}")
async def gpt_get_sacred_threads(user_id: str):
    """Get tag relationships and evolution patterns"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get commonly co-occurring tags
        cursor.execute("""
            SELECT 
                t1.name as tag1,
                t2.name as tag2,
                COUNT(*) as co_occurrences
            FROM entry_tags et1
            JOIN entry_tags et2 ON et1.message_id = et2.message_id AND et1.tag_id < et2.tag_id
            JOIN tags t1 ON et1.tag_id = t1.id
            JOIN tags t2 ON et2.tag_id = t2.id
            JOIN messages m ON et1.message_id = m.id
            WHERE m.user_id = ?
            GROUP BY t1.name, t2.name
            HAVING co_occurrences >= 3
            ORDER BY co_occurrences DESC
            LIMIT 20
        """, (user_id,))
        
        tag_relationships = []
        for row in cursor.fetchall():
            tag_relationships.append({
                "tags": [row["tag1"], row["tag2"]],
                "strength": row["co_occurrences"]
            })
        
        # Get tag evolution timeline
        cursor.execute("""
            SELECT 
                t.name as tag,
                COUNT(*) as usage_count,
                MIN(m.timestamp) as first_seen,
                MAX(m.timestamp) as last_seen,
                julianday(MAX(m.timestamp)) - julianday(MIN(m.timestamp)) as days_active
            FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            JOIN messages m ON et.message_id = m.id
            WHERE m.user_id = ?
            GROUP BY t.name
            HAVING usage_count >= 3
            ORDER BY first_seen
        """, (user_id,))
        
        tag_timeline = []
        for row in cursor.fetchall():
            tag_timeline.append({
                "tag": row["tag"],
                "usage_count": row["usage_count"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "days_active": int(row["days_active"]),
                "status": "active" if (datetime.utcnow() - datetime.fromisoformat(row["last_seen"].replace('Z', '+00:00'))).days < 30 else "dormant"
            })
        
        # Identify tag clusters by category and usage patterns
        cursor.execute("""
            SELECT 
                t.category,
                GROUP_CONCAT(t.name) as tags,
                COUNT(DISTINCT t.id) as tag_count,
                COUNT(DISTINCT et.message_id) as total_uses
            FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            JOIN messages m ON et.message_id = m.id
            WHERE m.user_id = ? AND t.category IS NOT NULL
            GROUP BY t.category
            ORDER BY total_uses DESC
        """, (user_id,))
        
        tag_clusters = []
        for row in cursor.fetchall():
            tag_clusters.append({
                "category": row["category"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "tag_count": row["tag_count"],
                "total_uses": row["total_uses"]
            })
        
        cursor.close()
        conn.close()
        
        logger.info(f"🕸️ Retrieved sacred threads for user {user_id}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "tag_relationships": tag_relationships,
            "tag_timeline": tag_timeline,
            "tag_clusters": tag_clusters,
            "insights": {
                "total_relationships": len(tag_relationships),
                "active_tags": sum(1 for t in tag_timeline if t["status"] == "active"),
                "dormant_tags": sum(1 for t in tag_timeline if t["status"] == "dormant"),
                "strongest_connection": tag_relationships[0] if tag_relationships else None
            }
        }
        
    except Exception as e:
        logger.error(f"Get sacred threads failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sacred threads: {str(e)}")

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

# Enhancement System API Endpoints
@app.post("/api/interactions")
async def save_interaction_log(interaction: InteractionLogCreate):
    """Save interaction log data to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO interaction_logs 
            (user_id, action_type, response_time, success, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            interaction.user_id,
            interaction.action_type,
            interaction.response_time,
            interaction.success,
            interaction.error_message,
            json.dumps(interaction.metadata) if interaction.metadata else None
        ))
        
        log_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"📊 Saved interaction log {log_id} for user {interaction.user_id}: {interaction.action_type}")
        
        return {
            "status": "success",
            "id": log_id,
            "message": "Interaction log saved successfully"
        }
        
    except Exception as e:
        logger.error(f"Save interaction log failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save interaction log: {str(e)}")

@app.post("/api/enhancements")
async def save_enhancement_suggestion(enhancement: EnhancementSuggestionCreate):
    """Save enhancement suggestion to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO enhancement_suggestions 
            (suggestion_id, title, description, category, priority, reasoning, 
             triggered_by, user_context, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            enhancement.suggestion_id,
            enhancement.title,
            enhancement.description,
            enhancement.category,
            enhancement.priority,
            enhancement.reasoning,
            enhancement.triggered_by,
            json.dumps(enhancement.user_context) if enhancement.user_context else None,
            enhancement.status
        ))
        
        suggestion_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"💡 Saved enhancement suggestion {enhancement.suggestion_id}: {enhancement.title}")
        
        return {
            "status": "success",
            "id": suggestion_id,
            "suggestion_id": enhancement.suggestion_id,
            "message": "Enhancement suggestion saved successfully"
        }
        
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"Enhancement suggestion with ID '{enhancement.suggestion_id}' already exists")
    except Exception as e:
        logger.error(f"Save enhancement suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save enhancement suggestion: {str(e)}")

@app.get("/api/interactions/stats")
async def get_interaction_stats(days: int = 7):
    """Get interaction statistics for the last N days"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get total interactions
        cursor.execute("""
            SELECT COUNT(*) as total FROM interaction_logs
            WHERE timestamp >= ?
        """, (start_date.isoformat(),))
        total_interactions = cursor.fetchone()["total"]
        
        # Get breakdown by action type
        cursor.execute("""
            SELECT action_type, COUNT(*) as count
            FROM interaction_logs
            WHERE timestamp >= ?
            GROUP BY action_type
            ORDER BY count DESC
        """, (start_date.isoformat(),))
        action_breakdown = {row["action_type"]: row["count"] for row in cursor.fetchall()}
        
        # Get success rate
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                COUNT(*) as total
            FROM interaction_logs
            WHERE timestamp >= ?
        """, (start_date.isoformat(),))
        success_data = cursor.fetchone()
        success_rate = (success_data["successful"] / success_data["total"] * 100) if success_data["total"] > 0 else 100.0
        
        # Get average response time
        cursor.execute("""
            SELECT AVG(response_time) as avg_response_time
            FROM interaction_logs
            WHERE timestamp >= ? AND response_time IS NOT NULL
        """, (start_date.isoformat(),))
        avg_response_time = cursor.fetchone()["avg_response_time"] or 0
        
        # Get interactions by day
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM interaction_logs
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (start_date.isoformat(),))
        daily_interactions = [{"date": row["date"], "count": row["count"]} for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "period_days": days,
            "total_interactions": total_interactions,
            "action_breakdown": action_breakdown,
            "success_rate": round(success_rate, 2),
            "average_response_time": round(avg_response_time, 2) if avg_response_time else 0,
            "daily_interactions": daily_interactions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get interaction stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get interaction stats: {str(e)}")

@app.get("/api/enhancements")
async def get_enhancement_suggestions(
    priority: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """Get enhancement suggestions with optional filters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query with optional filters
        query = "SELECT * FROM enhancement_suggestions WHERE 1=1"
        params = []
        
        if priority:
            query += " AND priority = ?"
            params.append(priority)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY suggested_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        suggestions = []
        
        for row in cursor.fetchall():
            suggestion = dict(row)
            # Parse JSON fields
            if suggestion["user_context"]:
                suggestion["user_context"] = json.loads(suggestion["user_context"])
            suggestions.append(suggestion)
        
        cursor.close()
        conn.close()
        
        return {
            "status": "success",
            "suggestions": suggestions,
            "count": len(suggestions),
            "filters": {
                "priority": priority,
                "category": category,
                "status": status,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Get enhancement suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement suggestions: {str(e)}")

# ========================================
# TEMPORAL AWARENESS API ENDPOINTS
# ========================================

# Initialize temporal awareness components
signal_detector = TemporalSignalDetector()
state_manager = TemporalStateManager(DATABASE_FILE)
summary_generator = TemporalSummaryGenerator(DATABASE_FILE)

# Initialize timestamp synchronization components
timezone_manager = TimezoneManager()
timestamp_synchronizer = TimestampSynchronizer(DATABASE_FILE)
temporal_validator = TemporalValidator(DATABASE_FILE)

@app.post("/api/temporal/mark-signal")
async def mark_temporal_signal(user_id: str, signal_data: TemporalSignalCreate):
    """Mark a detected temporal boundary signal"""
    try:
        logger.info(f"📅 Marking temporal signal for user {user_id}: {signal_data.signal_type}")
        
        # Parse timestamp if provided
        signal_timestamp = datetime.now()
        if signal_data.signal_timestamp:
            try:
                signal_timestamp = datetime.fromisoformat(signal_data.signal_timestamp)
            except ValueError:
                signal_timestamp = datetime.now()
        
        # Create temporal signal object
        signal = TemporalSignal(
            signal_type=SignalType(signal_data.signal_type),
            confidence=signal_data.confidence,
            detected_text=signal_data.detected_text,
            signal_timestamp=signal_timestamp,
            entry_id=signal_data.entry_id,
            metadata=signal_data.metadata
        )
        
        # Record the signal
        state_manager.record_temporal_signal(user_id, signal, signal_data.entry_id)
        
        return {
            "status": "success",
            "message": "Temporal signal marked successfully",
            "signal_type": signal_data.signal_type,
            "confidence": signal_data.confidence,
            "timestamp": signal_timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Mark temporal signal failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to mark temporal signal: {str(e)}")

@app.get("/api/temporal/state/{user_id}")
async def get_temporal_state(user_id: str):
    """Get current temporal state for user"""
    try:
        logger.info(f"🔍 Getting temporal state for user {user_id}")
        
        temporal_state = state_manager.get_temporal_state(user_id)
        
        return {
            "status": "success",
            "user_id": temporal_state.user_id,
            "timezone": temporal_state.timezone,
            "boundaries": {
                "last_day_start": temporal_state.last_day_start.isoformat() if temporal_state.last_day_start else None,
                "last_day_end": temporal_state.last_day_end.isoformat() if temporal_state.last_day_end else None,
                "last_week_start": temporal_state.last_week_start.isoformat() if temporal_state.last_week_start else None,
                "last_week_end": temporal_state.last_week_end.isoformat() if temporal_state.last_week_end else None,
                "last_month_start": temporal_state.last_month_start.isoformat() if temporal_state.last_month_start else None,
                "last_month_end": temporal_state.last_month_end.isoformat() if temporal_state.last_month_end else None,
                "last_year_start": temporal_state.last_year_start.isoformat() if temporal_state.last_year_start else None,
                "last_year_end": temporal_state.last_year_end.isoformat() if temporal_state.last_year_end else None,
            },
            "updated_at": temporal_state.updated_at.isoformat() if temporal_state.updated_at else None
        }
        
    except Exception as e:
        logger.error(f"Get temporal state failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get temporal state: {str(e)}")

@app.post("/api/temporal/detect-signals")
async def detect_temporal_signals(detect_request: TemporalSignalDetectRequest):
    """Analyze entry content for temporal signals"""
    try:
        logger.info(f"🔍 Detecting temporal signals in content")
        
        # Parse timestamp if provided
        entry_timestamp = datetime.now()
        if detect_request.entry_timestamp:
            try:
                entry_timestamp = datetime.fromisoformat(detect_request.entry_timestamp)
            except ValueError:
                entry_timestamp = datetime.now()
        
        # Detect signals
        signals = signal_detector.detect_signals(detect_request.content, entry_timestamp)
        
        # Convert signals to response format
        detected_signals = []
        for signal in signals:
            detected_signals.append({
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "detected_text": signal.detected_text,
                "signal_timestamp": signal.signal_timestamp.isoformat(),
                "metadata": signal.metadata
            })
        
        return {
            "status": "success",
            "content_analyzed": True,
            "signals_found": len(detected_signals),
            "signals": detected_signals,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Detect temporal signals failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect temporal signals: {str(e)}")

@app.get("/api/temporal/summary/{user_id}")
async def generate_temporal_summary(user_id: str, period_type: str = "weekly", 
                                  start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Generate a temporal period summary"""
    try:
        logger.info(f"📊 Generating {period_type} temporal summary for user {user_id}")
        
        # Calculate period dates if not provided
        if not start_date or not end_date:
            now = datetime.now()
            user_tz = state_manager.get_user_timezone(user_id)
            
            if period_type == "daily":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif period_type == "weekly":
                # Get start of week (Monday)
                days_since_monday = now.weekday()
                start_date = now - timedelta(days=days_since_monday)
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
            elif period_type == "monthly":
                start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                # Get last day of month
                if now.month == 12:
                    next_month = now.replace(year=now.year + 1, month=1, day=1)
                else:
                    next_month = now.replace(month=now.month + 1, day=1)
                end_date = next_month - timedelta(days=1)
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                raise HTTPException(status_code=400, detail="Invalid period_type. Use: daily, weekly, monthly")
        else:
            # Parse provided dates
            try:
                start_date = datetime.fromisoformat(start_date)
                end_date = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format: YYYY-MM-DDTHH:MM:SS")
        
        # Generate summary
        summary = summary_generator.generate_period_summary(user_id, period_type, start_date, end_date)
        
        return {
            "status": "success",
            "summary": summary,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Generate temporal summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate temporal summary: {str(e)}")

@app.get("/api/temporal/missing-signals/{user_id}")
async def get_missing_temporal_signals(user_id: str, days_back: int = 7):
    """Suggest missing temporal boundaries based on recent activity"""
    try:
        logger.info(f"🔍 Analyzing missing temporal signals for user {user_id}")
        
        # Get temporal state
        temporal_state = state_manager.get_temporal_state(user_id)
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_back)
        
        # Analyze missing signals
        missing_signals = []
        suggestions = []
        
        # Check for missing day boundaries
        if not temporal_state.last_day_start or temporal_state.last_day_start < cutoff_date:
            missing_signals.append("day_start")
            suggestions.append({
                "signal_type": "day_start",
                "suggestion": "Consider marking the beginning of your day with a morning reflection",
                "confidence": 0.7,
                "urgency": "medium"
            })
        
        if not temporal_state.last_day_end or temporal_state.last_day_end < cutoff_date:
            missing_signals.append("day_end")
            suggestions.append({
                "signal_type": "day_end",
                "suggestion": "Evening reflections can help close your day with intention",
                "confidence": 0.7,
                "urgency": "medium"
            })
        
        # Check for missing week boundaries
        if not temporal_state.last_week_start or temporal_state.last_week_start < (now - timedelta(weeks=2)):
            missing_signals.append("week_start")
            suggestions.append({
                "signal_type": "week_start",
                "suggestion": "Weekly planning and intention-setting can enhance your awareness",
                "confidence": 0.6,
                "urgency": "low"
            })
        
        # Check for missing month boundaries
        if not temporal_state.last_month_start or temporal_state.last_month_start < (now - timedelta(days=60)):
            missing_signals.append("month_start")
            suggestions.append({
                "signal_type": "month_start",
                "suggestion": "Monthly reflections provide valuable perspective on your growth",
                "confidence": 0.5,
                "urgency": "low"
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "analysis_period_days": days_back,
            "missing_signals": missing_signals,
            "suggestions": suggestions,
            "temporal_awareness_score": len(missing_signals) / 8.0,  # 8 possible boundary types
            "analyzed_at": now.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get missing temporal signals failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze missing temporal signals: {str(e)}")

# Background service endpoint for scheduled boundary detection
@app.post("/api/temporal/auto-detect-boundaries")
async def auto_detect_temporal_boundaries():
    """Automatically detect and mark temporal boundaries (for scheduled execution)"""
    try:
        logger.info("🕐 Running automatic temporal boundary detection")
        
        now = datetime.now()
        detected_boundaries = []
        
        # Check for natural temporal boundaries
        # This would typically be called by a scheduled service
        
        # Day boundaries
        if now.hour == 0 and now.minute < 5:  # Early morning
            detected_boundaries.append({
                "boundary_type": "day_start",
                "timestamp": now.isoformat(),
                "confidence": 1.0,
                "automatic": True
            })
        elif now.hour == 23 and now.minute > 55:  # Late evening
            detected_boundaries.append({
                "boundary_type": "day_end",
                "timestamp": now.isoformat(),
                "confidence": 1.0,
                "automatic": True
            })
        
        # Week boundaries (Monday morning)
        if now.weekday() == 0 and now.hour < 12:  # Monday before noon
            detected_boundaries.append({
                "boundary_type": "week_start",
                "timestamp": now.isoformat(),
                "confidence": 0.9,
                "automatic": True
            })
        elif now.weekday() == 6 and now.hour > 18:  # Sunday evening
            detected_boundaries.append({
                "boundary_type": "week_end",
                "timestamp": now.isoformat(),
                "confidence": 0.9,
                "automatic": True
            })
        
        # Month boundaries
        if now.day == 1 and now.hour < 12:  # First day of month
            detected_boundaries.append({
                "boundary_type": "month_start",
                "timestamp": now.isoformat(),
                "confidence": 0.8,
                "automatic": True
            })
        elif now.day >= 28:  # Last few days of month
            # Check if tomorrow is the first day of next month
            tomorrow = now + timedelta(days=1)
            if tomorrow.day == 1:
                detected_boundaries.append({
                    "boundary_type": "month_end",
                    "timestamp": now.isoformat(),
                    "confidence": 0.8,
                    "automatic": True
                })
        
        return {
            "status": "success",
            "detected_boundaries": detected_boundaries,
            "detection_timestamp": now.isoformat(),
            "automatic_detection": True
        }
        
    except Exception as e:
        logger.error(f"Auto-detect temporal boundaries failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to auto-detect temporal boundaries: {str(e)}")

# ========================================
# TEMPORAL AGGREGATION API ENDPOINTS
# ========================================

@app.get("/api/temporal/activity-patterns/{user_id}")
async def get_activity_patterns(user_id: str, days_back: int = 30, granularity: str = "hourly"):
    """Get comprehensive activity patterns analysis"""
    try:
        summary_generator = TemporalSummaryGenerator(get_database_path())
        patterns = summary_generator.get_activity_patterns(user_id, days_back)
        return {"status": "success", "user_id": user_id, "activity_patterns": patterns}
    except Exception as e:
        logger.error(f"Activity patterns analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze activity patterns: {str(e)}")

@app.get("/api/temporal/correlations/{user_id}")
async def get_temporal_correlations(user_id: str, days_back: int = 30, correlation_type: str = "tags_time"):
    """Analyze temporal correlations in user data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        if correlation_type == "tags_time":
            cursor.execute('''
                SELECT t.name, strftime('%H', COALESCE(m.local_timestamp, m.timestamp)) as hour,
                       COUNT(*) as usage_count
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                JOIN messages m ON et.message_id = m.id
                WHERE m.user_id = ? AND m.timestamp >= ?
                GROUP BY t.name, hour HAVING usage_count >= 2
                ORDER BY t.name, hour
            ''', (user_id, cutoff_date.isoformat()))
            
            correlations = {}
            for row in cursor.fetchall():
                tag_name, hour, count = row
                if tag_name not in correlations:
                    correlations[tag_name] = {}
                correlations[tag_name][hour] = count
        else:
            correlations = {"message": f"Correlation type {correlation_type} not implemented yet"}
        
        conn.close()
        return {
            "status": "success",
            "user_id": user_id,
            "correlation_type": correlation_type,
            "analysis_period_days": days_back,
            "correlations": correlations
        }
    except Exception as e:
        logger.error(f"Temporal correlations analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze temporal correlations: {str(e)}")

@app.get("/api/temporal/insights/{user_id}")
async def get_temporal_insights(user_id: str, insight_type: str = "comprehensive", days_back: int = 30):
    """Generate advanced temporal insights and recommendations"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        insights = []
        recommendations = []
        
        # Find peak activity hours
        cursor.execute('''
            SELECT strftime('%H', COALESCE(local_timestamp, timestamp)) as hour,
                   COUNT(*) as entry_count
            FROM messages 
            WHERE user_id = ? AND timestamp >= ?
            GROUP BY hour
            ORDER BY entry_count DESC
            LIMIT 3
        ''', (user_id, cutoff_date.isoformat()))
        
        peak_hours = cursor.fetchall()
        if peak_hours:
            best_hour = peak_hours[0][0]
            insights.append({
                "type": "productivity_peak",
                "title": f"Peak activity at {best_hour}:00",
                "description": f"You're most active around {best_hour}:00",
                "confidence": 0.8
            })
            recommendations.append({
                "type": "scheduling",
                "title": "Optimize your schedule",
                "description": f"Consider important tasks around {best_hour}:00"
            })
        
        conn.close()
        return {
            "status": "success",
            "user_id": user_id,
            "insights": insights,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Temporal insights generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate temporal insights: {str(e)}")

# ========================================
# TIMESTAMP SYNCHRONIZATION API ENDPOINTS
# ========================================

@app.post("/api/timestamp/override")
async def override_entry_timestamp(request: TimestampOverrideRequest):
    """Override an entry's timestamp with manual correction"""
    try:
        logger.info(f"⏰ Overriding timestamp for entry {request.entry_id}")
        
        # Get user_id from entry
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM messages WHERE id = ?", (request.entry_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Entry not found")
        
        user_id = result[0]
        
        # Determine timezone
        timezone_name = request.timezone_name or timezone_manager.get_user_timezone(DATABASE_FILE, user_id)
        
        # Override timestamp
        success = timestamp_synchronizer.override_timestamp(
            entry_id=request.entry_id,
            new_timestamp=request.new_timestamp,
            timezone_name=timezone_name,
            user_id=user_id,
            reason=request.reason or "Manual override"
        )
        
        if success:
            return {
                "status": "success",
                "message": "Timestamp overridden successfully",
                "entry_id": request.entry_id,
                "new_timestamp": request.new_timestamp,
                "timezone": timezone_name,
                "reason": request.reason
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to override timestamp")
            
    except Exception as e:
        logger.error(f"Timestamp override failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to override timestamp: {str(e)}")

@app.put("/api/timestamp/validate")
async def validate_timestamp(request: TimestampValidationRequest):
    """Validate a timestamp against content and user patterns"""
    try:
        logger.info("🔍 Validating timestamp against content")
        
        # Parse timestamp
        if request.timestamp:
            timestamp = datetime.fromisoformat(request.timestamp)
        else:
            timestamp = datetime.utcnow()
        
        # Default timezone if not provided
        timezone_name = request.timezone_name or "America/Chicago"
        
        # For validation, we need a user_id - this would normally come from auth
        # For now, use a default validation without user patterns
        validation = temporal_validator.validate_entry_timestamp(
            content=request.content,
            timestamp=timestamp,
            timezone_name=timezone_name,
            user_id="validation_user"  # Placeholder
        )
        
        return {
            "status": "success",
            "validation": {
                "severity": validation.severity.value,
                "confidence": validation.confidence,
                "message": validation.message,
                "suggested_correction": validation.suggested_correction,
                "metadata": validation.metadata
            },
            "timestamp_analyzed": timestamp.isoformat(),
            "timezone": timezone_name
        }
        
    except Exception as e:
        logger.error(f"Timestamp validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate timestamp: {str(e)}")

@app.put("/api/user/{user_id}/timezone")
async def update_user_timezone(user_id: str, request: TimezoneUpdateRequest):
    """Update user's timezone preference"""
    try:
        logger.info(f"🌍 Updating timezone for user {user_id} to {request.timezone_name}")
        
        # Validate timezone
        if not timezone_manager.validate_timezone(request.timezone_name):
            suggestions = timezone_manager.get_timezone_suggestions(request.timezone_name)
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timezone: {request.timezone_name}. Suggestions: {suggestions[:5]}"
            )
        
        # Update timezone
        success = timezone_manager.set_user_timezone(DATABASE_FILE, user_id, request.timezone_name)
        
        if success:
            return {
                "status": "success",
                "message": "Timezone updated successfully",
                "user_id": user_id,
                "new_timezone": request.timezone_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update timezone")
            
    except Exception as e:
        logger.error(f"Timezone update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update timezone: {str(e)}")

@app.post("/api/timestamp/bulk-correct")
async def bulk_correct_timestamps(request: BulkTimestampCorrectionRequest):
    """Bulk correct timestamps for timezone changes"""
    try:
        logger.info(f"📦 Bulk correcting timestamps for user {request.user_id}")
        
        # Validate timezone
        if not timezone_manager.validate_timezone(request.new_timezone):
            raise HTTPException(status_code=400, detail=f"Invalid timezone: {request.new_timezone}")
        
        # Prepare date range
        date_range = None
        if request.start_date and request.end_date:
            date_range = (request.start_date, request.end_date)
        
        # Perform bulk correction
        result = timestamp_synchronizer.bulk_correct_timestamps(
            user_id=request.user_id,
            timezone_correction=request.new_timezone,
            date_range=date_range
        )
        
        return {
            "status": "success" if result["success"] else "error",
            "corrections_made": result["corrections_made"],
            "total_entries": result.get("total_entries", 0),
            "errors": result.get("errors", []),
            "new_timezone": request.new_timezone,
            "date_range": date_range,
            "reason": request.reason
        }
        
    except Exception as e:
        logger.error(f"Bulk timestamp correction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk correct timestamps: {str(e)}")

@app.get("/api/timezone/suggestions")
async def get_timezone_suggestions(partial_name: Optional[str] = None, country_code: Optional[str] = None):
    """Get timezone suggestions"""
    try:
        suggestions = timezone_manager.get_timezone_suggestions(partial_name, country_code)
        
        return {
            "status": "success",
            "suggestions": suggestions,
            "query": {
                "partial_name": partial_name,
                "country_code": country_code
            }
        }
        
    except Exception as e:
        logger.error(f"Timezone suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timezone suggestions: {str(e)}")

@app.get("/api/timestamp/validation-report/{user_id}")
async def get_validation_report(user_id: str, days_back: int = 7, min_score: float = 0.5):
    """Get timestamp validation report for entries with low confidence scores"""
    try:
        logger.info(f"📊 Generating validation report for user {user_id}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get entries with low validation scores
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        
        cursor.execute("""
            SELECT id, content, utc_timestamp, local_timestamp, timezone_at_creation,
                   temporal_validation_score, timestamp_source
            FROM messages 
            WHERE user_id = ? AND utc_timestamp > ? AND temporal_validation_score < ?
            ORDER BY temporal_validation_score ASC, utc_timestamp DESC
            LIMIT 20
        """, (user_id, cutoff_date, min_score))
        
        entries = cursor.fetchall()
        conn.close()
        
        validation_issues = []
        for entry in entries:
            entry_id, content, utc_ts, local_ts, timezone_name, score, source = entry
            
            # Re-validate entry
            try:
                utc_dt = datetime.fromisoformat(utc_ts)
                validation = temporal_validator.validate_entry_timestamp(
                    content, utc_dt, timezone_name, user_id
                )
                
                validation_issues.append({
                    "entry_id": entry_id,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "utc_timestamp": utc_ts,
                    "local_timestamp": local_ts,
                    "timezone": timezone_name,
                    "validation_score": score,
                    "timestamp_source": source,
                    "current_validation": {
                        "severity": validation.severity.value,
                        "confidence": validation.confidence,
                        "message": validation.message
                    }
                })
            except Exception as e:
                validation_issues.append({
                    "entry_id": entry_id,
                    "error": f"Re-validation failed: {str(e)}"
                })
        
        return {
            "status": "success",
            "user_id": user_id,
            "analysis_period_days": days_back,
            "min_validation_score": min_score,
            "issues_found": len(validation_issues),
            "validation_issues": validation_issues,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Validation report failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate validation report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting enhanced journaling server with temporal awareness and timestamp synchronization on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)