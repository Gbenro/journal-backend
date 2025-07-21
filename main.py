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

# Configure logging - Last updated: 2025-07-21 for Railway deployment
# This ensures proper logging across development and production environments
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Enhanced fluency models
class EntryUpdateRequest(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    energy_signature: Optional[str] = None
    intention_flag: Optional[bool] = None

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
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
        
        # Add enhanced columns to messages table if they don't exist
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN intention_flag BOOLEAN DEFAULT FALSE")
        except:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN manual_energy_signature TEXT")
        except:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN relationship_mentions JSON")
        except:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE messages ADD COLUMN revision_count INTEGER DEFAULT 0")
        except:
            pass  # Column already exists
        
        # Insert predefined tags if they don't exist
        for tag_data in PREDEFINED_TAGS:
            cursor.execute("""
                INSERT OR IGNORE INTO tags (name, category, is_predefined, color, description)
                VALUES (?, ?, TRUE, ?, ?)
            """, (tag_data["name"], tag_data["category"], tag_data["color"], tag_data["description"]))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Enhanced SQLite database with tags initialized successfully at: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed at {db_path}: {e}")
        return False

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
    """Initialize database on startup with comprehensive logging
    Railway deployment test - 2025-07-21 - Verifying persistent volume mount"""
    logger.info("🚀 Starting Mirror Scribe Backend with Intelligent Tags...")
    logger.info(f"📁 Environment: {'Railway' if is_railway_environment() else 'Local'}")
    logger.info(f"💾 Database path: {get_database_path()}")
    logger.info(f"📂 Data directory: {'/app/data' if is_railway_environment() else 'local'}")
    
    # Initialize database
    success = init_database()
    
    if success:
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
            conn.close()
            
            logger.info(f"📝 Existing journal entries: {message_count}")
            logger.info(f"🏷️ Available tags: {tag_count}")
            logger.info(f"🔗 Tag applications: {tag_applications}")
        except Exception as e:
            logger.warning(f"⚠️ Could not check existing data: {e}")
        
        logger.info("✅ Mirror Scribe Backend ready with persistent storage!")
    else:
        logger.warning("⚠️ Database initialization failed, but continuing...")

@app.get("/health")
async def health_check():
    """Enhanced health check with persistent storage info"""
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
        logger.error(f"Health check failed: {e}")
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

@app.post("/api/save")
async def save_message(message: MessageRequest):
    """Save a journal entry with enhanced tag support"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute(
            "INSERT INTO messages (content, user_id, timestamp) VALUES (?, ?, ?)",
            (message.content, message.user_id, datetime.utcnow().isoformat())
        )
        
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
        
        # Add auto tags if enabled
        if message.auto_tag:
            tagger = AutoTagger()
            auto_tags = tagger.apply_auto_tags(message.content, message.manual_tags)
            applied_tags.extend(auto_tags)
        
        # Apply all tags to the entry
        apply_tags_to_entry(conn, message_id, applied_tags)
        
        # Extract and track relationships
        relationship_analyzer = RelationshipAnalyzer()
        relationships = relationship_analyzer.extract_relationships(message.content)
        relationship_names = [rel.get("name") for rel in relationships if rel.get("name")]
        
        # Update relationship mentions in message
        cursor.execute("""
            UPDATE messages SET relationship_mentions = ? WHERE id = ?
        """, (json.dumps(relationship_names), message_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Update relationship tracking (done after commit to avoid long transactions)
        if relationships:
            entry_timestamp = datetime.utcnow().isoformat()
            relationship_analyzer.update_relationship_tracking(message.user_id, message_id, relationships, entry_timestamp)
        
        logger.info(f"💾 Saved message {message_id} with {len(applied_tags)} tags for user {message.user_id}")
        
        return {
            "status": "success",
            "message_id": message_id,
            "timestamp": datetime.utcnow().isoformat(),
            "applied_tags": applied_tags,
            "tag_count": len(applied_tags)
        }
        
    except Exception as e:
        logger.error(f"Save message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save message: {str(e)}")

@app.get("/api/messages/{user_id}")
async def get_messages(user_id: str, limit: int = 100, offset: int = 0, tags: Optional[str] = None):
    """Get messages for a user with optional tag filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if tags:
            # Filter by tags
            tag_list = [tag.strip() for tag in tags.split(",")]
            placeholders = ",".join("?" * len(tag_list))
            
            cursor.execute(f"""
                SELECT DISTINCT m.id, m.content, m.user_id, m.timestamp
                FROM messages m
                JOIN entry_tags et ON m.id = et.message_id
                JOIN tags t ON et.tag_id = t.id
                WHERE m.user_id = ? AND t.name IN ({placeholders})
                ORDER BY m.timestamp DESC
                LIMIT ? OFFSET ?
            """, [user_id] + tag_list + [limit, offset])
        else:
            # Get all messages
            cursor.execute(
                """SELECT id, content, user_id, timestamp 
                   FROM messages 
                   WHERE user_id = ? 
                   ORDER BY timestamp DESC 
                   LIMIT ? OFFSET ?""",
                (user_id, limit, offset)
            )
        
        rows = cursor.fetchall()
        
        # Get tags for each message
        messages = []
        for row in rows:
            message = dict(row)
            
            # Get tags for this message
            cursor.execute("""
                SELECT t.name, t.color, t.category, et.confidence, et.is_auto_tagged
                FROM tags t
                JOIN entry_tags et ON t.id = et.tag_id
                WHERE et.message_id = ?
            """, (message["id"],))
            
            tags_data = cursor.fetchall()
            message["tags"] = [dict(tag) for tag in tags_data]
            messages.append(message)
        
        cursor.close()
        conn.close()
        
        logger.info(f"📖 Retrieved {len(messages)} messages for user {user_id}")
        
        return {
            "status": "success",
            "messages": messages,
            "count": len(messages),
            "filtered_by_tags": tags.split(",") if tags else None
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 Starting enhanced tagging server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)