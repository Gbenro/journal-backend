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