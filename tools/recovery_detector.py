"""
Recovery event detector for FDB logs.
Purpose: Detect and analyze recovery events, especially MasterRecoveryState
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from data_transfer_object.event_dto import EventModel


class RecoveryEvent:
    """Represents a recovery event"""
    
    def __init__(self, 
                 start_event: EventModel,
                 state_code: int,
                 state_name: str,
                 cause: Optional[str] = None,
                 related_events: Optional[List[EventModel]] = None):
        """
        Initialize recovery event
        
        Args:
            start_event: The event that started this recovery
            state_code: Recovery state code (0-14)
            state_name: Recovery state name
            cause: Detected cause of recovery
            related_events: Related events (CodeCoverage, failures, etc.)
        """
        self.start_event = start_event
        self.state_code = state_code
        self.state_name = state_name
        self.cause = cause
        self.related_events = related_events or []
        self.timestamp = start_event.ts
    
    def __repr__(self):
        return f"RecoveryEvent(state={self.state_code}:{self.state_name}, cause={self.cause}, time={self.timestamp})"


class RecoveryDetector:
    """Detector for recovery events in FDB logs"""
    
    # Recovery state codes and their meanings
    RECOVERY_STATES = {
        0: "reading_coordinated_state",
        1: "locking_coordinated_state",
        2: "recruiting_proxies",
        3: "reading_transaction_system_state",
        4: "configuration_missing",
        5: "configuration_never_created",
        6: "configuration_invalid",
        7: "recruiting_transaction_servers",
        8: "initializing_transaction_servers",
        9: "recovery_transaction",
        10: "writing_coordinated_state",
        11: "accepting_commits",
        12: "all_logs_recruited",
        13: "storage_recovered",
        14: "fully_recovered"
    }
    
    # Known recovery causes from CodeCoverage comments
    KNOWN_CAUSES = [
        "Terminated due to tLog failure",
        "Terminated due to storage server failure",
        "Terminated due to commit proxy failure",
        "Terminated due to GRV proxy failure",
        "Terminated due to resolver failure",
        "Terminated due to master failure",
        "Terminated due to coordinator failure",
        "Configuration change",
        "Manual recovery",
        "Network partition",
        "Datacenter failure",
    ]
    
    def __init__(self, look_back_seconds: float = 5.0):
        """
        Initialize recovery detector
        
        Args:
            look_back_seconds: How far back to look for cause events before recovery
        """
        self.look_back_seconds = look_back_seconds
        self.stats = {
            'total_recoveries': 0,
            'recoveries_with_cause': 0,
            'recoveries_without_cause': 0,
        }
    
    def detect_recoveries(self, events: List[EventModel], 
                         include_codecoverage: bool = True) -> List[RecoveryEvent]:
        """
        Detect all recovery events in the log stream
        
        Args:
            events: List of log events
            include_codecoverage: Whether to include CodeCoverage events in analysis
            
        Returns:
            List of detected RecoveryEvent objects
        """
        recoveries = []
        
        # Filter out CodeCoverage if requested (for AI challenge)
        if not include_codecoverage:
            events = [e for e in events if e.event != "CodeCoverage"]
        
        # Build index of events by time for efficient lookup
        events_by_time = sorted(events, key=lambda e: e.ts if e.ts else datetime.min)
        
        for i, event in enumerate(events_by_time):
            if event.event == "MasterRecoveryState":
                # Extract recovery state
                state_code = self._extract_state_code(event)
                if state_code is None:
                    continue
                
                state_name = self.RECOVERY_STATES.get(state_code, "unknown")
                
                # Look for cause in previous events
                cause = self._find_recovery_cause(event, events_by_time[:i], include_codecoverage)
                
                # Find related events
                related = self._find_related_events(event, events_by_time, i)
                
                recovery = RecoveryEvent(
                    start_event=event,
                    state_code=state_code,
                    state_name=state_name,
                    cause=cause,
                    related_events=related
                )
                
                recoveries.append(recovery)
                
                self.stats['total_recoveries'] += 1
                if cause:
                    self.stats['recoveries_with_cause'] += 1
                else:
                    self.stats['recoveries_without_cause'] += 1
        
        return recoveries
    
    def _extract_state_code(self, event: EventModel) -> Optional[int]:
        """Extract recovery state code from event"""
        if event.fields_json and 'StatusCode' in event.fields_json:
            try:
                return int(event.fields_json['StatusCode'])
            except (ValueError, TypeError):
                pass
        return None
    
    def _find_recovery_cause(self, 
                            recovery_event: EventModel, 
                            previous_events: List[EventModel],
                            include_codecoverage: bool) -> Optional[str]:
        """
        Find the likely cause of a recovery by looking at previous events
        
        Args:
            recovery_event: The recovery event
            previous_events: Events before the recovery
            include_codecoverage: Whether to check CodeCoverage events
            
        Returns:
            Cause description if found, None otherwise
        """
        if not recovery_event.ts:
            return None
        
        # Look back for potential causes
        cutoff_time = recovery_event.ts - timedelta(seconds=self.look_back_seconds)
        
        recent_events = [
            e for e in previous_events 
            if e.ts and e.ts >= cutoff_time
        ]
        
        # Check CodeCoverage events first (they often contain explicit cause)
        if include_codecoverage:
            for event in reversed(recent_events):
                if event.event == "CodeCoverage":
                    comment = event.fields_json.get('Comment', '') if event.fields_json else ''
                    # Check for known causes
                    for known_cause in self.KNOWN_CAUSES:
                        if known_cause.lower() in comment.lower():
                            return comment
        
        # Check for failure events
        for event in reversed(recent_events):
            if not event.event:
                continue
            
            event_lower = event.event.lower()
            
            # Check for specific failure types
            if 'fail' in event_lower or 'error' in event_lower or 'terminated' in event_lower:
                return f"Detected failure event: {event.event}"
            
            # Check for high severity events
            if event.severity and event.severity >= 40:  # Error severity
                return f"High severity event: {event.event} (severity {event.severity})"
        
        # If we can't find a specific cause
        return None
    
    def _find_related_events(self, 
                            recovery_event: EventModel,
                            all_events: List[EventModel],
                            recovery_index: int) -> List[EventModel]:
        """
        Find events related to this recovery
        
        Args:
            recovery_event: The recovery event
            all_events: All events
            recovery_index: Index of recovery event in all_events
            
        Returns:
            List of related events
        """
        if not recovery_event.ts:
            return []
        
        related = []
        
        # Look at events within the time window
        cutoff_time = recovery_event.ts - timedelta(seconds=self.look_back_seconds)
        
        # Check events before recovery
        for i in range(max(0, recovery_index - 100), recovery_index):
            event = all_events[i]
            if event.ts and event.ts >= cutoff_time:
                # Include high severity, failures, or interesting events
                if (event.severity and event.severity >= 30) or \
                   (event.event and any(k in event.event.lower() 
                                       for k in ['fail', 'error', 'terminated', 'codecoverage'])):
                    related.append(event)
        
        return related
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = self.stats.copy()
        if stats['total_recoveries'] > 0:
            stats['cause_detection_rate'] = stats['recoveries_with_cause'] / stats['total_recoveries']
        else:
            stats['cause_detection_rate'] = 0.0
        return stats

