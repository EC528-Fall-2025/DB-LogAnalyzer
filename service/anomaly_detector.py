"""
Anomaly Detection Gate for FoundationDB Log Analysis
Purpose: Filter logs to reduce token cost before passing to LLM

Based on mentor feedback:
- Focus on basic metrics: bytes_in, bytes_out, bytes_written, bytes_read
- Use simple statistical methods (e.g., Z-score) to detect anomalies
- Support both DuckDB aggregation and raw log filtering approaches
"""
from typing import List, Dict, Any, Optional, Tuple
from dto.event import EventModel
import statistics
import math


class MetricAnomalyDetector:
    """
    Simple statistical anomaly detector for FDB metrics.
    Uses Z-score and threshold-based detection.
    """
    
    KEY_METRICS = {
        'BytesInput', 'BytesStored', 'BytesQueried', 
        'BytesInput', 'BytesDurable', 'BytesWritten',
        'FinishedQueries', 'Mutations', 
        'UpdateLatency', 'ReadLatency',
        'Max', 'P99', 'P95',  # Latency percentiles
        'QueryQueue',
    }
    
    INTERESTING_EVENTS = {
        'StorageMetrics', 'DiskMetrics', 'GRVProxyMetrics',
        'UpdateLatencyMetrics', 'ReadLatencyMetrics',
        'CommitLatencyMetrics', 'GetValueMetrics',
    }
    
    def __init__(self, z_score_threshold: float = 1.5):
        """
        Initialize anomaly detector.
        
        Args:
            z_score_threshold: Z-score threshold for anomaly detection (default: 1.5)
                              1.5 means ~13% of data points will be marked as anomalies
                              (more sensitive to catch more potential issues)
        """
        self.z_score_threshold = z_score_threshold
        self.stats = {
            'total_events': 0,
            'filtered_events': 0,
            'anomalies_detected': 0,
            'by_method': {
                'z_score': 0,
                'threshold': 0,
                'interesting_event': 0,
            }
        }
    
    def detect_anomalies(self, events: List[EventModel]) -> List[Tuple[EventModel, List[str]]]:
        """
        Detect anomalies in a list of events.
        
        Args:
            events: List of parsed log events
            
        Returns:
            List of (event, reasons) tuples for anomalous events only
        """
        self.stats['total_events'] = len(events)
        anomalies = []
        
        interesting_events = self._filter_by_event_type(events)
        metric_anomalies = self._detect_metric_anomalies(interesting_events)
        
        for event, reasons in metric_anomalies:
            if reasons:
                anomalies.append((event, reasons))
                self.stats['anomalies_detected'] += 1
        
        self.stats['filtered_events'] = len(events) - len(anomalies)
        return anomalies
    
    def _filter_by_event_type(self, events: List[EventModel]) -> List[EventModel]:
        """Filter events to only include interesting types"""
        filtered = []
        for event in events:
            if event.event in self.INTERESTING_EVENTS:
                filtered.append(event)
                self.stats['by_method']['interesting_event'] += 1
        return filtered if filtered else events  # If no interesting events, return all
    
    def _detect_metric_anomalies(self, events: List[EventModel]) -> List[Tuple[EventModel, List[str]]]:
        """
        Detect anomalies using statistical methods.
        
        This implements the Z-score approach mentioned by the student in the meeting.
        """
        if len(events) < 3:
            return [(e, ['insufficient_data']) for e in events]
        
        metric_values = {}
        event_metrics = {}
        
        for event in events:
            if not event.fields_json:
                continue
            
            event_metrics[event.event_id] = {}
            
            for key, value in event.fields_json.items():
                try:
                    numeric_value = self._parse_numeric(value)
                    if numeric_value is not None and numeric_value > 0:  # Skip zeros and negatives
                        if key not in metric_values:
                            metric_values[key] = []
                        metric_values[key].append(numeric_value)
                        event_metrics[event.event_id][key] = numeric_value
                except:
                    continue
        
        metric_stats = {}
        for metric, values in metric_values.items():
            if len(values) < 3:
                continue
            try:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                metric_stats[metric] = {'mean': mean, 'stdev': stdev, 'values': values}
            except:
                continue
        
        result = []
        for event in events:
            reasons = []
            
            if event.event_id not in event_metrics:
                result.append((event, reasons))
                continue
            
            for key, numeric_value in event_metrics[event.event_id].items():
                if key not in metric_stats:
                    continue
                
                try:
                    stats = metric_stats[key]
                    if stats['stdev'] == 0:
                        continue
                    
                    z_score = abs((numeric_value - stats['mean']) / stats['stdev'])
                    
                    if z_score > self.z_score_threshold:
                        reasons.append(f'z_score_anomaly_{key}')
                        self.stats['by_method']['z_score'] += 1
                    
                    if z_score > 3.0:
                        reasons.append(f'extreme_value_{key}')
                    
                    if self._check_threshold_violation(key, numeric_value):
                        reasons.append(f'threshold_violation_{key}')
                        self.stats['by_method']['threshold'] += 1
                        
                except:
                    continue
            
            result.append((event, reasons))
        
        return result
    
    def _parse_numeric(self, value: Any) -> Optional[float]:
        """Parse a value to numeric, handling various formats"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if ' ' in value:
                parts = value.split()
                try:
                    return max(float(p) for p in parts if p not in ['-1', 'inf'])
                except:
                    return None
            try:
                return float(value)
            except:
                return None
        return None
    
    def _check_threshold_violation(self, metric: str, value: float) -> bool:
        """
        Check if a metric violates known thresholds.
        Based on FDB operational knowledge.
        """
        thresholds = {
            'Max': 1.0,  # Max latency > 1s
            'P99': 0.5,  # P99 latency > 500ms
            'P95': 0.3,  # P95 latency > 300ms
            'QueryQueue': 100,  # Queue depth > 100
        }
        
        return metric in thresholds and value > thresholds[metric]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        stats = self.stats.copy()
        if stats['total_events'] > 0:
            stats['filter_rate'] = stats['filtered_events'] / stats['total_events']
            stats['anomaly_rate'] = stats['anomalies_detected'] / stats['total_events']
        else:
            stats['filter_rate'] = 0.0
            stats['anomaly_rate'] = 0.0
        return stats
    
    def estimate_token_savings(self, avg_tokens_per_event: int = 200) -> Dict[str, Any]:
        """
        Estimate token reduction by filtering.
        
        Addresses two key issues: too many tokens and AI context overflow.
        """
        total_tokens = self.stats['total_events'] * avg_tokens_per_event
        anomaly_tokens = self.stats['anomalies_detected'] * avg_tokens_per_event
        saved_tokens = total_tokens - anomaly_tokens
        
        return {
            'total_events': self.stats['total_events'],
            'anomalies_detected': self.stats['anomalies_detected'],
            'filtered_events': self.stats['filtered_events'],
            'total_tokens_without_filter': total_tokens,
            'total_tokens_with_filter': anomaly_tokens,
            'tokens_saved': saved_tokens,
            'token_reduction_rate': saved_tokens / total_tokens if total_tokens > 0 else 0,
        }


class SimpleLogFilter:
    """
    Simple rule-based log filter.
    This is the "Filter Approach" mentioned by the mentor.
    """
    
    HIGH_SEVERITY = 30  # FDB severity >= 30 indicates warnings/errors
    
    def __init__(self, 
                 filter_by_severity: bool = True,
                 filter_by_event_type: bool = True,
                 interesting_event_types: Optional[set] = None):
        """
        Initialize simple log filter.
        
        Args:
            filter_by_severity: Filter by severity level
            filter_by_event_type: Filter by event type
            interesting_event_types: Set of event types to keep
        """
        self.filter_by_severity = filter_by_severity
        self.filter_by_event_type = filter_by_event_type
        self.interesting_event_types = interesting_event_types or MetricAnomalyDetector.INTERESTING_EVENTS
        
        self.stats = {
            'total': 0,
            'passed': 0,
            'filtered': 0,
        }
    
    def filter_events(self, events: List[EventModel]) -> List[EventModel]:
        """
        Filter events based on simple rules.
        
        Returns:
            List of events that passed the filter
        """
        self.stats['total'] = len(events)
        passed = []
        
        for event in events:
            should_keep = False
            
            if self.filter_by_severity and event.severity and event.severity >= self.HIGH_SEVERITY:
                should_keep = True
            
            if self.filter_by_event_type and event.event in self.interesting_event_types:
                should_keep = True
            
            if should_keep:
                passed.append(event)
        
        self.stats['passed'] = len(passed)
        self.stats['filtered'] = len(events) - len(passed)
        return passed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        stats = self.stats.copy()
        if stats['total'] > 0:
            stats['filter_rate'] = stats['filtered'] / stats['total']
        else:
            stats['filter_rate'] = 0.0
        return stats


class HybridFilter:
    """
    Hybrid approach combining DuckDB-style aggregation and log filtering.
    
    First uses statistical methods to identify anomalies (e.g., metric drops),
    then passes filtered logs to AI for analysis.
    """
    
    def __init__(self, z_score_threshold: float = 2.5):
        self.metric_detector = MetricAnomalyDetector(z_score_threshold)
        self.simple_filter = SimpleLogFilter()
    
    def filter_and_detect(self, events: List[EventModel]) -> Tuple[List[EventModel], List[Tuple[EventModel, List[str]]]]:
        """Apply both filtering and anomaly detection"""
        filtered_events = self.simple_filter.filter_events(events)
        anomalies = self.metric_detector.detect_anomalies(filtered_events)
        
        return filtered_events, anomalies
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from both approaches"""
        simple_stats = self.simple_filter.get_stats()
        metric_stats = self.metric_detector.get_stats()
        token_stats = self.metric_detector.estimate_token_savings()
        
        return {
            'simple_filter': simple_stats,
            'metric_detector': metric_stats,
            'token_savings': token_stats,
            'combined_filter_rate': simple_stats.get('filter_rate', 0),
            'anomaly_detection_rate': metric_stats.get('anomaly_rate', 0),
        }

