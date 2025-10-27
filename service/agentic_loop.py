"""
Agentic Loop Core - End-to-end automation for anomaly detection and recommendation
Purpose: Orchestrate load → filter → detect → match → recommend workflow
"""
from typing import List, Dict, Any, Optional
from dto.event import EventModel
from service.parser import LogParser
from service.anomaly_detector import MetricAnomalyDetector, SimpleLogFilter
from service.recovery_detector import RecoveryDetector, RecoveryEvent
from service.ai_agent import AIAgent
import json


class AgenticLoopResult:
    """Result from an agentic loop execution"""
    
    def __init__(self):
        """Initialize result container"""
        self.total_events = 0
        self.filtered_events = 0
        self.anomalies_detected = 0
        self.recoveries_detected = 0
        self.recommendations: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
    
    def add_recommendation(self, 
                          anomaly_type: str,
                          severity: str,
                          description: str,
                          solution: str,
                          evidence: Optional[Dict] = None):
        """Add a recommendation to results"""
        self.recommendations.append({
            'anomaly_type': anomaly_type,
            'severity': severity,
            'description': description,
            'solution': solution,
            'evidence': evidence or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_events': self.total_events,
            'filtered_events': self.filtered_events,
            'anomalies_detected': self.anomalies_detected,
            'recoveries_detected': self.recoveries_detected,
            'recommendations': self.recommendations,
            'stats': self.stats
        }


class AgenticLoop:
    """
    Core agentic loop orchestrator
    
    Workflow:
    1. Load logs from file
    2. Filter events (reduce noise)
    3. Detect anomalies (metric-based and recovery-based)
    4. AI Agent analyzes and diagnoses issues
    5. Generate intelligent recommendations
    """
    
    def __init__(self, 
                 z_score_threshold: float = 2.0,
                 recovery_lookback: float = 5.0,
                 auto_filter: bool = True,
                 use_ai: bool = True,
                 api_key: Optional[str] = None):
        """
        Initialize agentic loop
        
        Args:
            z_score_threshold: Threshold for metric anomaly detection
            recovery_lookback: Seconds to look back for recovery causes
            auto_filter: Automatically choose appropriate filter
            use_ai: Use AI (Gemini) for deep analysis
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.parser = LogParser()
        self.metric_detector = MetricAnomalyDetector(z_score_threshold)
        self.simple_filter = SimpleLogFilter()
        self.recovery_detector = RecoveryDetector(recovery_lookback)
        self.auto_filter = auto_filter
        self.use_ai = use_ai
        
        # Initialize AI Agent (required for pure AI diagnosis)
        if not use_ai:
            raise RuntimeError(
                "AI Agent is required. This is an AI-driven agentic loop.\n"
                "Set GEMINI_API_KEY environment variable or use --api-key argument.\n"
                "Get API key at: https://aistudio.google.com/"
            )
        
        if not api_key:
            import os
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                raise RuntimeError(
                    "API key not found. Set GEMINI_API_KEY environment variable or use --api-key argument."
                )
        
        try:
            self.ai_agent = AIAgent(api_key=api_key)
        except Exception as e:
            raise RuntimeError(
                f"AI Agent initialization failed: {e}\n"
                "Ensure API key is correct and network is available."
            )
    
    def run(self, 
            log_path: str, 
            limit: Optional[int] = None,
            include_codecoverage: bool = True) -> AgenticLoopResult:
        """
        Run the full agentic loop
        
        Args:
            log_path: Path to log file
            limit: Maximum number of events to process (None = all)
            include_codecoverage: Whether to include CodeCoverage events
            
        Returns:
            AgenticLoopResult with findings and recommendations
        """
        result = AgenticLoopResult()
        
        print(f"🚀 Starting Agentic Loop analysis...")
        print(f"📁 Log file: {log_path}")
        print(f"🔍 Include CodeCoverage: {'Yes' if include_codecoverage else 'No'}")
        print()
        
        print("📥 Step 1/5: Loading log events...")
        events = self._load_events(log_path, limit)
        result.total_events = len(events)
        print(f"   ✓ Loaded {result.total_events} events")
        print()
        
        if not events:
            print("❌ No events found, exiting")
            return result
        
        print("🔧 Step 2/5: Smart filtering events...")
        filtered_events = self._smart_filter(events)
        result.filtered_events = len(filtered_events)
        print(f"   ✓ Remaining after filtering {result.filtered_events} events "
              f"(filtered: {(1 - result.filtered_events/result.total_events)*100:.1f}%)")
        print()
        
        print("🤖 Step 3/5: AI Agent analyzing...")
        print("   → Evaluating log data...")
        
        print("   → Detecting recovery events...")
        recoveries = self.recovery_detector.detect_recoveries(events, include_codecoverage)
        result.recoveries_detected = len(recoveries)
        print(f"   ✓ Found {result.recoveries_detected} recovery events")
        
        print("   → Analyzing metric anomalies...")
        metric_anomalies = self.metric_detector.detect_anomalies(filtered_events)
        result.anomalies_detected = len([a for a in metric_anomalies if a[1]])
        print(f"   ✓ Identified {result.anomalies_detected} metric anomalies")
        print()
        
        print("🧠 Step 4/5: AI Agent generating recommendations...")
        print("   → Integrating analysis results...")
        self._agent_generate_recommendations(metric_anomalies, recoveries, events, result)
        print(f"   ✓ Generated {len(result.recommendations)} recommendations")
        print()
        
        print("📈 Step 5/5: Compiling statistics...")
        result.stats = self._compile_stats()
        print(f"   ✓ Statistics compiled")
        print()
        
        return result
    
    def _load_events(self, log_path: str, limit: Optional[int]) -> List[EventModel]:
        """Load events from log file"""
        events = []
        for i, event in enumerate(self.parser.parse_logs(log_path)):
            if limit and i >= limit:
                break
            events.append(event)
        return events
    
    def _smart_filter(self, events: List[EventModel]) -> List[EventModel]:
        """
        Intelligently filter events based on log characteristics
        
        This implements the "automatic filter selection" requirement
        """
        if not self.auto_filter:
            return events
        
        # Analyze log characteristics
        total_events = len(events)
        high_severity_count = sum(1 for e in events if e.severity and e.severity >= 30)
        recovery_count = sum(1 for e in events if e.event == "MasterRecoveryState")
        codecoverage_count = sum(1 for e in events if e.event == "CodeCoverage")
        
        # Decision logic for filter selection
        if codecoverage_count > total_events * 0.5:
            # Lots of CodeCoverage events - filter them out
            print("   → Many CodeCoverage events detected, will be filtered")
            return [e for e in events if e.event != "CodeCoverage"]
        
        elif high_severity_count > total_events * 0.1:
            # Many high severity events - use severity filter
            print("   → Many high-severity events detected, using severity filter")
            return self.simple_filter.filter_events(events)
        
        else:
            # Default: light filtering
            print("   → Using lightweight filter")
            # Filter out very low severity and boring events
            return [e for e in events 
                   if e.event not in ['BuggifySection'] and 
                      (not e.severity or e.severity >= 10)]
    
    def _agent_generate_recommendations(self,
                                        metric_anomalies: List[tuple],
                                        recoveries: List[RecoveryEvent],
                                        all_events: List[EventModel],
                                        result: AgenticLoopResult):
        """AI Agent generates recommendations using pure AI diagnosis"""
        for recovery in recoveries:
            print(f"   🤖 AI Agent diagnosing recovery (state {recovery.state_code}: {recovery.state_name})...")
            
            agent_result = self.ai_agent.analyze_recovery_with_tools(recovery, all_events)
            
            cause = agent_result.get('cause', 'Unknown')
            confidence = agent_result.get('confidence', 0)
            recommendations = agent_result.get('recommendations') or ['No recommendations']
            reasoning = agent_result.get('reasoning', 'None')
            recommendations_str = '; '.join(recommendations) if recommendations else 'No recommendations'
            
            if confidence >= 80:
                severity = "high"
            elif confidence >= 50:
                severity = "medium"
            else:
                severity = "low"
            
            result.add_recommendation(
                anomaly_type=f"Recovery Event (Pure AI Diagnosis)",
                severity=severity,
                description=f"Recovery event (state {recovery.state_code}: {recovery.state_name})\n🤖 AI diagnosis: {cause}\n📊 Confidence: {confidence}%",
                solution=f"💡 AI recommendations:\n{recommendations_str}\n\n🧠 AI reasoning:\n{reasoning}",
                evidence={
                    'timestamp': str(recovery.timestamp) if recovery.timestamp else None,
                    'state_code': recovery.state_code,
                    'state_name': recovery.state_name,
                    'ai_diagnosis': agent_result,
                    'diagnosis_method': 'Pure AI - Zero heuristics',
                    'related_events_analyzed': len(recovery.related_events)
                }
            )
        
        seen_types = set()
        for event, reasons in metric_anomalies:
            if not reasons:
                continue
            
            anomaly_key = f"metric_{event.event}_{'_'.join(reasons)}"
            if anomaly_key not in seen_types:
                print(f"   🤖 AI analyzing metric anomaly: {event.event}...")
                
                agent_result = self.ai_agent.analyze_anomaly_with_agent(event, reasons, all_events)
                recommendations = agent_result.get('recommendations') or []
                recommendations_str = '; '.join(recommendations) if recommendations else 'No specific recommendations'
                
                result.add_recommendation(
                    anomaly_type=f"Metric Anomaly (AI Agent)",
                    severity=agent_result.get('severity', 'medium'),
                    description=f"Detected metric anomaly: {event.event}. AI Agent root cause: {agent_result.get('root_cause', 'Unknown')}",
                    solution=f"AI Agent recommendations: {recommendations_str}",
                    evidence={
                        'event_type': event.event,
                        'timestamp': str(event.ts) if event.ts else None,
                        'reasons': reasons,
                        'agent_analysis': agent_result
                    }
                )
                seen_types.add(anomaly_key)
    
    def _compile_stats(self) -> Dict[str, Any]:
        """Compile all statistics"""
        return {
            'metric_detector': self.metric_detector.get_stats(),
            'recovery_detector': self.recovery_detector.get_stats(),
            'simple_filter': self.simple_filter.get_stats(),
            'token_savings': self.metric_detector.estimate_token_savings()
        }
    
    def print_results(self, result: AgenticLoopResult):
        """Print results in a readable format"""
        print("=" * 80)
        print("🎯 AGENTIC LOOP ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        # Summary statistics
        print("📊 Overall Statistics:")
        print(f"   • Total events: {result.total_events}")
        print(f"   • Filtered events: {result.filtered_events}")
        print(f"   • Anomalies detected: {result.anomalies_detected}")
        print(f"   • Recoveries detected: {result.recoveries_detected}")
        print(f"   • Recommendations generated: {len(result.recommendations)}")
        print()
        
        # Recommendations
        if result.recommendations:
            print("💡 Detected Issues and Recommendations:")
            print()
            
            for i, rec in enumerate(result.recommendations, 1):
                severity_emoji = {
                    'low': '🟢',
                    'medium': '🟡',
                    'high': '🟠',
                    'critical': '🔴'
                }.get(rec['severity'], '⚪')
                
                print(f"{i}. {severity_emoji} [{rec['severity'].upper()}] {rec['anomaly_type']}")
                print(f"   Description: {rec['description']}")
                print(f"   Recommendations: {rec['solution']}")
                
                if rec.get('evidence'):
                    evidence = rec['evidence']
                    if 'timestamp' in evidence and evidence['timestamp']:
                        print(f"   Time: {evidence['timestamp']}")
                    if 'event_type' in evidence:
                        print(f"   Event type: {evidence['event_type']}")
                
                print()
        else:
            print("✅ No obvious issues detected! System appears healthy.")
            print()
        
        # Token savings
        if 'token_savings' in result.stats:
            ts = result.stats['token_savings']
            print("💰 Token savings (for LLM analysis):")
            print(f"   • Original tokens: {ts['total_tokens_without_filter']:,}")
            print(f"   • Filtered tokens: {ts['total_tokens_with_filter']:,}")
            print(f"   • Tokens saved: {ts['tokens_saved']:,} ({ts['token_reduction_rate']*100:.1f}%)")
            print()
        
        print("=" * 80)

