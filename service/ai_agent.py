"""
AI Agent - Central decision maker for the agentic loop
Purpose: AI agent orchestrates all analysis, calling tools as needed
"""
from typing import List, Dict, Any, Optional
import json
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from dto.event import EventModel
from service.recovery_detector import RecoveryDetector, RecoveryEvent


# Structured output schemas
class ToolInsights(BaseModel):
    """Insights from different analysis tools"""
    heuristic_findings: str = Field(description="What patterns were detected by heuristic analysis", default="")
    ai_pattern_match: str = Field(description="Pattern matching from AI analysis", default="")
    deep_analysis: str = Field(description="Insights from deep AI reasoning", default="")


class RecoveryAnalysisResult(BaseModel):
    """Structured output for recovery analysis"""
    analysis_approach: str = Field(description="Analysis method: Pure AI forensic analysis")
    cause: str = Field(description="The likely root cause of the recovery")
    confidence: int = Field(description="Confidence score 0-100", ge=0, le=100)
    evidence: List[str] = Field(description="Key evidence points", default_factory=list)
    recommendations: List[str] = Field(description="Specific actionable recommendations", min_items=1)
    reasoning: str = Field(description="Step-by-step reasoning process")
    tool_insights: ToolInsights = Field(
        description="Insights from each analysis tool",
        default_factory=ToolInsights
    )


class AnomalyAnalysisResult(BaseModel):
    """Structured output for anomaly analysis"""
    root_cause: str = Field(description="The underlying issue")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    impact: str = Field(description="Description of business impact")
    recommendations: List[str] = Field(description="Specific actions to take", min_items=1)
    reasoning: str = Field(description="Analysis process and logic")


class AIAgent:
    """
    AI Agent that orchestrates the entire analysis process
    
    Pure AI-driven forensic analysis without heuristic shortcuts
    """
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initialize AI agent
        
        Args:
            api_key: Google Gemini API key
            model: Model name (default: gemini-2.5-flash)
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-genai not installed")
        
        self.api_key = api_key
        self.model_name = model
        self.client = genai.Client(api_key=api_key)
        
        # Heuristic detector for tool access
        self.recovery_detector = RecoveryDetector()
        
        print(f"🤖 AI Agent initialized (model: {self.model_name})")
        print(f"   Mode: Pure AI forensic analysis")
    
    def analyze_recovery_with_tools(self, 
                                    recovery: RecoveryEvent,
                                    all_events: List[EventModel]) -> Dict[str, Any]:
        """
        AI agent analyzes recovery using available tools
        
        Args:
            recovery: Recovery event to analyze
            all_events: All events for context
            
        Returns:
            Complete analysis with cause, recommendations, and reasoning
        """
        # Build context for the agent
        context = self._build_agent_context(recovery, all_events)
        
        # Create the agent prompt with tool descriptions
        prompt = f"""You are an expert AI agent analyzing FoundationDB logs. Your goal is to diagnose the cause of a Master Recovery event and provide actionable recommendations.

## Recovery Event Details
- Time: {recovery.timestamp}
- State Code: {recovery.state_code} ({recovery.state_name})
- Related Events: {len(recovery.related_events)} events in the 5 seconds before recovery

## Your Mission

You are the ONLY diagnostic authority. No pre-analysis has been done.
Your task: Examine the raw event sequence and determine what caused this recovery.

**No CodeCoverage hints. No heuristic shortcuts. Pure forensic analysis.**

## Event Context (last 10 events before recovery)
{context}

## Analysis Guidelines

1. **Look for Failure Signals**
   - Errors, exceptions, timeouts
   - Component disconnections or restarts
   - Configuration changes
   - Resource exhaustion (memory, disk, connections)
   - Network issues

2. **Identify Patterns**
   - What changed right before recovery?
   - Are there any high-severity events?
   - Do you see process failures, file errors, or crashes?
   - Are there timing correlations?

3. **Consider FoundationDB Architecture**
   - Master, TLog, Storage, Proxy roles
   - Coordinators and cluster coordination
   - Recovery triggers (component failure, configuration change, etc.)

4. **Infer the Root Cause**
   - What is the most likely explanation?
   - What evidence supports your conclusion?
   - How confident are you (0-100%)?

Be thorough, specific, and actionable. This analysis will be used by operators to fix production issues.

Provide your analysis with:
- analysis_approach: "Pure AI forensic analysis"
- cause: The likely root cause (be specific)
- confidence: Your confidence score 0-100
- evidence: List of key evidence from the events you examined
- recommendations: List of 2-4 specific, actionable fixes
- reasoning: Your step-by-step forensic analysis
- tool_insights: Object with:
  - heuristic_findings: "N/A - Pure AI analysis"
  - ai_pattern_match: "N/A - Pure AI analysis"  
  - deep_analysis: Summary of your investigation process
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                    response_schema=RecoveryAnalysisResult,
                )
            )
            
            # With structured output, response.text is guaranteed to be valid JSON
            response_text = response.text if response and hasattr(response, 'text') and response.text else ""
            if not response_text:
                return {
                    'analysis_approach': 'error',
                    'cause': 'Empty AI response',
                    'confidence': 0,
                    'evidence': [],
                    'recommendations': ['Retry analysis'],
                    'reasoning': 'No response text from AI',
                    'tool_insights': {}
                }
            
            # Parse structured JSON response with detailed error reporting
            result = self._safe_json_parse(response_text, "Recovery analysis")
            if result is None:
                # Parsing failed, return safe fallback
                return {
                    'analysis_approach': 'error',
                    'cause': 'JSON parsing failed - see logs above',
                    'confidence': 0,
                    'evidence': [],
                    'recommendations': ['Check API response format', 'Retry with different prompt'],
                    'reasoning': 'Structured output parsing failed despite API schema',
                    'tool_insights': {
                        'heuristic_findings': '',
                        'ai_pattern_match': '',
                        'deep_analysis': ''
                    }
                }
            
            # Ensure all required fields exist
            result.setdefault('evidence', [])
            result.setdefault('recommendations', ['No recommendations'])
            if 'tool_insights' not in result:
                result['tool_insights'] = {
                    'heuristic_findings': '',
                    'ai_pattern_match': '',
                    'deep_analysis': ''
                }
            
            # Add metadata
            result['agent_model'] = self.model_name
            result['agent_mode'] = 'tool-augmented'
            
            return result
            
        except Exception as e:
            print(f"❌ AI Agent analysis failed: {e}")
            return {
                'analysis_approach': 'error',
                'cause': f'Agent error: {str(e)}',
                'confidence': 0,
                'evidence': [],
                'recommendations': ['Check agent configuration and retry'],
                'reasoning': f'Exception during analysis: {str(e)}'
            }
    
    def analyze_anomaly_with_agent(self,
                                   event: EventModel,
                                   anomaly_reasons: List[str],
                                   context_events: List[EventModel]) -> Dict[str, Any]:
        """
        AI agent analyzes metric anomalies
        
        Args:
            event: The anomalous event
            anomaly_reasons: Detected anomaly indicators
            context_events: Recent events for context
            
        Returns:
            Analysis with root cause and recommendations
        """
        # Build context
        event_str = f"Type: {event.event}, Time: {event.ts}, Severity: {event.severity}"
        reasons_str = ", ".join(anomaly_reasons) if anomaly_reasons else "No specific reasons"
        
        if context_events and len(context_events) > 0:
            context_str = "\n".join([
                f"- {e.event} at {e.ts} (severity: {e.severity})"
                for e in context_events[-5:] if e.ts
            ])
        else:
            context_str = "No context events"
        
        prompt = f"""You are an expert AI agent analyzing FoundationDB metric anomalies.

## Anomaly Details
Event: {event_str}
Detected Anomalies: {reasons_str}

## Recent Context
{context_str}

## Available Analysis Tools

1. **Statistical Analysis** - Z-score based anomaly detection (already detected the anomaly)
2. **Pattern Recognition** - Identify common failure patterns
3. **Historical Analysis** - Compare with past incidents
4. **Your Deep Analysis** - Understand the business impact and root cause

## Your Task

Analyze this metric anomaly and provide:

Provide concise analysis:
1. root_cause: Brief description of the underlying issue (1-2 sentences)
2. severity: One of low/medium/high/critical
3. impact: Brief business impact (1 sentence)
4. recommendations: List of 2-4 specific, actionable steps
5. reasoning: Concise analysis (2-3 sentences max)

Keep responses brief and actionable - avoid lengthy explanations.
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1500,  # Increased from 1000 to prevent truncation
                    response_mime_type="application/json",
                    response_schema=AnomalyAnalysisResult,
                )
            )
            
            # With structured output, response.text is guaranteed to be valid JSON
            response_text = response.text if response and hasattr(response, 'text') and response.text else ""
            if not response_text:
                return {
                    'root_cause': 'Empty AI response',
                    'severity': 'medium',
                    'impact': 'Unknown',
                    'recommendations': ['Retry analysis'],
                    'reasoning': 'No response text from AI'
                }
            
            # Parse structured JSON response with detailed error reporting
            result = self._safe_json_parse(response_text, "Anomaly analysis")
            if result is None:
                # Parsing failed, return safe fallback
                return {
                    'root_cause': 'JSON parsing failed - see logs above',
                    'severity': 'medium',
                    'impact': 'Unknown',
                    'recommendations': ['Check API response format', 'Retry analysis'],
                    'reasoning': 'Structured output parsing failed despite API schema'
                }
            
            # Ensure all required fields exist
            result.setdefault('recommendations', ['No recommendations'])
            result.setdefault('severity', 'medium')
            
            return result
            
        except Exception as e:
            return {
                'root_cause': f'Analysis error: {str(e)}',
                'severity': 'medium',
                'impact': 'Unknown',
                'recommendations': ['Retry analysis'],
                'reasoning': 'Exception occurred'
            }
    
    def _build_agent_context(self, 
                            recovery: RecoveryEvent,
                            all_events: List[EventModel]) -> str:
        """Build context string for agent"""
        lines = []
        
        # Get events before recovery
        if recovery.timestamp:
            recent = [e for e in all_events if e.ts and e.ts < recovery.timestamp][-10:]
        else:
            recent = all_events[-10:] if all_events else []
        
        for i, event in enumerate(recent, 1):
            lines.append(
                f"{i}. [{event.event}] at {event.ts} "
                f"(severity: {event.severity}, role: {event.role})"
            )
            if event.fields_json:
                # Show a few important fields
                important = {k: v for k, v in list(event.fields_json.items())[:3]}
                if important:
                    lines.append(f"   Fields: {json.dumps(important, ensure_ascii=False)}")
        
        return '\n'.join(lines) if lines else "No events found"
    
    def _safe_json_parse(self, response_text: str, context: str = "response") -> Optional[Dict[str, Any]]:
        """
        Safely parse JSON with better error handling
        
        Args:
            response_text: JSON string to parse
            context: Context for error messages
            
        Returns:
            Parsed dict or None if parsing fails
        """
        if not response_text:
            print(f"⚠️  {context}: Empty response")
            return None
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️  {context} JSON parse error: {e}")
            print(f"   Error at line {e.lineno}, col {e.colno}")
            
            lines = response_text.split('\n')
            if e.lineno <= len(lines):
                error_line = lines[e.lineno - 1]
                print(f"   Error line: {error_line[:100]}...")
            
            if len(response_text) > 200:
                print(f"   Start: {response_text[:100]}...")
                print(f"   End: ...{response_text[-100:]}")
            else:
                print(f"   Full response: {response_text}")
            
            return None
    
    def _parse_agent_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse agent's JSON response (legacy fallback)
        
        Note: With structured output, this is rarely needed.
        Kept as fallback for edge cases or non-structured responses.
        """
        # Safety check
        if not response_text or response_text is None:
            return {
                'cause': 'Empty response',
                'confidence': 0,
                'recommendations': ['Retry analysis'],
                'reasoning': 'Response text is empty or None'
            }
        
        try:
            # Extract JSON from markdown
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text
            
            parsed = json.loads(json_text)
            
            # Ensure recommendations is always a list, never None
            if 'recommendations' in parsed and parsed['recommendations'] is None:
                parsed['recommendations'] = []
            elif 'recommendations' not in parsed:
                parsed['recommendations'] = []
                
            return parsed
        except json.JSONDecodeError:
            # Fallback: return raw text
            return {
                'cause': response_text[:200] if len(response_text) > 200 else response_text,
                'confidence': 50,
                'recommendations': ['Review agent response manually'],
                'reasoning': 'Failed to parse JSON response',
                'raw_response': response_text
            }

