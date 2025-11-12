"""
Agentic Loop Core - End-to-end automation for anomaly detection and recommendation
Purpose: Orchestrate load → filter → detect → match → recommend workflow
"""
import duckdb
import glob
import os
import sys
import json
from typing import Optional
from service.chunker import process_logs  
from service.storage import StorageService
import google.generativeai as genai

class AgenticLoop:

    def __init__(self, directory_path: str, 
                 db_path: Optional[str] = None,
                 api_key: Optional[str] = None, 
                 schema_design: Optional[str] = None,
                 use_ai: bool =  True):
        self.directory_path = directory_path
        self.db_path = db_path or "fdb_logs.duckdb"
        self.schema_design = "/Users/vanshikachaddha/Documents/Boston University/Fourth Year/Cloud Computing/DB-LogAnalyzer/data/schema.sql"
        self.use_ai = use_ai
        self._initialize_gemini()

        self._load_events()
    
    def _initialize_gemini(self):
        """Initialize the Gemini API with the API key from environment"""
        # Read your API key from environment (already exported)
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not set. Run `export GEMINI_API_KEY='your-key'` first.")

        # Configure Gemini client
        genai.configure(api_key=api_key)

        # Load the embedding model
        self.embedding_model = genai.GenerativeModel("text-embedding-004")

        print("✅ Gemini initialized successfully!")

    def _load_events(self):
        """Handles loading log files and populating the database"""

        if not os.path.exists(self.directory_path):
            print(f"Error: Directory path does not exist: {self.directory_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Target database: {self.directory_path}")
        files_to_load = []

        files_to_load = sorted(glob.glob((os.path.join(self.directory_path, "*.xml"))))
        print(f"Following {len(files_to_load)} files loaded: {files_to_load}")
        if not files_to_load:
            print(f"No .xml logs found in directory: {self.directory_path}")
        
        storage_service = StorageService(self.db_path)

        # Initialize database (if needed)
        if self.schema_design:
            storage_service.init_db(self.schema_design)
        else:
            storage_service.init_db()
        
        # Check if data already exists
        count = 0
        # if storage_service.check_events_loaded():
        #     count = storage_service.get_event_count()
        #     response = input(f"Database already contains {count} events, continue loading? (y/n): ")
        #     if response.lower() != 'y':
        #         print("Operation cancelled")
        #         storage_service.close()
        #         return      
        
        # Load logs from files_to_load
        total_loaded = 0
        try:
            for path in files_to_load:
                new_events_loaded = storage_service.load_logs_from_file(log_path = path, event_id_offset = count)
                count += new_events_loaded
                total_loaded += new_events_loaded
                print(f"Successfully loaded {new_events_loaded} events from {path}!")
        except Exception as e:
            print(f"Load failed: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            storage_service.close()

        print(f"Successfully loaded {total_loaded} events from {len(files_to_load)} file(s)!")
        self.process_logs(self.db_path)
    
    def process_logs(self, db_path: str):
        """
        Fetch logs from DuckDB, chunk, split, and store them in DuckDB.
        """
        print("Starting chunking process...")
        
        # Fetch logs from DuckDB
        logs = self.fetch_logs_from_duckdb(db_path)

        # Split logs into chunks (assuming the chunking logic is defined)
        log_chunks = self.chunk_logs(logs)

        # Generate embeddings for each chunk
        chunks_with_embeddings = self.generate_embeddings(log_chunks)

        # Store embeddings in DuckDB
        self.store_embeddings(db_path, chunks_with_embeddings)

        print("Processing and storing chunks completed.")

    def fetch_logs_from_duckdb(self, db_path: str):
        """Fetch logs from DuckDB in batches"""
        connection = duckdb.connect(db_path)
        
        query = "SELECT event_id, event, raw_json FROM events"
        result = connection.execute(query).fetchall()
        logs = [{"event_id": row[0], "event": row[1], "raw_json": json.loads(row[2])} for row in result]
        
        connection.close()
        
        return logs

    def chunk_logs(self, logs):
        """Chunk logs into smaller parts based on some logic"""
        # For simplicity, let's assume each log entry is a chunk here
        # In a real scenario, you can chunk logs based on time, event types, etc.
        chunks = []
        for log in logs:
            event_text = log["event"]
            chunk_size = 512  # Set chunk size limit for Gemini embedding input
            chunks.extend([event_text[i:i + chunk_size] for i in range(0, len(event_text), chunk_size)])
        return chunks


    def generate_embeddings(self, log_chunks):
        """Generate embeddings for each log chunk using Gemini API (with retry + throttle)."""
        print(f"Generating embeddings for {len(log_chunks)} chunks...")

        embeddings = []
        for idx, chunk in enumerate(log_chunks):
            # Gemini API requires non-empty strings
            if not chunk.strip():
                continue

            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk
                    )
                    embeddings.append(response["embedding"])
                    # Random small sleep to avoid rate limit bursts
                    time.sleep(random.uniform(0.2, 0.6))
                    break  # success → exit retry loop
                except Exception as e:
                    print(f"[Attempt {attempt+1}/3] Embedding failed for chunk {idx}: {e}")
                    time.sleep(1.5 * (attempt + 1))  # exponential backoff
            else:
                print(f"⚠️ Giving up on chunk {idx} after 3 failed attempts.")

        print(f"✅ Successfully embedded {len(embeddings)} / {len(log_chunks)} chunks.")
        return embeddings
    
    def store_embeddings(self, db_path: str, chunks_with_embeddings):
        """Store embeddings into DuckDB"""
        connection = duckdb.connect(db_path)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                event_id VARCHAR,
                embedding JSON
            );
        """)

        for idx, embedding in enumerate(chunks_with_embeddings):
            insert_query = """
                INSERT INTO chunk_embeddings (event_id, embedding)
                VALUES (?, ?);
            """
            connection.execute(insert_query, (f"event_{idx}", json.dumps(embedding)))  # Store as JSON

        connection.close()
        print(f"✅ Stored embeddings for {len(chunks_with_embeddings)} chunks.")



def main():
    """
    Main function to test the AgenticLoop class and its methods.
    """
    # Path to your log directory and DuckDB database
    log_directory_path = "/Users/vanshikachaddha/Documents/Boston University/Fourth Year/Cloud Computing/DB-LogAnalyzer/logs"  # Replace with actual path to log files 
    # Initialize AgenticLoop instance
    agent = AgenticLoop(directory_path=log_directory_path)
    
    # Test the process_logs function
    #agent._load_events()  # This will load events and call process_logs internally


    # Connect to the DuckDB file
    conn = duckdb.connect('fdb_logs.duckdb')

    # Query to list all tables
    tables = conn.execute("SHOW TABLES").fetchall()

    # Print the tables
    print("Tables in the database:")
    for table in tables:
        print(table[0])  # Print table names

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()






            
            



        



# class AgenticLoopResult:
#     """Result from an agentic loop execution"""
    
#     def __init__(self):
#         """Initialize result container"""
#         self.total_events = 0
#         self.filtered_events = 0
#         self.anomalies_detected = 0
#         self.recoveries_detected = 0
#         self.recommendations: List[Dict[str, Any]] = []
#         self.stats: Dict[str, Any] = {}
    
#     def add_recommendation(self, 
#                           anomaly_type: str,
#                           severity: str,
#                           description: str,
#                           solution: str,
#                           evidence: Optional[Dict] = None):
#         """Add a recommendation to results"""
#         self.recommendations.append({
#             'anomaly_type': anomaly_type,
#             'severity': severity,
#             'description': description,
#             'solution': solution,
#             'evidence': evidence or {}
#         })
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary"""
#         return {
#             'total_events': self.total_events,
#             'filtered_events': self.filtered_events,
#             'anomalies_detected': self.anomalies_detected,
#             'recoveries_detected': self.recoveries_detected,
#             'recommendations': self.recommendations,
#             'stats': self.stats
#         }


# class AgenticLoop:
#     """
#     Core agentic loop orchestrator
    
#     Workflow:
#     1. Load logs from file
#     2. Filter events (reduce noise)
#     3. Detect anomalies (metric-based and recovery-based)
#     4. AI Agent analyzes and diagnoses issues
#     5. Generate intelligent recommendations
#     """
    
#     def __init__(self, 
#                  z_score_threshold: float = 2.0,
#                  recovery_lookback: float = 5.0,
#                  auto_filter: bool = True,
#                  use_ai: bool = True,
#                  api_key: Optional[str] = None):
#         """
#         Initialize agentic loop
        
#         Args:
#             z_score_threshold: Threshold for metric anomaly detection
#             recovery_lookback: Seconds to look back for recovery causes
#             auto_filter: Automatically choose appropriate filter
#             use_ai: Use AI (Gemini) for deep analysis
#             api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
#         """
#         self.parser = LogParser()
#         self.metric_detector = MetricAnomalyDetector(z_score_threshold)
#         self.simple_filter = SimpleLogFilter()
#         self.recovery_detector = RecoveryDetector(recovery_lookback)
#         self.auto_filter = auto_filter
#         self.use_ai = use_ai
        
#         # Initialize AI Agent (required for pure AI diagnosis)
#         if not use_ai:
#             raise RuntimeError(
#                 "AI Agent is required. This is an AI-driven agentic loop.\n"
#                 "Set GEMINI_API_KEY environment variable or use --api-key argument.\n"
#                 "Get API key at: https://aistudio.google.com/"
#             )
        
#         if not api_key:
#             import os
#             api_key = os.environ.get('GEMINI_API_KEY')
#             if not api_key:
#                 raise RuntimeError(
#                     "API key not found. Set GEMINI_API_KEY environment variable or use --api-key argument."
#                 )
        
#         try:
#             self.ai_agent = AIAgent(api_key=api_key)
#         except Exception as e:
#             raise RuntimeError(
#                 f"AI Agent initialization failed: {e}\n"
#                 "Ensure API key is correct and network is available."
#             )
    
#     def run(self, 
#             log_path: str, 
#             limit: Optional[int] = None,
#             include_codecoverage: bool = True) -> AgenticLoopResult:
#         """
#         Run the full agentic loop
        
#         Args:
#             log_path: Path to log file
#             limit: Maximum number of events to process (None = all)
#             include_codecoverage: Whether to include CodeCoverage events
            
#         Returns:
#             AgenticLoopResult with findings and recommendations
#         """
#         result = AgenticLoopResult()
        
#         print(f"Starting Agentic Loop analysis...")
#         print(f"Log file: {log_path}")
#         print(f"Include CodeCoverage: {'Yes' if include_codecoverage else 'No'}")
#         print()
        
#         print(" Step 1/5: Loading log events...")
#         events = self._load_events(log_path, limit)
#         result.total_events = len(events)
#         print(f"    Loaded {result.total_events} events")
#         print()
        
#         if not events:
#             print(" No events found, exiting")
#             return result
        
#         print(" Step 2/5: Smart filtering events...")
#         filtered_events = self._smart_filter(events)
#         result.filtered_events = len(filtered_events)
#         print(f"    Remaining after filtering {result.filtered_events} events "
#               f"(filtered: {(1 - result.filtered_events/result.total_events)*100:.1f}%)")
#         print()
        
#         print(" Step 3/5: AI Agent analyzing...")
#         print("   → Evaluating log data...")
        
#         print("   → Detecting recovery events...")
#         recoveries = self.recovery_detector.detect_recoveries(events, include_codecoverage)
#         result.recoveries_detected = len(recoveries)
#         print(f"   ✓ Found {result.recoveries_detected} recovery events")
        
#         print("   → Analyzing metric anomalies...")
#         metric_anomalies = self.metric_detector.detect_anomalies(filtered_events)
#         result.anomalies_detected = len([a for a in metric_anomalies if a[1]])
#         print(f"   ✓ Identified {result.anomalies_detected} metric anomalies")
#         print()
        
#         print(" Step 4/5: AI Agent generating recommendations...")
#         print("   → Integrating analysis results...")
#         self._agent_generate_recommendations(metric_anomalies, recoveries, events, result)
#         print(f"   ✓ Generated {len(result.recommendations)} recommendations")
#         print()
        
#         print(" Step 5/5: Compiling statistics...")
#         result.stats = self._compile_stats()
#         print(f"   ✓ Statistics compiled")
#         print()
        
#         return result
    

#     def _load_events(self, log_path: str, limit: Optional[int]) -> List[EventModel]:
#         """Load events from log file"""
#         events = []
#         for i, event in enumerate(self.parser.parse_logs(log_path)):
#             if limit and i >= limit:
#                 break
#             events.append(event)
#         return events
    
#     def _smart_filter(self, events: List[EventModel]) -> List[EventModel]:
#         """
#         Intelligently filter events based on log characteristics
        
#         This implements the "automatic filter selection" requirement
#         """
#         if not self.auto_filter:
#             return events
        
#         # Analyze log characteristics
#         total_events = len(events)
#         high_severity_count = sum(1 for e in events if e.severity and e.severity >= 30)
#         recovery_count = sum(1 for e in events if e.event == "MasterRecoveryState")
#         codecoverage_count = sum(1 for e in events if e.event == "CodeCoverage")
        
#         # Decision logic for filter selection
#         if codecoverage_count > total_events * 0.5:
#             # Lots of CodeCoverage events - filter them out
#             print("   → Many CodeCoverage events detected, will be filtered")
#             return [e for e in events if e.event != "CodeCoverage"]
        
#         elif high_severity_count > total_events * 0.1:
#             # Many high severity events - use severity filter
#             print("   → Many high-severity events detected, using severity filter")
#             return self.simple_filter.filter_events(events)
        
#         else:
#             # Default: light filtering
#             print("   → Using lightweight filter")
#             # Filter out very low severity and boring events
#             return [e for e in events 
#                    if e.event not in ['BuggifySection'] and 
#                       (not e.severity or e.severity >= 10)]
    
#     def _agent_generate_recommendations(self,
#                                         metric_anomalies: List[tuple],
#                                         recoveries: List[RecoveryEvent],
#                                         all_events: List[EventModel],
#                                         result: AgenticLoopResult):
#         """AI Agent generates recommendations using pure AI diagnosis"""
#         for recovery in recoveries:
#             print(f"    AI Agent diagnosing recovery (state {recovery.state_code}: {recovery.state_name})...")
            
#             agent_result = self.ai_agent.analyze_recovery_with_tools(recovery, all_events)
            
#             cause = agent_result.get('cause', 'Unknown')
#             confidence = agent_result.get('confidence', 0)
#             recommendations = agent_result.get('recommendations') or ['No recommendations']
#             reasoning = agent_result.get('reasoning', 'None')
#             recommendations_str = '; '.join(recommendations) if recommendations else 'No recommendations'
            
#             if confidence >= 80:
#                 severity = "high"
#             elif confidence >= 50:
#                 severity = "medium"
#             else:
#                 severity = "low"
            
#             result.add_recommendation(
#                 anomaly_type=f"Recovery Event (Pure AI Diagnosis)",
#                 severity=severity,
#                 description=f"Recovery event (state {recovery.state_code}: {recovery.state_name})\n🤖 AI diagnosis: {cause}\n📊 Confidence: {confidence}%",
#                 solution=f" AI recommendations:\n{recommendations_str}\n\n🧠 AI reasoning:\n{reasoning}",
#                 evidence={
#                     'timestamp': str(recovery.timestamp) if recovery.timestamp else None,
#                     'state_code': recovery.state_code,
#                     'state_name': recovery.state_name,
#                     'ai_diagnosis': agent_result,
#                     'diagnosis_method': 'Pure AI - Zero heuristics',
#                     'related_events_analyzed': len(recovery.related_events)
#                 }
#             )
        
#         seen_types = set()
#         for event, reasons in metric_anomalies:
#             if not reasons:
#                 continue
            
#             anomaly_key = f"metric_{event.event}_{'_'.join(reasons)}"
#             if anomaly_key not in seen_types:
#                 print(f"    AI analyzing metric anomaly: {event.event}...")
                
#                 agent_result = self.ai_agent.analyze_anomaly_with_agent(event, reasons, all_events)
#                 recommendations = agent_result.get('recommendations') or []
#                 recommendations_str = '; '.join(recommendations) if recommendations else 'No specific recommendations'
                
#                 result.add_recommendation(
#                     anomaly_type=f"Metric Anomaly (AI Agent)",
#                     severity=agent_result.get('severity', 'medium'),
#                     description=f"Detected metric anomaly: {event.event}. AI Agent root cause: {agent_result.get('root_cause', 'Unknown')}",
#                     solution=f"AI Agent recommendations: {recommendations_str}",
#                     evidence={
#                         'event_type': event.event,
#                         'timestamp': str(event.ts) if event.ts else None,
#                         'reasons': reasons,
#                         'agent_analysis': agent_result
#                     }
#                 )
#                 seen_types.add(anomaly_key)
    
#     def _compile_stats(self) -> Dict[str, Any]:
#         """Compile all statistics"""
#         return {
#             'metric_detector': self.metric_detector.get_stats(),
#             'recovery_detector': self.recovery_detector.get_stats(),
#             'simple_filter': self.simple_filter.get_stats(),
#             'token_savings': self.metric_detector.estimate_token_savings()
#         }
    
#     def run_on_events(self,
#                       events: List[EventModel],
#                       include_codecoverage: bool = True) -> AgenticLoopResult:
#         """Run the loop on an in-memory event list (DB, API, etc.)."""
#         result = AgenticLoopResult()
#         result.total_events = len(events)
#         if not events:
#             return result

#         # Step 2: filter
#         filtered_events = self._smart_filter(events)
#         result.filtered_events = len(filtered_events)

#         # Step 3: detect
#         recoveries = self.recovery_detector.detect_recoveries(events, include_codecoverage)
#         result.recoveries_detected = len(recoveries)

#         metric_anomalies = self.metric_detector.detect_anomalies(filtered_events)
#         result.anomalies_detected = len([a for a in metric_anomalies if a[1]])

#         # Step 4: AI
#         self._agent_generate_recommendations(metric_anomalies, recoveries, events, result)

#         # Step 5: stats
#         result.stats = self._compile_stats()
#         return result

    
#     def print_results(self, result: AgenticLoopResult):
#         """Print results in a readable format"""
#         print("=" * 80)
#         print(" AGENTIC LOOP ANALYSIS RESULTS")
#         print("=" * 80)
#         print()
        
#         # Summary statistics
#         print("Overall Statistics:")
#         print(f"   • Total events: {result.total_events}")
#         print(f"   • Filtered events: {result.filtered_events}")
#         print(f"   • Anomalies detected: {result.anomalies_detected}")
#         print(f"   • Recoveries detected: {result.recoveries_detected}")
#         print(f"   • Recommendations generated: {len(result.recommendations)}")
#         print()
        
#         # Recommendations
#         if result.recommendations:
#             print("💡 Detected Issues and Recommendations:")
#             print()
            
#             for i, rec in enumerate(result.recommendations, 1):
#                 severity_emoji = {
#                     'low': '🟢',
#                     'medium': '🟡',
#                     'high': '🟠',
#                     'critical': '🔴'
#                 }.get(rec['severity'], '⚪')
                
#                 print(f"{i}. {severity_emoji} [{rec['severity'].upper()}] {rec['anomaly_type']}")
#                 print(f"   Description: {rec['description']}")
#                 print(f"   Recommendations: {rec['solution']}")
                
#                 if rec.get('evidence'):
#                     evidence = rec['evidence']
#                     if 'timestamp' in evidence and evidence['timestamp']:
#                         print(f"   Time: {evidence['timestamp']}")
#                     if 'event_type' in evidence:
#                         print(f"   Event type: {evidence['event_type']}")
                
#                 print()
#         else:
#             print("No obvious issues detected! System appears healthy.")
#             print()
        
#         # Token savings
#         if 'token_savings' in result.stats:
#             ts = result.stats['token_savings']
#             print("💰 Token savings (for LLM analysis):")
#             print(f"   • Original tokens: {ts['total_tokens_without_filter']:,}")
#             print(f"   • Filtered tokens: {ts['total_tokens_with_filter']:,}")
#             print(f"   • Tokens saved: {ts['tokens_saved']:,} ({ts['token_reduction_rate']*100:.1f}%)")
#             print()
        
#         print("=" * 80)

