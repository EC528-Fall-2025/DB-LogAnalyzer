"""
DuckDB + Anomaly Detection + AI Workflow

Hybrid approach: Use DuckDB to find anomaly points, 
then extract and compress logs for AI analysis.
"""
import sys
from pathlib import Path
from service.parser import LogParser
from service.storage import StorageService
from service.anomaly_detector import MetricAnomalyDetector
from datetime import datetime, timedelta
import json


class DuckDBAIWorkflow:
    """
    Workflow that combines DuckDB aggregation with anomaly detection
    to prepare filtered logs for AI analysis.
    """
    
    def __init__(self, db_path: str = "workflow_demo.duckdb"):
        self.db_path = db_path
        self.storage = StorageService(db_path)
        self.detector = MetricAnomalyDetector(z_score_threshold=1.5)
    
    def step1_load_logs_to_duckdb(self, log_file: str) -> int:
        """Load logs into DuckDB"""
        print("=" * 80)
        print("STEP 1: Load Logs into DuckDB")
        print("=" * 80)
        
        schema_path = "data/schema.sql"
        if Path(schema_path).exists():
            self.storage.init_db(schema_path)
            print(f"✓ Initialized database with schema from {schema_path}")
        else:
            self.storage.init_db()
            print(f"✓ Initialized database with default schema")
        
        count = self.storage.load_logs_from_file(log_file)
        print(f"✓ Loaded {count} events into DuckDB")
        
        return count
    
    def step2_analyze_with_duckdb(self) -> dict:
        """Use DuckDB to aggregate metrics and find anomalies"""
        print("\n" + "=" * 80)
        print("STEP 2: DuckDB Aggregation - Find Anomaly Points")
        print("=" * 80)
        
        latency_query = """
        SELECT 
            event_id,
            ts,
            event,
            machine_id,
            CAST(json_extract_string(fields_json, '$.Max') AS DOUBLE) as max_latency,
            CAST(json_extract_string(fields_json, '$.P99') AS DOUBLE) as p99_latency,
            CAST(json_extract_string(fields_json, '$.Mean') AS DOUBLE) as mean_latency
        FROM events
        WHERE event LIKE '%LatencyMetrics'
            AND json_extract_string(fields_json, '$.Max') IS NOT NULL
        ORDER BY ts
        """
        
        result = self.storage.query(latency_query)
        df = result.df()
        
        print(f"✓ Found {len(df)} latency metric events")
        
        if len(df) > 0:
            print(f"\nLatency Statistics:")
            print(f"  Max latency range: {df['max_latency'].min():.3f}s - {df['max_latency'].max():.3f}s")
            print(f"  Mean latency avg: {df['mean_latency'].mean():.3f}s")
            
            high_latency = df[df['max_latency'] > 1.0]
            print(f"\n🔍 Found {len(high_latency)} events with Max latency > 1.0s")
            
            if not high_latency.empty:
                print(f"\nAnomaly Points (from DuckDB aggregation):")
                for idx, row in high_latency.head(3).iterrows():
                    print(f"  - Event {row['event_id']}: {row['event']} at {row['ts']}")
                    print(f"    Max: {row['max_latency']:.3f}s, P99: {row['p99_latency']:.3f}s")
        
        return {
            'latency_df': df,
            'total_events': len(df),
            'anomaly_points': high_latency if len(df) > 0 else None
        }
    
    def step3_extract_context_logs(self, anomaly_event_ids: list) -> list:
        """Extract surrounding logs for each anomaly point"""
        print("\n" + "=" * 80)
        print("STEP 3: Extract Context Logs Around Anomalies")
        print("=" * 80)
        
        if not anomaly_event_ids:
            print("No anomaly event IDs provided")
            return []
        
        context_window = 5
        
        all_context_logs = []
        
        for event_id in anomaly_event_ids[:3]:
            event_id = int(event_id)
            query = f"""
            SELECT 
                event_id,
                ts,
                severity,
                event,
                machine_id,
                raw_json,
                fields_json
            FROM events
            WHERE CAST(event_id AS INTEGER) BETWEEN {event_id - context_window} AND {event_id + context_window}
            ORDER BY CAST(event_id AS INTEGER)
            """
            
            result = self.storage.query(query)
            context_df = result.df()
            
            print(f"\n✓ Event {event_id}: Extracted {len(context_df)} events as context")
            all_context_logs.extend(context_df.to_dict('records'))
        
        print(f"\n✓ Total context logs extracted: {len(all_context_logs)}")
        
        return all_context_logs
    
    def step4_apply_anomaly_detector(self, log_file: str) -> list:
        """Apply statistical anomaly detection for additional filtering"""
        print("\n" + "=" * 80)
        print("STEP 4: Apply Statistical Anomaly Detection")
        print("=" * 80)
        
        parser = LogParser()
        events = list(parser.parse_logs(log_file))
        anomalies = self.detector.detect_anomalies(events)
        
        print(f"✓ Detected {len(anomalies)} anomalies using Z-score method")
        
        stats = self.detector.get_stats()
        print(f"\nDetection breakdown:")
        print(f"  Z-score anomalies: {stats['by_method']['z_score']}")
        print(f"  Threshold violations: {stats['by_method']['threshold']}")
        
        return anomalies
    
    def step5_prepare_for_ai(self, anomalies: list, max_events: int = 20) -> dict:
        """Prepare filtered logs for AI analysis"""
        print("\n" + "=" * 80)
        print("STEP 5: Prepare Data for AI Analysis")
        print("=" * 80)
        
        limited_anomalies = anomalies[:max_events]
        ai_input = {
            'summary': {
                'total_anomalies': len(anomalies),
                'included_in_analysis': len(limited_anomalies),
                'token_estimate': len(limited_anomalies) * 200
            },
            'anomalies': []
        }
        
        for event, reasons in limited_anomalies:
            anomaly_data = {
                'event_id': event.event_id,
                'timestamp': str(event.ts) if event.ts else None,
                'event_type': event.event,
                'machine_id': event.machine_id,
                'severity': event.severity,
                'detection_reasons': reasons,
                'key_metrics': {}
            }
            
            if event.fields_json:
                for key in ['Max', 'P99', 'Mean', 'Count', 'QueryQueue']:
                    if key in event.fields_json:
                        anomaly_data['key_metrics'][key] = event.fields_json[key]
            
            ai_input['anomalies'].append(anomaly_data)
        
        print(f"✓ Prepared {len(limited_anomalies)} anomalies for AI")
        print(f"✓ Estimated tokens: {ai_input['summary']['token_estimate']:,}")
        
        return ai_input
    
    def step6_generate_ai_prompt(self, ai_input: dict) -> str:
        """Generate formatted AI prompt"""
        print("\n" + "=" * 80)
        print("STEP 6: Generate AI Prompt")
        print("=" * 80)
        
        prompt = f"""You are analyzing FoundationDB logs to identify and explain anomalies.

Summary:
- Total anomalies detected: {ai_input['summary']['total_anomalies']}
- Events in this analysis: {ai_input['summary']['included_in_analysis']}

Anomalous Events:
"""
        
        for i, anomaly in enumerate(ai_input['anomalies'][:5], 1):  # Show first 5 in prompt
            prompt += f"""
{i}. Event #{anomaly['event_id']}: {anomaly['event_type']}
   Time: {anomaly['timestamp']}
   Machine: {anomaly['machine_id']}
   Detection: {', '.join(anomaly['detection_reasons'][:3])}
   Key Metrics: {anomaly['key_metrics']}
"""
        
        prompt += """

Please analyze these anomalies and:
1. Identify common patterns
2. Explain potential root causes
3. Suggest what might be wrong with the system
4. Recommend next steps for investigation
"""
        
        print("✓ Generated AI prompt")
        print(f"\nPrompt preview (first 500 chars):")
        print("-" * 80)
        print(prompt[:500] + "...")
        print("-" * 80)
        
        return prompt
    
    def run_complete_workflow(self, log_file: str, output_file: str = None):
        """Run complete workflow from logs to AI-ready output"""
        print("\n" + "=" * 80)
        print("COMPLETE WORKFLOW: DuckDB → Anomaly Detection → AI")
        print("=" * 80)
        print()
        
        try:
            event_count = self.step1_load_logs_to_duckdb(log_file)
            duckdb_results = self.step2_analyze_with_duckdb()
            
            if duckdb_results.get('anomaly_points') is not None and not duckdb_results['anomaly_points'].empty:
                anomaly_ids = duckdb_results['anomaly_points']['event_id'].tolist()
                context_logs = self.step3_extract_context_logs(anomaly_ids)
            
            anomalies = self.step4_apply_anomaly_detector(log_file)
            ai_input = self.step5_prepare_for_ai(anomalies, max_events=20)
            prompt = self.step6_generate_ai_prompt(ai_input)
            
            if output_file:
                output_data = {
                    'ai_input': ai_input,
                    'prompt': prompt
                }
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\n✓ Saved output to {output_file}")
            
            print("\n" + "=" * 80)
            print("WORKFLOW COMPLETE")
            print("=" * 80)
            print(f"\n✅ Results:")
            print(f"  - Total events: {event_count}")
            print(f"  - Anomalies detected: {len(anomalies)}")
            print(f"  - Ready for AI: {ai_input['summary']['included_in_analysis']} events")
            print(f"  - Token estimate: {ai_input['summary']['token_estimate']:,}")
            
            print(f"\n💡 Next step:")
            print(f"  Copy the prompt to Claude/GPT for analysis")
            
            return ai_input, prompt
            
        finally:
            self.storage.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DuckDB + AI Workflow Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Run complete workflow
  python example_duckdb_ai_workflow.py data/sample_log.json
  
  # Save output for AI
  python example_duckdb_ai_workflow.py data/sample_log.json --output ai_input.json
        """
    )
    
    parser.add_argument('log_file', help='Path to log file')
    parser.add_argument('--output', help='Output file for AI input (JSON)')
    parser.add_argument('--db', default='workflow_demo.duckdb', help='DuckDB database path')
    
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    workflow = DuckDBAIWorkflow(db_path=args.db)
    workflow.run_complete_workflow(args.log_file, args.output)


if __name__ == '__main__':
    main()

