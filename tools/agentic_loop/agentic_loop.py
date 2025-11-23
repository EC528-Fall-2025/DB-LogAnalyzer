"""
Agentic Loop Core - End-to-end automation for anomaly detection and recommendation.

Purpose: Orchestrate load → filter → detect → match → recommend workflow with embeddings
"""
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Any

import duckdb
import google.generativeai as genai

from data_transfer_object.event_dto import EventModel
from tools.anomaly_detector import MetricAnomalyDetector
from tools.parser import LogParser
from tools.recovery_detector import RecoveryDetector, RecoveryEvent


class AgenticResult:
    """Result object from AgenticLoop execution"""
    
    def __init__(self):
        self.total_events = 0
        self.events_processed = 0
        self.chunks_created = 0
        self.embeddings_generated = 0
        self.anomalies = []
        self.recoveries = []
        self.anomaly_stats = {}
        self.recovery_stats = {}
        self.embedding_stats = {}
        self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization"""
        return {
            'total_events': self.total_events,
            'events_processed': self.events_processed,
            'chunks_created': self.chunks_created,
            'embeddings_generated': self.embeddings_generated,
            'anomalies_count': len(self.anomalies),
            'recoveries_count': len(self.recoveries),
            'anomaly_stats': self.anomaly_stats,
            'recovery_stats': self.recovery_stats,
            'embedding_stats': self.embedding_stats,
            'errors': self.errors,
        }


class AgenticLoop:
    """Agentic loop for processing FDB logs with embeddings and anomaly detection."""

    def __init__(
        self,
        z_score_threshold: float = 2.0,
        recovery_lookback: float = 5.0,
        auto_filter: bool = True,
        use_ai: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize AgenticLoop.

        Args:
            z_score_threshold: Z-score threshold for anomaly detection
            recovery_lookback: Seconds to look back for recovery causes
            auto_filter: Whether to automatically filter events
            use_ai: Whether to use AI features (embeddings)
            api_key: Google Gemini API key (or uses GEMINI_API_KEY env var)
        """
        self.z_score_threshold = z_score_threshold
        self.recovery_lookback = recovery_lookback
        self.auto_filter = auto_filter
        self.use_ai = use_ai
        self.api_key = api_key
        
        # Initialize components
        self.anomaly_detector = MetricAnomalyDetector(z_score_threshold=z_score_threshold)
        self.recovery_detector = RecoveryDetector(look_back_seconds=recovery_lookback)
        self.parser = LogParser()
        
        # Initialize Gemini if AI is enabled
        if self.use_ai:
            self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize the Gemini API with the API key."""
        # Use provided API key or environment variable
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Provide api_key parameter or set GEMINI_API_KEY env var."
            )

        # Configure Gemini client
        genai.configure(api_key=api_key)
        print("✓ Gemini initialized successfully!")

    def run(
        self,
        log_path: str,
        limit: Optional[int] = None,
        include_codecoverage: bool = True,
        db_path: Optional[str] = None,
    ) -> AgenticResult:
        """
        Run agentic loop on a log file.
        
        Args:
            log_path: Path to log file
            limit: Maximum number of events to process
            include_codecoverage: Whether to include CodeCoverage events
            db_path: Optional DuckDB path to store embeddings
            
        Returns:
            AgenticResult object with results
        """
        print(f"Processing log file: {log_path}")
        
        result = AgenticResult()
        
        # Parse log file
        events = []
        try:
            for i, event in enumerate(self.parser.parse_logs(log_path)):
                if limit is not None and i >= limit:
                    break
                events.append(event)
            result.total_events = len(events)
            result.events_processed = len(events)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            print(f"Parse error: {e}", file=sys.stderr)
            return result
        
        # Process events (detect anomalies and recoveries)
        return self._process_events(events, include_codecoverage, db_path, result)
    
    def run_on_events(
        self,
        events: List[EventModel],
        include_codecoverage: bool = True,
        db_path: Optional[str] = None,
    ) -> AgenticResult:
        """
        Run agentic loop on pre-loaded events.

        Args:
            events: List of EventModel instances
            include_codecoverage: Whether to include CodeCoverage events
            db_path: Optional DuckDB path to store embeddings

        Returns:
            AgenticResult object with results
        """
        print(f"Processing {len(events)} pre-loaded events")
        
        result = AgenticResult()
        result.total_events = len(events)
        result.events_processed = len(events)
        
        return self._process_events(events, include_codecoverage, db_path, result)
    
    def _process_events(
        self,
        events: List[EventModel],
        include_codecoverage: bool,
        db_path: Optional[str],
        result: AgenticResult,
    ) -> AgenticResult:
        """Internal method to process events and generate embeddings."""
        
        # Detect anomalies
        print(f"🔍 Detecting anomalies (z-score threshold: {self.z_score_threshold})...")
        anomalies = self.anomaly_detector.detect_anomalies(events)
        result.anomalies = [(event, reasons) for event, reasons in anomalies if reasons]
        result.anomaly_stats = self.anomaly_detector.get_stats()
        print(f"   ✓ Found {len(result.anomalies)} anomalous events")
        
        # Detect recoveries
        print(f"🔄 Detecting recoveries (lookback: {self.recovery_lookback}s)...")
        recoveries = self.recovery_detector.detect_recoveries(events, include_codecoverage)
        result.recoveries = recoveries
        result.recovery_stats = self.recovery_detector.get_stats()
        print(f"   ✓ Found {len(recoveries)} recovery events")
        
        # Generate embeddings if AI is enabled
        if self.use_ai and events:
            print(f"🧠 Generating embeddings...")
            
            # Chunk events for embedding
            chunks = self._chunk_events(events)
            result.chunks_created = len(chunks)
            
            # Generate embeddings
            chunks_with_embeddings = self.generate_embeddings(chunks)
            result.embeddings_generated = len(chunks_with_embeddings)
            result.embedding_stats = {
                'chunks_total': len(chunks),
                'embeddings_generated': len(chunks_with_embeddings),
                'success_rate': len(chunks_with_embeddings) / len(chunks) if chunks else 0,
            }
            
            # Store embeddings if db_path provided
            if db_path and chunks_with_embeddings:
                print(f"💾 Storing embeddings in {db_path}...")
                self.store_embeddings(db_path, chunks_with_embeddings)
        
        return result
    
    def _chunk_events(self, events: List[EventModel]) -> List[Dict[str, Any]]:
        """Chunk events into smaller parts for embedding."""
        chunks = []
        chunk_size = 512  # Character limit for Gemini embedding input
        
        for event in events:
            # Create text representation
            event_text = event.event or ""
            fields_text = json.dumps(event.fields_json) if event.fields_json else "{}"
            full_text = f"{event_text}\n{fields_text}"
            
            # If text fits in chunk, use as single chunk
            if len(full_text) <= chunk_size:
                chunks.append({
                        "text": full_text,
                    "event_id": str(event.event_id) if event.event_id else None,
                    "event": event_text,
                    "timestamp": event.ts.isoformat() if event.ts else None,
                })
            else:
                # Split into multiple chunks
                for i in range(0, len(full_text), chunk_size):
                    chunk_text = full_text[i:i + chunk_size]
                    chunks.append({
                            "text": chunk_text,
                        "event_id": str(event.event_id) if event.event_id else None,
                        "event": event_text,
                        "timestamp": event.ts.isoformat() if event.ts else None,
                            "chunk_index": i // chunk_size,
                    })

        return chunks

    def generate_embeddings(self, log_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for each log chunk using Gemini API (with retry + throttle).

        Args:
            log_chunks: List of chunk dictionaries

        Returns:
            List of dictionaries with chunks and their embeddings
        """
        if not self.use_ai:
            print("⚠️  AI disabled, skipping embedding generation.")
            return []

        print(f"   Generating embeddings for {len(log_chunks)} chunks...")

        chunks_with_embeddings = []

        for idx, chunk in enumerate(log_chunks):
            chunk_text = chunk.get("text", "")

            # Gemini API requires non-empty strings
            if not chunk_text or not chunk_text.strip():
                continue

            for attempt in range(3):  # Retry up to 3 times
                try:
                    # Generate embedding using Gemini
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk_text,
                        task_type="retrieval_document",
                    )

                    # Extract embedding from result
                    embedding = result.get("embedding")
                    if embedding:
                        chunks_with_embeddings.append({
                                **chunk,
                                "embedding": embedding,
                        })
                        # Random small sleep to avoid rate limit bursts
                        time.sleep(random.uniform(0.1, 0.3))
                        break  # success → exit retry loop
                    else:
                        raise ValueError("No embedding returned from API")

                except Exception as e:
                    if attempt < 2:
                        print(f"   ⚠️  [Attempt {attempt+1}/3] Embedding failed for chunk {idx}: {e}")
                        time.sleep(1.0 * (attempt + 1))  # exponential backoff
            else:
                        print(f"   ❌ Giving up on chunk {idx} after 3 failed attempts.")
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"   Progress: {idx + 1}/{len(log_chunks)} chunks processed...")

        print(f"   ✓ Successfully embedded {len(chunks_with_embeddings)} / {len(log_chunks)} chunks.")
        return chunks_with_embeddings

    def store_embeddings(self, db_path: str, chunks_with_embeddings: List[Dict[str, Any]]):
        """Store embeddings into DuckDB."""
        if not chunks_with_embeddings:
            print("   ⚠️  No embeddings to store.")
            return

        try:
            connection = duckdb.connect(db_path)

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    event_id VARCHAR,
                    chunk_index INTEGER,
                    event_type VARCHAR,
                    chunk_text TEXT,
                    embedding JSON,
                    timestamp TIMESTAMP
                );
                """
            )

            for chunk_data in chunks_with_embeddings:
                insert_query = """
                    INSERT INTO chunk_embeddings (event_id, chunk_index, event_type, chunk_text, embedding, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?);
                """
                timestamp = None
                if chunk_data.get("timestamp"):
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(chunk_data["timestamp"])
                    except:
                        pass
                
                connection.execute(
                    insert_query,
                    (
                        chunk_data.get("event_id") or "",
                        chunk_data.get("chunk_index", 0),
                        chunk_data.get("event") or "",
                        chunk_data.get("text", ""),
                        json.dumps(chunk_data.get("embedding", [])),
                        timestamp,
                    ),
                )

            connection.commit()
            connection.close()
            print(f"   ✓ Stored {len(chunks_with_embeddings)} embeddings in database.")
        except Exception as e:
            print(f"   ❌ Error storing embeddings: {e}", file=sys.stderr)

    def print_results(self, result: AgenticResult):
        """Print formatted results to console."""
        print("\n" + "=" * 80)
        print("AGENTIC LOOP RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total events: {result.total_events}")
        print(f"   Events processed: {result.events_processed}")
        
        if result.anomaly_stats:
            print(f"\n🔍 Anomaly Detection:")
            print(f"   Anomalies detected: {len(result.anomalies)}")
            print(f"   Anomaly rate: {result.anomaly_stats.get('anomaly_rate', 0):.2%}")
            if result.anomalies:
                print(f"   Top anomaly reasons:")
                reasons_count = {}
                for _, reasons in result.anomalies:
                    for reason in reasons:
                        reasons_count[reason] = reasons_count.get(reason, 0) + 1
                for reason, count in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     - {reason}: {count}")
        
        if result.recovery_stats:
            print(f"\n🔄 Recovery Detection:")
            print(f"   Recoveries found: {len(result.recoveries)}")
            if result.recoveries:
                print(f"   Recoveries with cause: {result.recovery_stats.get('recoveries_with_cause', 0)}")
                print(f"   Cause detection rate: {result.recovery_stats.get('cause_detection_rate', 0):.2%}")
                print(f"   Recovery states:")
                states_count = {}
                for recovery in result.recoveries:
                    state = f"{recovery.state_code}:{recovery.state_name}"
                    states_count[state] = states_count.get(state, 0) + 1
                for state, count in sorted(states_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     - {state}: {count}")
        
        if result.embedding_stats:
            print(f"\n🧠 Embedding Generation:")
            print(f"   Chunks created: {result.chunks_created}")
            print(f"   Embeddings generated: {result.embeddings_generated}")
            print(f"   Success rate: {result.embedding_stats.get('success_rate', 0):.2%}")
        
        if result.errors:
            print(f"\n❌ Errors:")
            for error in result.errors:
                print(f"   - {error}")
        
        print("\n" + "=" * 80)


def main():
    """Main function to test the AgenticLoop class."""
    print("AgenticLoop test - use main.py or cli_wrapper instead")


if __name__ == "__main__":
    main()
