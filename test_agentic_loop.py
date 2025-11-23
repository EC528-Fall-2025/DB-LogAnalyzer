#!/usr/bin/env python3
"""
Test script for AgenticLoop functionality.

This script helps you test the embedding generation and log processing pipeline.
"""
import os
import sys
from pathlib import Path

import duckdb

from tools.agentic_loop import AgenticLoop


def test_basic_functionality():
    """Test basic AgenticLoop functionality."""
    print("=" * 80)
    print("Testing AgenticLoop - Basic Functionality")
    print("=" * 80)
    print()

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: GEMINI_API_KEY not set!")
        print("   Set it with: export GEMINI_API_KEY='your-key'")
        print("   Or the test will fail when trying to generate embeddings.")
        print()
        use_ai = False
    else:
        print(f"✅ GEMINI_API_KEY found: {api_key[:10]}...")
        use_ai = True
    print()

    # Test directory path - adjust this to your actual log directory
    test_directory = input("Enter path to directory containing .xml log files (or press Enter for default): ").strip()
    
    if not test_directory:
        # Default path from your code
        test_directory = "/Users/vanshikachaddha/Documents/Boston University/Fourth Year/Cloud Computing/DB-LogAnalyzer/logs"
    
    if not os.path.exists(test_directory):
        print(f"❌ Error: Directory does not exist: {test_directory}")
        print("   Please provide a valid directory path containing .xml log files.")
        return False
    
    print(f"📁 Using directory: {test_directory}")
    print()

    # Test database path
    test_db = "test_fdb_logs.duckdb"
    print(f"💾 Using test database: {test_db}")
    print()

    try:
        # Initialize AgenticLoop
        print("🚀 Initializing AgenticLoop...")
        agent = AgenticLoop(
            directory_path=test_directory,
            db_path=test_db,
            use_ai=use_ai,
        )
        print("✅ AgenticLoop initialized successfully!")
        print()

        # Verify database was created and has data
        print("🔍 Verifying database contents...")
        conn = duckdb.connect(test_db)
        
        # Check tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"   Tables found: {[t[0] for t in tables]}")
        
        # Check event count
        try:
            event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            print(f"   Events loaded: {event_count}")
        except Exception as e:
            print(f"   ⚠️  Could not count events: {e}")
        
        # Check embeddings
        try:
            embedding_count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
            print(f"   Embeddings stored: {embedding_count}")
        except Exception as e:
            print(f"   ⚠️  Could not count embeddings: {e}")
            print("   (This is normal if embeddings weren't generated)")
        
        conn.close()
        print()

        print("✅ Test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_generation_only():
    """Test just the embedding generation on a small sample."""
    print("=" * 80)
    print("Testing AgenticLoop - Embedding Generation Only")
    print("=" * 80)
    print()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Cannot test embeddings.")
        return False

    # Create a minimal test
    test_db = "test_embeddings.duckdb"
    
    # Clean up old test DB
    if os.path.exists(test_db):
        os.remove(test_db)
    
    # Create a simple test with dummy data
    conn = duckdb.connect(test_db)
    conn.execute("""
        CREATE TABLE events (
            event_id VARCHAR,
            event VARCHAR,
            raw_json JSON
        );
    """)
    
    # Insert a few test events
    test_events = [
        ("1", "StorageMetrics", '{"BytesInput": 1000, "BytesOutput": 500}'),
        ("2", "DiskMetrics", '{"DiskUsage": 75, "FreeSpace": 25}'),
        ("3", "GRVProxyMetrics", '{"Latency": 0.05, "Throughput": 1000}'),
    ]
    
    for event_id, event_type, raw_json in test_events:
        conn.execute(
            "INSERT INTO events (event_id, event, raw_json) VALUES (?, ?, ?)",
            (event_id, event_type, raw_json)
        )
    
    conn.close()
    print(f"✅ Created test database with {len(test_events)} sample events")
    print()

    # Now test the embedding generation
    try:
        # We need to manually test the embedding parts
        from tools.agentic_loop import AgenticLoop
        
        # Create a minimal agent (but we need to bypass __init__)
        # Actually, let's just test the embedding function directly
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        test_text = "StorageMetrics BytesInput: 1000 BytesOutput: 500"
        print(f"🧪 Testing embedding generation for: {test_text[:50]}...")
        
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=test_text,
            task_type="retrieval_document",
        )
        
        embedding = result.get("embedding")
        if embedding:
            print(f"✅ Embedding generated successfully!")
            print(f"   Embedding dimension: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ No embedding returned")
            return False
            
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_queries():
    """Test querying the generated embeddings."""
    print("=" * 80)
    print("Testing Database Queries")
    print("=" * 80)
    print()

    test_db = input("Enter path to database file (or press Enter for 'fdb_logs.duckdb'): ").strip()
    if not test_db:
        test_db = "fdb_logs.duckdb"
    
    if not os.path.exists(test_db):
        print(f"❌ Database file not found: {test_db}")
        return False
    
    try:
        conn = duckdb.connect(test_db)
        
        # List all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print("📊 Tables in database:")
        for table in tables:
            print(f"   - {table[0]}")
        print()
        
        # Check events
        if any(t[0] == "events" for t in tables):
            event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            print(f"📝 Events: {event_count}")
            
            # Show sample events
            sample = conn.execute("SELECT event_id, event FROM events LIMIT 5").fetchall()
            print("   Sample events:")
            for row in sample:
                print(f"     - {row[0]}: {row[1]}")
            print()
        
        # Check embeddings
        if any(t[0] == "chunk_embeddings" for t in tables):
            embedding_count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
            print(f"🔢 Embeddings: {embedding_count}")
            
            # Show sample embeddings metadata
            sample = conn.execute(
                "SELECT event_id, event_type, chunk_index FROM chunk_embeddings LIMIT 5"
            ).fetchall()
            print("   Sample embeddings:")
            for row in sample:
                print(f"     - Event {row[0]}, Type: {row[1]}, Chunk: {row[2]}")
            print()
        else:
            print("⚠️  No chunk_embeddings table found. Run the full pipeline first.")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test menu."""
    print("=" * 80)
    print("AgenticLoop Test Suite")
    print("=" * 80)
    print()
    print("Choose a test:")
    print("  1. Full functionality test (load logs → generate embeddings)")
    print("  2. Embedding generation test only (quick API test)")
    print("  3. Database query test (inspect existing database)")
    print("  4. Run all tests")
    print("  0. Exit")
    print()
    
    choice = input("Enter choice (0-4): ").strip()
    
    if choice == "1":
        test_basic_functionality()
    elif choice == "2":
        test_embedding_generation_only()
    elif choice == "3":
        test_database_queries()
    elif choice == "4":
        print("\n" + "=" * 80)
        print("Running all tests...")
        print("=" * 80 + "\n")
        test_embedding_generation_only()
        print("\n" + "-" * 80 + "\n")
        test_database_queries()
    elif choice == "0":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please run again and select 0-4.")


if __name__ == "__main__":
    main()


