import duckdb
import json
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def make_log_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns an instance of RecursiveCharacterTextSplitter with the desired chunk size and overlap.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjusted size to handle large logs
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

def chunk_logs_by_master_recovery(events: List[dict]) -> List[Document]:
    """
    Chunk logs into separate documents based on Type="MasterRecoveryState" events.
    """
    chunks = []
    current_chunk = []

    for event in events:
        current_chunk.append(event)

        if event["event"] == "MasterRecoveryState":  # Check if event is of type MasterRecoveryState
            doc = Document(
                page_content="\n".join([str(e) for e in current_chunk]),  # Join the chunk into a single string
                metadata={"chunk_type": "pre-recovery", "start_event_id": current_chunk[0]["event_id"], "end_event_id": event["event_id"]}
            )
            chunks.append(doc)
            current_chunk = []  # Start new chunk

    # If there are remaining events after the last recovery event, include them in a chunk
    if current_chunk:
        doc = Document(
            page_content="\n".join([str(e) for e in current_chunk]),
            metadata={"chunk_type": "final_chunk", "start_event_id": current_chunk[0]["event_id"], "end_event_id": current_chunk[-1]["event_id"]}
        )
        chunks.append(doc)

    return chunks

def split_docs(documents: List[Document]) -> List[Document]:
    """
    Split the documents into smaller chunks based on character length using the log splitter.
    """
    splitter = make_log_splitter()  # Create a new instance of RecursiveCharacterTextSplitter
    chunks = splitter.split_documents(documents)

    # Add chunk indices to metadata for tracking
    for idx, ch in enumerate(chunks):
        ch.metadata["chunk_index"] = idx
    return chunks

def store_chunks_in_duckdb(db_path: str, chunks: List[Document]) -> None:
    """
    Store the chunked logs in DuckDB for easy retrieval and embedding.
    """
    connection = duckdb.connect(db_path)
    
    # Create the chunks table if it doesn't exist
    connection.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id UUID PRIMARY KEY,
        start_event_id VARCHAR,
        end_event_id VARCHAR,
        chunk_content TEXT,
        chunk_metadata JSON
    );
    """)

    # Insert the chunks into DuckDB
    for chunk in chunks:
        start_event_id = chunk.metadata.get("start_event_id", None)
        end_event_id = chunk.metadata.get("end_event_id", None)
        chunk_content = chunk.page_content
        chunk_metadata = json.dumps(chunk.metadata)

        connection.execute("""
        INSERT INTO chunks (chunk_id, start_event_id, end_event_id, chunk_content, chunk_metadata)
        VALUES (?, ?, ?, ?, ?)
        """, (str(uuid4()), start_event_id, end_event_id, chunk_content, chunk_metadata))

    connection.close()

def process_logs(db_path: str, logs: List[dict]):
    """
    Full log processing pipeline:
    1. Chunk logs based on Type="MasterRecoveryState" events
    2. Further split the chunks into smaller units for analysis
    3. Store chunks in DuckDB
    """
    print("Starting the log processing pipeline...")

    # Step 2: Chunk the logs based on MasterRecoveryState events
    chunks = chunk_logs_by_master_recovery(logs)

    # Step 3: Further split the chunks into smaller units for embedding
    final_chunks = split_docs(chunks)

    # Step 4: Store the chunked logs in DuckDB
    store_chunks_in_duckdb(db_path, final_chunks)

    print(f"Successfully processed and stored {len(final_chunks)} chunks.")
