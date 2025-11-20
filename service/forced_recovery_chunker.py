"""
Forced recovery chunking service.

Chunks FoundationDB logs starting at CodeCoverage events whose comments map to
the "Forced Recovery" phase and ending at the first MasterRecoveryState event
with StatusCode 14 (fully recovered).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

from dto.event import EventModel
from service.parser import LogParser


@dataclass
class ForcedRecoveryChunk:
    """Chunk of events bounded by a forced recovery trigger and recovery completion."""

    start_event: EventModel
    events: List[EventModel]
    end_event: Optional[EventModel] = None

    @property
    def start_comment(self) -> str:
        """Return CodeCoverage comment for the chunk start."""
        if self.start_event.fields_json:
            return str(self.start_event.fields_json.get("Comment", ""))
        return ""

    @property
    def is_complete(self) -> bool:
        """Return whether the chunk reached StatusCode 14."""
        return self.end_event is not None

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary for CLI display."""
        return {
            "start_event_id": self.start_event.event_id,
            "start_time": self.start_event.ts.isoformat() if self.start_event.ts else None,
            "start_comment": self.start_comment,
            "end_event_id": self.end_event.event_id if self.end_event else None,
            "end_time": self.end_event.ts.isoformat() if self.end_event and self.end_event.ts else None,
            "event_count": len(self.events),
            "complete": self.is_complete,
        }

    def to_dict(self, include_events: bool = False) -> Dict[str, Any]:
        """Serialize chunk to a dictionary."""
        data = self.summary()
        if include_events:
            data["events"] = [ForcedRecoveryChunk._serialize_event(event) for event in self.events]
        return data

    @staticmethod
    def _serialize_event(event: EventModel) -> Dict[str, Any]:
        """Convert EventModel to JSON-serializable dict."""
        payload = {
            "event_id": event.event_id,
            "ts": event.ts.isoformat() if event.ts else None,
            "severity": event.severity,
            "event": event.event,
            "process": event.process,
            "role": event.role,
            "pid": event.pid,
            "machine_id": event.machine_id,
            "address": event.address,
            "trace_file": event.trace_file,
            "src_line": event.src_line,
        }
        payload["raw_json"] = event.raw_json
        payload["fields_json"] = event.fields_json
        return payload

    def to_xml(self) -> str:
        """Convert chunk events to XML format matching original FDB trace format."""
        root = ET.Element("Trace")
        
        for event in self.events:
            # Use raw_json which contains all original fields
            event_elem = ET.SubElement(root, "Event")
            
            # Copy all attributes from raw_json, preserving original field names
            for key, value in event.raw_json.items():
                # Convert value to string, handling None
                event_elem.set(key, str(value) if value is not None else "")
        
        # Generate XML string with declaration
        xml_str = '<?xml version="1.0"?>\n'
        xml_str += ET.tostring(root, encoding='unicode')
        return xml_str


class ForcedRecoveryChunker:
    """Chunks logs based on forced recovery CodeCoverage triggers."""

    def __init__(
        self,
        knowledge_base_path: Optional[Path] = None,
        parser: Optional[LogParser] = None,
    ):
        """
        Initialize chunker.

        Args:
            knowledge_base_path: Path to knowledge base JSONL containing recovery phases.
            parser: Optional LogParser instance (useful for dependency injection or testing).
        """
        default_kb_path = Path(__file__).resolve().parents[1] / "samples" / "fdb_recovery_knowledgebase.jsonl"
        self.knowledge_base_path = knowledge_base_path or default_kb_path
        self._trigger_comments = self._load_trigger_comments(self.knowledge_base_path)
        self.parser = parser or LogParser()

    def _load_trigger_comments(self, kb_path: Path) -> List[str]:
        """Load Forced Recovery comment examples from knowledge base."""
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

        trigger_comments: List[str] = []
        with kb_path.open("r", encoding="utf-8") as handle:
            buffer = ""
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    trigger_comments.extend(self._extract_comments_from_entry(buffer))
                    buffer = ""
                    continue
                buffer += line

            # Process trailing buffer
            trigger_comments.extend(self._extract_comments_from_entry(buffer))
        # Deduplicate while preserving order
        seen = set()
        unique_comments = []
        for comment in trigger_comments:
            if comment not in seen:
                seen.add(comment)
                unique_comments.append(comment)
        return unique_comments
    
    def _extract_comments_from_entry(self, entry_text: str) -> List[str]:
        """Extract Forced Recovery comments from a JSON or JSONL entry."""
        if not entry_text:
            return []
        try:
            entry = json.loads(entry_text)
        except json.JSONDecodeError:
            return []
        if not isinstance(entry, dict):
            return []
        if entry.get("recovery_phase") != "Forced Recovery":
            return []
        comments: List[str] = []
        for example in entry.get("examples", []) or []:
            comments.append(str(example).lower())
        representative = entry.get("representative_comment")
        if representative:
            comments.append(str(representative).lower())
        return comments

    def chunk_log_file(
        self,
        log_path: str,
        include_events: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Parse log file and return forced recovery chunks.

        Args:
            log_path: Path to log file (XML, JSON, plaintext).
            include_events: Whether to include serialized events inside each chunk.

        Returns:
            List of chunk dictionaries.
        """
        chunks = self.chunk_events(self.parser.parse_logs(log_path))
        return [chunk.to_dict(include_events=include_events) for chunk in chunks]

    def chunk_events(self, events: Iterable[EventModel]) -> List[ForcedRecoveryChunk]:
        """
        Build forced recovery chunks from event iterable.

        Args:
            events: Iterable of EventModel instances.

        Returns:
            List of ForcedRecoveryChunk objects.
        """
        chunks: List[ForcedRecoveryChunk] = []
        current_start: Optional[EventModel] = None
        current_events: List[EventModel] = []

        for event in events:
            if self._is_trigger_event(event):
                if current_start is not None:
                    # Finish previous chunk as incomplete before starting new one.
                    chunks.append(ForcedRecoveryChunk(start_event=current_start, events=list(current_events)))
                current_start = event
                current_events = [event]
                continue

            if current_start is None:
                continue

            current_events.append(event)
            if self._is_terminal_event(event):
                chunks.append(
                    ForcedRecoveryChunk(
                        start_event=current_start,
                        events=list(current_events),
                        end_event=event,
                    )
                )
                current_start = None
                current_events = []

        if current_start is not None:
            chunks.append(ForcedRecoveryChunk(start_event=current_start, events=list(current_events)))

        return chunks

    def _is_trigger_event(self, event: EventModel) -> bool:
        """Return True if event is a forced recovery CodeCoverage trigger with Covered=1."""
        if event.event != "CodeCoverage":
            return False
        
        # Check if Covered=1 (only events that actually happened)
        if event.fields_json:
            covered = event.fields_json.get("Covered")
            # Handle both string and int representations
            if covered is not None:
                covered_str = str(covered).strip().strip('"').strip('""')
                if covered_str != "1":
                    return False
            else:
                # If Covered field is missing, skip this event
                return False
        else:
            return False
        
        # Check if comment matches forced recovery patterns
        comment = ""
        if event.fields_json:
            comment = str(event.fields_json.get("Comment", "")).lower()
        if not comment:
            return False
        return any(example in comment for example in self._trigger_comments)

    @staticmethod
    def _is_terminal_event(event: EventModel) -> bool:
        """Return True if event is MasterRecoveryState with StatusCode 14."""
        if event.event != "MasterRecoveryState":
            return False
        status_code = None
        if event.fields_json:
            status_code = event.fields_json.get("StatusCode")
        if status_code is None:
            return False
        status_str = str(status_code).strip()
        # Remove common quote characters (straight and curved)
        status_str = status_str.strip('"').strip('""')
        return status_str == "14"

    def export_chunks_to_xml(
        self,
        log_path: str,
        output_dir: str,
        prefix: str = "chunk",
        create_index: bool = True,
    ) -> List[str]:
        """
        Parse log file, chunk it, and export each chunk to a separate XML file.

        Args:
            log_path: Path to input log file (XML, JSON, plaintext).
            output_dir: Directory where chunk XML files will be written.
            prefix: Filename prefix for chunk files (default: "chunk").
            create_index: Whether to create an index file with chunk metadata (default: True).

        Returns:
            List of paths to created XML files.

        Example:
            chunker.export_chunks_to_xml(
                "samples/sample1.xml",
                "output/chunks",
                prefix="recovery_chunk"
            )
            # Creates: recovery_chunk_0.xml, recovery_chunk_1.xml, ...
            # Also creates: chunks_index.json with metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        chunks = self.chunk_events(self.parser.parse_logs(log_path))
        created_files = []
        chunk_metadata = []

        for idx, chunk in enumerate(chunks):
            filename = f"{prefix}_{idx}.xml"
            filepath = output_path / filename
            
            # Write chunk to XML file
            with filepath.open("w", encoding="utf-8") as f:
                f.write(chunk.to_xml())
            
            created_files.append(str(filepath))
            
            # Collect metadata
            metadata = {
                "chunk_index": idx,
                "chunk_label": f"{prefix}_{idx}",
                "xml_file": filename,
                "source_log": Path(log_path).name,
                "start_event_id": chunk.start_event.event_id,
                "start_time": chunk.start_event.ts.isoformat() if chunk.start_event.ts else None,
                "start_comment": chunk.start_comment,
                "end_event_id": chunk.end_event.event_id if chunk.end_event else None,
                "end_time": chunk.end_event.ts.isoformat() if chunk.end_event and chunk.end_event.ts else None,
                "event_count": len(chunk.events),
                "is_complete": chunk.is_complete,
                "file_size_bytes": filepath.stat().st_size,
            }
            chunk_metadata.append(metadata)

        # Create index file if requested
        if create_index and chunk_metadata:
            index_file = output_path / "chunks_index.json"
            with index_file.open("w", encoding="utf-8") as f:
                json.dump({
                    "source_log": Path(log_path).name,
                    "total_chunks": len(chunks),
                    "output_directory": str(output_path),
                    "chunks": chunk_metadata
                }, f, indent=2)
            print(f"   Index file: {index_file}")

        return created_files


