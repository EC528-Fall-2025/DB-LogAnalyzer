from typing import List
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

def make_log_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

def explode_examples_per_doc(doc: Document) -> List[Document]:
    """
    One chunk per example, carrying label + clean summary (not entire original page_content).
    """
    examples = doc.metadata.get("examples") or []
    label = doc.metadata.get("label", "")
    summary = doc.metadata.get("summary", "")  # ✅ now present

    if not examples:
        # No examples: just return the original doc; splitter will handle size if needed
        return [doc]

    out: List[Document] = []
    for ex in examples:
        text = (
            f"Label: {label}\n"
            f"Summary: {summary}\n\n"
            f"Example:\n- {ex}"
        )
        out.append(
            Document(
                page_content=text,
                metadata={**doc.metadata}  # keep cluster/type/source_file etc.
            )
        )
    return out

def split_docs(documents: List[Document]) -> List[Document]:
    # Expand into per-example docs
    expanded: List[Document] = []
    for d in documents:
        expanded.extend(explode_examples_per_doc(d))

    # Light character split (usually no-op for short examples, but safe)
    splitter = make_log_splitter()
    chunks = splitter.split_documents(expanded)

    for idx, ch in enumerate(chunks):
        ch.metadata["chunk_index"] = idx
    return chunks
