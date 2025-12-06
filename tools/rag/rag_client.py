"""RAG client that talks to Vertex RAG store (regional only, no global)."""

import os
from typing import Optional

from google import genai
from google.genai import types


class RAGClient:
    """Lightweight wrapper around the Vertex RAG Engine using google.genai."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        corpus_resource: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        model: Optional[str] = None,
        use_adc: Optional[bool] = None,
    ):
        # Resolve config
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        self.location = (
            location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("RAG_REGION")
            or os.environ.get("RAG_LOCATION")
        )
        self.corpus_resource = (
            corpus_resource
            or os.environ.get("RAG_CORPUS_RESOURCE")
            or os.environ.get("RAG_CORPUS")
        )
        self.model_name = model or os.environ.get("RAG_MODEL") or "gemini-1.5-flash-001"

        if not self.project:
            raise ValueError("Missing project. Set GOOGLE_CLOUD_PROJECT or pass project.")
        if not self.location:
            raise ValueError("Missing region. Set GOOGLE_CLOUD_LOCATION or pass location (must match corpus region).")
        if not self.corpus_resource:
            raise ValueError("Missing RAG corpus resource. Set RAG_CORPUS_RESOURCE or pass corpus_resource.")

        # Auth: prefer ADC (required for regional Vertex RAG). Do not mix api_key with project/location.
        use_adc = use_adc if use_adc is not None else bool(
            os.environ.get("RAG_USE_ADC") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if not use_adc:
            api_key = api_key or os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GEMINI_API_KEY")

        client_args = {
            "vertexai": True,
            "project": self.project,
            "location": self.location,
        }
        # Passing api_key together with project/location triggers mutual exclusion errors; rely on ADC when possible.
        if api_key and not use_adc:
            client_args["api_key"] = api_key

        self.client = genai.Client(**client_args)

        self.tool = types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus=self.corpus_resource
                        )
                    ]
                )
            )
        )

    def retrieve(self, query_text: str) -> dict:
        """Execute a retrieval-augmented query against the configured corpus."""
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query_text)],
            )
        ]

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
            top_p=1.0,
            tools=[self.tool],
        )

        response = {"text": "", "chunks": []}
        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        ):
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                response["text"] += chunk.text or ""
                try:
                    response["chunks"].append(chunk.to_dict())
                except Exception:
                    response["chunks"].append(str(chunk))

        return response
