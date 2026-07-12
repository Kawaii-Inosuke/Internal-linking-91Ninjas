"""Internal Linking Assistant — recommends internal links for a new blog post.

See PRD.md / TRD.md. Package modules:
  - config      : env-based configuration
  - chunking    : split blog content into embeddable chunks (TRD §5)
  - embeddings  : Gemini embedding client (TRD §2/§4)
  - db          : Postgres + pgvector access layer (TRD §3)
  - ingest      : ingestion pipeline (TRD §4)
"""

__version__ = "0.1.0"
