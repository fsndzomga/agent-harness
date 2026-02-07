"""GAIA Agent v1 — improved smolagents-based GAIA agent.

Built on top of HAL Generalist with:
- Smart search with query reformulation and retry
- Wikipedia lookup for factual queries
- Persistent file-based memory (save_note / read_notes)
- Hybrid context compaction (sliding window + observation truncation)
- Enhanced file reading (audio, images/OCR, docx, json, html)
- 60 max steps with compaction to stay within context limits
"""
