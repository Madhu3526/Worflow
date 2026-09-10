# Offline Workflow Discovery & Process Intelligence Engine

Offline-first system that discovers workflows inside enterprise documents (PDF, DOCX, TXT), indexes them as graphs, and answers workflow-oriented questions with **explainable** confidence — no LLMs, no cloud APIs, no database.

## Features

- Semantic parsing of PDF / DOCX / TXT with page & section metadata
- Semantic chunking (headings, numbered steps, bullets, tables — never character splits)
- CPU coreference resolution (`fastcoref` with heuristic fallback)
- spaCy dependency-based action extraction
- Discourse sequence-marker edges (`after`, `before`, `then`, `if`, …)
- Multi-signal workflow segmentation (structural + lexical + continuity)
- NetworkX workflow graphs + interactive pyvis visualization
- Template-based summaries (deterministic, auditable)
- Sentence-Transformers (`all-MiniLM-L6-v2`) + local FAISS index
- Similarity: set overlap · Needleman–Wunsch · graph edit distance
- Deduplication / evidence merge
- Streamlit UI: Upload · Workflow Browser · Query

## Requirements

- Python **3.10+**
- CPU only (works offline after the one-time model download)

## Setup

```bash
cd Worflow
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first run of embeddings downloads `all-MiniLM-L6-v2` from Hugging Face (cached locally). After that, all inference is offline.

## Run the Streamlit app

```bash
streamlit run app.py
```

1. **Upload** — drop PDF/DOCX/TXT files and run the full discovery pipeline  
2. **Workflow Browser** — inspect summaries, evidence, and interactive graphs  
3. **Query** — ask e.g. *Where is the Major Weld Repair procedure?* and see confidence breakdowns  

Sidebar: corpus stats and **Re-index**.

## Run tests

```bash
pytest -q
```

Integration tests exercise the full pipeline on `tests/fixtures/weld_repair.txt`.

## Project layout

```
parser/              Document parsing (PDF/DOCX/TXT)
chunking/            Semantic boundary chunking
coref/               Coreference resolution
actions/             spaCy action extraction
sequence_markers/    Discourse transition edges
segmentation/        Workflow boundary detection
graph/               NetworkX + pyvis
workflow/            Template summaries
embeddings/          MiniLM + FAISS
similarity/          Overlap / NW / GED + dedup
retrieval/           Evidence + explainable answers
storage/             Local JSON + meta index
pipeline/            End-to-end orchestration
config/settings.yaml Config (paths, weights, patterns)
app.py               Streamlit entrypoint
```

## Configuration

Edit `config/settings.yaml` to tune:

- Lexical segmentation patterns per document type
- Similarity weights (`set_overlap` / `sequence_alignment` / `structural`)
- Dedup cosine threshold
- FAISS / storage paths

## Example answer shape

```
Question: Where is the Major Weld Repair procedure?
Matched Workflow: inspect crack → remove damaged material → grind surface → …
Matched Pages: 1–1
Evidence: "Remove damaged material." / "Preheat the component." / …
Confidence: 87% (set overlap: 0.80, sequence alignment: 0.75, structural: 0.90)
```

## Persistence (no database)

| Artifact | Path (default) |
|----------|----------------|
| Workflow JSON | `data/workflows/wf_*.json` |
| Meta index | `data/workflows_meta.json` |
| FAISS index | `data/indexes/workflows.index` |
| FAISS metadata | `data/indexes/faiss_meta.json` |
| Logs | `logs/engine.log` |

## Notes

- **Coreference:** `fastcoref` is preferred when compatible with your `transformers` version. If the model fails to load, the engine automatically falls back to a deterministic heuristic resolver so the pipeline stays fully offline-capable.
- **First-run downloads:** spaCy `en_core_web_sm` and Sentence-Transformers `all-MiniLM-L6-v2` are downloaded once and cached; subsequent runs need no network.
- **Windows console:** if printing summaries with `→` raises an encoding error, set `PYTHONIOENCODING=utf-8`.
