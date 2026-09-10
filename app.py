"""
Offline Workflow Discovery & Process Intelligence Engine
Streamlit UI — Upload | Workflow Browser | Query
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is on sys.path when launched via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force reload local modules so Streamlit updates them when code changes
for mod in ["config", "graph", "embeddings", "pipeline", "retrieval", "similarity", "storage", "workflow", "actions", "chunking", "coref", "parser", "segmentation", "sequence_markers", "models"]:
    to_delete = [k for k in sys.modules if k == mod or k.startswith(mod + ".")]
    for k in to_delete:
        del sys.modules[k]

from config import get_config, setup_logging
from graph import WorkflowGraphBuilder, graph_to_pyvis_html
from embeddings import FaissWorkflowIndex
from pipeline import WorkflowDiscoveryPipeline
from retrieval import QueryEngine
from similarity import Deduplicator, WorkflowSimilarityEngine, format_workflow_pair_comparison
from storage import WorkflowStore
from workflow.sanitize import is_spurious_workflow_node, sanitize_workflow

# Bump when retrieval/action logic changes so Streamlit cache refreshes.
ENGINE_VERSION = "20260719-compare-workflows"

st.set_page_config(
    page_title="Workflow Discovery Engine",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline(_engine_version: str = ENGINE_VERSION) -> WorkflowDiscoveryPipeline:
    config = get_config()
    setup_logging(config)
    return WorkflowDiscoveryPipeline(config)


@st.cache_resource
def get_store(_engine_version: str = ENGINE_VERSION) -> WorkflowStore:
    return WorkflowStore(get_config())


@st.cache_resource
def get_deduplicator(_engine_version: str = ENGINE_VERSION) -> Deduplicator:
    pipeline = get_pipeline()
    config = get_config()
    return Deduplicator(config, pipeline.index.embedder, pipeline.similarity)


def workflow_select_options(workflows: list) -> dict[str, object]:
    return {
        f"{wf.title_hint or 'Workflow'} | {wf.document} | "
        f"p.{wf.pages[0] if wf.pages else '?'} ({wf.workflow_id})": wf
        for wf in workflows
    }


@st.cache_resource
def get_query_engine(_engine_version: str = ENGINE_VERSION) -> QueryEngine:
    pipeline = get_pipeline()
    config = get_config()
    return QueryEngine(
        config,
        store=pipeline.store,
        index=pipeline.index,
        similarity=WorkflowSimilarityEngine(config, pipeline.index.embedder),
    )


def invalidate_engine_cache() -> None:
    get_store.clear()
    get_pipeline.clear()
    get_query_engine.clear()
    get_deduplicator.clear()


def read_disk_corpus_stats() -> dict[str, int]:
    """Always read corpus metrics from disk (not cached in-memory index)."""
    import json

    config = get_config()
    workflows_dir = config.paths.resolve("workflows_dir")
    meta_path = config.paths.resolve("meta_file")
    faiss_meta_path = config.paths.resolve("faiss_meta")

    workflow_count = len(list(workflows_dir.glob("wf_*.json")))
    documents = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        documents = int((meta.get("stats") or {}).get("documents_processed", 0))

    vectors = 0
    if faiss_meta_path.exists():
        faiss_meta = json.loads(faiss_meta_path.read_text(encoding="utf-8"))
        vectors = len(faiss_meta.get("entries") or [])

    return {
        "documents": documents,
        "workflows": workflow_count,
        "vectors": vectors,
    }


def sync_corpus_cache_with_disk() -> None:
    """Drop cached pipeline/index when disk corpus changes (e.g. after Clear corpus)."""
    disk = read_disk_corpus_stats()
    prev = st.session_state.get("disk_corpus_snapshot")
    snapshot = (disk["workflows"], disk["vectors"])
    if prev is not None and prev != snapshot:
        invalidate_engine_cache()
    st.session_state["disk_corpus_snapshot"] = snapshot


def sync_engine_version() -> None:
    """Clear cached pipeline/store when code version changes (Streamlit hot-reload safe)."""
    previous = st.session_state.get("engine_version")
    if previous != ENGINE_VERSION:
        invalidate_engine_cache()
        st.session_state["engine_version"] = ENGINE_VERSION
        st.session_state.pop("disk_corpus_snapshot", None)


def corpus_has_stale_nodes(store: WorkflowStore) -> bool:
    return any(
        is_spurious_workflow_node(node)
        for wf in store.load_all()
        for node in wf.nodes
    )


def clear_corpus_files() -> int:
    """Remove all workflow JSON files and reset corpus metadata on disk."""
    store = WorkflowStore(get_config())
    removed = store.clear_all()
    config = get_config()
    FaissWorkflowIndex(config).clear()
    store.update_stats(
        documents_processed=0,
        workflows_found=0,
        actions_extracted=0,
        indexed_vectors=0,
    )
    return removed


def render_sidebar() -> None:
    st.sidebar.title("Corpus")
    disk = read_disk_corpus_stats()

    st.sidebar.metric("Documents processed", disk["documents"])
    st.sidebar.metric("Workflows found", disk["workflows"])
    st.sidebar.metric("Indexed vectors", disk["vectors"])

    if disk["workflows"] == 0:
        st.sidebar.info("Corpus is empty. Upload documents in the **Upload** tab.")

    if st.sidebar.button("Sanitize & re-index", use_container_width=True):
        if disk["workflows"] == 0:
            st.sidebar.warning("No workflows on disk to repair.")
        else:
            with st.sidebar.status("Cleaning workflows and rebuilding index..."):
                pipeline = get_pipeline()
                sanitized, indexed = pipeline.repair_corpus()
                invalidate_engine_cache()
                st.session_state.pop("disk_corpus_snapshot", None)
            st.sidebar.success(
                f"Updated {sanitized} workflow(s), indexed {indexed} vectors."
            )
            st.rerun()

    if st.sidebar.button("Re-index all workflows", use_container_width=True):
        if disk["workflows"] == 0:
            st.sidebar.warning("No workflows on disk to index.")
        else:
            with st.sidebar.status("Rebuilding FAISS index..."):
                pipeline = get_pipeline()
                count = pipeline.reindex_all()
                invalidate_engine_cache()
                st.session_state.pop("disk_corpus_snapshot", None)
            st.sidebar.success(f"Indexed {count} vectors")
            st.rerun()

    if st.sidebar.button("Clear corpus", use_container_width=True):
        with st.sidebar.status("Clearing stored workflows and index..."):
            invalidate_engine_cache()
            removed = clear_corpus_files()
            st.session_state.pop("disk_corpus_snapshot", None)
        st.sidebar.warning(
            f"Removed {removed} workflow(s). Re-upload documents to rebuild the corpus."
        )
        st.rerun()

    store = get_store()
    if disk["workflows"] > 0 and corpus_has_stale_nodes(store):
        st.sidebar.error(
            "Corpus contains outdated step artifacts (e.g. `perform 6`). "
            "Click **Sanitize & re-index** or clear and re-upload."
        )

    st.sidebar.caption(f"Engine {ENGINE_VERSION} · offline · CPU-only · no LLMs")


def tab_upload(pipeline: WorkflowDiscoveryPipeline) -> None:
    st.subheader("Upload & discover workflows")
    st.write(
        "Upload PDF, DOCX, or TXT. The engine parses, chunks, classifies sentences, "
        "extracts actions & discourse markers, segments workflows, builds graphs, "
        "and indexes them locally. After code updates, use **Clear corpus** then re-upload."
    )
    uploaded = st.file_uploader(
        "Document",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if not uploaded:
        return

    if st.button("Run discovery pipeline", type="primary"):
        progress = st.progress(0.0)
        results = []
        for i, file in enumerate(uploaded):
            suffix = Path(file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getvalue())
                tmp_path = Path(tmp.name)
            try:
                with st.status(f"Processing {file.name}...", expanded=True) as status:
                    st.write("Parsing → chunking → classify → actions → markers → segmentation → graph → index")
                    result = pipeline.process_file(tmp_path, reindex=True)
                    # Restore original filename in saved workflows
                    for wf in result.workflows:
                        wf.document = file.name
                        pipeline.store.save(wf)
                    status.update(label=f"Done: {file.name}", state="complete")
                results.append((file.name, result))
            finally:
                tmp_path.unlink(missing_ok=True)
            progress.progress((i + 1) / len(uploaded))

        invalidate_engine_cache()
        st.session_state.pop("disk_corpus_snapshot", None)

        for name, result in results:
            st.success(
                f"**{name}**: {len(result.chunks)} chunks · "
                f"{result.action_count} actions · "
                f"{len(result.workflows)} workflows"
                + (f" · {result.merged_count} merged" if result.merged_count else "")
            )
            for wf in result.workflows:
                with st.expander(f"{wf.title_hint or wf.workflow_id} — {wf.summary[:120]}"):
                    st.write(wf.summary)
                    st.caption(
                        f"Pages: {wf.pages} · Nodes: {len(wf.nodes)} · Edges: {len(wf.edges)}"
                    )


def tab_browser(store: WorkflowStore) -> None:
    st.subheader("Discovered workflows")
    workflows = [sanitize_workflow(wf) for wf in store.load_all()]
    if not workflows:
        st.info("No workflows yet. Upload a document in the Upload tab.")
        return

    options = {
        f"{wf.title_hint or 'Workflow'} | {wf.document} | "
        f"p.{wf.pages[0] if wf.pages else '?'} ({wf.workflow_id})": wf
        for wf in workflows
    }
    selected_key = st.selectbox("Select workflow", list(options.keys()))
    workflow = options[selected_key]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Summary**")
        st.write(workflow.summary)
        st.markdown("**Pages**")
        pages = workflow.pages
        st.write(
            f"{pages[0]}–{pages[-1]}" if len(pages) > 1 else (pages[0] if pages else "n/a")
        )
        st.markdown("**Sections**")
        st.write(", ".join(workflow.sections) or "—")
        st.markdown("**Nodes**")
        for node in workflow.ordered_nodes:
            actor = f" [{node.actor}]" if node.actor else ""
            st.write(f"{node.order + 1}. {node.label()}{actor} — p.{node.page}")

    with col2:
        st.markdown("**Interactive graph**")
        builder = WorkflowGraphBuilder(get_config())
        graph = builder.to_networkx(workflow.nodes, workflow.edges)
        html = graph_to_pyvis_html(graph, height="520px")
        components.html(html, height=540, scrolling=True)

    st.markdown("**Evidence sources**")
    for ev in workflow.evidence_sources[:20]:
        loc = f"p.{ev.page}" if ev.page else ""
        sec = f" · {ev.section}" if ev.section else ""
        st.markdown(f"- {loc}{sec}: _{ev.sentence}_")


def tab_query() -> None:
    st.subheader("Ask a workflow question")
    st.write(
        "Questions are matched via FAISS retrieval plus query-to-workflow similarity "
        "(semantic summary match + action set overlap) with cited evidence."
    )
    question = st.text_input(
        "Question",
        placeholder="Where is the Major Weld Repair procedure?",
    )
    top_k = st.slider("Top-K candidates", 1, 10, 5)

    if not question or not st.button("Search", type="primary"):
        return

    engine = get_query_engine()
    with st.spinner("Searching workflows..."):
        answer = engine.answer(question, top_k=top_k)

    if not answer.matched_workflow_id:
        st.warning(answer.explanation)
        return

    st.markdown("### Answer")
    st.code(answer.explanation, language=None)

    if answer.scores:
        if answer.scores.comparison_type.value == "query_to_workflow":
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{answer.confidence:.0%}")
            c2.metric("Semantic", f"{(answer.scores.semantic_similarity or 0.0):.2f}")
            c3.metric("Set overlap", f"{answer.scores.set_overlap:.2f}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence", f"{answer.confidence:.0%}")
            c2.metric("Set overlap", f"{answer.scores.set_overlap:.2f}")
            c3.metric("Sequence", f"{(answer.scores.sequence_alignment or 0.0):.2f}")
            c4.metric("Structural", f"{(answer.scores.structural or 0.0):.2f}")

    st.markdown("### Ranked candidates")
    for i, cand in enumerate(answer.candidates, start=1):
        wf = cand.workflow
        with st.expander(
            f"#{i} {wf.title_hint or wf.workflow_id} — confidence {cand.scores.blended:.0%}"
        ):
            st.write(wf.summary)
            if cand.scores.comparison_type.value == "query_to_workflow":
                st.caption(
                    f"semantic={(cand.scores.semantic_similarity or 0.0):.2f} · "
                    f"set={cand.scores.set_overlap:.2f} · "
                    f"faiss={cand.faiss_score}"
                )
            else:
                st.caption(
                    f"set={cand.scores.set_overlap:.2f} · "
                    f"seq={(cand.scores.sequence_alignment or 0.0):.2f} · "
                    f"struct={(cand.scores.structural or 0.0):.2f} · "
                    f"faiss={cand.faiss_score}"
                )
            for ev in cand.evidence[:5]:
                st.markdown(f"- p.{ev.page}: _{ev.sentence}_")


def tab_compare(store: WorkflowStore) -> None:
    st.subheader("Compare workflows")
    st.write(
        "Run **WORKFLOW_TO_WORKFLOW** scoring (set overlap + sequence alignment + "
        "structural similarity) — the same graph comparison Module 12 logs during dedup."
    )
    workflows = [sanitize_workflow(wf) for wf in store.load_all()]
    if len(workflows) < 2:
        st.info("Need at least two workflows in the corpus to compare.")
        return

    options = workflow_select_options(workflows)
    keys = list(options.keys())
    col1, col2 = st.columns(2)
    with col1:
        left_key = st.selectbox("Workflow A", keys, index=0, key="compare_left")
    with col2:
        default_right = min(1, len(keys) - 1)
        right_key = st.selectbox("Workflow B", keys, index=default_right, key="compare_right")

    left = options[left_key]
    right = options[right_key]
    if left.workflow_id == right.workflow_id:
        st.warning("Select two different workflows.")
        return

    if st.button("Compare", type="primary"):
        deduper = get_deduplicator()
        comparison = deduper.compare_pair(left, right)
        scores = comparison.graph_scores

        st.markdown("### Graph similarity (WORKFLOW_TO_WORKFLOW)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", f"{scores.blended:.0%}")
        c2.metric("Set overlap", f"{scores.set_overlap:.2f}")
        c3.metric("Sequence", f"{(scores.sequence_alignment or 0.0):.2f}")
        c4.metric("Structural", f"{(scores.structural or 0.0):.2f}")
        st.caption(format_workflow_pair_comparison(comparison))

        st.markdown("### Dedup check (Module 12)")
        st.write(
            f"Summary cosine: **{comparison.summary_cosine:.2f}** "
            f"(threshold **{comparison.dedup_threshold:.2f}**)"
        )
        if comparison.is_duplicate:
            st.error("Would be flagged as a **duplicate** and merged during indexing.")
        else:
            st.success("Would **not** be merged — treated as distinct procedures.")

        with st.expander("Workflow A chain"):
            st.write(" → ".join(n.label() for n in left.ordered_nodes))
        with st.expander("Workflow B chain"):
            st.write(" → ".join(n.label() for n in right.ordered_nodes))


def main() -> None:
    sync_engine_version()
    sync_corpus_cache_with_disk()
    st.title("Workflow Discovery & Process Intelligence")
    st.caption("Offline-first · graph-based · explainable · no LLMs")
    render_sidebar()
    pipeline = get_pipeline()
    store = get_store()

    upload_tab, browse_tab, query_tab, compare_tab = st.tabs(
        ["Upload", "Workflow Browser", "Query", "Compare Workflows"]
    )
    with upload_tab:
        tab_upload(pipeline)
    with browse_tab:
        tab_browser(store)
    with query_tab:
        tab_query()
    with compare_tab:
        tab_compare(store)


if __name__ == "__main__":
    main()
