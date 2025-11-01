"""
Gradio user interface for run management.

The UI allows users to browse past runs recorded in the database, inspect
parameters and results, and resume interrupted runs.  It communicates
directly with the pipeline functions to execute resumes and reflects
status updates in real time.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Any, List

import gradio as gr

from .config import RunConfig
from .db import init_db, list_runs, get_run
from .pipeline import run_pipeline


def launch_ui(db_path: Path, embeddings_dir: Path, host: str = "127.0.0.1", port: int = 7860) -> None:
    """Launch the Gradio run manager UI.

    Parameters
    ----------
    db_path: Path
        Path to the SQLite database file.
    embeddings_dir: Path
        Directory where embeddings are stored (passed through to resume).
    host: str
        Host address for the server.
    port: int
        Port for the server.
    """
    engine = init_db(db_path)

    # Parameter description dictionary for UI display
    param_desc = {
        "min_face_size": "Discard faces smaller than this many pixels.",
        "sim_threshold": "Cosine similarity threshold for kNN graph construction.",
        "merge_threshold": "Similarity threshold for merging small clusters.",
        "min_cluster_size": "Minimum number of faces to consider a valid identity.",
        "knn_k": "Number of neighbours queried for kNN graph.",
        "start_month": "Month that starts the academic year (1–12).",
        "hdbscan": "Use HDBSCAN instead of connected components.",
        "hdbscan_min_cluster_size": "HDBSCAN minimum cluster size.",
        "hdbscan_min_samples": "HDBSCAN minimum samples.",
        "thumbs_size": "Maximum side length for thumbnails.",
        "thumbs_workers": "Number of workers for thumbnail generation.",
        "thumb_regen": "Regenerate thumbnails even if cached.",
        "use_phash": "Compute perceptual hashes to skip duplicates.",
    }

    def refresh_run_list(status: str) -> List[str]:
        # status can be 'all', 'running', 'interrupted', 'done', 'no_faces'
        with engine.begin() as conn:
            runs = list_runs(conn, None if status == 'all' else status)
            return [f"{r['id']}: {r['status']}" for r in runs]

    def load_run_details(run_label: str) -> Dict[str, Any]:
        # run_label is "id: status"
        try:
            run_id = int(run_label.split(":")[0])
        except Exception:
            return {}
        with engine.begin() as conn:
            r = get_run(conn, run_id)
        if not r:
            return {}
        params = r.get('parameters') or {}
        # Flatten parameter description
        param_table = [[key, json.dumps(value), param_desc.get(key, '')] for key, value in params.items()]
        details = {
            'run_id': run_id,
            'status': r['status'],
            'input_dir': r['input_dir'],
            'output_root': r['output_root'],
            'embeddings_dir': r['embeddings_dir'],
            'model_name': r['model_name'],
            'start_time': str(r['start_time']),
            'end_time': str(r['end_time']),
            'n_images': r.get('n_images'),
            'n_faces': r.get('n_faces'),
            'album_path': r.get('album_path'),
            'command_line': r.get('command_line'),
            'params_table': param_table,
        }
        return details

    def resume_run_action(run_label: str) -> str:
        try:
            run_id = int(run_label.split(":")[0])
        except Exception:
            return "Invalid run"
        # Start a thread to run pipeline resume to avoid blocking UI
        def _resume():
            cfg = RunConfig(
                input_dir=Path('.'),
                output_root=Path('.'),
                db_path=db_path,
                embeddings_dir=embeddings_dir,
                command_line=f"resume {run_id}",
                extra={'resume_run': run_id}
            )
            run_pipeline(cfg)
        thread = threading.Thread(target=_resume, daemon=True)
        thread.start()
        return f"Resume triggered for run {run_id}. Check logs for progress."

    with gr.Blocks() as demo:
        gr.Markdown("# Classface Run Manager")
        status_filter = gr.Dropdown(["all", "running", "interrupted", "done", "no_faces"], value="all", label="Filter by status")
        run_list = gr.Dropdown(choices=refresh_run_list("all"), label="Runs", interactive=True)
        refresh_btn = gr.Button("Refresh list")
        details = gr.JSON(label="Run details")
        resume_btn = gr.Button("▶️ Resume selected run")
        resume_msg = gr.Textbox(label="Resume status", interactive=False)

        status_filter.change(lambda s: refresh_run_list(s), inputs=status_filter, outputs=run_list)
        refresh_btn.click(lambda s: refresh_run_list(status_filter.value), inputs=run_list, outputs=run_list)
        run_list.change(load_run_details, inputs=run_list, outputs=details)
        resume_btn.click(resume_run_action, inputs=run_list, outputs=resume_msg)
    demo.launch(server_name=host, server_port=port, share=False)