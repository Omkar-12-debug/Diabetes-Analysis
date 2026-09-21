"""Sequential Pipeline Runner for Diabetes MLOps Training DAG.

Executes all DAG stages in deterministic sequential order without requiring
a background Linux daemon or Celery/Postgres Airflow infrastructure.
Usable directly on Windows and during automated CI/CD runs:

    uv run python -m dags.pipeline_runner
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dags.diabetes_ml_pipeline import diabetes_dag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("pipeline_runner")


def run_pipeline() -> Dict[str, Any]:
    """Execute all tasks defined in the diabetes_ml_training_pipeline DAG sequentially.

    Returns:
        Dict[str, Any]: Execution summary containing task timings, statuses, and outputs.

    Raises:
        RuntimeError: If any pipeline stage fails.
    """
    logger.info("=" * 75)
    logger.info("DIABETES ML TRAINING PIPELINE — SEQUENTIAL RUNNER")
    logger.info("DAG ID: %s", diabetes_dag.dag_id)
    logger.info("Start Time: %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 75)

    # Resolve tasks in topological dependency order
    try:
        tasks = diabetes_dag.topological_sort()
    except Exception as exc:
        logger.error("Dependency resolution failed: %s", exc)
        raise RuntimeError(f"DAG dependency resolution error: {exc}") from exc

    task_results: List[Dict[str, Any]] = []
    pipeline_start_time = time.perf_counter()
    overall_status = "SUCCESS"

    for idx, task in enumerate(tasks, start=1):
        task_id = task.task_id
        logger.info("-" * 75)
        logger.info("[%d/%d] EXECUTING TASK: %s", idx, len(tasks), task_id)
        logger.info("-" * 75)

        t0 = time.perf_counter()
        try:
            output = task.execute()
            duration = time.perf_counter() - t0
            logger.info(">>> TASK COMPLETED: %s (Duration: %.2fs)", task_id, duration)
            task_results.append({
                "task_id": task_id,
                "status": "SUCCESS",
                "duration_seconds": round(duration, 3),
                "output": output,
                "error": None,
            })
        except Exception as exc:
            duration = time.perf_counter() - t0
            logger.exception(">>> TASK FAILED: %s (Duration: %.2fs) — Error: %s", task_id, duration, exc)
            task_results.append({
                "task_id": task_id,
                "status": "FAILED",
                "duration_seconds": round(duration, 3),
                "output": None,
                "error": str(exc),
            })
            overall_status = "FAILED"
            break

    total_pipeline_time = time.perf_counter() - pipeline_start_time

    # Print summary performance table
    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"{'Task Name':<32} | {'Duration (s)':<14} | {'Status':<10}")
    print("-" * 75)
    for res in task_results:
        print(f"{res['task_id']:<32} | {res['duration_seconds']:<14.2f} | {res['status']:<10}")
    print("-" * 75)
    print(f"Total Pipeline Runtime: {total_pipeline_time:.2f} seconds")
    print(f"Overall Status:        {overall_status}")
    print("=" * 75 + "\n")

    summary = {
        "dag_id": diabetes_dag.dag_id,
        "overall_status": overall_status,
        "total_duration_seconds": round(total_pipeline_time, 3),
        "executed_tasks_count": len(task_results),
        "total_tasks_count": len(tasks),
        "task_results": task_results,
    }

    if overall_status != "SUCCESS":
        failed_tasks = [r["task_id"] for r in task_results if r["status"] == "FAILED"]
        raise RuntimeError(f"Pipeline failed at stage(s): {', '.join(failed_tasks)}")

    return summary


def main() -> None:
    """CLI entrypoint for running the pipeline via `python -m dags.pipeline_runner`."""
    try:
        run_pipeline()
        sys.exit(0)
    except Exception as exc:
        logger.error("Pipeline runner encountered a fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
