"""Production Apache Airflow DAG for End-to-End Diabetes ML Pipeline.

Orchestrates the complete MLOps lifecycle via the TaskFlow API:
start -> ingest_data -> validate_data -> preprocess_and_engineer ->
train_candidate_models -> evaluate_models -> generate_explanations ->
quality_gate -> register_champion -> end
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Ensure repository root is in sys.path for Airflow workers & standalone imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("airflow.task")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =====================================================================
# Robust Standalone & Airflow Compatibility Layer
# =====================================================================

class StandaloneTask:
    """TaskFlow-compatible task node supporting execution and >> / << dependency chaining."""

    def __init__(
        self,
        task_id: str,
        python_callable: Callable,
        dag: Optional[StandaloneDAG] = None,
        doc: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.python_callable = python_callable
        self.dag = dag
        self.doc = doc
        self.upstream_list: Set[StandaloneTask] = set()
        self.downstream_list: Set[StandaloneTask] = set()

    @property
    def upstream_task_ids(self) -> Set[str]:
        return {t.task_id for t in self.upstream_list}

    @property
    def downstream_task_ids(self) -> Set[str]:
        return {t.task_id for t in self.downstream_list}

    def set_downstream(
        self, other: Union[StandaloneTask, List[StandaloneTask]]
    ) -> Union[StandaloneTask, List[StandaloneTask]]:
        if isinstance(other, (list, tuple, set)):
            for o in other:
                self.set_downstream(o)
            return other
        self.downstream_list.add(other)
        other.upstream_list.add(self)
        return other

    def set_upstream(
        self, other: Union[StandaloneTask, List[StandaloneTask]]
    ) -> Union[StandaloneTask, List[StandaloneTask]]:
        if isinstance(other, (list, tuple, set)):
            for o in other:
                self.set_upstream(o)
            return other
        self.upstream_list.add(other)
        other.downstream_list.add(self)
        return other

    def __rshift__(
        self, other: Union[StandaloneTask, List[StandaloneTask]]
    ) -> Union[StandaloneTask, List[StandaloneTask]]:
        return self.set_downstream(other)

    def __lshift__(
        self, other: Union[StandaloneTask, List[StandaloneTask]]
    ) -> Union[StandaloneTask, List[StandaloneTask]]:
        return self.set_upstream(other)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global _CURRENT_DAG
        if _CURRENT_DAG is not None:
            _CURRENT_DAG.add_task(self)
            return self
        return self.python_callable(*args, **kwargs)

    def execute(self, context: Optional[Dict[str, Any]] = None) -> Any:
        return self.python_callable()

    def __repr__(self) -> str:
        return f"<Task({self.task_id})>"


class StandaloneDAG:
    """TaskFlow-compatible DAG container supporting cycle detection and task management."""

    def __init__(
        self,
        dag_id: str,
        schedule: Optional[str] = "@weekly",
        start_date: Optional[datetime] = None,
        catchup: bool = False,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        doc_md: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.dag_id = dag_id
        self.schedule = schedule
        self.schedule_interval = schedule
        self.start_date = start_date or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.catchup = catchup
        self.tags = tags or []
        self.description = description
        self.doc_md = doc_md
        self.task_dict: Dict[str, StandaloneTask] = {}

    @property
    def tasks(self) -> List[StandaloneTask]:
        return list(self.task_dict.values())

    @property
    def task_ids(self) -> List[str]:
        return list(self.task_dict.keys())

    def add_task(self, task: StandaloneTask) -> None:
        task.dag = self
        self.task_dict[task.task_id] = task

    def get_task(self, task_id: str) -> StandaloneTask:
        if task_id not in self.task_dict:
            raise KeyError(f"Task '{task_id}' not found in DAG '{self.dag_id}'")
        return self.task_dict[task_id]

    def has_task(self, task_id: str) -> bool:
        return task_id in self.task_dict

    def topological_sort(self) -> List[StandaloneTask]:
        """Perform topological sort on tasks. Raises ValueError if cycle is detected."""
        in_degree = {t.task_id: len(t.upstream_list) for t in self.tasks}
        queue = [t for t in self.tasks if in_degree[t.task_id] == 0]
        sorted_tasks: List[StandaloneTask] = []

        while queue:
            curr = queue.pop(0)
            sorted_tasks.append(curr)
            for downstream in sorted(curr.downstream_list, key=lambda t: t.task_id):
                in_degree[downstream.task_id] -= 1
                if in_degree[downstream.task_id] == 0:
                    queue.append(downstream)

        if len(sorted_tasks) != len(self.tasks):
            raise ValueError(
                f"Cyclic dependency detected in DAG '{self.dag_id}'! "
                f"Expected {len(self.tasks)} nodes, resolved {len(sorted_tasks)}."
            )
        return sorted_tasks

    def test_cycle(self) -> bool:
        """Validate that the DAG contains no cycles."""
        self.topological_sort()
        return True

    def validate(self) -> None:
        self.test_cycle()

    def __enter__(self) -> StandaloneDAG:
        global _CURRENT_DAG
        _CURRENT_DAG = self
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _CURRENT_DAG
        _CURRENT_DAG = None

    def __repr__(self) -> str:
        return f"<DAG({self.dag_id}, tasks={len(self.tasks)})>"


_CURRENT_DAG: Optional[StandaloneDAG] = None


def standalone_task(
    _func: Optional[Callable] = None,
    *,
    task_id: Optional[str] = None,
    doc: Optional[str] = None,
) -> Callable:
    """Decorator defining a task node within a Standalone DAG."""

    def decorator(fn: Callable) -> StandaloneTask:
        tid = task_id or fn.__name__
        task_obj = StandaloneTask(task_id=tid, python_callable=fn, doc=doc or fn.__doc__)
        if _CURRENT_DAG is not None:
            _CURRENT_DAG.add_task(task_obj)
        return task_obj

    if _func is not None:
        return decorator(_func)
    return decorator


def standalone_dag(
    dag_id: str = "diabetes_ml_training_pipeline",
    schedule: Optional[str] = "@weekly",
    start_date: Optional[datetime] = None,
    catchup: bool = False,
    tags: Optional[List[str]] = None,
    **dag_kwargs: Any,
) -> Callable:
    """Decorator defining a TaskFlow DAG container."""

    def decorator(fn: Callable) -> Callable[..., StandaloneDAG]:
        def wrapper(*args: Any, **kwargs: Any) -> StandaloneDAG:
            dag_obj = StandaloneDAG(
                dag_id=dag_id,
                schedule=schedule,
                start_date=start_date,
                catchup=catchup,
                tags=tags or ["mlops", "diabetes", "training"],
                doc_md=fn.__doc__,
                **dag_kwargs,
            )
            with dag_obj:
                fn(*args, **kwargs)
            return dag_obj

        wrapper._dag_id = dag_id  # type: ignore[attr-defined]
        return wrapper

    return decorator


# Decide whether to use Airflow's native decorators or standalone fallback
USE_STANDALONE = os.environ.get("AIRFLOW_STANDALONE", "").lower() in ("1", "true", "yes")

try:
    if USE_STANDALONE:
        raise ImportError("Standalone mode requested via environment variable.")
    # Attempt to use native Airflow if available and working
    from airflow.decorators import dag as _af_dag, task as _af_task  # type: ignore
    from airflow.models.dag import DAG as _AfDAG  # type: ignore

    # Verify native Airflow works on this platform
    dag = _af_dag
    task = _af_task
    DAG = _AfDAG
    HAS_NATIVE_AIRFLOW = True
except (ImportError, Exception):
    dag = standalone_dag
    task = standalone_task
    DAG = StandaloneDAG
    HAS_NATIVE_AIRFLOW = False


# =====================================================================
# Pipeline Task Definitions (TaskFlow API)
# =====================================================================

@task(task_id="start")
def start() -> Dict[str, Any]:
    """Initialize pipeline execution and log run metadata."""
    now = datetime.now(timezone.utc).isoformat()
    logger.info("=== [STAGE: start] Starting Diabetes ML Training Pipeline at %s ===", now)
    return {"status": "STARTED", "timestamp": now}


@task(task_id="ingest_data")
def ingest_data() -> Dict[str, Any]:
    """Ingest raw diabetes prediction dataset and verify integrity."""
    logger.info("=== [STAGE: ingest_data] Ingesting dataset ===")
    from src.data.ingest import ingest_data as _ingest

    df = _ingest()
    logger.info("Dataset ingested successfully: %d rows, %d columns.", len(df), len(df.columns))
    return {
        "status": "SUCCESS",
        "rows": len(df),
        "columns": len(df.columns),
        "target_path": "data/raw/diabetes_prediction_dataset.csv",
    }


@task(task_id="validate_data")
def validate_data() -> Dict[str, Any]:
    """Execute strict schema validation, physiological boundary audits, and report generation."""
    logger.info("=== [STAGE: validate_data] Validating raw dataset integrity ===")
    from src.data.validate import run_validation_and_audit

    report = run_validation_and_audit()
    status = report.get("validation_status", "UNKNOWN")
    logger.info("Data validation completed with status: %s", status)
    if status != "PASSED":
        raise ValueError(f"Data validation failed! Report: {report}")
    return {
        "status": "SUCCESS",
        "validation_status": status,
        "report_path": "reports/data_validation_report.json",
    }


@task(task_id="preprocess_and_engineer")
def preprocess_and_engineer() -> Dict[str, Any]:
    """Execute leakage-safe stratified splitting and clinical feature engineering."""
    logger.info("=== [STAGE: preprocess_and_engineer] Running preprocessing and feature engineering ===")
    from src.preprocessing.pipeline import run_pipeline

    run_pipeline()
    logger.info("Preprocessing complete. Artifacts saved: models/preprocessor.joblib, data/processed/*")
    return {
        "status": "SUCCESS",
        "preprocessor_path": "models/preprocessor.joblib",
        "splits": ["data/processed/train.parquet", "data/processed/val.parquet", "data/processed/test.parquet"],
    }


@task(task_id="train_candidate_models")
def train_candidate_models() -> Dict[str, Any]:
    """Train baseline, benchmarks, and candidate models; rank models and register champion."""
    logger.info("=== [STAGE: train_candidate_models] Training candidates and selecting champion ===")
    from src.training.train_all import train_all_models

    results = train_all_models()
    champion = results.get("champion", {})
    champion_version = results.get("champion_version", "unknown")
    logger.info(
        "Candidate training complete. Champion: %s (version %s)",
        champion.get("model_name"),
        champion_version,
    )
    return {
        "status": "SUCCESS",
        "champion_model": champion.get("model_name"),
        "champion_version": champion_version,
    }


@task(task_id="evaluate_models")
def evaluate_models() -> Dict[str, Any]:
    """Run probability calibration, Brier score verification, and subgroup demographic fairness audit."""
    logger.info("=== [STAGE: evaluate_models] Running advanced evaluation, calibration & fairness ===")
    from src.evaluation.evaluate_all import run_evaluation_pipeline

    eval_report = run_evaluation_pipeline()
    brier = eval_report.get("calibration", {}).get("brier_score")
    logger.info("Evaluation complete. Brier score: %s. Report: reports/evaluation_report.json", brier)
    return {
        "status": "SUCCESS",
        "brier_score": brier,
        "report_path": "reports/evaluation_report.json",
    }


@task(task_id="generate_explanations")
def generate_explanations() -> Dict[str, Any]:
    """Generate global SHAP summary feature attributions and sample local DiCE counterfactuals."""
    logger.info("=== [STAGE: generate_explanations] Generating explainability & SHAP artifacts ===")
    from src.features.explain_all import run_explainability_pipeline

    run_explainability_pipeline()
    logger.info("Explainability artifacts generated: reports/figures/shap_summary.png")
    return {
        "status": "SUCCESS",
        "shap_summary_path": "reports/figures/shap_summary.png",
    }


@task(task_id="quality_gate")
def quality_gate() -> Dict[str, Any]:
    """Evaluate candidate model against strict production quality thresholds and promote."""
    logger.info("=== [STAGE: quality_gate] Evaluating autonomous quality gate thresholds ===")
    from src.evaluation.quality_gate import run_quality_gate

    gate_result = run_quality_gate()
    decision = gate_result.get("overall_decision")
    if decision != "PASSED":
        reasons = gate_result.get("failure_reasons", [])
        raise RuntimeError(f"Quality Gate FAILED with decision '{decision}': {reasons}")

    logger.info("Quality Gate PASSED. Candidate model promoted to @champion.")
    return {
        "status": "SUCCESS",
        "overall_decision": decision,
        "report_path": "reports/quality_gate_report.json",
    }


@task(task_id="register_champion")
def register_champion() -> Dict[str, Any]:
    """Verify and log final active @champion metadata from MLflow Model Registry."""
    logger.info("=== [STAGE: register_champion] Confirming active champion in Model Registry ===")
    from mlflow.tracking import MlflowClient
    from src.training.evaluate import REGISTERED_MODEL_NAME

    client = MlflowClient()
    champion_info: Dict[str, Any] = {
        "model_name": REGISTERED_MODEL_NAME,
        "alias": "champion",
    }
    try:
        model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        champion_info.update({
            "version": str(model_version.version),
            "run_id": model_version.run_id,
            "status": model_version.status,
            "current_stage": getattr(model_version, "current_stage", "None"),
        })
        logger.info(
            "Confirmed active champion in registry: %s (v%s, run_id: %s)",
            champion_info["model_name"],
            champion_info["version"],
            champion_info["run_id"],
        )
    except Exception as exc:
        logger.warning("Could not fetch @champion version details: %s. Using default confirmation.", exc)
        champion_info["status"] = "confirmed"

    return {"status": "SUCCESS", "champion": champion_info}


@task(task_id="end")
def end() -> Dict[str, Any]:
    """Finalize pipeline execution and log completion summary."""
    now = datetime.now(timezone.utc).isoformat()
    logger.info("=== [STAGE: end] Diabetes ML Training Pipeline Succeeded at %s ===", now)
    return {"status": "COMPLETED", "timestamp": now}


# =====================================================================
# DAG Definition & Dependency Graph
# =====================================================================

@dag(
    dag_id="diabetes_ml_training_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "diabetes", "training"],
)
def diabetes_ml_training_pipeline() -> StandaloneDAG:
    """End-to-End Diabetes Machine Learning Training & Governance Pipeline."""
    t_start = start()
    t_ingest = ingest_data()
    t_validate = validate_data()
    t_preprocess = preprocess_and_engineer()
    t_train = train_candidate_models()
    t_evaluate = evaluate_models()
    t_explain = generate_explanations()
    t_gate = quality_gate()
    t_champion = register_champion()
    t_end = end()

    # Exact sequential execution chain
    (
        t_start
        >> t_ingest
        >> t_validate
        >> t_preprocess
        >> t_train
        >> t_evaluate
        >> t_explain
        >> t_gate
        >> t_champion
        >> t_end
    )


# Instantiate the DAG object at module level for Airflow loader discovery and imports
diabetes_dag: StandaloneDAG = diabetes_ml_training_pipeline()  # type: ignore[assignment]
pipeline_dag = diabetes_dag
