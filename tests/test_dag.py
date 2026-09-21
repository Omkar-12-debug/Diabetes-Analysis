"""Unit and integration test suite for Apache Airflow DAG and Pipeline Runner.

Tests verify:
- DAG loading, attributes, and tags
- Expected task presence and completeness
- Exact sequential task dependency structure
- Graph acyclicity and cycle detection resilience
- Standalone pipeline runner deterministic execution
"""

import pytest
from dags.diabetes_ml_pipeline import (
    StandaloneDAG,
    StandaloneTask,
    diabetes_dag,
    diabetes_ml_training_pipeline,
)
from dags.pipeline_runner import run_pipeline


EXPECTED_CORE_TASKS = [
    "ingest_data",
    "validate_data",
    "preprocess_and_engineer",
    "train_candidate_models",
    "evaluate_models",
    "generate_explanations",
    "quality_gate",
    "register_champion",
]

FULL_TASK_CHAIN = [
    "start",
    "ingest_data",
    "validate_data",
    "preprocess_and_engineer",
    "train_candidate_models",
    "evaluate_models",
    "generate_explanations",
    "quality_gate",
    "register_champion",
    "end",
]


def test_dag_loading_and_attributes() -> None:
    """Verify DAG object loads cleanly with expected metadata and configurations."""
    assert diabetes_dag is not None
    assert diabetes_dag.dag_id == "diabetes_ml_training_pipeline"
    assert diabetes_dag.schedule in ("@weekly", None)
    assert diabetes_dag.catchup is False
    assert set(diabetes_dag.tags).issuperset({"mlops", "diabetes", "training"})


def test_dag_expected_tasks_present() -> None:
    """Verify DAG contains all core tasks required by the specification."""
    task_ids = set(diabetes_dag.task_ids)
    for expected_task in EXPECTED_CORE_TASKS:
        assert expected_task in task_ids, f"Required task '{expected_task}' missing from DAG."

    # Verify boundary tasks
    assert "start" in task_ids
    assert "end" in task_ids
    assert len(diabetes_dag.tasks) == len(FULL_TASK_CHAIN)


def test_dag_dependency_chain_and_order() -> None:
    """Verify the exact linear task dependency chain: start -> ... -> end."""
    for idx in range(len(FULL_TASK_CHAIN) - 1):
        curr_id = FULL_TASK_CHAIN[idx]
        next_id = FULL_TASK_CHAIN[idx + 1]

        curr_task = diabetes_dag.get_task(curr_id)
        next_task = diabetes_dag.get_task(next_id)

        assert next_id in curr_task.downstream_task_ids, (
            f"Expected '{next_id}' to be downstream of '{curr_id}', but found {curr_task.downstream_task_ids}"
        )
        assert curr_id in next_task.upstream_task_ids, (
            f"Expected '{curr_id}' to be upstream of '{next_id}', but found {next_task.upstream_task_ids}"
        )


def test_dag_acyclicity_and_topological_sort() -> None:
    """Verify DAG contains no circular dependencies and sorts in valid order."""
    # Validate the production DAG is strictly acyclic
    assert diabetes_dag.test_cycle() is True
    sorted_tasks = diabetes_dag.topological_sort()
    assert len(sorted_tasks) == len(FULL_TASK_CHAIN)
    assert [t.task_id for t in sorted_tasks] == FULL_TASK_CHAIN


def test_cycle_detection_resilience() -> None:
    """Verify cycle detector raises ValueError when a circular dependency is introduced."""
    test_dag = StandaloneDAG(dag_id="cyclic_test_dag")
    t1 = StandaloneTask(task_id="task_a", python_callable=lambda: None)
    t2 = StandaloneTask(task_id="task_b", python_callable=lambda: None)
    test_dag.add_task(t1)
    test_dag.add_task(t2)

    # Introduce a cyclic dependency: t1 >> t2 >> t1
    t1 >> t2 >> t1

    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        test_dag.topological_sort()


def test_all_tasks_have_callables() -> None:
    """Verify each task node has an associated executable Python callable."""
    for task in diabetes_dag.tasks:
        assert callable(task.python_callable), f"Task '{task.task_id}' has non-callable function."


def test_pipeline_runner_execution() -> None:
    """Verify standalone pipeline runner executes the complete workflow with SUCCESS status."""
    summary = run_pipeline()
    assert summary["overall_status"] == "SUCCESS"
    assert summary["executed_tasks_count"] == len(FULL_TASK_CHAIN)
    assert summary["total_duration_seconds"] > 0

    for task_res in summary["task_results"]:
        assert task_res["status"] == "SUCCESS", f"Task '{task_res['task_id']}' failed: {task_res['error']}"
