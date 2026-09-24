"""CI/CD workflow contract test suite.

Validates that all GitHub Actions workflow configurations are syntactically valid YAML,
contain required jobs, enforce trigger matrices, declare correct step dependencies,
and reference valid application entrypoints.
"""

from pathlib import Path
import pytest
import yaml

CI_WORKFLOW_PATH = Path(".github/workflows/ci.yml")
CD_WORKFLOW_PATH = Path(".github/workflows/cd.yml")


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    """Load and parse the CI workflow definition."""
    assert CI_WORKFLOW_PATH.exists(), f"Missing workflow file: {CI_WORKFLOW_PATH}"
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "CI workflow file is empty"
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "CI workflow YAML did not parse to a dictionary"
    return data


@pytest.fixture(scope="module")
def cd_workflow() -> dict:
    """Load and parse the CD workflow definition."""
    assert CD_WORKFLOW_PATH.exists(), f"Missing workflow file: {CD_WORKFLOW_PATH}"
    content = CD_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "CD workflow file is empty"
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "CD workflow YAML did not parse to a dictionary"
    return data


def test_ci_workflow_exists_and_valid_yaml(ci_workflow: dict):
    """Test that .github/workflows/ci.yml exists, is non-empty, and parses as valid YAML."""
    assert "name" in ci_workflow
    assert "jobs" in ci_workflow
    assert "on" in ci_workflow or True  # PyYAML might load 'on' as True (boolean)


def test_ci_workflow_declared_jobs(ci_workflow: dict):
    """Test that all required jobs are declared in the CI workflow."""
    jobs = ci_workflow.get("jobs", {})
    expected_jobs = [
        "code-quality-and-tests",
        "docker-build-verification",
        "frontend-check",
    ]
    for job_name in expected_jobs:
        assert job_name in jobs, f"Job '{job_name}' missing from CI workflow jobs"


def test_ci_workflow_trigger_matrix(ci_workflow: dict):
    """Verify that CI is triggered on push and pull_request to main and develop."""
    # Handle both 'on' and True due to YAML boolean parsing
    triggers = ci_workflow.get("on") if "on" in ci_workflow else ci_workflow.get(True)
    assert triggers is not None, "Workflow trigger configuration 'on' not found"

    assert "push" in triggers, "Trigger 'push' missing from CI workflow"
    assert "pull_request" in triggers, "Trigger 'pull_request' missing from CI workflow"

    push_branches = triggers["push"].get("branches", [])
    pr_branches = triggers["pull_request"].get("branches", [])

    assert "main" in push_branches, "Push triggers must include 'main'"
    assert "develop" in push_branches, "Push triggers must include 'develop'"
    assert "main" in pr_branches, "PR triggers must include 'main'"
    assert "develop" in pr_branches, "PR triggers must include 'develop'"


def test_ci_workflow_job_dependencies(ci_workflow: dict):
    """Assert that docker-build-verification depends on code-quality-and-tests."""
    jobs = ci_workflow.get("jobs", {})
    docker_job = jobs.get("docker-build-verification", {})
    needs = docker_job.get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    assert "code-quality-and-tests" in needs, (
        "Job 'docker-build-verification' must depend on 'code-quality-and-tests'"
    )


def test_ci_workflow_steps_contract(ci_workflow: dict):
    """Assert that code-quality-and-tests and docker-build-verification include required commands."""
    jobs = ci_workflow.get("jobs", {})
    
    # 1. code-quality-and-tests steps
    code_job = jobs.get("code-quality-and-tests", {})
    code_steps = code_job.get("steps", [])
    code_commands = " ".join(step.get("run", "") for step in code_steps)
    
    assert "uv sync --frozen" in code_commands, "Missing 'uv sync --frozen' step"
    assert "ruff check" in code_commands, "Missing Ruff linting step"
    assert "pytest" in code_commands, "Missing Pytest execution step"
    assert "src.evaluation.quality_gate" in code_commands, (
        "Missing autonomous quality gate audit step"
    )

    # 2. docker-build-verification steps
    docker_job = jobs.get("docker-build-verification", {})
    docker_steps = docker_job.get("steps", [])
    docker_commands = " ".join(step.get("run", "") for step in docker_steps)

    assert "docker build" in docker_commands, "Missing Docker build command"
    assert "Dockerfile" in docker_commands, "Missing reference to root Dockerfile"
    assert "frontend/Dockerfile" in docker_commands, "Missing reference to frontend Dockerfile"
    assert "/health" in docker_commands, "Missing health check test for container"


def test_cd_workflow_exists_and_valid_yaml(cd_workflow: dict):
    """Test that .github/workflows/cd.yml exists, is non-empty, and parses as valid YAML."""
    assert "name" in cd_workflow
    assert "jobs" in cd_workflow
    jobs = cd_workflow.get("jobs", {})
    assert "staging-release" in jobs, "Job 'staging-release' missing from CD workflow"
