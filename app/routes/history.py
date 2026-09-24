"""Routes for querying assessment trajectory history and binding delayed ground-truth feedback."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import (
    DelayedLabelInput,
    DelayedLabelResponse,
    HistoryListResponse,
)
from app.services.history_service import history_service

router = APIRouter(tags=["History & Delayed Labels"])


@router.get(
    "/history/{patient_id}",
    response_model=HistoryListResponse,
    summary="Get Patient Assessment History",
    description="Retrieves chronological assessment trajectory for a specific patient to visualize longitudinal risk progression.",
)
def get_patient_history(
    patient_id: str,
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    """Retrieve all chronological assessment events for a patient."""
    records = history_service.get_patient_history(db=db, patient_id=patient_id)
    dtos = [history_service.to_dto(r) for r in records]
    return HistoryListResponse(
        patient_id=patient_id,
        total_records=len(dtos),
        assessments=dtos,
    )


@router.get(
    "/history",
    response_model=HistoryListResponse,
    summary="Get Global Assessment History",
    description="Retrieves recent assessments across all patients with pagination support.",
)
def get_global_history(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of assessments to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    """Retrieve paginated global assessment history records."""
    records, total = history_service.get_all_history(db=db, limit=limit, offset=offset)
    dtos = [history_service.to_dto(r) for r in records]
    return HistoryListResponse(
        patient_id=None,
        total_records=total,
        assessments=dtos,
    )


@router.post(
    "/feedback/ground-truth",
    response_model=DelayedLabelResponse,
    summary="Record Delayed Ground Truth Label",
    description="Ingests confirmed diagnostic verification (0: Non-Diabetic, 1: Diabetic) for an existing assessment.",
)
def record_ground_truth(
    label_input: DelayedLabelInput,
    db: Session = Depends(get_db),
) -> DelayedLabelResponse:
    """Bind confirmed diabetic clinical outcome to a previous assessment record."""
    try:
        label_record = history_service.record_ground_truth(
            db=db,
            assessment_id=label_input.assessment_id,
            verified_label=label_input.verified_diabetes_label,
            notes=label_input.notes,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    return DelayedLabelResponse(
        id=label_record.id,
        assessment_id=label_record.assessment_id,
        patient_id=label_record.patient_id,
        verified_diabetes_label=label_record.verified_diabetes_label,
        verified_at=label_record.verified_at.isoformat() if label_record.verified_at else "",
        notes=label_record.notes,
        message="Ground truth clinical verification successfully recorded.",
    )
