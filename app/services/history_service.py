"""Service layer for assessment history persistence and delayed ground-truth feedback."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session, joinedload

from app.db.models import AssessmentHistory, GroundTruthLabel
from app.schemas import AssessmentHistoryResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HistoryService:
    """Service handling assessment persistence and diagnostic ground-truth tracking."""

    @staticmethod
    def log_assessment(
        db: Session,
        patient_id: str,
        input_features: Dict[str, Any],
        prediction_response: PredictionResponse,
    ) -> AssessmentHistory:
        """Persist a patient inference assessment record into the database.

        Args:
            db: Active SQLAlchemy session.
            patient_id: Unique patient identifier.
            input_features: Raw physiological and biometric inputs evaluated.
            prediction_response: Prediction outcome from model inference.

        Returns:
            Created AssessmentHistory instance.
        """
        record = AssessmentHistory(
            patient_id=patient_id,
            input_features=input_features,
            risk_probability=prediction_response.probability,
            risk_score=prediction_response.risk_score,
            risk_category=prediction_response.risk_category,
            model_version=prediction_response.model_version,
            created_at=datetime.now(timezone.utc),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(
            "Logged assessment #%d for patient '%s' (Risk: %d - %s)",
            record.id,
            patient_id,
            record.risk_score,
            record.risk_category,
        )
        return record

    @staticmethod
    def get_patient_history(db: Session, patient_id: str) -> List[AssessmentHistory]:
        """Retrieve sequential assessment trajectory for a specific patient chronologically.

        Args:
            db: Active SQLAlchemy session.
            patient_id: Unique patient identifier.

        Returns:
            List of AssessmentHistory records sorted from earliest to latest.
        """
        return (
            db.query(AssessmentHistory)
            .options(joinedload(AssessmentHistory.ground_truth))
            .filter(AssessmentHistory.patient_id == patient_id)
            .order_by(AssessmentHistory.created_at.asc(), AssessmentHistory.id.asc())
            .all()
        )

    @staticmethod
    def get_all_history(
        db: Session,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[AssessmentHistory], int]:
        """Retrieve global assessment logs with pagination.

        Args:
            db: Active SQLAlchemy session.
            limit: Maximum records to return.
            offset: Number of records to skip.

        Returns:
            Tuple of (records_list, total_record_count).
        """
        total = db.query(AssessmentHistory).count()
        records = (
            db.query(AssessmentHistory)
            .options(joinedload(AssessmentHistory.ground_truth))
            .order_by(AssessmentHistory.created_at.desc(), AssessmentHistory.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return records, total

    @staticmethod
    def record_ground_truth(
        db: Session,
        assessment_id: int,
        verified_label: int,
        notes: Optional[str] = None,
    ) -> GroundTruthLabel:
        """Bind confirmed clinical diagnostic outcome to an existing assessment.

        Args:
            db: Active SQLAlchemy session.
            assessment_id: Primary key of AssessmentHistory record.
            verified_label: Clinical diagnosis (0: Non-Diabetic, 1: Diabetic).
            notes: Optional diagnostic details.

        Returns:
            Saved GroundTruthLabel instance.

        Raises:
            KeyError: If the specified assessment_id does not exist.
        """
        assessment = (
            db.query(AssessmentHistory)
            .filter(AssessmentHistory.id == assessment_id)
            .first()
        )
        if not assessment:
            raise KeyError(f"Assessment record with id {assessment_id} does not exist.")

        # Check for existing ground truth label
        label_record = (
            db.query(GroundTruthLabel)
            .filter(GroundTruthLabel.assessment_id == assessment_id)
            .first()
        )

        now_utc = datetime.now(timezone.utc)
        if label_record:
            label_record.verified_diabetes_label = verified_label
            label_record.notes = notes
            label_record.verified_at = now_utc
        else:
            label_record = GroundTruthLabel(
                assessment_id=assessment_id,
                patient_id=assessment.patient_id,
                verified_diabetes_label=verified_label,
                notes=notes,
                verified_at=now_utc,
            )
            db.add(label_record)

        db.commit()
        db.refresh(label_record)
        logger.info(
            "Bound ground truth label (%d) to assessment #%d (patient '%s')",
            verified_label,
            assessment_id,
            assessment.patient_id,
        )
        return label_record

    @staticmethod
    def to_dto(record: AssessmentHistory) -> AssessmentHistoryResponse:
        """Transform SQLAlchemy ORM record into Pydantic DTO."""
        gt = record.ground_truth
        return AssessmentHistoryResponse(
            id=record.id,
            patient_id=record.patient_id,
            created_at=record.created_at.isoformat() if record.created_at else "",
            input_features=record.input_features or {},
            risk_probability=record.risk_probability,
            risk_score=record.risk_score,
            risk_category=record.risk_category,
            model_version=record.model_version,
            verified_diabetes_label=gt.verified_diabetes_label if gt else None,
            verified_at=gt.verified_at.isoformat() if gt and gt.verified_at else None,
            notes=gt.notes if gt else None,
        )


history_service = HistoryService()
