"""SQLAlchemy ORM models for AssessmentHistory and GroundTruthLabel.

Tracks longitudinal patient assessment events and binds delayed clinical diagnostic feedback.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class AssessmentHistory(Base):
    """Clinical assessment history record capturing raw features and model inference."""

    __tablename__ = "assessment_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(128), index=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    input_features = Column(JSON, nullable=False)
    risk_probability = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_category = Column(String(32), nullable=False)
    model_version = Column(String(32), nullable=False)

    # One-to-one relationship to verified ground truth label
    ground_truth = relationship(
        "GroundTruthLabel",
        back_populates="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )


class GroundTruthLabel(Base):
    """Delayed clinical verification binding true diabetic diagnostic outcomes to assessments."""

    __tablename__ = "ground_truth_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    assessment_id = Column(
        Integer,
        ForeignKey("assessment_history.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    patient_id = Column(String(128), index=True, nullable=False)
    verified_diabetes_label = Column(Integer, nullable=False)  # 0 or 1
    verified_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    notes = Column(Text, nullable=True)

    assessment = relationship("AssessmentHistory", back_populates="ground_truth")
