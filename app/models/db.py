import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Float, JSON, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class Facility(Base):
    """A hospital or clinic using SonoAI."""
    __tablename__ = "facilities"

    id            = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name          = Column(String(255), nullable=False)
    location      = Column(String(255))
    tier          = Column(Integer, default=1)          # 1=core, 2=analytics, 3=audit
    active        = Column(Boolean, default=True)
    orthanc_url   = Column(String(255))                 # local Orthanc instance
    created_at    = Column(DateTime, default=datetime.utcnow)

    users  = relationship("User", back_populates="facility")
    scans  = relationship("Scan", back_populates="facility")


class User(Base):
    """Sonographers and admins at a facility."""
    __tablename__ = "users"

    id            = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    facility_id   = Column(UUID(as_uuid=False), ForeignKey("facilities.id"), nullable=False)
    email         = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name     = Column(String(255))
    role          = Column(String(50), default="sonographer")  # sonographer | admin
    active        = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    facility = relationship("Facility", back_populates="users")


class Scan(Base):
    """A single ultrasound scan submitted for AI inference."""
    __tablename__ = "scans"

    id               = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    facility_id      = Column(UUID(as_uuid=False), ForeignKey("facilities.id"), nullable=False)
    submitted_by     = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    orthanc_study_id = Column(String(255))              # Orthanc study UID
    dicom_path       = Column(String(512))              # anonymised storage path
    modality         = Column(String(20))               # US (ultrasound)
    scan_mode        = Column(String(20))               # B-mode | M-mode | Doppler
    indication       = Column(String(100))              # OB-biometry | POCUS | etc.
    status           = Column(String(50), default="queued")  # queued|processing|done|error
    emergency_flag   = Column(Boolean, default=False)
    created_at       = Column(DateTime, default=datetime.utcnow)
    processed_at     = Column(DateTime)

    facility = relationship("Facility", back_populates="scans")
    result   = relationship("InferenceResult", back_populates="scan", uselist=False)


class InferenceResult(Base):
    """AI inference output for a scan."""
    __tablename__ = "inference_results"

    id                   = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    scan_id              = Column(UUID(as_uuid=False), ForeignKey("scans.id"), nullable=False)
    model_version        = Column(String(50))

    # Fetal biometry (mm)
    bpd_mm               = Column(Float)   # biparietal diameter
    hc_mm                = Column(Float)   # head circumference
    ac_mm                = Column(Float)   # abdominal circumference
    fl_mm                = Column(Float)   # femur length

    # Gestational age
    gestational_age_days = Column(Integer)
    ga_confidence        = Column(Float)   # 0–1

    # Emergency flag
    emergency_flag       = Column(Boolean, default=False)
    emergency_confidence = Column(Float)   # 0–1
    emergency_reason     = Column(String(255))

    # Anomaly detection
    anomalies_detected   = Column(JSON)    # list of {label, confidence}

    # Full structured report
    dicom_report_path    = Column(String(512))
    raw_output           = Column(JSON)

    inference_ms         = Column(Integer)  # latency in ms
    created_at           = Column(DateTime, default=datetime.utcnow)

    scan = relationship("Scan", back_populates="result")
