from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ── Facility ──────────────────────────────────────────────────────────────────
class FacilityCreate(BaseModel):
    name: str
    location: Optional[str] = None
    tier: int = 1
    orthanc_url: Optional[str] = None


class FacilityOut(BaseModel):
    id: str
    name: str
    location: Optional[str]
    tier: int
    active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── User ──────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: str = "sonographer"
    facility_id: str


class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    facility_id: str
    active: bool

    class Config:
        from_attributes = True


# ── Scan ──────────────────────────────────────────────────────────────────────
class ScanSubmit(BaseModel):
    orthanc_study_id: str
    scan_mode: str = "B-mode"
    indication: str = "OB-biometry"


class ScanOut(BaseModel):
    id: str
    facility_id: str
    status: str
    emergency_flag: bool
    scan_mode: Optional[str]
    indication: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ── Inference result ──────────────────────────────────────────────────────────
class AnomalyItem(BaseModel):
    label: str
    confidence: float


class InferenceResultOut(BaseModel):
    id: str
    scan_id: str
    model_version: Optional[str]

    bpd_mm: Optional[float]
    hc_mm: Optional[float]
    ac_mm: Optional[float]
    fl_mm: Optional[float]

    gestational_age_days: Optional[int]
    ga_confidence: Optional[float]

    emergency_flag: bool
    emergency_confidence: Optional[float]
    emergency_reason: Optional[str]

    anomalies_detected: Optional[List[AnomalyItem]]

    inference_ms: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Analytics ─────────────────────────────────────────────────────────────────
class FacilityStats(BaseModel):
    facility_id: str
    total_scans: int
    emergency_flags: int
    avg_inference_ms: Optional[float]
    scans_last_30_days: int
