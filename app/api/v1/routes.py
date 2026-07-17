from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
import uuid
import tempfile

from app.core.database import get_db
from app.core.security import (
    verify_password, create_access_token, hash_password, get_current_user
)
from app.models import db as dbm, schemas
from app.services.inference import engine
from app.services.anonymisation import anonymise_dicom, extract_scan_metadata
from app.services.orthanc import orthanc

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health", tags=["system"])
async def health():
    orthanc_ok = await orthanc.health()
    return {
        "status": "ok",
        "orthanc": "connected" if orthanc_ok else "unreachable",
        "orthanc_url_debug": orthanc.base,
        "models_loaded": engine._loaded,
        "version": "1.0.0",
    }


# ── Auth ──────────────────────────────────────────────────────────────────────
@router.post("/auth/token", response_model=schemas.TokenResponse, tags=["auth"])
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(dbm.User).filter(
        dbm.User.email == form.username,
        dbm.User.active == True
    ).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = create_access_token({"sub": user.id, "role": user.role, "facility": user.facility_id})
    return {"access_token": token}


@router.post("/users", response_model=schemas.UserOut, tags=["auth"])
async def create_user(
    payload: schemas.UserCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    existing = db.query(dbm.User).filter(dbm.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = dbm.User(
        id=str(uuid.uuid4()),
        facility_id=payload.facility_id,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ── Facilities ────────────────────────────────────────────────────────────────
@router.get("/facilities", response_model=list[schemas.FacilityOut], tags=["facilities"])
async def list_facilities(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    if current_user.get("role") != "admin":
        # Sonographers only see their own facility
        return db.query(dbm.Facility).filter(
            dbm.Facility.id == current_user["facility"]
        ).all()
    return db.query(dbm.Facility).filter(dbm.Facility.active == True).all()


@router.post("/facilities", response_model=schemas.FacilityOut, tags=["facilities"])
async def create_facility(
    payload: schemas.FacilityCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    facility = dbm.Facility(id=str(uuid.uuid4()), **payload.model_dump())
    db.add(facility)
    db.commit()
    db.refresh(facility)
    return facility


# ── Scans ─────────────────────────────────────────────────────────────────────
@router.post("/scans/upload", response_model=schemas.ScanOut, tags=["scans"])
async def upload_scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scan_mode: str = "B-mode",
    indication: str = "OB-biometry",
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload a DICOM file directly.
    The file is anonymised immediately, then queued for inference.
    """
    if not file.filename.endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only .dcm files accepted")

    scan_id = str(uuid.uuid4())

    # Write upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(await file.read())
        raw_path = Path(tmp.name)

    # Anonymise immediately
    anon_path = Path(f"/tmp/anon_{scan_id}.dcm")
    try:
        anon_study_uid, _ = anonymise_dicom(raw_path, anon_path)
    except Exception as e:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {e}")
    raw_path.unlink(missing_ok=True)

    # Create scan record
    scan = dbm.Scan(
        id=scan_id,
        facility_id=current_user["facility"],
        submitted_by=current_user["sub"],
        scan_mode=scan_mode,
        indication=indication,
        dicom_path=str(anon_path),
        status="queued",
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    # Run inference in background
    background_tasks.add_task(_run_inference, scan_id, anon_path, db)

    return scan


@router.get("/scans/{scan_id}", response_model=schemas.ScanOut, tags=["scans"])
async def get_scan(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    scan = db.query(dbm.Scan).filter(dbm.Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return scan


@router.get("/scans/{scan_id}/result", response_model=schemas.InferenceResultOut, tags=["scans"])
async def get_result(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = db.query(dbm.InferenceResult).filter(
        dbm.InferenceResult.scan_id == scan_id
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not ready yet")
    return result


@router.get("/scans", response_model=list[schemas.ScanOut], tags=["scans"])
async def list_scans(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    q = db.query(dbm.Scan).filter(
        dbm.Scan.facility_id == current_user["facility"]
    ).order_by(dbm.Scan.created_at.desc()).limit(limit)
    return q.all()


# ── Analytics ─────────────────────────────────────────────────────────────────
@router.get("/facilities/{facility_id}/stats",
            response_model=schemas.FacilityStats, tags=["analytics"])
async def facility_stats(
    facility_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    from sqlalchemy import func
    from datetime import timedelta

    scans = db.query(dbm.Scan).filter(dbm.Scan.facility_id == facility_id)
    total = scans.count()
    flags = scans.filter(dbm.Scan.emergency_flag == True).count()
    last30 = scans.filter(
        dbm.Scan.created_at >= datetime.utcnow() - timedelta(days=30)
    ).count()

    avg_ms_row = db.query(func.avg(dbm.InferenceResult.inference_ms)).join(
        dbm.Scan, dbm.Scan.id == dbm.InferenceResult.scan_id
    ).filter(dbm.Scan.facility_id == facility_id).scalar()

    return schemas.FacilityStats(
        facility_id=facility_id,
        total_scans=total,
        emergency_flags=flags,
        avg_inference_ms=float(avg_ms_row) if avg_ms_row else None,
        scans_last_30_days=last30,
    )


# ── Background task ───────────────────────────────────────────────────────────
def _run_inference(scan_id: str, dicom_path: Path, db: Session):
    """Background task: run ML inference and save result."""
    scan = db.query(dbm.Scan).filter(dbm.Scan.id == scan_id).first()
    if not scan:
        return

    try:
        scan.status = "processing"
        db.commit()

        output = engine.run_inference(dicom_path)

        result = dbm.InferenceResult(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            model_version=output["model_version"],
            bpd_mm=output["bpd_mm"],
            hc_mm=output["hc_mm"],
            ac_mm=output["ac_mm"],
            fl_mm=output["fl_mm"],
            gestational_age_days=output["gestational_age_days"],
            ga_confidence=output["ga_confidence"],
            emergency_flag=output["emergency_flag"],
            emergency_confidence=output["emergency_confidence"],
            emergency_reason=output["emergency_reason"],
            anomalies_detected=output["anomalies_detected"],
            inference_ms=output["inference_ms"],
        )
        db.add(result)

        scan.status = "done"
        scan.emergency_flag = output["emergency_flag"]
        scan.processed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        scan.status = "error"
        db.commit()
        raise e
    finally:
        dicom_path.unlink(missing_ok=True)
