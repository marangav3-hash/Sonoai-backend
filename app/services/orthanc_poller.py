"""
Orthanc poller.
Periodically checks Orthanc for new studies pushed from ultrasound machines
(via DICOM C-STORE) or uploaded manually, downloads + anonymises them,
creates Scan records, and triggers inference — the same pipeline used for
direct app uploads.
"""
import asyncio
import tempfile
import uuid as uuid_lib
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models import db as dbm
from app.services.anonymisation import anonymise_dicom
from app.services.orthanc import orthanc

log = structlog.get_logger()
settings = get_settings()


async def _process_study(study_id: str, db: Session):
    """Download, anonymise, and queue inference for a single new Orthanc study."""
    try:
        instances = await orthanc.list_instances(study_id)
        if not instances:
            log.warning("orthanc_study_empty", study_id=study_id)
            return

        # Use the first instance in the study (single-image ultrasound capture)
        instance_id = instances[0]

        scan_id = str(uuid_lib.uuid4())
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            raw_path = Path(tmp.name)
        await orthanc.download_instance(instance_id, raw_path)

        anon_path = Path(f"/tmp/anon_{scan_id}.dcm")
        try:
            anon_study_uid, _ = anonymise_dicom(raw_path, anon_path)
        except Exception as e:
            log.error("orthanc_anonymise_failed", study_id=study_id, error=str(e))
            raw_path.unlink(missing_ok=True)
            return
        raw_path.unlink(missing_ok=True)

        scan = dbm.Scan(
            id=scan_id,
            facility_id=settings.ORTHANC_DEFAULT_FACILITY_ID,
            submitted_by=None,
            orthanc_study_id=study_id,
            dicom_path=str(anon_path),
            modality="US",
            scan_mode="B-mode",
            indication="OB-biometry",
            status="queued",
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)

        log.info("orthanc_study_ingested", study_id=study_id, scan_id=scan_id)

        # Run inference inline (poller runs in its own background task, not a request handler)
        from app.api.v1.routes import _run_inference
        _run_inference(scan_id, anon_path, db)

    except Exception as e:
        log.error("orthanc_process_study_failed", study_id=study_id, error=str(e))


async def poll_once():
    """Check Orthanc for studies not yet ingested, and process each."""
    db = SessionLocal()
    try:
        if not await orthanc.health():
            log.warning("orthanc_poll_skipped_unreachable")
            return

        study_ids = await orthanc.list_studies()
        for study_id in study_ids:
            already_ingested = (
                db.query(dbm.Scan)
                .filter(dbm.Scan.orthanc_study_id == study_id)
                .first()
            )
            if already_ingested:
                continue
            await _process_study(study_id, db)
    except Exception as e:
        log.error("orthanc_poll_failed", error=str(e))
    finally:
        db.close()


async def poller_loop():
    """Background loop: poll Orthanc on a fixed interval, forever."""
    log.info("orthanc_poller_starting", interval=settings.ORTHANC_POLL_INTERVAL_SECONDS)
    while True:
        await poll_once()
        await asyncio.sleep(settings.ORTHANC_POLL_INTERVAL_SECONDS)
