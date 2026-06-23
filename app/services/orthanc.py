"""
Orthanc DICOM client.
Orthanc is the open-source DICOM server running on the edge device.
This service polls Orthanc for new studies and triggers the inference pipeline.
"""
import httpx
import asyncio
from pathlib import Path
from typing import List, Optional
import structlog
from app.core.config import get_settings

log = structlog.get_logger()
settings = get_settings()

BASE = settings.ORTHANC_URL
AUTH = (settings.ORTHANC_USER, settings.ORTHANC_PASSWORD)


class OrthancClient:

    def __init__(self, base_url: str = BASE):
        self.base = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base,
            auth=AUTH,
            timeout=30.0,
        )

    async def health(self) -> bool:
        """Check if Orthanc is reachable."""
        try:
            r = await self.client.get("/system")
            return r.status_code == 200
        except Exception:
            return False

    async def list_studies(self) -> List[str]:
        """Return all study IDs currently in Orthanc."""
        r = await self.client.get("/studies")
        r.raise_for_status()
        return r.json()

    async def get_study_info(self, study_id: str) -> dict:
        r = await self.client.get(f"/studies/{study_id}")
        r.raise_for_status()
        return r.json()

    async def list_instances(self, study_id: str) -> List[str]:
        """Return all DICOM instance IDs within a study."""
        info = await self.get_study_info(study_id)
        instances = []
        for series_id in info.get("Series", []):
            r = await self.client.get(f"/series/{series_id}")
            r.raise_for_status()
            instances.extend(r.json().get("Instances", []))
        return instances

    async def download_instance(self, instance_id: str, dest: Path) -> Path:
        """Download a DICOM instance to disk."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = await self.client.get(f"/instances/{instance_id}/file")
        r.raise_for_status()
        dest.write_bytes(r.content)
        log.info("dicom_downloaded", instance_id=instance_id, path=str(dest))
        return dest

    async def upload_dicom_report(self, report_path: Path) -> Optional[str]:
        """Upload a DICOM structured report back into Orthanc (write-back to PACS)."""
        try:
            with open(report_path, "rb") as f:
                r = await self.client.post("/instances", content=f.read())
            r.raise_for_status()
            instance_id = r.json().get("ID")
            log.info("report_uploaded_to_orthanc", instance_id=instance_id)
            return instance_id
        except Exception as e:
            log.error("report_upload_failed", error=str(e))
            return None

    async def delete_instance(self, instance_id: str):
        """Remove a processed instance from Orthanc (after anonymisation + cloud sync)."""
        await self.client.delete(f"/instances/{instance_id}")
        log.info("instance_deleted_from_orthanc", instance_id=instance_id)

    async def close(self):
        await self.client.aclose()


# Singleton
orthanc = OrthancClient()
