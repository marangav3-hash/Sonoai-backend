"""
DICOM anonymisation service.
Strips all patient-identifying tags before images are stored or sent to cloud.
This is a legal requirement under Kenya Data Protection Act 2019.
"""
import pydicom
import hashlib
import uuid
from pathlib import Path
from typing import Tuple
import structlog

log = structlog.get_logger()

# Tags to completely remove (patient identity)
REMOVE_TAGS = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1030),  # PatientWeight
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0040, 0xA123),  # PersonName
    (0x0008, 0x009C),  # ConsultingPhysicianName
]

# Tags to replace with anonymised values (needed for structural integrity)
REPLACE_TAGS = {
    (0x0020, 0x000D): None,  # StudyInstanceUID → new UID
    (0x0020, 0x000E): None,  # SeriesInstanceUID → new UID
    (0x0008, 0x0018): None,  # SOPInstanceUID → new UID
    (0x0020, 0x0010): "ANON",  # StudyID
    (0x0008, 0x0050): "ANON",  # AccessionNumber
}


def _new_uid(original_uid: str) -> str:
    """Generate a deterministic but anonymised UID from the original."""
    hashed = hashlib.sha256(original_uid.encode()).hexdigest()[:16]
    return f"2.25.{int(hashed, 16)}"


def anonymise_dicom(input_path: Path, output_path: Path) -> Tuple[str, str]:
    """
    Anonymise a DICOM file.
    Returns (anon_study_uid, anon_series_uid).
    """
    log.info("anonymising_dicom", input=str(input_path))
    try:
        ds = pydicom.dcmread(str(input_path), force=True)
    except Exception as e:
        log.error("dicom_read_failed", input=str(input_path), error=str(e))
        raise ValueError(f"Could not read DICOM file: {e}")

    # Remove identifying tags
    for tag in REMOVE_TAGS:
        if tag in ds:
            del ds[tag]

   # Replace with anonymised values
    def _get_uid(tag):
        elem = ds.get(tag)
        if elem is not None and elem.value:
            return str(elem.value)
        return str(uuid.uuid4())

    original_study_uid  = _get_uid((0x0020, 0x000D))
    original_series_uid = _get_uid((0x0020, 0x000E))
    original_sop_uid    = _get_uid((0x0008, 0x0018))

    anon_study_uid  = _new_uid(original_study_uid)
    anon_series_uid = _new_uid(original_series_uid)
    anon_sop_uid    = _new_uid(original_sop_uid)

    def _set_uid(tag, value):
        if tag in ds:
            ds[tag].value = value
        else:
            ds.add_new(tag, "UI", value)

    _set_uid((0x0020, 0x000D), anon_study_uid)
    _set_uid((0x0020, 0x000E), anon_series_uid)
    _set_uid((0x0008, 0x0018), anon_sop_uid)

    for tag, value in REPLACE_TAGS.items():
        if tag in ds and value is not None:
            ds[tag].value = value

    # Add de-identification note
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "SonoAI-Anon-v1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))

    log.info("anonymisation_complete",
             anon_study_uid=anon_study_uid,
             output=str(output_path))

    return anon_study_uid, anon_series_uid


def extract_scan_metadata(dicom_path: Path) -> dict:
    """Extract safe, non-identifying metadata from a DICOM file."""
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    return {
        "modality":      str(getattr(ds, "Modality", "US")),
        "rows":          int(getattr(ds, "Rows", 0)),
        "columns":       int(getattr(ds, "Columns", 0)),
        "photometric":   str(getattr(ds, "PhotometricInterpretation", "")),
        "ultrasound_regions": len(getattr(ds, "SequenceOfUltrasoundRegions", [])),
    }
