from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    APP_SECRET_KEY: str = "dev-secret-change-in-prod"
    APP_DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql://sonoai:sonoai_pass@localhost:5432/sonoai_db"

    # Storage
    STORAGE_ENDPOINT: str = "http://localhost:9000"
    STORAGE_ACCESS_KEY: str = "minioadmin"
    STORAGE_SECRET_KEY: str = "minioadmin"
    STORAGE_BUCKET_ANONYMISED: str = "sonoai-anonymised"
    STORAGE_BUCKET_MODELS: str = "sonoai-models"

    # Orthanc
    ORTHANC_URL: str = "http://localhost:8042"
    ORTHANC_USER: str = "orthanc"
    ORTHANC_PASSWORD: str = "orthanc"

    # JWT
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Models
    MODEL_BIOMETRY_PATH: str = "/models/biometry_v1.pt"
    MODEL_EMERGENCY_PATH: str = "/models/emergency_flag_v1.pt"
    MODEL_DEVICE: str = "cpu"

    # Alerts
    ALERT_WEBHOOK_URL: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
