from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLML_", extra="ignore")

    data_dir: Path = Path(".data")
    random_state: int = 42
    n_jobs: int = -1
    optuna_trials: int = 8
    mlflow_tracking_uri: str = ""
    plot_format: str = "png"
    feature_engineering: bool = False
    feature_rules: Path | None = None
    log_level: str = "WARNING"
    log_file: Path | None = None

    @field_validator("feature_rules", "log_file", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def _set_mlflow_default(self) -> "Settings":
        if not self.mlflow_tracking_uri:
            uri = f"sqlite:///{self.service_dir / 'mlflow.db'}"
            object.__setattr__(self, "mlflow_tracking_uri", uri)
        return self

    # ------------------------------------------------------------------
    # Directory layout under data_dir
    # ------------------------------------------------------------------

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def service_dir(self) -> Path:
        """Service files: MLflow DB, matplotlib cache, etc."""
        return self.data_dir / "service"

    @property
    def runs_dir(self) -> Path:
        """Root for all method run outputs, grouped by task group."""
        return self.data_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        """Run-all summaries, exports, and other generated reports."""
        return self.data_dir / "reports"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.service_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
