from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLML_", extra="ignore")

    data_dir: Path = Path(".data")
    random_state: int = 42
    n_jobs: int = -1
    optuna_trials: int = 8
    mlflow_tracking_uri: str = "sqlite:///.data/mlflow.db"
    plot_format: str = "png"
    feature_engineering: bool = False
    feature_rules: Path | None = None

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    return settings
