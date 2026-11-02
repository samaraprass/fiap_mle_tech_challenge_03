import os
from pathlib import Path
import yaml
from pydantic import BaseModel

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class AppConfig(BaseModel):
    title: str = "Medical Triage API"
    version: str = "1.0.0"
    description: str = "API de triagem automatizada de laudos médicos com NLP para suporte à decisão clínica hospitalar"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

class ModelConfig(BaseModel):
    model_path: str = "models/baseline_model.joblib"
    metadata_path: str = "models/classes_mapping.json"

class Settings:
    def __init__(self):
        config_path = ROOT_DIR / "config.yaml"
        config_data = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception:
                config_data = {}

        app_dict = config_data.get("app", {})
        model_dict = config_data.get("model", {})

        self.app = AppConfig(
            title=os.getenv("APP_TITLE", app_dict.get("title", "Medical Triage API")),
            version=os.getenv("APP_VERSION", app_dict.get("version", "1.0.0")),
            description=os.getenv("APP_DESC", app_dict.get("description", "API de triagem de laudos")),
            host=os.getenv("APP_HOST", app_dict.get("host", "0.0.0.0")),
            port=int(os.getenv("APP_PORT", app_dict.get("port", 8000))),
            log_level=os.getenv("LOG_LEVEL", app_dict.get("log_level", "INFO"))
        )

        model_p = os.getenv("MODEL_PATH", model_dict.get("model_path", "models/baseline_model.joblib"))
        meta_p = os.getenv("METADATA_PATH", model_dict.get("metadata_path", "models/classes_mapping.json"))

        self.model = ModelConfig(
            model_path=str(ROOT_DIR / model_p if not os.path.isabs(model_p) else model_p),
            metadata_path=str(ROOT_DIR / meta_p if not os.path.isabs(meta_p) else meta_p)
        )

settings = Settings()

