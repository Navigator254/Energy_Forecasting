from pathlib import Path
from dotenv import load_dotenv


def load_project_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)
