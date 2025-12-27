import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["token"] = os.getenv("HUGGINGFACE_TOKEN")
    
    return config


def set_random_seeds(seed: int) -> None:
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)


