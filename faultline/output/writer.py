import json
import re
from datetime import datetime, timezone
from pathlib import Path

from faultline.models.types import FeatureMap


def write_feature_map(feature_map: FeatureMap, output_path: str | None = None) -> str:
    """
    Writes the feature map to a JSON file.

    When output_path is not specified, generates a unique filename using the
    repository name and current UTC timestamp so each run produces a new file
    and history is preserved:
        .faultline/feature-map-{repo-slug}-{YYYYMMDD-HHMMSS}.json

    Returns the path where the file was saved.
    """
    if output_path is not None:
        path = Path(output_path)
    else:
        slug = _repo_slug(feature_map.repo_path)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = Path.home() / ".faultline" / f"feature-map-{slug}-{ts}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    data = feature_map.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str))

    return str(path)


def _repo_slug(repo_path: str) -> str:
    """Converts a repo path to a safe filename component."""
    name = Path(repo_path).name
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "repo"
