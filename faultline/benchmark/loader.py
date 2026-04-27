"""YAML ground-truth loaders for the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # PyYAML — already a dependency of faultline
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for the benchmark harness; "
        "install with `pip install pyyaml`."
    ) from _exc


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpectedFeature:
    """One ground-truth feature for a repo.

    A detected feature counts as the same feature when its name is
    either ``name`` itself or one of ``aliases``. ``must_include``
    lists files that must end up attributed to this feature for the
    attribution metric to give it credit.
    """

    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    must_include: tuple[str, ...] = ()
    package_hint: str = ""

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name,) + self.aliases


@dataclass(frozen=True)
class ExpectedFlow:
    """One expected user-facing flow."""

    name: str
    feature: str = ""
    description: str = ""


@dataclass(frozen=True)
class ExpectedAttribution:
    """One sample (path → expected feature) for the attribution metric."""

    path: str
    expected: str


@dataclass
class BenchmarkSpec:
    """All ground-truth artefacts for a single repo, gathered."""

    repo: str
    features: list[ExpectedFeature] = field(default_factory=list)
    flows: list[ExpectedFlow] = field(default_factory=list)
    attribution: list[ExpectedAttribution] = field(default_factory=list)


# ── Loaders ──────────────────────────────────────────────────────────


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a YAML mapping at the top level, got "
            f"{type(data).__name__}"
        )
    return data


def load_expected_features(path: Path | str) -> list[ExpectedFeature]:
    """Load ``expected_features.yaml`` and return a list of features."""
    data = _read_yaml(Path(path))
    raw = data.get("features") or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'features' must be a list, got {type(raw).__name__}")

    out: list[ExpectedFeature] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: feature entry must be a mapping")
        name = (entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"{path}: feature entry missing 'name'")
        if name in seen:
            raise ValueError(f"{path}: duplicate feature name {name!r}")
        seen.add(name)

        aliases_raw = entry.get("aliases") or []
        if not isinstance(aliases_raw, list):
            raise ValueError(f"{path}: aliases for {name!r} must be a list")
        aliases = tuple(str(a).strip() for a in aliases_raw if str(a).strip())

        must_raw = entry.get("must_include") or []
        if not isinstance(must_raw, list):
            raise ValueError(f"{path}: must_include for {name!r} must be a list")
        must_include = tuple(str(p).strip() for p in must_raw if str(p).strip())

        out.append(ExpectedFeature(
            name=name,
            description=(entry.get("description") or "").strip(),
            aliases=aliases,
            must_include=must_include,
            package_hint=(entry.get("package_hint") or "").strip(),
        ))
    return out


def load_expected_flows(path: Path | str) -> list[ExpectedFlow]:
    """Load ``expected_flows.yaml``."""
    data = _read_yaml(Path(path))
    raw = data.get("flows") or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'flows' must be a list")

    out: list[ExpectedFlow] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: flow entry must be a mapping")
        name = (entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"{path}: flow entry missing 'name'")
        out.append(ExpectedFlow(
            name=name,
            feature=(entry.get("feature") or "").strip(),
            description=(entry.get("description") or "").strip(),
        ))
    return out


def load_expected_attribution(path: Path | str) -> list[ExpectedAttribution]:
    """Load ``expected_attribution_sample.yaml``."""
    data = _read_yaml(Path(path))
    raw = data.get("samples") or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'samples' must be a list")

    out: list[ExpectedAttribution] = []
    seen_paths: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: sample entry must be a mapping")
        sp = (entry.get("path") or "").strip()
        ex = (entry.get("expected") or "").strip()
        if not sp or not ex:
            raise ValueError(f"{path}: sample needs both 'path' and 'expected'")
        if sp in seen_paths:
            raise ValueError(f"{path}: duplicate sample path {sp!r}")
        seen_paths.add(sp)
        out.append(ExpectedAttribution(path=sp, expected=ex))
    return out


def load_benchmark(repo: str, root: Path | str) -> BenchmarkSpec:
    """Load all three ground-truth files for one repo (some may be empty).

    ``root`` is the directory holding e.g. ``benchmarks/<repo>/``.
    Missing files are tolerated (treated as empty lists) so partial
    benchmarks still score what's available.
    """
    base = Path(root) / repo
    features = []
    flows = []
    attribution = []

    f_path = base / "expected_features.yaml"
    if f_path.exists():
        features = load_expected_features(f_path)
    fl_path = base / "expected_flows.yaml"
    if fl_path.exists():
        flows = load_expected_flows(fl_path)
    at_path = base / "expected_attribution_sample.yaml"
    if at_path.exists():
        attribution = load_expected_attribution(at_path)

    return BenchmarkSpec(
        repo=repo,
        features=features,
        flows=flows,
        attribution=attribution,
    )
