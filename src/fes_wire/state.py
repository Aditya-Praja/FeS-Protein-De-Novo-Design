import hashlib
import json
import logging
import os
from pathlib import Path
from contextlib import contextmanager
from importlib.metadata import version

LOG = logging.getLogger(__name__)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


@contextmanager
def stage(run, name, inputs, config, force=False):
    sources = {str(p): digest(p) for p in sorted(map(Path, inputs))}
    code = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    software = {n: version(n) for n in ("numpy", "scipy", "biopython")}
    payload = dict(inputs=sources, config=config, code=code, software=software)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(run) / name / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "stage.json"
    lock = folder / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Stage locked at {lock}. Check running processes before removing stale lock"
        ) from exc
    os.close(fd)
    try:
        cached = False
        if manifest.exists() and not force:
            previous = load(manifest)
            cached = previous.get("status") == "complete" and all(
                Path(p).is_file() and digest(p) == h for p, h in previous["outputs"].items()
            )
        LOG.info("%s %s", "Reusing" if cached else "Running", name)
        yield folder, cached
        outputs = {
            str(p): digest(p)
            for p in folder.rglob("*")
            if p.is_file() and p not in (manifest, lock)
        }
        dump(manifest, dict(payload, status="complete", outputs=outputs))
        dump(Path(run) / (name + ".json"), {"directory": str(folder), "manifest": str(manifest)})
    except Exception:
        dump(manifest, dict(payload, status="failed"))
        raise
    finally:
        lock.unlink()


def latest(run, name):
    pointer = load(Path(run) / (name + ".json"))
    manifest = load(pointer["manifest"])
    if manifest.get("status") != "complete":
        raise ValueError(f"Upstream stage {name} incomplete")
    for p, h in {**manifest["inputs"], **manifest["outputs"]}.items():
        if not Path(p).is_file() or digest(p) != h:
            raise ValueError(f"Stale {name} stage, rerun upstream workflow: {p}")
    return Path(pointer["directory"])
