"""Manual Slurm entry point, dry run unless --execute is explicitly supplied."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from fes_wire.state import latest, load, dump, digest
from fes_wire.rfd3 import REVISION

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-id", required=True)
p.add_argument("--index", type=int, default=0)
p.add_argument("--execute", action="store_true")
a = p.parse_args()
run = Path("runs") / a.run_id
prepared = latest(run, "rfd_inputs")
jobs = load(prepared / "jobs.json")
if not 0 <= a.index < len(jobs):
    p.error("Job-array index outside prepared job manifest")
job = jobs[a.index]
checkpoint = os.environ.get("RFD3_CHECKPOINT")
if not checkpoint or not Path(checkpoint).is_file():
    p.error("RFD3_CHECKPOINT must point to an existing checkpoint")
runtime_spec = load(job["inputs"])
for specification in runtime_spec.values():
    specification["input"] = str((prepared / specification["input"]).resolve())
resolved = run / "job_inputs" / f"job_{a.index:05d}.json"
dump(resolved, runtime_spec)
command = [
    "rfd3",
    "design",
    f"inputs={resolved.resolve()}",
    f"out_dir={job['out_dir']}",
    f"ckpt_path={checkpoint}",
    f"seed={job['seed']}",
    "diffusion_batch_size=1",
    "n_batches=1",
    "inference_sampler.kind=symmetry",
    "prevalidate_inputs=True",
    "skip_existing=True",
    "dump_trajectories=False",
    *shlex.split(os.environ.get("FES_RFD3_EXTRA_ARGS", "")),
]
print(shlex.join(command), flush=True)
if a.execute:
    from importlib.metadata import distribution

    package = distribution("rc-foundry")
    origin = json.loads(package.read_text("direct_url.json") or "{}")
    if origin.get("vcs_info", {}).get("commit_id") != REVISION:
        p.error("Install the pinned Foundry Git revision from docs/execution.md")
    folder = Path(job["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        command=command, revision=REVISION, checkpoint_sha256=digest(checkpoint), job=job
    )
    provenance_path = folder / "execution.json"
    if (
        provenance_path.exists()
        and load(provenance_path).get("checkpoint_sha256") != provenance["checkpoint_sha256"]
    ):
        p.error("Checkpoint changed within an existing run. Prepare a new run ID")
    dump(provenance_path, dict(provenance, status="running"))
    result = subprocess.run(command, check=False)
    finals = list(folder.glob("*_model_*.cif.gz"))
    complete = (
        result.returncode == 0
        and bool(finals)
        and all(x.with_name(x.name.removesuffix(".cif.gz") + ".json").exists() for x in finals)
    )
    dump(
        provenance_path,
        dict(provenance, status="complete" if complete else "failed", returncode=result.returncode),
    )
    sys.exit(0 if complete else (result.returncode or 2))
