"""Run the pinned RFD3 parser and symmetry builder without loading a model."""

import argparse
from pathlib import Path
import numpy as np
from rfd3.inference.input_parsing import DesignInputSpecification
from fes_wire.state import latest, load, dump

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
prepared = latest(Path("runs") / args.run_id, "rfd_inputs")

reports = []
for name, settings in load(prepared / "inputs.json").items():
    settings["input"] = str((prepared / settings["input"]).resolve())
    array = DesignInputSpecification(**settings).build()
    order = int(settings["symmetry"]["id"][1:])
    fes = array[array.res_name == "FES"]
    inventory = dict(zip(*np.unique(fes.atom_name, return_counts=True)))
    if inventory != {"FE1": order, "FE2": order, "S1": order, "S2": order}:
        raise ValueError(f"RFD3 altered FES atom inventory: {inventory}")
    for atom in fes:
        expected = "Fe" if atom.atom_name.startswith("FE") else "S"
        if atom.element.upper() != expected.upper():
            raise ValueError(f"Incorrect element for {atom.atom_name}: {atom.element}")
    reports.append(
        dict(
            name=name,
            atom_count=len(array),
            fes_atom_inventory={k: int(v) for k, v in inventory.items()},
        )
    )
dump(Path("runs") / args.run_id / "rfd3_parser_validation.json", reports)
print("RFD3 parser and symmetry construction passed. No inference was executed.")
