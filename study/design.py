"""Reconstruct fixed cohorts and geometry-only catalogues; no split selection."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import engine as prior

ROOT = prior.ROOT
OUT = ROOT / "run/primary"
PROTOCOL = ROOT / "protocol.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frames():
    """Recreate exactly the prior numerical cohort and map eligibility."""
    OUT.mkdir(exist_ok=True)
    prior.OUT = OUT
    acquisition = json.loads((prior.BASE / "acquisition.json").read_text())
    for item in acquisition["files"]:
        assert digest(prior.BASE / item.get("name", "county_adjacency2023.txt")) == item["sha256"]
    extraction = json.loads((prior.BASE / "schema-extraction.json").read_text())
    for name, sha in extraction["derived_sha256"].items():
        assert digest(prior.BASE / name) == sha
    names, graph, aliases = prior.county_metadata()
    plants, centers, paudit = prior.inspect_plants(names, aliases)
    census, caudit = prior.inspect_census()
    patches, _ = prior.make_patches(graph)
    assignment = {county: "+".join(sorted(p)) for p in patches for county in p}
    plants["map"] = plants.source.map(assignment)
    domains, controls = {}, {}
    for domain, frame in [("census", census), ("egrid", plants.dropna(subset=["map"]))]:
        inventory, eligible = prior.inventory_maps(domain, frame, frame.groupby("map", sort=True), centers)
        domains[domain] = {str(key): rows for key, rows, _, _ in eligible}
        controls[domain] = {"zero_mass_maps": sum(x["zero_mass_control"] for x in inventory),
                            "single_location_maps": sum(x["fine_locations"] == 1 for x in inventory),
                            "all_inventory": inventory}
    assert len(domains["census"]) == 60 and len(domains["egrid"]) == 369
    return domains, centers, {"census": caudit, "egrid": paudit}, controls


def catalog_for(frame, centers):
    """All distinct crossing 3-mile buffer masks, selected without fine errors."""
    candidates = centers[centers.LAT.between(frame.lat.min() - .1, frame.lat.max() + .1)
                         & centers.LON.between(frame.lon.min() - .15, frame.lon.max() + .15)]
    sources = [source.index.to_numpy() for _, source in frame.groupby("source") if source.truth.sum() > 0]
    found = {}
    for center in candidates.itertuples():
        _, _, meters = prior.GEOD.inv(np.full(len(frame), center.LON), np.full(len(frame), center.LAT),
                                      frame.lon.to_numpy(), frame.lat.to_numpy())
        mask = np.asarray(meters) <= 4828.032
        if not any(mask[g].any() and not mask[g].all() for g in sources):
            continue
        key = np.packbits(mask, bitorder="little").tobytes().hex()
        if key not in found:
            found[key] = {"query_id": hashlib.sha256(bytes.fromhex(key)).hexdigest()[:12],
                          "mask_hex": key, "n": len(frame), "first_ORISPL": str(center.ORISPL),
                          "all_center_ids": [], "lat": center.LAT, "lon": center.LON}
        found[key]["all_center_ids"].append(str(center.ORISPL))
    return list(found.values())
