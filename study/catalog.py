"""Reconstruct the frozen geometry-only query catalog, without changing splits."""
import json
import time
import pandas as pd
from design import load_frames, catalog_for, OUT, PROTOCOL, digest


def main():
    started = time.perf_counter()
    domains, centers, _, _ = load_frames()
    rows = []
    for domain, maps in domains.items():
        for key, frame in maps.items():
            for q in catalog_for(frame, centers):
                rows.append({"domain": domain, "map": key, **q,
                             "all_center_ids": json.dumps(q["all_center_ids"])})
    target = OUT/"registered-query-catalog.csv"
    assert not target.exists(), "Never overwrite a registered catalog"
    pd.DataFrame(rows).to_csv(target, index=False)
    assert digest(target) == json.loads(PROTOCOL.read_text())["query_catalog_sha256"]
    (OUT/"registration-costs.json").write_text(json.dumps({"queries":len(rows),"elapsed_seconds":time.perf_counter()-started},indent=2)+"\n")
    print("Reconstructed all", len(rows), "registered masks; frozen hash matches.")


if __name__ == "__main__":
    main()
