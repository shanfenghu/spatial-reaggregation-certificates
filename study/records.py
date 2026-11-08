"""Extract the exact pinned agency DBF/workbook versions, preserving blanks."""
import os
from pathlib import Path
import csv
import hashlib
import json
import platform
import struct
import time
import zipfile

import openpyxl

BASE = Path(os.environ.get("ALLOCATION_INPUTS", Path(__file__).resolve().parents[1] / "data/cache"))


def census_records():
    """Decode the original DBF and verify IDs/points against record bounds."""
    with zipfile.ZipFile(BASE / "tl_2020_44_tabblock20.zip") as archive:
        assert archive.testzip() is None
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts
                   for n in archive.namelist())
        raw = archive.read("tl_2020_44_tabblock20.dbf")
        count, header, length = struct.unpack_from("<IHH", raw, 4)
        fields = []
        for offset in range(32, header - 1, 32):
            if raw[offset] == 13:
                break
            spec = raw[offset:offset + 32]
            name, kind, width = spec[:11].split(b"\0")[0].decode(), chr(spec[11]), spec[16]
            assert kind in ["C", "N"]
            fields.append((name, kind, width))
        assert 1 + sum(w for _, _, w in fields) == length
        assert len(raw) in [header + count * length, header + count * length + 1]
        records = []
        for k in range(count):
            start = header + k * length
            assert raw[start:start + 1] == b" "
            row, offset = {}, start + 1
            for name, kind, width in fields:
                value = raw[offset:offset + width].decode("ascii").strip()
                row[name] = int(value) if kind == "N" and value else value
                offset += width
            records.append(row)
        with archive.open("tl_2020_44_tabblock20.shp") as shape:
            head = shape.read(100)
            assert struct.unpack_from(">i", head, 0)[0] == 9994
            assert struct.unpack_from("<i", head, 32)[0] == 5
            failures = []
            for k, row in enumerate(records, 1):
                index, words = struct.unpack(">ii", shape.read(8))
                content = shape.read(words * 2)
                assert index == k and struct.unpack_from("<i", content)[0] == 5
                xmin, ymin, xmax, ymax = struct.unpack_from("<dddd", content, 4)
                lat, lon = float(row["INTPTLAT20"]), float(row["INTPTLON20"])
                if not (xmin - 1e-7 <= lon <= xmax + 1e-7 and ymin - 1e-7 <= lat <= ymax + 1e-7):
                    failures.append(row["GEOID20"])
            assert shape.read() == b""
        with (BASE / "census-block-records.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        return {"records": count, "fields": [f[0] for f in fields],
                "same_record_point_bbox_failures": failures,
                "crs_wkt": archive.read("tl_2020_44_tabblock20.prj").decode(),
                "support_limit": "Bounding-box check only; no full polygon containment claim."}


def plant_records():
    """Use field-code header row, preserving blanks separately from zero."""
    workbook = openpyxl.load_workbook(BASE / "egrid2023_data_rev2.xlsx", read_only=True, data_only=True)
    output = {}
    for name, columns, file in [
        ("PLNT23", ["YEAR", "ORISPL", "PSTATABB", "FIPSST", "FIPSCNTY", "CNTYNAME",
                    "LAT", "LON", "PLFUELCT", "NAMEPCAP", "PLNOXAN", "PLNGENAN"], "plant-records.csv"),
        ("ST23", ["YEAR", "PSTATABB", "STNOXAN"], "state-inventory-records.csv")]:
        iterator = workbook[name].iter_rows(values_only=True)
        next(iterator)
        header = next(iterator)
        assert all(c in header for c in columns)
        indices = [header.index(c) for c in columns]
        rows = [dict(zip(columns, [r[i] for i in indices])) for r in iterator if r[indices[0]] is not None]
        with (BASE / file).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        output[name] = {"rows": len(rows), "columns": columns,
                        "missing": {c: sum(r[c] is None for r in rows) for c in columns}}
    workbook.close()
    return output


def main():
    started = time.perf_counter()
    manifest = json.loads((BASE / "acquisition.json").read_text())
    for record in manifest["files"]:
        name = record.get("name", "county_adjacency2023.txt")
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == record["sha256"]
    result = {"census": census_records(), "egrid": plant_records(),
              "python": platform.python_version(), "openpyxl": openpyxl.__version__,
              "extraction_seconds": time.perf_counter() - started}
    for name in ["census-block-records.csv", "plant-records.csv", "state-inventory-records.csv"]:
        result.setdefault("derived_sha256", {})[name] = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
    (BASE / "schema-extraction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"census_records": result["census"]["records"],
                      "census_bbox_failures": len(result["census"]["same_record_point_bbox_failures"]),
                      "plant_records": result["egrid"]["PLNT23"]["rows"],
                      "extraction_seconds": result["extraction_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
