"""Join Skogsstyrelsen 500 m Sverige-indexruta attributes onto a rasters_to_csv output.

The index grid is aligned to SWEREF99 TM at 500 m spacing, with feature
properties reporting cell *centers* in SWEREF99Ost/Nord. For every (x, y) row
in the CSV we look up the enclosing 500 m cell by its center and attach the
indexruta properties as new columns.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

import pandas as pd

CELL = 500
SERVICE = (
    "https://geodpags.skogsstyrelsen.se/arcgis/rest/services/"
    "Geodataportal/GeodataportalVisaSverigeindexruta/MapServer/0/query"
)
ATTRS = ["BK", "PageName", "Storruta", "CenterLanNamn", "CenterKommunNamn"]


def fetch_features(minx, miny, maxx, maxy):
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "3006",
        "outSR": "3006",
        "outFields": ",".join(["SWEREF99Ost", "SWEREF99Nord", *ATTRS]),
        "f": "geojson",
    }
    url = SERVICE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["features"]


def cell_center(v):
    """Lower-left of the 500 m cell + 250 → center, matching SWEREF99Ost/Nord."""
    return (v // CELL) * CELL + CELL // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in")
    ap.add_argument("csv_out")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    minx, maxx = df.x.min(), df.x.max()
    miny, maxy = df.y.min(), df.y.max()
    print(f"CSV bbox: {minx},{miny} → {maxx},{maxy}  ({len(df):,} rows)")

    feats = fetch_features(minx, miny, maxx, maxy)
    print(f"Fetched {len(feats)} indexruta features")

    lookup = {}
    for f in feats:
        p = f["properties"]
        lookup[(int(p["SWEREF99Ost"]), int(p["SWEREF99Nord"]))] = {k: p.get(k) for k in ATTRS}

    cx = (df.x.astype(int) // CELL) * CELL + CELL // 2
    cy = (df.y.astype(int) // CELL) * CELL + CELL // 2
    keys = list(zip(cx, cy))

    for attr in ATTRS:
        df[attr] = [lookup.get(k, {}).get(attr) for k in keys]

    missing = df[ATTRS[0]].isna().sum()
    print(f"Rows with no matching cell: {missing:,}")

    df.to_csv(args.csv_out, index=False)
    print(f"Wrote {args.csv_out}")


if __name__ == "__main__":
    main()
