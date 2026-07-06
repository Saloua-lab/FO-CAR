"""
prepare_incarmusic.py

Builds InCarMusic_clean.csv from the original Data_InCarMusic.xlsx
(InCarMusic dataset, Baltrunas et al.), sheet "ContextualRating".

INPUT:  Data_InCarMusic.xlsx, sheet "ContextualRating"
        columns: UserID, ItemID, Rating, DrivingStyle, landscape, mood,
                 naturalphenomena, RoadType, sleepiness, trafficConditions,
                 weather -- each context column is either "NA" or a free-text
                 category value (e.g. "sunny", "traffic jam").
OUTPUT: InCarMusic_clean.csv (comma-separated, same row order as source)

ENCODING SCHEME (reverse-engineered against the existing clean file):
  For each of the 8 context columns independently: collect the distinct
  non-NA values that appear anywhere in the column, sort them
  ALPHABETICALLY, and assign integer codes 1..N in that sorted order.
  "NA" maps to -1 (missing/not-applicable, consistent with every other
  dataset in this project). This matches the dataset's own "Context Factor"
  reference sheet, whose ids increase in alphabetical order within each
  dimension (e.g. weather: cloudy=1, rainy=2, snowing=3, sunny=4).
"""

import openpyxl

INPUT_FILE = "Data_InCarMusic.xlsx"
SHEET_NAME = "ContextualRating"
OUTPUT_FILE = "InCarMusic_clean_rebuilt.csv"

CTX_COLS = ["DrivingStyle", "landscape", "mood", "naturalphenomena",
            "RoadType", "sleepiness", "trafficConditions", "weather"]

def main():
    wb = openpyxl.load_workbook(INPUT_FILE, data_only=True)
    ws = wb[SHEET_NAME]

    rows_raw = list(ws.iter_rows(min_row=1, values_only=True))
    header = [str(h).strip() for h in rows_raw[0]]
    data = rows_raw[1:]
    print(f"Loaded {len(data)} rows, columns: {header}")

    col_idx = {name: header.index(name.strip()) for name in
               ["UserID", "ItemID", "Rating"] + [c.strip() for c in
               ["DrivingStyle", "landscape", "mood", "naturalphenomena",
                "RoadType", "sleepiness", "trafficConditions", "weather"]]}

    # Collect distinct non-NA values per context column, across ALL rows.
    distinct = {c: set() for c in CTX_COLS}
    for r in data:
        for c in CTX_COLS:
            v = r[col_idx[c.strip()]]
            v = str(v).strip() if v is not None else "NA"
            if v.upper() != "NA" and v != "":
                distinct[c].add(v)

    # Alphabetical order -> code 1..N per column, EXCEPT naturalphenomena,
    # which empirically follows morning/day time/afternoon/night (not
    # alphabetical) in the real InCarMusic_clean.csv -- verified directly
    # against known rows rather than assumed.
    EXPLICIT_ORDER = {
        "naturalphenomena": ["morning", "day time", "afternoon", "night"],
    }
    code_map = {}
    for c in CTX_COLS:
        if c in EXPLICIT_ORDER:
            ordered = EXPLICIT_ORDER[c]
            assert set(ordered) == distinct[c], f"{c}: {ordered} vs {distinct[c]}"
        else:
            ordered = sorted(distinct[c])
        code_map[c] = {v: i + 1 for i, v in enumerate(ordered)}
        print(f"  {c}: {code_map[c]}")

    out_rows = []
    for r in data:
        uid = int(r[col_idx["UserID"]])
        iid = int(r[col_idx["ItemID"]])
        rating = int(r[col_idx["Rating"]])
        ctx_vals = []
        for c in CTX_COLS:
            v = r[col_idx[c.strip()]]
            v = str(v).strip() if v is not None else "NA"
            if v.upper() == "NA" or v == "":
                ctx_vals.append(-1)
            else:
                ctx_vals.append(code_map[c][v])
        out_rows.append([uid, iid, rating] + ctx_vals)

    out_header = ["userID", "itemID", "rating"] + CTX_COLS
    with open(OUTPUT_FILE, "w") as f:
        f.write(",".join(out_header) + "\n")
        for row in out_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Saved {len(out_rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
