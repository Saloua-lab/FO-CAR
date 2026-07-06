"""
prepare_depaulmovie.py

Builds DePaulMovie_clean.csv from the original ratings_DePaulMovie.csv
(DePaul Movie dataset, Zheng et al.).

INPUT:  ratings_DePaulMovie.csv (comma-separated)
        columns: userid, itemid (IMDB tt-code, e.g. "tt1499658"), rating,
                 Time, Location, Companion -- each context column is either
                 "NA" or a free-text category value.
OUTPUT: DePaulMovie_clean.csv (comma-separated, same row order as source)

ENCODING SCHEME (reverse-engineered against the existing clean file):
  - userID: kept as-is (already numeric in the source).
  - itemID: the IMDB tt-code strings are NOT numeric, so they're remapped
    to sequential integers 1, 2, 3, ... in order of FIRST APPEARANCE in the
    file (verified: e.g. tt1499658 -> 1, tt0405422 -> 2, tt0109830 -> 3,
    even though these are not in tt-code sorted order).
  - Context columns: NOT alphabetical (unlike InCarMusic) -- verified
    directly against known rows:
      Time:      Weekday=1, Weekend=2
      Location:  Home=1,    Cinema=2
      Companion: Alone=1,   Family=2, Partner=3
    "NA" maps to -1 (missing/not-applicable).
"""

INPUT_FILE = "ratings_DePaulMovie.csv"
OUTPUT_FILE = "DePaulMovie_clean_rebuilt.csv"

# Verified directly against DePaulMovie_clean.csv -- not alphabetical.
TIME_CODE = {"Weekday": 1, "Weekend": 2}
LOCATION_CODE = {"Home": 1, "Cinema": 2}
COMPANION_CODE = {"Alone": 1, "Family": 2, "Partner": 3}

def main():
    item_ids = {}
    next_item = 1
    out_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            uid = int(parts[0].strip())
            item_str = parts[1].strip()
            rating = int(round(float(parts[2].strip())))

            iid = item_ids.get(item_str)
            if iid is None:
                iid = next_item
                next_item += 1
                item_ids[item_str] = iid

            def code(v, table):
                v = v.strip()
                if v.upper() == "NA" or v == "":
                    return -1
                return table[v]

            t = code(parts[3], TIME_CODE)
            l = code(parts[4], LOCATION_CODE)
            c = code(parts[5], COMPANION_CODE)

            out_rows.append([uid, iid, rating, t, l, c])

    out_header = ["userID", "itemID", "rating", "Time", "Location", "Companion"]
    with open(OUTPUT_FILE, "w") as f:
        f.write(",".join(out_header) + "\n")
        for row in out_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Distinct items: {len(item_ids)}")
    print(f"Saved {len(out_rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
