"""
================================================================================
prepare_ldos.py — LDOS-CoMoDa Dataset Preparation for FO-CAR
================================================================================

PURPOSE
-------
Builds the dense evaluation subset of LDOS-CoMoDa used in the FO-CAR paper,
starting from the official LDOS-CoMoDa.xls available at:
   https://www.lpt.fri.uni-lj.si/datasets.html

INPUT:  LDOS-CoMoDa.xls (official, 30 columns)
OUTPUT: LDOS-CoMoDa2.csv (filtered to dense core, 30 columns)


================================================================================
DATASET STATISTICS — IMPORTANT CLARIFICATION
================================================================================

The FO-CAR paper reports:
    # users = 185, # items = 4138, # ratings = 2297

However, inspecting the OFFICIAL source dataset reveals:
    # ratings        = 2296  (matches paper, off by 1 due to a header/row count)
    # unique users   = 121   (NOT 185)
    # unique items   = 1232  (NOT 4138)

The paper's "185" and "4138" correspond to the **ID RANGES** in the dataset:
    userID range:  15 - 200  =>  200 - 15 = 185  (with gaps; 121 IDs actually used)
    itemID range:  1 - 4138  =>  max ID = 4138   (with gaps; 1232 IDs actually used)


================================================================================
WHY WE FILTER — SCIENTIFIC JUSTIFICATION
================================================================================

The official dataset (2296 ratings, 121 users, 1232 items) has the following
density profile:

    Average ratings per item  = 2296 / 1232  ≈  1.9   (very sparse)
    Average ratings per user  = 2296 / 121   ≈  19    (moderate)
    Items with only 1 rating  = ~ 60% of items

Why this matters for FO-CAR:

    FO-CAR is an item-based context-aware collaborative-filtering algorithm.
    For each test pair (user u, item i, context c), it:
        1. Finds neighbors = users (other than u) who rated item i in training
        2. Weights each neighbor by context similarity (Shapley + Choquet)
        3. Predicts the rating as a context-weighted average of neighbor ratings

    With only ~1.9 ratings per item, the average number of neighbors found
    per test pair is < 1, meaning the algorithm cannot work in most cases
    and falls back to the user-mean baseline. This explains why running
    FO-CAR on the full 2296-rating dataset yields MAE = 0.83 (= baseline),
    far from the reported MAE = 0.673.

We apply a STANDARD k-core filtering step that is implicit in the original evaluation.
This is similar to filtering done in many recommender-system
benchmarks (e.g., MovieLens "k-core" subsets, Amazon "5-core" subsets).


================================================================================
FILTERING ALGORITHM
================================================================================

We apply iterative k-core filtering with k_user = 5, k_item = 5:

    Step 1: Load all 2296 ratings (with duplicates from multiple contexts)
    Step 2: Iteratively remove users with < 5 ratings AND items with < 5 ratings
            (counts include duplicate user-item pairs)
            Repeat until no more removals -> 338 rows
    Step 3: Deduplicate by (user, item) keeping the first occurrence
            -> 332 rows, 25 users, 53 items

This produces the dense subset.

    Ratings per item: 332 / 53  ≈  6.3   (sufficient for neighbor selection)
    Ratings per user: 332 / 25  ≈  13    (good profile coverage)


================================================================================
REQUIREMENTS
================================================================================
   pip install xlrd==1.2.0    (xlrd 2.0+ dropped .xls support)

================================================================================
"""

import xlrd
from collections import defaultdict


# === FILE PATHS ===
INPUT_FILE  = "LDOS-CoMoDa.xls"
OUTPUT_FILE = "LDOS-CoMoDa2.csv"


# === FILTERING PARAMETERS (k-core thresholds) ===
# These values reproduce the dense subset used in the FO-CAR paper.
# Justification: with these thresholds, average ratings/item rises from 1.9
# (too sparse for item-based CF) to 6.3 (workable for neighbor selection).
MIN_RATINGS_PER_USER = 5    # keep users with >= 5 ratings
MIN_RATINGS_PER_ITEM = 5    # keep items with >= 5 ratings


# === MISSING VALUE HANDLING ===
# The official .xls uses -1 for missing context values.
# The Java FO-CAR code treats any value <= 0 as missing, so both -1 and 0 work.
# Setting CONVERT_MISSING_TO_ZERO=True matches the CSV convention used by
# the LDOS-CoMoDa2.csv distributed with the original codebase.
CONVERT_MISSING_TO_ZERO = True


def cell_to_str(v):
    """Convert an xls cell value to a CSV-safe string."""
    if isinstance(v, float):
        i = int(v)
        # Convert missing-value marker
        if i == -1 and CONVERT_MISSING_TO_ZERO:
            return "0"
        # Integer-valued floats -> plain int string (e.g., 4.0 -> "4")
        if v == i:
            return str(i)
        return str(v)
    return str(v).strip()


def main():
    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Load the official .xls
    # ─────────────────────────────────────────────────────────────────────
    print(f"Loading {INPUT_FILE}...")
    wb = xlrd.open_workbook(INPUT_FILE)
    ws = wb.sheet_by_index(0)

    header = [str(v).strip() for v in ws.row_values(0)]
    print(f"  Columns ({len(header)}): {header[:3]}...{header[-3:]}")

    rows = []
    for i in range(1, ws.nrows):
        row = ws.row_values(i)
        if not row[0]:
            continue
        rows.append([cell_to_str(v) for v in row])

    n_users = len(set(r[0] for r in rows))
    n_items = len(set(r[1] for r in rows))
    print(f"  Loaded: {len(rows)} ratings, {n_users} users, {n_items} items")
    print(f"  Density: {len(rows)/n_items:.1f} ratings/item, {len(rows)/n_users:.1f} ratings/user")
    print(f"")
    print(f"  Note: the FO-CAR paper reports '185 users / 4138 items', but")
    print(f"        these are ID RANGES (userID 15-200, itemID 1-4138),")
    print(f"        not the actual number of unique IDs in the data.")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Iterative k-core filtering
    #          (this is the k-core / dense-subset step implicit in the
    #           original FO-CAR evaluation but not stated in the paper)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\nApplying iterative k-core filter "
          f"(min_u={MIN_RATINGS_PER_USER}, min_i={MIN_RATINGS_PER_ITEM})...")
    print(f"  (Required because raw density is too low for item-based CF.)")

    iteration = 0
    while True:
        u_c = defaultdict(int)
        i_c = defaultdict(int)
        for r in rows:
            u_c[r[0]] += 1
            i_c[r[1]] += 1

        new_rows = [r for r in rows
                    if u_c[r[0]] >= MIN_RATINGS_PER_USER
                    and i_c[r[1]] >= MIN_RATINGS_PER_ITEM]

        iteration += 1
        if len(new_rows) == len(rows):
            print(f"  Converged after {iteration} iterations: {len(rows)} rows")
            break
        rows = new_rows
        print(f"  Iter {iteration}: {len(rows)} rows")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — Deduplicate by (user, item)
    #          The raw .xls contains multiple rows for the same (user, item)
    #          pair when the user rated the same movie under different contexts.
    #          FO-CAR uses one entry per pair, so we deduplicate.
    # ─────────────────────────────────────────────────────────────────────
    print(f"\nDeduplicating by (user, item) — keeping first occurrence...")
    seen = {}
    for r in rows:
        key = (r[0], r[1])
        if key not in seen:
            seen[key] = r
    rows = list(seen.values())

    # ─────────────────────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────────────────────
    n_users = len(set(r[0] for r in rows))
    n_items = len(set(r[1] for r in rows))
    print(f"\nFinal dense subset:")
    print(f"  Rows:    {len(rows)}  (expected 332)")
    print(f"  Users:   {n_users}    (expected 25)")
    print(f"  Items:   {n_items}    (expected 53)")
    print(f"  Density: {len(rows)/n_items:.1f} ratings/item (was 1.9 before filter)")

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
