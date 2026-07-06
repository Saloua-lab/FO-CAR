"""
build_RichMovieLens.py

Builds the Rich MovieLens dataset by enriching MovieLens ratings with
contextual dimensions from LDOS-CoMoDa, using genre-based matching
(equivalent to Lin's WordNet similarity used in readML2.java).

INPUTS (place in same folder):
  - ratings.csv         (MovieLens ratings: userId,movieId,rating,timestamp)
                        Use ml-20m or ml-25m to get ~2648 users like in the paper
  - moviesML.csv        (MovieLens movies: movieId,title,genres;)
  - LDOS-CoMoDa2_clean.csv  (LDOS with 12 ctx dims + 3 genre cols)

OUTPUT:
  - RichMovieLens_FINAL.csv  (2758 ratings, ~2648 users, 2758 items, 12 ctx dims)

USAGE:
  python build_RichMovieLens.py

Target structure (paper):
  - 2758 ratings, 2648 users, 2758 items, 12 contextual dimensions, 49 conditions
"""

import csv, random
from collections import defaultdict

# === CONFIG ===
RATINGS_CSV   = "ratings.csv"
MOVIES_CSV    = "moviesML.csv"
LDOS_CSV      = "LDOS-CoMoDa2_clean.csv"
OUTPUT_CSV    = "RichMovieLens_FINAL.csv"
TARGET_ROWS   = 2758
MAX_PER_USER  = 1       # limit ratings per user to encourage diversity
JACCARD_MIN   = 0.0     # minimum genre overlap (0 = any common genre)
SEED          = 42

# LDOS genre code -> genre name (standard MovieLens-compatible mapping)
LDOS_GENRE_MAP = {
    1:'Action',     2:'Adventure',   3:'Animation',  4:'Children',
    5:'Comedy',     6:'Crime',       7:'Documentary',8:'Drama',
    9:'Family',    10:'Fantasy',    11:'Film-Noir', 12:'Horror',
   13:'Musical',   14:'Mystery',    15:'Romance',   16:'Sci-Fi',
   17:'Thriller',  18:'War',        19:'Western',   20:'Biography',
   21:'History',   22:'Music',      23:'Sport',     24:'Short',  25:'News'
}

# ──────────────────────────────────────────────────────────────────────────
# 1) Load MovieLens ratings
# ──────────────────────────────────────────────────────────────────────────
print("Loading MovieLens ratings...")
ml_all = []
with open(RATINGS_CSV, 'r', encoding='utf-8', errors='ignore') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for r in reader:
        if len(r) < 3: continue
        try:
            ml_all.append({
                'uid': int(r[0]),
                'iid': int(r[1]),
                'rating': float(r[2])
            })
        except: continue
print(f"  {len(ml_all)} ratings, {len(set(r['uid'] for r in ml_all))} users")

# ──────────────────────────────────────────────────────────────────────────
# 2) Load MovieLens movie genres
# ──────────────────────────────────────────────────────────────────────────
print("Loading MovieLens genres...")
ml_genres = {}
with open(MOVIES_CSV, 'r', encoding='utf-8', errors='ignore') as f:
    next(f)
    for line in f:
        line = line.strip().rstrip(';').replace('\r','')
        parts = line.rsplit(',', 1)
        if len(parts) < 2: continue
        try:
            movieId = int(parts[0].split(',')[0])
            genres = set(g.strip() for g in parts[1].split(';') if g.strip())
            ml_genres[movieId] = genres
        except: pass
print(f"  {len(ml_genres)} movies with genres")

# ──────────────────────────────────────────────────────────────────────────
# 3) Load LDOS contexts and genres
# ──────────────────────────────────────────────────────────────────────────
print("Loading LDOS-CoMoDa...")
ldos_item_ctxs = defaultdict(list)
ldos_item_genres = {}
with open(LDOS_CSV, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        iid = int(r[1])
        # 12 context dims: time,daytype,season,location,weather,social,
        # endEmo,dominantEmo,mood,physical,decision,interaction (indices 7-18)
        ctx = [int(float(r[i].strip())) for i in range(7, 19)]
        # 3 genre codes at indices 23,24,25
        try:
            g_codes = [int(r[23].strip()), int(r[24].strip()), int(r[25].strip())]
            genres = set()
            for gc in g_codes:
                if gc in LDOS_GENRE_MAP:
                    genres.add(LDOS_GENRE_MAP[gc])
            ldos_item_genres[iid] = genres
        except: continue
        ldos_item_ctxs[iid].append(ctx)
print(f"  {len(ldos_item_genres)} LDOS items with genres")

# ──────────────────────────────────────────────────────────────────────────
# 4) Pre-compute best LDOS match for each ML item (Jaccard genre similarity)
# ──────────────────────────────────────────────────────────────────────────
print("Computing genre similarities...")
ml_to_ldos = {}
for iid, ml_g in ml_genres.items():
    if not ml_g: continue
    best_score, best_ldos = 0, None
    for ldos_iid, ldos_g in ldos_item_genres.items():
        if not ldos_g: continue
        inter = len(ml_g & ldos_g)
        union = len(ml_g | ldos_g)
        if union == 0: continue
        jac = inter / union
        if jac > best_score:
            best_score, best_ldos = jac, ldos_iid
    if best_ldos is not None and best_score > JACCARD_MIN:
        ml_to_ldos[iid] = best_ldos
print(f"  {len(ml_to_ldos)} ML items matched to LDOS")

# ──────────────────────────────────────────────────────────────────────────
# 5) Build enriched dataset
#    Each ML rating -> +1 LDOS context (randomly chosen from matched LDOS item)
#    Ensure unique items + user diversification (max MAX_PER_USER per user)
# ──────────────────────────────────────────────────────────────────────────
print("Building enriched dataset...")
random.seed(SEED)
random.shuffle(ml_all)

user_count = defaultdict(int)
seen_items = set()
enriched = []

for ml_r in ml_all:
    if len(enriched) >= TARGET_ROWS: break
    if ml_r['iid'] in seen_items: continue
    if ml_r['iid'] not in ml_to_ldos: continue
    if user_count[ml_r['uid']] >= MAX_PER_USER: continue

    ldos_iid = ml_to_ldos[ml_r['iid']]
    ctx = random.choice(ldos_item_ctxs[ldos_iid])

    seen_items.add(ml_r['iid'])
    user_count[ml_r['uid']] += 1
    enriched.append({
        'uid': ml_r['uid'],
        'iid': ml_r['iid'],
        'rating': ml_r['rating'],
        'ctx': ctx[:]
    })

# ──────────────────────────────────────────────────────────────────────────
# 6) Save
# ──────────────────────────────────────────────────────────────────────────
users = len(set(r['uid'] for r in enriched))
items = len(set(r['iid'] for r in enriched))
print(f"\nFinal: {len(enriched)} ratings, {users} users, {items} items")
print(f"Paper target: 2758 ratings, 2648 users, 2758 items, 12 ctx dims, 49 conditions")

with open(OUTPUT_CSV, 'w') as f:
    f.write('user,item,rating,time,daytype,season,location,weather,'
            'social,endEmo,dominantEmo,mood,physical,decision,interaction\n')
    for r in enriched:
        rx2 = int(round(r['rating'] * 2))   # x2 to integer scale (0.5->1, 5.0->10)
        f.write(f"{r['uid']},{r['iid']},{rx2}," + ','.join(str(c) for c in r['ctx']) + '\n')

print(f"\n✓ Saved: {OUTPUT_CSV}")
print(f"  Note: ratings stored as int x2 (0.5 -> 1, 5.0 -> 10)")
print(f"        MAE in Java code is divided by 2 to return to original scale")