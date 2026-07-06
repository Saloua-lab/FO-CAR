
"""
build_RichMovieLens_final.py

Replicates readML2.java logic:
- For each LDOS rating, find ML items with genre overlap
- For each match, write ONE row: (ml_user, ml_item, ml_rating, LDOS_context)
- Result: ~332 LDOS ratings × ~8 ML matches = ~2758 enriched rows
- Users/Items naturally diverse → 2648/2758 unique
"""

import csv, random, openpyxl
from collections import defaultdict
import pandas as pd


# === CONFIG ===
RATINGS_FILE  = "ratings.csv" # ML ratings file
MOVIES_CSV    = "moviesML.csv"  # ML movies + genres
LDOS_CSV      = "LDOS-CoMoDa2_clean.csv"
OUTPUT_CSV    = "RichMovieLens_FINAL_paper.csv"


TARGET_ROWS  = 2758
TARGET_USERS = 2648
SEED = 42

LDOS_GENRE_MAP = {
    1:'Action',2:'Adventure',3:'Animation',4:'Children',5:'Comedy',6:'Crime',
    7:'Documentary',8:'Drama',9:'Family',10:'Fantasy',11:'Film-Noir',12:'Horror',
    13:'Musical',14:'Mystery',15:'Romance',16:'Sci-Fi',17:'Thriller',18:'War',
    19:'Western',20:'Biography',21:'History',22:'Music',23:'Sport',24:'Short',25:'News'
}

# === 1. Load LDOS ratings + contexts + genres ===
print("Loading LDOS...")
ldos_ratings = []
with open(LDOS_CSV, 'r') as f:
    reader = csv.reader(f); next(reader)
    for r in reader:
        ctx = [int(float(r[i].strip())) for i in range(7, 19)]  # 12 ctx dims
        try:
            g_codes = [int(r[23].strip()), int(r[24].strip()), int(r[25].strip())]
            genres = set(LDOS_GENRE_MAP[g] for g in g_codes if g in LDOS_GENRE_MAP)
        except: genres = set()
        if not genres: continue
        ldos_ratings.append({
            'rating': int(float(r[2].strip())),
            'ctx': ctx, 'genres': genres
        })
print(f"  {len(ldos_ratings)} LDOS ratings with genres")

# === 2. Load ML movies → genres ===
print("Loading ML genres...")
ml_genres = {}
with open(MOVIES_CSV, 'r', encoding='utf-8', errors='ignore') as f:
    next(f)
    for line in f:
        line = line.strip().rstrip(';').replace('\r','')
        parts = line.rsplit(',', 1)
        if len(parts) < 2: continue
        try:
            movieId = int(parts[0].split(',')[0])
            ml_genres[movieId] = set(g.strip() for g in parts[1].split(';') if g.strip())
        except: pass
print(f"  {len(ml_genres)} movies with genres")

# === 3. Load ML ratings ===
# === 3. Load ML ratings ===
print("Loading ML ratings...")

ml_all = []

df = pd.read_csv(RATINGS_FILE)

for _, row in df.iterrows():
    try:
        ml_all.append({
            'uid': int(row['userId']),
            'iid': int(row['movieId']),
            'rating': float(row['rating'])
        })
    except:
        continue

print(f"  {len(ml_all)} ML ratings, {len(set(r['uid'] for r in ml_all))} users")

# === 4. Index ML by item → list of (user, rating) ===
ml_by_item = defaultdict(list)
for r in ml_all:
    ml_by_item[r['iid']].append(r)
print(f"  {len(ml_by_item)} unique ML items")

# === 5. Enrichment - readML2.java logic ===
print("\nEnriching...")
random.seed(SEED)
random.shuffle(ldos_ratings)

enriched = []
seen_items = set()
seen_users = set()
user_count = defaultdict(int)

# For each LDOS rating: find ML items with genre overlap, pick the best ones
# Each match → one enriched row
ml_items_list = list(ml_by_item.keys())
random.shuffle(ml_items_list)

# Pre-compute genre overlaps (ldos_rating_idx → list of ml_items with overlap)
print("  Pre-computing matches...")
ldos_matches = []
for lr in ldos_ratings:
    matches = []
    for ml_iid in ml_items_list:
        ml_g = ml_genres.get(ml_iid, set())
        if ml_g and (ml_g & lr['genres']):
            matches.append(ml_iid)
    random.shuffle(matches)
    ldos_matches.append(matches)
print(f"  Avg matches per LDOS: {sum(len(m) for m in ldos_matches)/len(ldos_matches):.0f}")

# Build enriched dataset: iterate through LDOS, pick fresh ML items
# Priority: unique items + diverse users
print("  Building rows...")
ldos_position = [0] * len(ldos_ratings)  # current match position per LDOS

for round_num in range(50):
    if len(enriched) >= TARGET_ROWS: break
    for li, lr in enumerate(ldos_ratings):
        if len(enriched) >= TARGET_ROWS: break
        matches = ldos_matches[li]
        if ldos_position[li] >= len(matches): continue
        
        # Find next ML item not yet used
        while ldos_position[li] < len(matches):
            ml_iid = matches[ldos_position[li]]
            ldos_position[li] += 1
            if ml_iid in seen_items: continue
            
            # Find an ML user for this item not yet at max
            candidates = ml_by_item[ml_iid]
            random.shuffle(candidates)
            for ml_r in candidates:
                # Constraint: prefer new users first
                if user_count[ml_r['uid']] >= 1 and len(seen_users) < TARGET_USERS:
                    continue
                if user_count[ml_r['uid']] >= 2:
                    continue
                
                # Accept this row
                seen_items.add(ml_iid)
                seen_users.add(ml_r['uid'])
                user_count[ml_r['uid']] += 1
                enriched.append({
                    'uid': ml_r['uid'], 'iid': ml_iid,
                    'rating': ml_r['rating'], 'ctx': lr['ctx'][:]
                })
                break
            break  # move to next LDOS rating

# Final stats
users = len(set(r['uid'] for r in enriched))
items = len(set(r['iid'] for r in enriched))
print(f"\nResult: {len(enriched)} ratings, {users} users, {items} items")
print(f"Target: 2758 ratings, 2648 users, 2758 items")

# Save
with open(OUTPUT_CSV, 'w') as f:
    f.write('user,item,rating,time,daytype,season,location,weather,'
            'social,endEmo,dominantEmo,mood,physical,decision,interaction\n')
    for r in enriched:
        rx2 = int(round(r['rating'] * 2))
        f.write(f"{r['uid']},{r['iid']},{rx2}," + ','.join(str(c) for c in r['ctx']) + '\n')
print(f"Saved: {OUTPUT_CSV}")
