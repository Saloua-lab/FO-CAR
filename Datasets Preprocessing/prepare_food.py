"""
prepare_food.py

Cleans the Food (AIST) raw dataset into a CSV ready for FO-CAR Java code.

INPUT:  FoodData.csv (raw with rich attributes, ratings 1-5)
OUTPUT: Food_clean.csv (5 columns: user, item, rating, situation, virtualflag)

Context encoding:
  Situation:   hungry=1, normal=2, full=3
  VirtualFlag: 0=real situation -> 1, 1=virtual situation -> 2

Dataset stats:
  6360 ratings, 212 users, 20 items, 2 ctx dims, ~6 conditions
  Each user rates 30 items: 5 items x 2 situations (real) + 5 items x 4 situations (virtual)
"""
import csv

INPUT_FILE  = "../data/raw/FoodData.csv"
OUTPUT_FILE = "../data/processed/Food_clean.csv"

SITUATION_MAP = {'hungry':1, 'normal':2, 'full':3}

def main():
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Columns: AnswerID-MenuID-situation.key, AnswerID, MenuID, VirtualFlag, situation, ..., rating, ...
        for r in reader:
            try:
                uid = int(r[1])  # AnswerID
                iid = int(r[2])  # MenuID
                vflag = int(r[3])  # 0 or 1
                situation = r[4].strip()  # hungry/normal/full
                rating = int(r[10])  # rating column
                
                sit_code = SITUATION_MAP.get(situation, -1)
                vf_code = vflag + 1  # 0->1, 1->2
                
                if rating < 1 or rating > 5: continue
                if sit_code < 0: continue
                
                rows.append([uid, iid, rating, sit_code, vf_code])
            except: continue
    
    print(f"Cleaned rows: {len(rows)}")
    print(f"Users: {len(set(r[0] for r in rows))}")
    print(f"Items: {len(set(r[1] for r in rows))}")
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write('user,item,rating,situation,virtualflag\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
