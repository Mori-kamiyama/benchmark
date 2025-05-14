import os
import math
import json
import csv

# 最適解の推定値 
# 4x4での最適解をBFSで探索した結果最適解は5~7手程度であるとわかった
# これをもとに、今回の課題は盤面サイズに対して二次関数的に手数が増加するため
# 5 ≤ Mopt​(4) ≤ 7 ⟹ 5/16​n^2 ≤ Mopt​(n) ≤ 7/16​n^2 (4 のとき n^2 =16)となり
# 推定値は3/8n^2程度が推定値である
def optimal_moves(n):
    return math.floor(3 / 8 * n ** 2)

# 各ペアのマンハッタン距離を計算し、その総和を返す
def manhattan_dis(data):
    table = data["problem"]["field"]["entities"]
    size = len(table)
    positions = {}

    for i in range(size):
        for j in range(size):
            val = table[i][j]
            if val == 0:
                continue
            if val not in positions:
                positions[val] = []
            positions[val].append((i, j))

    total_distance = 0
    for pos_list in positions.values():
        for i in range(len(pos_list)):
            for j in range(i + 1, len(pos_list)):
                x1, y1 = pos_list[i]
                x2, y2 = pos_list[j]
                total_distance += abs(x1 - x2) + abs(y1 - y2)
    return total_distance

# 各ペアの距離に応じて距離を縮めるため最適解は一つのペアを作るのに最低でも1手使うため、
# マンハッタン距離からペア数(完成時には全てが1になるため)を引き二回オーバーにカウントしているため2で割ればいい
def man_min(D, size):
    P = size ** 2 / 2
    return math.floor((D - P) / 2)

# 毎回マンハッタン距離を1ずつ縮めれば必ず揃うため最大はD
def man_max(D):
    return D

PROBLEM_DIR = "./problems"
OUTPUT_FILE = "problems_summary.csv"

rows = []

folders = [
    d for d in os.listdir(PROBLEM_DIR)
    if os.path.isdir(os.path.join(PROBLEM_DIR, d))
]
folders.sort(key=lambda x: int(x.split("x")[0]))  # e.g. "10x10" → 10
for folder in folders:
    folder_path = os.path.join(PROBLEM_DIR, folder)

    for fname in sorted(
        os.listdir(folder_path),
        key=lambda x: int(x.replace(".json", "").lstrip("p"))
    ):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
            size = data["problem"]["field"]["size"]
            problem_id = int(fname.replace(".json", "").lstrip("p"))
            D = manhattan_dis(data)
            rows.append({
                "problem_id": problem_id,
                "size": size,
                "optimal_moves": optimal_moves(size),
                "n_squared": size * size,
                "manhattan_min": man_min(D, size),
                "manhattan_max": man_max(D)
            })

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Saved {len(rows)} problems to {OUTPUT_FILE}")