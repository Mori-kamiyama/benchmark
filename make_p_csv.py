import os
import math
import json
import csv

# 最適解の推定値 (ユーザー提供の計算式に基づく)
# ユーザーからのフィードバックにより、最適手数の推定には以下の計算式を使用します:
# y = 1/8 * n^3 - n^2 + 7*n - 14  (ここで n は盤面のサイズ)
# 計算結果が1未満になる場合は、最小手数として1を返します。
# この計算式は、以前使用されていた 3/8 * n^2 とは異なる経験則または分析に基づくものです。
def optimal_moves(n):
    """盤面サイズnに基づいて最適手数を推定します (ユーザー提供の計算式)。"""
    calculated_value = (1/8 * n**3) - (n**2) + (7*n) - 14
    return max(1, math.floor(calculated_value))

# 各ペアのマンハッタン距離を計算し、その総和を返す
def manhattan_dis(data):
    """
    盤面データから、同じ値を持つすべての牌のペア間のマンハッタン距離の総和を計算します。
    例えば、値「5」の牌が3つある場合、(5a, 5b), (5a, 5c), (5b, 5c) の3ペアの距離を計算し総和に加えます。
    """
    table = data["problem"]["field"]["entities"]
    size = len(table)
    positions = {} # 各値の牌が存在する位置(行,列)を格納する辞書

    # 盤面をスキャンして、各値 (val) の牌がどの位置 (i, j) にあるかを記録
    for i in range(size):
        for j in range(size):
            val = table[i][j]
            if val not in positions:
                positions[val] = []
            positions[val].append((i, j))

    total_distance = 0
    # 各値ごとに、その値を持つ牌のペアのマンハッタン距離を計算
    for pos_list in positions.values(): # 例: 値'5'の牌の位置リスト [(x1,y1), (x2,y2), (x3,y3)]
        # 同じ値を持つ牌が複数ある場合のみ距離計算を行う
        for i in range(len(pos_list)):
            for j in range(i + 1, len(pos_list)): # 重複しないようにペアを選択 (例: (A,B)は計算するが(B,A)はしない)
                x1, y1 = pos_list[i]
                x2, y2 = pos_list[j]
                total_distance += abs(x1 - x2) + abs(y1 - y2) # マンハッタン距離を加算
    return total_distance

# 各ペアのマンハッタン距離の合計 D から、理論的な最小手数を推定します。
# P: 総ペア数 (size * size / 2)。各ペアを最終的に隣接させるには最低1手は必要と仮定。
# (D - P) / 2: Dには各ペア間の距離が最短経路で含まれる。
# Pを引くのは、各ペアが最終的に同じ位置に来る「連結コスト」のようなもの。
# 2で割るのは、移動が2つの牌に影響する、または距離の重複カウントを補正する意図か (要確認、経験則の可能性あり)。
# ※この計算式は経験則や特定の仮定に基づく可能性があります。
def man_min(D, size):
    """マンハッタン距離の総和(D)と盤面サイズ(size)から最小推定手数を計算します。"""
    P = size ** 2 / 2  # 盤面上の総ペア数
    # この式は、経験則や特定のヒューリスティックに基づいている可能性があります。
    # DからPを引いた値を2で割ることで、最小手数を見積もっています。
    return math.floor((D - P) / 2)

# 毎回マンハッタン距離を1ずつ縮めれば必ず揃うため最大はD
def man_max(D):
    """マンハッタン距離の総和(D)は、あり得る最大手数の一つと考えられます。"""
    return D

PROBLEM_DIR = "./problems"
OUTPUT_FILE = "problems_summary.csv"

rows = [] # CSVに出力するための行データを格納するリスト

# 問題ディレクトリ内のサブディレクトリ（例: "4x4", "5x5"）を取得
folders = [
    d for d in os.listdir(PROBLEM_DIR)
    if os.path.isdir(os.path.join(PROBLEM_DIR, d)) and 'x' in d # 'x' を含むディレクトリのみ対象
]
# フォルダ名をサイズ基準でソート (例: "4x4", "5x5", ..., "10x10", ...)
folders.sort(key=lambda x: int(x.split("x")[0]))

for folder in folders:
    folder_path = os.path.join(PROBLEM_DIR, folder)

    # 各フォルダ内のJSONファイル名を取得し、問題番号基準でソート (例: "p0.json", "p1.json", ...)
    # "p" を取り除いて数値化することでソートキーとする
    problem_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".json")],
        key=lambda x: int(x.replace(".json", "").lstrip("p"))
    )

    for fname in problem_files:
        fpath = os.path.join(folder_path, fname)
        try:
            # スカスカ JSON を除外
            if os.path.getsize(fpath) == 0:
                raise ValueError("empty file")
            # 正常なら読み込む
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  Skipping {fname}: {e}")
            continue
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
