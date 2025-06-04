import os
import json
import random
from hashlib import sha256

FIXED_STARTS_AT = 1700000000  # 固定のUnixタイムスタンプ。問題セットの再現性を高めるために使用 (他の要因が同じ場合)
MAX_TRIES = 1000              # 無限ループ回避用の制限

def generate_entities(size):
    """
    盤面エンティティのリストを生成します。
    各エンティティ（牌）の値は整数で、盤面上でペアになるように設定されます。
    盤面のセル数が奇数の場合、最後の1つのエンティティはペアになりません。
    """
    total_cells = size * size
    num_pairs = total_cells // 2

    values = []
    # ペアとなる値を生成 (例: 0,0, 1,1, 2,2, ...)
    for v_idx in range(num_pairs):
        values.extend([v_idx, v_idx])

    # セル数が奇数の場合、最後の1つのエンティティを追加 (例: ..., N,N, N+1)
    if total_cells % 2 == 1:
        values.append(num_pairs) # ペアの次の値を使用

    random.shuffle(values) # 値をシャッフルしてランダムな盤面にする

    # 1次元リストを2次元リスト（盤面）に変換
    return [values[i * size:(i + 1) * size] for i in range(size)]

def hash_entities(entities):
    """
    盤面エンティティのハッシュ値を計算します。
    JSONにシリアライズする際にキーをソートすることで、
    同じ盤面であれば（行や列の内部的な順序が異なっていても）
    常に同じハッシュ値が得られるようにし、重複検出の安定性を高めます。
    """
    # Use JSON serialization to handle any integer values and ensure canonical representation
    return sha256(json.dumps(entities, sort_keys=True).encode('utf-8')).hexdigest()

def generate_unique_problem(size, existing_hashes):
    """重複を避けて一意の問題を生成"""
    for _ in range(MAX_TRIES):
        entities = generate_entities(size)
        h = hash_entities(entities)
        if h not in existing_hashes:
            existing_hashes.add(h)
            return {
                "startsAt": FIXED_STARTS_AT,
                "problem": {
                    "field": {
                        "size": size,
                        "entities": entities
                    }
                }
            }
    raise RuntimeError(f"❌ {size}x{size} で一意な問題が生成できませんでした（{MAX_TRIES}回試行）")

def save_problems(base_dir="./problems", sizes=range(4, 25, 2), per_size=100):
    existing_hashes = set()

    for size in sizes:
        folder_name = os.path.join(base_dir, f"{size}x{size}")
        os.makedirs(folder_name, exist_ok=True)

        for idx in range(per_size):
            problem = generate_unique_problem(size, existing_hashes)

            file_name = f"p{idx:03}.json"   # p000 – p099 など
            file_path = os.path.join(folder_name, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(problem, f, indent=2)

            print(f"✅ Generated: {file_path}")

if __name__ == "__main__":
    save_problems(
        base_dir="problems",
        sizes=range(4, 25, 2),  # 4x4～24x24
        per_size=100           # 各サイズ100個
    )
