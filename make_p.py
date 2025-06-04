import os
import json
import random
from hashlib import sha256

FIXED_STARTS_AT = 1700000000  # 固定のUnixタイムスタンプ
MAX_TRIES = 1000              # 無限ループ回避用の制限

def generate_entities(size):
    """
    盤面を生成:
    - 0〜(pairs-1) の数字を2つずつ用意（ただし最大255）
    - 余りセルが1つある場合は最後に追加で1つ番号を割り当てる
    """
    total = size * size
    pairs = total // 2


    values = []
    for v in range(pairs):
        values.extend([v, v])
    if total % 2 == 1:
        values.append(pairs)

    random.shuffle(values)
    return [values[i * size:(i + 1) * size] for i in range(size)]

def hash_entities(entities):
    """盤面をハッシュ化して重複検出用キーを作成"""
    # Use JSON serialization to handle any integer values
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
