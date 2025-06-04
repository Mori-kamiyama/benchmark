#!/usr/bin/env python3
"""
A*アルゴリズムベンチマーク結果 Streamlit可視化アプリ

これで実行
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import subprocess
import time
import signal
import psutil

# ページ設定
st.set_page_config(
        page_title="ベンチマーク結果",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
        )


# カスタムCSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
    
    /* --- レイアウト修正の核となる部分 --- */
    height: 100%; /* 親要素(カラム)の高さ一杯に広がる */
    display: flex; /* Flexboxを有効化 */
    flex-direction: column; /* 要素を縦に並べる */
}

.metric-card h3 {
    /* タイトル (例: "解決率") */
    margin-bottom: 0.5rem; /* タイトルと数値の間の余白 */
}

.metric-card h2 {
    /* メインの数値 (例: "83/83") */
    margin-bottom: 0.25rem; /* 数値と説明文の間の余白 */
}

.metric-card p {
    /* 説明文 (例: "100.0%") */
    margin-top: auto; /* ★★★ これが重要: 要素をコンテナの下部に押しやる ★★★ */
    color: #64748b; /* 説明文の文字色を少し薄くする */
    font-size: 0.9rem; /* 説明文のフォントサイズを少し小さくする */
    padding-top: 0.5rem; /* 上の要素との間に少し余白を確保 */
}

/* 各カードの左枠線の色 */
.success-metric {
    border-left-color: #51cf66;
}
.warning-metric {
    border-left-color: #ffd43b;
}
.info-metric {
    border-left-color: #339af0;
}
</style>
""", unsafe_allow_html=True)

def to_serializable(obj):
    """json.dump の default で使う: NumPy 型 → ネイティブ型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # ここに来るのは str や bool など json が扱える型
    return obj

def calculate_metrics(df):
    """評価指標を計算"""
    if df.empty:
        return df

    # 最適解推定式 (ユーザー提供のカスタム式) との差分
    # y = 1/8 * n^3 - n^2 + 7*n - 14
    # 結果が1未満の場合は1とする
    s = df['size'] # For brevity
    calculated_value = (1/8 * s**3) - (s**2) + (7*s) - 14
    df['estimated_optimal'] = np.maximum(1, calculated_value) # np.maximum は Series にも対応
    df['diff_from_estimated'] = df['num_moves'] - df['estimated_optimal']

    # n²との差分
    df['n_squared'] = df['size'] * df['size']
    df['diff_from_n_squared'] = df['num_moves'] - df['n_squared']

    # 探索効率 (solved problems per second based on nodes explored)
    df['search_efficiency'] = df['nodes_explored'] / (df['calculation_time_ms'] / 1000)

    # 総合スコア（参考値）- 小さいほど良い
    # 各指標を0-1の範囲に正規化し、合計する。
    # calculation_time_ms と num_moves は小さいほど良いため、そのまま正規化。
    # search_efficiency は大きいほど良いため、 (1 - 正規化値) を使用してスコアに寄与させる。
    # 1e-10 はゼロ除算を避けるための微小値。
    if len(df) > 1:  # 正規化のために複数のデータが必要
        time_normalized = (df['calculation_time_ms'] - df['calculation_time_ms'].min()) / \
                          (df['calculation_time_ms'].max() - df['calculation_time_ms'].min() + 1e-10)
        moves_normalized = (df['num_moves'] - df['num_moves'].min()) / \
                           (df['num_moves'].max() - df['num_moves'].min() + 1e-10)
        # search_efficiency は高いほど良いため、スコアとしては (1 - 正規化値) を使うことで、低い方が良い指標に合わせる
        efficiency_normalized = 1 - ((df['search_efficiency'] - df['search_efficiency'].min()) / \
                                     (df['search_efficiency'].max() - df['search_efficiency'].min() + 1e-10))

        df['composite_score'] = time_normalized + moves_normalized + efficiency_normalized
    else:
        # データポイントが1つしかない場合は正規化できないため、デフォルトスコアを設定
        df['composite_score'] = 1.0

    # Add Manhattan distance diffs if columns are available
    if 'manhattan_min' in df.columns and 'num_moves' in df.columns:
        df['diff_from_manhattan_min'] = df['num_moves'] - df['manhattan_min']
    if 'manhattan_max' in df.columns and 'num_moves' in df.columns:
        df['diff_from_manhattan_max'] = df['num_moves'] - df['manhattan_max']

    return df


def calculate_summary(df):
    """サマリー統計を計算"""
    if df.empty or 'solved' not in df.columns:
        return None

    solved_df = df[df['solved'] == True]

    summary = {
            'total_problems': len(df),
            'solved_problems': len(solved_df),
            'total_calculation_time_ms': df['calculation_time_ms'].sum(),
            'avg_calculation_time_ms': df['calculation_time_ms'].mean(),
            'total_actual_time_ms': df['actual_execution_time_ms'].sum(),
            'avg_actual_time_ms': df['actual_execution_time_ms'].mean(),
            'total_nodes_explored': df['nodes_explored'].sum(),
            'avg_nodes_explored': df['nodes_explored'].mean(),
            'avg_moves': df['num_moves'].mean(),
            'avg_diff_from_estimated': df['diff_from_estimated'].mean(),
            'avg_diff_from_n_squared': df['diff_from_n_squared'].mean(),
            'avg_search_efficiency': df['search_efficiency'].mean(),
            'avg_composite_score': df['composite_score'].mean()
            }
    return summary


@st.cache_data
def load_benchmark_results():
    """ベンチマーク結果を読み込む"""
    results_file = Path("benchmark_results.csv")
    summary_file = Path("benchmark_summary.json")
    problems_summary_file = Path("problems_summary.csv")
    problems_summary_df = None

    try:
        if problems_summary_file.exists():
            problems_summary_df = pd.read_csv(problems_summary_file)
            if problems_summary_df.empty:
                st.warning(f"問題概要ファイル '{problems_summary_file}' は空です。マンハッタン距離比較は利用できません。")
                problems_summary_df = None
        else:
            st.info(f"問題概要ファイル '{problems_summary_file}' が見つかりません。マンハッタン距離比較は利用できません。")
    except pd.errors.EmptyDataError: # Should be caught by problems_summary_df.empty check, but as a safeguard
        st.warning(f"問題概要ファイル '{problems_summary_file}' は空です。マンハッタン距離比較は利用できません。")
        problems_summary_df = None
    except pd.errors.ParserError:
        st.error(f"問題概要ファイル '{problems_summary_file}' の解析に失敗しました。マンハッタン距離比較は利用できません。")
        problems_summary_df = None
    except FileNotFoundError: # Should be caught by .exists(), but as a safeguard
        st.info(f"問題概要ファイル '{problems_summary_file}' が見つかりません。マンハッタン距離比較は利用できません。")
        problems_summary_df = None


    if not results_file.exists():
        return pd.DataFrame(), None

    try:
        df = pd.read_csv(results_file)
    except pd.errors.EmptyDataError:
        st.warning(f"ベンチマーク結果ファイル '{results_file}' は空です。")
        return pd.DataFrame(), None
    except pd.errors.ParserError:
        st.error(f"ベンチマーク結果ファイル '{results_file}' の解析に失敗しました。ファイルが破損している可能性があります。")
        return pd.DataFrame(), None

    if problems_summary_df is not None and not problems_summary_df.empty and 'problem_id' in df.columns:
        try:
            # Ensure 'problem_id' in df is string before stripping, and handle potential errors if it's not
            if pd.api.types.is_string_dtype(df['problem_id']):
                df['problem_id_numeric'] = df['problem_id'].str.lstrip('p').astype(int)
            elif pd.api.types.is_numeric_dtype(df['problem_id']): # If it's already numeric (e.g. from old file)
                df['problem_id_numeric'] = df['problem_id'].astype(int)
            else: # Try converting to string first
                df['problem_id_numeric'] = df['problem_id'].astype(str).str.lstrip('p').astype(int)

            df = pd.merge(
                df,
                problems_summary_df,
                left_on=['size', 'problem_id_numeric'],
                right_on=['size', 'problem_id'], # Assuming 'problem_id' in problems_summary.csv is already numeric
                how='left',
                suffixes=('', '_summary')
            )
            if 'problem_id_summary' in df.columns:
                df = df.drop(columns=['problem_id_summary'])
            # We might want to keep problem_id_numeric for other uses or drop it:
            # if 'problem_id_numeric' in df.columns:
            #     df = df.drop(columns=['problem_id_numeric'])
        except ValueError as e:
            st.error(f"ベンチマーク結果の 'problem_id' の形式が不正なため、問題概要データとのマージに失敗しました: {e}")
        except Exception as e: # Catch any other unexpected error during merge
            st.error(f"問題概要データとのマージ中に予期せぬエラーが発生しました: {e}")


    summary = None
    if summary_file.exists():
        try:
            # First, check if the file is not empty to avoid an error
            if summary_file.stat().st_size > 0:
                with summary_file.open('r', encoding='utf-8') as f:
                    summary = json.load(f)
            else:
                st.warning(f"警告: '{summary_file}' は空です。無視されます。")
        except json.JSONDecodeError as e:
            st.warning(f"'{summary_file}' の読み込みに失敗しました。ファイルが破損している可能性があります。")
            st.error(f"エラー詳細: {e}")
            st.info("ベンチマークを再実行すると、このファイルは自動的に修復（上書き）されます。")
            # On error, ensure summary remains None so the app can proceed
            summary = None

    return df, summary


def kill_process_tree(process):
    """プロセスツリー全体を確実に終了する"""
    try:
        # psutilを使用して指定されたプロセスの情報を取得
        parent = psutil.Process(process.pid)
        # そのプロセスの子プロセスを再帰的に全て取得
        children = parent.children(recursive=True)

        # まず子プロセスから順に終了させる
        for child in children:
            try:
                child.terminate()  # SIGTERMを送信
            except psutil.NoSuchProcess:
                pass  # プロセスが既に存在しない場合は何もしない

        # 次に親プロセスを終了させる
        try:
            parent.terminate()  # SIGTERMを送信
        except psutil.NoSuchProcess:
            pass  # プロセスが既に存在しない場合は何もしない

        # terminateシグナルで終了しなかったプロセスがいないか確認し、強制終了する
        # wait_procsでプロセスの終了を待つ（タイムアウト付き）
        gone, still_alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in still_alive: # タイムアウト後も生存しているプロセスに対して
            try:
                p.kill()  # SIGKILLを送信して強制終了
            except psutil.NoSuchProcess:
                pass  # プロセスが既に存在しない場合は何もしない

    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as e:
        # 指定されたプロセスが存在しない、アクセス権がない、その他のOSエラーの場合
        # subprocessの標準的な方法でプロセス終了を試みる
        st.warning(f"psutilでのプロセスツリー終了中にエラー: {e}。標準的な終了処理を試みます。")
        try:
            process.terminate()  # SIGTERMを送信
            process.wait(timeout=3)  # 終了を待つ（タイムアウト付き）
        except subprocess.TimeoutExpired:
            process.kill()  # タイムアウトならSIGKILLを送信
            process.wait()
        except OSError as final_e: # killでもエラーが出る場合
            st.error(f"標準的なプロセス終了処理でもエラー: {final_e}。")
            pass # これ以上できることは少ない


def run_single_problem(executable_path, problem_path):
    """単一の問題を実行し、結果を返す"""
    cat_process = None
    astar_process = None

    try:
        start_time = time.perf_counter()

        # cat problem_file | ./astar_manhattan の形で実行
        cat_process = subprocess.Popen(
                ["cat", problem_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
                )

        astar_process = subprocess.Popen(
                [executable_path],
                stdin=cat_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
                )

        cat_process.stdout.close()

        stdout, stderr = astar_process.communicate(timeout=60)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000  # ミリ秒

        if astar_process.returncode != 0:
            return None, f"実行エラー: {stderr}"

        try:
            output_data = json.loads(stdout.strip())
            output_data['actual_execution_time_ms'] = execution_time
            output_data['problem_path'] = problem_path
            return output_data, None
        except json.JSONDecodeError as e:
            return None, f"JSON解析エラー: {e}\n出力: {stdout}"

    except subprocess.TimeoutExpired:
        # タイムアウト時の処理を改善
        error_msg = "タイムアウト (60秒)"

        # プロセスを確実に終了
        if astar_process is not None:
            kill_process_tree(astar_process)
        if cat_process is not None:
            kill_process_tree(cat_process)

        return None, error_msg

    except Exception as e:
        # その他のエラー時もプロセスをクリーンアップ
        if astar_process is not None:
            try:
                kill_process_tree(astar_process)
            except:
                pass
        if cat_process is not None:
            try:
                kill_process_tree(cat_process)
            except:
                pass

        return None, f"予期せぬ実行エラー: {str(e)}"


def run_benchmark_with_config(executable_path_str, problems_dir_str, problems_per_size, min_size, max_size):
    """設定を使用してベンチマークを実行（サイズ範囲指定付き）"""
    executable_path = Path(executable_path_str)
    problems_dir = Path(problems_dir_str)

    if not executable_path.exists():
        st.error(f"実行ファイル '{executable_path}' が見つかりません")
        return

    if not problems_dir.is_dir(): # Changed from exists() to is_dir() for clarity
        st.error(f"問題ディレクトリ '{problems_dir}' が見つかりません")
        return

    # 問題ファイルを取得するロジック
    problem_files = []
    # problemsディレクトリ内のフォルダ('4x4', '5x5'など)を名前からサイズを読み取りソートして取得
    size_dirs = sorted(
        [d for d in problems_dir.iterdir() if d.is_dir() and 'x' in d.name],  # 'x' を含むディレクトリのみを対象
        key=lambda d: int(d.name.split('x')[0])
    )

    for size_dir in size_dirs:
        try:
            # ディレクトリ名 (例: "4x4") からサイズ (例: 4) を取得
            current_size = int(size_dir.name.split('x')[0])
            # 指定されたサイズ範囲外であれば、このディレクトリ内の問題はスキップ
            if not (min_size <= current_size <= max_size):
                continue
        except (ValueError, IndexError):
            # '4x4' のような形式でないディレクトリは無視
            continue

        # 各サイズディレクトリ内の問題ファイル（.json）を名前順でソートして取得
        files_in_dir = sorted(size_dir.glob("*.json"), key=lambda path: int(path.stem.lstrip('p')))
        # 指定された問題数だけスライスして、全体のリストに追加
        problem_files.extend(files_in_dir[:problems_per_size])

    if not problem_files:
        st.error(f"指定された範囲 ({min_size}x{min_size}〜{max_size}x{max_size}) で問題ファイル(.json)が見つかりません")
        return

    st.info(f"実行する問題数: {len(problem_files)}")

    progress_bar = st.progress(0, text="ベンチマーク実行準備中...")
    results = []  # 成功した問題の結果を格納するリスト
    failed_problems_details = [] # 失敗した問題の詳細を格納するリスト
    start_time = time.time()

    for i, problem_path in enumerate(problem_files):
        progress = (i + 1) / len(problem_files)
        # UIに進捗を表示 (例: "実行中 (10/100): p010.json")
        progress_bar.progress(progress, text=f"実行中 ({i + 1}/{len(problem_files)}): {problem_path.name}")

        # 個別の問題を実行し、結果とエラー(あれば)を取得
        result, error = run_single_problem(str(executable_path), str(problem_path))

        if result: # 実行成功時
            try:
                # 問題ファイルパスからサイズと問題IDを抽出
                # 例: problems/4x4/p001.json -> path_parts = ('problems', '4x4', 'p001.json')
                path_parts = problem_path.parts
                size_str = path_parts[-2]  # '4x4'
                problem_id_str = problem_path.stem  # 'p001'
                size = int(size_str.split('x')[0]) # 4

                # 結果にIDとサイズ情報を追加してリストに格納
                result['problem_id'] = problem_id_str
                result['size'] = size
                results.append(result)
            except (IndexError, ValueError) as e:
                # パス解析に失敗した場合 (通常は起こり得ないが念のため)
                st.warning(f"パス '{problem_path}' からサイズまたはIDの解析に失敗: {e}")
        else: # 実行失敗時
            # UIに警告を表示し、失敗リストに詳細を記録
            st.warning(f"問題 {problem_path.name} でエラー: {error}")
            failed_problems_details.append({'name': problem_path.name, 'path': str(problem_path), 'error': error})

    end_time = time.time() # ベンチマーク全体の終了時刻

    if results:
        df = pd.DataFrame(results)
        df = calculate_metrics(df)
        results_csv_path = Path("benchmark_results.csv")
        summary_json_path = Path("benchmark_summary.json")
        df.to_csv(results_csv_path, index=False)

        summary = calculate_summary(df)
        if summary:
            with summary_json_path.open('w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=to_serializable)

        progress_bar.progress(1.0, text="完了！")
        st.success(f"ベンチマーク実行完了！ (合計時間: {end_time - start_time:.2f}秒)")
        st.success(f"成功: {len(results)}/{len(problem_files)} 問題")

        if failed_problems_details:
            st.error(f"実行中に {len(failed_problems_details)} 件のエラーが発生しました。")
            with st.expander("⚠️ ベンチマーク実行エラーの詳細を表示", expanded=True):
                for failure in failed_problems_details:
                    st.markdown(f"**問題ファイル:** `{failure['name']}`")
                    st.markdown(f"**パス:** `{failure['path']}`")
                    st.markdown(f"**エラー内容:**")
                    st.error(f"{failure['error']}") # Using st.error for the message itself
                    st.markdown("---") # Separator

        load_benchmark_results.clear()
        st.rerun()
    else:
        st.error("実行に成功した問題がありませんでした")
        if failed_problems_details: # Also show errors if no problems succeeded
            st.error(f"実行中に {len(failed_problems_details)} 件のエラーが発生しました。")
            with st.expander("⚠️ ベンチマーク実行エラーの詳細を表示", expanded=True):
                for failure in failed_problems_details:
                    st.markdown(f"**問題ファイル:** `{failure['name']}`")
                    st.markdown(f"**パス:** `{failure['path']}`")
                    st.markdown(f"**エラー内容:**")
                    st.error(f"{failure['error']}")
                    st.markdown("---")


def display_metrics_overview(summary):
    """メトリクス概要を表示"""
    st.header("📊 パフォーマンス概要")

    col1, col2, col3, col4 = st.columns(4)
    solved_percentage = (summary['solved_problems'] / summary['total_problems'] * 100) if summary['total_problems'] > 0 else 0

    with col1:
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h3>解決率</h3>
            <h2>{summary.get('solved_problems', 0)}/{summary.get('total_problems', 0)}</h2>
            <p>{solved_percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card info-metric">
            <h3>平均計算時間</h3>
            <h2>{summary.get('avg_calculation_time_ms', 0):.2f}ms</h2>
            <p>アルゴリズム内部</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card warning-metric">
            <h3>平均探索ノード</h3>
            <h2>{summary.get('avg_nodes_explored', 0):,.0f}</h2>
            <p>nodes</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>平均手数</h3>
            <h2>{summary.get('avg_moves', 0):.1f}</h2>
            <p>moves</p>
        </div>
        """, unsafe_allow_html=True)


def plot_time_performance(df):
    """時間性能の可視化"""
    st.subheader("⏱️ 時間性能分析")

    col1, col2 = st.columns(2)

    with col1:
        size_time = df.groupby('size')['calculation_time_ms'].agg(['mean', 'std']).reset_index()
        fig = px.line(size_time, x='size', y='mean',
                      title='サイズ別平均計算時間',
                      labels={'mean': '平均計算時間 (ms)', 'size': 'パズルサイズ'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x='calculation_time_ms',
                           title='計算時間分布',
                           labels={'calculation_time_ms': '計算時間 (ms)', 'count': '問題数'},
                           nbins=50)
        st.plotly_chart(fig, use_container_width=True)


def plot_search_performance(df):
    """探索性能の可視化"""
    st.subheader("🔍 探索性能分析")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(df, x='size', y='nodes_explored',
                     title='サイズ別探索ノード数分布',
                     labels={'nodes_explored': '探索ノード数', 'size': 'パズルサイズ'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='size', y='search_efficiency',
                         color='calculation_time_ms',
                         title='探索効率 vs パズルサイズ',
                         labels={'search_efficiency': '探索効率 (nodes/sec)',
                                 'size': 'パズルサイズ',
                                 'calculation_time_ms': '計算時間 (ms)'},
                         color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)


def plot_solution_quality(df):
    """解の品質分析"""
    st.subheader("🎯 解の品質分析")
    st.caption("""
    ここでは、実際の解決手数が理論的な推定値や限界値とどの程度異なるかを分析します。
    - 「推定最適解」は、ユーザー提供の計算式 (y = 1/8 n³ - n² + 7n - 14, 結果は最小1) に基づく理論的な最適手数です。グラフではこの推定値からの差分（実際の手数 - 推定最適解）を示します。
    - 「理論最大値(n²)」は、手数の単純な上限の目安です。グラフではこのn²からの差分（実際の手数 - n²）を示します。
    """)
    if 'diff_from_estimated' not in df.columns:
        st.info("解の品質を分析するためのデータが不足しています。")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='size', y='diff_from_estimated',
                         color='num_moves',
                         title='推定最適解 (カスタム式) からの差分',
                         labels={'diff_from_estimated': '差分 (moves)',
                                 'size': 'パズルサイズ',
                                 'num_moves': '実際の手数'},
                         color_continuous_scale='RdYlBu_r')
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="理論最適値")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='size', y='diff_from_n_squared',
                         color='composite_score',
                         title='理論最大値(n²)からの差分',
                         labels={'diff_from_n_squared': '差分 (moves)',
                                 'size': 'パズルサイズ',
                                 'composite_score': '総合スコア'},
                         color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)


def plot_manhattan_comparison(df):
    """マンハッタン距離との比較"""
    if all(col in df.columns for col in ['diff_from_manhattan_min', 'diff_from_manhattan_max']):
        st.subheader("📏 マンハッタン距離ベース比較")
        st.caption("""
        マンハッタン距離は、各牌の現在位置から目的位置までの縦横の移動距離の総和に関連するヒューリスティックです。
        ここでは、A*探索で得られた実際の解決手数が、問題の初期状態のマンハッタン距離から計算される理論的な手数範囲（最小・最大推定値）とどの程度異なるかを示します。
        これにより、ヒューリスティック関数（この場合はマンハッタン距離に関連する何か）の品質や、解がこれらの推定範囲内に収まるかを評価するのに役立ちます。
        """)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='size', y='diff_from_manhattan_min',
                         title='マンハッタン最小値からの差分分布',
                         labels={'diff_from_manhattan_min': '差分 (moves)', 'size': 'パズルサイズ'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, x='size', y='diff_from_manhattan_max',
                         title='マンハッタン最大値からの差分分布',
                         labels={'diff_from_manhattan_max': '差分 (moves)', 'size': 'パズルサイズ'})
            st.plotly_chart(fig, use_container_width=True)


def plot_comprehensive_analysis(df):
    """総合分析 (各指標を0-100のスコアに正規化)"""
    st.subheader("📈 総合分析")
    st.info("各指標を0-100のスコアに変換して表示します。値が100に近いほど、そのサイズでのパフォーマンスが良いことを示します。")

    # NaNを含む行は集計から除外
    agg_df = df.dropna(subset=['calculation_time_ms', 'nodes_explored', 'num_moves'])

    if agg_df.empty:
        st.warning("総合分析を表示するための有効なデータがありません。")
        return

    size_summary = agg_df.groupby('size').agg({
        'calculation_time_ms': 'mean',
        'nodes_explored': 'mean',
        'num_moves': 'mean',
        'search_efficiency': 'mean',
        'composite_score': 'mean'
        }).reset_index()

    # 正規化対象のメトリクスとその評価タイプ ('lower_is_better' または 'higher_is_better')
    metrics_to_scale = {
            'calculation_time_ms': 'lower_is_better',  # 計算時間は短いほど良い
            'nodes_explored': 'lower_is_better',       # 探索ノード数は少ないほど良い
            'num_moves': 'lower_is_better',            # 手数は少ないほど良い
            'composite_score': 'lower_is_better',      # 総合スコアは小さいほど良い
            'search_efficiency': 'higher_is_better'    # 探索効率は高いほど良い
            }

    scaled_summary = size_summary.copy()

    for metric, scale_type in metrics_to_scale.items():
        if metric in scaled_summary.columns:
            min_val = scaled_summary[metric].min()
            max_val = scaled_summary[metric].max()

            # ゼロ除算を避けるためのチェック (minとmaxが同じ場合、全ての値が同じなのでスコアは100とする)
            if (max_val - min_val) == 0:
                scaled_summary[metric] = 100.0  # 全て同じ値なら満点（または中間点、ここでは100）
                continue

            # Min-Max スケーリング を0-100の範囲で行う
            if scale_type == 'lower_is_better':
                # 値が小さいほどスコアが100に近づくように正規化
                # (例: min=10, max=110 の時、値が10ならスコア100、値が110ならスコア0)
                scaled_summary[metric] = 100 * (1 - (scaled_summary[metric] - min_val) / (max_val - min_val))
            else: # 'higher_is_better'
                # 値が大きいほどスコアが100に近づくように正規化
                # (例: min=10, max=110 の時、値が10ならスコア0、値が110ならスコア100)
                scaled_summary[metric] = 100 * ((scaled_summary[metric] - min_val) / (max_val - min_val))

    # 表示するメトリクスの順番と名前を定義
    display_metrics = {
            'calculation_time_ms': '時間効率スコア',
            'nodes_explored': '探索空間スコア',
            'num_moves': '解品質スコア',
            'search_efficiency': '探索効率スコア',
            'composite_score': '総合スコア'
            }

    available_metrics = [m for m in display_metrics.keys() if m in scaled_summary.columns]

    if available_metrics:
        # インデックスをサイズに設定し、表示する列を選択
        heatmap_data = scaled_summary.set_index('size')[available_metrics].T
        # インデックス（Y軸のラベル）を日本語に置換
        heatmap_data = heatmap_data.rename(index=display_metrics)

        fig = px.imshow(heatmap_data,
                        title='サイズ別パフォーマンススコア ヒートマップ',
                        labels={'x': 'パズルサイズ', 'y': '評価指標', 'color': 'スコア (0-100)'},
                        color_continuous_scale='RdYlGn', # 赤(悪い) -> 黄 -> 緑(良い) のカラースケール
                        aspect="auto"
                        )
        st.plotly_chart(fig, use_container_width=True)
def display_detailed_table(df):
    """詳細データテーブル"""
    st.subheader("📋 詳細データ")
    st.caption("""
    **表の主な指標の説明:**
    - **size:** パズルのサイズ (例: 4x4の場合, 4)。
    - **problem_id:** 問題の識別子。
    - **solved:** 問題が解決されたかどうか (True/False)。
    - **num_moves:** 解決までの実際の手数。
    - **calculation_time_ms:** アルゴリズムによる計算時間 (ミリ秒)。
    - **nodes_explored:** 探索中に展開されたノードの総数。
    - **search_efficiency:** 探索効率 (ノード/秒)。
    - **diff_from_estimated:** 「推定最適解 (カスタム式: 1/8 n³ - n² + 7n - 14, 最小1)」からの実際の手数の差。
    - **composite_score:** 計算時間, 手数, 探索効率を正規化して組み合わせた総合スコア（このスコアが小さいほど良いと評価されます）。
    - **error:** 問題解決に失敗した場合のエラーメッセージ。
    (上記に加えて, `problems_summary.csv` とのマージに成功した場合, `manhattan_min`, `manhattan_max` などの列も表示されることがあります。)
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        size_filter = st.multiselect("パズルサイズ選択",
                                     options=sorted(df['size'].dropna().unique().astype(int)),
                                     default=sorted(df['size'].dropna().unique().astype(int)))
    with col2:
        # NaNを無視して最小・最大を計算
        min_time, max_time = float(df['calculation_time_ms'].min()), float(df['calculation_time_ms'].max())
        time_range = st.slider("計算時間範囲 (ms)", min_value=min_time, max_value=max_time, value=(min_time, max_time))

    with col3:
        # ## 変更: フィルタの選択肢を増やす ##
        solved_filter = st.selectbox("表示対象", options=["すべて", "解決済みのみ", "失敗のみ"], index=0)

    filtered_df = df[df['size'].isin(size_filter)]

    # 時間でのフィルタリングは解決済みのものにのみ適用
    solved_part = filtered_df[filtered_df['solved'] == True]
    failed_part = filtered_df[filtered_df['solved'] == False]

    solved_part = solved_part[
            (solved_part['calculation_time_ms'] >= time_range[0]) &
            (solved_part['calculation_time_ms'] <= time_range[1])
            ]

    filtered_df = pd.concat([solved_part, failed_part])

    if solved_filter == "解決済みのみ":
        filtered_df = filtered_df[filtered_df['solved'] == True]
    elif solved_filter == "失敗のみ":
        filtered_df = filtered_df[filtered_df['solved'] == False]


    # ## 変更: error列を追加 ##
    display_columns = [
            'size', 'problem_id', 'solved', 'num_moves', 'calculation_time_ms',
            'nodes_explored', 'search_efficiency',
            'diff_from_estimated', 'composite_score', 'error'
            ]
    available_columns = [col for col in display_columns if col in filtered_df.columns]

    # NaNを 'N/A' に置換して表示
    st.dataframe(filtered_df[available_columns].fillna('N/A').round(4), use_container_width=True, height=400)
    st.write(f"表示データ数: {len(filtered_df)} / {len(df)}")

def main():
    st.title("🔍 高専プロコン競技 ベンチマーク結果ダッシュボード")

    st.sidebar.title("⚙️ 操作パネル")
    executable_path_str = st.sidebar.text_input("実行ファイルパス", value="./astar_manhattan")
    problems_dir_str = st.sidebar.text_input("問題ディレクトリ", value="problems")
    executable_path = Path(executable_path_str)
    problems_dir = Path(problems_dir_str)

    # ## 追加: 問題ディレクトリから利用可能なサイズの範囲を自動検出 ##
    min_avail = 4
    max_avail = 24 # デフォルト値
    # problem_path_obj = Path(problems_dir) # Already a Path object
    if problems_dir.exists() and problems_dir.is_dir():
        available_sizes = []
        for d in problems_dir.iterdir():
            if d.is_dir():
                try:
                    size = int(d.name.split('x')[0])
                    available_sizes.append(size)
                except (ValueError, IndexError):
                    pass # '4x4'のような名前でないディレクトリは無視
        if available_sizes:
            min_avail = min(available_sizes)
            max_avail = max(available_sizes)

    # ## 追加: サイズ範囲を指定するための入力欄 ##
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 実行範囲の設定")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_size = st.number_input("最小サイズ", min_value=min_avail, max_value=max_avail, value=min_avail, step=2)
    with col2:
        max_size = st.number_input("最大サイズ", min_value=min_avail, max_value=max_avail, value=max_avail, step=2)

    if min_size > max_size:
        st.sidebar.error("最小サイズが最大サイズを超えています。")

    problems_per_size = st.sidebar.number_input(
            "各サイズで実行する問題数",
            min_value=1,
            value=5,
            step=1,
            help="各サイズ（4x4, 5x5...）ごとに、ここで指定した数の問題を実行します。ファイル名順に取得します。"
            )
    st.sidebar.markdown("---")

    # ## 変更: ベンチマーク実行ボタンのロジック ##
    if st.sidebar.button("🚀 ベンチマーク実行", type="primary", disabled=(min_size > max_size)):
        # ## 変更: min_sizeとmax_sizeを引数に追加 ##
        run_benchmark_with_config(str(executable_path), str(problems_dir), problems_per_size, min_size, max_size)

    df, summary = load_benchmark_results()

    if df.empty:
        st.warning("⚠️ ベンチマーク結果が見つかりません。")
        st.info("サイドバーの「ベンチマーク実行」ボタンを押して、まずベンチマークを実行してください。")
        return

    st.sidebar.success(f"✅ データ読み込み完了")
    st.sidebar.info(f"問題数: {len(df)}")
    st.sidebar.info(f"サイズ範囲: {df['size'].min()}x{df['size'].min()} - {df['size'].max()}x{df['size'].max()}")

    if summary:
        display_metrics_overview(summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏱️ 時間性能", "🔍 探索性能", "🎯 解品質", "📈 総合分析", "📋 詳細データ"
        ])

    with tab1:
        plot_time_performance(df)
    with tab2:
        plot_search_performance(df)
    with tab3:
        plot_solution_quality(df)
        plot_manhattan_comparison(df)
    with tab4:
        plot_comprehensive_analysis(df)
    with tab5:
        display_detailed_table(df)

    st.markdown("---")
    st.markdown("🚀 A*アルゴリズムベンチマークシステム | Built with Streamlit")

if __name__ == "__main__":
    main()
