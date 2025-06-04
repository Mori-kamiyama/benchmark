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

    # 最適解推定式 3n²/8 との差分
    df['estimated_optimal'] = (3 * df['size'] * df['size']) / 8
    df['diff_from_estimated'] = df['num_moves'] - df['estimated_optimal']

    # n²との差分
    df['n_squared'] = df['size'] * df['size']
    df['diff_from_n_squared'] = df['num_moves'] - df['n_squared']

    # 探索効率 (solved problems per second based on nodes explored)
    df['search_efficiency'] = df['nodes_explored'] / (df['calculation_time_ms'] / 1000)

    # 総合スコア（参考値）- 小さいほど良い
    if len(df) > 1:  # 正規化のために複数のデータが必要
        time_normalized = (df['calculation_time_ms'] - df['calculation_time_ms'].min()) / (
                df['calculation_time_ms'].max() - df['calculation_time_ms'].min() + 1e-10)
        moves_normalized = (df['num_moves'] - df['num_moves'].min()) / (
                df['num_moves'].max() - df['num_moves'].min() + 1e-10)
        efficiency_normalized = 1 - ((df['search_efficiency'] - df['search_efficiency'].min()) / (
            df['search_efficiency'].max() - df['search_efficiency'].min() + 1e-10))

        df['composite_score'] = time_normalized + moves_normalized + efficiency_normalized
    else:
        df['composite_score'] = 1.0

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
    results_file = "benchmark_results.csv"
    summary_file = "benchmark_summary.json"

    if not os.path.exists(results_file):
        return pd.DataFrame(), None

    df = pd.read_csv(results_file)

    summary = None
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                # First, check if the file is not empty to avoid an error
                if os.path.getsize(summary_file) > 0:
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
        # psutilを使用してプロセスツリーを取得
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)

        # 子プロセスから順に終了
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # 親プロセスを終了
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass

        # 少し待ってから強制終了
        gone, still_alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in still_alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        # プロセスが既に終了している場合やアクセス権限がない場合
        # 通常のプロセス終了を試行
        try:
            process.terminate()
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except OSError:
            pass


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


def run_benchmark_with_config(executable_path, problems_dir, problems_per_size, min_size, max_size):
    """設定を使用してベンチマークを実行（サイズ範囲指定付き）"""
    if not os.path.exists(executable_path):
        st.error(f"実行ファイル '{executable_path}' が見つかりません")
        return

    if not os.path.exists(problems_dir):
        st.error(f"問題ディレクトリ '{problems_dir}' が見つかりません")
        return

    # 問題ファイルを取得するロジック
    problem_files = []
    # problemsディレクトリ内のフォルダ('4x4', '5x5'など)をソートして取得
    size_dirs = sorted([d for d in Path(problems_dir).iterdir() if d.is_dir()], key=lambda d: int(d.name.split('x')[0]))

    for size_dir in size_dirs:
        try:
            # ## 追加: ディレクトリ名からサイズをパース ##
            current_size = int(size_dir.name.split('x')[0])
            # ## 追加: 指定されたサイズ範囲外であればスキップ ##
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
    results = []
    start_time = time.time()

    for i, problem_path in enumerate(problem_files):
        progress = (i + 1) / len(problem_files)
        progress_bar.progress(progress, text=f"実行中 ({i + 1}/{len(problem_files)}): {problem_path.name}")

        result, error = run_single_problem(executable_path, str(problem_path))

        if result:
            try:
                path_parts = problem_path.parts
                size_str = path_parts[-2]
                problem_id_str = problem_path.stem
                size = int(size_str.split('x')[0])

                result['problem_id'] = problem_id_str
                result['size'] = size
                results.append(result)
            except (IndexError, ValueError) as e:
                st.warning(f"パス '{problem_path}' からサイズまたはIDの解析に失敗: {e}")
        else:
            st.warning(f"問題 {problem_path.name} でエラー: {error}")

    end_time = time.time()

    if results:
        df = pd.DataFrame(results)
        df = calculate_metrics(df)
        df.to_csv("benchmark_results.csv", index=False)

        summary = calculate_summary(df)
        if summary:
            with open("benchmark_summary.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=to_serializable)

        progress_bar.progress(1.0, text="完了！")
        st.success(f"ベンチマーク実行完了！ (合計時間: {end_time - start_time:.2f}秒)")
        st.success(f"成功: {len(results)}/{len(problem_files)} 問題")

        load_benchmark_results.clear()
        st.rerun()
    else:
        st.error("実行に成功した問題がありませんでした")


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
    if 'diff_from_estimated' not in df.columns:
        st.info("解の品質を分析するためのデータが不足しています。")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='size', y='diff_from_estimated',
                         color='num_moves',
                         title='推定最適解(3n²/8)からの差分',
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

    # 正規化対象のメトリクス
    metrics_to_scale = {
            'calculation_time_ms': 'lower_is_better',
            'nodes_explored': 'lower_is_better',
            'num_moves': 'lower_is_better',
            'composite_score': 'lower_is_better',
            'search_efficiency': 'higher_is_better'
            }

    scaled_summary = size_summary.copy()

    for metric, scale_type in metrics_to_scale.items():
        if metric in scaled_summary.columns:
            min_val = scaled_summary[metric].min()
            max_val = scaled_summary[metric].max()

            # ゼロ除算を避ける
            if (max_val - min_val) == 0:
                scaled_summary[metric] = 100.0
                continue

            # Min-Max スケーリング
            if scale_type == 'lower_is_better':
                # 値が小さいほど100に近づく
                scaled_summary[metric] = 100 * (1 - (scaled_summary[metric] - min_val) / (max_val - min_val))
            else: # higher_is_better
                # 値が大きいほど100に近づく
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
    executable_path = st.sidebar.text_input("実行ファイルパス", value="./astar_manhattan")
    problems_dir = st.sidebar.text_input("問題ディレクトリ", value="problems")

    # ## 追加: 問題ディレクトリから利用可能なサイズの範囲を自動検出 ##
    min_avail = 4
    max_avail = 24 # デフォルト値
    problem_path_obj = Path(problems_dir)
    if problem_path_obj.exists() and problem_path_obj.is_dir():
        available_sizes = []
        for d in problem_path_obj.iterdir():
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
        run_benchmark_with_config(executable_path, problems_dir, problems_per_size, min_size, max_size)

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
