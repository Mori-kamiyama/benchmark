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
import datetime  # 日時を扱うために追加
import signal
import psutil
import glob  # ファイル検索のために追加

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
    height: 100%;
    display: flex;
    flex-direction: column;
}
.metric-card h3 {
    margin-bottom: 0.5rem;
}
.metric-card h2 {
    margin-bottom: 0.25rem;
}
.metric-card p {
    margin-top: auto;
    color: #64748b;
    font-size: 0.9rem;
    padding-top: 0.5rem;
}
.success-metric { border-left-color: #51cf66; }
.warning-metric { border-left-color: #ffd43b; }
.info-metric { border-left-color: #339af0; }
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
    return obj


def calculate_metrics(df):
    """評価指標を計算"""
    if df.empty:
        return df

    # 既存の計算に加えて、存在チェックを追加
    df_copy = df.copy()
    df_copy['estimated_optimal'] = (3 * df_copy['size'] * df_copy['size']) / 8
    df_copy['diff_from_estimated'] = df_copy['num_moves'] - df_copy['estimated_optimal']
    df_copy['n_squared'] = df_copy['size'] * df_copy['size']
    df_copy['diff_from_n_squared'] = df_copy['num_moves'] - df_copy['n_squared']
    df_copy['search_efficiency'] = df_copy['nodes_explored'] / (df_copy['calculation_time_ms'] / 1000)

    if len(df_copy) > 1:
        time_normalized = (df_copy['calculation_time_ms'] - df_copy['calculation_time_ms'].min()) / (
                df_copy['calculation_time_ms'].max() - df_copy['calculation_time_ms'].min() + 1e-10)
        moves_normalized = (df_copy['num_moves'] - df_copy['num_moves'].min()) / (
                df_copy['num_moves'].max() - df_copy['num_moves'].min() + 1e-10)
        efficiency_normalized = 1 - ((df_copy['search_efficiency'] - df_copy['search_efficiency'].min()) / (
                df_copy['search_efficiency'].max() - df_copy['search_efficiency'].min() + 1e-10))
        df_copy['composite_score'] = time_normalized + moves_normalized + efficiency_normalized
    else:
        df_copy['composite_score'] = 1.0

    return df_copy


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
        'avg_moves': df['num_moves'].mean()
    }

    # オプショナルな列も平均を計算
    for col in ['diff_from_estimated', 'diff_from_n_squared', 'search_efficiency', 'composite_score']:
        if col in df.columns:
            summary[f'avg_{col}'] = df[col].mean()

    return summary


@st.cache_data
def load_benchmark_results(results_csv_path):
    """【変更】指定されたパスからベンチマーク結果を読み込む"""
    if not results_csv_path or not os.path.exists(results_csv_path):
        return pd.DataFrame(), None

    # JSONファイルパスをCSVパスから生成
    summary_file = str(Path(results_csv_path).with_suffix('.json'))

    df = pd.read_csv(results_csv_path)

    summary = None
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                if os.path.getsize(summary_file) > 0:
                    summary = json.load(f)
                else:
                    st.warning(f"警告: '{summary_file}' は空です。")
        except json.JSONDecodeError as e:
            st.warning(f"'{summary_file}' の読み込みに失敗しました。")
            st.error(f"エラー詳細: {e}")
            summary = None  # エラー時も動作を継続

    # 読み込んだ後に再度サマリーを計算する（JSONがない場合のため）
    if not summary:
        recalculated_df = calculate_metrics(df.copy())
        summary = calculate_summary(recalculated_df)

    return df, summary


def kill_process_tree(process):
    """プロセスツリー全体を確実に終了する"""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        gone, still_alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in still_alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
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
        cat_process = subprocess.Popen(["cat", problem_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        astar_process = subprocess.Popen([executable_path], stdin=cat_process.stdout, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
        cat_process.stdout.close()
        stdout, stderr = astar_process.communicate(timeout=60)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000

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
        error_msg = "タイムアウト (60秒)"
        if astar_process: kill_process_tree(astar_process)
        if cat_process: kill_process_tree(cat_process)
        return None, error_msg
    except Exception as e:
        if astar_process: kill_process_tree(astar_process)
        if cat_process: kill_process_tree(cat_process)
        return None, f"予期せぬ実行エラー: {str(e)}"


def run_benchmark_with_config(executable_path, problems_dir, problems_per_size, min_size, max_size):
    """【変更】設定を使用してベンチマークを実行し、結果を名前を付けて保存"""
    if not os.path.exists(executable_path):
        st.error(f"実行ファイル '{executable_path}' が見つかりません")
        return
    if not os.path.exists(problems_dir):
        st.error(f"問題ディレクトリ '{problems_dir}' が見つかりません")
        return

    problem_files = []
    size_dirs = sorted([d for d in Path(problems_dir).iterdir() if d.is_dir()], key=lambda d: int(d.name.split('x')[0]))
    for size_dir in size_dirs:
        try:
            current_size = int(size_dir.name.split('x')[0])
            if not (min_size <= current_size <= max_size):
                continue
        except (ValueError, IndexError):
            continue
        files_in_dir = sorted(size_dir.glob("*.json"), key=lambda path: int(path.stem.lstrip('p')))
        problem_files.extend(files_in_dir[:problems_per_size])

    if not problem_files:
        st.error(f"指定された範囲 ({min_size}x{min_size}〜{max_size}x{max_size}) で問題ファイルが見つかりません")
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
                size_str = problem_path.parts[-2]
                result['problem_id'] = problem_path.stem
                result['size'] = int(size_str.split('x')[0])
                results.append(result)
            except (IndexError, ValueError) as e:
                st.warning(f"パス '{problem_path}' からサイズまたはIDの解析に失敗: {e}")
        else:
            st.warning(f"問題 {problem_path.name} でエラー: {error}")
            # 失敗したケースも記録する
            try:
                size_str = problem_path.parts[-2]
                size = int(size_str.split('x')[0])
                results.append({'solved': False, 'size': size, 'problem_id': problem_path.stem, 'error': error})
            except (IndexError, ValueError):
                results.append({'solved': False, 'size': np.nan, 'problem_id': problem_path.stem, 'error': error})

    end_time = time.time()
    if results:
        df = pd.DataFrame(results)
        df = calculate_metrics(df.copy())
        summary = calculate_summary(df)

        # 【要件1, 2】結果の保存先とファイル名を設定
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        executable_name = Path(executable_path).stem
        base_filename = f"{executable_name}_{timestamp}"

        results_csv_path = os.path.join(results_dir, f"{base_filename}.csv")
        results_json_path = os.path.join(results_dir, f"{base_filename}.json")

        df.to_csv(results_csv_path, index=False)
        if summary:
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=to_serializable)

        progress_bar.progress(1.0, text="完了！")
        st.success(f"ベンチマーク実行完了！ (合計時間: {end_time - start_time:.2f}秒)")
        st.success(f"結果は '{results_csv_path}' に保存されました。")
        load_benchmark_results.clear()
        st.rerun()
    else:
        st.error("実行に成功した問題がありませんでした")


def display_metrics_overview(summary, header="📊 パフォーマンス概要"):
    """メトリクス概要を表示"""
    st.header(header)
    if not summary:
        st.warning("サマリーデータがありません。")
        return

    col1, col2, col3, col4 = st.columns(4)
    solved_percentage = (summary.get('solved_problems', 0) / summary.get('total_problems', 1) * 100)

    with col1:
        st.markdown(f"""<div class="metric-card success-metric">
            <h3>解決率</h3><h2>{summary.get('solved_problems', 0)}/{summary.get('total_problems', 0)}</h2>
            <p>{solved_percentage:.1f}%</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card info-metric">
            <h3>平均計算時間</h3><h2>{summary.get('avg_calculation_time_ms', 0):.2f}ms</h2>
            <p>アルゴリズム内部</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card warning-metric">
            <h3>平均探索ノード</h3><h2>{summary.get('avg_nodes_explored', 0):,.0f}</h2>
            <p>nodes</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <h3>平均手数</h3><h2>{summary.get('avg_moves', 0):.1f}</h2>
            <p>moves</p></div>""", unsafe_allow_html=True)


# --- 可視化関数群 (変更なし) ---
def plot_time_performance(df):
    st.subheader("⏱️ 時間性能分析")
    col1, col2 = st.columns(2)
    with col1:
        size_time = df.groupby('size')['calculation_time_ms'].agg(['mean', 'std']).reset_index()
        fig = px.line(size_time, x='size', y='mean', title='サイズ別平均計算時間',
                      labels={'mean': '平均計算時間 (ms)', 'size': 'パズルサイズ'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='calculation_time_ms', title='計算時間分布',
                           labels={'calculation_time_ms': '計算時間 (ms)', 'count': '問題数'}, nbins=50)
        st.plotly_chart(fig, use_container_width=True)


def plot_search_performance(df):
    st.subheader("🔍 探索性能分析")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x='size', y='nodes_explored', title='サイズ別探索ノード数分布',
                     labels={'nodes_explored': '探索ノード数', 'size': 'パズルサイズ'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x='size', y='search_efficiency', color='calculation_time_ms',
                         title='探索効率 vs パズルサイズ',
                         labels={'search_efficiency': '探索効率 (nodes/sec)', 'size': 'パズルサイズ',
                                 'calculation_time_ms': '計算時間 (ms)'}, color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)


def plot_solution_quality(df):
    st.subheader("🎯 解の品質分析")
    if 'diff_from_estimated' not in df.columns:
        st.info("解の品質を分析するためのデータが不足しています。")
        return
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='size', y='diff_from_estimated', color='num_moves', title='推定最適解(3n²/8)からの差分',
                         labels={'diff_from_estimated': '差分 (moves)', 'size': 'パズルサイズ',
                                 'num_moves': '実際の手数'}, color_continuous_scale='RdYlBu_r')
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="理論最適値")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x='size', y='diff_from_n_squared', color='composite_score',
                         title='理論最大値(n²)からの差分',
                         labels={'diff_from_n_squared': '差分 (moves)', 'size': 'パズルサイズ',
                                 'composite_score': '総合スコア'}, color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)


def plot_comprehensive_analysis(df):
    st.subheader("📈 総合分析")
    st.info("各指標を0-100のスコアに変換して表示。100に近いほど良好なパフォーマンスを示します。")
    agg_df = df.dropna(subset=['calculation_time_ms', 'nodes_explored', 'num_moves'])
    if agg_df.empty:
        st.warning("総合分析を表示するための有効なデータがありません。")
        return

    size_summary = agg_df.groupby('size').agg(
        calculation_time_ms=('calculation_time_ms', 'mean'),
        nodes_explored=('nodes_explored', 'mean'),
        num_moves=('num_moves', 'mean'),
        search_efficiency=('search_efficiency', 'mean'),
        composite_score=('composite_score', 'mean')
    ).reset_index()

    metrics_to_scale = {
        'calculation_time_ms': 'lower_is_better', 'nodes_explored': 'lower_is_better',
        'num_moves': 'lower_is_better', 'composite_score': 'lower_is_better',
        'search_efficiency': 'higher_is_better'
    }
    scaled_summary = size_summary.copy()
    for metric, scale_type in metrics_to_scale.items():
        if metric in scaled_summary.columns:
            min_val, max_val = scaled_summary[metric].min(), scaled_summary[metric].max()
            if (max_val - min_val) == 0:
                scaled_summary[metric] = 100.0
                continue
            if scale_type == 'lower_is_better':
                scaled_summary[metric] = 100 * (1 - (scaled_summary[metric] - min_val) / (max_val - min_val))
            else:
                scaled_summary[metric] = 100 * ((scaled_summary[metric] - min_val) / (max_val - min_val))

    display_metrics = {
        'calculation_time_ms': '時間効率スコア', 'nodes_explored': '探索空間スコア',
        'num_moves': '解品質スコア', 'search_efficiency': '探索効率スコア',
        'composite_score': '総合スコア'
    }
    available_metrics = [m for m in display_metrics.keys() if m in scaled_summary.columns]
    if available_metrics:
        heatmap_data = scaled_summary.set_index('size')[available_metrics].T.rename(index=display_metrics)
        fig = px.imshow(heatmap_data, title='サイズ別パフォーマンススコア ヒートマップ',
                        labels={'x': 'パズルサイズ', 'y': '評価指標', 'color': 'スコア (0-100)'},
                        color_continuous_scale='RdYlGn', aspect="auto")
        st.plotly_chart(fig, use_container_width=True)


def display_detailed_table(df):
    st.subheader("📋 詳細データ")
    if df.empty:
        st.warning("表示するデータがありません。")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        size_filter = st.multiselect("パズルサイズ選択", options=sorted(df['size'].dropna().unique().astype(int)),
                                     default=sorted(df['size'].dropna().unique().astype(int)))
    with col2:
        min_time, max_time = float(df['calculation_time_ms'].min()), float(df['calculation_time_ms'].max())
        time_range = st.slider("計算時間範囲 (ms)", min_value=min_time, max_value=max_time, value=(min_time, max_time))
    with col3:
        solved_filter = st.selectbox("表示対象", options=["すべて", "解決済みのみ", "失敗のみ"], index=0)

    filtered_df = df[df['size'].isin(size_filter)]
    solved_part = filtered_df[filtered_df['solved'] == True]
    failed_part = filtered_df[filtered_df['solved'] == False]
    solved_part = solved_part[
        (solved_part['calculation_time_ms'] >= time_range[0]) & (solved_part['calculation_time_ms'] <= time_range[1])]
    filtered_df = pd.concat([solved_part, failed_part])

    if solved_filter == "解決済みのみ":
        filtered_df = filtered_df[filtered_df['solved'] == True]
    elif solved_filter == "失敗のみ":
        filtered_df = filtered_df[filtered_df['solved'] == False]

    display_columns = ['size', 'problem_id', 'solved', 'num_moves', 'calculation_time_ms', 'nodes_explored',
                       'search_efficiency', 'diff_from_estimated', 'composite_score', 'error']
    available_columns = [col for col in display_columns if col in filtered_df.columns]
    st.dataframe(filtered_df[available_columns].fillna('N/A').round(4), use_container_width=True, height=400)
    st.write(f"表示データ数: {len(filtered_df)} / {len(df)}")


def individual_analysis_page(csv_files):
    """【新規】個別分析ページのUIとロジック"""
    st.sidebar.title("⚙️ 操作パネル")

    # --- データ選択 ---
    if not csv_files:
        st.warning("⚠️ ベンチマーク結果が見つかりません。")
        st.info("下の「ベンチマーク実行」ボタンを押して、まずベンチマークを実行してください。")
    else:
        # 【要件3】 results内のデータを選んで表示
        file_options = {Path(f).stem: f for f in csv_files}
        selected_option = st.sidebar.selectbox("分析するデータを選択", options=list(file_options.keys()))
        selected_csv_path = file_options[selected_option]

        df, summary = load_benchmark_results(selected_csv_path)

        if df.empty:
            st.error("選択されたデータの読み込みに失敗しました。")
            return

        st.sidebar.success(f"✅ データ読み込み完了: {selected_option}")
        st.sidebar.info(f"問題数: {len(df)}")
        st.sidebar.info(f"サイズ範囲: {df['size'].min()}x{df['size'].min()} - {df['size'].max()}x{df['size'].max()}")

        if summary:
            display_metrics_overview(summary)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏱️ 時間性能", "🔍 探索性能", "🎯 解品質", "📈 総合分析", "📋 詳細データ"])
        with tab1:
            plot_time_performance(df[df['solved'] == True])
        with tab2:
            plot_search_performance(df[df['solved'] == True])
        with tab3:
            plot_solution_quality(df[df['solved'] == True])
        with tab4:
            plot_comprehensive_analysis(df[df['solved'] == True])
        with tab5:
            display_detailed_table(df)

    # --- ベンチマーク実行 ---
    with st.sidebar.expander("ベンチマーク実行", expanded=not csv_files):
        executable_path = st.text_input("実行ファイルパス", value="./astar_manhattan")
        problems_dir = st.text_input("問題ディレクトリ", value="problems")

        min_avail, max_avail = 4, 24
        problem_path_obj = Path(problems_dir)
        if problem_path_obj.exists() and problem_path_obj.is_dir():
            available_sizes = [int(d.name.split('x')[0]) for d in problem_path_obj.iterdir() if
                               d.is_dir() and d.name.split('x')[0].isdigit()]
            if available_sizes:
                min_avail, max_avail = min(available_sizes), max(available_sizes)

        col1, col2 = st.columns(2)
        min_size = col1.number_input("最小サイズ", min_value=min_avail, max_value=max_avail, value=min_avail, step=1)
        max_size = col2.number_input("最大サイズ", min_value=min_avail, max_value=max_avail, value=max_avail, step=1)
        if min_size > max_size: st.error("最小サイズが最大サイズを超えています。")

        problems_per_size = st.number_input("各サイズで実行する問題数", min_value=1, value=5, step=1)

        if st.button("🚀 ベンチマーク実行", type="primary", disabled=(min_size > max_size)):
            run_benchmark_with_config(executable_path, problems_dir, problems_per_size, min_size, max_size)


def comparison_page(csv_files):
    """【新規】比較分析ページのUIとロジック"""
    st.title("📊 パフォーマンス比較")

    if len(csv_files) < 2:
        st.warning("比較するには少なくとも2つのベンチマーク結果が必要です。")
        st.info("「個別分析」ページでベンチマークを2回以上実行してください。")
        return

    file_options = {Path(f).stem: f for f in csv_files}

    col1, col2 = st.columns(2)
    with col1:
        selection_a = st.selectbox("比較対象 A", options=list(file_options.keys()), index=0, key="comp_a")
    with col2:
        selection_b = st.selectbox("比較対象 B", options=list(file_options.keys()), index=1, key="comp_b")

    if selection_a == selection_b:
        st.error("同じデータは比較できません。異なるデータを選択してください。")
        return

    # データの読み込みと準備
    df_a, summary_a = load_benchmark_results(file_options[selection_a])
    df_b, summary_b = load_benchmark_results(file_options[selection_b])

    # --- 概要比較 ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        display_metrics_overview(summary_a, header=f"概要: {selection_a}")
    with col2:
        display_metrics_overview(summary_b, header=f"概要: {selection_b}")

    # --- グラフ比較 ---
    st.markdown("---")
    st.header("📈 グラフでの性能比較")

    # 時間性能
    st.subheader("⏱️ サイズ別平均計算時間")
    time_a = df_a[df_a['solved'] == True].groupby('size')['calculation_time_ms'].mean().reset_index()
    time_b = df_b[df_b['solved'] == True].groupby('size')['calculation_time_ms'].mean().reset_index()
    fig_time = go.Figure()
    fig_time.add_trace(
        go.Scatter(x=time_a['size'], y=time_a['calculation_time_ms'], mode='lines+markers', name=selection_a))
    fig_time.add_trace(
        go.Scatter(x=time_b['size'], y=time_b['calculation_time_ms'], mode='lines+markers', name=selection_b))
    fig_time.update_layout(xaxis_title="パズルサイズ", yaxis_title="平均計算時間 (ms)")
    st.plotly_chart(fig_time, use_container_width=True)

    # 探索性能
    st.subheader("🔍 サイズ別平均探索ノード数")
    nodes_a = df_a[df_a['solved'] == True].groupby('size')['nodes_explored'].mean().reset_index()
    nodes_b = df_b[df_b['solved'] == True].groupby('size')['nodes_explored'].mean().reset_index()
    fig_nodes = go.Figure()
    fig_nodes.add_trace(
        go.Scatter(x=nodes_a['size'], y=nodes_a['nodes_explored'], mode='lines+markers', name=selection_a))
    fig_nodes.add_trace(
        go.Scatter(x=nodes_b['size'], y=nodes_b['nodes_explored'], mode='lines+markers', name=selection_b))
    fig_nodes.update_layout(xaxis_title="パズルサイズ", yaxis_title="平均探索ノード数")
    st.plotly_chart(fig_nodes, use_container_width=True)


def main():
    """【変更】メイン関数: ページ選択とディスパッチ"""
    st.sidebar.title("📄 ページ選択")
    page = st.sidebar.radio("表示するページを選択", ["個別分析", "比較分析"], label_visibility="collapsed")

    st.sidebar.markdown("---")

    # 結果ディレクトリの確認とファイルリストの取得
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")), reverse=True)

    st.title("🔍 高専プロコン競技 ベンチマーク結果ダッシュボード")

    if page == "個別分析":
        individual_analysis_page(csv_files)
    elif page == "比較分析":
        comparison_page(csv_files)

    st.markdown("---")
    st.markdown("🚀 A*アルゴリズムベンチマークシステム | Built with Streamlit")


if __name__ == "__main__":
    main()