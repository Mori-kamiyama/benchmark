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


def get_timeout_config_path(executable_path: str) -> str:
    """実行ファイルパスから設定ファイルパスを生成する"""
    exec_path = Path(executable_path)
    config_filename = f"{exec_path.stem}_config.json"
    return str(exec_path.parent / config_filename)


def load_timeout_from_config(config_path: str) -> int:
    """設定ファイルからタイムアウト値を読み込む"""
    default_timeout = 60
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        timeout = config_data.get("timeout")
        if isinstance(timeout, int) and 10 <= timeout <= 300:
            return timeout
        else:
            # Log or warn about invalid timeout value, then return default
            st.warning(f"設定ファイル '{config_path}' のタイムアウト値が無効です (値: {timeout})。デフォルト値 {default_timeout}秒 を使用します。")
            return default_timeout
    except FileNotFoundError:
        # Log or inform that config file was not found, using default
        # st.info(f"設定ファイル '{config_path}' が見つかりません。デフォルトタイムアウト {default_timeout}秒 を使用します。")
        return default_timeout
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Log or warn about error in config file
        st.warning(f"設定ファイル '{config_path}' の読み込み中にエラーが発生しました: {e}。デフォルト値 {default_timeout}秒 を使用します。")
        return default_timeout


def save_timeout_to_config(config_path: str, timeout_value: int):
    """設定ファイルにタイムアウト値を保存する"""
    try:
        # Ensure the directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({"timeout": timeout_value}, f, indent=2)
        st.toast(f"タイムアウト設定 ({timeout_value}秒) を '{config_path}' に保存しました。", icon="✅")
    except IOError as e:
        st.error(f"設定ファイル '{config_path}' の保存中にエラーが発生しました: {e}")
    except Exception as e:
        st.error(f"タイムアウト設定の保存中に予期せぬエラーが発生しました: {e}")


def run_single_problem(executable_path, problem_path):
    """単一の問題を実行し、結果を返す"""
    config_path = get_timeout_config_path(executable_path)
    configured_timeout = load_timeout_from_config(config_path)

    cat_process = None
    astar_process = None
    try:
        start_time = time.perf_counter()
        cat_process = subprocess.Popen(["cat", problem_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        astar_process = subprocess.Popen([executable_path], stdin=cat_process.stdout, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
        cat_process.stdout.close()
        stdout, stderr = astar_process.communicate(timeout=configured_timeout)
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
        error_msg = f"タイムアウト ({configured_timeout}秒)"
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
            # 設定されたタイムアウト値を取得してサマリーに追加
            config_path = get_timeout_config_path(executable_path)
            configured_timeout = load_timeout_from_config(config_path)
            summary['timeout_seconds'] = configured_timeout # Use a more descriptive key like 'timeout_seconds'

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
        timeout_seconds = st.number_input("タイムアウト秒数 (10-300秒)", min_value=10, max_value=300, value=60, step=1, key="timeout_seconds_input")

        if st.button("🚀 ベンチマーク実行", type="primary", disabled=(min_size > max_size)):
            # タイムアウト設定を保存
            current_executable_path = executable_path # st.text_inputの現在の値を取得
            config_path = get_timeout_config_path(current_executable_path)
            save_timeout_to_config(config_path, timeout_seconds)

            # ベンチマーク実行
            run_benchmark_with_config(current_executable_path, problems_dir, problems_per_size, min_size, max_size)


# --- Helper Functions for Comparison Page ---

def load_and_merge_benchmark_data(
    selected_options: list[str],
    file_map: dict[str, str]
) -> tuple[pd.DataFrame | None, list[str], dict[str, dict]]:
    """Loads and merges data for selected benchmarks."""
    all_dfs = []
    all_df_names = []
    all_summaries = {}

    for option_stem in selected_options:
        filepath = file_map[option_stem]
        df, summary = load_benchmark_results(filepath) # Assuming load_benchmark_results is defined elsewhere
        if df is not None and not df.empty:
            # Ensure 'problem_id' is string for consistent merging
            if 'problem_id' in df.columns:
                df['problem_id'] = df['problem_id'].astype(str)
            if 'size' in df.columns: # Ensure size is int
                 df['size'] = df['size'].astype(int)

            all_dfs.append(df)
            all_df_names.append(option_stem)
            if summary is not None:
                all_summaries[option_stem] = summary
        else:
            st.warning(f"結果 '{option_stem}' のデータの読み込みに失敗したか、空です。スキップします。")

    if len(all_dfs) < 1: # Need at least one to start merging/displaying
        st.warning("有効なデータセットが1つも見つかりませんでした。")
        return None, [], {}

    if not all_dfs: # Should be caught by len(all_dfs) < 1 already
        return None, [], {}

    # Merge DataFrames
    merged_df = all_dfs[0].copy()
    # Ensure 'solved' column is boolean and handle potential non-boolean types before suffixing.
    # Also, make sure essential merge keys 'size', 'problem_id' are present.
    if not all({'size', 'problem_id'}.issubset(merged_df.columns)):
        st.error(f"最初のデータフレーム '{all_df_names[0]}' に 'size' または 'problem_id' がありません。マージを中止します。")
        return None, [], {}

    if f"solved" in merged_df.columns: # Original column name before suffix
        merged_df[f"solved"] = merged_df[f"solved"].astype('boolean')
    merged_df = merged_df.add_suffix(f"_{all_df_names[0]}")
    # Rename merge keys to not have suffix from the first df
    merged_df.rename(columns={
        f"size_{all_df_names[0]}": "size",
        f"problem_id_{all_df_names[0]}": "problem_id"
    }, inplace=True)


    for i in range(1, len(all_dfs)):
        current_df_name = all_df_names[i]
        current_df_to_merge = all_dfs[i].copy()

        if not all({'size', 'problem_id'}.issubset(current_df_to_merge.columns)):
            st.warning(f"データフレーム '{current_df_name}' に 'size' または 'problem_id' がありません。このデータセットをスキップします。")
            continue

        if f"solved" in current_df_to_merge.columns:
            current_df_to_merge[f"solved"] = current_df_to_merge[f"solved"].astype('boolean')
        current_df_to_merge = current_df_to_merge.add_suffix(f"_{current_df_name}")
        current_df_to_merge.rename(columns={
            f"size_{current_df_name}": "size",
            f"problem_id_{current_df_name}": "problem_id"
        }, inplace=True)

        merged_df = pd.merge(merged_df, current_df_to_merge, on=['size', 'problem_id'], how='outer')

    return merged_df, all_df_names, all_summaries


def calculate_benchmark_improvement_rates(
    merged_df: pd.DataFrame,
    baseline_name: str,
    all_df_names: list[str],
    metrics_to_compare: list[str]
) -> pd.DataFrame:
    """Calculates improvement rates for specified metrics against a baseline."""
    if not baseline_name or merged_df.empty:
        return merged_df

    # Make a copy to avoid SettingWithCopyWarning
    # merged_df_processed = merged_df.copy() # Already a copy if coming from load_and_merge
    # No, merged_df from load_and_merge is the actual merged_df, so copy is good.
    merged_df_processed = merged_df.copy()


    for df_name in all_df_names:
        if df_name == baseline_name:
            continue

        for metric in metrics_to_compare:
            baseline_col = f"{metric}_{baseline_name}"
            current_col = f"{metric}_{df_name}"
            rate_col_name = f"{metric}_improvement_rate_vs_{baseline_name}_for_{df_name}"

            if baseline_col in merged_df_processed.columns and current_col in merged_df_processed.columns:
                baseline_values = merged_df_processed[baseline_col].astype(float)
                current_values = merged_df_processed[current_col].astype(float)

                improvement_rate = (baseline_values - current_values) / baseline_values * 100

                improvement_rate[ (baseline_values == 0) & (current_values == 0) ] = 0
                improvement_rate[ (baseline_values == 0) & (current_values > 0) ] = -np.inf

                improvement_rate.replace([np.inf, -np.inf], np.nan, inplace=True)
                merged_df_processed[rate_col_name] = improvement_rate # Keep NaNs for now
            else:
                st.warning(f"レート計算に必要なカラムが見つかりません: {baseline_col} または {current_col}")

    return merged_df_processed


def create_comparison_summary_table_df(
    all_summaries: dict[str, dict],
    merged_df: pd.DataFrame,
    all_df_names: list[str],
    baseline_name: str | None,
    basic_metric_keys: dict[str, str], # e.g. {'total_problems': 'Total Problems'}
    rate_metric_specs: dict[str, str] # e.g. {'calculation_time_ms': 'Avg Calc Time Improvement %'}
) -> pd.DataFrame:
    """Creates the summary table DataFrame for comparison."""

    table_rows = []

    # Basic summary metrics
    for internal_key, display_name in basic_metric_keys.items():
        row_data = {'Metric': display_name}
        for df_name in all_df_names:
            summary = all_summaries.get(df_name, {})
            val = summary.get(internal_key)
            if pd.isna(val):
                row_data[df_name] = "N/A"
            elif isinstance(val, float):
                if 'time' in internal_key or 'ms' in internal_key : # crude check for time
                     row_data[df_name] = f"{val:.2f}"
                elif 'nodes' in internal_key :
                     row_data[df_name] = f"{val:,.0f}"
                elif 'moves' in internal_key:
                     row_data[df_name] = f"{val:.1f}"
                else:
                     row_data[df_name] = f"{val:.2f}" # Default float formatting
            else: # int or string
                row_data[df_name] = val
        table_rows.append(row_data)

    # Improvement rate metrics
    if baseline_name and not merged_df.empty:
        for internal_metric, display_name in rate_metric_specs.items():
            row_data = {'Metric': display_name}
            for df_name in all_df_names:
                if df_name == baseline_name:
                    row_data[df_name] = "Baseline"
                else:
                    rate_col = f"{internal_metric}_improvement_rate_vs_{baseline_name}_for_{df_name}"
                    if rate_col in merged_df.columns:
                        avg_rate = merged_df[rate_col].mean()
                        row_data[df_name] = f"{avg_rate:.1f}%" if pd.notna(avg_rate) else "N/A"
                    else:
                        row_data[df_name] = "N/A (no data)"
            table_rows.append(row_data)

    if not table_rows: # If no data at all
        return pd.DataFrame(columns=['Metric'] + all_df_names).set_index('Metric')

    summary_df = pd.DataFrame(table_rows)
    summary_df.set_index('Metric', inplace=True)
    return summary_df


def generate_comparison_line_charts_figures(
    merged_df: pd.DataFrame,
    all_df_names: list[str],
    chart_specs: list[dict] # Each dict: {'metric_key': 'col_name_part', 'title': 'Chart Title', 'yaxis_title': 'Y Axis'}
) -> list[go.Figure]:
    """Generates line charts for comparing benchmark performance metrics."""
    figures = []
    if merged_df.empty:
        return figures

    for spec in chart_specs:
        fig = go.Figure()
        st.subheader(spec['title']) # Display title before the chart

        for df_name in all_df_names:
            solved_col = f"solved_{df_name}"
            metric_col = f"{spec['metric_key']}_{df_name}"

            if solved_col in merged_df.columns and metric_col in merged_df.columns:
                # Ensure boolean type for solved_col before filtering
                # This might be redundant if already cast during merge, but safe.
                try: # Add try-except for astype if column is all NaN from an outer join
                    merged_df[solved_col] = merged_df[solved_col].astype('boolean')
                    solved_data_for_benchmark = merged_df[merged_df[solved_col].fillna(False)] # Treat NaN as False for solved
                except TypeError: # Handle cases where astype('boolean') fails e.g. mixed types not convertible
                     st.caption(f"注意: '{df_name}' の解決状態カラム ({solved_col}) の型変換に問題があり、グラフ生成をスキップする可能性があります。")
                     continue


                if not solved_data_for_benchmark.empty:
                    agg_data = solved_data_for_benchmark.groupby('size')[metric_col].mean().reset_index()
                    fig.add_trace(go.Scatter(x=agg_data['size'], y=agg_data[metric_col], mode='lines+markers', name=df_name))
                else:
                    st.caption(f"注意: '{df_name}' には '{spec['title']}' の解決済み問題データがありません。")
            else:
                st.caption(f"注意: '{df_name}' の '{spec['title']}' または解決状態カラム ({metric_col} or {solved_col}) がマージ後データに存在しません。")

        fig.update_layout(xaxis_title="パズルサイズ", yaxis_title=spec['yaxis_title'], legend_title_text='ベンチマーク')
        figures.append(fig)

    return figures


# --- Main Application Pages ---

def comparison_page(csv_files):
    """比較分析ページのUIとロジック"""
    st.title("📊 パフォーマンス比較")

    if len(csv_files) < 2:
        st.warning("比較するには少なくとも2つのベンチマーク結果が必要です。")
        st.info("「個別分析」ページでベンチマークを2回以上実行してください。")
        return

    file_options = {Path(f).stem: f for f in csv_files}
    selected_benchmark_options = st.multiselect(
        "比較するベンチマーク結果を選択 (2つ以上)",
        options=list(file_options.keys()),
        key="benchmark_multiselect"
    )

    if len(selected_benchmark_options) < 2:
        st.warning("比較するには少なくとも2つのベンチマーク結果を選択してください。")
        return

    # 1. Load and Merge Data
    merged_df, all_df_names, all_summaries = load_and_merge_benchmark_data(
        selected_benchmark_options, file_options
    )

    if merged_df is None or merged_df.empty:
        st.error("データの読み込みまたはマージに失敗しました。処理を続行できません。")
        return
    if len(all_df_names) < 2: # Check again after loading in case some failed
        st.warning("比較を実行するには、少なくとも2つの有効なデータセットを読み込む必要があります。")
        return


    # 2. Baseline Selection and Rate Calculation
    st.markdown("---")
    baseline_name = st.selectbox(
        "ベースラインにするベンチマークを選択",
        options=all_df_names, # Use names of successfully loaded DFs
        index=0,
        key="baseline_selection"
    )

    metrics_for_rate_calculation = ['calculation_time_ms', 'nodes_explored']
    if baseline_name:
        merged_df = calculate_benchmark_improvement_rates(
            merged_df, baseline_name, all_df_names, metrics_for_rate_calculation
        )

    # 3. Display Summary Table
    st.markdown("---")
    st.subheader("📊 総合サマリー比較")
    basic_summary_metrics = {
        'total_problems': 'Total Problems',
        'solved_problems': 'Solved Problems',
        'avg_calculation_time_ms': 'Avg Calculation Time (ms)',
        'avg_nodes_explored': 'Avg Nodes Explored',
        'avg_moves': 'Avg Moves'
    }
    rate_metrics_for_summary = {
        'calculation_time_ms': 'Avg Calc Time Improvement %',
        'nodes_explored': 'Avg Nodes Explored Impr. %'
    }
    summary_table_df = create_comparison_summary_table_df(
        all_summaries, merged_df, all_df_names, baseline_name,
        basic_summary_metrics, rate_metrics_for_summary
    )
    st.dataframe(summary_table_df, use_container_width=True)

    # 4. Display Charts
    st.markdown("---")
    st.header("📈 グラフでの性能比較")

    chart_specs = [
        {'metric_key': 'calculation_time_ms', 'title': '⏱️ サイズ別平均計算時間', 'yaxis_title': '平均計算時間 (ms)'},
        {'metric_key': 'nodes_explored', 'title': '🔍 サイズ別平均探索ノード数', 'yaxis_title': '平均探索ノード数'}
    ]

    if not merged_df.empty:
        chart_figures = generate_comparison_line_charts_figures(merged_df, all_df_names, chart_specs)
        for fig in chart_figures:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("マージされたデータが空のため、グラフを表示できません。")

    # Debug view
    if st.checkbox("マージされたDataFrameを表示 (デバッグ用)"):
        st.dataframe(merged_df)
        st.write(f"カラム名: {merged_df.columns.tolist()}")


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