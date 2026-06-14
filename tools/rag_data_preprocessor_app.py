# a011_make_rag_data_customer.py
# カスタマーサポートFAQデータのRAG前処理（モデル選択機能付き・独立版）
# streamlit run a011_make_rag_data_customer.py --server.port=8501

import streamlit as st
import pandas as pd
import re
import io
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from functools import wraps

# 基本ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# 設定管理（独立実装）
# ==================================================
class AppConfig:
    """アプリケーション設定（独立実装）"""

    # 利用可能なモデル
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-audio-preview",
        "gpt-4o-mini-audio-preview",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4",
        "o4-mini"
    ]

    DEFAULT_MODEL = "gpt-5-mini"

    # モデル料金（1000トークンあたりのドル）
    MODEL_PRICING = {
        "gpt-4o"                   : {"input": 0.005, "output": 0.015},
        "gpt-4o-mini"              : {"input": 0.00015, "output": 0.0006},
        "gpt-4o-audio-preview"     : {"input": 0.01, "output": 0.02},
        "gpt-4o-mini-audio-preview": {"input": 0.00025, "output": 0.001},
        "gpt-4.1"                  : {"input": 0.0025, "output": 0.01},
        "gpt-4.1-mini"             : {"input": 0.0001, "output": 0.0004},
        "o1"                       : {"input": 0.015, "output": 0.06},
        "o1-mini"                  : {"input": 0.003, "output": 0.012},
        "o3"                       : {"input": 0.03, "output": 0.12},
        "o3-mini"                  : {"input": 0.006, "output": 0.024},
        "o4"                       : {"input": 0.05, "output": 0.20},
        "o4-mini"                  : {"input": 0.01, "output": 0.04},
    }

    # モデル制限
    MODEL_LIMITS = {
        "gpt-4o"      : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4o-mini" : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1"     : {"max_tokens": 128000, "max_output": 4096},
        "gpt-4.1-mini": {"max_tokens": 128000, "max_output": 4096},
        "o1"          : {"max_tokens": 128000, "max_output": 32768},
        "o1-mini"     : {"max_tokens": 128000, "max_output": 65536},
        "o3"          : {"max_tokens": 200000, "max_output": 100000},
        "o3-mini"     : {"max_tokens": 200000, "max_output": 100000},
        "o4"          : {"max_tokens": 256000, "max_output": 128000},
        "o4-mini"     : {"max_tokens": 256000, "max_output": 128000},
    }

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルの制限を取得"""
        return cls.MODEL_LIMITS.get(model, {"max_tokens": 128000, "max_output": 4096})

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """モデルの料金を取得"""
        return cls.MODEL_PRICING.get(model, {"input": 0.00015, "output": 0.0006})


# ==================================================
# RAG設定（独立実装）
# ==================================================
class RAGConfig:
    """RAGデータ前処理の設定"""

    DATASET_CONFIGS = {
        "customer_support_faq": {
            "name"            : "カスタマーサポート・FAQ",
            "icon"            : "💬",
            "required_columns": ["question", "answer"],
            "description"     : "カスタマーサポートFAQデータセット",
            "combine_template": "{question} {answer}"
        }
    }

    @classmethod
    def get_config(cls, dataset_type: str) -> Dict[str, Any]:
        """データセット設定の取得"""
        return cls.DATASET_CONFIGS.get(dataset_type, {
            "name"            : "未知のデータセット",
            "icon"            : "❓",
            "required_columns": [],
            "description"     : "未知のデータセット",
            "combine_template": "{}"
        })


# ==================================================
# トークン管理（独立実装）
# ==================================================
class TokenManager:
    """トークン数の管理（簡易版）"""

    @staticmethod
    def count_tokens(text: str, model: str = None) -> int:
        """テキストのトークン数をカウント（簡易推定）"""
        if not text:
            return 0
        # 簡易推定: 1文字 = 0.5トークン（日本語）、1単語 = 1トークン（英語）
        japanese_chars = len([c for c in text if ord(c) > 127])
        english_chars = len(text) - japanese_chars
        return int(japanese_chars * 0.5 + english_chars * 0.25)

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """API使用コストの推定"""
        pricing = AppConfig.get_model_pricing(model)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


# ==================================================
# UI関数（独立実装）
# ==================================================
def select_model(key: str = "model_selection") -> str:
    """モデル選択UI"""
    models = AppConfig.AVAILABLE_MODELS
    default_model = AppConfig.DEFAULT_MODEL

    try:
        default_index = models.index(default_model)
    except ValueError:
        default_index = 0

    selected = st.sidebar.selectbox(
        "🤖 モデルを選択",
        models,
        index=default_index,
        key=key,
        help="利用するOpenAIモデルを選択してください"
    )

    return selected


def show_model_info(selected_model: str) -> None:
    """選択されたモデルの情報を表示"""
    try:
        limits = AppConfig.get_model_limits(selected_model)
        pricing = AppConfig.get_model_pricing(selected_model)

        with st.sidebar.expander("📊 選択モデル情報", expanded=False):
            # 基本情報
            col1, col2 = st.columns(2)
            with col1:
                st.write("**最大入力**")
                st.write(f"{limits['max_tokens']:,}")
            with col2:
                st.write("**最大出力**")
                st.write(f"{limits['max_output']:,}")

            # 料金情報
            st.write("**料金（1000トークン）**")
            st.write(f"- 入力: ${pricing['input']:.5f}")
            st.write(f"- 出力: ${pricing['output']:.5f}")

            # モデル特性
            if selected_model.startswith("o"):
                st.info("🧠 推論特化モデル")
                st.write("高度な推論タスクに最適化")
            elif "audio" in selected_model:
                st.info("🎵 音声対応モデル")
                st.write("音声入力・出力に対応")
            elif "gpt-4o" in selected_model:
                st.info("👁️ マルチモーダルモデル")
                st.write("テキスト・画像の理解が可能")
            else:
                st.info("💬 標準対話モデル")
                st.write("一般的な対話・テキスト処理")

            # RAG用途での推奨度
            st.write("**RAG用途推奨度**")
            if selected_model in ["gpt-4o-mini", "gpt-4.1-mini"]:
                st.success("✅ 最適（コスト効率良好）")
            elif selected_model in ["gpt-4o", "gpt-4.1"]:
                st.info("💡 高品質（コスト高）")
            elif selected_model.startswith("o"):
                st.warning("⚠️ 推論特化（RAG用途には過剰）")
            else:
                st.info("💬 標準的な性能")

    except Exception as e:
        logger.error(f"モデル情報表示エラー: {e}")
        st.sidebar.error("モデル情報の取得に失敗しました")


# ==================================================
# デコレータ（独立実装）
# ==================================================
def error_handler(func):
    """エラーハンドリングデコレータ"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"エラーが発生しました: {str(e)}")
            return None

    return wrapper


# ==================================================
# データ処理関数（独立実装）
# ==================================================
def clean_text(text: str) -> str:
    """テキストのクレンジング処理"""
    if pd.isna(text) or text == "":
        return ""

    # 改行を空白に置換
    text = str(text).replace('\n', ' ').replace('\r', ' ')

    # 連続した空白を1つの空白にまとめる
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を除去
    text = text.strip()

    # 引用符の正規化
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text


def combine_columns(row: pd.Series, dataset_type: str = "customer_support_faq") -> str:
    """複数列を結合して1つのテキストにする"""
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    # 各列からテキストを抽出・クレンジング
    cleaned_values = {}
    for col in required_columns:
        value = row.get(col, '')
        cleaned_values[col.lower()] = clean_text(str(value))

    # 結合
    combined = " ".join(cleaned_values.values())
    return combined.strip()


def validate_data(df: pd.DataFrame, dataset_type: str = None) -> List[str]:
    """データの検証"""
    issues = []

    # 基本統計
    issues.append(f"総行数: {len(df)}")
    issues.append(f"総列数: {len(df.columns)}")

    # 必須列の確認
    if dataset_type:
        config_data = RAGConfig.get_config(dataset_type)
        required_columns = config_data["required_columns"]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"⚠️ 必須列が不足: {missing_columns}")
        else:
            issues.append(f"✅ 必須列確認済み: {required_columns}")

    # 各列の空値確認
    for col in df.columns:
        empty_count = df[col].isna().sum() + (df[col] == '').sum()
        if empty_count > 0:
            percentage = (empty_count / len(df)) * 100
            issues.append(f"{col}列: 空値 {empty_count}個 ({percentage:.1f}%)")

    # 重複行の確認
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"⚠️ 重複行: {duplicate_count}個")
    else:
        issues.append("✅ 重複行なし")

    return issues


def load_dataset(uploaded_file, dataset_type: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """データセットの読み込みと基本検証"""
    try:
        # CSVファイルの読み込み
        df = pd.read_csv(uploaded_file)

        # 基本検証
        validation_results = validate_data(df, dataset_type)

        logger.info(f"データセット読み込み完了: {len(df)}行, {len(df.columns)}列")
        return df, validation_results

    except Exception as e:
        logger.error(f"データセット読み込みエラー: {e}")
        raise


def process_rag_data(df: pd.DataFrame, dataset_type: str, combine_columns_option: bool = True) -> pd.DataFrame:
    """RAGデータの前処理を実行"""
    # 基本的な前処理
    df_processed = df.copy()

    # 重複行の除去
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    duplicates_removed = initial_rows - len(df_processed)

    # 空行の除去
    df_processed = df_processed.dropna(how='all')
    empty_rows_removed = initial_rows - duplicates_removed - len(df_processed)

    # インデックスのリセット
    df_processed = df_processed.reset_index(drop=True)

    logger.info(f"前処理完了: 重複除去={duplicates_removed}行, 空行除去={empty_rows_removed}行")

    # 各列のクレンジング
    config_data = RAGConfig.get_config(dataset_type)
    required_columns = config_data["required_columns"]

    for col in required_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(clean_text)

    # 列の結合（オプション）
    if combine_columns_option:
        df_processed['Combined_Text'] = df_processed.apply(
            lambda row: combine_columns(row, dataset_type),
            axis=1
        )

    return df_processed


def create_download_data(df: pd.DataFrame, include_combined: bool = True, dataset_type: str = None) -> Tuple[
    str, Optional[str]]:
    """ダウンロード用データの作成"""
    try:
        # CSVデータの作成
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = csv_buffer.getvalue()

        # 結合テキストデータの作成
        text_data = None
        if include_combined and 'Combined_Text' in df.columns:
            text_data = df['Combined_Text'].to_string(index=False)

        return csv_data, text_data

    except Exception as e:
        logger.error(f"create_download_data エラー: {e}")
        raise


def display_statistics(df_original: pd.DataFrame, df_processed: pd.DataFrame, dataset_type: str = None) -> None:
    """処理前後の統計情報を表示"""
    st.subheader("📊 統計情報")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("元の行数", len(df_original))
    with col2:
        st.metric("処理後の行数", len(df_processed))
    with col3:
        removed_rows = len(df_original) - len(df_processed)
        st.metric("除去された行数", removed_rows)

    # 結合テキストの分析
    if 'Combined_Text' in df_processed.columns:
        st.subheader("📝 結合後テキスト分析")
        text_lengths = df_processed['Combined_Text'].str.len()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均文字数", f"{text_lengths.mean():.0f}")
        with col2:
            st.metric("最大文字数", text_lengths.max())
        with col3:
            st.metric("最小文字数", text_lengths.min())


def estimate_token_usage(df_processed: pd.DataFrame, selected_model: str) -> None:
    """処理済みデータのトークン使用量推定"""
    try:
        if 'Combined_Text' in df_processed.columns:
            # サンプルテキストでトークン数を推定
            sample_texts = df_processed['Combined_Text'].head(10).tolist()
            total_chars = df_processed['Combined_Text'].str.len().sum()

            if sample_texts:
                sample_text = " ".join(sample_texts)
                sample_tokens = TokenManager.count_tokens(sample_text, selected_model)
                sample_chars = len(sample_text)

                if sample_chars > 0:
                    # 全体のトークン数を推定
                    estimated_total_tokens = int((total_chars / sample_chars) * sample_tokens)

                    with st.expander("🔢 トークン使用量推定", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("推定総トークン数", f"{estimated_total_tokens:,}")
                        with col2:
                            avg_tokens_per_record = estimated_total_tokens / len(df_processed)
                            st.metric("平均トークン/レコード", f"{avg_tokens_per_record:.0f}")
                        with col3:
                            # embedding用のコスト推定（参考値）
                            embedding_cost = (estimated_total_tokens / 1000) * 0.0001
                            st.metric("推定embedding費用", f"${embedding_cost:.4f}")

                        st.info(f"💡 選択モデル「{selected_model}」での推定値")

    except Exception as e:
        logger.error(f"トークン使用量推定エラー: {e}")
        st.error("トークン使用量の推定に失敗しました")


# ==================================================
# ファイル保存関数
# ==================================================
def create_output_directory() -> Path:
    """OUTPUTディレクトリの作成"""
    try:
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)

        # 書き込み権限のテスト
        test_file = output_dir / ".test_write"
        try:
            test_file.write_text("test", encoding='utf-8')
            if test_file.exists():
                test_file.unlink()
                logger.info("書き込み権限テスト: 成功")
        except Exception as e:
            raise PermissionError(f"書き込み権限テストに失敗: {e}")

        logger.info(f"OUTPUTディレクトリ準備完了: {output_dir.absolute()}")
        return output_dir

    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {e}")
        raise


def save_files_to_output(df_processed, dataset_type: str, csv_data: str, text_data: str = None) -> Dict[str, str]:
    """処理済みデータをOUTPUTフォルダに保存"""
    try:
        output_dir = create_output_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # CSVファイルの保存
        csv_filename = f"preprocessed_{dataset_type}_{len(df_processed)}rows_{timestamp}.csv"
        csv_path = output_dir / csv_filename

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_data)

        if csv_path.exists():
            saved_files['csv'] = str(csv_path)
            logger.info(f"CSVファイル保存完了: {csv_path}")

        # テキストファイルの保存
        if text_data and len(text_data.strip()) > 0:
            txt_filename = f"{dataset_type}.txt"
            txt_path = output_dir / txt_filename

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_data)

            if txt_path.exists():
                saved_files['txt'] = str(txt_path)
                logger.info(f"テキストファイル保存完了: {txt_path}")

        # メタデータファイルの保存
        metadata = {
            "dataset_type"        : dataset_type,
            "processed_rows"      : len(df_processed),
            "processing_timestamp": timestamp,
            "created_at"          : datetime.now().isoformat(),
            "files_created"       : list(saved_files.keys())
        }

        metadata_filename = f"metadata_{dataset_type}_{timestamp}.json"
        metadata_path = output_dir / metadata_filename

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if metadata_path.exists():
            saved_files['metadata'] = str(metadata_path)
            logger.info(f"メタデータファイル保存完了: {metadata_path}")

        return saved_files

    except Exception as e:
        logger.error(f"ファイル保存エラー: {e}")
        raise


# ==================================================
# カスタマーサポートFAQ特有の処理関数
# ==================================================
def validate_customer_support_data_specific(df) -> List[str]:
    """カスタマーサポートFAQデータ特有の検証"""
    support_issues = []

    # サポート関連用語の存在確認
    support_keywords = [
        '問題', '解決', 'トラブル', 'エラー', 'サポート', 'ヘルプ', '対応',
        'problem', 'issue', 'error', 'help', 'support', 'solution', 'troubleshoot'
    ]

    if 'question' in df.columns:
        questions_with_support_terms = 0
        for _, row in df.iterrows():
            question_text = str(row.get('question', '')).lower()
            if any(keyword in question_text for keyword in support_keywords):
                questions_with_support_terms += 1

        support_ratio = (questions_with_support_terms / len(df)) * 100
        support_issues.append(f"サポート関連用語を含む質問: {questions_with_support_terms}件 ({support_ratio:.1f}%)")

    # 回答の長さ分析
    if 'answer' in df.columns:
        answer_lengths = df['answer'].astype(str).str.len()
        avg_answer_length = answer_lengths.mean()
        if avg_answer_length < 50:
            support_issues.append(f"⚠️ 平均回答長が短い可能性: {avg_answer_length:.0f}文字")
        else:
            support_issues.append(f"✅ 適切な回答長: 平均{avg_answer_length:.0f}文字")

    return support_issues


def show_usage_instructions(dataset_type: str = "customer_support_faq") -> None:
    """使用方法の説明を表示"""
    st.markdown("---")
    st.subheader("📖 使用方法")
    st.markdown("""
    1. **モデル選択**: サイドバーでRAG用途に適したモデルを選択
    2. **CSVファイルをアップロード**: question, answer 列を含むCSVファイルを選択
    3. **前処理を実行**: 以下の処理が自動で実行されます：
       - 改行の除去
       - 連続した空白の統一
       - 重複行の除去
       - 空行の除去
       - 引用符の正規化
    4. **複数列結合**: Vector Store/RAG用に最適化された自然な文章として結合
    5. **トークン使用量確認**: 選択モデルでのトークン数とコストを推定
    6. **ダウンロード**: 前処理済みデータをCSV形式でダウンロード

    **Vector Store用最適化:**
    - 自然な文章として結合（ラベル文字列なし）
    - OpenAI embeddingモデルに最適化
    - 検索性能が向上
    """)


# ==================================================
# メイン処理関数
# ==================================================
@error_handler
def main():
    """メイン処理関数"""

    # データセットタイプの設定
    DATASET_TYPE = "customer_support_faq"

    # ページ設定
    try:
        st.set_page_config(
            page_title="カスタマーサポートFAQデータ前処理",
            page_icon="💬",
            layout="wide"
        )
    except st.errors.StreamlitAPIException:
        pass

    st.title("💬 カスタマーサポートFAQデータ前処理アプリ")
    st.markdown("---")

    # =================================================
    # モデル選択機能
    # =================================================
    st.sidebar.title("💬 カスタマーサポートFAQ")
    st.sidebar.markdown("---")

    # モデル選択
    selected_model = select_model(key="rag_model_selection")

    # 選択されたモデル情報を表示
    show_model_info(selected_model)

    st.sidebar.markdown("---")
    # =================================================

    # サイドバー設定
    st.sidebar.header("前処理設定")
    combine_columns_option = st.sidebar.checkbox(
        "複数列を結合する（Vector Store用）",
        value=True,
        help="複数列を結合してRAG用テキストを作成"
    )
    show_validation = st.sidebar.checkbox(
        "データ検証を表示",
        value=True,
        help="データの品質検証結果を表示"
    )

    # カスタマーサポートデータ特有の設定
    with st.sidebar.expander("💬 サポートデータ設定", expanded=False):
        st.checkbox(
            "書式を保護",
            value=True,
            help="回答内の重要な書式を保護"
        )
        st.checkbox(
            "質問を正規化",
            value=True,
            help="質問文の表記ゆれを統一"
        )

    # ファイルアップロード
    st.subheader("📁 データファイルのアップロード")

    # 選択されたモデル情報を表示（メインエリア）
    with st.expander("📊 選択中のモデル情報", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🤖 選択モデル: **{selected_model}**")
        with col2:
            limits = AppConfig.get_model_limits(selected_model)
            st.info(f"📏 最大トークン: **{limits['max_tokens']:,}**")

    uploaded_file = st.file_uploader(
        "カスタマーサポートFAQデータのCSVファイルをアップロードしてください",
        type=['csv'],
        help="question, answer の2列を含むCSVファイル"
    )

    if uploaded_file is not None:
        try:
            # ファイル情報の確認
            st.info(f"📁 ファイル: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

            # セッション状態でファイル処理状況を管理
            file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"

            # ファイルが変更された場合は再読み込み
            if st.session_state.get('current_file_key') != file_key:
                with st.spinner("ファイルを読み込み中..."):
                    df, validation_results = load_dataset(uploaded_file, DATASET_TYPE)

                # セッション状態に保存
                st.session_state['current_file_key'] = file_key
                st.session_state['original_df'] = df
                st.session_state['validation_results'] = validation_results
                st.session_state['original_rows'] = len(df)
                st.session_state['file_processed'] = False

                logger.info(f"新しいファイルを読み込み: {len(df)}行")
            else:
                # セッション状態から取得
                df = st.session_state['original_df']
                validation_results = st.session_state['validation_results']
                logger.info(f"セッション状態からファイルを取得: {len(df)}行")

            st.success(f"ファイルが正常に読み込まれました。行数: {len(df)}")

            # 元データの表示
            st.subheader("📋 元データプレビュー")
            st.dataframe(df.head(10))

            # データ検証結果の表示
            if show_validation:
                st.subheader("🔍 データ検証")

                # 基本検証結果
                for issue in validation_results:
                    st.info(issue)

                # カスタマーサポートデータ特有の検証
                support_issues = validate_customer_support_data_specific(df)
                if support_issues:
                    st.write("**カスタマーサポートデータ特有の分析:**")
                    for issue in support_issues:
                        st.info(issue)

            # 前処理実行
            st.subheader("⚙️ 前処理実行")

            if st.button("前処理を実行", type="primary", key="process_button"):
                try:
                    with st.spinner("前処理中..."):
                        # RAGデータの前処理
                        df_processed = process_rag_data(
                            df.copy(),
                            DATASET_TYPE,
                            combine_columns_option
                        )

                    st.success("前処理が完了しました！")

                    # セッション状態に処理済みデータを保存
                    st.session_state['processed_df'] = df_processed
                    st.session_state['file_processed'] = True

                    # 前処理後のデータ表示
                    st.subheader("✅ 前処理後のデータプレビュー")
                    st.dataframe(df_processed.head(10))

                    # 統計情報の表示
                    display_statistics(df, df_processed, DATASET_TYPE)

                    # 選択されたモデルでのトークン使用量推定
                    estimate_token_usage(df_processed, selected_model)

                    # カスタマーサポートデータ特有の後処理分析
                    if 'Combined_Text' in df_processed.columns:
                        st.subheader("💬 カスタマーサポートデータ特有の分析")

                        # 結合テキストのサポート用語分析
                        combined_texts = df_processed['Combined_Text']
                        support_keywords = ['問題', 'エラー', 'トラブル', 'サポート', 'ヘルプ']

                        keyword_counts = {}
                        for keyword in support_keywords:
                            count = combined_texts.str.contains(keyword, case=False).sum()
                            keyword_counts[keyword] = count

                        if keyword_counts:
                            st.write("**サポート関連用語の出現頻度:**")
                            for keyword, count in keyword_counts.items():
                                percentage = (count / len(df_processed)) * 100
                                st.write(f"- {keyword}: {count}件 ({percentage:.1f}%)")

                        # 質問の長さ分布
                        if 'question' in df_processed.columns:
                            question_lengths = df_processed['question'].str.len()
                            st.write("**質問の長さ統計:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("平均質問長", f"{question_lengths.mean():.0f}文字")
                            with col2:
                                st.metric("最長質問", f"{question_lengths.max()}文字")
                            with col3:
                                st.metric("最短質問", f"{question_lengths.min()}文字")

                    logger.info(f"カスタマーサポートFAQデータ処理完了: {len(df)} → {len(df_processed)}行")

                except Exception as process_error:
                    st.error(f"前処理エラー: {str(process_error)}")
                    logger.error(f"前処理エラー: {process_error}")

            # 処理済みデータがある場合のみダウンロード・保存セクションを表示
            if st.session_state.get('file_processed', False) and 'processed_df' in st.session_state:
                df_processed = st.session_state['processed_df']

                # ダウンロード・保存セクション
                st.subheader("💾 ダウンロード・保存")

                # ダウンロード用データの作成（キャッシュ）
                if 'download_data' not in st.session_state or st.session_state.get('download_data_key') != file_key:
                    with st.spinner("ダウンロード用データを準備中..."):
                        csv_data, text_data = create_download_data(
                            df_processed,
                            combine_columns_option,
                            DATASET_TYPE
                        )
                        st.session_state['download_data'] = (csv_data, text_data)
                        st.session_state['download_data_key'] = file_key
                else:
                    csv_data, text_data = st.session_state['download_data']

                # ブラウザダウンロード
                st.write("**📥 ブラウザダウンロード**")
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📊 CSV形式でダウンロード",
                        data=csv_data,
                        file_name=f"preprocessed_{DATASET_TYPE}_{len(df_processed)}rows.csv",
                        mime="text/csv",
                        help="前処理済みのカスタマーサポートFAQデータをCSV形式でダウンロード"
                    )

                with col2:
                    if text_data:
                        st.download_button(
                            label="📝 テキスト形式でダウンロード",
                            data=text_data,
                            file_name="customer_support_faq.txt",
                            mime="text/plain",
                            help="Vector Store/RAG用に最適化された結合テキスト"
                        )

                # ローカル保存
                st.write("**💾 ローカルファイル保存（OUTPUTフォルダ）**")

                if st.button("🔄 OUTPUTフォルダに保存", type="secondary", key="save_button"):
                    try:
                        with st.spinner("ファイル保存中..."):
                            saved_files = save_files_to_output(
                                df_processed,
                                DATASET_TYPE,
                                csv_data,
                                text_data
                            )

                        if saved_files:
                            st.success("✅ ファイル保存完了！")

                            # 保存されたファイル一覧を表示
                            with st.expander("📂 保存されたファイル一覧", expanded=True):
                                for file_type, file_path in saved_files.items():
                                    if Path(file_path).exists():
                                        file_size = Path(file_path).stat().st_size
                                        st.write(f"**{file_type.upper()}**: `{file_path}` ({file_size:,} bytes) ✅")
                                    else:
                                        st.write(f"**{file_type.upper()}**: `{file_path}` ❌ ファイルが見つかりません")

                                # OUTPUTフォルダの場所を表示
                                output_path = Path("OUTPUT").resolve()
                                st.write(f"**保存場所**: `{output_path}`")
                                file_count = len(list(output_path.glob("*")))
                                st.write(f"**フォルダ内ファイル数**: {file_count}個")
                        else:
                            st.error("❌ ファイル保存に失敗しました")

                    except Exception as save_error:
                        st.error(f"❌ ファイル保存エラー: {str(save_error)}")
                        logger.error(f"保存エラー: {save_error}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"ファイル読み込みエラー: {e}")

    else:
        st.info("👆 CSVファイルをアップロードしてください")

    # 使用方法の説明
    show_usage_instructions(DATASET_TYPE)

    # セッション状態の表示（デバッグ用）
    if st.sidebar.checkbox("🔧 セッション状態を表示", value=False):
        with st.sidebar.expander("セッション状態", expanded=False):
            st.write(f"**選択モデル**: {selected_model}")
            st.write(f"**ファイル処理済み**: {st.session_state.get('file_processed', False)}")

            if 'original_df' in st.session_state:
                df = st.session_state['original_df']
                st.write(f"**元データ**: {len(df)}行")

            if 'processed_df' in st.session_state:
                df_processed = st.session_state['processed_df']
                st.write(f"**処理済みデータ**: {len(df_processed)}行")


# ==================================================
# アプリケーション実行
# ==================================================
if __name__ == "__main__":
    main()

# 実行コマンド:
# streamlit run a011_make_rag_data_customer.py --server.port=8501
