#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
log_viewer_page.py - 未回答ログ閲覧ページ
=======================================
エージェントが回答できなかった質問のログを表示・管理する画面。
"""

import streamlit as st

from services.log_service import clear_unanswered_logs, load_unanswered_logs


def show_log_viewer_page():
    """画面: 未回答ログ閲覧"""
    st.title("📊 未回答ログ (Unanswered Logs)")
    st.caption("エージェントがRAG検索で回答を見つけられなかった質問の履歴です。ナレッジベースの拡充に活用してください。")

    # ログ読み込み
    df_logs = load_unanswered_logs()

    # サイドバー操作
    with st.sidebar:
        st.header("⚙️ ログ操作")
        
        if st.button("🔄 最新情報を取得"):
            st.rerun()
            
        if not df_logs.empty:
            st.divider()
            if st.button("🗑️ ログを全消去", type="primary"):
                clear_unanswered_logs()
                st.success("ログを消去しました。")
                st.rerun()

    # メイン表示
    if df_logs.empty:
        st.info("現在、未回答の質問ログはありません。")
        return

    # 統計情報
    col1, col2 = st.columns(2)
    with col1:
        st.metric("未回答数", len(df_logs))
    with col2:
        # 理由の内訳
        if "reason" in df_logs.columns:
            reason_counts = df_logs["reason"].value_counts()
            top_reason = reason_counts.idxmax() if not reason_counts.empty else "N/A"
            st.metric("最多理由", top_reason)

    st.divider()

    # テーブル表示
    st.subheader("📋 ログ一覧")
    
    # フィルタリング機能
    search_text = st.text_input("🔍 ログを検索 (質問内容などでフィルタ)", "")
    if search_text:
        df_logs = df_logs[
            df_logs.astype(str).apply(lambda x: x.str.contains(search_text, case=False, na=False)).any(axis=1)
        ]

    st.dataframe(
        df_logs,
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("日時", format="YYYY-MM-DD HH:mm:ss"),
            "query": st.column_config.TextColumn("質問内容", width="large"),
            "collections": "検索コレクション",
            "reason": "理由",
            "agent_response": "エージェント応答"
        },
        hide_index=True
    )

    # ダウンロードボタン
    csv = df_logs.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 CSVダウンロード",
        data=csv,
        file_name="unanswered_questions_log.csv",
        mime="text/csv",
    )
