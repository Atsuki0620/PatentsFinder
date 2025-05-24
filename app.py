import os
import json
import yaml
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from src.utils.patent_utils import PatentSearchUtils

# ─── Secrets 取得 ────────────────────────────────────────────
sa_info = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")
if not sa_info:
    st.error("GCP のサービスアカウント情報が設定されていません。GCP_SERVICE_ACCOUNT を設定してください。")
    st.stop()
credentials = service_account.Credentials.from_service_account_info(json.loads(sa_info))

openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OpenAI API Key が設定されていません。OPENAI_API_KEY を設定してください。")
    st.stop()

@st.cache_resource
def load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource
def get_utils() -> PatentSearchUtils:
    cfg = load_config()
    return PatentSearchUtils(
        config=cfg,
        credentials=credentials,
        openai_api_key=openai_key
    )

def main():
    st.title("🔍 特許調査支援システム")
    mode = st.sidebar.radio("検索モード", ["キーワード検索", "類似特許検索"], index=0)
    if mode == "キーワード検索":
        keyword_search()
    else:
        similar_search()

def keyword_search():
    utils = get_utils()
    user_input = st.text_area("検索条件（自然文）を入力してください", height=120)
    if not user_input:
        return
    if st.button("検索実行"):
        with st.spinner("検索中…"):
            try:
                params = utils.generate_search_params(user_input)
                st.subheader("📝 生成された検索パラメータ")
                st.json(params)
                query = utils.build_query(params)
                df = utils.search_patents(query)
                st.subheader(f"🔎 検索結果: {len(df)} 件")
                st.dataframe(df)
                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("CSV ダウンロード", csv_data, "results.csv", "text/csv")
                utils.build_faiss_index(df)
                st.success("📦 FAISS インデックスを構築しました")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

def similar_search():
    utils = get_utils()
    query = st.text_area("技術内容を入力してください", height=120)
    if not query:
        return
    k = st.slider("類似件数", 1, 10, 5)
    show_summary = st.checkbox("要約を表示", True)
    if st.button("類似特許検索実行"):
        with st.spinner("類似特許検索中…"):
            try:
                results = utils.search_similar_patents(query, k)
                for i, patent in enumerate(results, start=1):
                    with st.expander(f"{i}. {patent.get('title', 'No Title')}" ):
                        st.write(f"- 公開番号: {patent.get('publication_number', '')}")
                        st.write(f"- 出願人: {patent.get('assignees', '')}")
                        st.write(f"- 抄録: {patent.get('abstract', '')}")
                        if show_summary:
                            summary = utils.generate_summary(patent.get('abstract', ''))
                            st.write(f"🔍 要約: {summary}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
