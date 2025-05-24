import os, json
import streamlit as st
from google.oauth2 import service_account

st.title("🔐 GCP Secret 読み込みテスト")

# ① st.secrets から
sa_json = st.secrets.get("GCP_SERVICE_ACCOUNT")

# ② 環境変数から（GitHub Actions 経由など）
if not sa_json:
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT")

if not sa_json:
    st.error("❌ Secret が見つかりません (st.secrets or ENV)。")
    st.stop()

# JSON パース
try:
    info = json.loads(sa_json)
except Exception as e:
    st.error(f"❌ JSON パース失敗: {e}")
    st.stop()

# 必須キーチェック
for k in ("type","project_id","private_key","client_email","token_uri"):
    if k not in info:
        st.error(f"❌ キー不足: {k}")
        st.stop()

# Credentials 生成テスト
try:
    creds = service_account.Credentials.from_service_account_info(info)
    st.success("✅ Credentials 生成 OK")
    st.write("プロジェクトID:", creds.project_id)
except Exception as e:
    st.error(f"❌ Credentials 生成失敗: {e}")
