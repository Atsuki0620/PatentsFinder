import streamlit as st
import json
from google.oauth2 import service_account

st.title("🔐 GCP Service Account Secret テスト")

# 1. st.secrets から JSON 文字列を取得
sa_json = st.secrets.get("GCP_SERVICE_ACCOUNT")
if not sa_json:
    st.error("❌ st.secrets['GCP_SERVICE_ACCOUNT'] が設定されていません。")
    st.stop()

# 2. JSON としてパース
try:
    info = json.loads(sa_json)
except json.JSONDecodeError as e:
    st.error(f"❌ JSON のパースに失敗しました: {e}")
    st.stop()

# 3. 必須キーのチェック
required_keys = ["type", "project_id", "private_key", "client_email", "token_uri"]
missing = [k for k in required_keys if k not in info]
if missing:
    st.error(f"❌ 以下のキーが足りません: {missing}")
    st.stop()

# 4. Credentials を生成してみる
try:
    creds = service_account.Credentials.from_service_account_info(info)
    st.success("✅ Credentials の生成に成功しました！")
    st.write("プロジェクトID:", creds.project_id)
except Exception as e:
    st.error(f"❌ Credentials の生成でエラー: {e}")
