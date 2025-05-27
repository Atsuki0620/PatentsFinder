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
    st.error("GCP のサービスアカウント情報が設定されていません。")
    st.stop()
credentials = service_account.Credentials.from_service_account_info(json.loads(sa_info))

openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OpenAI API Key が設定されていません。")
    st.stop()

# ─── キャッシュリソース定義 ───────────────────────────────────
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

# ─── 設定とクライアント準備 ─────────────────────────────────
config = load_config()
utils = get_utils()
openai_client = utils.openai_client
llm_model = utils.llm_model

initial_prompt = config["chat_flow"]["initial_prompt"]
proposal_prompt = config["chat_flow"]["proposal_prompt"]

# ─── セッションステート初期化 ─────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "question"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "awaiting_confirm" not in st.session_state:
    st.session_state.awaiting_confirm = False
if "initial_prompt_shown" not in st.session_state:
    st.session_state.initial_prompt_shown = False

# ─── チャット描画ヘルパー ────────────────────────────────────
def render_chat():
    for msg in st.session_state.chat_history:
        role = "assistant" if msg["role"] in ("assistant", "system") else "user"
        st.chat_message(role).write(msg["content"])

# ─── 質問フェーズ ───────────────────────────────────────────
def question_phase():
    # 初回は一度だけ初期プロンプトを表示
    if not st.session_state.initial_prompt_shown:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": initial_prompt
        })
        st.session_state.initial_prompt_shown = True

    render_chat()

    if not st.session_state.awaiting_confirm:
        user_input = st.chat_input("自由に入力してください…")
        if user_input:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            render_chat()

            # 解釈確認用メッセージを生成
            resp = openai_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "以下のユーザー発言を要約し、「こういう意味でお間違いないですか？」という自然文で確認メッセージを作成してください。"
                    },
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            interpretation = resp.choices[0].message.content.strip()

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": interpretation
            })
            st.session_state.awaiting_confirm = True
            render_chat()

    else:
        confirm = st.chat_input("この理解でよろしいですか？「はい」または「いいえ」でご回答ください。")
        if confirm:
            st.session_state.chat_history.append({
                "role": "user",
                "content": confirm
            })
            render_chat()

            if confirm.lower() in ["はい", "yes"]:
                st.session_state.mode = "proposal"
            else:
                # 誤解時は履歴を残して再入力を促す
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "すみません、誤解があったようです。もう一度教えてください！"
                })
                render_chat()

            st.session_state.awaiting_confirm = False

# ─── 提案フェーズ ───────────────────────────────────────────
def proposal_phase():
    render_chat()

    if "proposal" not in st.session_state:
        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": proposal_prompt},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                    if m["role"] == "user"
                ]
            ],
            temperature=0
        )
        st.session_state.proposal = resp.choices[0].message.content.strip()

    st.chat_message("assistant").write("こちらの検索方針でよろしいでしょうか？")
    st.code(st.session_state.proposal, language="json")

    col1, col2 = st.columns(2)
    if col1.button("検索実行"):
        st.session_state.mode = "execute"
    if col2.button("修正する"):
        # 「修正する」時だけ履歴とフラグをリセット
        st.session_state.chat_history = []
        st.session_state.initial_prompt_shown = False
        st.session_state.proposal = None
        st.session_state.mode = "question"

# ─── 実行フェーズ ───────────────────────────────────────────
def execute_phase():
    render_chat()
    params = json.loads(st.session_state.proposal)
    query = utils.build_query(params)
    df = utils.search_patents(query)

    st.chat_message("assistant").write(f"🔎 検索結果：{len(df)} 件です！")
    st.dataframe(df)
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV ダウンロード", csv_data, "results.csv", "text/csv")

# ─── メイン ───────────────────────────────────────────────
st.title("🔍 特許調査支援システム（チャットUI版）")

if st.session_state.mode == "question":
    question_phase()
elif st.session_state.mode == "proposal":
    proposal_phase()
else:
    execute_phase()
