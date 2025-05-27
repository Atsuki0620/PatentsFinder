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

config = load_config()
utils = get_utils()
openai_client = utils.openai_client
llm_model = utils.llm_model

# チャットフロー用プロンプト
initial_prompt = config["chat_flow"]["initial_prompt"]
proposal_prompt = config["chat_flow"]["proposal_prompt"]

# ─── チャット描画ヘルパー ────────────────────────────────────
def render_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant" or msg["role"] == "system":
            st.chat_message("assistant").write(msg["content"])
        else:
            st.chat_message("user").write(msg["content"])

# ─── 質問フェーズ ───────────────────────────────────────────
def question_phase():
    # 初期化
    if "mode" not in st.session_state:
        st.session_state.mode = "question"
        st.session_state.chat_history = []
        st.session_state.awaiting_confirm = False

    # 初回だけ初期プロンプトを表示
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "system",
            "content": initial_prompt
        })

    render_chat()

    # ユーザー入力 or 確認モードかで処理を分岐
    if not st.session_state.awaiting_confirm:
        user_input = st.chat_input("自由に入力してください…")
        if user_input:
            # ユーザー発言を履歴に
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            render_chat()

            # 解釈確認用プロンプトを作成して LLM に投げる
            resp = openai_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "以下のユーザーの発言を要約し、「こういう意味でお間違いないですか？」という形式で確認文を作成してください。"
                    },
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            interpretation = resp.choices[0].message.content.strip()

            # 解釈結果をアシスタント発言として追加
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": interpretation
            })
            st.session_state.awaiting_confirm = True
            render_chat()

    else:
        # 解釈確認のためのユーザー入力を待つ
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
                # 誤解があった場合、再入力を促す
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "すみません、誤解があったようです。もう一度教えてください！"
                })
                render_chat()
                # stay in question mode

            st.session_state.awaiting_confirm = False

# ─── 提案フェーズ ───────────────────────────────────────────
def proposal_phase():
    render_chat()
    if "proposal" not in st.session_state:
        # ユーザーとのやり取りを元に検索パラメータ JSON を生成
        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": proposal_prompt},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                    if m["role"] in ("user",)
                ]
            ],
            temperature=0
        )
        proposal = resp.choices[0].message.content.strip()
        st.session_state.proposal = proposal

    # 提案を表示
    st.chat_message("assistant").write("こちらの検索方針でよろしいでしょうか？")
    st.code(st.session_state.proposal, language="json")

    col1, col2 = st.columns(2)
    if col1.button("検索実行"):
        st.session_state.mode = "execute"
    if col2.button("修正する"):
        # 修正リクエスト → 質問フェーズに戻す
        st.session_state.mode = "question"
        st.session_state.chat_history = []
        st.session_state.proposal = None

# ─── 実行フェーズ ───────────────────────────────────────────
def execute_phase():
    render_chat()
    # proposal（JSON文字列）をパースして検索条件に
    params = json.loads(st.session_state.proposal)
    query = utils.build_query(params)
    df = utils.search_patents(query)
    st.chat_message("assistant").write(f"🔎 検索結果：{len(df)} 件です！")
    st.dataframe(df)
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV ダウンロード", csv_data, "results.csv", "text/csv")

# ─── メイン ───────────────────────────────────────────────
st.title("🔍 特許調査支援システム（チャットUI版）")

# ステート初期化（ユーザーがブラウザをリロードしたとき用）
if "mode" not in st.session_state:
    st.session_state.mode = "question"

if st.session_state.mode == "question":
    question_phase()
elif st.session_state.mode == "proposal":
    proposal_phase()
else:
    execute_phase()
