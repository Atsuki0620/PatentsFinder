import os
import json
import yaml
import streamlit as st
from google.oauth2 import service_account
from src.utils.patent_utils import PatentSearchUtils

# --- 質問フロー用設定ロード ---
@st.cache_resource
def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
questions = cfg['chat_flow']['questions']  # 質問リスト

# --- セッションステート初期化 ---
if 'mode' not in st.session_state:
    st.session_state.mode = 'question'      # question, proposal, execute
    st.session_state.step = 0               # 現在の質問インデックス
    st.session_state.chat_history = []      # 会話履歴
    st.session_state.proposal = None        # 検索方針

# --- Streamlit Secrets ---
sa_info = st.secrets.get("GCP_SERVICE_ACCOUNT") or os.getenv("GCP_SERVICE_ACCOUNT")
credentials = service_account.Credentials.from_service_account_info(json.loads(sa_info))
openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

@st.cache_resource
def get_utils():
    return PatentSearchUtils(
        config=cfg,
        credentials=credentials,
        openai_api_key=openai_key
    )

# --- フェーズごとの UI ---
def question_phase():
    q = questions[st.session_state.step]
    st.markdown(f"**Q: {q['text']}**")
    answer = st.text_input("回答を入力してください", key=f"ans_{st.session_state.step}")
    if st.button("送信", key=f"send_{st.session_state.step}"):
        st.session_state.chat_history.append({'role': 'user', 'content': answer})
        st.session_state.chat_history.append({'role': 'system', 'content': q['followup_prompt'].format(answer)})
        st.session_state.step += 1
        if st.session_state.step >= len(questions):
            st.session_state.mode = 'proposal'
        st.experimental_rerun()


def proposal_phase():
    utils = get_utils()
    # システムプロンプト
    prompt = cfg['chat_flow']['proposal_prompt']
    msgs = [{'role':'system','content': prompt}] + st.session_state.chat_history
    resp = utils.openai_client.chat.completions.create(
        model=utils.llm_model,
        messages=msgs,
        temperature=0
    )
    proposal = resp.choices[0].message.content.strip()
    st.session_state.proposal = proposal
    st.markdown("**提案された検索方針**")
    st.write(proposal)
    col1, col2 = st.columns(2)
    if col1.button("検索実行"):
        st.session_state.mode = 'execute'
        st.experimental_rerun()
    if col2.button("修正する"):
        st.session_state.mode = 'question'
        st.session_state.step = 0
        st.session_state.chat_history = []
        st.experimental_rerun()


def execute_phase():
    utils = get_utils()
    # ここで proposal から JSON パラメータ抽出 or build_query 実行
    params = json.loads(st.session_state.proposal)  # proposal に JSON が含まれている想定
    sql = utils.build_query(params)
    df = utils.search_patents(sql)
    st.subheader(f"🔎 検索結果: {len(df)} 件")
    st.dataframe(df)

# --- メイン ---
st.title("🔍 特許調査支援システム（チャットフロー版）")
if st.session_state.mode == 'question':
    question_phase()
elif st.session_state.mode == 'proposal':
    proposal_phase()
else:
    execute_phase()
