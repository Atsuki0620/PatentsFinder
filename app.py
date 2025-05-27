import os
import json
import yaml
import streamlit as st
from google.oauth2 import service_account
from src.utils.patent_utils import PatentSearchUtils

# --- 設定ロード ---
@st.cache_resource
def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
proposal_system_prompt = cfg['chat_flow']['proposal_prompt']

# カジュアルな初期プロンプト
initial_prompt = cfg['chat_flow'].get(
    'initial_prompt',
    "こんにちは！🚀 どんな特許情報をお探しですか？気になる技術や公開日、調査したい国やキーワードなど、思いつくまま教えてください♪"
)

# --- セッションステート初期化 ---
if 'mode' not in st.session_state:
    st.session_state.mode = 'question'      # question, proposal, execute
    st.session_state.chat_history = []      # 会話履歴
    st.session_state.proposal = None        # 提案内容

# --- Secrets 設定 ---
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

# --- チャットUI用関数 ---
def render_chat():
    for msg in st.session_state.chat_history:
        role = msg['role'] if msg['role'] != 'system' else 'assistant'
        st.chat_message(role).write(msg['content'])

# --- フェーズごとの UI ---
def question_phase():
    # 初期プロンプト表示
    if not any(m['role']=='system' and m['content']==initial_prompt for m in st.session_state.chat_history):
        st.session_state.chat_history.append({'role':'system','content': initial_prompt})
    render_chat()
    user_input = st.chat_input("自由に入力してください…")
    if user_input:
        st.session_state.chat_history.append({'role':'user', 'content': user_input})
        st.session_state.mode = 'proposal'
        st.experimental_rerun()


def proposal_phase():
    render_chat()
    # LLMによる提案生成
    utils = get_utils()
    messages = [
        {'role': 'system', 'content': proposal_system_prompt},
        {'role': 'user', 'content': st.session_state.chat_history[-1]['content']}
    ]
    resp = utils.openai_client.chat.completions.create(
        model=utils.llm_model,
        messages=messages,
        temperature=0
    )
    proposal = resp.choices[0].message.content.strip()
    st.session_state.proposal = proposal
    st.session_state.chat_history.append({'role':'assistant','content': proposal})
    render_chat()
    st.code(proposal, language='json')
    # 選択肢表示
    if st.button("🔍 この方針で検索実行"):
        st.session_state.mode = 'execute'
        st.experimental_rerun()
    if st.button("✏️ 方針を修正する"):
        st.session_state.mode = 'question'
        st.session_state.chat_history = []
        st.experimental_rerun()


def execute_phase():
    render_chat()
    utils = get_utils()
    try:
        params = json.loads(st.session_state.proposal)
    except json.JSONDecodeError:
        st.chat_message("assistant").write(
            "あれ？提案された内容がJSONとして読み込めませんでした。もう一度教えてもらえると助かります！"
        )
        if st.button("再入力する"):
            st.session_state.mode = 'question'
            st.session_state.chat_history = []
            st.experimental_rerun()
        return

    df = utils.search_patents(utils.build_query(params))
    st.chat_message("assistant").write(f"結果が出ました！{len(df)} 件の特許を見つけました🙂")
    st.dataframe(df)
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("CSV ダウンロード", csv_data, "results.csv", "text/csv")

# --- メイン ---
st.title("🔍 特許調査支援システム（チャットUI版）")
if st.session_state.mode == 'question':
    question_phase()
elif st.session_state.mode == 'proposal':
    proposal_phase()
else:
    execute_phase()
