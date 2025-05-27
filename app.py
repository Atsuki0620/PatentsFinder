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

# --- フェーズごとの UI ---
def question_phase():
    st.markdown(
        "**どんな特許を調べたいですか？技術内容、公開日（YYYY-MM-DD形式）、対象国（例：JP,US,EP）、およびキーワードを指定すると良い検索結果が得られます。回答例：\n"
              "技術内容：逆浸透膜の洗浄制御、公開日：2015-01-01以降、対象国：JP,US、キーワード：洗浄,メンブレン**"
    )
    answer = st.text_area("上記形式でご回答ください", height=120, key="question_input")
    if st.button("次へ"):  
        if answer.strip():
            st.session_state.chat_history.append({'role': 'user', 'content': answer})
            st.session_state.mode = 'proposal'
            st.experimental_rerun()


def proposal_phase():
    utils = get_utils()
    # システム＋ユーザーのやり取りを生成
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
    st.markdown("**提案された検索パラメータ（JSON形式）**")
    st.code(proposal, language='json')
    col1, col2 = st.columns(2)
    if col1.button("検索実行"):
        st.session_state.mode = 'execute'
        st.experimental_rerun()
    if col2.button("修正する"):
        st.session_state.mode = 'question'
        st.session_state.chat_history = []
        st.experimental_rerun()


def execute_phase():
    utils = get_utils()
    try:
        params = json.loads(st.session_state.proposal)
    except json.JSONDecodeError:
        st.error("提案内容が正しいJSON形式ではありません。再度修正してください。")
        if st.button("修正画面へ戻る"):
            st.session_state.mode = 'question'
            st.session_state.chat_history = []
            st.experimental_rerun()
        return

    sql = utils.build_query(params)
    df = utils.search_patents(sql)
    st.subheader(f"🔎 検索結果: {len(df)} 件")
    st.dataframe(df)
    csv_data = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("CSV ダウンロード", csv_data, "results.csv", "text/csv")

# --- メイン ---
st.title("🔍 特許調査支援システム（チャットフロー版）")
if st.session_state.mode == 'question':
    question_phase()
elif st.session_state.mode == 'proposal':
    proposal_phase()
else:
    execute_phase()
