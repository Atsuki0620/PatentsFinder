
import os
import json
import yaml
import streamlit as st
from google.oauth2 import service_account
from src.utils.patent_utils import PatentSearchUtils

# LangChain imports
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, ConversationChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ─── 設定・クライアント初期化 ───────────────────────────────
@st.cache_resource
def load_config():
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

config      = load_config()
utils       = PatentSearchUtils(config, 
                                credentials=service_account.Credentials.from_service_account_info(
                                    json.loads(os.getenv("GCP_SERVICE_ACCOUNT", "{}"))
                                ),
                                openai_api_key=os.getenv("OPENAI_API_KEY", "")
                               )
llm_model   = config["defaults"]["llm_model"]
emb_model   = config["defaults"]["embedding_model"]

# LangChain の準備
chat_llm    = ChatOpenAI(model_name=llm_model, temperature=0)
memory      = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# チャット用チェーン
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(config["chat_flow"]["initial_prompt"]),
    HumanMessagePromptTemplate.from_template("{user_input}")
])
conversation = ConversationChain(
    llm=chat_llm, 
    memory=memory, 
    prompt=chat_prompt
)

# 提案（JSON生成）用チェーン
proposal_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(config["chat_flow"]["proposal_prompt"]),
    HumanMessagePromptTemplate.from_template("{chat_history}")
])
proposal_chain = LLMChain(llm=chat_llm, prompt=proposal_prompt)

# ─── UI ────────────────────────────────────────────────────
st.title("🔍 特許調査支援システム（LangChain版チャットUI）")

# 初期化：Streamlit 再起動直後のみ
if "ready_for_proposal" not in st.session_state:
    st.session_state.ready_for_proposal = False

# １．チャット入力部
user_input = st.chat_input("自由に教えてください…")
if user_input:
    # ユーザーメッセージを描画
    st.chat_message("user").write(user_input)
    # LangChain に投げる
    ai_response = conversation.predict(user_input=user_input)
    st.chat_message("assistant").write(ai_response)

    # 一度でも質問フェーズが終わったら「提案ボタン」を表示
    st.session_state.ready_for_proposal = True

# ２．提案フェーズへの遷移
if st.session_state.ready_for_proposal:
    col1, col2 = st.columns(2)
    if col1.button("🔧 検索方針を生成する"):
        # conversation.memory.chat_history は System/Assistant/User メッセージの一覧
        history = "\n".join(
            m.content for m in memory.chat_history 
            if m.type in ("human","ai")
        )
        proposal = proposal_chain.run(chat_history=history)
        st.session_state.proposal = proposal
        st.session_state.mode = "proposal"

    if col2.button("🔄 会話を最初からやり直す"):
        memory.clear()
        st.session_state.ready_for_proposal = False
        if "proposal" in st.session_state:
            del st.session_state.proposal
        st.experimental_rerun()

# ３．検索実行フェーズ
if st.session_state.get("mode") == "proposal":
    st.markdown("### 提案された検索パラメータ（JSON）")
    st.code(st.session_state.proposal, language="json")
    if st.button("検索実行"):
        # JSON をパースして検索
        params = json.loads(st.session_state.proposal)
        df = utils.search_patents(utils.build_query(params))
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSVダウンロード", csv, "results.csv", "text/csv")
