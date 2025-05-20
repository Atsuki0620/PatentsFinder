import os
from dotenv import load_dotenv
import yaml
import pandas as pd
import streamlit as st
from src.utils.patent_utils import PatentSearchUtils

# 環境変数の読み込み
load_dotenv()

# 設定ファイルの読み込み
@st.cache_resource
def load_config():
    config_path = "config/config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# PatentSearchUtilsのインスタンス化
@st.cache_resource
def get_patent_utils():
    return PatentSearchUtils("config/config.yaml")

def main():
    st.title("特許調査支援システム")
    
    # サイドバーの設定
    st.sidebar.title("検索設定")
    search_type = st.sidebar.radio(
        "検索モード",
        ["キーワード検索", "類似特許検索"]
    )
    
    # メインコンテンツ
    if search_type == "キーワード検索":
        keyword_search()
    else:
        similar_search()

def keyword_search():
    st.header("キーワード検索")
    
    # 入力フォーム
    user_input = st.text_area(
        "検索条件を自然文で入力してください",
        height=100,
        placeholder="例：逆浸透膜の洗浄をAIによって最適化し、消費電力を抑える水処理プロセスに関する特許を探したい。"
    )
    
    if st.button("検索"):
        if user_input:
            with st.spinner("検索条件を生成中..."):
                utils = get_patent_utils()
                # 検索パラメータの生成
                params = utils.generate_search_params(user_input)
                
                # パラメータの表示
                st.subheader("生成された検索条件")
                st.json(params)
                
                # クエリの生成と実行
                with st.spinner("特許を検索中..."):
                    query = utils.build_query(params)
                    df = utils.search_patents(query)
                    
                    # 結果の表示
                    st.subheader(f"検索結果（{len(df)}件）")
                    st.dataframe(df)
                    
                    # CSVダウンロードボタン
                    st.download_button(
                        "結果をCSVでダウンロード",
                        df.to_csv(index=False).encode('utf-8-sig'),
                        "patent_search_results.csv",
                        "text/csv"
                    )
                    
                    # FAISSインデックスの構築
                    with st.spinner("類似検索用インデックスを構築中..."):
                        utils.build_faiss_index(df)
                        st.success("インデックスの構築が完了しました")

def similar_search():
    st.header("類似特許検索")
    
    # 入力フォーム
    query = st.text_area(
        "検索したい技術内容を入力してください",
        height=100,
        placeholder="例：AIで逆浸透膜の洗浄タイミングを最適化する技術"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        k = st.slider("表示する類似特許数", 1, 10, 3)
    with col2:
        show_summary = st.checkbox("AI要約を表示", value=True)
    
    if st.button("類似特許を検索"):
        if query:
            with st.spinner("類似特許を検索中..."):
                utils = get_patent_utils()
                results = utils.search_similar_patents(query, k)
                
                # 結果の表示
                for i, patent in enumerate(results, 1):
                    with st.expander(f"{i}. {patent['title']}", expanded=True):
                        st.write(f"公開番号: {patent['publication_number']}")
                        st.write(f"出願人: {patent['applicant']}")
                        st.write(f"出願日: {patent['filing_date']}")
                        
                        if show_summary:
                            with st.spinner("AI要約を生成中..."):
                                summary = utils.generate_summary(patent['abstract'])
                                st.write("AI要約:")
                                st.write(summary)
                        
                        st.write("要約:")
                        st.write(patent['abstract'])

if __name__ == "__main__":
    main()