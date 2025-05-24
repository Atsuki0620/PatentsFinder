import json
import yaml
from openai import OpenAI
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import faiss
from typing import List, Dict, Any

class PatentSearchUtils:
    def __init__(
        self,
        config: Dict[str, Any],
        credentials: service_account.Credentials,
        openai_api_key: str
    ):
        # 設定の読み込み
        self.config = config

        # OpenAI 新クライアントの初期化
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.llm_model = config['defaults']['llm_model']
        self.embedding_model = config['defaults']['embedding_model']
        self.publication_from = config['defaults']['publication_from']
        self.batch_size = config['defaults']['batch_size']

        # BigQuery クライアント
        self.bq_client = bigquery.Client(
            credentials=credentials,
            project=config['bigquery']['project_id']
        )
        self.dataset = config['bigquery']['dataset']
        self.table = config['bigquery']['table']
        self.limit = config['bigquery']['limit']

        # FAISS パス
        self.faiss_index_path = config['paths']['faiss_index']
        self.faiss_mapping_path = config['paths']['faiss_mapping']

        # プロンプト
        self.system_search_prompt = config['prompts']['system_search']
        self.system_summary_prompt = config['prompts']['system_summary']

    def generate_search_params(self, user_input: str) -> Dict[str, Any]:
        """
        自然文から検索パラメータ（JSON）を生成
        """
        resp = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': self.system_search_prompt},
                {'role': 'user',   'content': user_input}
            ],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)

    def build_query(self, params: Dict[str, Any]) -> str:
        """
        BigQuery 用の SQL を組み立て
        """
        clauses = []
        if params.get('ipc_codes'):
            codes = ", ".join([f"'{c}'" for c in params['ipc_codes']])
            clauses.append(f"ipc_code IN ({codes})")
        if params.get('assignees'):
            names = " OR ".join([
                f"LOWER(applicant) LIKE LOWER('%{name}%')" for name in params['assignees']
            ])
            clauses.append(f"({names})")
        pub_from = params.get('publication_from', self.publication_from)
        clauses.append(f"publication_date >= '{pub_from}'")

        where = " AND ".join(clauses)
        table_ref = f"`{self.config['bigquery']['project_id']}.{self.dataset}.{self.table}`"
        return f"SELECT * FROM {table_ref} WHERE {where} LIMIT {self.limit}"

    def search_patents(self, query: str) -> pd.DataFrame:
        """
        BigQuery で特許データを取得
        ""`

    def build_faiss_index(self, df: pd.DataFrame):
        """
        抽出結果から埋め込みを作成し、FAISS インデックスを保存
        ""`

    def search_similar_patents(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        FAISS インデックスを用いて類似特許を検索
        ""`

    def generate_summary(self, text: str) -> str:
        """
        与えられたテキストを要約
        ""`
