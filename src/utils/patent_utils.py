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

        # BigQuery クライアント: location 指定を追加
        bq_conf = config['bigquery']
        self.bq_client = bigquery.Client(
            credentials=credentials,
            project=bq_conf['project_id'],
            location=bq_conf.get('location', None)
        )
        self.dataset = bq_conf['dataset']
        self.table = bq_conf['table']
        self.limit = bq_conf['limit']

        # FAISS ファイルパス
        self.faiss_index_path = config['paths']['faiss_index']
        self.faiss_mapping_path = config['paths']['faiss_mapping']

        # プロンプト
        self.system_search_prompt = config['prompts']['system_search']
        self.system_summary_prompt = config['prompts']['system_summary']

    def generate_search_params(self, user_input: str) -> Dict[str, Any]:
        resp = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': self.system_search_prompt},
                {'role': 'user',   'content': user_input}
            ],
            temperature=0
        )
        return json.loads(resp.choices[0].message.content.strip())

    def build_query(self, params: Dict[str, Any]) -> str:
        """
        英語タイトル・抄録を取り、IPC／出願人を集約する BigQuery SQL を組み立て
        """
        # 日付は整数型(YYYYMMDD)として比較
        pub_from = params.get("publication_from", self.publication_from)
        # ユーザーが YYYY-MM-DD 形式を渡してくる場合はハイフンを除去
        if isinstance(pub_from, str) and "-" in pub_from:
            pub_from = pub_from.replace("-", "")
    
        # IPC フィルタ (コードプレフィックスで LIKE 検索)
        ipc_filters = []
        for code in params.get("ipc_codes", []):
            # 末尾に % を付けて PREFIX 検索
            ipc_filters.append(f"ipc.code LIKE '{code}%'")
        ipc_clause = " OR ".join(ipc_filters) or "TRUE"
    
        # 出願人フィルタ (部分一致)
        assignee_filters = []
        for name in params.get("assignees", []):
            assignee_filters.append(f"LOWER(assignee.name) LIKE LOWER('%{name}%')")
        assignee_clause = " OR ".join(assignee_filters) or "TRUE"
    
        table_ref = f"`{self.config['bigquery']['project_id']}.{self.dataset}.{self.table}`"
    
        sql = f"""
        SELECT
          p.publication_number,
          (SELECT v.text FROM UNNEST(p.title_localized)    AS v WHERE v.language='en' LIMIT 1) AS title,
          (SELECT v.text FROM UNNEST(p.abstract_localized) AS v WHERE v.language='en' LIMIT 1) AS abstract,
          p.publication_date,
          STRING_AGG(DISTINCT ipc.code, ',')     AS ipc_codes,
          STRING_AGG(DISTINCT assignee.name, ',') AS assignees
        FROM {table_ref} AS p,
             UNNEST(p.ipc)                 AS ipc,
             UNNEST(p.assignee_harmonized) AS assignee
        WHERE
          p.publication_date >= {pub_from}
          AND ({ipc_clause})
          AND ({assignee_clause})
        GROUP BY
          p.publication_number, title, abstract, p.publication_date
        LIMIT {self.limit}
        """
        return sql.strip()

    def search_patents(self, query: str) -> pd.DataFrame:
        job = self.bq_client.query(query)
        return job.to_dataframe()

    def build_faiss_index(self, df: pd.DataFrame):
        abstracts = df['abstract'].fillna('').tolist()
        embeddings = []
        for i in range(0, len(abstracts), self.batch_size):
            batch = abstracts[i:i + self.batch_size]
            resp = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=batch
            )
            embeddings.extend([d['embedding'] for d in resp.data])
        arr = np.array(embeddings, dtype='float32')
        index = faiss.IndexFlatL2(arr.shape[1])
        index.add(arr)
        faiss.write_index(index, self.faiss_index_path)

        with open(self.faiss_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

    def search_similar_patents(self, query: str, k: int) -> List[Dict[str, Any]]:
        index = faiss.read_index(self.faiss_index_path)
        mapping = json.load(open(self.faiss_mapping_path, 'r', encoding='utf-8'))
        resp = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        q_emb = np.array(resp.data[0]['embedding'], dtype='float32')
        _, I = index.search(np.array([q_emb]), k)
        df = pd.DataFrame(mapping)
        return [df.iloc[idx].to_dict() for idx in I[0]]

    def generate_summary(self, text: str) -> str:
        resp = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': self.system_summary_prompt},
                {'role': 'user',   'content': text}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
