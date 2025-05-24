import os
import json
import pytest
from google.oauth2 import service_account
from google.cloud import bigquery

def test_gcp_connectivity():
    # ① 環境変数から JSON を取得
    sa_json_str = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    assert sa_json_str, "環境変数 GCP_SERVICE_ACCOUNT_JSON が設定されていません"

    # ② JSON をパースし、認証情報を作成
    sa_info = json.loads(sa_json_str)
    creds   = service_account.Credentials.from_service_account_info(sa_info)
    client  = bigquery.Client(credentials=creds, project=sa_info["project_id"])

    # ③ データセット一覧を取得して件数をチェック
    datasets = list(client.list_datasets())
    assert isinstance(datasets, list), "datasets がリストではありません"
    print(f"✅ Connected to project {sa_info['project_id']}, found {len(datasets)} datasets")
