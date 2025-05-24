import os
import json
import sys
from google.oauth2 import service_account

def main():
    # 1. 環境変数から Secret を取得
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    if not sa_json:
        print("❌ エラー: 環境変数 GCP_SERVICE_ACCOUNT が設定されていません。")
        sys.exit(1)

    # 2. JSON としてパースし、キーの存在をチェック
    try:
        info = json.loads(sa_json)
    except json.JSONDecodeError as e:
        print(f"❌ エラー: JSON のパースに失敗しました: {e}")
        sys.exit(1)

    required_keys = ["type", "project_id", "private_key", "client_email", "token_uri"]
    missing = [k for k in required_keys if k not in info]
    if missing:
        print(f"❌ エラー: 以下のキーが Secret JSON に見つかりません: {missing}")
        sys.exit(1)

    # 3. Credentials オブジェクトを実際に作ってみる
    try:
        creds = service_account.Credentials.from_service_account_info(info)
    except Exception as e:
        print(f"❌ エラー: Credentials の生成に失敗しました: {e}")
        sys.exit(1)

    # 4. 成功メッセージ
    print("✅ 成功: GCP_SERVICE_ACCOUNT が読み込まれ、Credentials が生成できました。")

if __name__ == "__main__":
    main()
