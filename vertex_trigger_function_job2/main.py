# main.py - Cloud Function Trigger for K-BERT Job (Job 2)
import base64
import json
import os
import re
import datetime
import traceback
from google.cloud import aiplatform, storage
from google.cloud import secretmanager 

# --- 設定區 (Job 2) ---
PROJECT_ID = "project-e20fc94d-fa04-4164-9d1"
REGION = "asia-east1"

# 必須指向我們為 K-BERT 建立的映像檔
CONTAINER_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/kg-pipeline-repo/kg-pipeline-kbert:latest"
)

MACHINE_TYPE = "n1-standard-8" 
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1
STAGING_BUCKET_URI = "gs://kg_tv-subtitle-videos-20251016/vertex_staging/"

# --- 讀取 Secret 的函式 ---
def get_secret(secret_id, version="latest") -> str:
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"[CF-Job2] 警告：讀取 Secret '{secret_id}' 失敗：{e}")
        return ""

# --- 從 Secret 讀取服務帳號 ---
SERVICE_ACCOUNT = get_secret("vertex-sa-email")
storage_client = storage.Client()

def get_corresponding_json_path(bucket_name, ttl_file_name):
    base_name = os.path.splitext(ttl_file_name)[0]
    json_file_name = f"{base_name}_report.json"
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(json_file_name)
        if blob.exists():
            return f"gs://{bucket_name}/{json_file_name}"
        else:
            print(f"警告：找到了 {ttl_file_name}，但找不到對應的 {json_file_name}")
            return None
    except Exception as e:
        print(f"GCS 檢查 {json_file_name} 失敗: {e}")
        return None

def trigger_kbert_job(event, context):
    """Cloud Function 入口點（由 GCS->Pub/Sub 觸發）"""
    print(f"[CF-Job2] 收到事件: {getattr(context, 'event_id', 'N/A')}")

    if not SERVICE_ACCOUNT:
        print("[CF-Job2] 嚴重錯誤：無法從 Secret Manager 讀取 SERVICE_ACCOUNT。")
        return

    if "data" not in event:
        print("[CF-Job2] 錯誤：事件中缺少 'data' 欄位。")
        return

    try:
        payload = base64.b64decode(event["data"]).decode("utf-8")
        data = json.loads(payload)
        bucket_name = data.get("bucket")
        file_name = data.get("name")
        if (
            not bucket_name or not file_name or
            not bucket_name.endswith("-results") or
            not file_name.endswith(".ttl") or
            str(data.get("metageneration", "1")) != "1"
        ):
            print(f"[CF-Job2] 略過事件：gs://{bucket_name}/{file_name}")
            return
        print(f"[CF-Job2] 偵測到新的 .ttl 檔案：gs://{bucket_name}/{file_name}")
        gcs_ttl_path = f"gs://{bucket_name}/{file_name}"
        gcs_json_path = get_corresponding_json_path(bucket_name, file_name)
        if not gcs_json_path:
            print(f"[CF-Job2] 錯誤：找不到 {gcs_ttl_path} 對應的 _report.json。")
            return
    except Exception as e:
        print(f"[CF-Job2] 錯誤：解析事件失敗：{e}")
        traceback.print_exc()
        return

    try:
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=STAGING_BUCKET_URI,
        )
    except Exception as e:
        print(f"[CF-Job2] 錯誤：初始化 Vertex AI 失敗：{e}")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w-]", "_", os.path.splitext(file_name)[0])[:30]
    job_display_name = f"kbert_job_{safe_name}_{timestamp}"
    container_args = [
        f"--gcs_input_ttl={gcs_ttl_path}",
        f"--gcs_input_json={gcs_json_path}",
    ]

    try:
        print(f"[CF-Job2] 提交 K-BERT Job：{job_display_name}")
        print(f"      Image: {CONTAINER_IMAGE_URI}")
        print(f"      Args : {container_args}")
        print(f"      SA   : {SERVICE_ACCOUNT}")

        job = aiplatform.CustomContainerTrainingJob(
            display_name=job_display_name,
            container_uri=CONTAINER_IMAGE_URI,
        )
        job.run(
            machine_type=MACHINE_TYPE,
            accelerator_type=ACCELERATOR_TYPE,
            accelerator_count=ACCELERATOR_COUNT,
            service_account=SERVICE_ACCOUNT,
            args=container_args,
            sync=False, 
        )
        print("[CF-Job2] Vertex AI Job 提交請求已送出。")
    except Exception as e:
        print(f"[CF-Job2] 錯誤：提交 Vertex AI Job 失敗: {e}")
        traceback.print_exc()
        return
    print("[CF-Job2] Cloud Function 執行完畢。")