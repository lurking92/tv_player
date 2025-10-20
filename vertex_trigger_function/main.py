# main.py - Cloud Function Trigger for Vertex AI Custom Job (v5.11)
import base64
import json
import os
import re
import datetime
import traceback
from google.cloud import aiplatform


# 設定區
PROJECT_ID = "project-e20fc94d-fa04-4164-9d1"
REGION = "asia-east1"
CONTAINER_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/kg-pipeline-repo/kg-pipeline-gpu:latest"
)
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1
SERVICE_ACCOUNT = "175672098089-compute@developer.gserviceaccount.com"
STAGING_BUCKET_URI = "gs://kg_tv-subtitle-videos-20251016/vertex_staging/"


def trigger_vertex_job(event, context):
    """Cloud Function 入口點（由 GCS->Pub/Sub 觸發）"""
    print(f"[CF] 收到事件: {getattr(context, 'event_id', 'N/A')}, 類型: {getattr(context, 'event_type', 'N/A')}")

    # --- Step 1: 解析 GCS 事件 ---
    if "data" not in event:
        print("[CF] 錯誤：事件中缺少 'data' 欄位。")
        return

    try:
        payload = base64.b64decode(event["data"]).decode("utf-8")
        data = json.loads(payload)
        bucket_name = data.get("bucket")
        file_name = data.get("name")

        if not bucket_name or not file_name:
            print(f"[CF] 錯誤：payload 缺少 bucket 或 name：{data}")
            return

        # 忽略：資料夾事件、結果bucket事件、非新建立版本事件
        if (
            file_name.endswith("/") or
            bucket_name.endswith("-results") or
            str(data.get("metageneration", "1")) != "1"
        ):
            print(f"[CF] 略過事件：gs://{bucket_name}/{file_name}")
            return

        print(f"[CF] 偵測到新檔案：gs://{bucket_name}/{file_name}")

    except Exception as e:
        print(f"[CF] 錯誤：解析事件失敗：{e}")
        traceback.print_exc()
        return

    # Step 2: 初始化 Vertex AI SDK 
    try:
        if not PROJECT_ID:
            print("[CF] 錯誤：無法取得 PROJECT_ID。")
            return

        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=STAGING_BUCKET_URI,
        )
        print(f"[CF] Vertex AI SDK 初始化成功（Project={PROJECT_ID}, Region={REGION}, Staging={STAGING_BUCKET_URI}）。")

    except Exception as e:
        print(f"[CF] 錯誤：初始化 Vertex AI 失敗：{e}")
        traceback.print_exc()
        return

    # Step 3: 準備 Job 名稱與容器參數
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w-]", "_", os.path.splitext(file_name)[0])[:30]
    job_display_name = f"kg_job_{safe_name}_{timestamp}"

    container_args = [
        f"--bucket={bucket_name}",
        f"--file={file_name}",
    ]  # bucket內的 main.py 會用 argparse 接收

    # Step 4: 提交 Vertex AI Custom Job 
    try:
        print(f"[CF] 提交 Custom Job：{job_display_name}")
        print(f"      Image: {CONTAINER_IMAGE_URI}")
        print(f"      Args : {container_args}")
        print(f"      Spec : {MACHINE_TYPE}, GPU {ACCELERATOR_COUNT} x {ACCELERATOR_TYPE}")
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
            args=container_args,   # 由 Dockerfile 的 ENTRYPOINT 接收
            sync=False,            # 非同步：提交後立即返回，避免阻塞 Cloud Function
        )

        # 不要再讀 job.resource_name；提交後即可返回
        print("[CF] Vertex AI Job 提交請求已送出。請至 Vertex AI 介面查看狀態。")

    except Exception as e:
        print("[CF] 錯誤：提交 Vertex AI Job 失敗")
        print(f"[CF] 詳細錯誤：{e}")
        traceback.print_exc()
        return
    print("[CF] Cloud Function 執行完畢。")