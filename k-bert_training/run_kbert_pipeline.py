# run_kbert_pipeline.py 
import argparse
import json
import os
import sys
from pathlib import Path
from google.cloud import storage
import traceback 

try:
    from ttl_to_json_converter_6 import convert_observation_ttl_to_action_json
    from json_risk_detector_7 import initialize_kbert_model, analyze_json_for_risk
    print("[OK] 成功導入 ttl_to_json_converter_6 和 json_risk_detector_7。")
except ImportError as e:
    print(f"[CRITICAL] 導入本地 K-BERT 腳本失敗: {e}")
    sys.exit(1)

# (GCS 輔助函式 v2)
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS 檔案不存在：gs://{bucket_name}/{source_blob_name}")
        print(f"[GCS] 開始下載：gs://{bucket_name}/{source_blob_name}")
        Path(destination_file_name).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination_file_name)
        print(f"[GCS] 檔案已下載至：{destination_file_name}")
        return True
    except Exception as e:
        raise Exception(f"GCS 下載失敗：{e}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        print(f"[GCS] 開始上傳至：gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_filename(source_file_name)
        print(f"[GCS] 檔案上傳成功。")
        return True
    except Exception as e:
        raise Exception(f"GCS 上傳失敗：{e}")

def main():
    parser = argparse.ArgumentParser(description="K-BERT Risk Prediction Pipeline (Job 2)")
    parser.add_argument("--gcs_input_ttl", required=True, help="GCS 上的 .ttl 檔案路徑")
    parser.add_argument("--gcs_input_json", required=True, help="GCS 上的 _report.json 檔案路徑")
    args = parser.parse_args()

    try:
        # (步驟 1: 剖析路徑)
        try:
            ttl_bucket_name = args.gcs_input_ttl.split('/')[2]
            ttl_blob_name = '/'.join(args.gcs_input_ttl.split('/')[3:])
            json_bucket_name = args.gcs_input_json.split('/')[2]
            json_blob_name = '/'.join(args.gcs_input_json.split('/')[3:])
            base_name = os.path.splitext(os.path.basename(ttl_blob_name))[0]
            local_ttl_path = f"/tmp/{base_name}.ttl"
            local_json_path = f"/tmp/{base_name}_report.json"
            local_inferred_actions_path = f"/tmp/{base_name}_inferred_actions.json"
        except Exception as e:
            raise Exception(f"解析 GCS 路徑失敗: {e}")

        # (步驟 2: 下載檔案)
        print(f"\n[步驟 2a] 下載 Job 1 產生的 .ttl 和 _report.json...")
        download_from_gcs(ttl_bucket_name, ttl_blob_name, local_ttl_path)
        download_from_gcs(json_bucket_name, json_blob_name, local_json_path)

        # (步驟 3: 執行轉換)
        print(f"\n[步驟 2b] 開始執行 ttl_to_json_converter_6...")
        output_json_file = convert_observation_ttl_to_action_json(
            ttl_path=local_ttl_path, 
            output_dir="/tmp"
        )
        if not output_json_file or not os.path.exists(output_json_file):
             raise Exception("ttl_to_json_converter_6 未能產生輸出檔案。")
        print(f"步驟 2b 完成。推論的動作 JSON 位於: {output_json_file}")
        local_inferred_actions_path = output_json_file

        # (步驟 4: 執行預測)
        print(f"\n[步驟 2c] 開始執行 json_risk_detector_7...")
        initialize_kbert_model()
        kbert_results = analyze_json_for_risk(local_inferred_actions_path)
        if kbert_results is None:
            raise Exception("json_risk_detector_7 未回傳有效的分析結果。")
        print(f"步驟 2c 完成。K-BERT 分析結果: {kbert_results.get('summary')}")

        # (步驟 5: 合併報告)
        print(f"\n[步驟 2d] 合併 K-BERT 結果到 _report.json...")
        with open(local_json_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        report_data["analysis"]["kbert_analysis"] = kbert_results
        report_data["status"] = "completed_with_risk" if kbert_results.get("contains_danger") else "completed"
        with open(local_json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print("步驟 2d 完成。JSON 報告已在本地合併。")

        # (步驟 6: 上傳)
        print(f"\n[步驟 3] 上傳最終的 _report.json...")
        upload_to_gcs(json_bucket_name, local_json_path, json_blob_name)
        
        print("\n[SUCCESS] K-BERT 預測任務 (Job 2) 執行完畢。")

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Job 2 執行失敗: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()