# =================================================================
# main_gpu.py - 適用於 Vertex AI Job 的腳本 (v2.5-mod-notrans-fix2)
# Cloud Secret Manager 讀取 Gemini API Key
# BLIP + Grounding DINO -> 三元組 -> TTL
# 徹底移除 faster-whisper (import 和使用)
# 修正 DINO post_process 函數 (使用 .ipynb 邏輯)
# 防禦性修正 KeyError: 'artifacts' 
# =================================================================

# 1) 基本匯入
import os
import json
import base64
import cv2
import math
import subprocess
import shlex
import tempfile
from typing import List, Dict, Tuple
import hashlib
import re
import datetime
import traceback
import argparse

# 2) 服務 / 網路
from google.cloud import storage
from google.cloud import secretmanager
import google.generativeai as genai

# 3) AI/ML
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    pipeline,
)
from opencc import OpenCC
from retry import retry
import numpy as np


# 裝置設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全域物件（lazy-init，一次載入）
storage_client = storage.Client()

gemini_model = None
blip_processor = None
blip_model = None
g_dino_processor = None
g_dino_model = None
converter = None

# Log 輔助
def log_message(message: str) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    print(f"[{ts}] {message}", flush=True)

# 從 Secret Manager / 環境變數 讀取 Gemini API Key
def get_project_id_for_secret() -> str:
    return (
        os.environ.get("GCP_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or "project-e20fc94d-fa04-4164-9d1"
    )

def get_gemini_api_key() -> str:
    project_id = get_project_id_for_secret()
    secret_id = os.environ.get("GEMINI_SECRET_ID", "gemini-api-key")
    version = os.environ.get("GEMINI_SECRET_VERSION", "latest")
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        api_key = response.payload.data.decode("UTF-8")
        if api_key:
            log_message(f"成功從 Secret Manager 讀取 `{secret_id}`（版本：{version}）。")
            return api_key
    except Exception as e:
        log_message(f"警告：從 Secret Manager 讀取 `{secret_id}` 失敗：{e}")
    env_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if env_api_key:
        log_message("使用環境變數中的 Gemini API Key。")
        return env_api_key
    log_message("警告：未取得 Gemini API Key（Secret 與環境變數皆無）。")
    return ""

# 設定 Gemini
_API_KEY = get_gemini_api_key()
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
if _API_KEY:
    try:
        genai.configure(api_key=_API_KEY)
        gemini_model = genai.GenerativeModel(_GEMINI_MODEL)
        log_message(f"Gemini 已初始化（model={_GEMINI_MODEL}）。")
    except Exception as e:
        log_message(f"警告：初始化 Gemini 失敗：{e}")
        gemini_model = None
else:
    gemini_model = None

# Grounding DINO 的預設提示詞
DEFAULT_PROMPT = (
    "person. man. woman. elderly. child. baby. "
    "bed. pillow. blanket. sheet. mattress. nightstand. bedside table. "
    "lamp. desk lamp. floor lamp. ceiling light. "
    "sofa. couch. armchair. chair. dining chair. office chair. seat. "
    "table. dining table. coffee table. desk. furniture. "
    "television. tv. monitor. computer. laptop. screen. "
    "stove. oven. refrigerator. fridge. microwave. sink. faucet. tap. cabinet. "
    "cupboard. drawer. counter. "
    "pan. pot. bowl. cup. plate. dish. knife. fork. spoon. utensil. "
    "toilet. bathtub. shower. towel. mirror. sink. "
    "door. window. curtain. "
    "book. notebook. paper. pen. pencil. phone. remote. bottle. "
    "walker. cane. wheelchair"
)

# =========================
# 外部工具輔助 (不變)
# =========================
def run_ffmpeg_cmd(cmd_list) -> Tuple[int, str, str]:
    try:
        p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(timeout=300)
        return p.returncode, out.decode("utf-8", errors="ignore"), err.decode("utf-8", errors="ignore")
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate()
        log_message("錯誤：ffmpeg 指令逾時")
        return -1, out.decode("utf-8", errors="ignore"), err.decode("utf-8", errors="ignore") + "\nFFMPEG Timeout"

def ffprobe_json(path):
    cmd = ["ffprobe", "-v", "error", "-hide_banner", "-print_format", "json", "-show_streams", "-show_format", path]
    rc, out, err = run_ffmpeg_cmd(cmd)
    return rc, out, err

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1); area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def slug(s: str) -> str:
    s = (s or "").strip(); s = re.sub(r"\s+", "_", s); s = re.sub(r"[^\w\-]", "", s)
    return s or "unnamed"

def iri_pred(rel: str) -> str:
    r = (rel or "relatedTo").strip(); r = re.sub(r"\s+", "_", r); r = re.sub(r"[^\w\-]", "", r)
    return r or "relatedTo"

def edge_event_id(key: str) -> str:
    return "event_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

def format_timestamp_for_log(seconds: float) -> str:
    s = int(seconds); return f"[{s//60:02d}:{s%60:02d}]"

# 視覺標註繪製（OpenCV）(不變)
def draw_dino_predictions(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    img_bgr = frame.copy()
    def color_for(label: str):
        hv = int(hashlib.md5(label.encode()).hexdigest(), 16)
        r = (hv >> 16) & 0xFF; g = (hv >> 8) & 0xFF; b = hv & 0xFF
        return (b, g, r)
    for det in detections:
        try:
            label = str(det.get("label", "unknown")); score = float(det.get("score", 0)); box = det.get("bbox", [])
            if not box: continue
            x1, y1, x2, y2 = [int(p) for p in box]; color = color_for(label)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {score:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_bgr, (x1, y1 - th - bl), (x1 + tw, y1), color, -1)
            cv2.putText(img_bgr, text, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            log_message(f"警告：繪製標註框失敗: {e}")
    return img_bgr

# 模型初始化與推論
def initialize_models():
    global blip_processor, blip_model
    global g_dino_processor, g_dino_model, converter
    log_message(f"開始初始化模型... (使用裝置: {DEVICE})")

    # log_message("警告：Whisper 模型已暫時移除 (避開 cuDNN 錯誤)。")
    global whisper_model
    whisper_model = None

    if blip_processor is None or blip_model is None:
        log_message("載入 BLIP (large)...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model.to(DEVICE); blip_model.eval()
        log_message("BLIP 已載入。")

    if g_dino_processor is None or g_dino_model is None:
        log_message("載入 Grounding DINO (base)...")
        mname = "IDEA-Research/grounding-dino-base"
        g_dino_processor = AutoProcessor.from_pretrained(mname)
        g_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(mname)
        g_dino_model.to(DEVICE); g_dino_model.eval()
        log_message("Grounding DINO 已載入。")

    if converter is None:
        # log_message("載入 OpenCC (簡轉繁)...")
        converter = OpenCC("s2t")
        # log_message("OpenCC 已載入。")
    log_message("所有 AI 模型已完成初始化 (Whisper 已移除)。")

@torch.no_grad()
def generate_visual_description(image_pil: Image.Image) -> str:
    if blip_model is None or blip_processor is None:
        log_message("警告：BLIP 模型未載入。"); return ""
    try:
        inputs = blip_processor(images=image_pil, return_tensors="pt").to(DEVICE)
        output = blip_model.generate(**inputs, max_length=50, num_beams=5)
        return blip_processor.decode(output[0], skip_special_tokens=True).strip()
    except Exception as e:
        log_message(f"錯誤：BLIP 生成描述失敗: {e}"); return ""

# --- DINO 後處理輔助函數 (來自 .ipynb 檔案) ---
def _dino_post_process(
    outputs,
    input_ids,
    image_size_hw: Tuple[int, int],
    box_threshold: float,
    text_threshold: float,
) -> List[Dict]:
    global g_dino_processor
    h, w = image_size_hw
    try:
        processed = g_dino_processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]
    except TypeError:
        log_message("  (DINO: 偵測到 TypeError，退回使用位置參數 post_process)")
        processed = g_dino_processor.post_process_grounded_object_detection(
            outputs, input_ids, box_threshold, text_threshold, [(h, w)]
        )[0]
    text_labels = processed.get("text_labels", None)
    raw_labels = processed.get("labels", [])
    labels_out = [str(x) for x in text_labels] if text_labels is not None else [str(x) for x in raw_labels]
    boxes = processed.get("boxes", [])
    scores = processed.get("scores", [])
    dets: List[Dict] = []
    for b, s, lab in zip(boxes, scores, labels_out):
        x0, y0, x1, y2 = map(float, getattr(b, "tolist", lambda: b)())
        dets.append({"label": lab, "score": float(s), "bbox": [x0, y0, x1, y2]})
    return dets

@torch.no_grad()
def detect_objects_with_dino(image_pil: Image.Image, text_prompt: str) -> List[Dict]:
    if g_dino_model is None or g_dino_processor is None:
        log_message("警告：DINO 模型未載入。"); return []
    try:
        phrases = [p.strip() for p in text_prompt.split(".") if p.strip()]
        if not phrases: log_message("警告：DINO 提示詞為空。"); return []
        inputs = g_dino_processor(images=image_pil, text=phrases, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = g_dino_model(**inputs)
        detections = _dino_post_process(
            outputs=outputs,
            input_ids=inputs.input_ids,
            image_size_hw=image_pil.size[::-1],
            box_threshold=0.25,
            text_threshold=0.25
        )
        return detections
    except Exception as e:
        log_message(f"錯誤：DINO 偵測失敗: {e}")
        traceback.print_exc()
        return []

# 語意抽取（Gemini）(不變)
@retry(tries=3, delay=4)
def ie_chunk(text: str) -> str:
    # ... (程式碼不變) ...
    if not gemini_model:
        log_message("警告：Gemini 模型未設定，跳過語意抽取。"); return '{"entities":[],"relationships":[]}'
    SYSTEM = "You are an expert system designed to extract domain-agnostic knowledge graphs."
    USER_TMPL = "Extract entities and relationships from the following text:\n\nTEXT:\n{chunk}"
    try:
        resp = gemini_model.generate_content(
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM}]},
                {"role": "model", "parts": [{"text": "OK. I will extract entities and relationships in JSON."}]},
                {"role": "user", "parts": [{"text": USER_TMPL.format(chunk=text)}]},
            ],
            generation_config={"response_mime_type": "application/json"},
        )
        return resp.text or '{"entities":[],"relationships":[]}'
    except Exception as e:
        log_message(f"錯誤：Gemini API 失敗: {e}"); return '{"entities":[],"relationships":[]}'


# K-BERT（預留）
def analyze_with_kbert(visual_triples: List[Dict], text_triples: List[Dict], transcript: List[Dict]) -> Dict:
    log_message("資訊：K-BERT 分析（未實作），回傳略過。")
    return { "status": "skipped", "summary": "K-BERT analysis not yet implemented.", "contains_danger": False, "details": {} }

# GCS 上傳輔助 
def upload_file_to_results(
    video_bucket_name: str,
    gcs_file_path: str,
    local_file_path: str = None,
    content_type: str = "text/plain",
    report_data: dict = None,
):
    try:
        results_bucket_name = video_bucket_name + "-results"
        results_bucket = storage_client.bucket(results_bucket_name)
        blob = results_bucket.blob(gcs_file_path)
        if report_data is not None:
            data_to_upload = json.dumps(report_data, indent=2, ensure_ascii=False)
            blob.upload_from_string(data_to_upload, content_type=content_type)
        elif local_file_path and os.path.exists(local_file_path):
            blob.upload_from_filename(local_file_path, content_type=content_type)
        else:
            log_message(f"錯誤：上傳 {gcs_file_path} 失敗，缺少来源。"); return
        log_message(f"資訊：已上傳到 gs://{results_bucket_name}/{gcs_file_path}")
    except Exception as e:
        log_message(f"錯誤：上傳 {gcs_file_path} 失敗: {e}")
        # 如果上傳失敗，重新拋出錯誤 
        raise e # 確保上傳失敗時外層能捕捉到

# 主處理流程
def process_video_for_kg(video_bucket_name: str, video_file_name: str) -> None:
    log_message(f"背景任務開始: gs://{video_bucket_name}/{video_file_name}")
    video_base_name = os.path.splitext(video_file_name)[0]

    # 修正 KeyError:將 report 移到 try...except 外部
    report = {
        "source_video": f"gs://{video_bucket_name}/{video_file_name}",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "processing",
        "artifacts": {
            "report_json": f"{video_base_name}_report.json",
            "knowledge_graph_ttl": f"{video_base_name}.ttl",
            "dino_log_txt": f"{video_base_name}_dino.txt",
            "dino_images_dir": f"{video_base_name}_dino_images/",
        },
        "analysis": {
            "visual_analysis": {"status": "pending", "object_detections_log": [], "visual_triples": []},
            "audio_analysis": {"status": "pending", "transcript": [], "text_sections": []},
            "semantic_analysis": {"status": "pending", "entities": [], "relationships": []},
            "kbert_analysis": {"status": "pending", "summary": "", "contains_danger": False},
        },
        "errors": [],
    }-

    try:
        # 0) 初始化
        log_message("步驟 0：初始化模型...")
        initialize_models()
        log_message("步驟 0：初始化完成。")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_path = os.path.join(tmpdir, video_file_name)
            local_dino_images_dir = os.path.join(tmpdir, "dino_images")
            os.makedirs(local_dino_images_dir, exist_ok=True)
            dino_log_lines: List[str] = []

            # 下載影片
            log_message("步驟 0：下載影片中...")
            bucket = storage_client.bucket(video_bucket_name)
            blob = bucket.blob(video_file_name)
            if not blob.exists():
                raise FileNotFoundError(f"檔案不存在：gs://{video_bucket_name}/{video_file_name}")
            blob.download_to_filename(local_video_path)
            log_message("步驟 0：影片下載完成。")

            # 1) 視覺分析
            log_message("步驟 1：視覺分析（取樣每 2 秒）...")
            cap = cv2.VideoCapture(local_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(int(fps * 2.0), 1)
            frame_analysis_results: List[Tuple[float, List[Dict]]] = []
            frame_index = 0
            first_frame_size = None

            while cap.isOpened():
                if not cap.grab(): break
                if frame_index % step == 0:
                    ok, frame_bgr = cap.retrieve()
                    timestamp = frame_index / fps
                    if not ok: break
                    log_message(f"  影格 {frame_index} @ {timestamp:.2f}s")
                    
                    # 修正 KeyError:將 try...except 範圍縮小
                    detections = [] # 預設為空
                    try:
                        image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                        if first_frame_size is None: first_frame_size = image_pil.size
                        blip_caption = generate_visual_description(image_pil)
                        full_prompt = (blip_caption + " . " + DEFAULT_PROMPT).strip(". ")
                        
                        detections = detect_objects_with_dino(image_pil, full_prompt) # 呼叫修正後的 DINO
                    
                    except Exception as fe:
                        # 如果 DINO 或 BLIP 失敗
                        log_message(f"錯誤：影格 {frame_index} AI分析失敗：{fe}")
                        report["errors"].append(f"Frame {frame_index} analysis error: {str(fe)}")
                    
                    # 無論 AI 是否成功，都儲存frame分析結果 (可能是空 list)
                    frame_analysis_results.append((timestamp, detections))
                    
                    # 嘗試記錄日誌和儲存圖片 (現在 report['artifacts'] 必定存在)
                    try:
                        log_time = format_timestamp_for_log(timestamp)
                        log_dets = ", ".join([f"{d['label']}({d['score']:.2f})" for d in detections])
                        dino_log_lines.append(f"{log_time} {log_dets}")

                        annotated = draw_dino_predictions(frame_bgr, detections)
                        frame_filename = f"frame_{frame_index:06d}.jpg"
                        local_frame_path = os.path.join(local_dino_images_dir, frame_filename)
                        cv2.imwrite(local_frame_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        
                        report["analysis"]["visual_analysis"]["object_detections_log"].append({
                            "timestamp": round(timestamp, 2),
                            "blip_caption": locals().get("blip_caption", ""), # 使用 locals().get 避免 blip_caption 未定義
                            "detections_count": len(detections),
                            "annotated_image_path": f"{report['artifacts']['dino_images_dir']}{frame_filename}",
                        })
                    except Exception as log_e:
                         # 即使日誌記錄失敗，也不要讓迴圈崩潰
                        log_message(f"錯誤：影格 {frame_index} 日誌或存圖失敗：{log_e}")
                        report["errors"].append(f"Frame {frame_index} logging/saving error: {str(log_e)}")
                    

                frame_index += 1
            cap.release()
            report["analysis"]["visual_analysis"]["status"] = "completed"
            log_message(f"步驟 1：完成（總frame {frame_index}，featuring {len(frame_analysis_results)} 次）。")

            # 2) 視覺三元組 (現在應該會有結果了)
            log_message("步驟 2：生成視覺三元組...")
            if first_frame_size: W, H = first_frame_size
            else: W, H = 1, 1
            max_dim = float(max(W, H))
            visual_triples: List[Dict] = []
            for ts, dets in frame_analysis_results: # 這裡的 dets 可能是 DINO 成功或失敗回傳的空 list
                persons = [d for d in dets if "person" in str(d.get("label", "")).lower() and d.get("score", 0) >= 0.35]
                objs = [d for d in dets if "person" not in str(d.get("label", "")).lower() and d.get("score", 0) >= 0.40]
                for i, p in enumerate(persons, start=1):
                    x1, y1, x2, y2 = p["bbox"]
                    w, h = x2 - x1, y2 - y1
                    state = "lying_or_fall" if (h / (w + 1e-6) < 0.8 and w > h) else "upright"
                    visual_triples.append({ "head": f"person#{i}", "relation": "has_state", "tail": state, "time": round(ts, 2) })
                for i, p in enumerate(persons, start=1):
                    pb = p["bbox"]; px = (pb[0] + pb[2]) / 2.0; py = (pb[1] + pb[3]) / 2.0
                    for o in objs:
                        ob = o["bbox"]; ox = (ob[0] + ob[2]) / 2.0; oy = (ob[1] + ob[3]) / 2.0
                        dist = math.hypot(px - ox, py - oy) / max_dim
                        if iou(pb, ob) > 0.01 or dist < 0.2:
                            visual_triples.append({ "head": f"person#{i}", "relation": "near", "tail": str(o["label"]), "time": round(ts, 2) })
            report["analysis"]["visual_analysis"]["visual_triples"] = visual_triples
            log_message(f"步驟 2：完成（共 {len(visual_triples)} 條）。")


            # 3) 音訊分析 (已移除)
            # log_message("步驟 3：音訊分析（Whisper 已移除），略過。")
            report["analysis"]["audio_analysis"]["status"] = "skipped"
            report["analysis"]["audio_analysis"]["transcript"] = []
            report["analysis"]["audio_analysis"]["text_sections"] = []
            transcript = []
            sections = []

            # 4) 語意分析 (已移除)
            log_message("步驟 4：語意分析（Gemini）...")
            all_entities, all_relationships = [], []
            if not sections:
                log_message("資訊：無音訊文字片段，略過 Gemini 語意分析。")
            elif not gemini_model:
                log_message("警告：Gemini 未初始化，略過語意分析。")
                report["errors"].append("Gemini model not initialized.")
            else:
                for i, sec in enumerate(sections, start=1):
                    # Gemini 分析迴圈
                    pass 
            report["analysis"]["semantic_analysis"]["entities"] = all_entities
            report["analysis"]["semantic_analysis"]["relationships"] = all_relationships
            report["analysis"]["semantic_analysis"]["status"] = "skipped"
            log_message(f"步驟 4：完成（實體 0、關係 0）。")

            # 5) 產生 TTL (只含視覺)
            log_message("步驟 5：生成 TTL...")
            def norm_person(n: str):
                m = re.search(r"person[#\s_]*([0-9]+)", n, re.I)
                if m: return (f"person{m.group(1)}", int(m.group(1)), f"person#{m.group(1)}")
                return (slug(n), None, n)
            people, objects, postures = {}, {}, {}
            events = []
            for r in visual_triples: # 只會有視覺
                nid, tid, lbl = norm_person(r["head"])
                people[nid] = {"label": lbl, **({"trackId": tid} if tid else {})}
                tail = r["tail"]
                if str(tail).lower() in {"upright", "lying_or_fall"}: postures[slug(tail)] = tail
                else: objects[slug(tail)] = tail
                events.append({ "id": edge_event_id(str(r)), "type": "ex:VisualObservation", "agent": nid, "object": slug(tail), "relation": iri_pred(r["relation"]), "tOffset": r.get("time"), "source": "VisionPipeline" })
            
            events.sort(key=lambda ev: (0, ev["tOffset"]) if ev.get("tOffset") is not None else (1, 0))
            lines = [ "@prefix ex: <http://example.org/> .", "@prefix prov: <http://www.w3.org/ns/prov#> .", "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .", "@prefix schema: <http://schema.org/> .", "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", "" ]
            for ev in events:
                lines.append(f"ex:{ev['id']} a {ev['type']} ;")
                lines.append(f"    ex:agent ex:{ev['agent']} ;")
                lines.append(f"    ex:object ex:{ev['object']} ;")
                lines.append(f"    ex:relation ex:{ev['relation']} ;")
                if ev.get("tOffset") is not None: lines.append(f"    ex:timeOffsetS {ev['tOffset']:.2f} ;")
                lines.append(f"    prov:wasDerivedFrom ex:{ev['source']} .")
                lines.append("")
            for nid, meta in sorted(people.items()):
                lines.append(f"ex:{nid} a ex:Agent ;")
                tail = f"    rdfs:label \"{meta['label']}\""
                if "trackId" in meta: lines.append(tail + " ;"); lines.append(f"    ex:trackId {meta['trackId']} .")
                else: lines.append(tail + " .")
                lines.append("")
            for nid, lbl in sorted(objects.items()):
                lines.append(f"ex:{nid} a ex:Object ;"); lines.append(f"    rdfs:label \"{lbl}\" ."); lines.append("")
            for nid, lbl in sorted(postures.items()):
                lines.append(f"ex:{nid} a ex:Posture ;"); lines.append(f"    rdfs:label \"{lbl}\" ."); lines.append("")
            
            ttl_content = "\n".join(lines)
            local_ttl_path = os.path.join(tmpdir, f"{video_base_name}.ttl")
            with open(local_ttl_path, "w", encoding="utf-8") as f: f.write(ttl_content)
            log_message("步驟 5：TTL 生成完成。")

            # 6) K-BERT（預留）
            log_message("步驟 6：K-BERT 分析（預留）...")
            kbert_result = analyze_with_kbert(visual_triples, [], []) # 傳入空列表
            report["analysis"]["kbert_analysis"] = kbert_result
            report["status"] = "completed"
            log_message("步驟 6：完成。")

            # 7) 上傳所有成果
            log_message("步驟 7：上傳成果到 -results Bucket...")
            
            # report['artifacts'] 必定存在
            local_dino_txt_path = os.path.join(tmpdir, f"{video_base_name}_dino.txt")
            with open(local_dino_txt_path, "w", encoding="utf-8") as f: f.write("\n".join(dino_log_lines))
            upload_file_to_results(video_bucket_name, report["artifacts"]["dino_log_txt"], local_dino_txt_path, "text/plain")

            for img_name in os.listdir(local_dino_images_dir):
                local_img_path = os.path.join(local_dino_images_dir, img_name)
                gcs_img_path = f"{report['artifacts']['dino_images_dir']}{img_name}"
                upload_file_to_results(video_bucket_name, gcs_img_path, local_img_path, "image/jpeg")
            log_message(f"  已上傳 {len(os.listdir(local_dino_images_dir))} 張 DINO 標註圖像。")

            upload_file_to_results(video_bucket_name, report["artifacts"]["knowledge_graph_ttl"], local_ttl_path, "text/turtle")
            
            # Report JSON (最後上傳)
            upload_file_to_results(video_bucket_name, report["artifacts"]["report_json"], report_data=report, content_type="application/json")
            log_message("步驟 7：上傳完成。背景任務成功。")
            

    except Exception as e:
        log_message("!!!!!!!!!!!!!! 嚴重錯誤 !!!!!!!!!!!!!!")
        log_message(f"處理影片時發生錯誤：{e}")
        trace = traceback.format_exc()
        log_message("錯誤堆疊：\n" + trace)

        # 修正 KeyError:防禦性檢查 report
        if 'report' not in locals():
            report = {
                "source_video": f"gs://{video_bucket_name}/{video_file_name}",
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "error", "errors": ["嚴重初始化錯誤，report 物件未建立。"]
            }
        
        # 確保 artifacts 和 errors key 存在
        if "artifacts" not in report:
            report["artifacts"] = {"report_json": f"{video_base_name}_report.json"}
        if "errors" not in report:
            report["errors"] = []
        # END NEW

        report["status"] = "error"
        report["errors"].append(str(e))
        report["errors"].append(trace)
        try:
            # 修正 KeyError: 使用 .get() 安全地取得檔名 
            report_json_name = report.get("artifacts", {}).get("report_json", f"{video_base_name}_report.json")
            upload_file_to_results(video_bucket_name, report_json_name, report_data=report, content_type="application/json")
            log_message("錯誤報告已上傳。")
        except Exception as ue:
            log_message(f"上傳錯誤報告失敗：{ue}")

# 腳本執行入口 

if __name__ == "__main__":
    log_message("===================================")
    log_message(f"Vertex AI Custom Job Script (v2.5-mod-notrans-fix2) starting...")
    log_message(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    parser = argparse.ArgumentParser(description="Process video into a knowledge graph.")
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket containing the video.')
    parser.add_argument('--file', type=str, required=True, help='File name of the video in the bucket.')
    args = parser.parse_args()

    log_message(f"Processing gs://{args.bucket}/{args.file}")

    try:
        process_video_for_kg(args.bucket, args.file)
        log_message("===================================")
        log_message("Script finished successfully.")
        log_message("===================================")
    except Exception as e:
        log_message("!!!!!!!!!!!!!! SCRIPT FAILED !!!!!!!!!!!!!!")
        log_message(f"Main execution error: {e}")
        traceback.print_exc()
        log_message("===================================")
        exit(1)