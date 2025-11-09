import json
import os
from glob import glob
from typing import List, Dict, Any

# --- 設定路徑 (根據您的輸入進行更新) ---
RAW_JSON_FOLDER = r"E:\KGRC-RDF-kgrc4si\CompleteData\Episodes" 
KNOWLEDGE_FILE_PATH = r"D:\senior_project\code\k-bert_training\extracted_knowledge_triplets.json"
# 階段二最終輸出的標註數據集檔案
OUTPUT_JSON_PATH = r"D:\senior_project\code\k-bert_training\action_sequences_with_labels.json"

# --- 參數設定 ---
SEQUENCE_WINDOW_SIZE = 5 
# 多標籤分類的四個獨立標籤
RISK_NAMES = [
    "Fall", 
    "Walk_with_memory_loss", 
    "Fall_with_climb", 
    "Run_with_disorientation"
]

def load_knowledge(filepath: str) -> List[List[str]]:
    """載入階段一提取的知識三元組。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到知識檔案 {filepath}")
        return []
    except Exception as e:
        print(f"載入知識檔案時發生錯誤: {e}")
        return []

def generate_sequences_and_label(raw_json_folder: str, all_knowledge: List[List[str]]) -> List[Dict]:
    """
    讀取所有原始 JSON，切分動作序列，並進行輔助多標籤標註。
    """
    final_labeled_data = []
    
    # 將知識轉換為字典，以便查找
    knowledge_dict = {s: o for s, p, o in all_knowledge if p == 'is_STATE'}

    json_files_paths = glob(os.path.join(raw_json_folder, "*.json"))
    
    if not json_files_paths:
        print(f"錯誤：在 {raw_json_folder} 中找不到任何 .json 檔案。請檢查路徑和副檔名。")
        return []
        
    print(f"找到 {len(json_files_paths)} 個原始動作紀錄檔案，開始進行標註...")

    for filepath in json_files_paths:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告：檔案 {filename} 不是有效的 JSON 格式，跳過。")
            continue
            
        activities_data = raw_data.get("data", {})
        scene_id = activities_data.get("id", filename.replace('.json', ''))
        all_actions = activities_data.get("activities", [])
        
        if not all_actions:
            print(f"警告：檔案 {filename} (ID: {scene_id}) 中沒有找到 'activities'，跳過。")
            continue
        
        print(f"\n--- 處理檔案: {filename} (總動作數: {len(all_actions)}) ---")

        # 1. 序列切分 (Sliding Window)
        for i in range(0, len(all_actions) - SEQUENCE_WINDOW_SIZE + 1, 1): 
            current_sequence = all_actions[i:i + SEQUENCE_WINDOW_SIZE]
            
            # --- 輔助判斷：找出序列相關的危險知識 ---
            relevant_knowledge = [
                f"{obj}: {state}" 
                for obj, state in knowledge_dict.items() 
                if scene_id in obj and state in ["DIRTY", "WET", "OPEN", "HIGH", "UNLOCKED"] 
            ]
            
            print("\n" + "="*80)
            print(f"序列ID: {scene_id}_seq{i}")
            print(f"動作序列 ({SEQUENCE_WINDOW_SIZE} 個動作): {current_sequence}")
            
            if relevant_knowledge:
                print(f"潛在危險環境狀態輔助: {relevant_knowledge[:5]}...")
            else:
                print("潛在危險環境狀態輔助: (未找到特定危險狀態)")

            # 2. 手動標註環節：詢問四個獨立問題
            multi_label = []
            
            print("\n--- 請為此序列標註四種風險 (1: YES, 0: NO, [s]: 跳過此序列) ---")
            
            skip_sequence = False
            for risk_name in RISK_NAMES:
                while True:
                    try:
                        label_input = input(f"-> 是否包含【{risk_name}】的潛在/已發生危險 (1/0)？: ")
                        if label_input.lower() == 's':
                            skip_sequence = True
                            break
                        risk_label = int(label_input)
                        if risk_label in [0, 1]:
                            multi_label.append(risk_label)
                            break
                        else:
                            print("無效輸入，請輸入 1 或 0 或 [s]。")
                    except ValueError:
                        print("無效輸入，請輸入數字或 [s]。")
                
                if skip_sequence:
                    break

            # 3. 儲存結果
            if not skip_sequence:
                # 儲存的 event_label 現在是一個包含 4 個元素的列表！
                final_labeled_data.append({
                    "sequence_id": f"{scene_id}_seq{i}",
                    "activities": current_sequence,
                    "event_label": multi_label, # 儲存多標籤列表
                })
            
    return final_labeled_data

# --- 執行流程 ---
if __name__ == "__main__":
    
    # 1. 載入知識
    all_knowledge_triplets = load_knowledge(KNOWLEDGE_FILE_PATH)
    
    if not all_knowledge_triplets:
        print("無法進行標註，請先確保階段一正確執行並產生知識檔案。")
    else:
        # 2. 進行交互式標註
        labeled_data = generate_sequences_and_label(RAW_JSON_FOLDER, all_knowledge_triplets)
        
        # 3. 儲存最終的 K-BERT 數據集 (action_sequences_with_labels.json)
        if labeled_data:
            os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, ensure_ascii=False, indent=4)
                
            print(f"\n\n*** 多標籤標註完成！ ***")
            print(f"✅ 共 {len(labeled_data)} 條序列，已儲存到 {OUTPUT_JSON_PATH}")
        else:
            print("未生成任何標註數據。")