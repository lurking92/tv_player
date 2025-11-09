import json
import os
from typing import List, Dict, Any

# --- 設定路徑 (根據您之前的步驟結果進行更新) ---
# 階段一輸出的知識檔案
KNOWLEDGE_FILE_PATH = r"D:\senior_project\code\k-bert_training\extracted_knowledge_triplets.json"
# 階段二標註輸出的動作序列檔案
# 此檔案現在的 'event_label' 欄位是一個包含 4 個元素的列表
INPUT_ACTION_JSON_PATH = r"D:\senior_project\code\k-bert_training\action_sequences_with_labels.json"
# 階段二最終輸出的 K-BERT 訓練數據集 (JSONL 格式)
OUTPUT_KBERT_DATA_PATH = r"D:\senior_project\code\k-bert_training\kbert_train_data.jsonl"


def load_data(filepath: str) -> Any:
    """通用載入 JSON 或 JSONL 檔案。"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"錯誤：找不到檔案: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        # 假設標註檔案是單個 JSON 列表
        return json.load(f)

def actions_to_text(actions: List[str]) -> str:
    """將動作列表轉換為 K-BERT 可接受的文本序列。"""
    sentences = []
    for action in actions:
        # 將 ActionName_ID 轉換為可讀句子，例如：Walk_with_memory_loss6 -> The person walks with memory loss.
        parts = action.split('_')
        # 移除末尾的數字ID
        action_name_parts = [part for part in parts if not part.isdigit() and part != '']
        action_name = '_'.join(action_name_parts)
        
        # 簡單的動詞轉換
        if action_name:
             sentences.append(f"The person {action_name.lower().replace('_', ' ')}")
        else:
             # 如果解析失敗，保留原始動作
             sentences.append(f"The person performs action {action.lower().replace('_', ' ')}")

    return ". ".join(sentences) + "."


def combine_data_for_kbert(action_data: List[Dict], all_knowledge: List[List[str]]) -> List[Dict]:
    """
    整合動作文本、知識三元組和事件標籤。
    """
    kbert_dataset = []
    
    # 打印一次知識總數，確認無誤
    print(f"將 {len(all_knowledge)} 條知識三元組附加到每個樣本中...")

    for item in action_data:
        actions = item.get("activities", [])
        text_sequence = actions_to_text(actions)
        
        # 核心：這裡讀取的是標註工具輸出的 4 維列表標籤 (例如：[0, 1, 0, 0])
        # 為了安全，將預設值也設為 4 維列表
        label = item.get("event_label", [0, 0, 0, 0]) 
        
        # 核心步驟：將完整的知識列表附加到每個樣本中
        kbert_sample = {
            "text": text_sequence,
            # K-BERT 數據格式：將 (Subject, Predicate, Object) 列表賦值給 'triplets'
            "triplets": all_knowledge,
            #  label 現在是一個包含 4 個整數的列表
            "label": label 
        }
        kbert_dataset.append(kbert_sample)
        
    return kbert_dataset

def save_kbert_dataset(data: List[Dict], filepath: str):
    """將數據集以 JSON Lines 格式儲存。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\n成功將 K-BERT 訓練數據儲存到：{filepath} (共 {len(data)} 條記錄)")


# --- 執行流程 ---
if __name__ == "__main__":
    try:
        # 1. 載入知識 (extracted_knowledge_triplets.json)
        all_knowledge = load_data(KNOWLEDGE_FILE_PATH)
        
        # 2. 載入動作序列數據 (action_sequences_with_labels.json)
        action_data = load_data(INPUT_ACTION_JSON_PATH)

        print(f"成功載入 {len(all_knowledge)} 條知識和 {len(action_data)} 條已標註序列。")

        # 3. 整合與轉換
        final_kbert_data = combine_data_for_kbert(action_data, all_knowledge)

        # 4. 儲存 K-BERT 數據集 (kbert_train_data.jsonl)
        save_kbert_dataset(final_kbert_data, OUTPUT_KBERT_DATA_PATH)

    except FileNotFoundError as e:
        print(f"檔案讀取錯誤，請檢查路徑：{e}")
    except Exception as e:
        print(f"階段二執行失敗: {e}")