import torch
import json
import os
import random
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm # 用於顯示進度

# *** 導入您剛剛創建的自定義 DataLoader 模組 ***
from kbert_custom_dataloader import KbertDataset, create_kbert_dataloader
# **********************************************

# --- 設定路徑 ---
TRAIN_DATA_PATH = r"D:\senior_project\code\k-bert_training\kbert_train_data.jsonl"
MODEL_OUTPUT_DIR = r"D:\senior_project\code\k-bert_training\kbert_model_output_multilabel" # 建議更改輸出目錄以區分
PRE_TRAINED_MODEL = 'bert-base-uncased' 

# --- 參數設定 ---
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
# 更改為 4：對應四種特定的危險類型
NUM_LABELS = 4        
TRAIN_RATIO = 0.8     

# 定義四種風險的名稱 (供參考，未在訓練中使用)
RISK_NAMES = [
    "Fall", 
    "Walk_with_memory_loss", 
    "Fall_with_climb", 
    "Run_with_disorientation"
]

def load_and_split_data(filepath: str, train_ratio: float):
    """載入 JSONL 數據並劃分為訓練集和驗證集。"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    random.seed(42) # 確保可重複性
    random.shuffle(data)

    split_point = int(len(data) * train_ratio)
    train_data = data[:split_point]
    val_data = data[split_point:]
    
    print(f"數據集總數: {len(data)}")
    print(f"訓練集數量: {len(train_data)}")
    print(f"驗證集數量: {len(val_data)}")
    return train_data, val_data

# --- 模型訓練流程 ---
def train_kbert_detector():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # 1. 數據準備
    train_data, val_data = load_and_split_data(TRAIN_DATA_PATH, TRAIN_RATIO)
    
    # 2. 載入 Hugging Face 組件 (使用標準 BERT 作為 K-BERT 的替代品)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

    # *** 關鍵修改：設置 num_labels=4 並指定 problem_type 為多標籤 ***
    model = BertForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL, 
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification" 
    )
    
    # 3. 創建 DataLoader (使用我們自定義的)
    train_dataloader = create_kbert_dataloader(train_data, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE)
    val_dataloader = create_kbert_dataloader(val_data, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE)
    
    # 4. 訓練迴圈設置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # *** 關鍵修改：多標籤分類必須使用 BCEWithLogitsLoss ***
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"開始訓練於裝置: {device}")
    
    # 5. 訓練迴圈
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            
            # 將批次數據轉移到 device
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            # outputs.logits 的形狀為 (Batch_Size, 4)
            outputs = model(**inputs)
            logits = outputs.logits 
        
            # *** 損失計算：使用 BCEWithLogitsLoss，且 inputs['labels'] 已經是 float 類型 ***
            loss = criterion(logits, inputs['labels'])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} 訓練完成. 平均損失: {avg_train_loss:.4f}")
        
    # 6. 模型儲存
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"\n✅ 多標籤模型已儲存到：{MODEL_OUTPUT_DIR}")


# --- 執行訓練 (已簡潔化) ---
if __name__ == "__main__":
    try:
        train_kbert_detector()
    except Exception as e:
        print(f"模型訓練失敗，請檢查 PyTorch 和 Hugging Face 庫的安裝，以及數據集格式是否已修改：{e}")