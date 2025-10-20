import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch.nn.functional as F # 新增導入

# --- 設定參數 ---
CSV_FILE_PATH = "Dangerous_activity/dangerous_training_data_unlabeled.csv"
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_RUNS = 5

print(f"使用的設備: {DEVICE}")

# --- 自定義數據集類 ---
class EventDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 載入數據 (這部分只執行一次) ---
if not os.path.exists(CSV_FILE_PATH):
    print(f"錯誤: 找不到CSV檔案 '{CSV_FILE_PATH}'。請確認路徑正確。")
    exit()

df = pd.read_csv(CSV_FILE_PATH)
df['is_dangerous_event'] = pd.to_numeric(df['is_dangerous_event'], errors='coerce')
df.dropna(subset=['is_dangerous_event'], inplace=True)
df['is_dangerous_event'] = df['is_dangerous_event'].astype(int)

texts_full = df['event_description'].tolist()
labels_full = df['is_dangerous_event'].tolist()

print(f"載入 {len(texts_full)} 筆數據。其中 {sum(labels_full)} 筆標記為危險事件，{len(labels_full) - sum(labels_full)} 筆標記為非危險事件。")

# --- 初始化分詞器 (只執行一次) ---
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# --- 儲存每次運行的結果 ---
all_test_accuracy = []
all_test_precision = []
all_test_recall = []
all_test_f1 = []
best_f1_overall = -1
best_model_for_saving = None

print(f"\n--- 開始 {NUM_RUNS} 次訓練運行 ---")

for run in range(NUM_RUNS):
    print(f"\n--- 運行 {run + 1}/{NUM_RUNS} ---")

    # 在這裡計算或設定類別權重
    class_weights = torch.tensor([0.635, 2.35], dtype=torch.float).to(DEVICE)
    # 每次運行都重新初始化模型
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 每次運行都重新劃分數據集 ---
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts_full, labels_full, test_size=0.1, random_state=None, stratify=labels_full
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=(0.1 / 0.9), random_state=None, stratify=train_val_labels
    )

    train_dataset = EventDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = EventDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = EventDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"運行 {run + 1} - 訓練集大小: {len(train_dataset)}, 驗證集大小: {len(val_dataset)}, 測試集大小: {len(test_dataset)}")

    # --- 訓練循環 ---
    best_val_f1 = -1
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights) # 使用 F.cross_entropy 來計算加權損失
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1} 訓練平均損失: {avg_train_loss:.4f}")

        # --- 驗證階段 ---
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask) # 這裡也只傳入 input_ids 和 attention_mask
                loss = F.cross_entropy(outputs.logits, labels, weight=class_weights) # 這裡也使用 F.cross_entropy 計算加權損失
                logits = outputs.logits
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).flatten()
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary', zero_division=0)

        print(f"  Epoch {epoch + 1} 驗證平均損失: {avg_val_loss:.4f}")
        print(f"  Epoch {epoch + 1} 驗證準確度: {val_accuracy:.4f}")
        print(f"  Epoch {epoch + 1} 驗證精確度: {val_precision:.4f}")
        print(f"  Epoch {epoch + 1} 驗證召回率: {val_recall:.4f}")
        print(f"  Epoch {epoch + 1} 驗證 F1-Score: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()

    # --- 測試階段 (使用當前運行中驗證集表現最佳的模型) ---
    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).flatten()
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_true, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_true, test_preds, average='binary', zero_division=0)

    print(f"\n運行 {run + 1} - 測試集準確度: {test_accuracy:.4f}")
    print(f"運行 {run + 1} - 測試集精確度: {test_precision:.4f}")
    print(f"運行 {run + 1} - 測試集召回率: {test_recall:.4f}")
    print(f"運行 {run + 1} - 測試集 F1-Score: {test_f1:.4f}")

    all_test_accuracy.append(test_accuracy)
    all_test_precision.append(test_precision)
    all_test_recall.append(test_recall)
    all_test_f1.append(test_f1)

    if test_f1 > best_f1_overall:
        best_f1_overall = test_f1
        best_model_for_saving = model
        print(f"  運行 {run + 1} 的測試 F1-Score ({test_f1:.4f}) 是當前最佳，已暫存模型。")

print("\n--- 所有運行完成 ---")
print(f"總運行次數: {NUM_RUNS}")
print(f"測試集平均準確度: {np.mean(all_test_accuracy):.4f} ± {np.std(all_test_accuracy):.4f}")
print(f"測試集平均精確度: {np.mean(all_test_precision):.4f} ± {np.std(all_test_precision):.4f}")
print(f"測試集平均召回率: {np.mean(all_test_recall):.4f} ± {np.std(all_test_recall):.4f}")
print(f"測試集平均 F1-Score: {np.mean(all_test_f1):.4f} ± {np.std(all_test_f1):.4f}")


# 在所有運行結束後，保存表現最佳的模型
if best_model_for_saving is not None:
    output_dir = 'Dangerous_activity/best_dangerous_event_classifier_model/'
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = best_model_for_saving.module if hasattr(best_model_for_saving, 'module') else best_model_for_saving
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n已將最佳模型和分詞器保存到: {output_dir}")
else:
    print("\n沒有模型被保存，因為沒有達到有效的 F1-Score。")