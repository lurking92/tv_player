# knowledge-graph-for-elderly

##### 使用 Git LFS (Large File Storage) 來管理大型模型檔案（例如 K-BERT 模型權重）。若要成功下載所有檔案並執行訓練和預測，請在 clone 儲存庫前確保您的環境已配置 Git LFS。

---

### k-bert_training 危險事件偵測

#### 模型訓練操作步驟
可同時檢測以下四種高風險事件：
1.  Fall
2.  Walk_with_memory_loss
3.  Fall_with_climb
4.  Run_with_disorientation
   
---

#### 階段一：數據集準備 

此階段將原始動作序列轉換為 K-BERT 所需的 JSONL 訓練格式，並進行多標籤標註。

| 檔案 | 輸出 | 備註 |
| :--- | :--- | :--- |
| `1.1_labeling_tool.py` | `action_sequences_with_labels.json` | 進行交互式標註，每個序列輸出 **4 個獨立的 $0/1$ 標籤列表**。 |
| `2_dataset_generation.py` | `kbert_train_data.jsonl` | 整合標註結果與知識三元組。**最終 `label` 欄位為 4 維列表。** |

**操作步驟:**

1.  **標註**：運行 ` 1.1_labeling_tool.py`
2.  **生成數據集**：運行 ` 2_dataset_generation.py`

---

#### 階段二：模型訓練與配置

此階段使用 `kbert_train_data.jsonl` 訓練多標籤分類模型。

| 檔案 | 配置與輸出 | 輸出結果 (範例) |
| :--- | :--- | :--- |
| `kbert_custom_dataloader.py` | 標籤轉換為 `torch.float`。 | 無直接輸出，用於 `3_kbert_event_detector.py`。 |
| `3_kbert_event_detector.py` | 設置 `NUM_LABELS = 4`；使用 `BCEWithLogitsLoss`。 模型輸出至 `kbert_model_output_multilabel`。 | **訓練完成，平均損失持續下降：**<br>Epoch 1 損失: 0.4402<br>...<br>Epoch 5 損失: 0.0764 |

**操作步驟:**

1.  **執行訓練**：運行 ` 3_kbert_event_detector.py`

---

#### 階段三：模型評估 (多標籤指標)

此階段評估模型在驗證集上的性能，使用適用於多標籤分類的指標。

| 檔案 | 預測/指標計算 | 輸出結果  |
| :--- | :--- | :--- |
| `4_evaluate_model.py` | 預測使用 **Sigmoid & 0.5 閾值**；指標使用 **Micro/Macro F1-Score**。 | **各別標籤 F1-Score：**<br>Fall: 1.0000<br>Walk_with_memory_loss: 1.0000<br>Fall_with_climb: 0.0000 **(數據不足警告)**<br>Run_with_disorientation: 0.6667<br><br>**總體指標：**<br>Micro F1-Score: 0.9908<br>Macro F1-Score: 0.6667 |

**操作步驟:**

1.  **執行評估**：運行 `4_evaluate_model.py`

---

### 階段四：實際預測 (批量偵測)

此階段使用訓練好的多標籤模型對新的 JSON 動作序列進行批量風險偵測。

| 檔案 | 預測邏輯 | 輸出 |
| :--- | :--- | :--- |
| `5_predict_risk.py` | 載入模型，對輸入動作序列輸出 4 個機率和 $0/1$ 判斷。 | 輸出每個檔案的風險判斷及 4 維機率向量。 |

**操作步驟:**

1.  **執行預測**：運行 ` 5_predict_risk.py`

---
### 階段五：即時偵測流程串接 (TTL 轉換與 K-BERT 預測)
此階段將視覺管線輸出的低階 TTL 觀察，轉換為 K-BERT 模型輸入的動作序列 JSON，並進行實時風險偵測。

| 檔案 | 預測邏輯 | 輸出 |
| :--- | :--- | :--- |
| `6_ttl_to_json_converter.py` | TTL 轉換器：讀取 TTL 檔案，執行五分類邏輯，將低階觀察轉換為高階動作序列 JSON。 | 輸出 JSON 動作序列 (如 *_inferred_actions.json)。 |
| `7_json_risk_detector.py` | 實時預測器：載入訓練模型，讀取單一 JSON 動作序列，輸出 4 個機率和 $0/1$ 判斷。 | 輸出單一檔案的風險判斷及 4 維機率向量。 |

**操作步驟:**

1. 轉換 JSON：運行 6_ttl_to_json_converter.py
2. 偵測風險：運行 7_json_risk_detector.py 
