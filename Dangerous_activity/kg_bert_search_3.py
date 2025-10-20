import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import json
import rdflib
from rdflib.plugins.sparql import prepareQuery
import re

# --- 1. 設定與載入 BERT 模型 ---
MODEL_PATH = 'Dangerous_activity/best_dangerous_event_classifier_model/'
MAX_LEN = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"使用的設備: {DEVICE}")

if not os.path.exists(MODEL_PATH):
    print(f"錯誤: 找不到模型資料夾 '{MODEL_PATH}'。")
    exit()

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print(f"成功載入 BERT 模型和分詞器從: {MODEL_PATH}")
except Exception as e:
    print(f"載入 BERT 模型時發生錯誤: {e}")
    exit()

def classify_dangerous_action(description: str):
    """
    使用訓練好的 BERT 模型判斷事件描述是否為危險動作。
    """
    inputs = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id, probabilities[predicted_class_id]

# --- 2. 載入知識圖譜 ---
TTL_DIR = r"E:\KGRC-RDF-kgrc4si\CompleteData\RDF"
g = rdflib.Graph()
loaded_files = 0
print(f"\n--- 載入知識圖譜檔案從: {TTL_DIR} ---")
for filename in os.listdir(TTL_DIR):
    if filename.endswith(".ttl"):
        ttl_file_path = os.path.join(TTL_DIR, filename)
        try:
            g.parse(ttl_file_path, format="ttl")
            loaded_files += 1
        except Exception as e:
            print(f"載入 {filename} 時出錯: {e}")
print(f"成功載入 {loaded_files} 個 TTL 檔案。知識圖譜包含 {len(g)} 個三元組。")

# --- 3. 設定 JSON 資料夾路徑 ---
JSON_DIR = r"E:\KGRC-RDF-kgrc4si\CompleteData\Episodes"

if not os.path.exists(JSON_DIR):
    print(f"錯誤: 找不到資料夾 '{JSON_DIR}'。")
    exit()

# --- 4. 定義用於查詢單一事件的 SPARQL 查詢 (已優化) ---
sparql_query_for_event = """
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX vh2kg-an: <http://kgrc4si.home.kg/virtualhome2kg/ontology/action/>

    SELECT DISTINCT ?eventUri ?actionName ?mainObjectName ?targetObjectName
    WHERE {{
        ?eventUri vh2kg:action ?actionUri .
        FILTER regex(str(?eventUri), "{action_placeholder}", "i") .
        FILTER regex(str(?eventUri), "{scene_placeholder}", "i") .
        
        BIND(strafter(str(?actionUri), str(vh2kg-an:)) AS ?actionName)

        OPTIONAL {{
            ?eventUri vh2kg:mainObject ?mainObjectUri .
            OPTIONAL {{ ?mainObjectUri rdfs:label ?mainObjectNameLabel . }}
            BIND(IF(BOUND(?mainObjectNameLabel), ?mainObjectNameLabel, strafter(str(?mainObjectUri), str(ex:))) AS ?mainObjectName)
        }}

        OPTIONAL {{
            ?eventUri vh2kg:targetObject ?targetObjectUri .
            OPTIONAL {{ ?targetObjectUri rdfs:label ?targetObjectNameLabel . }}
            BIND(IF(BOUND(?targetObjectNameLabel), ?targetObjectNameLabel, strafter(str(?targetObjectUri), str(ex:))) AS ?targetObjectName)
        }}
    }}
    LIMIT 1 
"""
# --- 5. 設定篩選閾值 ---
DANGEROUS_THRESHOLD = 0.50 # 將這個值調高，可以篩掉信心度低的結果

# --- 6. 遍歷 JSON 檔案並處理輸出 ---
print(f"\n--- 開始分析 '{JSON_DIR}' 資料夾中的所有 JSON 檔案 ---")
print(f"*** 只會顯示信心度高於 {DANGEROUS_THRESHOLD:.2f} 的危險動作 ***")

for filename in sorted(os.listdir(JSON_DIR)):
    if filename.endswith(".json"):
        json_file_path = os.path.join(JSON_DIR, filename)
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'data' in data and 'activities' in data['data']:
                activities = data['data']['activities']
                
                match = re.search(r'(scene\d+)', filename, re.IGNORECASE)
                scene_name = match.group(1) if match else None
                
                if not scene_name:
                    continue
                
                # 用於該檔案的危險動作清單，使用集合來去重
                dangerous_actions_in_file = set()

                for i, action_uri_name in enumerate(activities):
                    query_to_execute = sparql_query_for_event.format(
                        action_placeholder=action_uri_name,
                        scene_placeholder=scene_name
                    )
                    
                    query_results = g.query(query_to_execute)
                    
                    if len(query_results) == 0:
                        continue
                        
                    for row in query_results:
                        action = str(row.actionName) if row.actionName else ""
                        main_obj = str(row.mainObjectName) if row.mainObjectName else ""
                        target_obj = str(row.targetObjectName) if row.targetObjectName else ""
                        
                        description_parts = [action]
                        if main_obj:
                            description_parts.append(f"MAIN_OBJECT:{main_obj}")
                        if target_obj:
                            description_parts.append(f"TARGET_OBJECT:{target_obj}")
                            
                        event_description = " ".join(part for part in description_parts if part).strip()
                        
                        if event_description:
                            predicted_label, probability = classify_dangerous_action(event_description)
                            
                            # 檢查是否為危險動作且信心度高於閾值
                            if predicted_label == 1 and probability >= DANGEROUS_THRESHOLD:
                                dangerous_actions_in_file.add((action_uri_name, event_description, probability))
                
                # 只有當該檔案有危險動作時才列出
                if dangerous_actions_in_file:
                    print("\n" + "="*50)
                    print(f"分析結果: {filename}")
                    print("="*50)
                    # 按信心度排序後再列印
                    sorted_actions = sorted(list(dangerous_actions_in_file), key=lambda x: x[2], reverse=True)
                    for original_action, full_description, probability in sorted_actions:
                        print(f"原始動作: '{original_action}'")
                        print(f"  完整描述: '{full_description}'")
                        print(f"  信心度: {probability:.4f}")
                        print("-" * 20)

        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")

print("\n分析結束。")