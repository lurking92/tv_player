import rdflib
import os
import json
from glob import glob

# 1. 定義資料夾路徑 (請確保這個路徑在您的運行環境中是可訪問的)
TTL_FOLDER_PATH = r"E:\KGRC-RDF-kgrc4si\CompleteData\RDF"

# 2. 定義命名空間 (Prefixes) - 保持不變
EX_NS = rdflib.Namespace("http://kgrc4si.home.kg/virtualhome2kg/instance/")
VH2KG_NS = rdflib.Namespace("http://kgrc4si.home.kg/virtualhome2kg/ontology/")

def load_and_merge_knowledge_graphs(folder_path: str) -> rdflib.Graph:
    """
    加載資料夾內所有 TTL 檔案，並合併到一個 rdflib 圖譜中。
    """
    g = rdflib.Graph()
    ttl_files = glob(os.path.join(folder_path, "*.ttl"))
    
    if not ttl_files:
        print(f"錯誤：在 {folder_path} 中找不到任何 .ttl 檔案。請檢查路徑和副檔名。")
        return g

    print(f"找到 {len(ttl_files)} 個 TTL 檔案，開始合併...")
    for file_path in ttl_files:
        try:
            # 逐一解析檔案並將三元組添加到圖譜 g 中
            g.parse(file_path, format='turtle')
            print(f"  成功加載：{os.path.basename(file_path)}")
        except Exception as e:
            print(f"  加載 {os.path.basename(file_path)} 時發生錯誤: {e}")
            
    print(f"所有檔案合併完成。圖譜中共有 {len(g)} 個三元組。")
    return g

def extract_knowledge_triplets(merged_graph: rdflib.Graph) -> list:
    """
    從合併後的圖譜中提取物件狀態的三元組 (Subject, Predicate, Object)。
    核心查詢邏輯與單檔案版本相同。
    """
    knowledge_triplets = []

    # 核心查詢邏輯：尋找所有物件的狀態 (vh2kg:isStateOf)
    # state_entity: 狀態實體 (e.g., ex:state0_bathroom11_scene1)
    # actual_object_uri: 實際物件實體 (e.g., ex:bathroom11_scene1)
    for state_entity, _, actual_object_uri in merged_graph.triples((None, VH2KG_NS.isStateOf, None)):
        
        object_name = actual_object_uri.toPython().split('/')[-1]
        
        # 提取該狀態實體的具體狀態 (vh2kg:state)
        for _, _, state_value_uri in merged_graph.triples((state_entity, VH2KG_NS.state, None)):
            
            # 提取狀態值 (e.g., CLEAN, CLOSED)
            state_predicate = state_value_uri.toPython().split('/')[-1]
            
            # 組合三元組：(Object, is_STATE, Value)
            triplet = (object_name, "is_STATE", state_predicate)
            knowledge_triplets.append(triplet)

    # 確保知識是唯一的（防止重複加載導致的重複三元組）
    return list(set(knowledge_triplets))

# --- 執行流程 ---

# 1. 加載和合併所有 TTL 檔案
full_knowledge_graph = load_and_merge_knowledge_graphs(TTL_FOLDER_PATH)

# 2. 從合併後的圖譜中提取所有相關的背景知識
all_knowledge = extract_knowledge_triplets(full_knowledge_graph)

print(f"\n提取出的唯一知識三元組總數：{len(all_knowledge)}")

# === 新增：將提取的知識儲存到檔案中 (用於階段二) ===
OUTPUT_DIR = r"D:\senior_project\code\k-bert_training" 
OUTPUT_FILENAME = "extracted_knowledge_triplets.json"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

try:
    # 檢查並創建輸出資料夾
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已創建輸出資料夾: {OUTPUT_DIR}")

    # 將元組轉換為列表以進行 JSON 序列化
    serializable_knowledge = [list(triplet) for triplet in all_knowledge]
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        # 使用 json.dump 儲存，設置 indent=4 方便閱讀
        json.dump(serializable_knowledge, f, ensure_ascii=False, indent=4)
        
    print(f"\n 成功將 {len(serializable_knowledge)} 條知識三元組儲存到：{OUTPUT_PATH}")

except Exception as e:
    print(f"寫入檔案時發生錯誤，請檢查路徑權限或磁碟：{e}")

# 儲存後，您會在專案根目錄看到一個名為 extracted_knowledge_triplets.json 的檔案。