import rdflib
import os
import csv

TTL_DIR = r"E:\KGRC-RDF-kgrc4si\CompleteData\RDF" # 您的 TTL 檔案目錄
OUTPUT_CSV = "Dangerous_activity/dangerous_training_data_unlabeled.csv" # 輸出的未標記資料檔

# **** 修改後的 SPARQL 查詢 ****
sparql_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX vh2kg-an: <http://kgrc4si.home.kg/virtualhome2kg/ontology/action-name/>
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/> # 根據您的數據，可能需要這個前綴來處理實例 URI

    SELECT ?actionName ?mainObjectName ?targetObjectName
    WHERE {
        ?eventUri a vh2kg:Event . # 找到所有 Event 實例

        # 提取動作名稱：從 vh2kg:action 屬性中提取 URI 的本地名稱
        ?eventUri vh2kg:action ?actionUri .
        BIND(strafter(str(?actionUri), str(vh2kg-an:)) AS ?actionName)

        OPTIONAL {
            ?eventUri vh2kg:mainObject ?mainObjectUri .
            OPTIONAL { ?mainObjectUri rdfs:label ?mainObjectNameLabel . } # 優先使用 rdfs:label
            BIND(IF(BOUND(?mainObjectNameLabel), ?mainObjectNameLabel, strafter(str(?mainObjectUri), str(ex:))) AS ?mainObjectName) # 否則從URI提取
        }

        OPTIONAL {
            ?eventUri vh2kg:targetObject ?targetObjectUri .
            OPTIONAL { ?targetObjectUri rdfs:label ?targetObjectNameLabel . } # 優先使用 rdfs:label
            BIND(IF(BOUND(?targetObjectNameLabel), ?targetObjectNameLabel, strafter(str(?targetObjectUri), str(ex:))) AS ?targetObjectName) # 否則從URI提取
        }
    }
"""

all_event_descriptions = set() # 使用 set 來自動去除重複的事件描述

for filename in os.listdir(TTL_DIR):
    if filename.endswith(".ttl"):
        ttl_file_path = os.path.join(TTL_DIR, filename)
        g = rdflib.Graph()
        try:
            g.parse(ttl_file_path, format="ttl")
            results = g.query(sparql_query)
            for row in results:
                action = str(row.actionName) if row.actionName else ""
                main_obj = str(row.mainObjectName) if row.mainObjectName else ""
                target_obj = str(row.targetObjectName) if row.targetObjectName else ""

                description_parts = [action]
                if main_obj:
                    description_parts.append(f"MAIN_OBJECT:{main_obj}")
                if target_obj:
                    description_parts.append(f"TARGET_OBJECT:{target_obj}")

                # 組合成描述句，例如 "GRAB MAIN_OBJECT:BREADSLICE" 或 "WALK"
                description = " ".join(part for part in description_parts if part).strip()

                if description:
                    all_event_descriptions.add(description)
        except Exception as e:
            print(f"處理 {filename} 時出錯: {e}")

# 將所有不重複的事件描述寫入 CSV 檔案
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['event_description', 'is_dangerous_event']) # 寫入標頭
    for desc in sorted(list(all_event_descriptions)):
        writer.writerow([desc, '']) # is_kitchen_event 欄暫時留空

print(f"成功將 {len(all_event_descriptions)} 筆不重複的事件描述匯出至 {OUTPUT_CSV}")
print("下一步：請手動打開這個 CSV 檔案，並在 'is_dangerous_event' 欄填入 1 (是) 或 0 (否)。")