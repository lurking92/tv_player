from numpy import rint
from rdflib import Graph
import re
import os
import json

# 設定 TTL 檔案所在的資料夾路徑
TTL_FOLDER_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"  # 將路徑改為你的資料夾路徑

def safe_extract(uri):
    """ 從 URI 提取物件名稱（去掉 shape_state0_ 和場景名稱） """
    if isinstance(uri, str):
        match = re.search(r"shape_state0_([a-zA-Z]+)", uri)  # 找出 "shape_state0_XXX"
        if match:
            return match.group(1).capitalize()  # 取出名稱並轉為首字母大寫
    return "N/A"

# SPARQL 查詢 (移除 LIMIT)
query_initial = """
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

    SELECT ?obj1 ?obj2
    WHERE {
        ?obj1 ?relation ?obj2 .

        # 只處理場景物件
        FILTER (STRSTARTS(STR(?obj1), STR(ex:shape_state0_)))
        FILTER (STRSTARTS(STR(?obj2), STR(ex:shape_state0_)))
    }
"""

# 儲存所有不重複的物品
all_unique_objects = set()

# 需要過濾掉的房間類型
room_types_to_filter = {"Bedroom", "Bathroom", "Livingroom", "Kitchen"}

# 遍歷資料夾中的所有檔案
for filename in os.listdir(TTL_FOLDER_PATH):
    if filename.endswith(".ttl"):
        ttl_file_path = os.path.join(TTL_FOLDER_PATH, filename)
        print(f"\nProcessing file: {ttl_file_path}")

        # 讀取 RDF 檔案
        g = Graph()
        try:
            g.parse(ttl_file_path, format="ttl")
            print(f"  OK RDF Graph has {len(g)} triples.")
            print(all_unique_objects)
            print(f"\ntotal nums of objs: {len(all_unique_objects)}")
        except Exception as e:
            print(f"  Error loading TTL file: {e}")
            continue  # 如果載入失敗，則處理下一個檔案

        # 執行查詢
        try:
            query_results = g.query(query_initial)
            print(f"  OK Query executed successfully. Found {len(query_results)} results.")
            for row in query_results:
                obj1_name = safe_extract(str(row["obj1"])) # type: ignore
                obj2_name = safe_extract(str(row["obj2"])) # type: ignore

                if obj1_name not in room_types_to_filter and obj1_name != "N/A":
                    all_unique_objects.add(obj1_name)
                if obj2_name not in room_types_to_filter and obj2_name != "N/A":
                    all_unique_objects.add(obj2_name)

        except Exception as e:
            print(f"  Error executing SPARQL query: {e}")

print(all_unique_objects)
print(f"\ntotal nums of objs: {len(all_unique_objects)}")