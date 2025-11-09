from rdflib import Graph
import re
import os
import json

# 設定 TTL 檔案所在的資料夾路徑
TTL_FOLDER_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"  # 將路徑改為你的資料夾路徑

# 設定輸出 JSON 檔案的資料夾路徑
OUTPUT_JSON_FOLDER = "Q7_initial" # 輸出的 JSON 檔案將存放在這個資料夾中
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)  # 創建資料夾，如果已存在則不報錯

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

    SELECT ?obj1 ?relation ?obj2
    WHERE {
        ?obj1 ?relation ?obj2 .

        # 只處理場景物件
        FILTER (STRSTARTS(STR(?obj1), STR(ex:shape_state0_)))
        FILTER (STRSTARTS(STR(?obj2), STR(ex:shape_state0_)))
    }
"""

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
        except Exception as e:
            print(f"  Error loading TTL file: {e}")
            continue  # 如果載入失敗，則處理下一個檔案

        # 執行查詢
        formatted_results = []
        try:
            query_results = g.query(query_initial)
            print(f"  OK Query executed successfully. Found {len(query_results)} results.")
            for row in query_results:
                obj1 = safe_extract(str(row["obj1"]))# type: ignore
                obj2 = safe_extract(str(row["obj2"])) # type: ignore
                relation = str(row["relation"]).split("/")[-1].upper()  # 取出關係名稱，轉為大寫# type: ignore

                formatted_results.append({"obj1": obj1, "obj2": obj2, "relation": relation})

        except Exception as e:
            print(f"  Error executing SPARQL query: {e}")

        # 構建最終的 JSON 輸出
        output_json = {
            "name": "Test Test",
            "scenario": os.path.splitext(filename)[0],  # 使用檔案名稱作為 scenario
            "initial_state_relationships": formatted_results # 使用格式化後的結果
        }

        # 輸出 JSON 到檔案
        output_filename = os.path.join(OUTPUT_JSON_FOLDER, f"{os.path.splitext(filename)[0]}.json")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)
            print(f"  OK JSON output saved to: {output_filename}")
        except Exception as e:
            print(f"  Error saving JSON output to: {output_filename} - {e}")

print("\nFinished processing all TTL files.")