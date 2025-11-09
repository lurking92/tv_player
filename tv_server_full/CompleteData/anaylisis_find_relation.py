# 只有"CLOSE","FACING","INSIDE","ON",這四個關係
# 這個檔案是用來找出所有的關係

from rdflib import Graph
import re
import json
import os

# 設定包含 TTL 檔案的目錄路徑
TTL_DIRECTORY_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"  # 替換為你的 TTL 目錄

# SPARQL 查詢，只抓取 relation
query = """
PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

SELECT DISTINCT ?relation
WHERE {
  ?obj1 ?relation ?obj2 .

  # 只處理場景物件
  FILTER (STRSTARTS(STR(?obj1), STR(ex:shape_state0_)))
  FILTER (STRSTARTS(STR(?obj2), STR(ex:shape_state0_)))
}
"""

all_relations = {}

# 遍歷目錄中的所有 TTL 檔案
for filename in os.listdir(TTL_DIRECTORY_PATH):
    if filename.endswith(".ttl"):
        file_path = os.path.join(TTL_DIRECTORY_PATH, filename)
        file_base_name = os.path.splitext(filename)[0]  # 去掉副檔名的檔名

        g = Graph()
        try:
            g.parse(file_path, format="ttl")
            print(f"OK: Loaded RDF data from {filename} ({len(g)} triples).")

            # 執行 SPARQL 查詢
            query_results = g.query(query)

            # 儲存所有不重複的 relation
            unique_relations = set()

            # 整理結果
            for row in query_results:
                relation_uri = str(row["relation"]) # type: ignore
                relation_name = relation_uri.split("/")[-1].upper()  # 取出關係名稱，轉為大寫
                unique_relations.add(relation_name)

            all_relations[file_base_name] = sorted(list(unique_relations))
            print(f"  Found {len(unique_relations)} unique relations.")
            print(f"  {sorted(list(unique_relations))}")

        except Exception as e:
            print(f"Error loading or querying TTL file {filename}: {e}")

# 轉為 JSON 格式
output = {
    "name": "All Files",
    "scenario": "All Scenarios",
    "relations_by_file": all_relations
}

# 輸出 JSON
output_json = json.dumps(output, indent=4)
print(output_json)