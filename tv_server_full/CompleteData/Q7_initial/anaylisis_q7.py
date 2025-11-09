from rdflib import Graph
import re

# 讀取 RDF 資料
rdf_file = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF\clean_kitchen1_scene1.ttl"
g = Graph()
g.parse(rdf_file, format="ttl")

def safe_extract(uri):
    """ 從 URI 提取物件名稱（去掉 shape_state0_ 和場景名稱） """
    if isinstance(uri, str):
        match = re.search(r"shape_state0_([a-zA-Z]+)", uri)  # 找出 "shape_state0_XXX"
        if match:
            return match.group(1).capitalize()  # 取出名稱並轉為首字母大寫
    return "N/A"

# SPARQL 查詢
query = """
PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

SELECT ?obj1 ?relation ?obj2
WHERE {
  ?obj1 ?relation ?obj2 .

  # 只處理場景物件
  FILTER (STRSTARTS(STR(?obj1), STR(ex:shape_state0_)))
  FILTER (STRSTARTS(STR(?obj2), STR(ex:shape_state0_)))
}
LIMIT 200
"""

# 執行 SPARQL 查詢
query_results = g.query(query)

# 整理結果
formatted_results = []
for row in query_results:
    obj1 = safe_extract(str(row["obj1"]))# type: ignore
    obj2 = safe_extract(str(row["obj2"])) # type: ignore
    relation = str(row["relation"]).split("/")[-1].upper()  # 取出關係名稱，轉為大寫# type: ignore

    formatted_results.append({"obj1": obj1, "obj2": obj2, "relation": relation})

# 轉為 JSON 格式
import json
output = {
    "name": "Test Test",
    "scenario": "scene1_Day1",
    "state_after_10s": formatted_results
}

# 輸出 JSON
output_json = json.dumps(output, indent=4)
print(output_json)
