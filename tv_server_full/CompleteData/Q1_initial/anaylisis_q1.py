import rdflib

# 設定要分析的特定 TTL 檔案路徑
TTL_FILE_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF\Admire_art1_scene1.ttl"  # 你可以修改為你想要分析的特定檔案

# 讀取 RDF 檔案
g = rdflib.Graph()
try:
    g.parse(TTL_FILE_PATH, format="ttl")
    print(f"OK RDF Graph has {len(g)} triples.")
except Exception as e:
    print(f"Error loading TTL file: {e}")
    exit(1)

# 定義 SPARQL 查詢
query = """
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?roomName (COUNT(?event) AS ?entryCount)
    WHERE {
    ?event a vh2kg:Event ;
            vh2kg:action vh2kg-an:walk ;
            vh2kg:mainObject ?room .
    ?room a ?roomType .
    FILTER regex(str(?roomType), "Room|Bedroom|Kitchen|Livingroom|Bathroom")
    BIND (STRAFTER(STR(?room), STR(ex:)) AS ?roomName)
    }
    GROUP BY ?roomName
    ORDER BY DESC(?entryCount)
"""

# 執行查詢
try:
    qres = g.query(query)
    print(f"OK Query executed successfully. Result type: {type(qres)}")
except Exception as e:
    print(f"Error SPARQL query error: {e}")
    exit(1)

# 確保 qres 是有效結果
if not isinstance(qres, rdflib.query.Result):
    print("Warning: Query did not return a valid Result object. Please check the SPARQL syntax.")
    exit(1)

# 檢查是否有結果
if len(qres) == 0:
    print("Warning: No results found. Please check your RDF data and SPARQL query.")

# 遍歷查詢結果並印出
from rdflib import Literal

for row in qres:
    if isinstance(row, tuple) and len(row) >= 2:
        room_name = str(row[0]) if row[0] else "N/A"
        entry_count_value = row[1]
        try:
            if hasattr(entry_count_value, 'value'):
                entry_count = int(entry_count_value.value) # type: ignore
            else:
                entry_count = int(entry_count_value) # type: ignore
            print(f"Room: {room_name}, Entry Count: {entry_count}")
        except ValueError:
            print(f"Warning: Could not convert entry count to integer: {entry_count_value}")
    else:
        print(f"Warning: Unexpected result format: {row}")