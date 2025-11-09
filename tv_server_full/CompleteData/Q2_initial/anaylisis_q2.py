import rdflib

# 設定 TTL 檔案路徑
TTL_FILE_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF\clean_kitchen1_scene1.ttl"

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
    PREFIX ho: <http://www.owl-ontologies.com/VirtualHome.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX : <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX ac: <http://kgrc4si.home.kg/virtualhome2kg/ontology/action/>
PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

SELECT (STRAFTER(STR(?action), "http://kgrc4si.home.kg/virtualhome2kg/ontology/action/") AS ?actionName) 
       (COUNT(?action) AS ?count ) 
WHERE { 
    #?activity :virtualHome ex:scene .  # 限定在 scene1-7 內
    ?activity :hasEvent ?event .        # 取得該場景內的事件
    ?event :action ?action .            # 取得事件的 action
}
GROUP BY ?action  # 按 action 名稱分組計數
ORDER BY DESC(?count)  # 按次數降序排列

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

# 遍歷查詢結果
for row in qres:
    if isinstance(row, tuple) and len(row) >= 2:
        event = str(row[0]) if row[0] else "N/A"
        action = str(row[1]) if row[1] else "N/A"
        print(f"OK {event} \t {action}")
    else:
        print(f"Warning: Unexpected result format: {row}")