import rdflib

# 創建一個 RDF 圖形
g = rdflib.Graph()

# 假設 clean_kitchen1_scene1.ttl 檔案位於相同的目錄下
# 如果不是，請提供檔案的完整路徑
try:
    g.parse("clean_kitchen1_scene1.ttl", format="turtle")
except FileNotFoundError:
    print("錯誤：找不到 clean_kitchen1_scene1.ttl 檔案。請確認檔案是否存在於目前的目錄或提供正確的路徑。")
    exit()

# 定義 SPARQL 查詢
qres = g.query(
    """
PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
select DISTINCT * where {
    ex:clean_kitchen1_scene1 :hasEvent ?event .
    ?event :action ?action .
}
    """
)

# 處理查詢結果
results = qres.bindings  # 直接使用 qres.bindings 存取結果

print(f"Total triples: {len(g)}")
