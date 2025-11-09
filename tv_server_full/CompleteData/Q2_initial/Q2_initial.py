import rdflib
import os
import json

# 設定 TTL 檔案所在的資料夾路徑
TTL_FOLDER_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"  # 將路徑改為你的資料夾路徑

# 設定輸出 JSON 檔案的資料夾路徑
OUTPUT_JSON_FOLDER = "Q2"
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)  # 創建資料夾，如果已存在則不報錯

# 定義 SPARQL 查詢
query = """
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX ac: <http://kgrc4si.home.kg/virtualhome2kg/ontology/action/>
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

    SELECT (STRAFTER(STR(?action), "http://kgrc4si.home.kg/virtualhome2kg/ontology/action/") AS ?actionName)
        (COUNT(?action) AS ?count )
    WHERE {
        ?activity vh2kg:hasEvent ?event .
        ?event vh2kg:action ?action .
    }
    GROUP BY ?action
    ORDER BY DESC(?count)
"""

# 遍歷資料夾中的所有檔案
for filename in os.listdir(TTL_FOLDER_PATH):
    if filename.endswith(".ttl"):
        ttl_file_path = os.path.join(TTL_FOLDER_PATH, filename)
        print(f"\nProcessing file: {ttl_file_path}")

        # 讀取 RDF 檔案
        g = rdflib.Graph()
        try:
            g.parse(ttl_file_path, format="ttl")
            print(f"  OK RDF Graph has {len(g)} triples.")
        except Exception as e:
            print(f"  Error loading TTL file: {e}")
            continue  # 如果載入失敗，繼續處理下一個檔案

        # 執行查詢
        try:
            qres = g.query(query)
            print(f"  OK Query executed successfully. Result type: {type(qres)}")
        except Exception as e:
            print(f"  Error SPARQL query error: {e}")
            continue  # 如果查詢失敗，繼續處理下一個檔案

        # 確保 qres 是有效結果
        if not isinstance(qres, rdflib.query.Result):
            print("  Warning: Query did not return a valid Result object. Please check the SPARQL syntax.")
            continue

        # 檢查是否有結果並寫入 JSON 檔案
        results_data = {}
        if len(qres) > 0:
            from rdflib.query import ResultRow  # 確保 ResultRow 類別可用
            for row in qres:
                if isinstance(row, ResultRow) and len(row) == 2:
                    action_name = str(row[0]) if row[0] else "N/A"
                    count_value = row[1]
                    try:
                        count = int(count_value)
                        results_data[action_name] = count
                    except ValueError:
                        print(f"  Warning: Could not convert count to integer for action '{action_name}'. Value: {count_value}")
                else:
                    print(f"  Warning: Unexpected row type or number of columns: {row}")

            json_filename = filename.replace(".ttl", ".json")
            json_filepath = os.path.join(OUTPUT_JSON_FOLDER, json_filename)

            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=4, ensure_ascii=False)
                print(f"  OK Results saved to JSON: {json_filepath}")
            except Exception as e:
                print(f"  Error saving results to JSON: {e}")
        else:
            print("  Warning: No query results found for this file.")

print("\nFinished processing all TTL files and saving results to JSON.")