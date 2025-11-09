import rdflib
import os
import csv

# 設定包含 TTL 檔案的目錄路徑
TTL_DIRECTORY_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"  # 替換為你的 TTL 目錄

# 定義 SPARQL 查詢，抓取房間名稱 + 編號
query = """
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dcterms: <http://purl.org/dc/terms/>

    SELECT DISTINCT ?roomLabel ?roomID
    WHERE {
        ?room rdf:type ?roomType ;
              rdfs:label ?roomLabel .  # 抓取房間名稱

        OPTIONAL { ?room dcterms:identifier ?roomID }  # 取得房間編號（如果有的話）

        # 確保類型是房間，而不是事件、物品
        FILTER (
            ?roomType IN (vh2kg:Bedroom, vh2kg:Kitchen, vh2kg:Livingroom, vh2kg:Bathroom)
        )
    }
"""

# 存儲所有不重複的房間
all_unique_rooms = set()

# 遍歷目錄中的所有 TTL 檔案
for filename in os.listdir(TTL_DIRECTORY_PATH):
    if filename.endswith(".ttl"):
        file_path = os.path.join(TTL_DIRECTORY_PATH, filename)

        g = rdflib.Graph()
        try:
            g.parse(file_path, format="ttl")
            print(f" OK: Loaded RDF data from {filename} ({len(g)} triples).")
            total_unique_rooms = len(all_unique_rooms)
            print(f"\n Not repeated: {total_unique_rooms}")
            print(f" RoomName: {sorted(all_unique_rooms)}")
        except Exception as e:
            print(f" Error loading TTL file {filename}: {e}")
            continue  # 如果檔案載入失敗，則跳過

        try:
            query_results = g.query(query)
            for row in query_results:
                room_label = str(row["roomLabel"]) # type: ignore
                room_id = str(row["roomID"]) if row["roomID"] else "unknown"  # type: ignore # 沒有 ID 則標記為 "unknown"

                # 組合完整的房間名稱
                full_room_name = f"{room_label}_{room_id}"
                all_unique_rooms.add(full_room_name)
        except Exception as e:
            print(f" Error executing SPARQL query on {filename}: {e}")
            
        try:
            csv_file_path = r"D:\KGRC-RDF-kgrc4si\CompleteData\unique_room_names.csv"
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                for room_name in sorted(list(all_unique_rooms)):
                    writer.writerow([room_name])
            
        except Exception as e:
            print(f" Error executing SPARQL query on {filename}: {e}")

print(f"\n not repeated room are saved to csv file: {csv_file_path}")