import rdflib
import os
import json
import re
import glob

# 資料夾設定
TTL_FOLDER_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
OUTPUT_JSON_FOLDER = "Q2_final"
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)

# SPARQL 查詢語句
query = """
    PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX ac: <http://kgrc4si.home.kg/virtualhome2kg/ontology/action/>
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

    SELECT (STRAFTER(STR(?action), "http://kgrc4si.home.kg/virtualhome2kg/ontology/action/") AS ?actionName)
           (COUNT(?action) AS ?count)
    WHERE {
        ?activity vh2kg:hasEvent ?event .
        ?event vh2kg:action ?action .
    }
    GROUP BY ?action
    ORDER BY DESC(?count)
"""

# 處理 Episodes JSON
for json_filename in os.listdir(EPISODES_FOLDER):
    if json_filename.endswith(".json"):
        json_path = os.path.join(EPISODES_FOLDER, json_filename) # type: ignore
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        except Exception as e:
            print(f"\n[Error] Failed to load JSON: {json_filename} - {e}")
            continue

        scenario_id = episode_data.get('data', {}).get('id', '')
        match = re.search(r"scene(\d+)_Day\d+", scenario_id)
        scene_number = match.group(1) if match else "1"

        print(f"\n[Processing JSON] {json_filename}")
        print(f"  Scenario ID   : {scenario_id}")
        print(f"  Scene Number  : {scene_number}")

        activities = episode_data.get('data', {}).get('activities', [])
        ttl_files_to_use = []

        for activity in activities:
            ttl_filename = f"{activity}_scene{scene_number}.ttl"
            ttl_path = os.path.join(TTL_FOLDER_PATH, ttl_filename)
            if os.path.exists(ttl_path):
                ttl_files_to_use.append(ttl_path)
            else:
                fallback = glob.glob(os.path.join(TTL_FOLDER_PATH, f"{activity}_scene*.ttl"))
                if fallback:
                    ttl_files_to_use.append(fallback[0])

        if not ttl_files_to_use:
            fallback_scene_ttls = glob.glob(os.path.join(TTL_FOLDER_PATH, f"*scene{scene_number}.ttl"))
            ttl_files_to_use.extend(fallback_scene_ttls)

        if not ttl_files_to_use:
            scene1_fallback = glob.glob(os.path.join(TTL_FOLDER_PATH, f"*scene1.ttl"))
            ttl_files_to_use.extend(scene1_fallback)

        all_results_data = {}

        # 對每個 TTL 執行 SPARQL 查詢
        for ttl_path in ttl_files_to_use:
            print(f"\n  [TTL] {os.path.basename(ttl_path)}")
            g = rdflib.Graph()
            try:
                g.parse(ttl_path, format="ttl")
                print(f"    OK RDF Graph has {len(g)} triples.")
            except Exception as e:
                print(f"    Error loading TTL file: {e}")
                continue

            try:
                qres = g.query(query)
                print(f"    OK Query executed successfully.")
            except Exception as e:
                print(f"    Error SPARQL query: {e}")
                continue

            if not isinstance(qres, rdflib.query.Result):
                print("    Warning: Invalid SPARQL result object.")
                continue

            from rdflib.query import ResultRow
            for row in qres:
                if isinstance(row, ResultRow) and len(row) == 2:
                    action_name = str(row[0]) if row[0] else "N/A"
                    try:
                        count = int(row[1])
                        if action_name in all_results_data:
                            all_results_data[action_name] += count
                        else:
                            all_results_data[action_name] = count
                    except ValueError:
                        print(f"    Warning: Invalid count value for action '{action_name}' -> {row[1]}")
                else:
                    print(f"    Warning: Unexpected row: {row}")

        # 將結果轉換為新格式並寫入 JSON 結果檔案
        if all_results_data:
            # 將結果轉換為要求的格式
            formatted_output = {
                "name": "Test Test",  # 你可能需要調整這個名稱
                "senario": scenario_id,
                "answers": []
            }
            
            # 將動作計數轉為排序後的列表
            sorted_actions = sorted(all_results_data.items(), key=lambda x: x[1], reverse=True)
            for action_name, count in sorted_actions:
                formatted_output["answers"].append({
                    "name": action_name,
                    "number": count
                })
            
            output_filename = json_filename.replace(".json", ".json")
            output_path = os.path.join(OUTPUT_JSON_FOLDER, output_filename)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_output, f, indent=4, ensure_ascii=False)
                print(f"  OK Results saved: {output_path}")
            except Exception as e:
                print(f"  Error saving JSON: {e}")
        else:
            print("  Warning: No results gathered from TTL files.")

print("\nAll JSON files processed.")