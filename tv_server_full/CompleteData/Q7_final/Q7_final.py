import os
import json
from rdflib import Graph
import re
import glob

EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
RDF_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
OUTPUT_FOLDER = "Q7_final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def safe_extract(uri):
    """Extract object name from URI"""
    if isinstance(uri, str):
        match = re.search(r"shape_state\d+_([a-zA-Z]+)", uri)
        if match:
            return match.group(1).capitalize()
    return "N/A"

def get_initial_state_relationships(ttl_file_path):
    g = Graph()
    try:
        g.parse(ttl_file_path, format="ttl")
        print(f"        RDF Graph loaded: {ttl_file_path} ({len(g)} triples)")
    except Exception as e:
        print(f"        Error parsing {ttl_file_path}: {e}")
        return []

    relationship_set = set()
    formatted_results = []

    query_initial = """
        PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
        PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>

        SELECT ?obj1 ?relation ?obj2
        WHERE {
            ?obj1 ?relation ?obj2 .
            FILTER (STRSTARTS(STR(?obj1), STR(ex:shape_state)))
            FILTER (STRSTARTS(STR(?obj2), STR(ex:shape_state)))
        }
    """

    try:
        query_results = g.query(query_initial)
        print(f"        SPARQL executed successfully: {len(query_results)} results")
        for row in query_results:
            obj1 = safe_extract(str(row["obj1"])) # type: ignore
            obj2 = safe_extract(str(row["obj2"])) # type: ignore
            relation = str(row["relation"]).split("/")[-1].upper() # type: ignore

            if obj1 == "N/A" or obj2 == "N/A":
                continue

            # 為了排除同一TTL檔內的重複關係（但允許跨檔重複）
            rel_key = f"{obj1}|{relation}|{obj2}"
            if rel_key not in relationship_set:
                relationship_set.add(rel_key)
                formatted_results.append({
                    "obj1": obj1,
                    "obj2": obj2,
                    "relation": relation
                })
    except Exception as e:
        print(f"        Error executing SPARQL query: {e}")

    print(f"        Unique relationships extracted: {len(formatted_results)}")
    return formatted_results

# 遍歷 Episodes JSON 檔案
for json_filename in os.listdir(EPISODES_FOLDER): # type: ignore
    if json_filename.endswith(".json"):
        json_file_path = os.path.join(EPISODES_FOLDER, json_filename)

        try:
            with open(json_file_path, 'r') as f:
                episode_data = json.load(f)
                scenario_id = episode_data['data']['id']

                match = re.search(r"scene(\d+)_Day\d+", scenario_id)
                scene_number = "1"
                if match:
                    scene_number = match.group(1)

                print(f"\n[Processing File] {json_filename}")
                print(f"  Scenario ID   : {scenario_id}")
                print(f"  Scene Number  : {scene_number}")

                if 'activities' in episode_data['data'] and episode_data['data']['activities']:
                    all_relationships = []

                    for activity in episode_data['data']['activities']:
                        print(f"    [Activity] {activity}")
                        ttl_filename = f"{activity}_scene{scene_number}.ttl"
                        ttl_file_path = os.path.join(RDF_FOLDER, ttl_filename)
                        print(f"      Looking for TTL: {ttl_filename}")

                        if os.path.exists(ttl_file_path):
                            print(f"        Found TTL file for activity")
                            relationships = get_initial_state_relationships(ttl_file_path)
                            all_relationships.extend(relationships)
                            print(f"        Relationships added: {len(relationships)}")
                        else:
                            print(f"        TTL file not found: {ttl_filename}")
                            alternative_ttls = glob.glob(os.path.join(RDF_FOLDER, f"{activity}_scene*.ttl"))
                            if alternative_ttls:
                                print(f"        Trying alternative TTL: {os.path.basename(alternative_ttls[0])}")
                                relationships = get_initial_state_relationships(alternative_ttls[0])
                                all_relationships.extend(relationships)
                                print(f"        Relationships added (alternative): {len(relationships)}")

                    # 若找不到任何關係，使用 fallback 機制
                    if not all_relationships:
                        print(f"    No relationships from activities, trying fallback methods")
                        all_scene_ttls = glob.glob(os.path.join(RDF_FOLDER, f"*_scene{scene_number}.ttl"))
                        if all_scene_ttls:
                            print(f"      Using fallback TTL: {os.path.basename(all_scene_ttls[0])}")
                            all_relationships = get_initial_state_relationships(all_scene_ttls[0])
                        else:
                            print(f"      No TTL files found for scene {scene_number}")
                            fallback_files = glob.glob(os.path.join(RDF_FOLDER, f"*_scene1.ttl"))
                            if fallback_files:
                                print(f"      Fallback: Using scene1 TTL file: {os.path.basename(fallback_files[0])}")
                                all_relationships = get_initial_state_relationships(fallback_files[0])

                    print(f"  Total relationships collected: {len(all_relationships)}")

                    output_data = {
                        "name": "Test Test",
                        "scenario": scenario_id,
                        "answers": all_relationships
                    }

                    output_filename_base = os.path.splitext(json_filename)[0]
                    output_filename = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}.json")
                    with open(output_filename, 'w', encoding='utf-8') as outfile:
                        json.dump(output_data, outfile, indent=4, ensure_ascii=False)
                    print(f"  Output saved to: {output_filename}")
                else:
                    print(f"  No activities found in {json_filename}")

        except FileNotFoundError:
            print(f"  Error: JSON file {json_file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON: {e}")
        except KeyError as e:
            print(f"  Missing key in JSON: {e}")
        except Exception as e:
            print(f"  Unexpected error while processing {json_file_path}: {e}")

print("\nAll files processed.")
