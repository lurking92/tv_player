import os
import json
from rdflib import Graph
import re

EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
RDF_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
OUTPUT_FOLDER = "Q6_final" # 使用你指定的輸出資料夾名稱
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_action_at_10_seconds(ttl_file_path):
    g = Graph()
    try:
        g.parse(ttl_file_path, format="ttl")
    except Exception as e:
        print(f"Error parsing {ttl_file_path}: {e}")
        return None

    event_times = {}
    query_event_times = """
        PREFIX time: <http://www.w3.org/2006/time#>
        SELECT ?event ?duration
        WHERE {
            ?event a time:Duration ;
                   time:numericDuration ?duration .
        }
    """
    results_event_times = g.query(query_event_times)
    for row in results_event_times:
        event_times[str(row.event)] = float(row.duration) # type: ignore

    ordered_time_events = sorted(event_times.items(), key=lambda item: item[0]) # Order by event URI

    cumulative_time = 0
    target_time_event_uri = None
    for event_uri, duration in ordered_time_events:
        if cumulative_time < 10 and cumulative_time + duration >= 10:
            target_time_event_uri = event_uri
            break
        cumulative_time += duration

    if not target_time_event_uri:
        # If no event reaches 10 seconds, consider the last event before 10 seconds
        cumulative_time = 0
        last_event_before_10s = None
        for event_uri, duration in ordered_time_events:
            if cumulative_time + duration < 10:
                last_event_before_10s = event_uri
                cumulative_time += duration
            elif cumulative_time < 10 and cumulative_time + duration >= 10:
                target_time_event_uri = event_uri
                break
            elif cumulative_time >= 10:
                break
        if not target_time_event_uri:
            target_time_event_uri = last_event_before_10s


    if target_time_event_uri:
        query_action = f"""
            PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
            PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
            SELECT ?action
            WHERE {{
                ?event vh2kg:time <{target_time_event_uri}> ;
                       vh2kg:action ?action_uri .
                BIND (STRAFTER(STR(?action_uri), STR(vh2kg-an:)) AS ?action)
            }}
        """
        results_action = g.query(query_action)
        for row in results_action:
            return str(row.action).upper() # type: ignore

    return None

for json_filename in os.listdir(EPISODES_FOLDER):
    if json_filename.endswith(".json"):
        json_file_path = os.path.join(EPISODES_FOLDER, json_filename)
        action = None
        try:
            with open(json_file_path, 'r') as f:
                episode_data = json.load(f)
                scenario_id = episode_data['data']['id']
                match = re.search(r"scene(\d+)_Day", scenario_id)
                scene_number = "1"  # 預設值，以防沒有匹配到
                if match:
                    scene_number = match.group(1)

                if 'activities' in episode_data['data'] and episode_data['data']['activities']:
                    first_activity = episode_data['data']['activities'][0]
                    ttl_filename = f"{first_activity}_scene{scene_number}.ttl"
                    ttl_file_path = os.path.join(RDF_FOLDER, ttl_filename)

                    action = get_action_at_10_seconds(ttl_file_path)

                    output_data = {
                        "name": "Test Test",
                        "scenario": scenario_id,
                        "answers": [action] if action else []
                    }
                    print(f"Value of action for {json_filename}: {action}") # 方便查看每個檔案的處理結果
                    output_filename_base = os.path.splitext(json_filename)[0]
                    output_filename = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}.json")
                    with open(output_filename, 'w', encoding='utf-8') as outfile:
                        json.dump(output_data, outfile, indent=4, ensure_ascii=False)
                    print(f"Processed {json_filename}, output saved to {output_filename}")
                else:
                    print(f"No activities found in {json_filename}")

        except FileNotFoundError:
            print(f"Error: JSON file {json_file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {json_file_path}: {e}")
        except KeyError as e:
            print(f"KeyError in {json_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {json_file_path}: {e}")

print("Processing complete.")