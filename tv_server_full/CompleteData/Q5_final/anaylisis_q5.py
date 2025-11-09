import os
import json
from rdflib import Graph
import re

# 資料夾設定
EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
RDF_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
OUTPUT_FOLDER = "Q5_final"  # 使用指定的輸出資料夾名稱
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def format_time(seconds):
    """
    將秒數轉換為 "00h-00m-00s" 格式
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h-{minutes:02d}m-{secs:02d}s"

def get_grab_action_details(ttl_file_path, cumulative_time_offset=0):
    """
    1. 找到具有 grab action 的 event
    2. 根據抓取事件，利用前一個 event 的累積時間作為該事件的時間
    3. 根據 event 的 mainObject 找到該物件的 shape，並從中提取出房間名稱
    4. 考慮累積時間偏移量
    """
    g = Graph()
    try:
        g.parse(ttl_file_path, format="ttl")
        print(f"Parsed TTL file: {ttl_file_path}")
    except Exception as e:
        print(f"Error parsing {ttl_file_path}: {e}")
        return None, 0

    # 取得所有時間事件與其 duration
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
        event_times[str(row.event)] = float(row.duration)  # type: ignore
    print("Query: event_times - Results:")
    print(event_times)

    # 依 event URI 排序 (依字母排序，假設 URI 中含有順序)
    ordered_time_events = sorted(event_times.items(), key=lambda item: item[0])
    print("Ordered Time Events:")
    print(ordered_time_events)

    # 查詢具有 grab action 的 event (找出所有的grab action，不只第一個)
    query_grab_event = """
        PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
        PREFIX vh2kg-an: <http://kgrc4si.home.kg/virtualhome2kg/ontology/action/>
        SELECT ?event ?object ?timeEvent
        WHERE {
            ?event a vh2kg:Event ;
                   vh2kg:action vh2kg-an:grab ;
                   vh2kg:mainObject ?object ;
                   vh2kg:time ?timeEvent .
        }
    """
    results_grab_event = g.query(query_grab_event)
    
    grab_results = []
    for row in results_grab_event:
        grab_event_uri = str(row.event)  # type: ignore
        grab_object_uri = str(row.object)  # type: ignore
        grab_time_event_uri = str(row.timeEvent)  # type: ignore
        
        print(f"Found Grab Event URI: {grab_event_uri}")
        print(f"Grab Object URI: {grab_object_uri}")
        print(f"Grab Time Event URI: {grab_time_event_uri}")
        
        # 計算該 grab event 的時間，使用前一個 event 的累積時間
        grab_time_formatted = None
        cumulative_time = 0
        if grab_time_event_uri:
            for event_uri, duration in ordered_time_events:
                if event_uri == grab_time_event_uri:
                    break
                cumulative_time += duration
            
            # 加上前面 activity 的累積時間偏移量
            total_time = cumulative_time + cumulative_time_offset
            grab_time_formatted = format_time(total_time)
            print(f"Grab Time Formatted: {grab_time_formatted}")

        # 從 grab_object_uri 提取物件基本名稱 (例如 "pillow189")
        room_name = None
        if grab_object_uri:
            object_id = grab_object_uri.split('/')[-1]
            object_base = object_id.split('_')[0]
            print(f"Object ID: {object_id}")
            print(f"Object Base: {object_base}")

            # 使用 SPARQL 直接查詢，不依賴 URI 結構
            query_room = f"""
                PREFIX vh2kg: <http://kgrc4si.home.kg/virtualhome2kg/ontology/>
                PREFIX x3do: <http://www.web3d.org/specifications/x3d-namespace#>
                PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

                SELECT ?shape ?inside
                WHERE {{
                    ?shape a x3do:Shape ;
                           vh2kg:inside ?inside .

                    # 匹配所有可能包含該物件名稱的shape URI
                    FILTER(CONTAINS(STR(?shape), "{object_base}"))
                }}
            """
            print("Executing room query:")
            print(query_room)

            results_room = g.query(query_room)
            print(f"Found {len(list(results_room))} results")

            # 重新執行查詢並處理結果
            results_room = g.query(query_room)
            for row in results_room:
                shape_uri = str(row.shape)  # type: ignore
                inside_uri = str(row.inside)  # type: ignore
                print(f"Shape URI: {shape_uri}")
                print(f"Inside URI: {inside_uri}")

                # 從inside_uri中提取房間名稱
                room_patterns = ["bedroom", "kitchen", "bathroom", "livingroom", "living_room", "dining_room", "diningroom", "office"]
                for pattern in room_patterns:
                    if pattern in inside_uri.lower():
                        room_name = pattern.replace('_', '')
                        print(f"Found room name: {room_name}")
                        break

                if room_name:
                    break

            # 如果以上方法找不到房間，嘗試使用直接查詢
            if not room_name:
                print("嘗試直接查詢所有三元組...")
                # 直接把所有包含特定物件ID的三元組都找出來
                for s, p, o in g:
                    s_str = str(s)
                    p_str = str(p)
                    o_str = str(o)

                    # 檢查是否為我們要找的物件
                    if object_base in s_str:
                        print(f"Subject: {s_str}")
                        print(f"Predicate: {p_str}")
                        print(f"Object: {o_str}")

                        # 檢查是否有inside關係
                        if "inside" in p_str:
                            inside_uri = o_str
                            print(f"Found inside relation: {inside_uri}")

                            # 從inside_uri提取房間名稱
                            room_patterns = ["bedroom", "kitchen", "bathroom", "livingroom"]
                            for pattern in room_patterns:
                                if pattern in inside_uri.lower():
                                    room_name = pattern.replace('_', '')
                                    print(f"Found room name from triples: {room_name}")
                                    break

                            if room_name:
                                break

        print(f"Final Room Name: {room_name}")

        # 取得物件名稱，只保留字母部分並轉換為大寫 (例如 "pillow189" 取 "PILLOW")
        object_name = None
        if grab_object_uri:
            object_id = grab_object_uri.split('/')[-1]
            object_name = ''.join(c for c in object_id.split('_')[0] if not c.isdigit()).upper()
        print(f"Object Name: {object_name}")

        if grab_time_formatted and room_name and object_name:
            grab_results.append({
                "time": grab_time_formatted,
                "room": room_name,
                "object": object_name
            })

    # 計算此 TTL 文件的總持續時間，用於下一個 activity 的時間偏移
    total_duration = sum(duration for _, duration in event_times.items())
    
    return grab_results, total_duration

# 遍歷所有的 JSON 檔
for json_filename in os.listdir(EPISODES_FOLDER):
    if json_filename.endswith(".json"):
        json_file_path = os.path.join(EPISODES_FOLDER, json_filename)
        try:
            with open(json_file_path, 'r') as f:
                print(f"\nProcessing JSON file: {json_file_path}")
                episode_data = json.load(f)
                scenario_id = episode_data['data']['id']
                print(f"Scenario ID: {scenario_id}")
                
                match = re.search(r"scene(\d+)_Day", scenario_id)
                scene_number = "1"  # 預設值，以防沒有匹配到
                if match:
                    scene_number = match.group(1)
                print(f"Scene Number: {scene_number}")

                # 初始化此 JSON 檔案的結果
                all_grab_details = []
                
                # 跨不同 activities 累積時間
                cumulative_time_offset = 0
                
                if 'activities' in episode_data['data'] and episode_data['data']['activities']:
                    activities = episode_data['data']['activities']
                    print(f"Found {len(activities)} activities")
                    
                    for activity in activities:
                        ttl_filename = f"{activity}_scene{scene_number}.ttl"
                        ttl_file_path = os.path.join(RDF_FOLDER, ttl_filename)
                        print(f"\nProcessing activity: {activity}")
                        print(f"TTL file path: {ttl_file_path}")
                        
                        # 檢查文件是否存在
                        if not os.path.exists(ttl_file_path):
                            print(f"TTL file not found: {ttl_file_path}")
                            continue
                        
                        # 獲取此活動中的抓取事件詳情，並得到此活動的總持續時間
                        grab_details, activity_duration = get_grab_action_details(ttl_file_path, cumulative_time_offset)
                        
                        print(f"Activity duration: {activity_duration} seconds")
                        print(f"Current cumulative time offset: {cumulative_time_offset} seconds")
                        
                        # 更新累積時間偏移量
                        cumulative_time_offset += activity_duration
                        print(f"New cumulative time offset: {cumulative_time_offset} seconds")
                        
                        # 如果有抓取事件，加入結果列表
                        if grab_details and len(grab_details) > 0:
                            all_grab_details.extend(grab_details)
                            print(f"Added {len(grab_details)} grab details. Total now: {len(all_grab_details)}")
                
                # 處理完此 JSON 檔後保存結果
                output_data = {
                    "name": "Test Test",
                    "senario": scenario_id,
                    "answers": all_grab_details
                }
                
                output_filename_base = os.path.splitext(json_filename)[0]
                output_filename = os.path.join(OUTPUT_FOLDER, f"{output_filename_base}.json")
                
                with open(output_filename, 'w', encoding='utf-8') as outfile:
                    json.dump(output_data, outfile, indent=4, ensure_ascii=False)
                
                print(f"\nCompleted processing {json_filename}")
                print(f"Found {len(all_grab_details)} grab details")
                print(f"Output saved to {output_filename}")

        except FileNotFoundError:
            print(f"Error: JSON file {json_file_path} not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {json_file_path}: {e}")
        except KeyError as e:
            print(f"KeyError in {json_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {json_file_path}: {e}")

print("\nProcessing complete.")