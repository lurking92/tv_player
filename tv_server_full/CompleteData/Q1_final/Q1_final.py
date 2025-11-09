import rdflib
import os
import json
import re
import glob
from collections import defaultdict

TTL_FOLDER_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
OUTPUT_JSON_FOLDER = "Q1_final"
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)

def extract_room_entries(ttl_content):
    """
    Extract actual room entries from TTL content.
    We want to count transitions where a character enters a room.
    """
    # Look for specific triples that indicate entering a room
    # The pattern matches where a character transitions to being inside a room
    room_pattern = r'vh2kg:inside\s+ex:shape_state\d+_(bedroom\d+|bathroom\d+|kitchen\d+|livingroom\d+)'
    
    # First, split the TTL content into separate statements/triples
    statements = re.split(r'\s*\.\s*', ttl_content)
    
    room_entries = []
    for statement in statements:
        # Find statements that contain character entering a room
        # We're looking for lines containing "vh2kg:hasPrevState" followed by "vh2kg:inside"
        # which would indicate a state transition into a room
        if "vh2kg:hasPrevState" in statement and "vh2kg:inside" in statement:
            matches = re.search(room_pattern, statement)
            if matches:
                room_entries.append(matches.group(1))
    
    return room_entries

def count_room_entries(ttl_files, scene_number):
    """
    Count how many times the character enters each room from the list of TTL files.
    """
    room_entry_counts = defaultdict(int)
    for ttl_path in ttl_files:
        try:
            with open(ttl_path, 'r', encoding='utf-8') as f:
                ttl_content = f.read()
            rooms = extract_room_entries(ttl_content)
            for room in rooms:
                room_key = f"{room}_scene{scene_number}"
                room_entry_counts[room_key] += 1
        except Exception as e:
            print(f"  Error reading TTL: {ttl_path} - {e}")
    return room_entry_counts

# Process each JSON file
for json_filename in os.listdir(EPISODES_FOLDER):
    if not json_filename.endswith(".json"):
        continue
    json_path = os.path.join(EPISODES_FOLDER, json_filename)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            episode_data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load JSON: {json_filename} - {e}")
        continue
    
    scenario_id = episode_data.get('data', {}).get('id', '')
    match = re.search(r"scene(\d+)", scenario_id)
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
    
    print(f"  TTL files to process (in activity order): {len(ttl_files_to_use)}")
    for ttl_file in ttl_files_to_use:
        print(f"    - {os.path.basename(ttl_file)}")
    
    room_counts = count_room_entries(ttl_files_to_use, scene_number)
    
    if room_counts:
        output_data = {
            "name": "Test Test",
            "senario": scenario_id,
            "answers": []
        }
        
        for room_name, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True):
            output_data["answers"].append({
                "name": room_name,
                "number": count
            })
        
        output_filename = json_filename.replace(".json", "_results.json")
        output_path = os.path.join(OUTPUT_JSON_FOLDER, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"  OK Results saved: {output_path}")
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"  Error saving JSON: {e}")
    else:
        print("  Warning: No room entries found.")

print("\nAll JSON files processed.")