import re
import os
import json
import glob

# 定義您的資料夾路徑，請根據您的實際路徑修改
TTL_FOLDER_PATH = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"

# 定義輸出檔案的名稱
OUTPUT_FILE_NAME = "output_obj.txt" 

def extract_item_name(line):
    """
    從一行文字中提取物品名稱。
    例如，從 'ex:state0_computer177_admire_art1_scene1' 中提取 'computer'。
    """
    # 使用正則表達式匹配 'ex:state0_' 後面跟著英文字母，然後是數字，再是下劃線。
    # 括號內的 ([a-zA-Z]+) 會捕獲所需的物品名稱。
    match = re.search(r'ex:state0_([a-zA-Z]+)\d+_', line)
    if match:
        return match.group(1) # 返回捕獲到的物品名稱
    return None

def process_files_and_extract_items(file_paths):
    """
    處理多個檔案，提取物品名稱並返回一個不重複的列表。

    Args:
        file_paths (list): 包含要處理的檔案路徑的列表。

    Returns:
        set: 包含所有提取到的不重複物品名稱的集合。
    """
    all_extracted_items = set() # 使用集合自動處理重複項

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' not found. Skipping...")
            continue
        if not os.path.isfile(file_path):
            print(f"Warning: Path '{file_path}' is not a file. Skipping...")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = extract_item_name(line)
                    if item:
                        all_extracted_items.add(item)
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
    return all_extracted_items

if __name__ == "__main__":
    # 用於儲存所有檔案中提取到的不重複物品名稱
    overall_unique_items = set()

    # 遍歷 EPISODES_FOLDER 中的每個 JSON 檔案
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
        print(f"   Scenario ID   : {scenario_id}")
        print(f"   Scene Number  : {scene_number}")
        
        activities = episode_data.get('data', {}).get('activities', [])
        ttl_files_to_process = [] # 為當前JSON檔案收集相關的TTL檔案路徑
        
        for activity in activities:
            ttl_filename = f"{activity}_scene{scene_number}.ttl"
            ttl_path = os.path.join(TTL_FOLDER_PATH, ttl_filename)
            
            if os.path.exists(ttl_path):
                ttl_files_to_process.append(ttl_path)
            else:
                # 備用方案：如果確切的檔案名不存在，嘗試使用 glob 找到相似的檔案
                fallback = glob.glob(os.path.join(TTL_FOLDER_PATH, f"{activity}_scene*.ttl"))
                if fallback:
                    ttl_files_to_process.append(fallback[0]) # 找到第一個匹配的檔案
                else:
                    print(f"   Warning: No TTL file found for activity '{activity}' and scene '{scene_number}'.")
        
        print(f"   TTL files to process for this JSON: {len(ttl_files_to_process)}")
        for ttl_file in ttl_files_to_process:
            print(f"     - {os.path.basename(ttl_file)}")
        
        # 使用現有的 process_files_and_extract_items 函數處理這些 TTL 檔案
        current_json_items = process_files_and_extract_items(ttl_files_to_process)
        overall_unique_items.update(current_json_items) # 將當前JSON提取的物品添加到總體集合中

    print("\n--- All JSON files processed. ---")

    # 將提取到的不重複物品寫入 output_obj.txt 檔案
    if overall_unique_items:
        output_path = os.path.join(os.getcwd(), OUTPUT_FILE_NAME) # 輸出到當前工作目錄
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in sorted(list(overall_unique_items)):
                    f.write(item + '\n')
            print(f"所有檔案中提取到的不重複物品已寫入至: {output_path}")
        except Exception as e:
            print(f"Error writing to {OUTPUT_FILE_NAME}: {e}")
    else:
        print("沒有從任何TTL檔案中找到符合模式的物品，未生成輸出檔案。")