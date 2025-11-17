# 功能: 實作五分類邏輯，將 VisualObservation TTL 檔案，轉換為 K-BERT 模型所需的 JSON 動作序列。

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, Namespace
import json
import os
import random
from typing import List, Dict

# RDF 命名空間定義
EX = Namespace("http://example.org/")

# ==============================================================================
# K-BERT 動作列表 (必須與您的訓練集動作完全一致！)
# ==============================================================================
ALL_KBERT_ACTIONS = [
    'Admire_art1', 'Admire_art2', 'Brush_teeth1', 'Clean_desk3', 'Clean_desk4', 'Clean_fridge1', 
    'Clean_kitchen1', 'Clean_kitchentable1', 'Clean_livingroom1', 'Clean_sink3', 'Clean_sink4', 
    'Clean_slippers_scene1', 'Clean_sofa1', 'Clean_stove1', 'Clean_television3', 'Clean_television4', 
    'Clear_table1', 'Cook_carrot1', 'Cook_fried_bread3', 'Cook_fried_bread4', 'Cook_potato_using_microwave1', 
    'Cook_potato_using_stove1', 'Cook_salmon1', 'Do_homework_on_paper1', 'Do_research_on_computer1', 
    'Do_work_on_computer1', 'Drink_alcohol1', 'Drink_alcohol_while_watching_television3', 'Drink_juice1', 
    'Drink_juice_while_watching_television3', 'Drink_milk1', 'Drink_milk_while_watching_television3', 
    'Drink_water1', 'Drink_wine1', 'Drink_wine_while_watching_television3', 'Eat_bread_while_watching_television3', 
    'Eat_breadslice1', 'Eat_cupcake1', 'Eat_cupcake_while_watching_television3', 
    
    # Fall 動作系列 (危險類別 1)
    'Fall_backward3', 'Fall_backward_while_walking_and_turning1', 'Fall_in_bathroom1', 
    'Fall_sideways_while_walking_forward1', 'Fall_while_climbing_at_somewhere_height1', 
    'Fall_while_during_getting_up_or_rising1', 'Fall_while_during_getting_up_or_rising2', 
    'Fall_while_initiation_of_walking1', 'Fall_while_initiation_of_walking2', 'Fall_while_preparing_meal1', 
    'Fall_while_preparing_meal2', 'Fall_while_sitting_down1', 'Fall_while_sitting_down_or_lowering1', 
    'Fall_while_standing_and_reaching1', 'Fall_while_standing_and_turning1', 
    'Fall_while_standing_at_somewhere_height1', 'Fall_while_standing_quietly1', 'Fall_while_walking_forward1', 
    
    'Fold_clothespants1', 'Fold_clothespile1', 'Fold_clothesshirt1', 'Get_out_of_bed1', 'Go_to_sleep1', 
    'Hand_washing1', 'Have_evening_beverage3', 'Have_evening_beverage4', 'Have_morning_beverage3', 
    'Have_morning_beverage4', 'LegOpp1', 'Make_bed1', 'Make_cold_cereal1', 'Make_hot_cereal1', 
    'Organize_closet1', 'Pet_cat1', 'Pick_up_dirty_dishes1', 'Pick_up_dirty_fork1', 'Pick_up_dirty_knife1', 
    'Pick_up_dirty_waterglass1', 'Pick_up_dirty_wineglass1', 'Place_fork1', 'Place_knife1', 'Place_plate1', 
    'Place_waterglass1', 'Place_wine1', 'Place_wineglass1', 'Prepare_for_brushing_teeth1', 
    'Put_away_groceries_from_fridge1', 'Put_away_groceries_from_fridge2', 'Put_away_groceries_from_fridge3', 
    'Put_groceries_in_fridge13', 'Put_groceries_in_fridge14', 'Put_groceries_in_fridge15', 
    'Put_groceries_in_fridge16', 'Put_groceries_in_fridge17', 'Put_groceries_in_fridge18', 
    'Put_groceries_in_fridge19', 'Put_groceries_in_fridge20', 'Put_groceries_in_fridge21', 
    'Put_groceries_in_fridge22', 'Put_groceries_in_fridge23', 'Put_groceries_in_fridge24', 
    'Put_slippers_in_closet1', 'Put_slippers_in_closet2', 'Read_bedtime_story1', 'Read_book1', 
    'Read_textbook1', 'Relax_on_bed1', 'Relax_on_sofa1', 'Relax_on_sofa_while_watching_television3', 
    'Rinse_toothbrush1', 
    
    # Run_with_disorientation 動作系列 (危險類別 4)
    'Run_with_disorientation_scene1', 'Run_with_disorientation_scene3', 'Run_with_disorientation_scene5', 
    'Run_with_disorientation_scene6', 
    
    'Set_up_table1', 'Shoulder_stretch1', 'Social_media_checks1', 'Squat_jumps1', 'Squats1', 
    'Stand_on_coffee_table1', 'Straddle_splits1', 'Throw_trash2', 'Throw_trash3', 'Turn_off_light5', 
    'Turn_off_light6', 'Turn_off_light7', 'Turn_off_light8', 'Turn_on_light5', 'Turn_on_light6', 
    'Turn_on_light7', 'Turn_on_light8', 'Use_bathroom1', 'Use_phone3', 'Use_phone4', 'Use_toilet1', 
    
    # Walk_with_memory_loss 動作系列 (危險類別 2)
    'Walk_with_memory_loss5', 'Walk_with_memory_loss6', 'Walk_with_memory_loss7', 'Walk_with_memory_loss8', 
    
    'Watch_television3', 'Workout1', 'Write_notes1', 'add_classes', 'add_places', 'affordance_20231127', 
    'vh2kg_schema'
]

# 將動作分類到三個主要群組，以便隨機取樣
FALL_ACTIONS = [a for a in ALL_KBERT_ACTIONS if a.startswith('Fall_')]
WALK_MEMORY_LOSS_ACTIONS = [a for a in ALL_KBERT_ACTIONS if a.startswith('Walk_with_memory_loss')]
RUN_DISORIENTATION_ACTIONS = [a for a in ALL_KBERT_ACTIONS if a.startswith('Run_with_disorientation')]
# 其他所有動作，作為常規動作的抽樣池 (危險類別 5)
NORMAL_ACTIONS = [a for a in ALL_KBERT_ACTIONS if not a.startswith(('Fall_', 'Walk_with_memory_loss', 'Run_with_disorientation'))]


# --- 核心轉換函式 ---
def convert_observation_ttl_to_action_json(ttl_path: str, output_dir: str = ".") -> str:
    """
    實作五分類邏輯：將 TTL 觀察推論為四種危險動作或一種常規動作。
    """
    g = Graph()
    try:
        g.parse(ttl_path, format="ttl")
        print(f"[INFO] 成功載入 TTL: {os.path.basename(ttl_path)}")
    except Exception as e:
        print(f"[ERROR] 解析 TTL 檔案 {ttl_path} 失敗: {e}")
        return ""

    # 1. 儲存並排序所有觀察結果 (按時間戳排序)
    observations = []
    for event_uri, _, _ in g.triples((None, RDF.type, EX["VisualObservation"])):
        agent_uri = g.value(event_uri, EX.agent)
        object_uri = g.value(event_uri, EX.object)
        relation_uri = g.value(event_uri, EX.relation)
        time_offset = g.value(event_uri, EX.timeOffsetS)
        
        observations.append({
            'time': float(time_offset) if time_offset else 0.0,
            'agent': str(agent_uri).split(':')[-1], 
            'object': str(object_uri).split(':')[-1], 
            'relation': str(relation_uri).split(':')[-1]
        })
    
    observations.sort(key=lambda x: x['time'])
    print(f"[INFO] 總共 {len(observations)} 條低階觀察數據。")

    # 2. 應用規則進行動作推論 (動作識別核心 - 五分類)
    action_sequence = []
    last_action_time = 0.0
    
    # 設置時間閾值，用於區分 Run (快速) 和 Walk (慢速)
    RUN_THRESHOLD = 0.5  # 小於 0.5 秒的間隔，假設為快速移動
    WALK_THRESHOLD = 2.0  # 大於 2.0 秒的間隔，假設為停滯或慢速行走
    
    for obs in observations:
        time = obs['time']
        action_name = None
        time_diff = time - last_action_time
        
        # --- 規則 1: 偵測 Fall 動作 (最高優先級) ---
        if obs['object'] == 'lying_or_fall' and obs['relation'] == 'has_state':
            # 隨機選擇一個 Fall_ 開頭的動作，以模擬訓練數據的多樣性
            if FALL_ACTIONS:
                action_name = random.choice(FALL_ACTIONS)
                
        # --- 規則 2/3: 偵測 Walk/Run (基於 near 關係和時間間隔) ---
        elif obs['relation'] == 'near':
             
             if time_diff < RUN_THRESHOLD and time_diff > 0.0:
                 # 偵測 Run_with_disorientation (危險類別 4)
                 if RUN_DISORIENTATION_ACTIONS:
                     action_name = random.choice(RUN_DISORIENTATION_ACTIONS)

             elif time_diff > WALK_THRESHOLD:
                 # 偵測 Walk_with_memory_loss (危險類別 2)
                 if WALK_MEMORY_LOSS_ACTIONS:
                     action_name = random.choice(WALK_MEMORY_LOSS_ACTIONS)
        
        # --- 規則 4: 偵測常規動作 (低優先級) ---
        # 排除跌倒狀態，且時間間隔適中，視為常規活動 (危險類別 5)
        if action_name is None and obs['object'] != 'lying_or_fall' and time_diff > 1.0:
            if NORMAL_ACTIONS:
                action_name = random.choice(NORMAL_ACTIONS)


        # 如果推論出動作，則加入序列
        if action_name:
            # 確保不會重複加入上一個動作
            if not action_sequence or action_name != action_sequence[-1]:
                action_sequence.append(action_name)
                last_action_time = time
                print(f"  [Action] 推論出: {action_name} @ {time:.2f}s")


    # 3. 構造目標 JSON 格式 (與 K-BERT 輸入格式一致)
    final_activities = action_sequence
    # # 測試，加上一個已知的危險動作
    # if FALL_ACTIONS:
    #      print("[DEBUG] 強制附加 'Fall_while_walking_forward1' 用於測試。")
    #      final_activities.append(random.choice(FALL_ACTIONS))
    # # --- [!! 測試結束 !!] ---
    if not final_activities:
         # 如果沒有推論出任何高階動作，給一個預設的低風險動作
         final_activities = ["Get_out_of_bed1"] 
    
    base_filename = os.path.splitext(os.path.basename(ttl_path))[0]
    
    json_data = {
        "statusCode": 200,
        "method": "POST_Inferred", 
        "message": "Inferred Success",
        "data": {
            "id": f"{base_filename}_inferred", 
            "title": base_filename,
            "scene": 99, 
            "activities": final_activities
        }
    }
    
    output_json_path = os.path.join(output_dir, f"{base_filename}_inferred_actions.json")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[SUCCESS] 動作序列 JSON 儲存於: {output_json_path}")
    return output_json_path

# --- 主執行部分 (測試和示範) ---
if __name__ == "__main__":
    # 這是測試用的虛擬 TTL 內容，用於模擬跌倒、快速移動和常規活動的混合序列
    TEST_TTL_PATH = "test_video_data.ttl" 
    
    VIRTUAL_TTL_CONTENT = """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:event_1 a ex:VisualObservation ; ex:agent ex:person1 ; ex:object ex:upright ; ex:relation ex:has_state ; ex:timeOffsetS 0.50 .
ex:event_2 a ex:VisualObservation ; ex:agent ex:person1 ; ex:object ex:desk_counter ; ex:relation ex:near ; ex:timeOffsetS 0.60 . # (快速移動)
ex:event_3 a ex:VisualObservation ; ex:agent ex:person1 ; ex:object ex:wall ; ex:relation ex:near ; ex:timeOffsetS 0.90 . # (快速移動)
ex:event_4 a ex:VisualObservation ; ex:agent ex:person1 ; ex:object ex:kitchen_table ; ex:relation ex:near ; ex:timeOffsetS 4.00 . # (慢速移動)
ex:event_5 a ex:VisualObservation ; ex:agent ex:person1 ; ex:object ex:lying_or_fall ; ex:relation ex:has_state ; ex:timeOffsetS 7.80 . # (跌倒)
ex:event_6 a ex:VisualObservation ; ex:agent ex:person1 ; ex:object ex:wall ; ex:relation ex:near ; ex:timeOffsetS 12.00 . # (常規動作)
ex:person1 a ex:Agent ; rdfs:label "person#1" .
ex:desk_counter a ex:Object ; rdfs:label "desk counter" .
ex:lying_or_fall a ex:Posture ; rdfs:label "lying_or_fall" .
"""
    
    with open(TEST_TTL_PATH, "w", encoding="utf-8") as f:
        f.write(VIRTUAL_TTL_CONTENT)
            
    # 執行轉換
    output_json_file = convert_observation_ttl_to_action_json(TEST_TTL_PATH)
    
    if output_json_file:
        print("\n--- 輸出 JSON 內容範例 ---")
        with open(output_json_file, 'r', encoding='utf-8') as f:
             print(f.read())
        print("\n-------------------------")
        
        print(f"\n[NEXT STEP] 現在請使用 7_json_risk_detector.py 讀取此檔案：{output_json_file}")
    