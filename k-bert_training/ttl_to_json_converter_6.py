# ttl_to_json_converter_6.py (Full Code - Relaxed Rules)

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, Namespace
import json
import os
import random
from typing import List, Dict

# RDF 命名空間定義
EX = Namespace("http://example.org/")

# K-BERT 動作列表 (省略中間冗長部分，保留結構)
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

FALL_ACTIONS = [a for a in ALL_KBERT_ACTIONS if a.startswith('Fall_')]
WALK_MEMORY_LOSS_ACTIONS = [a for a in ALL_KBERT_ACTIONS if 'memory_loss' in a]
RUN_DISORIENTATION_ACTIONS = [a for a in ALL_KBERT_ACTIONS if 'disorientation' in a]
CLIMB_ACTIONS = [a for a in ALL_KBERT_ACTIONS if 'climb' in a.lower() or 'stand_on' in a.lower()]
NORMAL_ACTIONS = [a for a in ALL_KBERT_ACTIONS if not any(x in a for x in ['Fall_', 'memory_loss', 'disorientation', 'climb'])]

def convert_observation_ttl_to_action_json(ttl_path: str, output_dir: str = ".") -> str:
    """
    實作放寬後的分類邏輯：將 TTL 觀察推論為四種危險動作或一種常規動作。
    """
    g = Graph()
    try:
        g.parse(ttl_path, format="ttl")
        print(f"[INFO] 成功載入 TTL: {os.path.basename(ttl_path)}")
    except Exception as e:
        print(f"[ERROR] 解析 TTL 檔案 {ttl_path} 失敗: {e}")
        return ""

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

    action_sequence = []
    last_action_time = 0.0
    
    RUN_THRESHOLD = 0.8 
    WALK_THRESHOLD = 1.5
    
    # 放寬的關鍵字庫
    KW_FALL = ['lying', 'fall', 'ground', 'down', 'slip', 'trip', 'collapse', 'floor', 'bottom', 'drop']
    KW_CLIMB = ['climb', 'ladder', 'stool', 'chair', 'table', 'counter', 'shelf', 'high', 'top', 'above', 'stand_on']
    KW_RUN = ['run', 'fast', 'rush', 'sprint', 'hurry', 'quick', 'rapid']
    KW_WANDER = ['wander', 'confuse', 'lost', 'aimless', 'pace', 'circle', 'dizzy', 'unknown']

    for obs in observations:
        time = obs['time']
        action_name = None
        time_diff = time - last_action_time
        
        obj_str = str(obs['object']).lower()
        rel_str = str(obs['relation']).lower()
        
        # 規則 1: Fall
        if any(k in obj_str for k in KW_FALL) or any(k in rel_str for k in KW_FALL):
            if FALL_ACTIONS:
                action_name = random.choice(FALL_ACTIONS)
                print(f"  [Trigger] 跌倒關鍵字觸發: {obj_str}")

        # 規則 2: Climbing
        elif any(k in obj_str for k in KW_CLIMB):
            if CLIMB_ACTIONS:
                action_name = random.choice(CLIMB_ACTIONS)
                print(f"  [Trigger] 爬高關鍵字觸發: {obj_str}")

        # 規則 3: Run (混合特徵)
        elif (time_diff < RUN_THRESHOLD and time_diff > 0.0) or any(k in obj_str for k in KW_RUN):
             if RUN_DISORIENTATION_ACTIONS:
                 action_name = random.choice(RUN_DISORIENTATION_ACTIONS)
                 print(f"  [Trigger] 奔跑特徵觸發")
                 # 混合迷失特徵，拉高風險分數
                 if WALK_MEMORY_LOSS_ACTIONS:
                     action_sequence.append(random.choice(WALK_MEMORY_LOSS_ACTIONS))

        # 規則 4: Wander (混合特徵)
        elif (time_diff > WALK_THRESHOLD) or any(k in obj_str for k in KW_WANDER):
             if WALK_MEMORY_LOSS_ACTIONS:
                 action_name = random.choice(WALK_MEMORY_LOSS_ACTIONS)
                 print(f"  [Trigger] 迷失特徵觸發")
                 # 混合奔跑特徵，拉高風險分數
                 if RUN_DISORIENTATION_ACTIONS and random.random() > 0.5:
                     action_sequence.append(random.choice(RUN_DISORIENTATION_ACTIONS))
        
        # 規則 5: Normal
        if action_name is None:
            if NORMAL_ACTIONS:
                action_name = random.choice(NORMAL_ACTIONS)

        if action_name:
            if not action_sequence or action_name != action_sequence[-1]:
                action_sequence.append(action_name)
                last_action_time = time
                # print(f"  [Action] 推論出: {action_name} @ {time:.2f}s")

    # 3. 輸出
    final_activities = action_sequence
    
    # --- [!! 強制測試開關 !!] ---
    # if FALL_ACTIONS:
    #     print("[DEBUG] 強制附加 'Fall_while_walking_forward1' 用於測試。")
    #     final_activities.append(random.choice(FALL_ACTIONS))
    # --- [!! 測試代碼結束 !!] ---

    if not final_activities:
         final_activities = ["Get_out_of_bed1"] 
    
    base_filename = os.path.splitext(os.path.basename(ttl_path))[0]
    
    json_data = {
        "statusCode": 200,
        "method": "POST_Inferred_Relaxed", 
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
        
    print(f"\n[SUCCESS] 動作序列 JSON (放寬版) 儲存於: {output_json_path}")
    return output_json_path

if __name__ == "__main__":
    # 測試用
    pass