import pickle
import os

# 你的原始檔案路徑
object_path = r"D:\KGRC-RDF-kgrc4si\CompleteData\SceneGraph\Leisure\scene2\Use_phone3\0\Use phone3_0\object_bbox_and_relationship.pkl"
people_path2 = r"D:\KGRC-RDF-kgrc4si\CompleteData\SceneGraph\Leisure\scene2\Use_phone3\0\Use phone3_0\person_bbox.pkl"

# 決定輸出檔案的路徑 (與原始檔案在同一目錄下)
output_dir = os.path.dirname(object_path)
object_output_path = os.path.join(output_dir, "object_output.txt")
people_output_path = os.path.join(output_dir, "people_output.txt")

# 處理 object_path 的資料並寫入 object_output.txt
try:
    with open(object_path, 'rb') as object_file:
        object_data = pickle.load(object_file)
    with open(object_output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Data from '{os.path.basename(object_path)}':\n")
        if isinstance(object_data, dict):
            for key, value in object_data.items():
                outfile.write(f"  {key}: {value}\n")
        elif isinstance(object_data, list):
            for item in object_data:
                outfile.write(f"  {item}\n")
        else:
            outfile.write(f"  {object_data}\n")
    print(f"Data from '{os.path.basename(object_path)}' has been written to '{object_output_path}'")
except FileNotFoundError:
    print(f"找不到檔案：'{os.path.basename(object_path)}'")
except Exception as e:
    print(f"讀取 '{os.path.basename(object_path)}' 時發生錯誤：{e}")

print("\n" + "="*30 + "\n") # 加入分隔線

# 處理 people_path2 的資料並寫入 people_output.txt
try:
    with open(people_path2, 'rb') as people_file:
        people_data = pickle.load(people_file)
    with open(people_output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"Data from '{os.path.basename(people_path2)}':\n")
        if isinstance(people_data, dict):
            for key, value in people_data.items():
                outfile.write(f"  {key}: {value}\n")
        elif isinstance(people_data, list):
            for item in people_data:
                outfile.write(f"  {item}\n")
        else:
            outfile.write(f"  {people_data}\n")
    print(f"Data from '{os.path.basename(people_path2)}' has been written to '{people_output_path}'")
except FileNotFoundError:
    print(f"找不到檔案：'{os.path.basename(people_path2)}'")
except Exception as e:
    print(f"讀取 '{os.path.basename(people_path2)}' 時發生錯誤：{e}")