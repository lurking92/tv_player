import pickle

# 將 'your_file.pkl' 替換成你的檔案路徑
file_path = r"D:\KGRC-RDF-kgrc4si\CompleteData\SceneGraph\Abnormal\scene1\Fall_backward_while_walking_and_turning1\0\Fall backward while walking and turning1_0\object_bbox_and_relationship.pkl"

try:
    with open(file_path, 'rb') as file:
        # 使用 pickle.load() 載入檔案中的物件
        loaded_data = pickle.load(file)

        # 現在 'loaded_data' 變數中就包含了你儲存在 .pkl 檔案中的 Python 物件
        # 你可以根據這個物件的類型來查看和使用它

        # 範例：如果 .pkl 檔案儲存的是一個字典
        if isinstance(loaded_data, dict):
            print("dictionary：")
            for key, value in loaded_data.items():
                print(f"{key}: {value}")

        # 範例：如果 .pkl 檔案儲存的是一個列表
        elif isinstance(loaded_data, list):
            print("list：")
            for item in loaded_data:
                print(item)

        # 範例：如果 .pkl 檔案儲存的是其他類型的物件
        else:
            print("other：")
            print(loaded_data)

except FileNotFoundError:
    print(f"can't find '{file_path}'")
except Exception as e:
    print(f"error： {e}")