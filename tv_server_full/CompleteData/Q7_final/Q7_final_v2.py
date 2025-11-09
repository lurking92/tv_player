import os
import json
import re
import glob
from rdflib import Graph, URIRef

# 資料夾路徑
EPISODES_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
RDF_FOLDER = r"D:\KGRC-RDF-kgrc4si\CompleteData\RDF"
OUTPUT_FOLDER = "Q7_final"
# 確保輸出資料夾存在
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def safe_extract(uri):
    """從 shape_state URI 提取物件名稱，去除數字部分"""
    m = re.search(r"shape_state\d+_([A-Za-z]+)", uri)
    return m.group(1).capitalize() if m else "N/A"


def get_initial_and_10s_relationships(ttl_file_path):
    """
    讀取單一 RDF 文件，回傳該活動初始 state (0) 與 10 秒後 state 的關係清單
    """
    g = Graph()
    try:
        g.parse(ttl_file_path, format="ttl")
    except Exception as e:
        print(f"Error parsing RDF {ttl_file_path}: {e}")
        return []

    # 收集所有 Duration 事件及其持續時間
    durations = []
    for evt in g.subjects(
        predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
        object=URIRef("http://www.w3.org/2006/time#Duration")):
        num = g.value(subject=evt,
                      predicate=URIRef("http://www.w3.org/2006/time#numericDuration"))
        if num is None:
            continue
        try:
            durations.append((str(evt), float(num))) # type: ignore
        except:
            continue

    # 依 time_event 編號排序，以確保時間順序
    def event_idx(uri):
        m = re.search(r"time_event(\d+)_", uri)
        return int(m.group(1)) if m else -1
    durations.sort(key=lambda x: event_idx(x[0]))

    # 累積計算直到 >=10 秒，並記錄該事件編號
    acc = 0.0
    target_e_idx = None
    for uri, dur in durations:
        acc += dur
        if acc >= 10.0:
            m = re.search(r"time_event(\d+)_", uri)
            target_e_idx = int(m.group(1)) if m else None
            break
    if target_e_idx is None:
        return []

    # 10秒後的 shape_state 快照編號 = 事件編號 (不加1)
    target_state = target_e_idx

    # SPARQL 查詢：擷取所有 shape_state 節點間的關係
    query = """
    PREFIX ex: <http://kgrc4si.home.kg/virtualhome2kg/instance/>
    SELECT ?s ?p ?o WHERE {
      ?s ?p ?o .
      FILTER STRSTARTS(STR(?s), STR(ex:shape_state))
      FILTER STRSTARTS(STR(?o), STR(ex:shape_state))
    }
    """
    results = []
    seen = set()
    for row in g.query(query):
        us = str(row['s']) # type: ignore
        uo = str(row['o']) # type: ignore
        rel_uri = str(row['p']) # type: ignore
        # 解析 shape_state 編號
        m1 = re.search(r"shape_state(\d+)_", us)
        m2 = re.search(r"shape_state(\d+)_", uo)
        if not m1 or not m2:
            continue
        s1, s2 = int(m1.group(1)), int(m2.group(1))
        # 僅保留同一快照，且為初始(0)或10秒後(target_state)
        if s1 != s2 or s1 not in (0, target_state):
            continue
        rel = rel_uri.split('/')[-1].upper()
        o1 = safe_extract(us)
        o2 = safe_extract(uo)
        if o1 == 'N/A' or o2 == 'N/A':
            continue
        key = (s1, o1, rel, o2)
        if key in seen:
            continue
        seen.add(key)
        results.append({'time_state': s1, 'obj1': o1, 'relation': rel, 'obj2': o2})

    # 確保初始(0)在前，10秒(target_state)在後
    results.sort(key=lambda x: x['time_state'])
    return results


if __name__ == '__main__':
    # 逐場景處理
    for js in glob.glob(os.path.join(EPISODES_FOLDER, '*.json')):
        with open(js, encoding='utf-8') as f:
            data = json.load(f)['data']
        scen = data['id']                # 如 scene1_Day1
        scene_num = re.search(r"scene(\d+)_", scen).group(1) # type: ignore

        # 取該場景所有 RDF ttl 檔案
        ttl_files = sorted(glob.glob(os.path.join(RDF_FOLDER, f"*scene{scene_num}.ttl")))
        all_answers = []
        for ttl in ttl_files:
            rels = get_initial_and_10s_relationships(ttl)
            for r in rels:
                ans = {'obj1': r['obj1'], 'obj2': r['obj2'], 'relation': r['relation']}
                if ans not in all_answers:
                    all_answers.append(ans)

        # 組成最終輸出
        output = {
            'name': data.get('title', ''),
            'scenario': scen,
            'answers': all_answers
        }
        out_path = os.path.join(OUTPUT_FOLDER, f"q7_answer_{scen}.json")
        with open(out_path, 'w', encoding='utf-8') as fw:
            json.dump(output, fw, ensure_ascii=False, indent=4)
        print(f"Written {out_path}: total {len(all_answers)} relations")
