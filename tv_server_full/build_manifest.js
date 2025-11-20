// build_manifest.js (結合版)
// 讀 Episodes/*.json + 掃描 videos/*，輸出 public/manifest.json（敘述留空）
// 每個活動只輸出 { name, base, ext, narration:null }（不再含 angles）；
// 全域補上 angleSuffixMap，前端選角度時再用 base + 後綴 + ext 組 URL。
// 掃描額外的影片檔案作為 true_scene
// 讀取 risk_data.json 並注入風險分析數據

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Config
const EPISODES_DIR = path.join(__dirname, "Episodes");
const VIDEOS_ROOT = path.join(__dirname, "videos");
const OUTPUT_MANIFEST = path.join(__dirname, "public", "manifest.json");
// --- 讀取外部 JSON ---
const RISK_DATA_PATH = path.join(__dirname, "risk_data.json");

const DEFAULT_INTERVAL_SEC = 2.5;

const ANGLE_SUFFIX_MAP = { 1: "_0", 2: "_1", 3: "_2", 4: "_3", 5: "_4" };

const VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".webm"];

// --- 讀取函式 ---
let RISK_DATA = {};
try {
  if (fs.existsSync(RISK_DATA_PATH)) {
    const raw = fs.readFileSync(RISK_DATA_PATH, "utf-8");
    RISK_DATA = JSON.parse(raw);
    console.log(
      `[INFO] 成功載入 risk_data.json，包含 ${
        Object.keys(RISK_DATA).length
      } 筆資料。`
    );
  } else {
    console.warn(`[WARN] 找不到 risk_data.json，將不包含預先計算的風險數據。`);
  }
} catch (e) {
  console.error(`[ERROR] 讀取 risk_data.json 失敗: ${e.message}`);
}

// Helpers

function walkFiles(dir) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walkFiles(full)); // 展開遞迴結果
    else out.push(full);
  }
  return out;
}

function loadEpisodes(dir) {
  const scenes = [];
  if (!fs.existsSync(dir)) return scenes;
  const files = fs
    .readdirSync(dir)
    .filter((n) => n.toLowerCase().endsWith(".json"));
  for (const name of files) {
    const full = path.join(dir, name);
    try {
      const raw = JSON.parse(fs.readFileSync(full, "utf-8"));
      const payload =
        raw && typeof raw === "object" && raw.data ? raw.data : raw;

      const id = payload.id || name.replace(/\.json$/i, "");
      const title = id;

      let activities = [];
      if (Array.isArray(payload.activities)) {
        activities = payload.activities;
      } else if (Array.isArray(raw?.data?.activities)) {
        activities = raw.data.activities;
      }

      // --- 注入風險數據 ---
      const risk = RISK_DATA[id] || null;

      // 將 riskAnalysis 加入物件
      scenes.push({ id, title, activities, riskAnalysis: risk });
    } catch (e) {
      console.warn("JSON parse failed:", full, e.message);
    }
  }
  return scenes;
}

// 正規化（用於寬鬆比對活動名）：全小寫 + 移除非英數
function norm(s) {
  return String(s)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

// 尋找某個活動「任一角度檔」來回推出 base 與 ext；支援底線/空白混用、_1/-1 兩種後綴。
function findBaseAndExtForActivity(activityName, allVideoPaths) {
  // 允許 "Drink_milk1" 與 "Drink milk1" 兩種
  const nameCandidates = [activityName, activityName.replace(/_/g, " ")]; // 同時考慮 _1 與 -1 兩種後綴字元
  const suffixPairs = Object.values(ANGLE_SUFFIX_MAP).flatMap((sfx) => [
    sfx,
    sfx.replace("_", "-"),
  ]);

  const files = allVideoPaths.map((p) => {
    const fullLower = p.toLowerCase();
    const okExt = VIDEO_EXTS.some((ext) => fullLower.endsWith(ext));
    return { full: p, fullLower, okExt };
  });

  for (const cand of nameCandidates) {
    const candLower = cand.toLowerCase();
    for (const sfx of suffixPairs) {
      const needle = candLower + sfx;
      const hit = files.find((f) => f.okExt && f.fullLower.includes(needle));
      if (hit) {
        const rel =
          "/" + path.relative(__dirname, hit.full).replace(/\\/g, "/"); // 轉網站相對路徑
        const ext = path.extname(rel);
        const baseNoExt = rel.slice(0, -ext.length); // 去掉副檔名
        const base = baseNoExt.endsWith(sfx)
          ? baseNoExt.slice(0, -sfx.length)
          : baseNoExt; // 去掉後綴、加 encodeURI，避免空格等特殊字元
        return { base: encodeURI(base), ext };
      }
    }
  }
  return null;
}

// 掃描額外的影片檔案（不屬於任何場景的）
function scanStandaloneVideos(allVideoPaths, usedPaths) {
  const standaloneVideos = [];
  const usedSet = new Set(usedPaths.map((p) => p.toLowerCase()));

  for (const videoPath of allVideoPaths) {
    const ext = path.extname(videoPath).toLowerCase();
    if (!VIDEO_EXTS.includes(ext)) continue;

    const videoLower = videoPath.toLowerCase(); // 檢查這個影片是否已經被任何場景使用

    let isUsed = false;
    for (const usedPath of usedSet) {
      if (
        videoLower.includes(usedPath.toLowerCase()) ||
        usedPath.toLowerCase().includes(videoLower)
      ) {
        isUsed = true;
        break;
      }
    } // 檢查是否包含角度後綴，如果包含則可能是場景影片

    const hasAngleSuffix = Object.values(ANGLE_SUFFIX_MAP).some((suffix) =>
      videoLower.includes(suffix.toLowerCase())
    );

    if (!isUsed && !hasAngleSuffix) {
      const rel = "/" + path.relative(__dirname, videoPath).replace(/\\/g, "/");
      const filename = path.basename(videoPath, ext);

      standaloneVideos.push({
        name: filename,
        src: encodeURI(rel),
        title: filename,
      });
    }
  }

  return standaloneVideos;
}

function buildManifest() {
  const scenes = loadEpisodes(EPISODES_DIR);
  const allVideoPaths = walkFiles(VIDEOS_ROOT);
  const usedVideoPaths = [];

  const outScenes = scenes.map((scene) => {
    const acts = (scene.activities || []).map((name) => {
      const be = findBaseAndExtForActivity(name, allVideoPaths);
      if (be) {
        // 記錄已使用的影片路徑
        Object.values(ANGLE_SUFFIX_MAP).forEach((suffix) => {
          usedVideoPaths.push(`${be.base}${suffix}${be.ext}`);
        });
      }
      return {
        name,
        base: be?.base ?? null, // 例：/videos/.../Drink milk1
        ext: be?.ext ?? null, // 例：.mp4
        narration: null,
      };
    });
    console.log(
      `[SCAN] ${scene.id} activities=${
        acts.length
      } risk=${!!scene.riskAnalysis}`
    );

    // 回傳包含 riskAnalysis 的物件
    return {
      id: scene.id,
      title: scene.title,
      activities: acts,
      riskAnalysis: scene.riskAnalysis,
    };
  });

  // 掃描額外的影片檔案
  const standaloneVideos = scanStandaloneVideos(allVideoPaths, usedVideoPaths);
  console.log(`[SCAN] 額外影片數量: ${standaloneVideos.length}`);

  return {
    generatedAt: new Date().toISOString(),
    config: { defaultIntervalSec: DEFAULT_INTERVAL_SEC },
    angleSuffixMap: ANGLE_SUFFIX_MAP, // << 頂層補上去
    virtualScenes: outScenes, // 改名為 virtualScenes
    trueScenes: standaloneVideos, // 新增 trueScenes // 保持向後相容
    scenes: outScenes,
  };
}

// main 函式
function main() {
  const newManifest = buildManifest();
  let finalManifest;

  // 1. 嘗試載入舊的 manifest 檔案
  if (fs.existsSync(OUTPUT_MANIFEST)) {
    try {
      const existingManifest = JSON.parse(
        fs.readFileSync(OUTPUT_MANIFEST, "utf-8")
      );
      console.log(`已載入舊的 manifest.json，準備合併...`);

      // 2. 準備合併資料
      const oldTrueScenes = existingManifest.trueScenes || [];
      const newTrueScenes = newManifest.trueScenes || [];
      const existingNames = new Set(oldTrueScenes.map((v) => v.name));

      let newAddedCount = 0;
      const mergedTrueScenes = [...oldTrueScenes]; // 複製一份舊的

      // 3. 遍歷新的掃描結果，只添加不存在的
      for (const video of newTrueScenes) {
        if (!existingNames.has(video.name)) {
          mergedTrueScenes.push(video);
          newAddedCount++;
        }
      }

      // 4. 構建最終的 manifest 檔案
      finalManifest = {
        ...newManifest,
        generatedAt: new Date().toISOString(),
        trueScenes: mergedTrueScenes, // 用合併後的列表取代
      };

      console.log(`合併完成。新增 ${newAddedCount} 個影片。`);
    } catch (e) {
      console.warn("舊檔案載入或解析失敗，將重新生成新檔案。", e.message);
      finalManifest = newManifest;
    }
  } else {
    console.log("找不到舊的 manifest.json，將直接輸出新檔案。");
    finalManifest = newManifest;
  }

  // 5. 寫入新檔案
  fs.mkdirSync(path.dirname(OUTPUT_MANIFEST), { recursive: true });
  fs.writeFileSync(
    OUTPUT_MANIFEST,
    JSON.stringify(finalManifest, null, 2),
    "utf-8"
  );
  console.log(`已輸出: ${OUTPUT_MANIFEST}`);

  // 保持舊有的輸出格式
  console.log(`Virtual 情境數: ${finalManifest.virtualScenes.length}`);
  console.log(`True 影片數: ${finalManifest.trueScenes.length}`);
  finalManifest.virtualScenes.forEach((s) => {
    console.log(`- ${s.id} (${s.title}) 活動數=${s.activities.length}`);
  });
  finalManifest.trueScenes.forEach((v) => {
    console.log(`- ${v.name}`);
  });
}

main();
