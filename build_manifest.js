// build_manifest.js
// 讀 Episodes/*.json + 掃描 videos/*，輸出 public/manifest.json（敘述留空）
// 每個活動只輸出 { name, base, ext, narration:null }（不再含 angles）；
// 全域補上 angleSuffixMap，前端選角度時再用 base + 後綴 + ext 組 URL。

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// =====================
// Config
// =====================
const EPISODES_DIR = path.join(__dirname, "Episodes");
const VIDEOS_ROOT = path.join(__dirname, "videos");
const OUTPUT_MANIFEST = path.join(__dirname, "public", "manifest.json");

const DEFAULT_INTERVAL_SEC = 2.5;

const ANGLE_SUFFIX_MAP = { 1: "_0", 2: "_1", 3: "_2", 4: "_3", 5: "_4" };

const VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".webm"];

// =====================
// Helpers
// =====================

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

      scenes.push({ id, title, activities });
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
  const nameCandidates = [activityName, activityName.replace(/_/g, " ")];
  // 同時考慮 _1 與 -1 兩種後綴字元
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
          : baseNoExt; // 去掉後綴
        // 加 encodeURI，避免空格等特殊字元
        return { base: encodeURI(base), ext };
      }
    }
  }
  return null;
}

function buildManifest() {
  const scenes = loadEpisodes(EPISODES_DIR);
  const allVideoPaths = walkFiles(VIDEOS_ROOT);

  const outScenes = scenes.map((scene) => {
    const acts = (scene.activities || []).map((name) => {
      const be = findBaseAndExtForActivity(name, allVideoPaths);
      return {
        name,
        base: be?.base ?? null, // 例：/videos/.../Drink milk1
        ext: be?.ext ?? null, // 例：.mp4
        narration: null,
      };
    });
    console.log(`[SCAN] ${scene.id} activities=${acts.length}`);
    return { id: scene.id, title: scene.title, activities: acts };
  });

  return {
    generatedAt: new Date().toISOString(),
    config: { defaultIntervalSec: DEFAULT_INTERVAL_SEC },
    angleSuffixMap: ANGLE_SUFFIX_MAP, // << 頂層補上去
    scenes: outScenes,
  };
}

function main() {
  const manifest = buildManifest();
  fs.mkdirSync(path.dirname(OUTPUT_MANIFEST), { recursive: true });
  fs.writeFileSync(OUTPUT_MANIFEST, JSON.stringify(manifest, null, 2), "utf-8");
  console.log(`已輸出: ${OUTPUT_MANIFEST}`);
  console.log(`情境數: ${manifest.scenes.length}`);
  manifest.scenes.forEach((s) => {
    console.log(`- ${s.id} (${s.title}) 活動數=${s.activities.length}`);
  });
}

main();
