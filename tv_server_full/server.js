// Node.js 18+；package.json 需有 "type": "module"

import "dotenv/config";
import express from "express";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";
import multer from "multer";
import { Storage } from "@google-cloud/storage";

// 基本設定
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));
const PORT = process.env.PORT || 3000;

// --- GCS 設定 ---
const GCS_BUCKET_NAME =
  process.env.GCS_BUCKET_NAME || "kg_tv-subtitle-videos-20251016";
const GCS_RESULTS_BUCKET_NAME =
  process.env.GCS_RESULTS_BUCKET_NAME || `${GCS_BUCKET_NAME}-results`;

const storageClient = new Storage();

// --- Multer：記憶體暫存 ---
const upload = multer({ storage: multer.memoryStorage() });

// --- 靜態檔案服務 ---
// 1)  index.html
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// 2) public/ 作為靜態資源（前端）
app.use(express.static(path.join(__dirname, "public")));

// 3) 便利測試：直接提供 /videos /subs 本機資料夾
app.use("/videos", express.static(path.join(__dirname, "videos")));
app.use("/subs", express.static(path.join(__dirname, "subs")));

// --- API：上傳影片到 GCS，讓後端（如 Cloud Run/Cloud Functions）接手分析 ---
app.post("/api/upload", upload.single("video"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "沒有上傳檔案。" });
  }
  if (!GCS_BUCKET_NAME) {
    return res.status(500).json({ error: "伺服器未設定 GCS_BUCKET_NAME。" });
  }

  try {
    const bucket = storageClient.bucket(GCS_BUCKET_NAME);
    const blob = bucket.file(req.file.originalname);
    const blobStream = blob.createWriteStream({ resumable: false });

    blobStream.on("error", (err) => {
      console.error("[GCS Upload Error]", err);
      res.status(500).json({ error: "上傳至 GCS 失敗。" });
    });

    blobStream.on("finish", () => {
      console.log(
        `[Upload] ${req.file.originalname} 已上傳至 GCS：${GCS_BUCKET_NAME}。後端分析服務將自動觸發。`
      );
      res.status(200).json({
        message: "檔案已上傳，後端分析已自動觸發。",
        fileName: req.file.originalname,
        bucket: GCS_BUCKET_NAME,
      });
    });

    blobStream.end(req.file.buffer);
  } catch (error) {
    console.error("[Upload] 上傳時發生錯誤:", error);
    res.status(500).json({ error: "上傳檔案失敗。" });
  }
});

// API：Gemini 即時影像分析（單張影像 dataURL）
app.post("/api/analyze", async (req, res) => {
  try {
    const { dataUrl } = req.body || {};
    if (!dataUrl || typeof dataUrl !== "string") {
      return res.status(400).json({ error: "missing dataUrl" });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res
        .status(500)
        .json({ error: "server missing GEMINI_API_KEY in .env file" });
    }

    const model = "gemini-2.5-flash";
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

    const base64Data = dataUrl.split(",")[1];
    if (!base64Data) {
      return res.status(400).json({ error: "Invalid dataUrl format" });
    }

    const payload = {
      contents: [
        {
          parts: [
            {
              text: `你是一個專業的安全監控系統，請仔細分析這張圖片中人物的動作和姿勢。

                請判斷是否出現以下四種情況，即使只是輕微跡象也要標記為 true：

                1. fall (跌倒)：人物是否有跌倒、倒地、失去平衡、不正常的躺臥姿勢？
                2. climbing (爬高)：人物是否在攀爬任何物體，包括椅子、桌子、樓梯、牆壁等？
                3. running (奔跑)：人物是否在快速移動、跑步、或比正常走路更快的動作、或是手有大幅度擺動的狀況？
                4. disoriented (迷失方向)：人物是否看起來困惑、徘徊不定、重複相同動作、或行為異常？

                請以 JSON 格式回應，只包含這四個布林值，格式如下：
                {"fall":false,"climbing":false,"running":false,"disoriented":false}

                重要：
                - 如果不確定，請標記為 true
                - 只回傳 JSON，不要任何其他文字
                - 仔細觀察人物的姿勢、動作和周圍環境`,
            },
            {
              inline_data: {
                mime_type: "image/jpeg",
                data: base64Data,
              },
            },
          ],
        },
      ],
      // generationConfig 可視需要加入
      // generationConfig: { temperature: 0.1, maxOutputTokens: 100 }
    };

    const resp = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errorText = await resp.text();
      console.error(
        `[AI] Google Gemini API 錯誤: ${resp.status} ${resp.statusText}`
      );
      return res
        .status(resp.status)
        .json({ error: `Google Gemini API error`, details: errorText });
    }

    const data = await resp.json();
    let raw = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";

    // 嘗試抽取 JSON
    let result = null;
    if (typeof raw === "string") {
      raw = raw.replace(/```json|```/g, "").trim();
      const jsonMatch = raw.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          result = JSON.parse(jsonMatch[0]);
        } catch (e) {
          console.error("[AI] JSON 解析失敗:", e.message);
        }
      }
    }

    if (!result) {
      result = {
        fall: false,
        climbing: false,
        running: false,
        disoriented: false,
      };
    }

    const finalResult = {
      fall: !!result.fall,
      climbing: !!result.climbing,
      running: !!result.running,
      disoriented: !!result.disoriented,
    };

    res.json({
      result: finalResult,
      raw: raw,
      model: model,
      timestamp: new Date().toISOString(),
    });
  } catch (e) {
    console.error("[AI] analyze error:", e);
    res.status(500).json({
      error: String(e?.message || e),
      timestamp: new Date().toISOString(),
    });
  }
});

// API：查詢分析結果（從結果桶讀取 *_report.json）
app.get("/api/result/:fileName", async (req, res) => {
  const { fileName } = req.params;
  const reportFileName = `${fileName.split(".")[0]}_report.json`;

  try {
    const file = storageClient
      .bucket(GCS_RESULTS_BUCKET_NAME)
      .file(reportFileName);
    const [exists] = await file.exists();

    if (exists) {
      const [content] = await file.download();
      res.status(200).json({
        status: "completed",
        data: JSON.parse(content.toString()),
      });
    } else {
      res.status(202).json({ status: "processing" });
    }
  } catch (error) {
    console.error("[Result] 查詢結果時發生錯誤:", error);
    res.status(500).json({ status: "error", message: "無法取得結果" });
  }
});

// API：下載分析報告（檔案下載）
app.get("/api/download/:fileName", async (req, res) => {
  const { fileName } = req.params;
  const reportFileName = `${fileName.split(".")[0]}_report.json`;

  try {
    const file = storageClient
      .bucket(GCS_RESULTS_BUCKET_NAME)
      .file(reportFileName);
    const [content] = await file.download();

    res.setHeader(
      "Content-Disposition",
      `attachment; filename=${reportFileName}`
    );
    res.setHeader("Content-Type", "application/json");
    res.send(content);
  } catch (error) {
    console.error("[Download] 下載檔案時發生錯誤:", error);
    res.status(404).send("找不到檔案");
  }
});
// --- 下載 TTL 檔 ---
app.get("/api/download/ttl/:fileName", async (req, res) => {
  const { fileName } = req.params;
  const baseName = fileName.split(".")[0];
  const ttlFileName = `${baseName}.ttl`;
  try {
    const file = storageClient
      .bucket(GCS_RESULTS_BUCKET_NAME)
      .file(ttlFileName);
    const [exists] = await file.exists();
    if (!exists) return res.status(404).send("找不到知識圖譜檔案");

    res.setHeader("Content-Disposition", `attachment; filename=${ttlFileName}`);
    res.setHeader("Content-Type", "text/turtle");
    file.createReadStream().pipe(res);
  } catch (error) {
    console.error("[Download TTL] 下載檔案時發生錯誤:", error);
    res.status(500).send("下載知識圖譜檔案時發生錯誤");
  }
});

// --- 下載 DINO TXT 檔 ---
app.get("/api/download/dino_txt/:fileName", async (req, res) => {
  const { fileName } = req.params;
  const baseName = fileName.split(".")[0];
  const dinoTxtFileName = `${baseName}_dino.txt`;
  try {
    const file = storageClient
      .bucket(GCS_RESULTS_BUCKET_NAME)
      .file(dinoTxtFileName);
    const [exists] = await file.exists();
    if (!exists) return res.status(404).send("找不到 DINO TXT 檔案");

    res.setHeader(
      "Content-Disposition",
      `attachment; filename=${dinoTxtFileName}`
    );
    res.setHeader("Content-Type", "text/plain");
    file.createReadStream().pipe(res);
  } catch (error) {
    console.error("[Download DINO TXT] 下載檔案時發生錯誤:", error);
    res.status(500).send("下載 DINO TXT 檔案時發生錯誤");
  }
});

// --- 下載 DINO 圖片 ZIP 檔 ---
app.get("/api/download/dino_zip/:fileName", async (req, res) => {
  const { fileName } = req.params;
  const baseName = fileName.split(".")[0];
  const dinoZipFileName = `${baseName}_dino_images.zip`;
  try {
    const file = storageClient
      .bucket(GCS_RESULTS_BUCKET_NAME)
      .file(dinoZipFileName);
    const [exists] = await file.exists();
    if (!exists) return res.status(404).send("找不到 DINO 圖片 ZIP 檔案");

    res.setHeader(
      "Content-Disposition",
      `attachment; filename=${dinoZipFileName}`
    );
    res.setHeader("Content-Type", "application/zip");
    file.createReadStream().pipe(res);
  } catch (error) {
    console.error("[Download DINO ZIP] 下載檔案時發生錯誤:", error);
    res.status(500).send("下載 DINO 圖片 ZIP 檔案時發生錯誤");
  }
});

// --- 啟動 ---
app.listen(PORT, () => {
  console.log(`靜態伺服器運行於 http://localhost:${PORT}`);
  console.log(`即時分析模型: google/gemini-1.5-flash-latest`);
  console.log(
    `Gemini API Key 狀態: ${
      process.env.GEMINI_API_KEY ? "已設定" : "未設定 (.env 檔案)"
    }`
  );
  console.log(`GCS 影片bucket: ${GCS_BUCKET_NAME}`);
  console.log(`GCS 結果bucket: ${GCS_RESULTS_BUCKET_NAME}`);
});
