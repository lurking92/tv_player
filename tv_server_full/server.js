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

// API：Gemini 即時影像風險偵測（快速通道，僅回傳布林值）
app.post("/api/check-risk", async (req, res) => {
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

    // 關鍵：使用 Flash 模型並強制 JSON 輸出，以求最快速度
    const model = "gemini-2.5-flash";
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

    const base64Data = dataUrl.split(",")[1];
    if (!base64Data) {
      return res.status(400).json({ error: "Invalid dataUrl format" });
    }

    // 關鍵：Prompt 大幅簡化，只要求 JSON，不要求報告！
    const prompt = `
    你是一位 AI 安全助手。請只根據這張即時影像，判斷是否有以下四種立即性風險 (fall, climbing, running, disoriented)。
    
    請嚴格以 JSON 格式回應，只包含四個布林值：
    {
      "fall": boolean,
      "climbing": boolean,
      "running": boolean,
      "disoriented": boolean
    }`;

    const payload = {
      contents: [
        {
          parts: [
            { text: prompt },
            { inline_data: { mime_type: "image/jpeg", data: base64Data } },
          ],
        },
      ],
      // 關鍵：強制 API 回傳 JSON
      generationConfig: { responseMimeType: "application/json" },
    };

    const resp = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errorText = await resp.text();
      console.error(`[AI-Fast] Google Gemini API 錯誤: ${resp.status}`);
      return res
        .status(resp.status)
        .json({ error: `Google Gemini API error`, details: errorText });
    }

    const data = await resp.json();
    const content = data?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";

    // 解析 JSON
    let parsed = {};
    try {
      parsed = JSON.parse(content || "{}");
    } catch (e) {
      console.error("[AI-Fast] JSON 解析失敗:", e.message);
      parsed = {};
    }

    // 確保回傳的 JSON 格式正確
    const finalAnalysis = {
      fall: !!parsed?.fall,
      climbing: !!parsed?.climbing,
      running: !!parsed?.running,
      disoriented: !!parsed?.disoriented,
    };

    return res.json({ result: finalAnalysis });
  } catch (e) {
    console.error("[AI-Fast] check-risk error:", e);
    return res.status(500).json({
      error: String(e?.message || e),
      timestamp: new Date().toISOString(),
    });
  }
});

// API：Gemini 即時影像分析（單張影像 dataURL + 背景知識，"以當前畫面為主"）
app.post("/api/analyze", async (req, res) => {
  try {
    const { dataUrl, context, fastResult } = req.body || {};

    if (!dataUrl) return res.status(400).json({ error: "missing dataUrl" });
    const apiKey = process.env.GEMINI_API_KEY;
    const model = "gemini-2.5-flash";
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const base64Data = dataUrl.split(",")[1];

    // 1. 準備背景資訊
    const visualStr = context?.visual_triples?.length
      ? `- 參考背景 - 視覺物件: ${JSON.stringify(
          context.visual_triples.slice(0, 3)
        )}...`
      : "";
    const kbertStr = context?.kbert_summary
      ? `- 參考背景 - K-BERT 摘要: ${context.kbert_summary}`
      : "";

    // 2. 處理快篩強制指令
    let mandatoryInstruction = "";
    if (fastResult) {
      const detected = [];
      if (fastResult.fall) detected.push("跌倒 (Fall)");
      if (fastResult.climbing) detected.push("攀爬 (Climbing)");
      if (fastResult.running) detected.push("奔跑 (Running)");
      if (fastResult.disoriented) detected.push("迷失方向 (Disoriented)");

      if (detected.length > 0) {
        mandatoryInstruction = `【強制指令】快篩系統已偵測到：${detected.join(
          "、"
        )}。報告必須包含此風險的描述。`;
      }
    }

    // 3.七大行為特徵量表 (作為參考準則插入)
    const behaviorChecklist = `
    【失智症行為評估準則 (請在報告中對照檢查)】：
    1. [空間] 頻繁進出某房間或遊蕩? (注意力渙散)
    2. [動作] 跌倒或攀爬高處? (高風險)
    3. [操作] 長時間開爐火無人看顧? (火災風險)
    4. [停留] 在非日常區域(如走廊)長時間呆滯?
    5. [地點] 在不恰當地點做出異常行為(如隨地便溺/睡覺)?
    6. [反應] 面對事件反應遲緩?
    7. [離家] 試圖獨自開門離開?`;

    // 4. 組合 Prompt
    const prompt = `你是一位專注於失智症長者居家安全的 AI 助手。請優先根據眼前的即時影像進行分析，並輔以參考以下背景資訊，提供安全評估。

    背景資訊 (僅供參考，以即時影像為主):
    ${visualStr}
    ${kbertStr}
    ${mandatoryInstruction}
    ${behaviorChecklist}

    分析2項任務：
    1. 判斷目前畫面，或是根據前面提供的圖片判斷連續動作中是否有以下四種立即性風險 (fall, climbing, running, disoriented)，即使只是輕微跡象也要標記為 true。此判斷必須基於即時影像。
    2. 根據即時影像，並參考【失智症行為評估準則】，生成一份簡短的中文風險評估報告 (report)，包含：
      - 危險動作傾向：描述目前畫面顯示的主要風險。${
        mandatoryInstruction
          ? "請務必回應強制指令，若有其他風險也可提出。"
          : "若無明顯危險則說明之。"
      } 若觀察到上述七點準則中的行為，請特別指出。
      - 環境風險評估：畫面中是否有容易絆倒的物品、危險物品(如爐火)、光線問題等？
      - 預防建議：針對上述畫面中觀察到的風險，提供 1-2 點具體建議。

    請嚴格以 JSON 格式回應，包含 analysis (四個布林值) 和 report (中文報告文字)，格式如下：
    {
      "analysis": { "fall": boolean, "climbing": boolean, "running": boolean, "disoriented": boolean },
      "report": "中文風險評估報告文字..."
    }

    重要：
    1. 優先依賴即時影像進行判斷與報告生成。
    2. 僅回傳 JSON 物件，無其他多餘文字。`;

    const payload = {
      contents: [
        {
          parts: [
            { text: prompt },
            { inline_data: { mime_type: "image/jpeg", data: base64Data } },
          ],
        },
      ],
      generationConfig: { responseMimeType: "application/json" },
    };

    const resp = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    const content = data?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";

    let parsed = {};
    try {
      parsed = JSON.parse(content.replace(/```json|```/g, "").trim());
    } catch (e) {
      parsed = { report: content };
    }

    return res.json({
      result: parsed.analysis,
      report: parsed.report || "無法生成報告",
      timestamp: new Date().toISOString(),
    });
  } catch (e) {
    console.error("[AI] analyze error:", e);
    return res.status(500).json({ error: String(e?.message || e) });
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

app.post("/api/download/ai_report", express.text(), (req, res) => {
  const reportText = req.body;
  if (!reportText || typeof reportText !== "string") {
    return res.status(400).send("缺少報告內容");
  }
  try {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `ai_risk_report_${ts}.txt`;
    res.setHeader("Content-Disposition", `attachment; filename=${filename}`);
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.send(reportText);
    console.log(`[Download AI Report] 已提供報告下載: ${filename}`);
  } catch (err) {
    console.error("[Download AI Report] 產生下載時發生錯誤:", err);
    res.status(500).send("無法產生報告下載");
  }
});

// --- 啟動 ---
app.listen(PORT, () => {
  console.log(`靜態伺服器運行於 http://localhost:${PORT}`);
  console.log(`即時分析模型: google/gemini-2.5-flash`);
  console.log(
    `Gemini API Key 狀態: ${
      process.env.GEMINI_API_KEY ? "已設定" : "未設定 (.env 檔案)"
    }`
  );
  console.log(`GCS 影片bucket: ${GCS_BUCKET_NAME}`);
  console.log(`GCS 結果bucket: ${GCS_RESULTS_BUCKET_NAME}`);
});
