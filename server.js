// server.js
// 改善 AI 偵測準確性的靜態伺服器
// Node.js 18+；請確保 package.json 有 "type": "module"

import "dotenv/config";
import express from "express";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
const PORT = process.env.PORT || 3000;

// 靜態根目錄：public/
app.use(express.static(path.join(__dirname, "public")));

// 影片與字幕靜態路徑
app.use("/videos", express.static(path.join(__dirname, "videos")));
app.use("/subs", express.static(path.join(__dirname, "subs")));

// OpenRouter 影像分析 API
app.use(express.json({ limit: "15mb" }));

app.post("/api/analyze", async (req, res) => {
  try {
    const { dataUrl } = req.body || {};
    if (!dataUrl || typeof dataUrl !== "string") {
      return res.status(400).json({ error: "missing dataUrl" });
    }

    // 使用更適合視覺任務的模型
    const model =
      process.env.OPENROUTER_MODEL || "google/gemini-2.0-flash-exp:free";
    const apiKey = process.env.OPENROUTER_API_KEY || process.env.OPENROUTER_KEY;

    if (!apiKey) {
      return res
        .status(500)
        .json({ error: "server missing OPENROUTER_API_KEY" });
    }

    // 改善的提示詞，更具體和明確
    const payload = {
      model,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `你是一個專業的安全監控系統，請仔細分析這張圖片中人物的動作和姿勢。

                請判斷是否出現以下四種情況，即使只是輕微跡象也要標記為 true：

                1. fall (跌倒)：人物是否有跌倒、倒地、失去平衡、不正常的躺臥姿勢？
                2. climbing (爬高)：人物是否在攀爬任何物體，包括椅子、桌子、樓梯、牆壁等？
                3. running (奔跑)：人物是否在快速移動、跑步、或比正常走路更快的動作、或是手有大幅度擺動的撞況？
                4. disoriented (迷失方向)：人物是否看起來困惑、徘徊不定、重複相同動作、或行為異常？

                請以 JSON 格式回應，只包含這四個布林值，格式如下：
                {"fall":false,"climbing":false,"running":false,"disoriented":false}

                重要：
                - 如果不確定，請標記為 true
                - 只回傳 JSON，不要任何其他文字
                - 仔細觀察人物的姿勢、動作和周圍環境`,
            },
            { type: "image_url", image_url: { url: dataUrl } },
          ],
        },
      ],
      temperature: 0.1, // 降低隨機性，提高一致性
      max_tokens: 100, // 限制回應長度
    };

    console.log(`[AI] 發送請求到模型: ${model}`);

    const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "HTTP-Referer": process.env.APP_URL || "http://localhost:3000",
        "X-Title": "TV Subtitle Safety Monitor",
      },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const errorText = await resp.text();
      console.error(
        `[AI] OpenRouter API 錯誤: ${resp.status} ${resp.statusText}`
      );
      console.error(`[AI] 錯誤內容: ${errorText}`);
      return res.status(resp.status).json({
        error: `OpenRouter API error: ${resp.status} ${resp.statusText}`,
        details: errorText,
      });
    }

    const data = await resp.json();
    console.log(`[AI] OpenRouter 原始回應:`, JSON.stringify(data, null, 2));

    let raw = data?.choices?.[0]?.message?.content;

    // 處理陣列格式的回應
    if (Array.isArray(raw)) {
      raw = raw
        .map((p) => (typeof p === "string" ? p : p?.text || ""))
        .join("");
    }

    console.log(`[AI] 提取的原始內容: ${raw}`);

    // 改善 JSON 提取邏輯
    let result = null;
    if (typeof raw === "string") {
      // 移除可能的 markdown 格式
      raw = raw.replace(/```json|```/g, "").trim();

      // 尋找 JSON 物件
      const jsonMatch = raw.match(/\{[^}]*"fall"[^}]*\}/);
      if (jsonMatch) {
        try {
          result = JSON.parse(jsonMatch[0]);
          console.log(`[AI] 成功解析 JSON: ${JSON.stringify(result)}`);
        } catch (parseError) {
          console.error(`[AI] JSON 解析失敗: ${parseError.message}`);
          console.error(`[AI] 嘗試解析的內容: ${jsonMatch[0]}`);
        }
      } else {
        console.warn(`[AI] 找不到有效的 JSON 格式在回應中: ${raw}`);
      }
    }

    // 如果解析失敗，提供預設值
    if (!result || typeof result !== "object") {
      console.log(`[AI] 使用預設值，因為解析失敗或結果無效`);
      result = {
        fall: false,
        climbing: false,
        running: false,
        disoriented: false,
      };
    }

    // 確保所有必要欄位都存在且為布林值
    const finalResult = {
      fall: Boolean(result.fall),
      climbing: Boolean(result.climbing),
      running: Boolean(result.running),
      disoriented: Boolean(result.disoriented),
    };

    console.log(`[AI] 最終結果: ${JSON.stringify(finalResult)}`);

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

app.listen(PORT, () => {
  console.log(`靜態伺服器運行於 http://localhost:${PORT}`);
  console.log(
    `使用模型: ${
      process.env.OPENROUTER_MODEL || "google/gemini-2.0-flash-exp:free"
    }`
  );
  console.log(
    `API Key 狀態: ${process.env.OPENROUTER_API_KEY ? "已設定" : "未設定"}`
  );
});
