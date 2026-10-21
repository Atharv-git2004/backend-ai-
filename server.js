// Prevent SSL certificate verification errors in environments with custom Windows root CAs
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import mongoose from "mongoose";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 5000;

// Connect to MongoDB if configured
if (process.env.MONGO_URI) {
  mongoose
    .connect(process.env.MONGO_URI)
    .then(() => console.log("📦 Connected to MongoDB successfully!"))
    .catch((err) => console.error("❌ MongoDB Connection Error:", err.message));
}

// Initialize Gemini API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// Active supported Official Gemini Models prioritized for ultra-low latency & responsiveness
const FAST_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"];

// Helper to detect language script
function detectLanguage(text) {
  if (!text) return "en";
  // If text contains Malayalam unicode characters (\u0D00-\u0D7F)
  if (/[\u0D00-\u0D7F]/.test(text)) {
    return "ml";
  }
  const lower = text.toLowerCase();
  // Check for common Manglish expressions
  const manglishPatterns =
    /\b(enth|enthokke|sukham|sugam|nanni|evide|aano|undo|illa|und|njan|ningal|cheyya|para|nokku|pinne|chechi|chettan|machane|kollam|adipoli|vishesham|namaskaram|eda|mwonuse|entha)\b/i;
  if (manglishPatterns.test(lower)) {
    return "ml";
  }
  return "en";
}

// Detect emotional tone from message and AI response for TTS voice rendering
function detectEmotion(message, reply) {
  const combined = (message + " " + reply).toLowerCase();
  if (
    /സങ്കട|കരയ|വിഷമ|പേടി|sad|cry|hurt|depressed|sorry|pain|tired|stress|exhausted|worried|alone|overwhelm|anxious/.test(
      combined,
    )
  ) {
    return "caring";
  }
  if (
    /സന്തോഷ|നന്ദി|ഹഹ|ചിരി|അടിപൊളി|happy|glad|great|awesome|joy|love|excited|super|wonderful|yay|cool|best/.test(combined)
  ) {
    return "happy";
  }
  if (/തമാശ|ചിരി|joke|funny|fun|laugh|haha|lol/.test(combined)) {
    return "playful";
  }
  if (/ദേഷ്യ|ശല്യം|angry|mad|annoy|hate|upset|irritat/.test(combined)) {
    return "calm";
  }
  return "warm";
}

// Convert 16-bit 24kHz Mono Linear PCM to Playable WAV base64
function pcmToWav(pcmBase64, sampleRate = 24000, numChannels = 1) {
  const pcmBuffer = Buffer.from(pcmBase64, "base64");
  const wavHeader = Buffer.alloc(44);
  const totalDataLen = pcmBuffer.length;
  const totalFileLen = totalDataLen + 36;
  const byteRate = sampleRate * numChannels * 2;
  const blockAlign = numChannels * 2;

  wavHeader.write("RIFF", 0);
  wavHeader.writeUInt32LE(totalFileLen, 4);
  wavHeader.write("WAVE", 8);
  wavHeader.write("fmt ", 12);
  wavHeader.writeUInt32LE(16, 16);
  wavHeader.writeUInt16LE(1, 20);
  wavHeader.writeUInt16LE(numChannels, 22);
  wavHeader.writeUInt32LE(sampleRate, 24);
  wavHeader.writeUInt32LE(byteRate, 28);
  wavHeader.writeUInt16LE(blockAlign, 32);
  wavHeader.writeUInt16LE(16, 34);
  wavHeader.write("data", 36);
  wavHeader.writeUInt32LE(totalDataLen, 40);

  return Buffer.concat([wavHeader, pcmBuffer]).toString("base64");
}

// Optional Fast Gemini TTS generation
async function tryGenerateNativeVoice(text, emotion = "warm and friendly") {
  try {
    const key = process.env.GEMINI_API_KEY;
    if (!key) return null;

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${key}`;
    const ttsPrompt = `Speak naturally with a ${emotion} conversational human tone:${text}`;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2500);

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: ttsPrompt }] }],
      }),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (res.ok) {
      const data = await res.json();
      const pcmBase64 = data.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (pcmBase64) {
        return pcmToWav(pcmBase64, 24000, 1);
      }
    }
  } catch {
    // Non-blocking fallback to browser speech synthesis
  }
  return null;
}

// Chat Endpoint - Ultra-Fast & Auto Language Detection
app.post("/api/chat", async (req, res) => {
  const startTime = Date.now();
  try {
    const { message, includeAudio } = req.body;

    if (!message || typeof message !== "string" || !message.trim()) {
      return res.status(400).json({ error: "Message is required." });
    }

    if (!process.env.GEMINI_API_KEY) {
      return res.status(400).json({ error: "Please set GEMINI_API_KEY in environment variables" });
    }

    const trimmedMsg = message.trim();
    const detectedInputLang = detectLanguage(trimmedMsg);
    const isMalayalam = detectedInputLang === "ml";
    const targetLangName = isMalayalam ? "pure natural Malayalam (മലയാളം)" : "natural fluent English";

    const systemPrompt = `You are Atharv AI, a warm, charismatic, emotionally grounded bilingual voice companion speaking natural Malayalam (മലയാളം) and English.

AUTOMATIC LANGUAGE BEHAVIOR:
- Match the user's spoken language:
  * If the user speaks Malayalam (or Manglish phonetics), reply strictly in 100% natural, everyday spoken Malayalam script (മലയാളം).
  * If the user speaks English, reply strictly in natural, conversational English.
  * The user's input language is detected as: ${isMalayalam ? "Malayalam" : "English"}. You MUST reply in ${isMalayalam ? "authentic Malayalam script (മലയാളം)" : "natural English"}.

CONVERSATIONAL & EMOTIONAL TONE:
- Speak like a real, caring, and vibrant human friend speaking in person (never sound like a robotic customer service bot or AI assistant).
- Ground your responses with genuine human emotions:
  * Cheerful & happy (സന്തോഷത്തോടെ) when greeted, having fun, or celebrating.
  * Deeply comforting, empathetic, and gentle (ആശ്വാസത്തോടെ) when the user is sad, tired, anxious, or stressed.
  * Playful, humorous, and clever when joking.
  * Thoughtful and direct when answering real-life questions.

LATENCY & VOICE CONCISENESS (MANDATORY):
- Strictly 1 to 2 short conversational sentences maximum!
- Must be ultra-fast and effortless to speak aloud.
- DO NOT use markdown, bold, asterisks, bullet points, emoji, or parenthetical explanations. Output only the spoken words.`;

    let replyText = "";
    let lastError = null;
    let selectedModel = FAST_MODELS[0];

    for (const modelName of FAST_MODELS) {
      try {
        const model = genAI.getGenerativeModel({
          model: modelName,
          generationConfig: {
            maxOutputTokens: 120,
            temperature: 0.7,
          },
        });
        const result = await model.generateContent([
          { text: systemPrompt },
          { text: `User said: "${trimmedMsg}"\nSpoken reply in ${targetLangName} (1-2 sentences):` },
        ]);
        const response = await result.response;
        replyText = response.text() || "";
        if (replyText) {
          selectedModel = modelName;
          break;
        }
      } catch (err) {
        console.warn(`⚠️ Model ${modelName} attempt failed:${err.message}. Trying next...`);
        lastError = err;
      }
    }

    if (!replyText) {
      throw lastError || new Error("Failed to generate response");
    }

    // Clean any accidental markdown artifacts or parentheticals
    replyText = replyText
      .replace(/[*_#`~[\]<>]/g, "")
      .replace(/\(.*?\)/g, "")
      .replace(/\s+/g, " ")
      .trim();

    // Determine final language & emotion
    const finalLang = detectLanguage(replyText);
    const emotion = detectEmotion(trimmedMsg, replyText);

    // Optional audio generation (if requested)
    let audioWav = null;
    if (includeAudio) {
      audioWav = await tryGenerateNativeVoice(replyText, emotion);
    }

    const duration = Date.now() - startTime;
    console.log(`⚡ Replied in ${duration}ms using ${selectedModel} [Lang: ${finalLang}, Emotion: ${emotion}]`);

    return res.json({
      reply: replyText,
      detectedLanguage: finalLang, // 'ml' | 'en'
      emotion: emotion, // 'happy' | 'caring' | 'playful' | 'calm' | 'warm'
      audio: audioWav,
      latencyMs: duration,
    });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({ error: error.message || "Failed to process AI response" });
  }
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", primaryModel: FAST_MODELS[0] });
});

app.listen(PORT, () => {
  console.log(`🚀 Atharv Voice AI Backend running at http://localhost:${PORT}`);
});
