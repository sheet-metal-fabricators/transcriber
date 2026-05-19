# 🎙 Audio Transcriber

Transcribe audio files with speaker labels and AI summary — powered by Groq Whisper + Claude.

## Features
- 🎯 Accurate transcription via Groq's Whisper large-v3
- 👥 Automatic speaker identification & labeling
- 📋 AI-generated summary and key points
- 💬 Sentiment analysis
- 💾 Download as .txt

## Deploy to Vercel (5 minutes)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/audio-transcriber.git
git push -u origin main
```

### 2. Deploy on Vercel
1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repo
3. Click **Deploy** (no env vars needed — keys are entered in the UI)

### 3. Get your free API keys
- **Groq** (free): https://console.groq.com/keys
- **Anthropic**: https://console.anthropic.com

## Local Development

```bash
npm install
npm run dev
```

Open http://localhost:3000

## Notes
- Max file size: 25MB (Groq limit)
- API keys are entered in the UI and sent securely via request headers — never stored
- Supports: M4A, MP3, WAV, OGG, MP4, WebM
