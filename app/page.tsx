'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import styles from './page.module.css'

type Stage = 'idle' | 'transcribing' | 'analyzing' | 'done' | 'error'
type Mode = 'file' | 'live'

interface Segment { start: number; end: number; text: string }
interface TranscriptResult { text: string; segments: Segment[]; language: string; duration: number }
interface Analysis { speakers: string[]; labeled_transcript: string; summary: string; key_points: string[]; sentiment: string }

const LANGUAGES = [
  { code: 'auto', label: '🌐 Auto Detect' },
  { code: 'en', label: '🇬🇧 English' },
  { code: 'hi', label: '🇮🇳 Hindi' },
  { code: 'ta', label: '🇮🇳 Tamil' },
  { code: 'te', label: '🇮🇳 Telugu' },
  { code: 'ml', label: '🇮🇳 Malayalam' },
  { code: 'kn', label: '🇮🇳 Kannada' },
  { code: 'mr', label: '🇮🇳 Marathi' },
  { code: 'bn', label: '🇧🇩 Bengali' },
  { code: 'ar', label: '🇸🇦 Arabic' },
  { code: 'fr', label: '🇫🇷 French' },
  { code: 'de', label: '🇩🇪 German' },
  { code: 'es', label: '🇪🇸 Spanish' },
  { code: 'pt', label: '🇵🇹 Portuguese' },
  { code: 'zh', label: '🇨🇳 Chinese' },
  { code: 'ja', label: '🇯🇵 Japanese' },
  { code: 'ko', label: '🇰🇷 Korean' },
  { code: 'ru', label: '🇷🇺 Russian' },
  { code: 'tr', label: '🇹🇷 Turkish' },
  { code: 'it', label: '🇮🇹 Italian' },
  { code: 'nl', label: '🇳🇱 Dutch' },
  { code: 'pl', label: '🇵🇱 Polish' },
  { code: 'ur', label: '🇵🇰 Urdu' },
  { code: 'fa', label: '🇮🇷 Persian' },
  { code: 'uk', label: '🇺🇦 Ukrainian' },
]

const TARGET_SAMPLE_RATE = 8000
const MAX_CHUNK_SECONDS = 120 // 2 min chunks = ~2MB WAV at 8kHz, well under Groq 25MB limit
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

async function splitAudioIntoChunks(file: File, onProgress?: (msg: string) => void): Promise<Blob[]> {
  onProgress?.('Decoding audio...')
  const arrayBuffer = await file.arrayBuffer()
  const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
  await audioCtx.close()
  const duration = audioBuffer.duration
  const numChunks = Math.ceil(duration / MAX_CHUNK_SECONDS)
  const chunkDuration = duration / numChunks
  const srcRate = audioBuffer.sampleRate
  const chunks: Blob[] = []
  for (let i = 0; i < numChunks; i++) {
    onProgress?.(`Preparing chunk ${i + 1} of ${numChunks}...`)
    const startSample = Math.floor(i * chunkDuration * srcRate)
    const endSample = Math.min(Math.floor((i + 1) * chunkDuration * srcRate), audioBuffer.length)
    const chunkLength = endSample - startSample
    const mono = new Float32Array(chunkLength)
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const channelData = audioBuffer.getChannelData(ch)
      for (let s = 0; s < chunkLength; s++) mono[s] += channelData[startSample + s]
    }
    for (let s = 0; s < chunkLength; s++) mono[s] /= audioBuffer.numberOfChannels
    const ratio = srcRate / TARGET_SAMPLE_RATE
    const outLength = Math.floor(chunkLength / ratio)
    const downsampled = new Float32Array(outLength)
    for (let s = 0; s < outLength; s++) downsampled[s] = mono[Math.floor(s * ratio)]
    chunks.push(encodeWav(downsampled, TARGET_SAMPLE_RATE))
  }
  return chunks
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const dataLen = samples.length * 2
  const buf = new ArrayBuffer(44 + dataLen)
  const v = new DataView(buf)
  const w = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  w(0, 'RIFF'); v.setUint32(4, 36 + dataLen, true)
  w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true)
  v.setUint16(20, 1, true); v.setUint16(22, 1, true)
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true)
  v.setUint16(32, 2, true); v.setUint16(34, 16, true)
  w(36, 'data'); v.setUint32(40, dataLen, true)
  let off = 44
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }
  return new Blob([buf], { type: 'audio/wav' })
}

async function transcribeBlob(blob: Blob, groqKey: string, language: string, label: string, retries = 5): Promise<{ text: string; segments: Segment[]; language: string; duration: number }> {
  const fd = new FormData()
  fd.append('file', new File([blob], 'audio.wav', { type: 'audio/wav' }))
  fd.append('model', 'whisper-large-v3')
  fd.append('response_format', 'verbose_json')
  fd.append('timestamp_granularities[]', 'segment')
  if (language !== 'auto') fd.append('language', language)
  const res = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${groqKey}` },
    body: fd,
  })
  if (!res.ok) {
    let errMsg = `HTTP ${res.status}`
    try {
      const err = await res.json()
      errMsg = err.error?.message || errMsg
    } catch { /* ignore parse error */ }
    if (res.status === 429 && retries > 0) {
      const waitMatch = errMsg.match(/try again in ([\d.]+)s/)
      const waitMs = waitMatch ? Math.ceil(parseFloat(waitMatch[1]) * 1000) + 500 : 4000
      await sleep(waitMs)
      return transcribeBlob(blob, groqKey, language, label, retries - 1)
    }
    if (res.status === 413) throw new Error(`${label}: File too large for Groq. Try a shorter clip.`)
    throw new Error(`${label} failed: ${errMsg}`)
  }
  return res.json()
}

export default function Home() {
  const [groqKey, setGroqKey] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [showKeys, setShowKeys] = useState(false)
  const [rememberKeys, setRememberKeys] = useState(false)
  const [savedBadge, setSavedBadge] = useState(false)
  const [mode, setMode] = useState<Mode>('file')
  const [file, setFile] = useState<File | null>(null)
  const [mediaUrl, setMediaUrl] = useState<string | null>(null)
  const [isVideo, setIsVideo] = useState(false)
  const [language, setLanguage] = useState('auto')
  const [stage, setStage] = useState<Stage>('idle')
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('')
  const [transcript, setTranscript] = useState<TranscriptResult | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<'transcript' | 'labeled' | 'summary' | 'export'>('summary')
  const [isDragging, setIsDragging] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [stealth, setStealth] = useState(false)

  // Live capture state
  const [liveRecording, setLiveRecording] = useState(false)
  const [liveSeconds, setLiveSeconds] = useState(0)
  const [liveWords, setLiveWords] = useState(0)
  const [liveTranscriptText, setLiveTranscriptText] = useState('')
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const liveChunksRef = useRef<Blob[]>([])
  const liveTimerRef = useRef<NodeJS.Timeout | null>(null)
  const liveStreamRef = useRef<MediaStream | null>(null)

  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const g = localStorage.getItem('groq_key')
    const a = localStorage.getItem('anthropic_key')
    if (g) { setGroqKey(g); setRememberKeys(true) }
    if (a) { setAnthropicKey(a); setRememberKeys(true) }
  }, [])

  useEffect(() => {
    if (rememberKeys) {
      if (groqKey) localStorage.setItem('groq_key', groqKey)
      if (anthropicKey) localStorage.setItem('anthropic_key', anthropicKey)
    }
  }, [groqKey, anthropicKey, rememberKeys])

  const handleSaveKeys = () => {
    if (rememberKeys) {
      localStorage.removeItem('groq_key')
      localStorage.removeItem('anthropic_key')
      setRememberKeys(false)
    } else {
      if (groqKey) localStorage.setItem('groq_key', groqKey)
      if (anthropicKey) localStorage.setItem('anthropic_key', anthropicKey)
      setRememberKeys(true)
      setSavedBadge(true)
      setTimeout(() => setSavedBadge(false), 2000)
    }
  }

  const handleFile = (f: File) => {
    setFile(f)
    setStage('idle')
    setTranscript(null)
    setAnalysis(null)
    setError('')
    if (mediaUrl) URL.revokeObjectURL(mediaUrl)
    setMediaUrl(URL.createObjectURL(f))
    setIsVideo(f.type.startsWith('video/'))
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Live Capture ──────────────────────────────────────────
  const startLiveCapture = async () => {
    if (!groqKey) { setError('Groq API key required'); return }
    setError('')
    try {
      // Request screen + system audio
      const stream = await (navigator.mediaDevices as unknown as {
        getDisplayMedia: (c: object) => Promise<MediaStream>
      }).getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          sampleRate: 44100,
        },
      })

      // Check if audio track was granted
      const audioTracks = stream.getAudioTracks()
      if (audioTracks.length === 0) {
        stream.getTracks().forEach(t => t.stop())
        setError('No audio captured. Make sure to check "Share system audio" in the dialog.')
        return
      }

      // Keep only audio, drop video to save bandwidth
      const audioStream = new MediaStream(audioTracks)
      liveStreamRef.current = stream

      liveChunksRef.current = []
      setLiveTranscriptText('')
      setLiveSeconds(0)
      setLiveWords(0)
      setLiveRecording(true)

      const recorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) liveChunksRef.current.push(e.data)
      }

      // Collect in 60-second intervals and transcribe each
      recorder.start(60000)

      let segmentIndex = 0
      let timeOffset = 0

      const processChunk = async () => {
        if (liveChunksRef.current.length <= segmentIndex) return
        const blob = new Blob(liveChunksRef.current.slice(segmentIndex), { type: 'audio/webm' })
        segmentIndex = liveChunksRef.current.length
        try {
          const result = await transcribeBlob(blob, groqKey, language, `Segment ${segmentIndex}`)
          const newText = result.text.trim()
          if (newText) {
            setLiveTranscriptText(prev => prev + (prev ? ' ' : '') + newText)
            setLiveWords(prev => prev + newText.split(/\s+/).filter(Boolean).length)
            timeOffset += result.duration || 60
          }
        } catch (err) {
          console.error('Live chunk error:', err)
        }
      }

      // Process every 60 seconds
      liveTimerRef.current = setInterval(async () => {
        setLiveSeconds(s => s + 1)
        if (liveChunksRef.current.length > segmentIndex) {
          await processChunk()
        }
      }, 1000)

      // Stop when screen share ends
      stream.getVideoTracks()[0]?.addEventListener('ended', () => stopLiveCapture())

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Could not start capture'
      if (msg.includes('Permission denied') || msg.includes('NotAllowedError')) {
        setError('Screen share was cancelled. Click "Start Live Capture" and then click Share.')
      } else {
        setError(msg)
      }
    }
  }

  const stopLiveCapture = async () => {
    if (liveTimerRef.current) clearInterval(liveTimerRef.current)
    mediaRecorderRef.current?.stop()
    liveStreamRef.current?.getTracks().forEach(t => t.stop())
    setLiveRecording(false)

    // Final transcription of remaining audio
    if (liveChunksRef.current.length > 0 && groqKey) {
      setStage('transcribing')
      setStatusMsg('Finalizing live transcript...')
      try {
        const blob = new Blob(liveChunksRef.current, { type: 'audio/webm' })
        const result = await transcribeBlob(blob, groqKey, language, 'Final segment')
        const finalText = liveTranscriptText + (liveTranscriptText ? ' ' : '') + result.text.trim()

        const tData: TranscriptResult = {
          text: finalText,
          segments: [],
          language: result.language || 'en',
          duration: liveSeconds,
        }
        setTranscript(tData)

        if (anthropicKey && finalText) {
          setStage('analyzing')
          setStatusMsg('Analyzing speakers...')
          const aRes = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'x-anthropic-key': anthropicKey, 'Content-Type': 'application/json' },
            body: JSON.stringify({ transcript: finalText, segments: [] }),
          })
          const aData = await aRes.json()
          if (aRes.ok) setAnalysis(aData)
        }

        setProgress(100)
        setStage('done')
        setActiveTab('summary')
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Finalization failed'
        setError(msg)
        setStage('error')
      }
    }
  }

  // ── File transcription ────────────────────────────────────
  const runFileTranscription = async () => {
    if (!file || !groqKey || !anthropicKey) return
    setError('')
    setStage('transcribing')
    setProgress(5)
    setStatusMsg('Preparing audio...')
    try {
      const chunks = await splitAudioIntoChunks(file, setStatusMsg)
      const totalChunks = chunks.length
      const allSegments: Segment[] = []
      let fullText = ''
      let detectedLanguage = 'en'
      let totalDuration = 0
      let timeOffset = 0

      for (let i = 0; i < totalChunks; i++) {
        if (i > 0) {
          setStatusMsg(`Chunk ${i} done. Waiting 3s...`)
          await sleep(3000)
        }
        setStatusMsg(`Transcribing chunk ${i + 1} of ${totalChunks}...`)
        setProgress(5 + Math.round((i / totalChunks) * 50))
        const result = await transcribeBlob(chunks[i], groqKey, language, `Chunk ${i + 1}/${totalChunks}`)
        if (result.segments) {
          result.segments.forEach(seg => allSegments.push({ start: seg.start + timeOffset, end: seg.end + timeOffset, text: seg.text }))
        }
        fullText += (i > 0 ? ' ' : '') + result.text.trim()
        detectedLanguage = result.language || detectedLanguage
        totalDuration += result.duration || 0
        timeOffset += result.duration || 0
      }

      const tData: TranscriptResult = { text: fullText, segments: allSegments, language: detectedLanguage, duration: totalDuration }
      setTranscript(tData)
      setProgress(60)
      setStatusMsg('Analyzing speakers...')
      setStage('analyzing')

      const aRes = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'x-anthropic-key': anthropicKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: tData.text, segments: tData.segments }),
      })
      const aData = await aRes.json()
      if (!aRes.ok) throw new Error(aData.error || 'Analysis failed')
      setAnalysis(aData)
      setProgress(100)
      setStage('done')
      setActiveTab('summary')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
      setStage('error')
    }
  }

  // ── Helpers ───────────────────────────────────────────────
  const copyText = (text: string) => navigator.clipboard.writeText(text)
  const downloadTxt = (text: string, name: string) => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
    a.download = name; a.click()
  }
  const downloadDocx = (labeled: string, summary: string, keyPoints: string[], name: string) => {
    let html = `<html><head><meta charset="UTF-8"></head><body style="font-family:Calibri,sans-serif;font-size:12pt;max-width:800px;margin:40px auto">`
    html += `<h1 style="color:#1a1a2e;border-bottom:2px solid #6c63ff;padding-bottom:8px">Call Transcript</h1>`
    html += `<h2 style="color:#444">Summary</h2><p style="line-height:1.6">${summary}</p>`
    html += `<h2 style="color:#444">Key Points</h2><ul>${keyPoints.map(p => `<li style="margin:4px 0">${p}</li>`).join('')}</ul>`
    html += `<h2 style="color:#444">Full Transcript</h2>`
    labeled.split('\n').filter(l => l.trim()).forEach(line => {
      const m = line.match(/^(.+?)(\[\d+:\d+\])?\s*:\s*(.*)/)
      if (m) html += `<p style="margin:8px 0"><strong style="color:#6c63ff">${m[1].trim()}${m[2] ? ' ' + m[2] : ''}:</strong> ${m[3]}</p>`
      else html += `<p style="margin:8px 0">${line}</p>`
    })
    html += '</body></html>'
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([html], { type: 'application/msword' }))
    a.download = name; a.click()
  }
  const downloadXlsx = (segments: Segment[], labeled: string, name: string) => {
    const rows = [['Timestamp', 'Speaker', 'Text']]
    labeled.split('\n').filter(l => l.trim()).forEach((line, i) => {
      const m = line.match(/^(.+?)\s*(\[(\d+:\d+)\])?\s*:\s*(.*)/)
      if (m) rows.push([m[3] || (segments[i] ? formatTime(segments[i].start) : ''), m[1].trim(), m[4].trim()])
    })
    const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' }))
    a.download = name; a.click()
  }
  const formatTime = (s: number) => `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`
  const formatDuration = (s: number) => `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`
  const formatSize = (b: number) => b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : (b / 1024).toFixed(0) + ' KB'
  const formatLiveTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`
  const sentimentColor = (s: string) => s === 'positive' ? '#3ecf8e' : s === 'negative' ? '#ef4444' : '#f59e0b'
  const seekTo = (t: number) => { if (mediaRef.current) { mediaRef.current.currentTime = t; mediaRef.current.play() } }
  const isReady = file && groqKey && anthropicKey && stage !== 'transcribing' && stage !== 'analyzing'

  // ── Stealth mode ──────────────────────────────────────────
  if (stealth) {
    return (
      <div style={{ position: 'fixed', inset: 0, background: '#fff', zIndex: 9999 }}>
        <button
          onClick={() => setStealth(false)}
          style={{
            position: 'fixed', bottom: 20, right: 20,
            background: '#1a1a2e', color: '#fff',
            border: 'none', borderRadius: 20,
            padding: '6px 14px', fontSize: 12,
            cursor: 'pointer', display: 'flex',
            alignItems: 'center', gap: 6, zIndex: 10000,
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: liveRecording ? '#30d158' : '#6c63ff', display: 'inline-block' }} />
          {liveRecording ? `REC ${formatLiveTime(liveSeconds)}` : 'Show'}
        </button>
      </div>
    )
  }

  return (
    <main className={styles.main}>

      {/* Header */}
      <div className={styles.header}>
        <div className={styles.logo}>🎙</div>
        <div style={{ flex: 1 }}>
          <h1 className={styles.title}>Audio Transcriber</h1>
          <p className={styles.subtitle}>Transcribe · Label Speakers · Summarize · 99 Languages · Any Size</p>
        </div>
        <button className={styles.stealthBtn} onClick={() => setStealth(true)} title="Hide this page (stealth mode)">
          👁 Hide
        </button>
      </div>

      {/* API Keys */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardLabel}>API Keys</span>
          <button className={styles.toggleBtn} onClick={() => setShowKeys(!showKeys)}>{showKeys ? 'Hide' : 'Show'}</button>
        </div>
        {showKeys && (
          <div className={styles.keyGrid}>
            <div className={styles.keyField}>
              <label>Groq API Key <span className={styles.tag}>Free</span></label>
              <input type="password" placeholder="gsk_..." value={groqKey} onChange={e => setGroqKey(e.target.value)} className={styles.input} />
              <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer" className={styles.link}>Get free key →</a>
            </div>
            <div className={styles.keyField}>
              <label>Anthropic API Key</label>
              <input type="password" placeholder="sk-ant-..." value={anthropicKey} onChange={e => setAnthropicKey(e.target.value)} className={styles.input} />
              <a href="https://console.anthropic.com" target="_blank" rel="noreferrer" className={styles.link}>Get key →</a>
            </div>
          </div>
        )}
        {(!groqKey || !anthropicKey) && <p className={styles.keyHint}>⚠ Enter both API keys to enable transcription</p>}
        {groqKey && anthropicKey && (
          <div className={styles.keyReadyRow}>
            <p className={styles.keyReady}>✓ API keys configured{rememberKeys && <span className={styles.savedNote}> · saved in this browser</span>}</p>
            <button className={`${styles.rememberBtn} ${rememberKeys ? styles.rememberOn : ''}`} onClick={handleSaveKeys}>
              {savedBadge ? '✓ Saved!' : rememberKeys ? '🔒 Forget keys' : '💾 Remember keys'}
            </button>
          </div>
        )}
      </div>

      {/* Mode tabs */}
      <div className={styles.modeTabs}>
        <button className={`${styles.modeTab} ${mode === 'file' ? styles.modeTabActive : ''}`} onClick={() => setMode('file')}>
          📁 Upload File
        </button>
        <button className={`${styles.modeTab} ${mode === 'live' ? styles.modeTabActive : ''}`} onClick={() => setMode('live')}>
          🔴 Live Capture
        </button>
      </div>

      {/* FILE MODE */}
      {mode === 'file' && (
        <div className={styles.card}>
          <div className={styles.cardLabel}>Upload Audio / Video</div>
          <div
            className={`${styles.dropZone} ${isDragging ? styles.dragging : ''} ${file ? styles.hasFile : ''}`}
            onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => !file && fileInputRef.current?.click()}
          >
            <input ref={fileInputRef} type="file" accept="audio/*,video/*" style={{ display: 'none' }} onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
            {file ? (
              <div className={styles.fileInfo}>
                <span className={styles.fileIcon}>{isVideo ? '🎬' : '🎵'}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className={styles.fileName}>{file.name}</div>
                  <div className={styles.fileMeta}>{formatSize(file.size)} · auto-split & compressed</div>
                </div>
                <button className={styles.changeBtn} onClick={e => { e.stopPropagation(); fileInputRef.current?.click() }}>Change</button>
              </div>
            ) : (
              <>
                <div className={styles.uploadIcon}>📂</div>
                <p className={styles.uploadText}>Drop audio or video here or <b>click to browse</b></p>
                <p className={styles.uploadFormats}>MP3 · M4A · WAV · MP4 · MOV · WebM · Any size</p>
              </>
            )}
          </div>

          {mediaUrl && (
            <div className={styles.playerWrap}>
              {isVideo
                ? <video ref={mediaRef as React.RefObject<HTMLVideoElement>} src={mediaUrl} controls className={styles.videoPlayer} onTimeUpdate={e => setCurrentTime((e.target as HTMLVideoElement).currentTime)} />
                : <audio ref={mediaRef as React.RefObject<HTMLAudioElement>} src={mediaUrl} controls className={styles.audioPlayer} onTimeUpdate={e => setCurrentTime((e.target as HTMLAudioElement).currentTime)} />
              }
            </div>
          )}

          <div className={styles.langRow}>
            <label className={styles.langLabel}>🌐 Language</label>
            <select className={styles.langSelect} value={language} onChange={e => setLanguage(e.target.value)}>
              {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
            </select>
            <span className={styles.langHint}>{language === 'auto' ? 'Auto detect' : 'Manually set'}</span>
          </div>
        </div>
      )}

      {/* LIVE CAPTURE MODE */}
      {mode === 'live' && (
        <div className={styles.card}>
          <div className={styles.cardLabel}>Live Capture — Teams · Zoom · Any App</div>

          <div className={styles.liveInstructions}>
            <div className={styles.liveStep}><span className={styles.liveNum}>1</span><span>Click <b>Start Live Capture</b> below</span></div>
            <div className={styles.liveStep}><span className={styles.liveNum}>2</span><span>Chrome shows a screen share dialog — pick <b>Entire Screen</b> or a specific <b>Tab/Window</b></span></div>
            <div className={styles.liveStep}><span className={styles.liveNum}>3</span><span>⚠ Check <b>"Share system audio"</b> at the bottom of that dialog</span></div>
            <div className={styles.liveStep}><span className={styles.liveNum}>4</span><span>Click <b>Share</b> — transcription starts automatically</span></div>
            <div className={styles.liveStep}><span className={styles.liveNum}>5</span><span>Click <b>👁 Hide</b> in the top-right to enter stealth mode during the call</span></div>
          </div>

          {liveRecording && (
            <div className={styles.liveStatus}>
              <div className={styles.liveIndicator}>
                <span className={styles.liveDot} />
                <span>RECORDING</span>
              </div>
              <span className={styles.liveTimer}>{formatLiveTime(liveSeconds)}</span>
              <span className={styles.liveWordCount}>{liveWords} words</span>
            </div>
          )}

          {liveTranscriptText && (
            <div className={styles.livePreview}>
              <div className={styles.livePreviewLabel}>Live transcript preview</div>
              <div className={styles.livePreviewText}>{liveTranscriptText}</div>
            </div>
          )}

          <div className={styles.langRow} style={{ marginTop: 14 }}>
            <label className={styles.langLabel}>🌐 Language</label>
            <select className={styles.langSelect} value={language} onChange={e => setLanguage(e.target.value)}>
              {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
            </select>
          </div>
        </div>
      )}

      {/* Action button */}
      {mode === 'file' ? (
        <button className={styles.runBtn} disabled={!isReady} onClick={runFileTranscription}>
          {stage === 'transcribing' || stage === 'analyzing'
            ? <><span className={styles.spinner} />{statusMsg}</>
            : '⚡ Transcribe & Analyze'}
        </button>
      ) : (
        <button
          className={`${styles.runBtn} ${liveRecording ? styles.runBtnStop : ''}`}
          disabled={!groqKey}
          onClick={liveRecording ? stopLiveCapture : startLiveCapture}
        >
          {liveRecording ? '⏹ Stop & Analyze' : '🔴 Start Live Capture'}
        </button>
      )}

      {/* Progress */}
      {(stage === 'transcribing' || stage === 'analyzing') && (
        <div className={styles.progressWrap}>
          <div className={styles.progressBar} style={{ width: `${progress}%` }} />
          <div className={styles.progressSteps}>
            <span className={stage === 'transcribing' ? styles.stepActive : styles.stepDone}>{stage === 'transcribing' ? '⏳' : '✓'} Transcribing</span>
            <span className={stage === 'analyzing' ? styles.stepActive : styles.stepWaiting}>{stage === 'analyzing' ? '⏳' : '○'} Analyzing</span>
          </div>
        </div>
      )}

      {error && <div className={styles.errorBox}><span>⚠</span> {error}</div>}

      {/* Results */}
      {stage === 'done' && analysis && transcript && (
        <div className={styles.results}>
          <div className={styles.statsRow}>
            <div className={styles.stat}><span className={styles.statVal}>{transcript.language?.toUpperCase() || '—'}</span><span className={styles.statKey}>Language</span></div>
            <div className={styles.stat}><span className={styles.statVal}>{formatDuration(transcript.duration)}</span><span className={styles.statKey}>Duration</span></div>
            <div className={styles.stat}><span className={styles.statVal}>{analysis.speakers.length}</span><span className={styles.statKey}>Speakers</span></div>
            <div className={styles.stat}><span className={styles.statVal} style={{ color: sentimentColor(analysis.sentiment) }}>{analysis.sentiment}</span><span className={styles.statKey}>Sentiment</span></div>
            <div className={styles.stat}><span className={styles.statVal}>{transcript.text.split(/\s+/).length}</span><span className={styles.statKey}>Words</span></div>
          </div>

          <div className={styles.tabs}>
            {(['summary', 'labeled', 'transcript', 'export'] as const).map(tab => (
              <button key={tab} className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ''}`} onClick={() => setActiveTab(tab)}>
                {tab === 'summary' ? '📋 Summary' : tab === 'labeled' ? '👥 Speakers' : tab === 'transcript' ? '📝 Transcript' : '⬇ Export'}
              </button>
            ))}
          </div>

          <div className={styles.tabContent}>
            {activeTab === 'summary' && (
              <div className={styles.summaryView}>
                <div className={styles.summarySection}><h3>Summary</h3><p>{analysis.summary}</p></div>
                <div className={styles.summarySection}><h3>Key Points</h3><ul className={styles.keyPoints}>{analysis.key_points.map((pt, i) => <li key={i}>{pt}</li>)}</ul></div>
                <div className={styles.summarySection}><h3>Speakers</h3><div className={styles.speakerTags}>{analysis.speakers.map((s, i) => <span key={i} className={styles.speakerTag}>{s}</span>)}</div></div>
              </div>
            )}
            {activeTab === 'labeled' && (
              <div className={styles.transcriptView}>
                <div className={styles.transcriptActions}>
                  <button className={styles.actionBtn} onClick={() => copyText(analysis.labeled_transcript)}>📋 Copy</button>
                  <button className={styles.actionBtn} onClick={() => downloadTxt(analysis.labeled_transcript, 'transcript-labeled.txt')}>💾 txt</button>
                </div>
                <div className={styles.segmentList}>
                  {analysis.labeled_transcript.split('\n').filter((l: string) => l.trim()).map((line: string, i: number) => {
                    const m = line.match(/^(.+?)\s*(\[\d+:\d+\])?\s*:\s*(.*)/)
                    if (m) {
                      const timeMatch = m[2]
                      const timeSecs = timeMatch ? (() => { const [mm, ss] = timeMatch.replace(/[\[\]]/g, '').split(':').map(Number); return mm * 60 + ss })() : null
                      return (
                        <div key={i} className={`${styles.segment} ${timeSecs !== null && currentTime >= timeSecs && currentTime < timeSecs + 10 ? styles.segmentActive : ''}`}
                          onClick={() => timeSecs !== null && seekTo(timeSecs)} style={{ cursor: timeSecs !== null ? 'pointer' : 'default' }}>
                          <span className={styles.segTime} style={{ color: '#a78bfa' }}>{m[1].trim()}{m[2] ? ' ' + m[2] : ''}</span>
                          <span className={styles.segText}>{m[3]}</span>
                        </div>
                      )
                    }
                    return <div key={i} className={styles.segment}><span className={styles.segText}>{line}</span></div>
                  })}
                </div>
              </div>
            )}
            {activeTab === 'transcript' && (
              <div className={styles.transcriptView}>
                <div className={styles.transcriptActions}>
                  <button className={styles.actionBtn} onClick={() => copyText(transcript.text)}>📋 Copy</button>
                  <button className={styles.actionBtn} onClick={() => downloadTxt(transcript.text, 'transcript-raw.txt')}>💾 txt</button>
                </div>
                <div className={styles.segmentList}>
                  {transcript.segments.length > 0 ? transcript.segments.map((seg, i) => (
                    <div key={i} className={`${styles.segment} ${currentTime >= seg.start && currentTime < seg.end ? styles.segmentActive : ''}`} onClick={() => seekTo(seg.start)}>
                      <span className={styles.segTime}>{formatTime(seg.start)}</span>
                      <span className={styles.segText}>{seg.text}</span>
                    </div>
                  )) : <pre className={styles.transcriptText}>{transcript.text}</pre>}
                </div>
              </div>
            )}
            {activeTab === 'export' && (
              <div className={styles.exportView}>
                <p className={styles.exportDesc}>Download in your preferred format</p>
                <div className={styles.exportGrid}>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>📄</div>
                    <div className={styles.exportInfo}><strong>Word Document (.doc)</strong><span>Formatted with summary, key points, and speaker dialogue</span></div>
                    <button className={styles.exportBtn} onClick={() => downloadDocx(analysis.labeled_transcript, analysis.summary, analysis.key_points, 'transcript.doc')}>Download</button>
                  </div>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>📊</div>
                    <div className={styles.exportInfo}><strong>Excel / CSV (.csv)</strong><span>Columns: Timestamp, Speaker, Text — opens in Excel</span></div>
                    <button className={styles.exportBtn} onClick={() => downloadXlsx(transcript.segments, analysis.labeled_transcript, 'transcript.csv')}>Download</button>
                  </div>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>📝</div>
                    <div className={styles.exportInfo}><strong>Plain Text (.txt)</strong><span>Speaker-labeled transcript as plain text</span></div>
                    <button className={styles.exportBtn} onClick={() => downloadTxt(analysis.labeled_transcript, 'transcript.txt')}>Download</button>
                  </div>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>⏱</div>
                    <div className={styles.exportInfo}><strong>Timestamped (.txt)</strong><span>Raw transcript with timestamps per segment</span></div>
                    <button className={styles.exportBtn} onClick={() => downloadTxt(transcript.segments.map(s => `[${formatTime(s.start)}] ${s.text}`).join('\n'), 'transcript-timestamped.txt')}>Download</button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  )
}
