'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import styles from './page.module.css'

type Stage = 'idle' | 'transcribing' | 'analyzing' | 'done' | 'error'

interface Segment {
  start: number
  end: number
  text: string
}

interface TranscriptResult {
  text: string
  segments: Segment[]
  language: string
  duration: number
}

interface Analysis {
  speakers: string[]
  labeled_transcript: string
  summary: string
  key_points: string[]
  sentiment: string
}

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

// Target 8kHz mono WAV — good enough for speech, ~1MB per minute
const TARGET_SAMPLE_RATE = 8000
const MAX_CHUNK_SECONDS = 300 // 5 minutes per chunk = ~5MB WAV at 8kHz

async function splitAudioIntoChunks(file: File, onProgress?: (msg: string) => void): Promise<Blob[]> {
  onProgress?.('Decoding audio (this may take a moment for large files)...')
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

    // Mix to mono at source rate
    const mono = new Float32Array(chunkLength)
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const channelData = audioBuffer.getChannelData(ch)
      for (let s = 0; s < chunkLength; s++) mono[s] += channelData[startSample + s]
    }
    for (let s = 0; s < chunkLength; s++) mono[s] /= audioBuffer.numberOfChannels

    // Downsample to TARGET_SAMPLE_RATE (8kHz) — reduces size ~6x vs 48kHz
    const ratio = srcRate / TARGET_SAMPLE_RATE
    const outLength = Math.floor(chunkLength / ratio)
    const downsampled = new Float32Array(outLength)
    for (let s = 0; s < outLength; s++) {
      downsampled[s] = mono[Math.floor(s * ratio)]
    }

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

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))

async function transcribeChunk(blob: Blob, groqKey: string, language: string, index: number, total: number, retries = 5): Promise<{ text: string; segments: Segment[]; language: string; duration: number }> {
  const fd = new FormData()
  const fileName = `chunk_${index + 1}.wav`
  fd.append('file', new File([blob], fileName, { type: 'audio/wav' }))
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
    const err = await res.json()
    const msg = err.error?.message || ''
    // Rate limit — wait and retry
    if (res.status === 429 && retries > 0) {
      const waitMatch = msg.match(/try again in ([\d.]+)s/)
      const waitMs = waitMatch ? Math.ceil(parseFloat(waitMatch[1]) * 1000) + 500 : 4000
      await sleep(waitMs)
      return transcribeChunk(blob, groqKey, language, index, total, retries - 1)
    }
    throw new Error(`Chunk ${index + 1}/${total} failed: ${msg || res.status}`)
  }

  return res.json()
}

export default function Home() {
  const [groqKey, setGroqKey] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [showKeys, setShowKeys] = useState(false)
  const [rememberKeys, setRememberKeys] = useState(false)
  const [savedBadge, setSavedBadge] = useState(false)
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
    const url = URL.createObjectURL(f)
    setMediaUrl(url)
    setIsVideo(f.type.startsWith('video/'))
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const formatDuration = (s: number) => `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`
  const formatSize = (b: number) => b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : (b / 1024).toFixed(0) + ' KB'
  const formatTime = (s: number) => `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`

  const run = async () => {
    if (!file || !groqKey || !anthropicKey) return
    setError('')
    setStage('transcribing')
    setProgress(5)
    setStatusMsg('Preparing audio...')

    try {
      setStatusMsg('Preparing audio...')

      const chunks = await splitAudioIntoChunks(file, setStatusMsg)
      const totalChunks = chunks.length

      // Transcribe each chunk
      const allSegments: Segment[] = []
      let fullText = ''
      let detectedLanguage = 'en'
      let totalDuration = 0
      let timeOffset = 0

      for (let i = 0; i < totalChunks; i++) {
        setStatusMsg(`Transcribing chunk ${i + 1} of ${totalChunks}...`)
        setProgress(5 + Math.round(((i) / totalChunks) * 50))

        // Throttle: wait 3s between chunks to stay under Groq free tier (20 RPM)
        if (i > 0) {
          setStatusMsg(`Chunk ${i} done. Waiting 3s to avoid rate limit...`)
          await sleep(3000)
        }

        const result = await transcribeChunk(chunks[i], groqKey, language, i, totalChunks)

        // Offset segment timestamps for chunks after the first
        if (result.segments) {
          result.segments.forEach(seg => {
            allSegments.push({
              start: seg.start + timeOffset,
              end: seg.end + timeOffset,
              text: seg.text,
            })
          })
        }

        fullText += (i > 0 ? ' ' : '') + result.text.trim()
        detectedLanguage = result.language || detectedLanguage
        totalDuration += result.duration || 0
        timeOffset += result.duration || 0
      }

      const tData: TranscriptResult = {
        text: fullText,
        segments: allSegments,
        language: detectedLanguage,
        duration: totalDuration,
      }

      setTranscript(tData)
      setProgress(60)
      setStatusMsg('Analyzing speakers and generating summary...')
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
      const message = err instanceof Error ? err.message : 'Something went wrong'
      setError(message)
      setStage('error')
    }
  }

  const copyText = (text: string) => navigator.clipboard.writeText(text)

  const downloadDocx = (labeledTranscript: string, summary: string, keyPoints: string[], fileName: string) => {
    const lines = labeledTranscript.split('\n')
    let html = `<html><head><meta charset="UTF-8"></head><body style="font-family:Calibri,sans-serif;font-size:12pt;max-width:800px;margin:40px auto">`
    html += `<h1 style="color:#1a1a2e;border-bottom:2px solid #6c63ff;padding-bottom:8px">Call Transcript</h1>`
    html += `<h2 style="color:#444;font-size:14pt">Summary</h2><p style="line-height:1.6">${summary}</p>`
    html += `<h2 style="color:#444;font-size:14pt">Key Points</h2><ul>${keyPoints.map(p => `<li style="margin:4px 0">${p}</li>`).join('')}</ul>`
    html += `<h2 style="color:#444;font-size:14pt">Full Transcript</h2>`
    lines.forEach(line => {
      if (!line.trim()) return
      const speakerMatch = line.match(/^(Speaker \d+|Agent|Customer|[^:]+)\s*(\[\d+:\d+\])?\s*:\s*(.*)/)
      if (speakerMatch) {
        html += `<p style="margin:8px 0"><strong style="color:#6c63ff">${speakerMatch[1]}${speakerMatch[2] ? ' ' + speakerMatch[2] : ''}:</strong> ${speakerMatch[3]}</p>`
      } else {
        html += `<p style="margin:8px 0">${line}</p>`
      }
    })
    html += '</body></html>'
    const blob = new Blob([html], { type: 'application/msword' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = fileName
    a.click()
  }

  const downloadXlsx = (segments: Segment[], labeledTranscript: string, fileName: string) => {
    // Build CSV that Excel opens natively
    const rows = [['Timestamp', 'Speaker', 'Text']]
    const lines = labeledTranscript.split('\n')
    let segIndex = 0
    lines.forEach(line => {
      if (!line.trim()) return
      const m = line.match(/^(.+?)\s*(\[(\d+:\d+)\])?\s*:\s*(.*)/)
      if (m) {
        const speaker = m[1].trim()
        const time = m[3] || (segments[segIndex] ? formatTimestamp(segments[segIndex].start) : '')
        const text = m[4].trim()
        rows.push([time, speaker, text])
        segIndex++
      }
    })
    const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n')
    const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = fileName
    a.click()
  }

  const formatTimestamp = (s: number) => `${Math.floor(s/60)}:${Math.floor(s%60).toString().padStart(2,'0')}`
  const downloadTxt = (text: string, name: string) => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }))
    a.download = name
    a.click()
  }

  const seekTo = (time: number) => {
    if (mediaRef.current) {
      mediaRef.current.currentTime = time
      mediaRef.current.play()
    }
  }

  const sentimentColor = (s: string) =>
    s === 'positive' ? '#3ecf8e' : s === 'negative' ? '#ef4444' : '#f59e0b'

  const isReady = file && groqKey && anthropicKey && stage !== 'transcribing' && stage !== 'analyzing'

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <div className={styles.logo}>🎙</div>
        <div>
          <h1 className={styles.title}>Audio Transcriber</h1>
          <p className={styles.subtitle}>Transcribe · Label Speakers · Summarize · 99 Languages · Any File Size</p>
        </div>
      </div>

      {/* API Keys */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardLabel}>API Keys</span>
          <button className={styles.toggleBtn} onClick={() => setShowKeys(!showKeys)}>
            {showKeys ? 'Hide' : 'Show'}
          </button>
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

      {/* Upload */}
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
                <div className={styles.fileMeta}>
                  {formatSize(file.size)}
                  <span className={styles.chunkBadge}> · large files auto-split & compressed</span>
                </div>
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

        {/* Built-in media player */}
        {mediaUrl && (
          <div className={styles.playerWrap}>
            {isVideo ? (
              <video
                ref={mediaRef as React.RefObject<HTMLVideoElement>}
                src={mediaUrl}
                controls
                className={styles.videoPlayer}
                onTimeUpdate={e => setCurrentTime((e.target as HTMLVideoElement).currentTime)}
              />
            ) : (
              <audio
                ref={mediaRef as React.RefObject<HTMLAudioElement>}
                src={mediaUrl}
                controls
                className={styles.audioPlayer}
                onTimeUpdate={e => setCurrentTime((e.target as HTMLAudioElement).currentTime)}
              />
            )}
          </div>
        )}

        <div className={styles.langRow}>
          <label className={styles.langLabel}>🌐 Language</label>
          <select className={styles.langSelect} value={language} onChange={e => setLanguage(e.target.value)}>
            {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
          <span className={styles.langHint}>{language === 'auto' ? 'Whisper will detect automatically' : 'Manually set — improves accuracy'}</span>
        </div>
      </div>

      {/* Run */}
      <button className={styles.runBtn} disabled={!isReady} onClick={run}>
        {stage === 'transcribing' || stage === 'analyzing'
          ? <><span className={styles.spinner} />{statusMsg}</>
          : '⚡ Transcribe & Analyze'}
      </button>

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
              <button key={tab} className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ''}`} onClick={() => setActiveTab(tab as 'transcript' | 'labeled' | 'summary' | 'export')}>
                {tab === 'summary' ? '📋 Summary' : tab === 'labeled' ? '👥 Speaker View' : tab === 'transcript' ? '📝 Transcript' : '⬇ Export'}
              </button>
            ))}
          </div>

          <div className={styles.tabContent}>
            {activeTab === 'summary' && (
              <div className={styles.summaryView}>
                <div className={styles.summarySection}><h3>Summary</h3><p>{analysis.summary}</p></div>
                <div className={styles.summarySection}>
                  <h3>Key Points</h3>
                  <ul className={styles.keyPoints}>{analysis.key_points.map((pt, i) => <li key={i}>{pt}</li>)}</ul>
                </div>
                <div className={styles.summarySection}>
                  <h3>Speakers Identified</h3>
                  <div className={styles.speakerTags}>{analysis.speakers.map((s, i) => <span key={i} className={styles.speakerTag}>{s}</span>)}</div>
                </div>
              </div>
            )}
            {activeTab === 'labeled' && (
              <div className={styles.transcriptView}>
                <div className={styles.transcriptActions}>
                  <button className={styles.actionBtn} onClick={() => copyText(analysis.labeled_transcript)}>📋 Copy</button>
                  <button className={styles.actionBtn} onClick={() => downloadTxt(analysis.labeled_transcript, 'transcript-labeled.txt')}>💾 Download .txt</button>
                </div>
                <div className={styles.segmentList}>
                  {analysis.labeled_transcript.split('\n').filter((l: string) => l.trim()).map((line: string, i: number) => {
                    const m = line.match(/^(.+?)\s*(\[\d+:\d+\])?\s*:\s*(.*)/)
                    if (m) {
                      const timeMatch = m[2]
                      const timeSecs = timeMatch ? (() => { const [mm, ss] = timeMatch.replace(/[\[\]]/g,'').split(':').map(Number); return mm*60+ss })() : null
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
                  <button className={styles.actionBtn} onClick={() => downloadTxt(transcript.text, 'transcript-raw.txt')}>💾 Download .txt</button>
                </div>
                <div className={styles.segmentList}>
                  {transcript.segments.length > 0 ? transcript.segments.map((seg, i) => (
                    <div
                      key={i}
                      className={`${styles.segment} ${currentTime >= seg.start && currentTime < seg.end ? styles.segmentActive : ''}`}
                      onClick={() => seekTo(seg.start)}
                    >
                      <span className={styles.segTime}>{formatTime(seg.start)}</span>
                      <span className={styles.segText}>{seg.text}</span>
                    </div>
                  )) : (
                    <pre className={styles.transcriptText}>{transcript.text}</pre>
                  )}
                </div>
              </div>
            )}
            {activeTab === 'export' && (
              <div className={styles.exportView}>
                <p className={styles.exportDesc}>Download the transcript in your preferred format</p>
                <div className={styles.exportGrid}>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>📄</div>
                    <div className={styles.exportInfo}>
                      <strong>Word Document (.doc)</strong>
                      <span>Formatted transcript with summary, key points, and speaker-labeled dialogue. Opens in Microsoft Word.</span>
                    </div>
                    <button className={styles.exportBtn} onClick={() => downloadDocx(analysis.labeled_transcript, analysis.summary, analysis.key_points, 'transcript.doc')}>Download</button>
                  </div>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>📊</div>
                    <div className={styles.exportInfo}>
                      <strong>Excel / CSV (.csv)</strong>
                      <span>Spreadsheet with columns: Timestamp, Speaker, Text. Opens directly in Excel.</span>
                    </div>
                    <button className={styles.exportBtn} onClick={() => downloadXlsx(transcript.segments, analysis.labeled_transcript, 'transcript.csv')}>Download</button>
                  </div>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>📝</div>
                    <div className={styles.exportInfo}>
                      <strong>Plain Text (.txt)</strong>
                      <span>Simple text file with the full transcript. Works everywhere.</span>
                    </div>
                    <button className={styles.exportBtn} onClick={() => downloadTxt(analysis.labeled_transcript, 'transcript.txt')}>Download</button>
                  </div>
                  <div className={styles.exportCard}>
                    <div className={styles.exportIcon}>⏱</div>
                    <div className={styles.exportInfo}>
                      <strong>Timestamped (.txt)</strong>
                      <span>Raw transcript with timestamps for each segment.</span>
                    </div>
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
