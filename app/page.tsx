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

const CHUNK_SIZE_MB = 24
const CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024

async function splitAudioIntoChunks(file: File): Promise<Blob[]> {
  // If file is under limit, return as-is
  if (file.size <= CHUNK_SIZE_BYTES) return [file]

  // Decode audio, split into equal time chunks
  const arrayBuffer = await file.arrayBuffer()
  const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
  await audioCtx.close()

  const duration = audioBuffer.duration
  const numChunks = Math.ceil(file.size / CHUNK_SIZE_BYTES)
  const chunkDuration = duration / numChunks
  const sampleRate = audioBuffer.sampleRate
  const chunks: Blob[] = []

  for (let i = 0; i < numChunks; i++) {
    const startSample = Math.floor(i * chunkDuration * sampleRate)
    const endSample = Math.min(Math.floor((i + 1) * chunkDuration * sampleRate), audioBuffer.length)
    const chunkLength = endSample - startSample

    // Mix to mono
    const mono = new Float32Array(chunkLength)
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const channelData = audioBuffer.getChannelData(ch)
      for (let s = 0; s < chunkLength; s++) mono[s] += channelData[startSample + s]
    }
    for (let s = 0; s < chunkLength; s++) mono[s] /= audioBuffer.numberOfChannels

    chunks.push(encodeWav(mono, sampleRate))
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

async function transcribeChunk(blob: Blob, groqKey: string, language: string, index: number, total: number): Promise<{ text: string; segments: Segment[]; language: string; duration: number }> {
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
    throw new Error(`Chunk ${index + 1}/${total} failed: ${err.error?.message || res.status}`)
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
  const [activeTab, setActiveTab] = useState<'transcript' | 'labeled' | 'summary'>('summary')
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
      // Split into chunks if needed
      const needsChunking = file.size > CHUNK_SIZE_BYTES
      setStatusMsg(needsChunking ? `Splitting file into chunks...` : 'Uploading to Groq Whisper...')

      const chunks = await splitAudioIntoChunks(file)
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
                  {file.size > CHUNK_SIZE_BYTES && <span className={styles.chunkBadge}> · will auto-split into {Math.ceil(file.size / CHUNK_SIZE_BYTES)} chunks</span>}
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
            {(['summary', 'labeled', 'transcript'] as const).map(tab => (
              <button key={tab} className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ''}`} onClick={() => setActiveTab(tab)}>
                {tab === 'summary' ? '📋 Summary' : tab === 'labeled' ? '👥 Speaker View' : '📝 Raw Transcript'}
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
                  <button className={styles.actionBtn} onClick={() => downloadTxt(analysis.labeled_transcript, 'transcript-labeled.txt')}>💾 Download</button>
                </div>
                <pre className={styles.transcriptText}>{analysis.labeled_transcript}</pre>
              </div>
            )}
            {activeTab === 'transcript' && (
              <div className={styles.transcriptView}>
                <div className={styles.transcriptActions}>
                  <button className={styles.actionBtn} onClick={() => copyText(transcript.text)}>📋 Copy</button>
                  <button className={styles.actionBtn} onClick={() => downloadTxt(transcript.text, 'transcript-raw.txt')}>💾 Download</button>
                </div>
                {/* Clickable segments with timestamps */}
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
          </div>
        </div>
      )}
    </main>
  )
}
