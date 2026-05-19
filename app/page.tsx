'use client'

import { useState, useRef, useCallback } from 'react'
import styles from './page.module.css'

type Stage = 'idle' | 'uploading' | 'transcribing' | 'analyzing' | 'done' | 'error'

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
  duration_note: string
}

export default function Home() {
  const [groqKey, setGroqKey] = useState('')
  const [anthropicKey, setAnthropicKey] = useState('')
  const [showKeys, setShowKeys] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [stage, setStage] = useState<Stage>('idle')
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('')
  const [transcript, setTranscript] = useState<TranscriptResult | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<'transcript' | 'labeled' | 'summary'>('summary')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setStage('idle')
    setTranscript(null)
    setAnalysis(null)
    setError('')
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [])

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m}m ${s}s`
  }

  const formatSize = (bytes: number) => {
    if (bytes > 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
    return (bytes / 1024).toFixed(0) + ' KB'
  }

  const run = async () => {
    if (!file || !groqKey || !anthropicKey) return
    setError('')
    setStage('transcribing')
    setProgress(10)
    setStatusMsg('Sending audio to Whisper…')

    try {
      // Step 1: Transcribe
      const fd = new FormData()
      fd.append('file', file)

      const tRes = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'x-groq-key': groqKey },
        body: fd,
      })

      const tData = await tRes.json()
      if (!tRes.ok) throw new Error(tData.error || 'Transcription failed')

      setTranscript(tData)
      setProgress(55)
      setStatusMsg('Transcript ready. Analyzing speakers…')
      setStage('analyzing')

      // Step 2: Analyze
      const aRes = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'x-anthropic-key': anthropicKey,
          'Content-Type': 'application/json',
        },
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

  const copyText = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const downloadTxt = (text: string, name: string) => {
    const blob = new Blob([text], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
  }

  const sentimentColor = (s: string) =>
    s === 'positive' ? '#3ecf8e' : s === 'negative' ? '#ef4444' : '#f59e0b'

  const isReady = file && groqKey && anthropicKey && stage !== 'transcribing' && stage !== 'analyzing'

  return (
    <main className={styles.main}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.logo}>🎙</div>
        <div>
          <h1 className={styles.title}>Audio Transcriber</h1>
          <p className={styles.subtitle}>Transcribe · Label Speakers · Summarize</p>
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
              <input
                type="password"
                placeholder="gsk_..."
                value={groqKey}
                onChange={e => setGroqKey(e.target.value)}
                className={styles.input}
              />
              <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer" className={styles.link}>
                Get free key →
              </a>
            </div>
            <div className={styles.keyField}>
              <label>Anthropic API Key</label>
              <input
                type="password"
                placeholder="sk-ant-..."
                value={anthropicKey}
                onChange={e => setAnthropicKey(e.target.value)}
                className={styles.input}
              />
              <a href="https://console.anthropic.com" target="_blank" rel="noreferrer" className={styles.link}>
                Get key →
              </a>
            </div>
          </div>
        )}
        {(!groqKey || !anthropicKey) && (
          <p className={styles.keyHint}>⚠ Enter both API keys above to enable transcription</p>
        )}
        {groqKey && anthropicKey && (
          <p className={styles.keyReady}>✓ API keys configured</p>
        )}
      </div>

      {/* Upload */}
      <div className={styles.card}>
        <div className={styles.cardLabel}>Upload Audio</div>
        <div
          className={`${styles.dropZone} ${isDragging ? styles.dragging : ''} ${file ? styles.hasFile : ''}`}
          onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*"
            style={{ display: 'none' }}
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          {file ? (
            <div className={styles.fileInfo}>
              <span className={styles.fileIcon}>🎵</span>
              <div>
                <div className={styles.fileName}>{file.name}</div>
                <div className={styles.fileMeta}>{formatSize(file.size)}</div>
              </div>
              <span className={styles.changeFile}>Click to change</span>
            </div>
          ) : (
            <>
              <div className={styles.uploadIcon}>📂</div>
              <p className={styles.uploadText}>Drop audio here or <b>click to browse</b></p>
              <p className={styles.uploadFormats}>M4A · MP3 · WAV · OGG · MP4 · WebM · max 25MB</p>
            </>
          )}
        </div>
      </div>

      {/* Run button */}
      <button
        className={styles.runBtn}
        disabled={!isReady}
        onClick={run}
      >
        {stage === 'transcribing' || stage === 'analyzing'
          ? <><span className={styles.spinner} /> {statusMsg}</>
          : '⚡ Transcribe & Analyze'
        }
      </button>

      {/* Progress */}
      {(stage === 'transcribing' || stage === 'analyzing') && (
        <div className={styles.progressWrap}>
          <div className={styles.progressBar} style={{ width: `${progress}%` }} />
          <div className={styles.progressSteps}>
            <span className={stage === 'transcribing' ? styles.stepActive : styles.stepDone}>
              {stage === 'transcribing' ? '⏳' : '✓'} Transcribing
            </span>
            <span className={stage === 'analyzing' ? styles.stepActive : styles.stepWaiting}>
              {stage === 'analyzing' ? '⏳' : '○'} Analyzing
            </span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className={styles.errorBox}>
          <span>⚠</span> {error}
        </div>
      )}

      {/* Results */}
      {stage === 'done' && analysis && transcript && (
        <div className={styles.results}>
          {/* Stats row */}
          <div className={styles.statsRow}>
            <div className={styles.stat}>
              <span className={styles.statVal}>{transcript.language?.toUpperCase() || '—'}</span>
              <span className={styles.statKey}>Language</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statVal}>{formatDuration(transcript.duration)}</span>
              <span className={styles.statKey}>Duration</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statVal}>{analysis.speakers.length}</span>
              <span className={styles.statKey}>Speakers</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statVal} style={{ color: sentimentColor(analysis.sentiment) }}>
                {analysis.sentiment}
              </span>
              <span className={styles.statKey}>Sentiment</span>
            </div>
            <div className={styles.stat}>
              <span className={styles.statVal}>{transcript.text.split(/\s+/).length}</span>
              <span className={styles.statKey}>Words</span>
            </div>
          </div>

          {/* Tabs */}
          <div className={styles.tabs}>
            {(['summary', 'labeled', 'transcript'] as const).map(tab => (
              <button
                key={tab}
                className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'summary' ? '📋 Summary' : tab === 'labeled' ? '👥 Speaker View' : '📝 Raw Transcript'}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className={styles.tabContent}>
            {activeTab === 'summary' && (
              <div className={styles.summaryView}>
                <div className={styles.summarySection}>
                  <h3>Summary</h3>
                  <p>{analysis.summary}</p>
                </div>
                <div className={styles.summarySection}>
                  <h3>Key Points</h3>
                  <ul className={styles.keyPoints}>
                    {analysis.key_points.map((pt, i) => (
                      <li key={i}>{pt}</li>
                    ))}
                  </ul>
                </div>
                <div className={styles.summarySection}>
                  <h3>Speakers Identified</h3>
                  <div className={styles.speakerTags}>
                    {analysis.speakers.map((s, i) => (
                      <span key={i} className={styles.speakerTag}>{s}</span>
                    ))}
                  </div>
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
                <pre className={styles.transcriptText}>{transcript.text}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  )
}
