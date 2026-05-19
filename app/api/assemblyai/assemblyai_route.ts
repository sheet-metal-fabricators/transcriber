import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const maxDuration = 300

const ASSEMBLYAI_BASE = 'https://api.assemblyai.com/v2'

async function poll(transcriptId: string, apiKey: string): Promise<Record<string, unknown>> {
  while (true) {
    const res = await fetch(`${ASSEMBLYAI_BASE}/transcript/${transcriptId}`, {
      headers: { authorization: apiKey },
    })
    const data = await res.json() as Record<string, unknown>
    if (data.status === 'completed') return data
    if (data.status === 'error') throw new Error(`AssemblyAI error: ${data.error}`)
    await new Promise(r => setTimeout(r, 3000))
  }
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = req.headers.get('x-assemblyai-key')
    if (!apiKey) return NextResponse.json({ error: 'AssemblyAI API key required' }, { status: 400 })

    const formData = await req.formData()
    const uploadUrl = formData.get('upload_url') as string
    const language = formData.get('language') as string || null

    if (!uploadUrl) return NextResponse.json({ error: 'No upload_url provided' }, { status: 400 })

    // Submit transcription job with speaker diarization
    const transcriptRes = await fetch(`${ASSEMBLYAI_BASE}/transcript`, {
      method: 'POST',
      headers: {
        authorization: apiKey,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        audio_url: uploadUrl,
        speaker_labels: true,
        speakers_expected: null, // auto-detect
        ...(language && language !== 'auto' ? { language_code: language } : { language_detection: true }),
      }),
    })

    const transcriptData = await transcriptRes.json() as Record<string, unknown>
    if (!transcriptRes.ok) throw new Error((transcriptData.error as string) || 'Failed to start transcription')

    // Poll until done
    const result = await poll(transcriptData.id as string, apiKey)

    // Format utterances into labeled transcript
    const utterances = result.utterances as Array<{ speaker: string; text: string; start: number; end: number }> || []
    const labeled = utterances
      .map(u => `Speaker ${u.speaker} [${formatTime(u.start / 1000)}]: ${u.text}`)
      .join('\n')

    const segments = utterances.map(u => ({
      start: u.start / 1000,
      end: u.end / 1000,
      text: u.text,
      speaker: u.speaker,
    }))

    return NextResponse.json({
      text: result.text,
      labeled_transcript: labeled,
      segments,
      language: result.language_code,
      duration: (result.audio_duration as number) || 0,
      speakers: utterances.map(u => `Speaker ${u.speaker}`).filter((s, i, arr) => arr.indexOf(s) === i),
    })

  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'AssemblyAI transcription failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
