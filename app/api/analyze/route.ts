import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'

export const runtime = 'nodejs'
export const maxDuration = 120

export async function POST(req: NextRequest) {
  try {
    const anthropicKey = req.headers.get('x-anthropic-key')
    if (!anthropicKey) return NextResponse.json({ error: 'Anthropic API key required' }, { status: 400 })

    const { transcript, segments } = await req.json()
    if (!transcript) return NextResponse.json({ error: 'No transcript provided' }, { status: 400 })

    const client = new Anthropic({ apiKey: anthropicKey })

    // Build segment text — pass full transcript, Claude handles long context well
    const MAX_CHARS = 30000
    const segmentText = segments && segments.length > 0
      ? segments
          .map((s: { start: number; end: number; text: string }) =>
            `[${formatTime(s.start)}] ${s.text.trim()}`
          )
          .join('\n')
          .slice(0, MAX_CHARS)
      : transcript.slice(0, MAX_CHARS)

    const message = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 8000,
      messages: [{
        role: 'user',
        content: `Analyze this call recording transcript. Identify speakers based on conversational patterns.

Transcript:
${segmentText}

Respond ONLY with a JSON object in this exact format, no markdown, no code fences, no extra text:
{"speakers":["Speaker 1","Speaker 2"],"labeled_transcript":"Speaker 1 [0:00]: Hello how are you?\\nSpeaker 2 [0:05]: I am fine thanks.","summary":"2-3 sentence summary of the call","key_points":["Key point 1","Key point 2","Key point 3","Key point 4","Key point 5"],"sentiment":"positive"}`
      }]
    })

    const content = message.content[0]
    if (content.type !== 'text') throw new Error('Unexpected response type')

    const raw = content.text.trim()
    const jsonMatch = raw.match(/\{[\s\S]*\}/)
    if (!jsonMatch) throw new Error('No JSON found in response')

    const analysis = JSON.parse(jsonMatch[0])
    return NextResponse.json(analysis)

  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Analysis failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}
