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

    // Truncate transcript if very long to avoid token limits
    const MAX_CHARS = 12000
    const segmentText = segments && segments.length > 0
      ? segments
          .map((s: { start: number; end: number; text: string }) =>
            `[${formatTime(s.start)}] ${s.text.trim()}`
          )
          .join('\n')
          .slice(0, MAX_CHARS)
      : transcript.slice(0, MAX_CHARS)

    const wasTruncated = transcript.length > MAX_CHARS

    const message = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      messages: [{
        role: 'user',
        content: `Analyze this call recording transcript and respond ONLY with a valid JSON object.

Transcript${wasTruncated ? ' (first portion)' : ''}:
${segmentText}

Return ONLY this JSON structure, no other text, no markdown, no code fences:
{"speakers":["Speaker 1","Speaker 2"],"labeled_transcript":"Speaker 1: Hello...\\nSpeaker 2: Hi...","summary":"Brief summary here","key_points":["Point 1","Point 2","Point 3"],"sentiment":"positive"}`
      }]
    })

    const content = message.content[0]
    if (content.type !== 'text') throw new Error('Unexpected response type')

    // Robustly extract JSON even if Claude adds extra text
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
