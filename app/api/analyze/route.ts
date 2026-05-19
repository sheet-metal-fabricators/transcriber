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

    const segmentText = segments
      ? segments.map((s: { start: number; end: number; text: string }) =>
          `[${formatTime(s.start)} - ${formatTime(s.end)}] ${s.text.trim()}`
        ).join('\n')
      : transcript

    const message = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      messages: [{
        role: 'user',
        content: `You are analyzing a call recording transcript. Your job is to:
1. Identify distinct speakers based on conversational patterns, topics, and speech style
2. Label them as "Speaker 1", "Speaker 2", etc. (or give descriptive labels if roles are clear, e.g. "Agent", "Customer")
3. Produce a clean labeled transcript
4. Write a concise summary

Here is the transcript with timestamps:
${segmentText}

Respond in this exact JSON format:
{
  "speakers": ["Speaker 1", "Speaker 2"],
  "labeled_transcript": "Speaker 1: Hello, how can I help you today?\nSpeaker 2: Hi, I wanted to ask about...",
  "summary": "This call was about...",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "sentiment": "positive|neutral|negative",
  "duration_note": "The call lasted approximately X minutes"
}

Return ONLY the JSON, no other text.`
      }]
    })

    const content = message.content[0]
    if (content.type !== 'text') throw new Error('Unexpected response type')

    const cleaned = content.text.replace(/```json\n?|\n?```/g, '').trim()
    const analysis = JSON.parse(cleaned)

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
