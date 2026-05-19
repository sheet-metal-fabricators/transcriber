import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const maxDuration = 300

export async function POST(req: NextRequest) {
  try {
    const apiKey = req.headers.get('x-assemblyai-key')
    if (!apiKey) return NextResponse.json({ error: 'AssemblyAI API key required' }, { status: 400 })

    const formData = await req.formData()
    const file = formData.get('file') as File
    if (!file) return NextResponse.json({ error: 'No file provided' }, { status: 400 })

    // Upload directly to AssemblyAI storage
    const uploadRes = await fetch('https://api.assemblyai.com/v2/upload', {
      method: 'POST',
      headers: {
        authorization: apiKey,
        'content-type': 'application/octet-stream',
      },
      body: await file.arrayBuffer(),
    })

    const uploadData = await uploadRes.json() as { upload_url?: string; error?: string }
    if (!uploadRes.ok) throw new Error(uploadData.error || 'Upload failed')

    return NextResponse.json({ upload_url: uploadData.upload_url })
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Upload failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
