# OpenNotes

A simple command-line tool for real-time audio transcription using AssemblyAI.

## Features

- 🎙️ Real-time audio recording and transcription
- 💾 Saves audio as MP3 files
- 📝 Generates text transcripts and summaries
- 🤖 AI-powered summaries using Lemur

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/opennotes.git
cd opennotes
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Create a `.env` file with your AssemblyAI API key:

```bash
ASSEMBLYAI_API_KEY=your_api_key_here
```

## Usage

1. List available audio devices:

```bash
python transcriber.py devices
```

1. Start recording:

```bash
python transcriber.py record
```

1. Transcribe an existing audio file:

```bash
python transcriber.py transcribe path/to/audio.mp3
```

## Output Files

The tool creates several files in the `recordings` directory:

- `recorded_audio_TIMESTAMP.mp3` - The recorded audio
- `transcript_TIMESTAMP_text.txt` - Plain text transcript
- `transcript_TIMESTAMP_summary.txt` - AI-generated summary
- `transcript_TIMESTAMP_lemur_summary.txt` - Detailed AI analysis

## Requirements

- Python 3.8+
- AssemblyAI API key
- Audio input device (microphone)

## License

MIT
