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
git clone https://github.com/alexkroman-assembly/opennotes.git
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

1. (macOS only) Install audio dependencies:

```bash
brew install portaudio blackhole-2ch
```

## Setting Up System Audio Recording (macOS)

To record your computer's audio:

1. Open Audio MIDI Setup (press Cmd+Space and search for it)
2. Click the "+" button in the bottom left and select "Create Multi-Output Device"
3. Name it "System Audio"
4. Select both "BlackHole 2ch" and your speakers in the right panel
5. Make sure "BlackHole 2ch" is checked
6. Close Audio MIDI Setup

Now when you run the recorder, select "BlackHole 2ch" as your input device to capture system audio.

## Usage

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
- `transcript_TIMESTAMP_summary.txt` - Default AI-generated summary
- `transcript_TIMESTAMP_lemur_summary.txt` - Lemur generated summary

## Requirements

- Python 3.8+
- AssemblyAI API key
- Audio input device (microphone)
- PortAudio (for audio capture)
- BlackHole 2ch (macOS only, for system audio capture)

## License

MIT
