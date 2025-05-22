# OpenNotes - Audio Transcription Tool

brew install blackhole-2ch ffmpeg

Aggregate Device Setup
On macOS, you can use the built-in Audio MIDI Setup utility to combine multiple audio devices into one "Aggregate Device." This lets you capture both your system audio (via BlackHole 2ch) and your microphone simultaneously. Here’s how:

Open Audio MIDI Setup:

Navigate to Applications > Utilities > Audio MIDI Setup and launch the app.
Create a New Aggregate Device:

In the lower left corner of the window, click the “+” button and select “Create Aggregate Device.”
Select Devices to Include:

In the right-hand panel, you'll see a list of available audio devices.
Check the box next to BlackHole 2ch and your microphone (this could be your built-in mic or an external one).
If available, enable “Drift Correction” for your microphone to help keep the audio in sync.
Rename the Aggregate Device:

IMPORTANT: Rename the aggregate device to "Meeting Recorder"

A Python-based tool for transcribing audio files using AssemblyAI.

## Features

- Transcribe audio files using AssemblyAI with speaker detection and summarization
- Support for both single file and batch processing
- Automatic creation of transcription directories
- Detailed meeting summaries with key points, action items, and sentiment analysis
- Separate files for full transcript and AI-generated summary

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/opennotes.git
cd opennotes
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API key:
```
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

## Usage

### Transcribe a Single File

```bash
python transcriber.py --file path/to/audio.wav
```

### Transcribe All Files in a Directory

```bash
python transcriber.py --recordings path/to/recordings --transcriptions path/to/output
```

### Command Line Options

- `--file`, `-f`: Single audio file to transcribe
- `--recordings`, `-r`: Directory containing audio recordings (default: "recordings")
- `--transcriptions`, `-t`: Directory to save transcriptions (default: "transcriptions")

## Supported Audio Formats

- WAV files (recommended)
- Other formats supported by AssemblyAI

## Output Format

For each audio file, two files are generated:

1. `{filename}.txt` - Contains:
   - Full transcript with speaker labels
   - Meeting summary with key points
   - Action items and decisions
   - Speaker-specific contributions

2. `{filename}_summary.txt` - Contains:
   - AI-generated comprehensive summary
   - Key decisions and agreements
   - Action items with assignments and deadlines
   - Main discussion points and takeaways
   - Issues raised and resolutions
   - Sentiment analysis
   - Speaker-specific contributions

## License

This project is licensed under the MIT License - see the LICENSE file for details.
