# OpenNotes

A real-time audio transcription tool that uses AssemblyAI's API to transcribe audio from your computer's microphone or system audio.

## Features

- Real-time audio transcription
- Support for multiple audio input devices
- Automatic saving of recordings and transcripts
- Terminal interface with live transcription display
- Support for system audio capture (e.g., recording from your computer's speakers)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/alexkroman-assembly/opennotes.git
cd opennotes
```

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

1. Install the required packages:

```bash
pip install -r requirements.txt
```

1. Create a `.env` file in the project root and add your AssemblyAI API key:

```bash
ASSEMBLYAI_API_KEY=your_api_key_here
```

1. (macOS only) Install audio dependencies:

```bash
brew install portaudio blackhole-2ch
```

Note: If you don't have Homebrew installed, you can install it from [brew.sh](https://brew.sh).

## Usage

### Basic Recording

To start recording from your default microphone:

```bash
python transcriber.py record
```

### Recording System Audio (macOS)

To record system audio on macOS:

1. First, create a Multi-Output Device for audio routing:
   - Open Audio MIDI Setup (Applications > Utilities > Audio MIDI Setup)
   - Click the "+" button in the bottom left and select "Create Multi-Output Device"
   - Name it "Audio Router"
   - In the right panel, check the boxes for:
     - Your speakers (e.g., "MacBook Pro Speakers")
     - "BlackHole 2ch"
   - Select "Audio Router" as your system output device in System Settings > Sound

1. Then, create an Aggregate Device for recording:
   - In Audio MIDI Setup, click the "+" button again
   - Select "Create Aggregate Device"
   - Name it "Meeting Recorder"
   - In the right panel, check the boxes in this order:
     - Your microphone (e.g., "MacBook Pro Microphone") to capture your voice
     - "BlackHole 2ch" to capture system audio
   - Run the recorder with the Meeting Recorder device:

```bash
python transcriber.py record
```

This setup will:

- Route all system audio to both your speakers and BlackHole
- Record your voice through the microphone
- Let you hear the meeting through your speakers
- Capture all system audio (including meeting participants)
- Create a complete transcript of the entire meeting
- Create a Lemur powered summary of the entire meeting

Note: Make sure to test the audio levels before starting an important meeting. You can use the "devices" command to verify your setup:

```bash
python transcriber.py devices
```

### Transcribing Existing Audio Files

To transcribe an existing audio file:

```bash
python transcriber.py transcribe path/to/your/audio/file.mp3
```

### Listing Available Devices

To see all available audio input devices:

```bash
python transcriber.py devices
```

## Output

The tool creates one directory:

- `recordings/`: Contains the recorded audio files

Each recording session generates:

1. A WAV audio file
2. A text file with the transcription
3. A JSON file with detailed transcription data including:
   - Full transcript
   - Speaker labels
   - Auto-highlights
   - Summary

## Requirements

- Python 3.8 or higher
- AssemblyAI API key
- For system audio capture on macOS: [BlackHole](https://github.com/ExistentialAudio/BlackHole)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
