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

To start recording from your microphone:

```bash
python transcriber.py record
```

### Recording System Audio and Meetings (macOS)

To record system audio on macOS:

1. First, create a Multi-Output Device for audio routing:
   - Open Audio MIDI Setup (Applications > Utilities > Audio MIDI Setup)
   - Click the "+" button in the bottom left and select "Create Multi-Output Device"
   - Name it "Audio Router"
   - In the right panel, check the boxes for:
     - Your speakers (e.g., "MacBook Pro Speakers") - must be first in list
     - "BlackHole 2ch" - must be second, check drift correction
   - Select "Audio Router" as your system output device in System Settings > Sound

1. Then, create an Aggregate Device for recording:
   - In Audio MIDI Setup, click the "+" button again
   - Select "Create Aggregate Device"
   - Name it "Meeting Recorder"
   - In the right panel, check the boxes in this order:
     - Your microphone (e.g., "MacBook Pro Microphone") to capture your voice (must be first)
     - Your speakers (e.g., "MacBook Speakers") to capture your voice (must be second, check drift detection)
     - "BlackHole 2ch" to capture system audio (must be third, check drift detection)

### Run the Recorder

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
