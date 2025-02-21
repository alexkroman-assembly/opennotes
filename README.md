# OpenNotes AI

## Open Source AI Notetaker for Your Meetings

Transcribe, summarize, search, and analyze all your team conversations—completely open source. Own your data, customize workflows, and gain AI-powered insights.

Supported Automatic Speech Recognition (ASR) Engines:

- Whisper Python: Open source, slower performance, lower quality.
- Whisper.cpp: Open source, fast performance, better quality.
- AssemblyAI: Fast and high quality (paid service).

## Features

- Multi-ASR Support: Choose between Whisper Python, Whisper.cpp, or Assembly based on your performance and quality needs.
- Aggregate Device: Capture system audio (e.g., microphone and virtual audio devices) for comprehensive audio recording.
- Command Line & Web Interface: Interact with transcripts via CLI or a web server.
- Fully Open Source: Customize, self-host, and integrate seamlessly into your workflow.

## Getting started

### Prerequsites

Before you begin, ensure you have the following installed:

- BlackHole 2ch (for audio routing)
- FFmpeg (for audio recording)
- Node.js and npm

Install BlackHole and FFmpeg using Homebrew:

```bash
brew install blackhole-2ch ffmpeg
```

Additionally, install AIChat using Homebrew:

```bash
brew install aichat
```

Then install Node.js dependencies:

```bash
npm install
```

### Aggregate Device Setup

On macOS, you can use the built-in Audio MIDI Setup utility to combine multiple audio devices into one "Aggregate Device." This lets you capture both your system audio (via BlackHole 2ch) and your microphone simultaneously. Here’s how:

Open Audio MIDI Setup:

- Navigate to Applications > Utilities > Audio MIDI Setup and launch the app.

Create a New Aggregate Device:

- In the lower left corner of the window, click the “+” button and select “Create Aggregate Device.”

Select Devices to Include:

- In the right-hand panel, you'll see a list of available audio devices.
- Check the box next to BlackHole 2ch and your microphone (this could be your built-in mic or an external one).
- If available, enable “Drift Correction” for your microphone to help keep the audio in sync.

Rename the Aggregate Device:

- **IMPORTANT:** Rename the aggregate device to "Meeting Recorder"

### ASR Installation

Choose one of the following ASR engines based on your requirements:

#### Whisper.cpp (Fast, Open Source)

- Clone and install from the Whisper.cpp GitHub repository.
- Copy the provided example.env to .env:

```bash
cp example.env .env
```

#### Whisper Python (Slow, Free)

Install Whisper using Homebrew:

```bash
brew install whisper
```

Open .env and add the path to your Whisper CPP executable.

#### Assembly (Fast, Paid)

- Copy the provided example.env to .env:

```bash
cp example.env .env
```

- Open .env and add your Assembly API key.

## Usage

### Recording Audio

To capture recordings, use:

```bash
npm run record
```

Recordings will be saved in the `recordings/` directory.

### Transcribing Recordings

Select the ASR engine for transcription:

#### Whisper Python

```bash
npm run transcribe:whisper-python
```

#### Whisper.cpp

```bash
npm run transcribe:whisper-cpp
```

#### Assembly

```bash
npm run transcribe:assembly
```

### Interacting with Transcripts

Use AIChat to explore your transcripts via CLI or web interface.

#### Command Line Interface (CLI)

Launch AIChat in the terminal:

```bash
npm run cli
```

#### Web Server

Start the web server to interact with transcripts in a browser:

```bash
npm run server
```

Open in your browser:

```bash
npm run browser
```

Or follow this link:

<http://localhost:8000/playground?model=default&rag=transcriber>

### License

This project is licensed under the MIT License.
