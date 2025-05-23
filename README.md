directions to brew install blackhole

- create aggregate device

check mic first

Name: Meeting Input
- check Blackhole 2ch
- check Macbook Pro Microphone and check drift correction

- create multioutput device

check speakers first

Name: Meeting Output
- check Blackhole 2ch
- check Macbook Pro Speakers and check drift correction


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

1. To start recording and transcribing:

```bash
python transcriber.py record
```

1. To transcribe an existing audio file:

```bash
python transcriber.py transcribe --file path/to/audio.wav
```

1. To transcribe all WAV files in a directory:

```bash
python transcriber.py transcribe --recordings path/to/recordings --transcriptions path/to/output
```

## Output

The tool generates two types of files for each recording:

1. A transcript file (`.txt`) containing the full conversation with speaker labels
1. A summary file (`_summary.txt`) containing an AI-generated summary of the conversation

## License

MIT
