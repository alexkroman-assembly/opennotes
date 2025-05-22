#!/usr/bin/env python3
import os
import sys
import json
import threading
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode
import click  # type: ignore
from dotenv import load_dotenv  # type: ignore
import pyaudio  # type: ignore
import websocket  # type: ignore
import subprocess

# Load environment variables
load_dotenv()

# AssemblyAI API configuration
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing AssemblyAI API key. Set ASSEMBLYAI_API_KEY in .env")

HEADERS = {
    "authorization": API_KEY,
    "content-type": "application/json"
}

# Streaming API Configuration
CONNECTION_PARAMS = {
    "sample_rate": 16000,
    "formatted_finals": True,
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Audio Configuration
FRAMES_PER_BUFFER = 800  # 50ms of audio (0.05s * 16000Hz)
SAMPLE_RATE = CONNECTION_PARAMS["sample_rate"]
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Global variables for audio stream and websocket
audio = None
stream = None
ws_app = None
audio_thread = None
stop_event = threading.Event()

FFMPEG_CMD = [
    "ffmpeg",
    "-f", "avfoundation",
    "-i", ":Meeting Recorder",
    "-ar", "16000",
    "-ac", "1",
    "-f", "s16le",
    "-"
]

def on_open(ws):
    """Called when the WebSocket connection is established."""
    print("WebSocket connection opened.")
    print(f"Connected to: {API_ENDPOINT}")

    def stream_ffmpeg_audio(ws):
        process = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            print("Starting ffmpeg audio streaming...")
            while not stop_event.is_set():
                data = process.stdout.read(3200)  # 100ms of audio at 16kHz, 16-bit mono
                if not data:
                    break
                ws.send(data, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(f"Error streaming audio: {e}")
        finally:
            process.terminate()
            print("ffmpeg audio streaming stopped.")

    global audio_thread
    audio_thread = threading.Thread(target=stream_ffmpeg_audio, args=(ws,))
    audio_thread.daemon = True
    audio_thread.start()

def on_message(ws, message):
    try:
        data = json.loads(message)
        msg_type = data.get('type')

        if msg_type == "Begin":
            session_id = data.get('id')
            expires_at = data.get('expires_at')
            print(f"\nSession began: ID={session_id}, ExpiresAt={datetime.fromtimestamp(expires_at)}")
        elif msg_type == "Turn":
            transcript = data.get('transcript', '')
            formatted = data.get('turn_is_formatted', False)

            if formatted:
                print('\r' + ' ' * 80 + '\r', end='')
                print(transcript)
            else:
                print(f"\r{transcript}", end='')
        elif msg_type == "Termination":
            audio_duration = data.get('audio_duration_seconds', 0)
            session_duration = data.get('session_duration_seconds', 0)
            print(f"\nSession Terminated: Audio Duration={audio_duration}s, Session Duration={session_duration}s")
    except json.JSONDecodeError as e:
        print(f"Error decoding message: {e}")
    except Exception as e:
        print(f"Error handling message: {e}")

def on_error(ws, error):
    """Called when a WebSocket error occurs."""
    print(f"\nWebSocket Error: {error}")
    stop_event.set()

def on_close(ws, close_status_code, close_msg):
    """Called when the WebSocket connection is closed."""
    print(f"\nWebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
    global stream, audio
    stop_event.set()

    if stream:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        stream = None
    if audio:
        audio.terminate()
        audio = None
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=1.0)

def record_audio(output_dir: Path) -> None:
    """Record audio using ffmpeg and stream to AssemblyAI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global ws_app

    def stream_ffmpeg_audio(ws):
        process = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            print("Starting ffmpeg audio streaming...")
            while not stop_event.is_set():
                data = process.stdout.read(3200)  # 100ms of audio at 16kHz, 16-bit mono
                if not data:
                    break
                ws.send(data, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(f"Error streaming audio: {e}")
        finally:
            process.terminate()
            print("ffmpeg audio streaming stopped.")

    def on_open(ws):
        print("WebSocket connection opened.")
        print(f"Connected to: {API_ENDPOINT}")
        global audio_thread
        audio_thread = threading.Thread(target=stream_ffmpeg_audio, args=(ws,))
        audio_thread.daemon = True
        audio_thread.start()

    ws_app = websocket.WebSocketApp(
        API_ENDPOINT,
        header={"Authorization": API_KEY},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws_app.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    try:
        while ws_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping...")
        stop_event.set()
        if ws_app and ws_app.sock and ws_app.sock.connected:
            try:
                terminate_message = {"type": "Terminate"}
                print(f"Sending termination message: {json.dumps(terminate_message)}")
                ws_app.send(json.dumps(terminate_message))
                time.sleep(5)
            except Exception as e:
                print(f"Error sending termination message: {e}")
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        stop_event.set()
        if ws_app:
            ws_app.close()
        ws_thread.join(timeout=2.0)
    finally:
        print("Cleanup complete. Exiting.")

def get_output_paths(base_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Get paths for transcript and summary files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / f"{base_path.stem}.txt",
        output_dir / f"{base_path.stem}_summary.txt"
    )

def upload_file(file_path: Path) -> str:
    """Upload a file to AssemblyAI and return the URL."""
    upload_url = "https://api.assemblyai.com/v2/upload"
    
    with open(file_path, "rb") as f:
        response = requests.post(
            upload_url,
            headers={"authorization": API_KEY},
            data=f
        )
    
    if response.status_code != 200:
        raise Exception(f"Upload failed with status {response.status_code}: {response.text}")
    
    return response.json()["upload_url"]

def transcribe_file(file_path: Path, output_dir: Path) -> None:
    """Transcribe a single audio file using AssemblyAI."""
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return

    transcript_file, summary_file = get_output_paths(file_path, output_dir)
    if transcript_file.exists() and summary_file.exists():
        print(f"Skipping {file_path}, transcription files already exist.")
        return

    try:
        print(f"Transcribing {file_path} with AssemblyAI...")

        # Upload the file
        audio_url = upload_file(file_path)
        
        # Submit transcription request
        transcript_url = "https://api.assemblyai.com/v2/transcript"
        transcript_request = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "summarization": True,
            "summary_model": "conversational",
            "summary_type": "bullets_verbose"
        }
        
        response = requests.post(transcript_url, json=transcript_request, headers=HEADERS)
        if response.status_code != 200:
            raise Exception(f"Transcription request failed with status {response.status_code}: {response.text}")
        
        transcript_id = response.json()["id"]
        
        # Poll for completion
        while True:
            response = requests.get(f"{transcript_url}/{transcript_id}", headers=HEADERS)
            if response.status_code != 200:
                raise Exception(f"Failed to get transcript status: {response.text}")
            
            status = response.json()["status"]
            if status == "completed":
                break
            elif status == "error":
                raise Exception(f"Transcription failed: {response.json().get('error')}")
            
            print("Transcription in progress...")
            time.sleep(3)
        
        # Get the completed transcript
        transcript_data = response.json()
        
        if not transcript_data.get("utterances"):
            print(f"Transcription failed or is incomplete for {file_path}")
            return

        # Save transcript
        transcript_content = "\n".join(
            f"Speaker {utt['speaker']}: {utt['text']}" 
            for utt in transcript_data["utterances"]
        )
        transcript_file.write_text(transcript_content)
        print(f"Transcript saved to {transcript_file}")

        # Save summary
        summary = transcript_data.get("summary", "No summary available.")
        summary_file.write_text(f"LLM Transcript Summary:\n{summary}")
        print(f"Summary saved to {summary_file}")

    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")

def transcribe_directory(input_dir: Path, output_dir: Path) -> None:
    """Transcribe all WAV files in a directory."""
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist.")
        return

    for wav_file in input_dir.glob("*.wav"):
        transcribe_file(wav_file, output_dir)

    print("All transcriptions complete.")

@click.group()
def cli():
    """Audio recording and transcription tool."""
    pass

@cli.command()
@click.option(
    "--recordings", "-r",
    default="recordings",
    help="Directory containing audio recordings"
)
@click.option(
    "--transcriptions", "-t",
    default="transcriptions",
    help="Directory to save transcriptions"
)
@click.option(
    "--file", "-f",
    help="Single audio file to transcribe"
)
def transcribe(recordings: str, transcriptions: str, file: Optional[str]):
    """Transcribe audio files using AssemblyAI."""
    try:
        if file:
            transcribe_file(Path(file), Path(transcriptions))
        else:
            transcribe_directory(Path(recordings), Path(transcriptions))
        print("Done!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option(
    "--output-dir", "-o",
    default="recordings",
    help="Directory to save recordings"
)
def record(output_dir: str):
    """Record audio using ffmpeg."""
    try:
        record_audio(Path(output_dir))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli() 