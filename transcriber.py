#!/usr/bin/env python3
import os
import sys
import json
import threading
import time
import requests  # type: ignore
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
import typer  # type: ignore
from dotenv import load_dotenv  # type: ignore
import sounddevice as sd  # type: ignore
import numpy as np  # type: ignore
import websocket  # type: ignore
from pydub import AudioSegment  # type: ignore
from rich.console import Console  # type: ignore
from typing import Optional, List, Dict, Any, Tuple
from requests.adapters import HTTPAdapter  # type: ignore
from urllib3.util.retry import Retry  # type: ignore
import pyfiglet  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

# Load environment variables
load_dotenv()

# Create console and show banner
console = Console()
banner = pyfiglet.figlet_format("opennotes", font="slant")
console.print(f"[bold blue]{banner}[/]")
console.print("[bold green]Real-time Audio Transcription Tool[/]\n")

# AssemblyAI API configuration
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing AssemblyAI API key. Set ASSEMBLYAI_API_KEY in .env")

# Audio Configuration
DTYPE = np.int16  # Keep this as it's a fundamental type

# Streaming API Configuration
CONNECTION_PARAMS: Dict[str, Any] = {
    "formatted_finals": True,
}
API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"

# Global variables
console = Console()
stop_event = threading.Event()
stream = None
ws_app = None
audio_thread = None
recorded_frames: List[bytes] = []  # Store audio frames for WAV file
current_device_id: Optional[int] = None  # Store the current device ID
current_streaming_file: Optional[Path] = None  # Store the current streaming file path
show_partials: bool = True  # Control whether to show partial transcriptions
current_session_dir: Optional[Path] = None  # Will be set when session begins
current_session_id: Optional[str] = None  # Store the current session ID

def get_input_device_id(device_name: str) -> Optional[int]:
    """Get the device ID for a given device name."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['name'] == device_name and device['max_input_channels'] > 0:
            return i
    return None

def list_devices() -> None:
    """List all available audio input devices."""
    devices = sd.query_devices()
    
    console.print("\nInput Devices:")
    input_devices = [(i, device) for i, device in enumerate(devices) if device['max_input_channels'] > 0]
    
    if not input_devices:
        console.print("No input devices found.")
        return
        
    for i, device in input_devices:
        console.print(f"{i}: {device['name']}")
        console.print(f"    Channels: {device['max_input_channels']}")
        console.print(f"    Sample Rate: {device['default_samplerate']} Hz")
        console.print()

def process_audio_data(indata: np.ndarray, for_streaming: bool = False) -> bytes:
    """Process audio data for both streaming and recording."""
    if for_streaming:
        # For streaming, we need mono audio at 16kHz with PCM S16LE encoding
        # Convert to float32 for processing, properly scaled
        audio_float = indata.astype(np.float32) / 32768.0  # Scale to [-1.0, 1.0]
        
        if audio_float.shape[1] > 1:  # If we have multiple channels
            # Mix down to mono by averaging channels
            audio_data = np.mean(audio_float, axis=1)
        else:
            # If already mono, just flatten
            audio_data = audio_float.flatten()
            
        # Convert back to int16, properly scaled
        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        
        # Ensure we have data
        if len(audio_data) == 0:
            return b''
            
        return audio_data.tobytes()
    else:
        # For recording, keep all channels but ensure proper shape
        audio_data = indata.reshape(-1, indata.shape[1])
        return audio_data.tobytes()

def on_open(ws):
    """Called when the WebSocket connection is established."""
    console.print("Connected to: " + API_ENDPOINT + "\n")

    def stream_audio():
        global stream
        try:
            # Get device info to determine correct channel count
            if current_device_id is None:
                raise Exception("No device selected")
            
            device_info = sd.query_devices(current_device_id)
            channels = device_info['max_input_channels']  # Use all available channels
            sample_rate = int(device_info['default_samplerate'])
            frames_per_buffer = int(sample_rate * 0.05)  # 50ms of audio
            
            # Update streaming API parameters with current sample rate
            global API_ENDPOINT
            CONNECTION_PARAMS["sample_rate"] = sample_rate
            API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"
            
            console.print(f"Using {channels} channels for device")
            console.print(f"Device sample rate: {sample_rate} Hz")
            console.print(f"Frame size: {frames_per_buffer} samples")
            
            if channels > 1:
                console.print("Note: Mixing down to mono for streaming, preserving all channels for recording")
            
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=DTYPE,
                device=current_device_id,
                callback=audio_callback
            ):
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            console.print(f"Error streaming audio: {e}")
        console.print("Audio streaming stopped.")

    global audio_thread
    audio_thread = threading.Thread(target=stream_audio)
    audio_thread.daemon = True
    audio_thread.start()

def audio_callback(indata, frames, time, status):
    """Callback for audio stream."""
    if status:
        console.print(f"[yellow]Audio callback status: {status}[/]")
    
    if not stop_event.is_set() and ws_app and ws_app.sock and ws_app.sock.connected:
        try:
            # Process audio data for streaming (mono) and recording (multi-channel)
            streaming_audio = process_audio_data(indata, for_streaming=True)
            recording_audio = process_audio_data(indata, for_streaming=False)
            
            # Send mono audio to WebSocket
            if streaming_audio:  # Only send if we have data
                ws_app.send(streaming_audio, websocket.ABNF.OPCODE_BINARY)
            
            # Store full multi-channel audio for recording
            if recording_audio:  # Only store if we have data
                recorded_frames.append(recording_audio)
        except Exception as e:
            console.print(f"[red]Error in audio callback: {e}[/]")
            console.print("[yellow]Debug info:[/]")
            console.print(f"- Input shape: {indata.shape}")
            console.print(f"- Input dtype: {indata.dtype}")
            console.print(f"- Streaming audio size: {len(streaming_audio) if streaming_audio else 0}")
            stop_event.set()

def on_message(ws, message):
    """Handle incoming WebSocket messages."""
    try:
        data = json.loads(message)
        msg_type = data.get('type')

        if msg_type == "Begin":
            session_id = data.get('id')
            expires_at = data.get('expires_at')

            # Create session-specific directory
            global current_streaming_file, current_session_dir, current_session_id
            current_session_id = session_id
            current_session_dir = Path("recordings") / session_id
            current_session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create streaming output file in session directory
            current_streaming_file = current_session_dir / f"streaming-{session_id}.txt"
            with open(current_streaming_file, "w") as f:
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Started at: {datetime.fromtimestamp(expires_at)}\n\n")
                console.print(f"Session began: ID={session_id}")
                console.print(f"Expires at: {datetime.fromtimestamp(expires_at)}")
                console.print(f"Session directory: {current_session_dir}")
                console.print("Recording started. Press Ctrl+C to stop.")
        elif msg_type == "Turn":
            transcript = data.get('transcript', '')
            formatted = data.get('turn_is_formatted', False)

            if transcript.strip():  # Only print if we have content
                # Show text based on format preference
                if formatted:
                    # Show final transcription on a new line with green color
                    console.print(f"[green]✓ {transcript}[/]")
                    # Always save formatted text to file
                    if current_streaming_file:
                        with open(current_streaming_file, "a") as f:
                            f.write(f"{transcript}\n")
                elif show_partials:  # Only show partials if enabled
                    # Show partial on a new line with yellow color and a different prefix
                    console.print(f"[yellow]⟳ {transcript}[/]")
        elif msg_type == "Termination":
            audio_duration = data.get('audio_duration_seconds', 0)
            session_duration = data.get('session_duration_seconds', 0)
            console.print("Session Terminated")
            console.print(f"Audio Duration: {audio_duration:.1f}s")
            console.print(f"Session Duration: {session_duration:.1f}s")
            console.print("\nRecording Complete")
            # Add session end info to streaming file
            if current_streaming_file:
                with open(current_streaming_file, "a") as f:
                    f.write(f"\nSession ended at: {datetime.now()}\n")
                    f.write(f"Audio Duration: {audio_duration:.1f}s\n")
                    f.write(f"Session Duration: {session_duration:.1f}s\n")
    except Exception as e:
        console.print(f"Error handling message: {str(e)}")

def on_error(ws, error):
    """Called when a WebSocket error occurs."""
    console.print(f"WebSocket Error: {error}")
    stop_event.set()

def on_close(ws, close_status_code, close_msg):
    """Called when the WebSocket connection is closed."""
    console.print(f"WebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
    global stream
    stop_event.set()

    if stream:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        stream = None
    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=1.0)

def save_audio_file(output_dir: Path, recorded_frames: list) -> Optional[Path]:
    """Save recorded audio frames to a WAV file."""
    if not recorded_frames:
        console.print("No audio data recorded.")
        return None
    
    try:
        # Get device info for sample rate and channels
        if current_device_id is not None:
            device_info = sd.query_devices(current_device_id)
            sample_rate = int(device_info['default_samplerate'])
            channels = device_info['max_input_channels']
        else:
            sample_rate = 48000  # Default to standard sample rate if no device
            channels = 2  # Default to stereo if no device
            
        # Combine all frames into a single array
        all_frames = b''.join(recorded_frames)
        audio_data = np.frombuffer(all_frames, dtype=DTYPE)
        
        # Create AudioSegment from numpy array
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=channels
        )
        
        # Export to WAV in the session directory
        filename = output_dir / f"audio-{current_session_id}.wav"
        audio_segment.export(str(filename), format="wav")
        
        console.print(f"Audio saved to: {filename}")
        console.print(f"Duration: {len(audio_segment) / 1000:.2f} seconds")
        return filename
        
    except Exception as e:
        console.print(f"Error saving WAV file: {e}")
        console.print("Debug info:")
        console.print(f"- Number of frames: {len(recorded_frames)}")
        if recorded_frames:
            console.print(f"- First frame size: {len(recorded_frames[0])} bytes")
            console.print(f"- Device channels: {channels}")
        return None

def transcribe_file(file_path: Path, options: Optional[Dict[str, Any]] = None, use_slam: bool = False) -> Dict[str, Any]:
    """Transcribe a file using AssemblyAI."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Upload the file
    upload_url = upload_file(file_path)
    
    # Prepare transcription request
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    headers = {
        "authorization": API_KEY,
        "content-type": "application/json"
    }
    
    data = {
        "audio_url": upload_url,
        "punctuate": True,
        "format_text": True,
        "speaker_labels": True,
        "auto_highlights": True,
        "summarization": True,
        "summary_model": "informative",
        "summary_type": "bullets"
    }
    
    # Add SLAM model if requested
    if use_slam:
        data["speech_model"] = "slam-1"
    
    if options:
        data.update(options)
    
    # Submit transcription request
    model_type = "SLAM" if use_slam else "Universal"
    console.print(f"\nSubmitting {model_type} transcription request...")
    session = create_session()
    response = session.post(transcript_url, json=data, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"{model_type} transcription request failed with status {response.status_code}: {response.text}")
    
    transcript_id = response.json()["id"]
    
    # Poll for completion
    console.print(f"\nWaiting for {model_type} transcription to complete...")
    while True:
        response = session.get(f"{transcript_url}/{transcript_id}", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get {model_type} transcript status: {response.text}")
        
        status = response.json()["status"]
        if status == "completed":
            return response.json()
        elif status == "error":
            raise Exception(f"{model_type} transcription failed: {response.json().get('error')}")
        
        time.sleep(3)

def save_transcript(transcript: Dict[str, Any], output_dir: Path) -> None:
    """Save transcript results to files."""
    base_path = output_dir / f"universal-{current_session_id}"
    
    # Save text only if available
    if "text" in transcript and transcript["text"]:
        with open(f"{base_path}.txt", "w") as f:
            f.write(transcript["text"])
        console.print(f"Universal transcript text: {base_path}.txt")
    else:
        console.print("No text content found in transcript")
    
    # Display the full transcript
    if "text" in transcript and transcript["text"]:
        console.print("\nUniversal Transcript:")
        console.print(transcript["text"])
        
        # Run Lemur task and save results
        try:
            console.print("\nGenerating Universal Lemur summary...")
            lemur_result = run_lemur_task(transcript["id"])
            
            # Save Lemur response
            lemur_path = output_dir / f"universal-lemur-{current_session_id}.txt"
            with open(lemur_path, "w") as f:
                f.write(lemur_result["response"])
            console.print(f"Universal Lemur summary: {lemur_path}")
            
            # Display Lemur response
            console.print("\nUniversal Lemur Summary:")
            console.print(lemur_result["response"])
        except Exception as e:
            console.print(f"Error generating Universal Lemur summary: {e}")

def record_audio(output_dir: Optional[Path] = None, device_name: Optional[str] = None, show_partials_flag: bool = False) -> None:
    """Record audio using sounddevice and stream to AssemblyAI."""
    # Ensure output_dir is set to 'recordings' if not provided
    if output_dir is None:
        output_dir = Path("recordings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global stream, ws_app, recorded_frames, current_device_id, show_partials, API_ENDPOINT, current_session_dir
    stop_event.clear()
    recorded_frames = []  # Reset recorded frames
    show_partials = show_partials_flag  # Set the global flag
    current_session_dir = None  # Will be set when session begins
    
    # Get device ID if device name is provided
    current_device_id = None
    if device_name:
        current_device_id = get_input_device_id(device_name)
        if current_device_id is None:
            console.print(f"[red]Error: Input device '{device_name}' not found.[/]")
            console.print("\n[yellow]Available input devices:[/]")
            for device in sd.query_devices():
                if device['max_input_channels'] > 0:
                    console.print(f"- {device['name']}")
            return
        console.print(f"\n[green]Using input device:[/] {device_name}")
    
    try:
        # Get device info and set up streaming parameters
        if current_device_id is not None:
            device_info = sd.query_devices(current_device_id)
            sample_rate = int(device_info['default_samplerate'])
            CONNECTION_PARAMS["sample_rate"] = sample_rate
            API_ENDPOINT = f"{API_ENDPOINT_BASE_URL}?{urlencode(CONNECTION_PARAMS)}"
            console.print(f"[green]Streaming with sample rate:[/] {sample_rate} Hz")
        
        # Initialize WebSocket connection
        console.print("\n[yellow]Initializing WebSocket connection...[/]\n")
        ws_app = websocket.WebSocketApp(
            API_ENDPOINT,
            header={"Authorization": API_KEY},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=ws_app.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        try:
            while ws_thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Ctrl+C received. Stopping...[/]")
            stop_event.set()
            if ws_app and ws_app.sock and ws_app.sock.connected:
                try:
                    ws_app.send(json.dumps({"type": "Terminate"}))
                    time.sleep(5)
                except Exception as e:
                    console.print(f"[red]Error sending termination message: {e}[/]")
            if ws_app:
                ws_app.close()
            ws_thread.join(timeout=2.0)
        except Exception as e:
            console.print(f"\n[red]An unexpected error occurred: {e}[/]")
            stop_event.set()
            if ws_app:
                ws_app.close()
            ws_thread.join(timeout=2.0)
        finally:
            console.print("[yellow]Recording stopped.[/]")
            # Save the WAV file
            if current_session_dir:
                audio_file = save_audio_file(current_session_dir, recorded_frames)
                if audio_file:
                    # Automatically transcribe the saved file
                    console.print("\n[yellow]Starting transcription...[/]")
                    try:
                        # Get regular transcript
                        transcript = transcribe_file(audio_file)
                        save_transcript(transcript, current_session_dir)
                        
                        # Get SLAM transcript
                        slam_transcript = transcribe_file(audio_file, use_slam=True)
                        slam_base_path = current_session_dir / f"slam-{current_session_id}"
                        
                        # Save SLAM transcript files
                        with open(f"{slam_base_path}_full.json", "w") as f:
                            json.dump(slam_transcript, f, indent=2)
                        
                        if "text" in slam_transcript and slam_transcript["text"]:
                            with open(f"{slam_base_path}_text.txt", "w") as f:
                                f.write(slam_transcript["text"])
                            console.print(f"SLAM transcript text: {slam_base_path}_text.txt")
                            
                            # Display SLAM transcript
                            console.print("\nSLAM Transcript:")
                            console.print(slam_transcript["text"])
                            
                            # Run Lemur task on SLAM transcript
                            try:
                                console.print("\nGenerating SLAM Lemur summary...")
                                slam_lemur_result = run_lemur_task(slam_transcript["id"])
                                
                                # Save SLAM Lemur response
                                slam_lemur_path = current_session_dir / f"slam-lemur-{current_session_id}.txt"
                                with open(slam_lemur_path, "w") as f:
                                    f.write(slam_lemur_result["response"])
                                console.print(f"SLAM Lemur summary: {slam_lemur_path}")
                                
                                # Display SLAM Lemur response
                                console.print("\nSLAM Lemur Summary:")
                                console.print(slam_lemur_result["response"])

                                # Generate and save embeddings
                                console.print("\nGenerating embeddings for SLAM transcript...")
                                embeddings = generate_embeddings(slam_transcript["id"])
                                save_embeddings(embeddings, current_session_dir, current_session_id)
                                
                            except Exception as e:
                                console.print(f"Error generating SLAM Lemur summary: {e}")
                            
                    except Exception as e:
                        console.print(f"[red]Error during transcription: {e}[/]")

    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {e}[/]")
    finally:
        stop_event.set()

def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def upload_file(file_path: Path) -> str:
    """Upload a file to AssemblyAI and return the upload URL."""
    console.print(f"\nUploading file: {file_path}")
    
    upload_url = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": API_KEY}
    
    session = create_session()
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                response = session.post(upload_url, headers=headers, data=f, timeout=30)
            
            if response.status_code == 200:
                return response.json()["upload_url"]
            else:
                console.print(f"Upload failed (attempt {attempt + 1}/{max_retries}): {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            console.print(f"Upload error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
        if attempt < max_retries - 1:
            console.print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    raise Exception("Failed to upload file after multiple attempts")

def run_lemur_task(transcript_id: str) -> Dict[str, Any]:
    """Run a Lemur task on the transcript."""
    url = "https://api.assemblyai.com/lemur/v3/generate/task"
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "final_model": "anthropic/claude-3-7-sonnet-20250219",
        "prompt": "summarize this transcript",
        "max_output_size": 3000,
        "temperature": 0,
        "transcript_ids": [transcript_id]
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"Lemur task failed: {response.text}")
    
    return response.json()

def generate_embeddings(transcript_id: str) -> Dict[str, Any]:
    """Generate embeddings for a transcript using a local model."""
    try:
        # Load the local model (using all-MiniLM-L6-v2 which is a good balance of speed and quality)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get the Lemur summary from the file
        if current_session_dir is None:
            raise ValueError("No active session directory found")
            
        summary_path = current_session_dir / f"slam-lemur-{current_session_id}.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"Lemur summary file not found: {summary_path}")
            
        with open(summary_path, 'r') as f:
            text = f.read()
            
        # Split text into chunks (sentences or paragraphs) to handle long texts
        # Simple sentence splitting - you might want to use a more sophisticated approach
        chunks = [chunk.strip() for chunk in text.split('.') if chunk.strip()]
        
        # Generate embeddings for each chunk
        embeddings = model.encode(chunks)
        
        # Create a structured response similar to what the API would return
        result = {
            "transcript_id": transcript_id,
            "model": "all-MiniLM-L6-v2",
            "embeddings": embeddings.tolist(),
            "chunks": chunks,
            "metadata": {
                "embedding_dimension": embeddings.shape[1],
                "num_chunks": len(chunks),
                "generated_at": datetime.now().isoformat(),
                "source": "slam-lemur-summary"
            }
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Local embedding generation failed: {str(e)}")

def save_embeddings(embeddings: Dict[str, Any], output_dir: Path, session_id: str) -> None:
    """Save embeddings to a JSON file."""
    embeddings_path = output_dir / f"embeddings-{session_id}.json"
    with open(embeddings_path, "w") as f:
        json.dump(embeddings, f, indent=2)
    console.print(f"Embeddings saved: {embeddings_path}")

def prompt_device_selection() -> Optional[str]:
    """Prompt the user to select an input device."""
    devices = sd.query_devices()
    input_devices = [(i, device) for i, device in enumerate(devices) if device['max_input_channels'] > 0]
    
    if not input_devices:
        console.print("No input devices found.")
        return None
    
    console.print("\nAvailable Input Devices:")
    for idx, (i, device) in enumerate(input_devices, 1):
        console.print(f"{idx}. {device['name']}")
        console.print(f"    Channels: {device['max_input_channels']}")
        console.print(f"    Sample Rate: {device['default_samplerate']} Hz")
        console.print()
    
    while True:
        try:
            choice = typer.prompt("Select a device number", type=int, default=1)
            if 1 <= choice <= len(input_devices):
                return input_devices[choice - 1][1]['name']  # Return the device name
            console.print(f"Please enter a number between 1 and {len(input_devices)}")
        except ValueError:
            console.print("Please enter a valid number")

def load_embeddings(recordings_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load all embedding files from the recordings directory."""
    embeddings = []
    for session_dir in recordings_dir.iterdir():
        if not session_dir.is_dir():
            continue
            
        # Look for embedding files in the session directory
        for file in session_dir.glob("embeddings-*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                embeddings.append((file, data))
            except Exception as e:
                console.print(f"Error loading embeddings from {file}: {e}")
    
    return embeddings

def search_embeddings(query: str, embeddings: List[Tuple[Path, Dict[str, Any]]], model: SentenceTransformer, top_k: int = 5) -> List[Tuple[float, str, Path]]:
    """Search through embeddings using semantic similarity."""
    # Generate query embedding
    query_embedding = model.encode(query)
    
    results = []
    for file_path, data in embeddings:
        # Get embeddings and chunks from the file
        doc_embeddings = np.array(data['embeddings'])
        chunks = data['chunks']
        
        # Calculate cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top matches for this document
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Convert embeddings file path to SLAM Lemur summary path
        session_dir = file_path.parent
        # Get the full session ID from the embeddings filename
        session_id = '-'.join(file_path.stem.split('-')[1:])  # Join all parts after 'embeddings-'
        summary_path = session_dir / f"slam-lemur-{session_id}.txt"
        
        for idx in top_indices:
            results.append((
                float(similarities[idx]),
                chunks[idx],
                summary_path
            ))
    
    # Sort all results by similarity
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

app = typer.Typer(help="Audio recording and transcription tool.")

@app.command()
def record(
    output_dir: Path = typer.Option(
        "recordings",
        "--output-dir", "-o",
        help="Directory to save recordings"
    )
) -> None:
    """Record audio using sounddevice."""
    try:
        # Prompt for device selection
        device = prompt_device_selection()
        if device is None:
            console.print("No device selected. Exiting.")
            sys.exit(1)
                    
        record_audio(output_dir, device, show_partials)
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@app.command()
def transcribe(
    file_path: Path = typer.Argument(..., help="Path to the audio file to transcribe"),
    output_dir: Path = typer.Option(
        "transcripts",
        "--output-dir", "-o",
        help="Directory to save transcriptions"
    )
) -> None:
    """Transcribe an audio file using AssemblyAI."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe the file
        transcript = transcribe_file(file_path)
        
        # Save the results
        save_transcript(transcript, output_dir)
        
        # Display summary if available
        if "summary" in transcript:
            console.print("\nSummary:")
            console.print(transcript["summary"])
        
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@app.command()
def devices() -> None:
    """List all available audio input devices."""
    try:
        list_devices()
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@app.command()
def search(
    recordings_dir: Path = typer.Option(
        "recordings",
        "--recordings-dir", "-r",
        help="Directory containing recordings and embeddings"
    ),
    top_k: int = typer.Option(
        5,
        "--top-k", "-k",
        help="Number of top results to return"
    )
) -> None:
    """Interactive search through all embeddings in the recordings directory."""
    try:
        # Load the embedding model
        console.print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load all embeddings
        console.print(f"Loading embeddings from {recordings_dir}...")
        embeddings = load_embeddings(recordings_dir)
        
        if not embeddings:
            console.print("No embeddings found in the recordings directory.")
            return
            
        console.print(f"Found {len(embeddings)} embedding files.")
        console.print("\nInteractive search mode. Type 'exit' or 'quit' to end the session.")
        console.print("Type 'help' for available commands.")
        
        while True:
            # Get search query from user
            query = typer.prompt("\nEnter search query")
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit']:
                console.print("Exiting search session.")
                break
                
            # Check for help command
            if query.lower() == 'help':
                console.print("\nAvailable commands:")
                console.print("  exit, quit - End the search session")
                console.print("  help - Show this help message")
                console.print("  clear - Clear the screen")
                console.print("\nOr enter any text to search through your recordings.")
                continue
                
            # Check for clear command
            if query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # Perform search
            console.print(f"\nSearching for: {query}")
            results = search_embeddings(query, embeddings, model, top_k)
            
            if not results:
                console.print("No results found.")
                continue
                
            # Display results
            console.print("\nTop matches:")
            for i, (score, text, file_path) in enumerate(results, 1):
                console.print(f"\n{i}. Similarity: {score:.3f}")
                console.print(f"   Summary: {file_path}")
                console.print(f"   Text: {text}")
            
    except Exception as e:
        console.print(f"Error during search: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app() 