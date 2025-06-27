# main.py - Enhanced FastAPI + WebSocket Voice Agent with Deepgram SDK
import asyncio
import json
import time
import uuid
import logging
import base64
import os
import threading
import struct
import wave
import io
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
from asyncio import Queue

from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    AgentWebSocketClient,
    AgentWebSocketEvents,
    AgentKeepAlive,
)
from deepgram.clients.agent.v1.websocket.options import SettingsOptions

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY environment variable is not set")

# OpenAI API key not needed when using Deepgram's managed LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY environment variable is not set")

# Audio configuration constants
DEEPGRAM_SAMPLE_RATE = 24000  # Deepgram Agent API requires 24kHz
DEEPGRAM_CHANNELS = 1  # Mono
DEEPGRAM_SAMPLE_WIDTH = 2  # 16-bit
CHUNK_DURATION_MS = 100  # 100ms chunks for low latency
CHUNK_SIZE = int(DEEPGRAM_SAMPLE_RATE * DEEPGRAM_CHANNELS * DEEPGRAM_SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000)

# Pydantic models for type safety
class SessionStartRequest(BaseModel):
    agent_id: str
    instructions: Optional[str] = None
    voice_config: Optional[Dict[str, Any]] = None
    audio_config: Optional[Dict[str, Any]] = None  # New field for audio settings

class AudioDataMessage(BaseModel):
    audio: str  # base64 encoded audio data
    format: str = "wav"  # React Native will send WAV
    sample_rate: Optional[int] = None  # Will be detected from WAV header
    channels: Optional[int] = None

@dataclass
class SessionMetrics:
    start_time: float
    audio_bytes_sent: int = 0
    audio_bytes_received: int = 0
    audio_chunks_processed: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    format_conversions: int = 0

class AudioProcessor:
    """Handles audio format conversion and validation"""
    
    @staticmethod
    def extract_wav_info(wav_data: bytes) -> tuple[int, int, int]:
        """Extract sample rate, channels, and sample width from WAV header"""
        try:
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    return sample_rate, channels, sample_width
        except Exception as e:
            logger.error(f"Failed to extract WAV info: {e}")
            return 0, 0, 0
    
    @staticmethod
    def extract_pcm_from_wav(wav_data: bytes) -> bytes:
        """Extract raw PCM data from WAV file"""
        try:
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    return wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            logger.error(f"Failed to extract PCM from WAV: {e}")
            return b''
    
    @staticmethod
    def resample_audio(pcm_data: bytes, from_rate: int, to_rate: int, 
                      channels: int = 1, sample_width: int = 2) -> bytes:
        """Simple audio resampling (linear interpolation)"""
        if from_rate == to_rate:
            return pcm_data
        
        try:
            # Convert bytes to samples
            if sample_width == 2:
                samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
            else:
                logger.warning(f"Unsupported sample width: {sample_width}")
                return pcm_data
            
            # Calculate resampling ratio
            ratio = to_rate / from_rate
            output_length = int(len(samples) * ratio)
            
            # Simple linear interpolation resampling
            resampled = []
            for i in range(output_length):
                source_index = i / ratio
                left_index = int(source_index)
                right_index = min(left_index + 1, len(samples) - 1)
                
                if left_index >= len(samples):
                    break
                    
                # Linear interpolation
                fraction = source_index - left_index
                interpolated = samples[left_index] * (1 - fraction) + samples[right_index] * fraction
                resampled.append(int(interpolated))
            
            # Convert back to bytes
            return struct.pack(f'<{len(resampled)}h', *resampled)
            
        except Exception as e:
            logger.error(f"Failed to resample audio: {e}")
            return pcm_data
    
    @staticmethod
    def convert_to_mono(pcm_data: bytes, channels: int, sample_width: int = 2) -> bytes:
        """Convert stereo/multi-channel audio to mono"""
        if channels == 1:
            return pcm_data
        
        try:
            if sample_width == 2:
                samples = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
                # Average channels to create mono
                mono_samples = []
                for i in range(0, len(samples), channels):
                    if i + channels <= len(samples):
                        avg = sum(samples[i:i+channels]) // channels
                        mono_samples.append(avg)
                
                return struct.pack(f'<{len(mono_samples)}h', *mono_samples)
            else:
                logger.warning(f"Unsupported sample width for mono conversion: {sample_width}")
                return pcm_data
                
        except Exception as e:
            logger.error(f"Failed to convert to mono: {e}")
            return pcm_data

class AgentSession:
    def __init__(self, user_id: str, agent_id: str, instructions: Optional[str] = None):
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())
        self.instructions = instructions or "You are a helpful AI assistant."
        
        # Deepgram SDK connection
        self.deepgram_client: Optional[DeepgramClient] = None
        self.connection: Optional[AgentWebSocketClient] = None
        self.client_ws: Optional[WebSocket] = None
        
        # State management
        self.is_connected = False
        self.is_listening = False
        self.metrics = SessionMetrics(start_time=time.time())
        
        # Threading support
        self._deepgram_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connection_ready_event = threading.Event()
        
        # Audio processing
        self.audio_processor = AudioProcessor()
        self.audio_queue: Queue = Queue()
        self._tasks: List[asyncio.Task] = []
        
        # Audio state tracking
        self.last_audio_time = time.time()
        self.audio_format_detected = False
        self.client_sample_rate = None
        self.client_channels = None
        
        # Thread-safe message queue for event handlers
        self._client_message_queue = Queue()
        
        logger.info(f"Created session {self.session_id} for user {user_id}")

    async def connect_to_deepgram(self) -> bool:
        """Establish connection to Deepgram Voice Agent API using SDK"""
        try:
            logger.info(f"Starting Deepgram connection for session {self.session_id}")
            
            # Initialize Deepgram client
            config = DeepgramClientOptions(
                options={
                    "keepalive": "true",
                },
            )
            
            self.deepgram_client = DeepgramClient(DEEPGRAM_API_KEY, config)
            self.connection = self.deepgram_client.agent.websocket.v("1")
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Configure agent settings
            options = self._get_agent_config()
            
            # Start the connection in a separate thread
            logger.info("Starting Deepgram connection thread...")
            self._deepgram_thread = threading.Thread(
                target=self._run_deepgram_connection,
                args=(options,),
                daemon=True
            )
            self._deepgram_thread.start()
            
            # Wait for connection to establish (with timeout)
            if self._connection_ready_event.wait(timeout=10):
                if self.is_connected:
                    logger.info(f"✅ Connected to Deepgram for session {self.session_id}")
                    return True
                else:
                    logger.error(f"❌ Failed to establish Deepgram connection for session {self.session_id}")
                    return False
            else:
                logger.error(f"❌ Connection timeout for session {self.session_id}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.is_connected = False
            return False

    def _run_deepgram_connection(self, options: SettingsOptions):
        """Run Deepgram connection in a separate thread"""
        try:
            logger.info(f"Starting Deepgram connection thread for session {self.session_id}")
            
            if self.connection.start(options):
                self.is_connected = True
                logger.info(f"✅ Deepgram connection started successfully for session {self.session_id}")
                self._connection_ready_event.set()
                
                # Start keep-alive in this thread
                self._keep_alive_loop()
            else:
                logger.error(f"❌ Failed to start Deepgram connection for session {self.session_id}")
                self.is_connected = False
                self._connection_ready_event.set()
                
        except Exception as e:
            logger.error(f"Error in Deepgram connection thread: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.is_connected = False
            self._connection_ready_event.set()

    def _keep_alive_loop(self):
        """Send keep-alive messages in the Deepgram thread"""
        while self.is_connected and not self._stop_event.is_set():
            try:
                time.sleep(5)
                if self.connection and self.is_connected:
                    self.connection.send(str(AgentKeepAlive()))
                    logger.debug(f"📡 Sent keep-alive for session {self.session_id}")
            except Exception as e:
                logger.error(f"Error sending keep-alive: {e}")
                break

    def _get_agent_config(self) -> SettingsOptions:
        """Generate Deepgram agent configuration optimized for React Native"""
        options = SettingsOptions()
        
        # Audio input configuration - optimized for mobile
        options.audio.input.encoding = "linear16"
        options.audio.input.sample_rate = DEEPGRAM_SAMPLE_RATE  # 24kHz required
        
        # Audio output configuration - optimized for mobile playback
        options.audio.output.encoding = "linear16"
        options.audio.output.sample_rate = DEEPGRAM_SAMPLE_RATE  # 24kHz for consistency
        options.audio.output.container = "wav"
        
        # Agent configuration - optimized for conversation
        options.agent.language = "en"
        options.agent.listen.provider.type = "deepgram"
        options.agent.listen.provider.model = "nova-3"
        
        # LLM configuration - use Deepgram's managed LLM instead of OpenAI
        # options.agent.think.provider.type = "openai"
        # options.agent.think.provider.model = "gpt-4o-mini"
        options.agent.think.provider.type = "deepgram"
        options.agent.think.prompt = self.instructions
        
        # TTS configuration - optimized for mobile
        options.agent.speak.provider.type = "deepgram"
        options.agent.speak.provider.model = "aura-2-thalia-en"
        
        # Conversation settings
        options.agent.greeting = "Hello! I'm ready to chat. How can I help you today?"
        
        logger.info(f"🔧 Agent configured for session {self.session_id}")
        return options

    def _setup_event_handlers(self):
        """Set up event handlers for Deepgram Agent WebSocket"""
        
        def on_welcome(*args, **kwargs):
            """Handle welcome event from Deepgram"""
            logger.info(f"🎉 Welcome event received for session {self.session_id}")
            self._connection_ready_event.set()  # Signal connection is ready
            # Queue message to be sent from main thread
            self._queue_client_message({
                "type": "agent_connected",
                "session_id": self.session_id,
                "message": "Voice agent is ready"
            })

        def on_settings_applied(*args, **kwargs):
            """Handle settings applied event"""
            logger.info(f"⚙️ Settings applied for session {self.session_id}")
            self._queue_client_message({
                "type": "settings_applied",
                "session_id": self.session_id
            })

        def on_conversation_text(*args, **kwargs):
            """Handle conversation text events"""
            if len(args) > 1:
                conversation_data = args[1]
                logger.info(f"💬 Conversation text: {conversation_data}")
                self._queue_client_message({
                    "type": "conversation_text",
                    "data": conversation_data.__dict__ if hasattr(conversation_data, '__dict__') else str(conversation_data)
                })

        def on_user_started_speaking(*args, **kwargs):
            """Handle user started speaking event"""
            logger.info(f"🎤 User started speaking in session {self.session_id}")
            self._queue_client_message({
                "type": "user_started_speaking",
                "session_id": self.session_id
            })

        def on_agent_thinking(*args, **kwargs):
            """Handle agent thinking event"""
            logger.info(f"🤔 Agent thinking in session {self.session_id}")
            self._queue_client_message({
                "type": "agent_thinking",
                "session_id": self.session_id
            })

        def on_agent_started_speaking(*args, **kwargs):
            """Handle agent started speaking event"""
            logger.info(f"🗣️ Agent started speaking in session {self.session_id}")
            # Queue message (audio queue clearing will happen in main thread)
            self._queue_client_message({
                "type": "agent_started_speaking",
                "session_id": self.session_id,
                "clear_audio_queue": True  # Signal to clear queue
            })

        def on_agent_audio_done(*args, **kwargs):
            """Handle agent audio done event"""
            logger.info(f"✅ Agent audio done for session {self.session_id}")
            self._queue_client_message({
                "type": "agent_audio_done",
                "session_id": self.session_id
            })

        def on_agent_audio_data(*args, **kwargs):
            """Handle audio data from agent"""
            if len(args) > 1:
                audio_data = args[1]
                if hasattr(audio_data, 'data') and audio_data.data:
                    audio_bytes = audio_data.data
                    self.metrics.audio_bytes_received += len(audio_bytes)
                    
                    # Convert audio to base64 for transmission to React Native
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    logger.debug(f"🔊 Received {len(audio_bytes)} bytes of audio from agent")
                    self._queue_client_message({
                        "type": "agent_audio_data",
                        "audio": audio_b64,
                        "format": "wav",
                        "sample_rate": DEEPGRAM_SAMPLE_RATE,
                        "channels": DEEPGRAM_CHANNELS
                    })

        def on_error(*args, **kwargs):
            """Handle error events"""
            error_msg = str(args[1]) if len(args) > 1 else "Unknown error"
            logger.error(f"❌ Deepgram error in session {self.session_id}: {error_msg}")
            self._queue_client_message({
                "type": "agent_error",
                "error": error_msg,
                "session_id": self.session_id
            })

        def on_close(*args, **kwargs):
            """Handle connection close events"""
            close_info = str(args[1]) if len(args) > 1 else "Connection closed"
            logger.info(f"🔌 Deepgram connection closed for session {self.session_id}: {close_info}")
            self.is_connected = False
            self._queue_client_message({
                "type": "agent_disconnected",
                "reason": close_info,
                "session_id": self.session_id
            })

        def on_unhandled(*args, **kwargs):
            """Handle unhandled events"""
            event_info = str(args[1]) if len(args) > 1 else "Unknown event"
            logger.debug(f"🔍 Unhandled event in session {self.session_id}: {event_info}")

        # Register all event handlers
        self.connection.on(AgentWebSocketEvents.Welcome, on_welcome)
        self.connection.on(AgentWebSocketEvents.SettingsApplied, on_settings_applied)
        self.connection.on(AgentWebSocketEvents.ConversationText, on_conversation_text)
        self.connection.on(AgentWebSocketEvents.UserStartedSpeaking, on_user_started_speaking)
        self.connection.on(AgentWebSocketEvents.AgentThinking, on_agent_thinking)
        self.connection.on(AgentWebSocketEvents.AgentStartedSpeaking, on_agent_started_speaking)
        self.connection.on(AgentWebSocketEvents.AgentAudioDone, on_agent_audio_done)
        self.connection.on(AgentWebSocketEvents.AudioData, on_agent_audio_data)
        self.connection.on(AgentWebSocketEvents.Error, on_error)
        self.connection.on(AgentWebSocketEvents.Close, on_close)
        self.connection.on(AgentWebSocketEvents.Unhandled, on_unhandled)
        
        logger.info(f"🔗 Event handlers registered for session {self.session_id}")

    async def send_audio(self, audio_data: bytes, format: str = "wav"):
        """Enhanced audio processing and sending to Deepgram with detailed logging"""
        if not self.is_connected or not self.connection:
            logger.warning(f"❌ Cannot send audio - session {self.session_id} not connected (is_connected: {self.is_connected}, connection: {self.connection is not None})")
            return

        try:
            self.last_audio_time = time.time()
            processed_audio = audio_data
            logger.info(f"🎵 Processing {len(audio_data)} bytes of {format} audio for session {self.session_id}")
            
            # Process WAV format from React Native
            if format.lower() == "wav":
                logger.debug(f"🔍 Extracting WAV header information...")
                # Extract audio information from WAV header
                sample_rate, channels, sample_width = self.audio_processor.extract_wav_info(audio_data)
                
                if sample_rate > 0:  # Valid WAV file
                    # Track client audio format
                    if not self.audio_format_detected:
                        self.client_sample_rate = sample_rate
                        self.client_channels = channels
                        self.audio_format_detected = True
                        logger.info(f"🎵 Detected audio format: {sample_rate}Hz, {channels}ch, {sample_width*8}bit")
                    
                    logger.debug(f"📊 Audio specs: {sample_rate}Hz, {channels}ch, {sample_width*8}bit")
                    
                    # Extract PCM data from WAV
                    logger.debug(f"🔄 Extracting PCM data from WAV...")
                    pcm_data = self.audio_processor.extract_pcm_from_wav(audio_data)
                    if not pcm_data:
                        logger.warning(f"❌ No PCM data extracted from WAV")
                        return
                    
                    logger.debug(f"✅ Extracted {len(pcm_data)} bytes of PCM data")
                    
                    # Convert to mono if needed
                    if channels > 1:
                        logger.debug(f"🔄 Converting {channels}ch to mono...")
                        pcm_data = self.audio_processor.convert_to_mono(pcm_data, channels, sample_width)
                        logger.debug(f"✅ Converted {channels}ch to mono ({len(pcm_data)} bytes)")
                    
                    # Resample to 24kHz if needed
                    if sample_rate != DEEPGRAM_SAMPLE_RATE:
                        logger.debug(f"🔄 Resampling {sample_rate}Hz → {DEEPGRAM_SAMPLE_RATE}Hz...")
                        pcm_data = self.audio_processor.resample_audio(
                            pcm_data, sample_rate, DEEPGRAM_SAMPLE_RATE, 1, sample_width
                        )
                        self.metrics.format_conversions += 1
                        logger.debug(f"✅ Resampled to {DEEPGRAM_SAMPLE_RATE}Hz ({len(pcm_data)} bytes)")
                    
                    processed_audio = pcm_data
                else:
                    logger.error(f"❌ Invalid WAV format received - cannot extract audio info")
                    return
            
            # Send processed audio to Deepgram
            if processed_audio:
                logger.info(f"🚀 Streaming {len(processed_audio)} bytes to Deepgram Agent API...")
                self.connection.send(processed_audio)
                self.metrics.audio_bytes_sent += len(processed_audio)
                self.metrics.audio_chunks_processed += 1
                
                logger.info(f"✅ Successfully streamed audio to Deepgram (session {self.session_id})")
                logger.debug(f"📊 Total audio sent: {self.metrics.audio_bytes_sent} bytes, Chunks: {self.metrics.audio_chunks_processed}")
            else:
                logger.warning(f"⚠️ No processed audio to send to Deepgram")
            
        except Exception as e:
            logger.error(f"❌ Error processing/sending audio for session {self.session_id}: {e}")
            import traceback
            logger.error(f"💥 Audio processing error traceback: {traceback.format_exc()}")

    async def send_audio_chunk(self, audio_chunk: bytes, format: str = "wav"):
        """Send individual audio chunk (optimized for streaming)"""
        await self.send_audio(audio_chunk, format)

    async def update_instructions(self, instructions: str):
        """Update agent instructions"""
        if self.connection:
            try:
                # Update the prompt
                update_message = {
                    "type": "UpdatePrompt",
                    "prompt": instructions
                }
                self.connection.send(json.dumps(update_message))
                self.instructions = instructions
                logger.info(f"📝 Updated instructions for session {self.session_id}")
            except Exception as e:
                logger.error(f"Error updating instructions: {e}")

    def _queue_client_message(self, message: Dict[str, Any]):
        """Queue message to be sent to client (thread-safe)"""
        try:
            self._client_message_queue.put_nowait(message)
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")

    async def _send_to_client(self, message: Dict[str, Any]):
        """Send message to client WebSocket"""
        if self.client_ws:
            try:
                await self.client_ws.send_text(json.dumps(message))
                self.metrics.messages_sent += 1
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
    
    async def _process_queued_messages(self):
        """Process queued messages from event handlers (called from main thread)"""
        while not self._client_message_queue.empty():
            try:
                message = self._client_message_queue.get_nowait()
                
                # Handle special actions
                if message.get("clear_audio_queue"):
                    await self._clear_audio_queue()
                    # Remove the flag before sending
                    message = {k: v for k, v in message.items() if k != "clear_audio_queue"}
                
                await self._send_to_client(message)
            except Exception as e:
                logger.error(f"Failed to process queued message: {e}")
                break

    async def _clear_audio_queue(self):
        """Clear pending audio queue (for barge-in)"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def get_session_stats(self) -> Dict[str, Any]:
        """Get detailed session statistics"""
        duration = time.time() - self.metrics.start_time
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "duration_seconds": round(duration, 2),
            "is_connected": self.is_connected,
            "audio_bytes_sent": self.metrics.audio_bytes_sent,
            "audio_bytes_received": self.metrics.audio_bytes_received,
            "audio_chunks_processed": self.metrics.audio_chunks_processed,
            "format_conversions": self.metrics.format_conversions,
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "client_audio_format": {
                "sample_rate": self.client_sample_rate,
                "channels": self.client_channels,
                "detected": self.audio_format_detected
            },
            "last_audio_time": self.last_audio_time
        }

    async def cleanup(self):
        """Clean up session resources"""
        logger.info(f"🧹 Cleaning up session {self.session_id}")
        
        # Signal thread to stop
        self._stop_event.set()
        self.is_connected = False
        
        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close Deepgram connection
        if self.connection:
            try:
                self.connection.finish()
            except Exception as e:
                logger.error(f"Error finishing Deepgram connection: {e}")
        
        # Wait for thread to finish
        if self._deepgram_thread and self._deepgram_thread.is_alive():
            self._deepgram_thread.join(timeout=5)
            
        # Log session metrics
        stats = self.get_session_stats()
        logger.info(f"📊 Session {self.session_id} final stats: "
                   f"Duration: {stats['duration_seconds']}s, "
                   f"Audio sent: {stats['audio_bytes_sent']} bytes, "
                   f"Audio received: {stats['audio_bytes_received']} bytes, "
                   f"Chunks processed: {stats['audio_chunks_processed']}, "
                   f"Format conversions: {stats['format_conversions']}")

class VoiceAgentBackend:
    def __init__(self):
        self.active_sessions: Dict[str, AgentSession] = {}
        
    async def create_session(self, user_id: str, agent_id: str, 
                           instructions: Optional[str] = None) -> tuple[str, bool]:
        """Create a new voice agent session"""
        session = AgentSession(user_id, agent_id, instructions)
        
        # Connect to Deepgram
        success = await session.connect_to_deepgram()
        if not success:
            await session.cleanup()
            return "", False
            
        self.active_sessions[session.session_id] = session
        logger.info(f"✅ Created session {session.session_id} for user {user_id}")
        return session.session_id, True
        
    async def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get active session by ID"""
        return self.active_sessions.get(session_id)
        
    async def cleanup_session(self, session_id: str):
        """Remove and cleanup session"""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            await session.cleanup()
            logger.info(f"🗑️ Cleaned up session {session_id}")
            
    async def cleanup_all_sessions(self):
        """Cleanup all active sessions"""
        tasks = []
        for session_id in list(self.active_sessions.keys()):
            tasks.append(self.cleanup_session(session_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        return {
            "active_sessions": len(self.active_sessions),
            "session_ids": list(self.active_sessions.keys()),
            "deepgram_sample_rate": DEEPGRAM_SAMPLE_RATE,
            "chunk_size": CHUNK_SIZE,
            "chunk_duration_ms": CHUNK_DURATION_MS
        }

# Global backend instance
voice_backend = VoiceAgentBackend()

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Voice Agent Backend starting up...")
    yield
    # Shutdown
    logger.info("Voice Agent Backend shutting down...")
    await voice_backend.cleanup_all_sessions()

app = FastAPI(title="Voice Agent Backend", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    backend_stats = voice_backend.get_backend_stats()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        **backend_stats
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, session in voice_backend.active_sessions.items():
        sessions.append(session.get_session_stats())
    return {
        "active_sessions": len(sessions),
        "sessions": sessions
    }

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get detailed information about a specific session"""
    session = await voice_backend.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.get_session_stats()


@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for direct PCM audio streaming from React Native frontend"""
    await websocket.accept()
    session: Optional[AgentSession] = None
    session_id = None
    
    # Create logger for this endpoint to match your log format
    audio_logger = logging.getLogger("app.routes.websocket_audio")
    
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        audio_logger.info(f"🔗 WebSocket connected: session {session_id}")
        
        # Send connected message to frontend
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "timestamp": time.time()
        }))
        
        # Create default session with character (adina)
        character = "adina"
        instructions = f"You are {character}, a helpful AI assistant."
        
        # Create session using existing backend with detailed logging
        audio_logger.info(f"🔧 Creating session for character {character}...")
        session_created_id, success = await voice_backend.create_session(
            session_id,  # Use session_id as user_id
            character,
            instructions
        )
        
        if success:
            session = voice_backend.active_sessions[session_created_id]
            session.client_ws = websocket
            audio_logger.info(f"🎯 Session created for character {character}, connected: {success}")
            audio_logger.info(f"✅ Session {session_created_id} is ready for audio processing")
            
            await websocket.send_text(json.dumps({
                "type": "session_ready",
                "character": character,
                "session_id": session_id
            }))
        else:
            audio_logger.error(f"❌ CRITICAL: Failed to create session {session_id}")
            audio_logger.error(f"💥 This means Deepgram connection failed - check API key and network")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Failed to connect to Deepgram - check server logs"
            }))
        
        # Main message loop
        while True:
            try:
                # Receive message from frontend
                message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
                
                if message["type"] == "websocket.receive":
                    if "text" in message:
                        # Handle text messages (character selection, etc.)
                        text_message = message["text"]
                        audio_logger.info(f"📨 Received text message: {text_message}")
                        
                        # Try to parse as JSON first
                        try:
                            data = json.loads(text_message)
                            message_type = data.get("type")
                            audio_logger.info(f"🔍 Parsed JSON message - type: {message_type}, data: {data}")
                            
                            if message_type == "character_selected":
                                # Handle proper JSON character selection
                                new_character = data.get("character", "adina")
                            elif message_type == "message":
                                # Handle frontend's "message" type (the missing piece!)
                                new_character = data.get("content", data.get("data", data.get("message", "adina")))
                                audio_logger.info(f"🎭 Frontend sent message type, character: {new_character}")
                            else:
                                # Handle unknown JSON message type
                                audio_logger.warning(f"⚠️ Unknown JSON message type: {message_type}")
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": f"Unknown message type: {message_type}"
                                }))
                                continue
                                
                        except json.JSONDecodeError:
                            # Not JSON - treat as plain character name (frontend compatibility)
                            new_character = text_message.strip()
                            audio_logger.info(f"🎭 Treating plain text as character selection: {new_character}")
                        
                        # Create session with selected character
                        new_instructions = f"You are {new_character}, a helpful AI assistant."
                        
                        # Cleanup existing session
                        if session:
                            await voice_backend.cleanup_session(session.session_id)
                        
                        # Create new session with selected character
                        session_created_id, success = await voice_backend.create_session(
                            session_id,
                            new_character,
                            new_instructions
                        )
                        
                        if success:
                            session = voice_backend.active_sessions[session_created_id]
                            session.client_ws = websocket
                            audio_logger.info(f"🎯 Session created for character {new_character}")
                            
                            await websocket.send_text(json.dumps({
                                "type": "session_ready",
                                "character": new_character,
                                "session_id": session_id
                            }))
                        else:
                            audio_logger.error(f"❌ Failed to create session for character {new_character}")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Failed to connect to Deepgram - check server logs"
                            }))
                    
                    elif "bytes" in message and session:
                        # Handle binary PCM audio data
                        pcm_audio = message["bytes"]
                        
                        if len(pcm_audio) > 0:
                            audio_logger.info(f"📥 Received {len(pcm_audio)} bytes of PCM audio from frontend")
                            
                            # Check if session is connected to Deepgram
                            if not session.is_connected:
                                audio_logger.warning(f"⚠️ Session {session_id} not connected to Deepgram, attempting reconnection...")
                                # Try to reconnect
                                success = await session.connect_to_deepgram()
                                if success:
                                    audio_logger.info(f"✅ Successfully reconnected session {session_id} to Deepgram")
                                else:
                                    audio_logger.error(f"❌ Failed to reconnect session {session_id} to Deepgram")
                                    continue
                            
                            # Send PCM audio to the session with enhanced logging
                            try:
                                audio_logger.info(f"🎤 Processing audio through conversational AI pipeline...")
                                await session.send_audio(pcm_audio, "wav")
                                audio_logger.info(f"📤 Successfully sent {len(pcm_audio)} bytes to Deepgram for session {session_id}")
                                
                                # Log session stats periodically
                                stats = session.get_session_stats()
                                audio_logger.debug(f"📊 Session stats - Audio sent: {stats['audio_bytes_sent']} bytes, Chunks: {stats['audio_chunks_processed']}")
                                
                            except Exception as e:
                                audio_logger.error(f"❌ Failed to process audio for session {session_id}: {e}")
                                # Send error back to frontend
                                await websocket.send_text(json.dumps({
                                    "type": "audio_error",
                                    "error": str(e),
                                    "session_id": session_id
                                }))
                        else:
                            audio_logger.warning(f"⚠️ Received empty audio data from frontend")
                            
                    elif "bytes" in message and not session:
                        # Handle case where binary data arrives without session
                        audio_logger.error(f"❌ Received {len(message['bytes'])} bytes of audio but NO ACTIVE SESSION")
                        audio_logger.error(f"💡 This means character selection failed - check above logs")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "No active session for binary audio data - character selection may have failed"
                        }))
                
            except asyncio.TimeoutError:
                # No message received, process any queued messages from event handlers
                if session:
                    await session._process_queued_messages()
                continue
                
    except WebSocketDisconnect:
        audio_logger.info(f"🔌 WebSocket disconnected: session {session_id}")
    except Exception as e:
        audio_logger.error(f"❌ WebSocket error for session {session_id}: {e}")
    finally:
        # Clean up session
        if session and session.session_id in voice_backend.active_sessions:
            try:
                await voice_backend.cleanup_session(session.session_id)
                audio_logger.info(f"🧹 Cleaned up session {session_id}")
            except Exception as e:
                audio_logger.error(f"Error cleaning up session {session_id}: {e}")

async def handle_text_message(websocket: WebSocket, session: Optional[AgentSession], 
                             message_text: str, user_id: str) -> Optional[AgentSession]:
    """Handle text-based control messages"""
    try:
        data = json.loads(message_text)
        message_type = data.get("type")
        
        if message_type == "start_session":
            # Start new session
            request = SessionStartRequest(**data)
            session_id, success = await voice_backend.create_session(
                user_id, 
                request.agent_id, 
                request.instructions
            )
            
            if success:
                session = voice_backend.active_sessions[session_id]
                session.client_ws = websocket
                
                await websocket.send_text(json.dumps({
                    "type": "session_started",
                    "session_id": session_id,
                    "agent_id": request.agent_id,
                    "audio_config": {
                        "required_sample_rate": DEEPGRAM_SAMPLE_RATE,
                        "required_channels": DEEPGRAM_CHANNELS,
                        "chunk_duration_ms": CHUNK_DURATION_MS,
                        "supported_formats": ["wav"]
                    }
                }))
                logger.info(f"✅ Started session {session_id} for user {user_id}")
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to connect to Deepgram"
                }))
                logger.error(f"❌ Failed to start session for user {user_id}")
                
        elif message_type == "audio_data":
            # Handle base64 encoded audio data (fallback method)
            if session and session.is_connected:
                try:
                    audio_data = base64.b64decode(data["audio"])
                    audio_format = data.get("format", "wav")
                    await session.send_audio(audio_data, audio_format)
                    logger.debug(f"📤 Processed base64 audio data for session {session.session_id}")
                except Exception as e:
                    logger.error(f"Error processing base64 audio: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}"
                    }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No active session or session not connected"
                }))
                
        elif message_type == "update_instructions":
            # Update agent instructions
            if session and session.is_connected:
                await session.update_instructions(data["instructions"])
                await websocket.send_text(json.dumps({
                    "type": "instructions_updated",
                    "session_id": session.session_id
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "No active session"
                }))
                
        elif message_type == "get_session_stats":
            # Get session statistics
            if session:
                stats = session.get_session_stats()
                await websocket.send_text(json.dumps({
                    "type": "session_stats",
                    "stats": stats
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No active session"
                }))
                
        elif message_type == "end_session":
            # End current session
            if session:
                session_id = session.session_id
                await voice_backend.cleanup_session(session_id)
                session = None
                await websocket.send_text(json.dumps({
                    "type": "session_ended",
                    "session_id": session_id
                }))
                logger.info(f"🔚 Ended session {session_id} for user {user_id}")
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No active session to end"
                }))
        
        elif message_type == "ping":
            # Handle ping for connection testing
            await websocket.send_text(json.dumps({
                "type": "pong",
                "timestamp": time.time()
            }))
            
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }))
            
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Invalid JSON message"
        }))
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Internal server error"
        }))
    
    return session

async def handle_binary_message(websocket: WebSocket, session: Optional[AgentSession], 
                               binary_data: bytes):
    """Handle binary audio messages (more efficient for streaming)"""
    if not session or not session.is_connected:
        # Send error as text message
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "No active session for binary audio data"
        }))
        return
    
    try:
        # Assume binary data is WAV format from React Native
        await session.send_audio(binary_data, "wav")
        logger.debug(f"📤 Processed binary audio data ({len(binary_data)} bytes) for session {session.session_id}")
        
    except Exception as e:
        logger.error(f"Error processing binary audio: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Binary audio processing error: {str(e)}"
        }))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    ) 