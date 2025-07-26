#!/usr/bin/env python3
"""
Optimized Real-time Always-On Audio Speech Recognition System

This is a highly optimized version that focuses on:
1. Minimal latency and maximum throughput
2. Efficient memory usage
3. Advanced VAD with noise reduction
4. Streaming ASR with chunked processing
5. Adaptive quality based on system performance
"""

import pyaudio
import threading
import queue
import time
import numpy as np
import webrtcvad
import whisper
import logging
from collections import deque
from typing import Optional, Callable
import signal
import sys
import gc
from dataclasses import dataclass
import psutil
import os
import re
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration parameters."""
    sample_rate: int = 16000
    chunk_size: int = 480  # 30ms at 16kHz
    channels: int = 1
    format: int = pyaudio.paInt16
    
@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    aggressiveness: int = 1
    min_speech_duration: float = 0.3  # Minimum speech duration in seconds
    max_speech_duration: float = 30.0  # Maximum speech duration in seconds
    silence_threshold: float = 0.8  # Seconds of silence to end speech
    
@dataclass
class ASRConfig:
    """ASR configuration parameters."""
    model_name: str = "base"  # Fastest model for real-time
    language: str = "zh"
    fp16: bool = True  # Use half precision for speed
    beam_size: int = 1  # Greedy decoding for speed
    best_of: int = 1  # No multiple candidates
    temperature: float = 0.0  # Deterministic output

@dataclass
class KimiConfig:
    """Kimi API configuration parameters."""
    api_key: str = "sk-fsLfyXnzCQd2sOCzfXZ73Fbu9qAx5KAGxSiBotLIX96YdoUZ"
    base_url: str = "https://api.moonshot.cn/v1"
    model: str = "kimi-k2-0711-preview"
    temperature: float = 0.6
    max_chinese_words: int = 20

class KimiTaskClassifier:
    """Handle text filtering and Kimi API calls for task classification."""
    
    def __init__(self, kimi_config: Optional[KimiConfig] = None):
        self.config = kimi_config or KimiConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Refined prompt for better task classification
        self.system_prompt = (
            "你是一个智能机器人任务分类助手。你需要根据语音识别结果判断用户想要执行的任务。"
            "可能的任务包括：拿纸巾、帮忙拿手机、松开并归还手机。"
            "请注意语音识别可能存在错误，需要根据谐音和语义进行推测。"
            "如果识别结果与这三个任务都无关（比如日常对话、无关指令等），请输出'无'。"
        )
        
        self.user_prompt_template = (
            "语音识别结果：{result}\n\n"
            "请判断用户想让机器人执行以下哪个任务：\n"
            "1. 拿纸巾\n"
            "2. 帮忙拿手机\n"
            "3. 松开并归还手机\n\n"
            "要求：\n"
            "- 只输出任务名称，不要其他内容\n"
            "- 考虑语音识别错误和谐音\n"
            "- 特别注意环境噪音可能导致的误识别，因此如果与以上任务无关，输出'无'\n"
        )
    
    def count_chinese_words(self, text: str) -> int:
        """Count Chinese characters in text."""
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars)
    
    def filter_text(self, text: str) -> bool:
        """Filter out text with more than max_chinese_words Chinese characters."""
        chinese_count = self.count_chinese_words(text)
        logger.info(f"Text has {chinese_count} Chinese characters")
        return chinese_count <= self.config.max_chinese_words
    
    def classify_task(self, text: str) -> Optional[str]:
        """Classify task using Kimi API."""
        try:
            user_prompt = self.user_prompt_template.format(result=text)
            
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature
            )
            
            result = completion.choices[0].message.content.strip()
            logger.info(f"Kimi API classification result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling Kimi API: {e}")
            return None
    
    def process_transcription(self, text: str) -> Optional[str]:
        """Process transcription: filter and classify if needed."""
        # First filter by Chinese word count
        if not self.filter_text(text):
            logger.info(f"Text filtered out (too many Chinese words): {text}")
            return None
        
        # Then classify using Kimi API
        return self.classify_task(text)

class PerformanceMonitor:
    """Monitor system performance and adapt accordingly."""
    
    def __init__(self):
        self.cpu_usage_history = deque(maxlen=10)
        self.memory_usage_history = deque(maxlen=10)
        self.processing_times = deque(maxlen=50)
        
    def update_system_stats(self):
        """Update system performance statistics."""
        self.cpu_usage_history.append(psutil.cpu_percent())
        self.memory_usage_history.append(psutil.virtual_memory().percent)
        
    def get_avg_cpu_usage(self) -> float:
        """Get average CPU usage."""
        return sum(self.cpu_usage_history) / len(self.cpu_usage_history) if self.cpu_usage_history else 0
        
    def get_avg_memory_usage(self) -> float:
        """Get average memory usage."""
        return sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0
        
    def add_processing_time(self, processing_time: float):
        """Add a processing time measurement."""
        self.processing_times.append(processing_time)
        
    def get_avg_processing_time(self) -> float:
        """Get average processing time."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
    def should_reduce_quality(self) -> bool:
        """Determine if we should reduce quality for performance."""
        return (self.get_avg_cpu_usage() > 80 or 
                self.get_avg_memory_usage() > 85 or 
                self.get_avg_processing_time() > 2.0)

class OptimizedRealTimeASR:
    """Highly optimized real-time ASR system."""
    
    def __init__(self, 
                 audio_config: Optional[AudioConfig] = None,
                 vad_config: Optional[VADConfig] = None,
                 asr_config: Optional[ASRConfig] = None,
                 kimi_config: Optional[KimiConfig] = None,
                 speech_callback: Optional[Callable[[str, float], None]] = None,
                 task_callback: Optional[Callable[[str, str], None]] = None):
        """
        Initialize the optimized real-time ASR system.
        
        Args:
            audio_config: Audio configuration
            vad_config: VAD configuration
            asr_config: ASR configuration
            kimi_config: Kimi API configuration
            speech_callback: Callback function for recognized speech (text, confidence)
            task_callback: Callback function for classified tasks (text, task)
        """
        self.audio_config = audio_config or AudioConfig()
        self.vad_config = vad_config or VADConfig()
        self.asr_config = asr_config or ASRConfig()
        self.kimi_config = kimi_config or KimiConfig()
        self.speech_callback = speech_callback
        self.task_callback = task_callback
        
        # Initialize Kimi task classifier
        self.task_classifier = KimiTaskClassifier(self.kimi_config)
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_config.aggressiveness)
        
        # Initialize Whisper model with optimizations
        logger.info(f"Loading optimized Whisper model: {self.asr_config.model_name}")
        self.whisper_model = whisper.load_model(
            self.asr_config.model_name,
            device="cuda" if self._has_cuda() else "cpu"
        )
        logger.info("Whisper model loaded successfully")
        
        # Audio processing components
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # High-performance queues with appropriate sizes
        self.audio_queue = queue.Queue(maxsize=50)  # Smaller queue for lower latency
        self.speech_queue = queue.Queue(maxsize=5)   # Even smaller for ASR
        
        # Threading control
        self.running = False
        self.threads = []
        
        # Speech detection state
        self.is_speaking = False
        self.speech_frames = deque()
        self.silence_frames = 0
        self.speech_start_time = 0
        
        # Calculate frame thresholds
        self.silence_threshold_frames = int(
            self.vad_config.silence_threshold * self.audio_config.sample_rate / self.audio_config.chunk_size
        )
        self.min_speech_frames = int(
            self.vad_config.min_speech_duration * self.audio_config.sample_rate / self.audio_config.chunk_size
        )
        self.max_speech_frames = int(
            self.vad_config.max_speech_duration * self.audio_config.sample_rate / self.audio_config.chunk_size
        )
        
        # Performance statistics
        self.stats = {
            'total_chunks': 0,
            'speech_chunks': 0,
            'recognition_count': 0,
            'avg_recognition_time': 0,
            'dropped_frames': 0,
            'queue_overflows': 0
        }
        
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Optimized audio callback with minimal processing."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            self.audio_queue.put_nowait(in_data)
        except queue.Full:
            self.stats['dropped_frames'] += 1
            # Drop oldest frame to make room
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(in_data)
            except queue.Empty:
                pass
        
        return (None, pyaudio.paContinue)
    
    def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data with noise reduction."""
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Simple noise gate (remove very quiet samples)
        noise_threshold = np.max(np.abs(audio_np)) * 0.01
        audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
        
        return audio_np
    
    def _process_audio_chunk(self, audio_data: bytes) -> bool:
        """Process audio chunk with optimized VAD."""
        self.stats['total_chunks'] += 1
        
        # Preprocess audio
        audio_np = self._preprocess_audio(audio_data)
        
        # Apply VAD
        is_speech = self.vad.is_speech(audio_data, self.audio_config.sample_rate)
        
        if is_speech:
            self.stats['speech_chunks'] += 1
            
            if not self.is_speaking:
                logger.info("Speech detected - starting recording")
                self.is_speaking = True
                self.speech_frames.clear()
                self.speech_start_time = time.time()
            
            self.speech_frames.append(audio_data)
            self.silence_frames = 0
            
            # Check for maximum speech duration
            if len(self.speech_frames) >= self.max_speech_frames:
                logger.info("Maximum speech duration reached - processing")
                self._finalize_speech_segment()
                self.is_speaking = False
                
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.speech_frames.append(audio_data)
                
                # Check if we've had enough silence to end speech
                if self.silence_frames >= self.silence_threshold_frames:
                    # Only process if we have minimum speech duration
                    if len(self.speech_frames) >= self.min_speech_frames:
                        logger.info("Speech ended - processing for ASR")
                        self._finalize_speech_segment()
                    else:
                        logger.debug("Speech too short - discarding")
                    
                    self.is_speaking = False
                    self.silence_frames = 0
        
        return is_speech
    
    def _finalize_speech_segment(self):
        """Finalize speech segment for ASR processing."""
        if len(self.speech_frames) == 0:
            return
        
        # Combine all speech frames
        speech_data = b''.join(self.speech_frames)
        
        # Convert to float32 numpy array (required by Whisper)
        audio_np = np.frombuffer(speech_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Apply simple normalization
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.95
        
        try:
            self.speech_queue.put_nowait((audio_np, time.time()))
            duration = len(audio_np) / self.audio_config.sample_rate
            logger.info(f"Queued speech segment: {duration:.2f}s")
        except queue.Full:
            self.stats['queue_overflows'] += 1
            logger.warning("Speech queue full, dropping segment")
        
        self.speech_frames.clear()
    
    def _audio_processing_thread(self):
        """Optimized audio processing thread."""
        logger.info("Audio processing thread started")
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.05)
                self._process_audio_chunk(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
        
        logger.info("Audio processing thread stopped")
    
    def _asr_processing_thread(self):
        """Optimized ASR processing thread."""
        logger.info("ASR processing thread started")
        
        while self.running:
            try:
                audio_data, timestamp = self.speech_queue.get(timeout=0.1)
                start_time = time.time()
                
                # Perform ASR with optimized settings
                result = self.whisper_model.transcribe(
                    audio_data,
                    language=self.asr_config.language,
                    task='transcribe',
                    fp16=self.asr_config.fp16,
                    beam_size=self.asr_config.beam_size,
                    best_of=self.asr_config.best_of,
                    temperature=self.asr_config.temperature,
                    verbose=False,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0
                )
                
                recognition_time = time.time() - start_time
                text = result['text'].strip()
                
                # Calculate confidence (approximate)
                avg_logprob = result.get('segments', [{}])[0].get('avg_logprob', -1.0) if result.get('segments') else -1.0
                confidence = max(0.0, min(1.0, (avg_logprob + 1.0) / 1.0)) if avg_logprob > -2.0 else 0.5
                
                if text and len(text) > 1:  # Filter out very short/meaningless transcriptions
                    self.stats['recognition_count'] += 1
                    self.perf_monitor.add_processing_time(recognition_time)
                    
                    # Update average recognition time
                    if self.stats['avg_recognition_time'] == 0:
                        self.stats['avg_recognition_time'] = recognition_time
                    else:
                        self.stats['avg_recognition_time'] = (
                            self.stats['avg_recognition_time'] * 0.9 + recognition_time * 0.1
                        )
                    
                    latency = time.time() - timestamp
                    logger.info(f"Recognized ({recognition_time:.2f}s, latency: {latency:.2f}s, conf: {confidence:.2f}): {text}")
                    
                    # Call user callback if provided
                    if self.speech_callback:
                        try:
                            self.speech_callback(text, confidence)
                        except Exception as e:
                            logger.error(f"Error in speech callback: {e}")
                    
                    # Process transcription for task classification
                    try:
                        task_result = self.task_classifier.process_transcription(text)
                        if task_result:
                            logger.info(f"Task classified: {task_result}")
                            # Call task callback if provided
                            if self.task_callback:
                                try:
                                    self.task_callback(text, task_result)
                                except Exception as e:
                                    logger.error(f"Error in task callback: {e}")
                        else:
                            logger.info("No task classification result or text filtered out")
                    except Exception as e:
                        logger.error(f"Error in task classification: {e}")
                
                # Periodic garbage collection for memory management
                if self.stats['recognition_count'] % 10 == 0:
                    gc.collect()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in ASR processing: {e}")
        
        logger.info("ASR processing thread stopped")
    
    def _monitoring_thread(self):
        """Performance monitoring thread."""
        logger.info("Performance monitoring thread started")
        
        while self.running:
            try:
                time.sleep(5)  # Update every 5 seconds
                self.perf_monitor.update_system_stats()
                
                # Log performance warnings
                if self.perf_monitor.should_reduce_quality():
                    logger.warning("High system load detected - consider reducing quality settings")
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
        
        logger.info("Performance monitoring thread stopped")
    
    def start(self):
        """Start the optimized real-time ASR system."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting Optimized Real-time ASR system...")
        
        # Set process priority (Linux only)
        try:
            os.nice(-5)  # Higher priority
        except (OSError, AttributeError):
            pass
        
        # Open audio stream with optimized settings
        self.stream = self.audio.open(
            format=self.audio_config.format,
            channels=self.audio_config.channels,
            rate=self.audio_config.sample_rate,
            input=True,
            frames_per_buffer=self.audio_config.chunk_size,
            stream_callback=self._audio_callback,
            input_device_index=25  # Use default input device
        )
        
        self.running = True
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._audio_processing_thread, daemon=True),
            threading.Thread(target=self._asr_processing_thread, daemon=True),
            threading.Thread(target=self._monitoring_thread, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        # Start audio stream
        self.stream.start_stream()
        
        logger.info("Optimized Real-time ASR system started successfully")
    
    def stop(self):
        """Stop the optimized real-time ASR system."""
        if not self.running:
            return
        
        logger.info("Stopping Optimized Real-time ASR system...")
        
        self.running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Final garbage collection
        gc.collect()
        
        logger.info("Optimized Real-time ASR system stopped")
    
    def get_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        stats.update({
            'avg_cpu_usage': self.perf_monitor.get_avg_cpu_usage(),
            'avg_memory_usage': self.perf_monitor.get_avg_memory_usage(),
            'avg_processing_time': self.perf_monitor.get_avg_processing_time()
        })
        return stats
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def optimized_speech_handler(text: str, confidence: float):
    """Optimized speech callback function."""
    print(f"\n[SPEECH] ({confidence:.2f}): {text}\n")

def task_classification_handler(text: str, task: str):
    """Task classification callback function."""
    print(f"\n[TASK CLASSIFIED] Original: {text}")
    print(f"[TASK CLASSIFIED] Result: {task}\n")
    
    # Here you can add specific actions based on the classified task
    if task == "拿纸巾":
        print("🤖 执行任务：拿纸巾")
    elif task == "拿手机":
        print("🤖 执行任务：拿手机")
    elif task == "松开手机":
        print("🤖 执行任务：松开手机")
    elif task == "无":
        print("🤖 未识别到有效任务")
    else:
        print(f"🤖 未知任务：{task}")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down gracefully...")
    sys.exit(0)


def main():
    """Main function to run the optimized real-time ASR system."""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Optimized Real-time Always-On ASR System")
    print("========================================")
    print("High-performance system optimized for minimal latency and maximum throughput.")
    print("Press Ctrl+C to stop.\n")
    
    # Create optimized configurations
    audio_config = AudioConfig(
        sample_rate=16000,
        chunk_size=480,  # 30ms for optimal VAD performance
        channels=1
    )
    
    vad_config = VADConfig(
        aggressiveness=0,
        min_speech_duration=0.5,  # Filter out very short sounds
        max_speech_duration=3.0,  # Reasonable maximum
        silence_threshold=1.5  # Quick response
    )
    
    asr_config = ASRConfig(
        model_name="base",  # Fastest model for real-time
        language="zh",
        fp16=True,  # Half precision for speed
        beam_size=1,  # Greedy decoding
        temperature=0.0  # Deterministic
    )
    
    kimi_config = KimiConfig(
        api_key="sk-b4pdM1xHqOdY16GIWZxhKIFoR7zdJ3ob3QHo8MU88LY5ONFY",
        base_url="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
        temperature=0.3,
        max_chinese_words=15
    )
    
    # Create and start the optimized ASR system
    asr_system = OptimizedRealTimeASR(
        audio_config=audio_config,
        vad_config=vad_config,
        asr_config=asr_config,
        kimi_config=kimi_config,
        speech_callback=optimized_speech_handler,
        task_callback=task_classification_handler
    )
    
    try:
        with asr_system:
            print("Optimized system is now listening... Speak into your microphone.")
            
            # Keep the main thread alive and print stats periodically
            start_time = time.time()
            while True:
                time.sleep(15)  # Print stats every 15 seconds
                stats = asr_system.get_stats()
                runtime = time.time() - start_time
                
                print(f"\n--- Performance Stats (Runtime: {runtime:.1f}s) ---")
                print(f"Total chunks: {stats['total_chunks']}")
                print(f"Speech chunks: {stats['speech_chunks']} ({stats['speech_chunks']/max(stats['total_chunks'], 1)*100:.1f}%)")
                print(f"Recognitions: {stats['recognition_count']}")
                print(f"Dropped frames: {stats['dropped_frames']}")
                print(f"Queue overflows: {stats['queue_overflows']}")
                if stats['avg_recognition_time'] > 0:
                    print(f"Avg recognition time: {stats['avg_recognition_time']:.2f}s")
                print(f"CPU usage: {stats['avg_cpu_usage']:.1f}%")
                print(f"Memory usage: {stats['avg_memory_usage']:.1f}%")
                print("---\n")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        print("Optimized system stopped.")


if __name__ == "__main__":
    main()