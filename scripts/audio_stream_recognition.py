import pyaudio
import numpy as np
import mlx_whisper
import threading
import queue
import time
import wave
import argparse
from collections import deque
import noisereduce as nr
from scipy import signal
from mlx_lm import load, generate

class AudioConfig:
    def __init__(self, args):
        self.FORMAT = self.get_format_from_string(args.format)
        self.CHANNELS = args.channels
        self.RATE = args.rate
        self.CHUNK = args.chunk
        self.NUMPY_DTYPE = self.get_numpy_dtype(self.FORMAT)
        self.BUFFER_DURATION = args.buffer_duration
        self.BUFFER_SIZE = int(self.RATE * self.BUFFER_DURATION)
        self.SILENCE_THRESHOLD = 0.005
        self.VOICE_ACTIVITY_THRESHOLD = 0.01
        self.SILENCE_DURATION = 1.0

    @staticmethod
    def get_format_from_string(format_str):
        format_dict = {
            'int8': pyaudio.paInt8,
            'int16': pyaudio.paInt16,
            'int32': pyaudio.paInt32,
            'float32': pyaudio.paFloat32
        }
        return format_dict.get(format_str.lower(), pyaudio.paInt16)

    @staticmethod
    def get_numpy_dtype(format):
        if format == pyaudio.paInt8:
            return np.int8
        elif format == pyaudio.paInt16:
            return np.int16
        elif format == pyaudio.paInt32:
            return np.int32
        elif format == pyaudio.paFloat32:
            return np.float32
        else:
            raise ValueError(f"Unsupported audio format: {format}")

class AudioCapture:
    def __init__(self, config, audio_queue):
        self.config = config
        self.audio_queue = audio_queue

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=self.config.NUMPY_DTYPE)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def capture_thread(self, is_running, args):
        audio = pyaudio.PyAudio()
        input_device_index = args.input_device if args.input_device is not None else self.get_input_device_index()

        if input_device_index is None:
            print("適切な入力デバイスが見つかりません。手動で指定してください。")
            return

        stream = audio.open(format=self.config.FORMAT,
                            channels=self.config.CHANNELS,
                            rate=self.config.RATE,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=self.config.CHUNK,
                            stream_callback=self.audio_callback)
        
        if args.debug:
            print(f"音声キャプチャスレッド開始 (デバイスインデックス: {input_device_index})")
        
        stream.start_stream()
        
        while is_running.is_set():
            time.sleep(0.1)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if args.debug:
            print("音声キャプチャスレッド終了")

    @staticmethod
    def get_input_device_index():
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "blackhole" in info["name"].lower():
                return i
        return None

class AudioProcessing:
    def __init__(self, config, audio_queue, processing_queue):
        self.config = config
        self.audio_queue = audio_queue
        self.processing_queue = processing_queue

    def processing_thread(self, is_running, args):
        buffer = deque(maxlen=self.config.BUFFER_SIZE)
        silence_start = None
        last_voice_activity = time.time()
        
        while is_running.is_set():
            try:
                data = self.audio_queue.get(timeout=0.1)
                buffer.extend(data)
                
                if self.has_voice_activity(data):
                    last_voice_activity = time.time()
                    silence_start = None
                elif silence_start is None:
                    silence_start = time.time()
                
                if len(buffer) >= self.config.BUFFER_SIZE:
                    current_time = time.time()
                    if current_time - last_voice_activity < self.config.SILENCE_DURATION:
                        audio_data = np.array(buffer)
                        processed_audio = self.preprocess_audio(audio_data)
                        self.processing_queue.put(processed_audio)
                    
                    overlap = int(self.config.BUFFER_SIZE * 0.05)
                    for _ in range(self.config.BUFFER_SIZE - overlap):
                        buffer.popleft()
            
            except queue.Empty:
                pass
            except Exception as e:
                print(f"\nエラー (処理スレッド): {e}", flush=True)

    def has_voice_activity(self, audio_data):
        normalized_data = self.normalize_audio(audio_data)
        rms = np.sqrt(np.mean(normalized_data**2))
        return rms > self.config.VOICE_ACTIVITY_THRESHOLD

    def normalize_audio(self, audio_data):
        if self.config.FORMAT == pyaudio.paFloat32:
            return np.clip(audio_data, -1.0, 1.0)
        elif self.config.FORMAT == pyaudio.paInt8:
            return audio_data.astype(np.float32) / 128.0
        elif self.config.FORMAT == pyaudio.paInt16:
            return audio_data.astype(np.float32) / 32768.0
        elif self.config.FORMAT == pyaudio.paInt32:
            return audio_data.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported audio format: {self.config.FORMAT}")

    def preprocess_audio(self, audio_data):
        reduced_noise = nr.reduce_noise(y=audio_data, sr=self.config.RATE)
        sos = signal.butter(10, [300, 3000], btype='band', fs=self.config.RATE, output='sos')
        filtered_audio = signal.sosfilt(sos, reduced_noise)
        return filtered_audio

class SpeechRecognition:
    def __init__(self, config, processing_queue, translation_queue, args):
        self.config = config
        self.processing_queue = processing_queue
        self.translation_queue = translation_queue
        self.args = args

    def recognition_thread(self, is_running):
        last_text = ""
        last_text_time = 0
        while is_running.is_set():
            try:
                audio_data = self.processing_queue.get(timeout=1)
                normalized_audio = self.normalize_audio(audio_data)
                
                if self.args.debug:
                    print("\n音声認識処理開始")
                    self.save_audio_debug(audio_data, f"debug_audio_{time.time()}.wav")
                
                try:
                    result = mlx_whisper.transcribe(normalized_audio,
                                                    language=self.args.language,
                                                    path_or_hf_repo=self.args.model_path)
                except Exception as e:
                    print(f"音声認識エラー: {e}")
                    continue
                
                text = result['text'].strip()
                
                current_time = time.time()
                if text and (text != last_text or current_time - last_text_time > 5):
                    self.print_with_strictly_controlled_linebreaks(text)
                    last_text_time = current_time
                    self.translation_queue.put(text)
                elif self.args.debug:
                    print("処理後のテキストが空か、直前の文と同じため出力をスキップします")
            
            except queue.Empty:
                if self.args.debug:
                    print("認識キューが空です")
            except Exception as e:
                print(f"\nエラー (認識スレッド): {e}", flush=True)

    def normalize_audio(self, audio_data):
        if self.config.FORMAT == pyaudio.paFloat32:
            return np.clip(audio_data, -1.0, 1.0)
        elif self.config.FORMAT == pyaudio.paInt8:
            return audio_data.astype(np.float32) / 128.0
        elif self.config.FORMAT == pyaudio.paInt16:
            return audio_data.astype(np.float32) / 32768.0
        elif self.config.FORMAT == pyaudio.paInt32:
            return audio_data.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported audio format: {self.config.FORMAT}")

    def save_audio_debug(self, audio_data, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.config.CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(self.config.FORMAT))
            wf.setframerate(self.config.RATE)
            wf.writeframes(audio_data.tobytes())

    @staticmethod
    def is_sentence_end(word):
        return word.endswith(('.', '!', '?')) and not word.endswith('...')

    @staticmethod
    def print_with_strictly_controlled_linebreaks(text):
        words = text.split()
        buffer = []
        for i, word in enumerate(words):
            buffer.append(word)
            
            if SpeechRecognition.is_sentence_end(word):
                print(' '.join(buffer), end='')
                if SpeechRecognition.is_sentence_end(word):
                    print('\n', end='', flush=True)
                else:
                    print(' ', end='', flush=True)
                buffer = []
            elif i == len(words) - 1:
                print(' '.join(buffer), end='', flush=True)
                buffer = []
        
        if buffer:
            print(' '.join(buffer), end='', flush=True)

class Translation:
    def __init__(self, translation_queue, args):
        self.translation_queue = translation_queue
        self.args = args
        self.load_model()
        self.last_reload_time = time.time()
        self.reload_interval = 60  # 1分ごとにモデルを再ロード
        self.consecutive_errors = 0
        self.max_consecutive_errors = 1
        self.error_cooldown = 2  # エラー後の待機時間（秒）
        self.failed_translations = []  # エラーとなった原文を保存するリスト

    def load_model(self):
        self.llm_model, self.llm_tokenizer = load(path_or_hf_repo=self.args.llm_model)

    def translation_thread(self, is_running):
        while is_running.is_set():
            try:
                if self.failed_translations:
                    text = self.failed_translations.pop(0)
                    if self.args.debug:
                        print(f"\n再翻訳を試みます: {text}\n")
                else:
                    text = self.translation_queue.get(timeout=1)
                
                processed_text = self.preprocess_text(text)
                translated_text = self.translate_text(processed_text)
                
                if self.is_valid_translation(translated_text):
                    print(f"\n翻訳: {translated_text}\n")
                    self.consecutive_errors = 0
                else:
                    if self.args.debug:
                        print(f"\n翻訳エラー: 有効な翻訳を生成できませんでした。原文: {text}\n")
                    self.handle_translation_error(text)
                
            except queue.Empty:
                if self.args.debug:
                    print("翻訳キューが空です")
            except Exception as e:
                print(f"\nエラー (翻訳スレッド): {e}", flush=True)
                self.handle_translation_error(text)
            
            self.check_model_reload()

    def translate_text(self, text):
        prompt = f"以下の英語を日本語に翻訳してください。翻訳のみを出力し、余計な説明は不要です:\n\n{text}\n\n日本語訳:"

        generation_params = {
            "temp": 0.3,
            "top_p": 0.95,
            "max_tokens": 256,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
        }
        if hasattr(self.llm_tokenizer, "apply_chat_template") and self.llm_tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        response = generate(self.llm_model, self.llm_tokenizer, prompt=prompt, **generation_params)
        return response.strip()

    @staticmethod
    def is_valid_translation(text):
        return bool(text) and len(set(text)) > 1 and not text.startswith('!!!') and not text.endswith('!!!')

    def handle_translation_error(self, text):
        self.consecutive_errors += 1
        self.failed_translations.append(text)
        if self.consecutive_errors >= self.max_consecutive_errors:
            if self.args.debug:
                print("連続エラーが発生しました。モデルを再ロードします。")
            self.load_model()
            self.consecutive_errors = 0
        time.sleep(self.error_cooldown)

    def check_model_reload(self):
        current_time = time.time()
        if current_time - self.last_reload_time > self.reload_interval:
            if self.args.debug:
                print("定期的なモデル再ロードを実行します。")
            self.load_model()
            self.last_reload_time = current_time

    @staticmethod
    def preprocess_text(text):
        text = text.replace("...", " ")
        text = text.replace("&", "and")
        
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text.strip()

class AudioRecognitionSystem:
    def __init__(self, args):
        self.args = args
        self.config = AudioConfig(args)
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        self.is_running = threading.Event()
        self.is_running.set()

        self.audio_capture = AudioCapture(self.config, self.audio_queue)
        self.audio_processing = AudioProcessing(self.config, self.audio_queue, self.processing_queue)
        self.speech_recognition = SpeechRecognition(self.config, self.processing_queue, self.translation_queue, args)
        self.translation = Translation(self.translation_queue, args)

    def run(self):
        threads = [
            threading.Thread(target=self.audio_capture.capture_thread, args=(self.is_running, self.args)),
            threading.Thread(target=self.audio_processing.processing_thread, args=(self.is_running, self.args)),
            threading.Thread(target=self.speech_recognition.recognition_thread, args=(self.is_running,)),
            threading.Thread(target=self.translation.translation_thread, args=(self.is_running,))
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.is_running.clear()
        
        for thread in threads:
            thread.join()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time Audio Recognition with Noise Reduction")
    parser.add_argument("--model_path", type=str, default="mlx-community/whisper-large-v3-turbo-q4",
                        help="Path or HuggingFace repo for the Whisper model")
    parser.add_argument("--language", type=str, default="ja",
                        help="Language code for speech recognition (e.g., 'en' for English, 'ja' for Japanese)")
    parser.add_argument("--format", type=str, default="int16",
                        choices=['int8', 'int16', 'int32', 'float32'],
                        help="Audio format (default: int16)")
    parser.add_argument("--rate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("--channels", type=int, default=1,
                        help="Number of channels (default: 1)")
    parser.add_argument("--chunk", type=int, default=1024,
                        help="Chunk size (default: 1024)")
    parser.add_argument("--input_device", type=int, help="Input device index (default: auto-detect Black Hole)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--buffer_duration", type=float, default=2.0,
                        help="Duration of audio buffer in seconds (default: 2.0)")
    parser.add_argument("--llm_model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit",
                        help="Path to the local LLM model for translation")
    return parser.parse_args()

def main():
    args = parse_arguments()
    system = AudioRecognitionSystem(args)
    system.run()

if __name__ == "__main__":
    main()

