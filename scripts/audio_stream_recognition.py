import pyaudio
import numpy as np
import mlx_whisper
import threading
import queue
import time
import wave
import argparse
import re
from collections import deque
import noisereduce as nr
from difflib import SequenceMatcher
from scipy import signal
from mlx_lm import load, generate, stream_generate

def get_format_from_string(format_str):
    format_dict = {
        'int8': pyaudio.paInt8,
        'int16': pyaudio.paInt16,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32
    }
    return format_dict.get(format_str.lower(), pyaudio.paInt16)

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

def normalize_audio(audio_data, format):
    if format == pyaudio.paFloat32:
        return np.clip(audio_data, -1.0, 1.0)
    elif format == pyaudio.paInt8:
        return audio_data.astype(np.float32) / 128.0
    elif format == pyaudio.paInt16:
        return audio_data.astype(np.float32) / 32768.0
    elif format == pyaudio.paInt32:
        return audio_data.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported audio format: {format}")

def get_input_device_index():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "blackhole" in info["name"].lower():
            return i
    return None

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

args = parse_arguments()

# 音声キャプチャの設定
FORMAT = get_format_from_string(args.format)
CHANNELS = args.channels
RATE = args.rate
CHUNK = args.chunk
NUMPY_DTYPE = get_numpy_dtype(FORMAT)

# バッファの設定
BUFFER_DURATION = args.buffer_duration
BUFFER_SIZE = int(RATE * BUFFER_DURATION)

# グローバル変数
audio_queue = queue.Queue()
processing_queue = queue.Queue()
translation_queue = queue.Queue() 
is_running = True
start_time = time.time()

# 無音検出のためのパラメータ
SILENCE_THRESHOLD = 0.005
VOICE_ACTIVITY_THRESHOLD = 0.01
SILENCE_DURATION = 1.0  # 無音と判断する秒数

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=NUMPY_DTYPE)
    audio_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)

def audio_capture_thread():
    audio = pyaudio.PyAudio()
    input_device_index = args.input_device if args.input_device is not None else get_input_device_index()

    if input_device_index is None:
        print("適切な入力デバイスが見つかりません。手動で指定してください。")
        return

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK,
                        stream_callback=audio_callback)
    
    if args.debug:
        print(f"音声キャプチャスレッド開始 (デバイスインデックス: {input_device_index})")
    
    stream.start_stream()
    
    while is_running:
        time.sleep(0.1)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    if args.debug:
        print("音声キャプチャスレッド終了")

def is_silence(audio_data):
    normalized_data = normalize_audio(audio_data, FORMAT)
    rms = np.sqrt(np.mean(normalized_data**2))
    return rms < SILENCE_THRESHOLD

def has_voice_activity(audio_data):
    normalized_data = normalize_audio(audio_data, FORMAT)
    rms = np.sqrt(np.mean(normalized_data**2))
    return rms > VOICE_ACTIVITY_THRESHOLD

def save_audio_debug(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

def preprocess_audio(audio_data):
    # ノイズ除去
    reduced_noise = nr.reduce_noise(y=audio_data, sr=RATE)
    
    # バンドパスフィルター（人間の音声周波数帯域を強調）
    sos = signal.butter(10, [300, 3000], btype='band', fs=RATE, output='sos')
    filtered_audio = signal.sosfilt(sos, reduced_noise)
    
    return filtered_audio

def is_sentence_end(word):
    return word.endswith(('.', '!', '?')) and not word.endswith('...')

def print_with_strictly_controlled_linebreaks(text):
    words = text.split()
    buffer = []
    for i, word in enumerate(words):
        buffer.append(word)
        
        if is_sentence_end(word):
            print(' '.join(buffer), end='')
            if is_sentence_end(word):
                print('\n', end='', flush=True)
            else:
                print(' ', end='', flush=True)
            buffer = []
        elif i == len(words) - 1:
            print(' '.join(buffer), end='', flush=True)
            buffer = []
    
    if buffer:
        print(' '.join(buffer), end='', flush=True)

def translate_text(text, model, tokenizer):
    prompt= f"以下の英語を日本語に翻訳してください:\n{text}\n\n日本語訳:"

    generation_params = {
        "temp": 0.8,
        "top_p": 0.95,
        "max_tokens": 256,
        "repetition_penalty": 1.1,
        "repetition_context_size": 20,
    }
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    response = generate(model, tokenizer, prompt=prompt, **generation_params)
    return response

def translation_thread():
    # Local LLMモデルの読み込み
    llm_model, llm_tokenizer = load(path_or_hf_repo=args.llm_model)
    
    while is_running:
        try:
            text = translation_queue.get(timeout=1)
            translated_text = translate_text(text, llm_model, llm_tokenizer)
            print(f"\n翻訳: {translated_text}\n")
            #print(' ', end='', flush=True)
        except queue.Empty:
            if args.debug:
                print("翻訳キューが空です")
        except Exception as e:
            print(f"\nエラー (翻訳スレッド): {e}", flush=True)

def audio_processing_thread():
    buffer = deque(maxlen=BUFFER_SIZE)
    silence_start = None
    last_voice_activity = time.time()
    
    while is_running:
        try:
            data = audio_queue.get(timeout=0.1)
            buffer.extend(data)
            
            if has_voice_activity(data):
                last_voice_activity = time.time()
                silence_start = None
            elif silence_start is None:
                silence_start = time.time()
            
            if len(buffer) >= BUFFER_SIZE:
                current_time = time.time()
                if current_time - last_voice_activity < SILENCE_DURATION:
                    audio_data = np.array(buffer)
                    # 前処理を追加
                    #processed_audio = preprocess_audio(audio_data)
                    processed_audio = audio_data
                    processing_queue.put(processed_audio)
                
                # オーバーラップを考慮してバッファをクリア
                overlap = int(BUFFER_SIZE * 0.05)  # 5%オーバーラップ
                for _ in range(BUFFER_SIZE - overlap):
                    buffer.popleft()
        
        except queue.Empty:
            pass
        except Exception as e:
            print(f"\nエラー (処理スレッド): {e}", flush=True)

def speech_recognition_thread():
    last_text = ""
    last_text_time = 0
    while is_running:
        try:
            audio_data = processing_queue.get(timeout=1)
            normalized_audio = normalize_audio(audio_data, FORMAT)
            
            if args.debug:
                print("\n音声認識処理開始")
                save_audio_debug(audio_data, f"debug_audio_{time.time()}.wav")
            
            try:
                result = mlx_whisper.transcribe(normalized_audio,
                                                language=args.language,
                                                path_or_hf_repo=args.model_path)
            except Exception as e:
                print(f"音声認識エラー: {e}")
                continue
            
            text = result['text'].strip()
            
            current_time = time.time()
            if text and (text != last_text or current_time - last_text_time > 5):
                print_with_strictly_controlled_linebreaks(text)
                print(' ', end='', flush=True)
                last_text_time = current_time
            elif args.debug:
                print("処理後のテキストが空か、直前の文と同じため出力をスキップします")
        
        except queue.Empty:
            if args.debug:
                print("認識キューが空です")
        except Exception as e:
            print(f"\nエラー (認識スレッド): {e}", flush=True)

def speech_recognition_thread():
    last_text = ""
    last_text_time = 0
    while is_running:
        try:
            audio_data = processing_queue.get(timeout=1)
            normalized_audio = normalize_audio(audio_data, FORMAT)
            
            if args.debug:
                print("\n音声認識処理開始")
                save_audio_debug(audio_data, f"debug_audio_{time.time()}.wav")
            
            try:
                result = mlx_whisper.transcribe(normalized_audio,
                                                language=args.language,
                                                path_or_hf_repo=args.model_path)
            except Exception as e:
                print(f"音声認識エラー: {e}")
                continue
            
            text = result['text'].strip()
            
            current_time = time.time()
            if text and (text != last_text or current_time - last_text_time > 5):
                print_with_strictly_controlled_linebreaks(text)
                #print(' ', end='', flush=True)
                last_text_time = current_time
                translation_queue.put(text)  # 翻訳キューに追加
            elif args.debug:
                print("処理後のテキストが空か、直前の文と同じため出力をスキップします")
        
        except queue.Empty:
            if args.debug:
                print("認識キューが空です")
        except Exception as e:
            print(f"\nエラー (認識スレッド): {e}", flush=True)

def main():
    global is_running

    threads = [
        threading.Thread(target=audio_capture_thread),
        threading.Thread(target=audio_processing_thread),
        threading.Thread(target=speech_recognition_thread),
        threading.Thread(target=translation_thread)
    ]
    
    # すべてのスレッドを開始
    for thread in threads:
        thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        is_running = False
    
    # すべてのスレッドが終了するのを待つ
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
