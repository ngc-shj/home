import pyaudio
import numpy as np
import mlx_whisper
import threading
import queue
import time
import wave
import argparse

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
    parser = argparse.ArgumentParser(description="Real-time Audio Recognition")
    parser.add_argument("--model_path", type=str, default="mlx-community/whisper-large-v3-turbo",
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
    return parser.parse_args()

args = parse_arguments()

# 音声キャプチャの設定
FORMAT = get_format_from_string(args.format)
CHANNELS = args.channels
RATE = args.rate
CHUNK = args.chunk
NUMPY_DTYPE = get_numpy_dtype(FORMAT)

# グローバル変数
audio_queue = queue.Queue()
is_running = True
start_time = time.time()

# 無音検出のためのパラメータ
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION = 0.3  # 無音と判断する秒数
VOICE_ACTIVITY_THRESHOLD = 0.01  # 音声活動を判断するための閾値

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
    if args.debug:
        print(f"RMS: {rms}, SILENCE_THRESHOLD: {SILENCE_THRESHOLD}")
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

def process_audio_thread():
    buffer = []
    silence_start = None
    has_activity = False

    while is_running:
        try:
            data = audio_queue.get(timeout=1)
            buffer.append(data)
            
            if has_voice_activity(data):
                has_activity = True
            
            if is_silence(data):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    if len(buffer) > 0 and has_activity:
                        audio_data = np.concatenate(buffer)
                        normalized_audio = normalize_audio(audio_data, FORMAT)
                        if args.debug:
                            print("無音検出: 音声認識処理開始")
                            save_audio_debug(audio_data, f"debug_audio_{time.time()}.wav")
                        
                        try:
                            result = mlx_whisper.transcribe(normalized_audio,
                                                            language=args.language,
                                                            path_or_hf_repo=args.model_path)
                        except Exception as e:
                            print(f"音声認識エラー: {e}")
                            buffer = []
                            silence_start = None
                            has_activity = False
                            continue
                        
                        text = result['text'].strip()
                        
                        if args.debug:
                            print(f"認識結果: {text}")
                        
                        if text:
                            elapsed_time = time.time() - start_time
                            print(f"\n[{elapsed_time:.2f}s] {text}", flush=True)
                        
                        buffer = []
                        silence_start = None
                        has_activity = False
            else:
                silence_start = None
            
        except queue.Empty:
            if args.debug:
                print("キューが空です")
        except Exception as e:
            print(f"\nエラー: {e}", flush=True)

def main():
    global is_running
    
    capture_thread = threading.Thread(target=audio_capture_thread)
    process_thread = threading.Thread(target=process_audio_thread)
    
    capture_thread.start()
    process_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        is_running = False
    
    capture_thread.join()
    process_thread.join()

if __name__ == "__main__":
    main()
