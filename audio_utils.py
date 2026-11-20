"""Audio loading helpers for Voyager recordings."""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Audio Loading Utility
# ---------------------------------------------------------------------------

def load_audio(file_path):
    """
    Loads audio file (WAV, MP3, FLAC, OGG) into float32 stereo numpy array: shape (N, 2), values in [-1.0, 1.0].
    Tries miniaudio first (fastest for MP3 & WAV), then soundfile, then scipy.io.wavfile,
    with standard library wave module as a universal WAV fallback.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. Try miniaudio (Fastest for MP3 & WAV)
    try:
        import miniaudio
        info = miniaudio.get_file_info(file_path)
        sample_rate = info.sample_rate
        dec = miniaudio.decode_file(file_path)
        if dec.sample_format == miniaudio.SampleFormat.SIGNED16:
            arr = np.frombuffer(dec.samples, dtype=np.int16).reshape(-1, dec.nchannels)
            arr = arr.astype(np.float32) / 32768.0
        elif dec.sample_format == miniaudio.SampleFormat.FLOAT32:
            arr = np.frombuffer(dec.samples, dtype=np.float32).reshape(-1, dec.nchannels)
        else:
            arr = np.frombuffer(dec.samples, dtype=np.int16).reshape(-1, dec.nchannels).astype(np.float32) / 32768.0

        if arr.shape[1] == 1:
            arr = np.column_stack([arr, arr])
        return arr[:, :2], sample_rate
    except Exception:
        pass

    # 2. Try soundfile (High quality for WAV, FLAC, OGG)
    try:
        import soundfile as sf
        arr, sample_rate = sf.read(file_path, dtype='float32')
        if arr.ndim == 1:
            arr = np.column_stack([arr, arr])
        return arr[:, :2], sample_rate
    except Exception:
        pass

    # 3. Try scipy.io.wavfile (WAV fallback)
    try:
        from scipy.io import wavfile
        sample_rate, arr = wavfile.read(file_path)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int32:
            arr = arr.astype(np.float32) / 2147483648.0
        elif arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = np.column_stack([arr, arr])
        return arr[:, :2], sample_rate
    except Exception:
        pass

    # 4. Standard library wave module (Universal standard library WAV fallback)
    try:
        import wave
        with wave.open(file_path, 'rb') as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw_bytes = wf.readframes(nframes)
            if sampwidth == 2:
                arr = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, nchannels)
                arr = arr.astype(np.float32) / 32768.0
            elif sampwidth == 4:
                arr = np.frombuffer(raw_bytes, dtype=np.int32).reshape(-1, nchannels)
                arr = arr.astype(np.float32) / 2147483648.0
            elif sampwidth == 1:
                arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, nchannels)
                arr = (arr.astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 3:
                # 24-bit PCM
                raw = np.frombuffer(raw_bytes, dtype=np.uint8)
                raw = raw.reshape(-1, nchannels, 3)
                int32_arr = (raw[:, :, 0].astype(np.int32) << 8) | (raw[:, :, 1].astype(np.int32) << 16) | (raw[:, :, 2].astype(np.int32) << 24)
                arr = int32_arr.astype(np.float32) / 2147483648.0
            else:
                arr = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, nchannels).astype(np.float32) / 32768.0
            if arr.shape[1] == 1:
                arr = np.column_stack([arr, arr])
            return arr[:, :2], framerate
    except Exception:
        pass

    raise RuntimeError(f"Could not load audio file '{file_path}' with any supported backend (miniaudio, soundfile, scipy, wave).")
