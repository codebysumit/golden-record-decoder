"""Command-line entry point for the Voyager decoder GUI and batch mode."""

import argparse
import os
import sys
import tkinter as tk

from batch_decoder import run_headless_batch
from voyager_app import VoyagerApp

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voyager Golden Record Live Decoder (Python/Tkinter)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--audio", "-a", type=str, default=None,
                        help="Path to audio file (defaults to resources/voyager.mp3)")
    parser.add_argument("--no-live", action="store_true",
                        help="Run in headless batch mode without GUI and save images")
    parser.add_argument("--interval-ms", type=int, default=30,
                        help="Display update interval in milliseconds")
    parser.add_argument("--spec-window", type=int, default=1024,
                        choices=[256, 512, 1024, 2048, 4096],
                        help="Initial spectrogram FFT window size")
    parser.add_argument("--duration", type=float, default=None,
                        help="Decode only the first N seconds of audio")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Initial playback / decode speed multiplier")
    parser.add_argument("--out-dir", type=str, default="gold=en_out",
                        help="Output directory for decoded images")

    args = parser.parse_args()

    if args.no_live:
        if not args.audio or not os.path.isfile(args.audio):
            project_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(project_dir, "resources", "voyager.mp3"),
                "voyager.mp3",
                "record.mp3",
            ]
            audio_path = next(
                (path for path in candidates if os.path.isfile(path)),
                None,
            )
            if audio_path is None:
                print("Error: Please specify the audio file path using --audio path/to/file.mp3",
                      file=sys.stderr)
                sys.exit(1)
            args.audio = audio_path

        print(f"[VOYAGER] Selected audio source: {args.audio}")
        run_headless_batch(audio_path=args.audio, out_dir=args.out_dir,
                           duration=args.duration, nfft=args.spec_window)
    else:
        root = tk.Tk()
        VoyagerApp(root=root, audio_path=args.audio, interval_ms=args.interval_ms,
                   initial_nfft=args.spec_window, duration=args.duration, speed=args.speed)
        root.mainloop()


if __name__ == "__main__":
    main()
