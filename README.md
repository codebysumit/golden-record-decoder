# Voyager Golden Record Decoder

A Python/Tkinter desktop application that decodes the stereo image data in
Voyager Golden Record audio. It displays the left and right channels side by side
with live waveforms, reconstructed grayscale or RGB images, and spectrograms.

## Requirements

- Python 3.10 or later
- The packages listed in `requirements.txt`
- Tkinter (included with standard Python on Windows and macOS; on Linux install
  your distribution's Tkinter package, such as `python3-tk`)

## Install and run

```powershell
python -m pip install -r requirements.txt
python app.py
```

Choose **File → Open Audio File** to load an MP3, WAV, FLAC, or OGG recording.
Use **File → Save Current Frames** to export the current channel images to
`golden_out/left` and `golden_out/right`. Playback, view, colour, catalog, and
contrast controls are available from the toolbar and **Tools** menu.
Playback sends the recording's left and right channels to the matching speaker;
the per-channel Play buttons let you mute either side independently.

The **Help → About** dialog shows the application version and author contact
links.

## Voyager Golden Record audio

Download Voyager Golden Record audio files from the
[Google Drive audio collection](https://drive.google.com/drive/folders/1m2LEz2lQREN4j2zHSpv1swIFNCc5olyO?usp=sharing).
After downloading a recording, open it with **File → Open Audio File** or pass
its path with `--audio`.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Play or pause |
| `+` / `-` | Change playback speed |
| `[` / `]` | Change spectrogram FFT window |
| `f` | Toggle Live Waterfall / Full Frame |
| `c` | Toggle RGB / grayscale |
| `k` | Toggle NASA image catalog |
| `e` | Toggle DC/contrast correction |
| `o` / `Ctrl+O` | Open an audio file |
| `s` / `Ctrl+S` | Save current frames |
| `Left` / `Right` | Seek both channels |
| `q` / `Esc` | Close the application |

## Headless export

Decode an audio file without opening the GUI. The exporter saves one PNG per
complete catalog image, combines RGB passes into a single image, and skips
images that are incomplete at the selected duration. PNGs preserve the decoded
700-pixel scanlines and all scanlines collected for each image; they are not
resized to the GUI display size.

```powershell
python app.py --no-live
```

By default, this uses `resources/voyager.mp3`. To select a different recording:

```powershell
python app.py --no-live --audio path\to\recording.mp3
```

Optional arguments:

- `--duration 60`: Decode only the first 60 seconds.
- `--out-dir golden_out`: Set the output directory.

## Project files

- `app.py`: GUI and headless command-line entry point.
- `voyager_app.py`: Tkinter GUI.
- `channel_decoder.py`: Per-channel decoding and rendering.
- `voyager_catalog.py`: Voyager image metadata.
- `audio_utils.py`: Audio loading helpers.
- `batch_decoder.py`: Headless image export.
- `golden_record_live.py`: Compatibility launcher for existing commands.
- `resources/voyager.mp3`: Included sample recording for headless export.
- `src/profile_photo.png`: Author image shown in the About dialog.

The bundled NASA image-catalog metadata includes material derived from
GoldenDecoder; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
