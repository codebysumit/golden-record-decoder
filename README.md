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

Click **Open Default Audio** to load the included Voyager recording, or choose
**File → Open Audio File** to load another MP3, WAV, FLAC, or OGG recording.
Use **File → Save Current Frames** to export the current channel images to
`golden_out/left` and `golden_out/right`. Playback, view, colour, catalog, and
contrast controls are available from the toolbar and **Tools** menu.
Playback sends the recording's left and right channels to the matching speaker;
the per-channel Play buttons let you mute either side independently.

The **Help → About** dialog shows the application version and author contact
links.

## Build a standalone Windows executable

Install the requirements, then run this command from the repository root in
PowerShell:

```powershell
$root = (Get-Location).Path
python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --console `
  --name golden_record_decoder `
  --icon "$root\src\256x256.ico" `
  --distpath "$root\build" `
  --workpath "$root\build\pyinstaller\work" `
  --specpath "$root\build\pyinstaller" `
  --add-data "$root\src;src" `
  --add-data "$root\resources\voyager.mp3;resources" `
  "$root\app.py"
```

The standalone, 64-bit Windows application is written to
`build/golden_record_decoder.exe`. It bundles Python, the application
dependencies, the app icons and profile image, and the sample `voyager.mp3`;
users do not need to install Python or these packages. Build it on Windows
with a 64-bit Python installation. Console mode keeps CLI output visible when
running commands such as `--no-live`; opening the executable from Explorer
also opens a console window alongside the GUI.

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

For the standalone executable, run it from a terminal to see progress and
errors there:

```powershell
.\build\golden_record_decoder.exe --no-live
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
- `resources/voyager.mp3`: Included default recording for GUI and headless use.
- `src/profile_photo.png`: Author image shown in the About dialog.
- `src/256x256.ico` and `src/icons/`: Application icon assets.
