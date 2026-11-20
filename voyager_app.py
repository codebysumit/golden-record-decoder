"""Tkinter user interface for the Voyager Golden Record decoder."""

import os
import sys
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import miniaudio
import numpy as np

from audio_utils import load_audio
from channel_decoder import ChannelDecoder

APP_VERSION = "1.0.0"
AUTHOR_NAME = "Sumit Das"
AUTHOR_GITHUB = "https://github.com/codebysumit/"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/codebysumit/"
AUTHOR_EMAIL = "codebysumit@gmail.com"


def get_src_asset(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", filename)


def load_photo_image(path, master, max_width, max_height):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not load image asset: {path}")

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1)
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = rgb_image.shape[:2]
    ppm_data = (
        f"P6\n{image_width} {image_height}\n255\n".encode("ascii")
        + rgb_image.tobytes()
    )
    return tk.PhotoImage(master=master, data=ppm_data, format="PPM")

# ---------------------------------------------------------------------------
# Responsive Voyager GUI Application
# ---------------------------------------------------------------------------

class VoyagerApp:
    def __init__(self, root, audio_path=None, interval_ms=30, initial_nfft=1024,
                 duration=None, speed=1.0):
        self.root = root
        self.audio_path = audio_path
        self.interval_ms = max(10, interval_ms)
        self.speed = speed
        self.duration_limit = duration
        self.initial_nfft = initial_nfft

        # State flags
        self.is_alive = True
        self.audio_loaded = False
        self.is_playing = False
        self.is_scrubbing = False
        self.is_minimized = False
        self.after_id = None

        # Channels
        self.left_ch = None
        self.right_ch = None
        self.audio_data = None
        self.sample_rate = 44100
        self.total_samples = 0
        self.duration_sec = 0.0
        self._audio_output_device = None
        self._audio_stream_generator = None
        self._audio_output_offset = 0.0
        self._audio_output_lock = threading.Lock()

        # Persistent PhotoImages and canvas item IDs
        self.photos = {}
        self.canvas_items = {}
        self.last_render_time = time.time()
        self.fps = 0.0

        self.build_ui()
        self.bind_events()

        if audio_path and os.path.isfile(audio_path):
            self.load_audio_file(audio_path)

        self.schedule_next_tick()

    def build_ui(self):
        self.root.title("Voyager Golden Record Live Decoder")
        self.root.geometry("1400x900")
        self.root.minsize(960, 600)
        self.root.configure(bg="#121212")

        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Open Audio File...", command=self.browse_audio_file,
                              accelerator="Ctrl+O")
        file_menu.add_command(label="Save Current Frames...", command=self.save_current_images,
                              accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menu_bar, tearoff=False)
        tools_menu.add_command(label="Play / Pause", command=self.toggle_play_pause,
                               accelerator="Space")
        tools_menu.add_command(label="Toggle View Mode", command=self.toggle_view_mode,
                               accelerator="F")
        tools_menu.add_command(label="Toggle Colour Mode", command=self.toggle_color_mode,
                               accelerator="C")
        tools_menu.add_separator()
        tools_menu.add_command(label="Toggle NASA Catalog",
                               command=lambda: self.chk_catalog.invoke(),
                               accelerator="K")
        tools_menu.add_command(label="Toggle DC/Contrast Fix",
                               command=lambda: self.chk_enhance.invoke(),
                               accelerator="E")
        menu_bar.add_cascade(label="Tools", menu=tools_menu)

        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu_bar)

        # Top Header Bar
        header = tk.Frame(self.root, bg="#1a1a1a", height=50)
        header.pack(side=tk.TOP, fill=tk.X)

        tk.Label(header, text="THE VOYAGER GOLDEN RECORD DECODER", font=("Segoe UI", 15, "bold"),
                 fg="#ffd700", bg="#1a1a1a").pack(side=tk.LEFT, padx=15, pady=6)

        # Global Master Toolbar
        toolbar = tk.Frame(self.root, bg="#212121", height=44, bd=1, relief=tk.FLAT)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=3)

        self.btn_browse = tk.Button(toolbar, text="📂 Open Audio File...", font=("Segoe UI", 9, "bold"),
                                    bg="#ff9800", fg="#000000", activebackground="#fb8c00",
                                    command=self.browse_audio_file, relief=tk.RAISED, padx=8)
        self.btn_browse.pack(side=tk.LEFT, padx=6, pady=4)

        self.btn_master_play = tk.Button(toolbar, text="▶ Play (Space)", font=("Segoe UI", 9, "bold"),
                                         bg="#2e7d32", fg="#ffffff", activebackground="#388e3c",
                                         command=self.toggle_play_pause, width=14, relief=tk.RAISED)
        self.btn_master_play.pack(side=tk.LEFT, padx=8, pady=4)

        # Speed selector
        tk.Label(toolbar, text="Speed:", fg="#aaaaaa", bg="#212121",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 2))
        self.speed_var = tk.StringVar(value=f"{self.speed}x")
        self.speed_combo = ttk.Combobox(toolbar, textvariable=self.speed_var,
                                        values=["0.5x", "1.0x", "2.0x", "4.0x", "8.0x"],
                                        width=5, state="readonly")
        self.speed_combo.pack(side=tk.LEFT, padx=2)
        self.speed_combo.bind("<<ComboboxSelected>>", self.on_speed_change)

        # View Mode selector
        tk.Label(toolbar, text="View:", fg="#aaaaaa", bg="#212121",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.view_var = tk.StringVar(value="Live Waterfall")
        self.view_combo = ttk.Combobox(toolbar, textvariable=self.view_var,
                                       values=["Live Waterfall", "Full Frame"],
                                       width=13, state="readonly")
        self.view_combo.pack(side=tk.LEFT, padx=2)
        self.view_combo.bind("<<ComboboxSelected>>", self.on_view_change)

        # Colour Mode selector: "Gray Scale" vs "RGB"
        tk.Label(toolbar, text="Colour:", fg="#aaaaaa", bg="#212121",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(10, 2))
        self.color_var = tk.StringVar(value="RGB")
        self.color_combo = ttk.Combobox(toolbar, textvariable=self.color_var,
                                        values=["Gray Scale", "RGB"],
                                        width=10, state="readonly")
        self.color_combo.pack(side=tk.LEFT, padx=2)
        self.color_combo.bind("<<ComboboxSelected>>", self.on_color_change)

        # NASA Catalog toggle checkbox (toggles VOYAGER_CATALOG on/off, default ON)
        self.catalog_var = tk.BooleanVar(value=True)
        self.chk_catalog = tk.Checkbutton(toolbar, text="NASA Catalog", variable=self.catalog_var,
                                          fg="#ffd700", bg="#212121", selectcolor="#111111",
                                          activebackground="#212121", activeforeground="#ffd700",
                                          font=("Segoe UI", 9, "bold"), command=self.on_catalog_toggle)
        self.chk_catalog.pack(side=tk.LEFT, padx=(8, 4))

        # DC Enhancement toggle
        self.enhance_var = tk.BooleanVar(value=True)
        self.chk_enhance = tk.Checkbutton(toolbar, text="DC/Contrast Fix", variable=self.enhance_var,
                                          fg="#00e5ff", bg="#212121", selectcolor="#111111",
                                          font=("Segoe UI", 9), command=self.on_enhance_toggle)
        self.chk_enhance.pack(side=tk.LEFT, padx=8)

        # FFT Window selector
        tk.Label(toolbar, text="FFT Window:", fg="#aaaaaa", bg="#212121",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 2))
        self.fft_var = tk.StringVar(value="1024 (23.2ms)")
        self.fft_combo = ttk.Combobox(toolbar, textvariable=self.fft_var,
                                      values=["256 (5.8ms)", "512 (11.6ms)", "1024 (23.2ms)",
                                              "2048 (46.4ms)", "4096 (92.9ms)"],
                                      width=14, state="readonly")
        self.fft_combo.pack(side=tk.LEFT, padx=2)
        self.fft_combo.bind("<<ComboboxSelected>>", self.on_fft_change)

        btn_save = tk.Button(toolbar, text="Save Frames", font=("Segoe UI", 9),
                             bg="#1565c0", fg="#ffffff", activebackground="#1976d2",
                             command=self.save_current_images, relief=tk.GROOVE)
        btn_save.pack(side=tk.RIGHT, padx=10, pady=4)

        # Timeline Scrubber Bar
        scrub_frame = tk.Frame(self.root, bg="#1a1a1a")
        scrub_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.lbl_time = tk.Label(scrub_frame, text="00:00 / 00:00", font=("Consolas", 10, "bold"),
                                 fg="#ffd700", bg="#1a1a1a", width=18)
        self.lbl_time.pack(side=tk.LEFT, padx=5)

        self.scrub_scale = tk.Scale(scrub_frame, from_=0, to=1000, orient=tk.HORIZONTAL,
                                    showvalue=False, bg="#2a2a2a", fg="#ffd700", troughcolor="#111111",
                                    highlightthickness=0, bd=0, sliderlength=20)
        self.scrub_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.scrub_scale.bind("<Button-1>", self.on_scrub_start)
        self.scrub_scale.bind("<B1-Motion>", self.on_scrub_motion)
        self.scrub_scale.bind("<ButtonRelease-1>", self.on_scrub_end)

        # Status bar packed BEFORE the expanding container so it never gets pushed off-screen
        status_frame = tk.Frame(self.root, bg="#0d0d0d", height=24)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_status = tk.Label(status_frame,
                                   text="No audio file loaded. Click '📂 Open Audio File...' to select a recording.",
                                   font=("Segoe UI", 9), fg="#ffb74d", bg="#0d0d0d")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.lbl_stats = tk.Label(status_frame, text="", font=("Consolas", 9),
                                  fg="#666666", bg="#0d0d0d")
        self.lbl_stats.pack(side=tk.RIGHT, padx=10)

        # Main Dual-Channel Container
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.main_container.grid_columnconfigure(0, weight=1, uniform="channel_col")
        self.main_container.grid_columnconfigure(1, weight=1, uniform="channel_col")
        self.main_container.grid_rowconfigure(0, weight=1)

        self.build_channel_column(self.main_container, col=0, ch_name="Left", accent_color="#00e5ff")
        self.build_channel_column(self.main_container, col=1, ch_name="Right", accent_color="#ff9100")

    def show_about_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("About Voyager Golden Record Decoder")
        dialog.configure(bg="#121212")
        dialog.resizable(False, False)
        dialog.transient(self.root)

        content = tk.Frame(dialog, bg="#121212", padx=20, pady=18)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="Voyager Golden Record Decoder",
                 font=("Segoe UI", 16, "bold"), fg="#ffd700",
                 bg="#121212").pack(anchor=tk.W)
        tk.Label(content, text=f"Version {APP_VERSION}",
                 font=("Segoe UI", 10), fg="#aaaaaa",
                 bg="#121212").pack(anchor=tk.W, pady=(2, 14))

        author_row = tk.Frame(content, bg="#121212")
        author_row.pack(fill=tk.X)

        photo_path = get_src_asset("profile_photo.png")
        if os.path.isfile(photo_path):
            author_photo = load_photo_image(photo_path, dialog, 200, 200)
            photo_label = tk.Label(author_row, image=author_photo, bd=0, bg="#121212")
            photo_label.image = author_photo
            photo_label.pack(side=tk.LEFT, padx=(0, 18))
        else:
            tk.Label(author_row, text="Author photo unavailable",
                     fg="#aaaaaa", bg="#121212", width=22, height=8).pack(
                         side=tk.LEFT, padx=(0, 18))

        details = tk.Frame(author_row, bg="#121212")
        details.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, expand=True)
        tk.Label(details, text=AUTHOR_NAME, font=("Segoe UI", 13, "bold"),
                 fg="#ffffff", bg="#121212").pack(anchor=tk.W)
        tk.Label(details, text="Developer & creator",
                 font=("Segoe UI", 10), fg="#aaaaaa",
                 bg="#121212").pack(anchor=tk.W, pady=(2, 10))
        tk.Label(details, text="Decode and explore the Voyager Golden Record's\n"
                               "stereo image data in real time.",
                 font=("Segoe UI", 9), fg="#dddddd", bg="#121212",
                 justify=tk.LEFT).pack(anchor=tk.W)

        links = tk.Frame(content, bg="#121212")
        links.pack(anchor=tk.W, pady=(16, 0))
        for label, url in (
            ("GitHub: codebysumit", AUTHOR_GITHUB),
            ("LinkedIn: Sumit Das", AUTHOR_LINKEDIN),
            (AUTHOR_EMAIL, f"mailto:{AUTHOR_EMAIL}"),
        ):
            ttk.Button(links, text=label,
                       command=lambda link=url: webbrowser.open(link)).pack(
                           side=tk.LEFT, padx=(0, 8))

        tk.Button(content, text="Close", command=dialog.destroy,
                  bg="#333333", fg="#ffffff", activebackground="#444444",
                  relief=tk.FLAT, padx=14).pack(anchor=tk.E, pady=(18, 0))
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()

    def build_channel_column(self, parent, col, ch_name, accent_color):
        frame = tk.Frame(parent, bg="#181818", bd=1, relief=tk.SOLID)
        frame.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        hdr_frame = tk.Frame(frame, bg="#1e1e1e")
        hdr_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=3)

        tk.Label(hdr_frame, text=f"{ch_name} Channel", font=("Segoe UI", 11, "bold"),
                 fg=accent_color, bg="#1e1e1e").pack(side=tk.LEFT, padx=8, pady=4)

        step_amount = 256 * 328
        btn_back = tk.Button(hdr_frame, text="⏪ Jump Back", font=("Segoe UI", 8),
                             bg="#333333", fg="#ffffff",
                             command=lambda ch=ch_name: self.jump_channel_name(ch, -step_amount))
        btn_back.pack(side=tk.LEFT, padx=4)

        btn_ch_play = tk.Button(hdr_frame, text="▶ Play", font=("Segoe UI", 8, "bold"),
                                bg="#2e7d32", fg="#ffffff",
                                command=lambda ch=ch_name: self.toggle_channel_name(ch))
        btn_ch_play.pack(side=tk.LEFT, padx=2)

        btn_fwd = tk.Button(hdr_frame, text="Jump Forward ⏩", font=("Segoe UI", 8),
                            bg="#333333", fg="#ffffff",
                            command=lambda ch=ch_name: self.jump_channel_name(ch, step_amount))
        btn_fwd.pack(side=tk.LEFT, padx=4)

        # Dynamic NASA Image Title & Author/Source Banner
        lbl_info = tk.Label(hdr_frame, text="", font=("Segoe UI", 9, "italic"),
                            fg="#ffd700", bg="#1e1e1e")
        lbl_info.pack(side=tk.RIGHT, padx=8, pady=4)

        wave_canvas = tk.Canvas(frame, bg="#000000", height=90, highlightthickness=1,
                                highlightbackground="#2a2a2a")
        wave_canvas.grid(row=1, column=0, sticky="ew", padx=4, pady=2)

        img_canvas = tk.Canvas(frame, bg="#080808", highlightthickness=1,
                               highlightbackground="#2a2a2a")
        img_canvas.grid(row=2, column=0, sticky="nsew", padx=4, pady=2)

        spec_canvas = tk.Canvas(frame, bg="#000000", height=110, highlightthickness=1,
                                highlightbackground="#2a2a2a")
        spec_canvas.grid(row=3, column=0, sticky="ew", padx=4, pady=2)

        if ch_name == "Left":
            self.btn_left_play = btn_ch_play
            self.canvas_left_wave = wave_canvas
            self.canvas_left_img = img_canvas
            self.canvas_left_spec = spec_canvas
            self.lbl_left_info = lbl_info
        else:
            self.btn_right_play = btn_ch_play
            self.canvas_right_wave = wave_canvas
            self.canvas_right_img = img_canvas
            self.canvas_right_spec = spec_canvas
            self.lbl_right_info = lbl_info

    def bind_events(self):
        self.root.bind("<Configure>", self.on_window_configure)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<space>", lambda e: self.toggle_play_pause())
        self.root.bind("q", lambda e: self.on_close())
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.bind("+", lambda e: self.change_speed(1))
        self.root.bind("=", lambda e: self.change_speed(1))
        self.root.bind("-", lambda e: self.change_speed(-1))
        self.root.bind("[", lambda e: self.step_fft_window(-1))
        self.root.bind("]", lambda e: self.step_fft_window(1))
        self.root.bind("f", lambda e: self.toggle_view_mode())
        self.root.bind("c", lambda e: self.toggle_color_mode())
        self.root.bind("k", lambda e: self.chk_catalog.invoke())
        self.root.bind("e", lambda e: self.chk_enhance.invoke())
        self.root.bind("s", lambda e: self.save_current_images())
        self.root.bind("o", lambda e: self.browse_audio_file())
        self.root.bind("<Control-o>", lambda e: self.browse_audio_file())
        self.root.bind("<Control-s>", lambda e: self.save_current_images())
        self.root.bind("<Left>", lambda e: self.jump_both(-256 * 328))
        self.root.bind("<Right>", lambda e: self.jump_both(256 * 328))

    def browse_audio_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Voyager Golden Record Audio Recording (WAV or MP3)",
            filetypes=[
                ("All Supported Audio (*.wav;*.mp3;*.flac;*.ogg)", "*.wav;*.mp3;*.flac;*.ogg"),
                ("WAV Files (*.wav)", "*.wav"),
                ("MP3 Files (*.mp3)", "*.mp3"),
                ("FLAC Files (*.flac)", "*.flac"),
                ("OGG Files (*.ogg)", "*.ogg"),
                ("All Files (*.*)", "*.*"),
            ]
        )
        if file_path:
            self.load_audio_file(file_path)

    def load_audio_file(self, file_path):
        """Loads audio and prepares it in READY / PAUSED state."""
        try:
            self.lbl_status.config(text=f"Loading: {os.path.basename(file_path)}...", fg="#ffd700")
            self.root.update_idletasks()

            audio_data, sr = load_audio(file_path)
            if self.duration_limit and self.duration_limit > 0:
                audio_data = audio_data[:int(self.duration_limit * sr)]

            self._close_audio_output()
            self.audio_data = audio_data
            self.sample_rate = sr
            self.total_samples = len(self.audio_data)
            self.duration_sec = self.total_samples / float(self.sample_rate)
            with self._audio_output_lock:
                self._audio_output_offset = 0.0

            # Initialize Decoders
            self.left_ch = ChannelDecoder(self.audio_data[:, 0], sr, name="Left", initial_offset=0)
            self.right_ch = ChannelDecoder(self.audio_data[:, 1], sr, name="Right", initial_offset=0)
            self.left_ch.set_fft_window(self.initial_nfft)
            self.right_ch.set_fft_window(self.initial_nfft)

            # Apply current UI modes
            use_cat = self.catalog_var.get()
            self.left_ch.use_catalog = use_cat
            self.right_ch.use_catalog = use_cat
            col_mode = 'gray' if 'Gray' in self.color_var.get() else 'rgb'
            self.left_ch.color_mode = col_mode
            self.right_ch.color_mode = col_mode
            v_mode = 'frame' if 'Frame' in self.view_var.get() else 'waterfall'
            self.left_ch.view_mode = v_mode
            self.right_ch.view_mode = v_mode

            self.scrub_scale.config(from_=0, to=self.total_samples - 1)
            self.audio_loaded = True
            self.is_playing = False  # Starts PAUSED

            self.update_play_button_ui()

            fname = os.path.basename(file_path)
            m_t, s_t = divmod(int(self.duration_sec), 60)
            self.lbl_status.config(
                text=f"Loaded: {fname} ({self.sample_rate} Hz, {m_t:02d}:{s_t:02d}) | Ready — Press 'Play' to start live stream!",
                fg="#a5d6a7"
            )

            self.render_all_panels()
            self.update_status_bar()

        except Exception as e:
            messagebox.showerror("Audio Load Error", f"Failed to load audio file:\n{e}")
            self.lbl_status.config(text=f"Error loading file: {e}", fg="#ef5350")

    def update_play_button_ui(self):
        if not self.audio_loaded:
            self.btn_master_play.config(text="▶ Play (Space)", bg="#424242", fg="#888888")
            return

        if self.is_playing:
            self.btn_master_play.config(text="⏸ Pause (Space)", bg="#e65100", fg="#ffffff")
            if hasattr(self, 'btn_left_play'):
                self.btn_left_play.config(text="⏸ Pause" if self.left_ch.go else "▶ Play",
                                          bg="#e65100" if self.left_ch.go else "#2e7d32")
            if hasattr(self, 'btn_right_play'):
                self.btn_right_play.config(text="⏸ Pause" if self.right_ch.go else "▶ Play",
                                           bg="#e65100" if self.right_ch.go else "#2e7d32")
        else:
            self.btn_master_play.config(text="▶ Play (Space)", bg="#2e7d32", fg="#ffffff")
            if hasattr(self, 'btn_left_play'):
                self.btn_left_play.config(text="▶ Play", bg="#2e7d32")
            if hasattr(self, 'btn_right_play'):
                self.btn_right_play.config(text="▶ Play", bg="#2e7d32")

    def _audio_output_stream(self):
        frame_count = yield
        while True:
            output = np.zeros((frame_count, 2), dtype=np.float32)
            with self._audio_output_lock:
                audio_data = self.audio_data
                playing = self.is_playing
                left_enabled = bool(self.left_ch and self.left_ch.go)
                right_enabled = bool(self.right_ch and self.right_ch.go)
                position = self._audio_output_offset
                speed = max(float(self.speed), 0.01)
                if audio_data is not None and playing and (left_enabled or right_enabled):
                    self._audio_output_offset = min(
                        position + frame_count * speed,
                        len(audio_data),
                    )

            if audio_data is not None and playing and (left_enabled or right_enabled):
                positions = position + np.arange(frame_count, dtype=np.float64) * speed
                valid = positions < len(audio_data)
                if np.any(valid):
                    indices = np.minimum(positions.astype(np.int64), len(audio_data) - 1)
                    next_indices = np.minimum(indices + 1, len(audio_data) - 1)
                    fractions = (positions - indices).astype(np.float32)[:, None]
                    samples = audio_data[indices] + (
                        audio_data[next_indices] - audio_data[indices]
                    ) * fractions
                    samples[~valid] = 0.0
                    if left_enabled:
                        output[:, 0] = samples[:, 0]
                    if right_enabled:
                        output[:, 1] = samples[:, 1]

            frame_count = yield output

    def _ensure_audio_output(self):
        if self._audio_output_device is not None:
            if self._audio_output_device.running:
                return True
            self._close_audio_output()

        device = None
        generator = None
        try:
            device = miniaudio.PlaybackDevice(
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=2,
                sample_rate=self.sample_rate,
                buffersize_msec=50,
                app_name="Voyager Golden Record Decoder",
            )
            generator = self._audio_output_stream()
            next(generator)
            device.start(generator)
        except miniaudio.MiniaudioError as exc:
            if device is not None:
                device.close()
            messagebox.showerror(
                "Audio Output Error",
                f"Could not start audio playback:\n{exc}",
                parent=self.root,
            )
            return False

        self._audio_output_device = device
        self._audio_stream_generator = generator
        return True

    def _close_audio_output(self):
        device = self._audio_output_device
        self._audio_output_device = None
        self._audio_stream_generator = None
        if device is not None:
            device.close()

    def toggle_play_pause(self):
        if not self.audio_loaded:
            self.browse_audio_file()
            return
        if not self.is_playing and not self._ensure_audio_output():
            return
        self.is_playing = not self.is_playing
        self.left_ch.go = self.is_playing
        self.right_ch.go = self.is_playing
        self.update_play_button_ui()

    def toggle_channel_name(self, ch_name):
        if not self.audio_loaded:
            return
        ch = self.left_ch if ch_name == "Left" else self.right_ch
        if not ch.go and not self._ensure_audio_output():
            return
        ch.go = not ch.go
        self.is_playing = (self.left_ch.go or self.right_ch.go)
        self.update_play_button_ui()

    def jump_channel_name(self, ch_name, delta):
        if not self.audio_loaded:
            return
        ch = self.left_ch if ch_name == "Left" else self.right_ch
        ch.seek(ch.offset + delta)
        self.render_all_panels()
        self.update_status_bar()

    def jump_both(self, delta):
        if not self.audio_loaded:
            return
        self.left_ch.seek(self.left_ch.offset + delta)
        self.right_ch.seek(self.right_ch.offset + delta)
        with self._audio_output_lock:
            self._audio_output_offset = float(self.left_ch.offset)
        self.render_all_panels()
        self.update_status_bar()

    # --- Live Responsive Seek Bar Handlers ---
    def on_scrub_start(self, event):
        if not self.audio_loaded:
            return
        self.is_scrubbing = True
        self.handle_scale_click(event)

    def on_scrub_motion(self, event):
        if not self.audio_loaded:
            return
        self.handle_scale_click(event)

    def on_scrub_end(self, event):
        if not self.audio_loaded:
            return
        self.handle_scale_click(event)
        self.is_scrubbing = False

    def handle_scale_click(self, event=None):
        if event is not None and hasattr(event, 'x'):
            width = self.scrub_scale.winfo_width()
            if width > 10:
                frac = max(0.0, min(1.0, float(event.x) / float(width)))
                target_spl = int(frac * (self.total_samples - 1))
            else:
                target_spl = int(self.scrub_scale.get())
        else:
            target_spl = int(self.scrub_scale.get())

        self.perform_seek(target_spl)

    def perform_seek(self, sample_offset):
        if not self.audio_loaded:
            return
        val = max(0, min(self.total_samples - 1, sample_offset))
        self.scrub_scale.set(val)
        with self._audio_output_lock:
            self._audio_output_offset = float(val)
        self.left_ch.seek(val)
        self.right_ch.seek(val)

        cur_sec = val / float(self.sample_rate)
        m_c, s_c = divmod(int(cur_sec), 60)
        m_t, s_t = divmod(int(self.duration_sec), 60)
        self.lbl_time.config(text=f"{m_c:02d}:{s_c:02d} / {m_t:02d}:{s_t:02d}")

        self.render_all_panels()
        self.update_status_bar()

    def on_window_configure(self, event):
        try:
            self.is_minimized = (self.root.state() == "iconic")
        except Exception:
            pass

    def on_close(self):
        self.is_alive = False
        self.is_playing = False
        self._close_audio_output()
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def on_speed_change(self, event=None):
        val_str = self.speed_var.get().replace("x", "")
        try:
            self.speed = float(val_str)
        except ValueError:
            self.speed = 1.0

    def change_speed(self, direction):
        speeds = [0.5, 1.0, 2.0, 4.0, 8.0]
        cur = self.speed
        idx = min(range(len(speeds)), key=lambda i: abs(speeds[i] - cur))
        new_idx = max(0, min(len(speeds) - 1, idx + direction))
        self.speed = speeds[new_idx]
        self.speed_var.set(f"{self.speed}x")

    def on_view_change(self, event=None):
        v = "frame" if "Frame" in self.view_var.get() else "waterfall"
        if self.left_ch:
            self.left_ch.view_mode = v
        if self.right_ch:
            self.right_ch.view_mode = v
        self.render_all_panels()

    def toggle_view_mode(self):
        cur = self.view_var.get()
        new_v = "Full Frame" if cur == "Live Waterfall" else "Live Waterfall"
        self.view_var.set(new_v)
        self.on_view_change()

    def on_color_change(self, event=None):
        if not self.catalog_var.get():
            # When catalog is toggled OFF, color RGB mode is disabled
            self.color_var.set("Gray Scale")
            mode = 'gray'
        else:
            c = self.color_var.get()
            mode = 'gray' if 'Gray' in c else 'rgb'

        if self.left_ch:
            self.left_ch.color_mode = mode
        if self.right_ch:
            self.right_ch.color_mode = mode
        self.render_all_panels()

    def toggle_color_mode(self):
        if not self.catalog_var.get():
            return
        cur = self.color_var.get()
        new_c = "RGB" if "Gray" in cur else "Gray Scale"
        self.color_var.set(new_c)
        self.on_color_change()

    def on_catalog_toggle(self):
        """Toggles VOYAGER_CATALOG metadata and RGB multi-pass color decoding on/off."""
        use_cat = self.catalog_var.get()
        if self.left_ch:
            self.left_ch.use_catalog = use_cat
            self.left_ch.current_meta = self.left_ch.get_image_meta(self.left_ch.offset)
        if self.right_ch:
            self.right_ch.use_catalog = use_cat
            self.right_ch.current_meta = self.right_ch.get_image_meta(self.right_ch.offset)

        if not use_cat:
            # When catalog is OFF: disable RGB mode, force Gray Scale!
            self.color_var.set("Gray Scale")
            self.color_combo.config(state="disabled")
            if self.left_ch:
                self.left_ch.color_mode = 'gray'
            if self.right_ch:
                self.right_ch.color_mode = 'gray'
        else:
            # When catalog is ON: re-enable RGB mode selection (default RGB)
            self.color_combo.config(state="readonly")
            self.color_var.set("RGB")
            if self.left_ch:
                self.left_ch.color_mode = 'rgb'
            if self.right_ch:
                self.right_ch.color_mode = 'rgb'

        self.render_all_panels()
        self.update_status_bar()

    def on_enhance_toggle(self):
        enh = self.enhance_var.get()
        if self.left_ch:
            self.left_ch.enhance = enh
        if self.right_ch:
            self.right_ch.enhance = enh
        self.render_all_panels()

    def on_fft_change(self, event=None):
        txt = self.fft_var.get()
        n = int(txt.split()[0])
        if self.left_ch:
            self.left_ch.set_fft_window(n)
        if self.right_ch:
            self.right_ch.set_fft_window(n)
        self.render_all_panels()

    def step_fft_window(self, direction):
        sizes = [256, 512, 1024, 2048, 4096]
        cur = self.left_ch.spec_nfft if self.left_ch else 1024
        idx = sizes.index(cur) if cur in sizes else 2
        new_idx = max(0, min(len(sizes) - 1, idx + direction))
        labels = ["256 (5.8ms)", "512 (11.6ms)", "1024 (23.2ms)", "2048 (46.4ms)", "4096 (92.9ms)"]
        self.fft_var.set(labels[new_idx])
        self.on_fft_change()

    def schedule_next_tick(self):
        if self.is_alive:
            self.after_id = self.root.after(self.interval_ms, self.timer_tick)

    def timer_tick(self):
        try:
            if self.audio_loaded and not self.is_minimized:
                if self.is_playing:
                    with self._audio_output_lock:
                        playback_finished = self._audio_output_offset >= self.total_samples
                    if playback_finished:
                        self.left_ch.go = False
                        self.right_ch.go = False
                        self.is_playing = False
                        self.update_play_button_ui()
                    else:
                        base_lines = max(1, int(round(1.8 * self.speed)))
                        if self.left_ch.go:
                            self.left_ch.step(base_lines)
                        if self.right_ch.go:
                            self.right_ch.step(base_lines)

                self.render_all_panels()
                self.update_status_bar()

            now = time.time()
            dt = now - self.last_render_time
            self.last_render_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        except Exception as e:
            print(f"Error in timer tick: {e}", file=sys.stderr)

        self.schedule_next_tick()

    def render_canvas_image(self, canvas, img_bgr, key):
        """Converts OpenCV BGR image to PPM and updates Tk canvas."""
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10 or h < 10:
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ok, ppm_bytes = cv2.imencode('.ppm', img_rgb)
        if not ok:
            return

        photo = tk.PhotoImage(data=ppm_bytes.tobytes())
        self.photos[key] = photo

        cx, cy = w // 2, h // 2
        if key in self.canvas_items:
            item_id = self.canvas_items[key]
            canvas.itemconfig(item_id, image=photo)
            canvas.coords(item_id, cx, cy)
        else:
            item_id = canvas.create_image(cx, cy, image=photo, anchor=tk.CENTER)
            self.canvas_items[key] = item_id

    def draw_placeholder(self, canvas, text, subtext=""):
        w = max(50, canvas.winfo_width())
        h = max(30, canvas.winfo_height())
        bg = np.full((h, w, 3), 16, dtype=np.uint8)
        cv2.rectangle(bg, (2, 2), (w - 3, h - 3), (35, 35, 35), 1)
        cv2.putText(bg, text, (max(10, w // 2 - len(text) * 4), h // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        if subtext:
            cv2.putText(bg, subtext, (max(10, w // 2 - len(subtext) * 3), h // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        self.render_canvas_image(canvas, bg, f"placeholder_{id(canvas)}")

    def render_channel(self, ch, wave_canvas, img_canvas, spec_canvas, prefix):
        w_w = max(50, wave_canvas.winfo_width())
        w_h = max(30, wave_canvas.winfo_height())
        self.render_canvas_image(wave_canvas, ch.render_waveform(w_w, w_h), f"{prefix}_wave")

        i_w = max(50, img_canvas.winfo_width())
        i_h = max(50, img_canvas.winfo_height())
        self.render_canvas_image(img_canvas, ch.render_image(i_w, i_h), f"{prefix}_img")

        s_w = max(50, spec_canvas.winfo_width())
        s_h = max(30, spec_canvas.winfo_height())
        self.render_canvas_image(spec_canvas, ch.render_spectrogram(s_w, s_h), f"{prefix}_spec")

    def render_all_panels(self):
        if not self.audio_loaded:
            self.draw_placeholder(self.canvas_left_wave, "Left Oscilloscope (Waiting for audio)")
            self.draw_placeholder(self.canvas_left_img, "VOYAGER LIVE DECODER",
                                  "Click 'Browse Audio File' to select recording")
            self.draw_placeholder(self.canvas_left_spec, "Left Spectrogram (Waiting for audio)")

            self.draw_placeholder(self.canvas_right_wave, "Right Oscilloscope (Waiting for audio)")
            self.draw_placeholder(self.canvas_right_img, "VOYAGER LIVE DECODER",
                                  "Click 'Browse Audio File' to select recording")
            self.draw_placeholder(self.canvas_right_spec, "Right Spectrogram (Waiting for audio)")
            return

        self.render_channel(self.left_ch, self.canvas_left_wave, self.canvas_left_img,
                            self.canvas_left_spec, "left")
        self.render_channel(self.right_ch, self.canvas_right_wave, self.canvas_right_img,
                            self.canvas_right_spec, "right")

    def update_status_bar(self):
        if not self.audio_loaded:
            return

        cur_offset = self.left_ch.offset
        cur_sec = cur_offset / float(self.sample_rate)

        if not self.is_scrubbing:
            self.scrub_scale.set(cur_offset)

        m_c, s_c = divmod(int(cur_sec), 60)
        m_t, s_t = divmod(int(self.duration_sec), 60)
        self.lbl_time.config(text=f"{m_c:02d}:{s_c:02d} / {m_t:02d}:{s_t:02d}")

        st_l = "LOCKED" if self.left_ch.sync_locked else "COASTING"
        st_r = "LOCKED" if self.right_ch.sync_locked else "COASTING"
        state_txt = "▶ PLAYING" if self.is_playing else "⏸ PAUSED"
        col_txt = self.color_var.get()

        use_cat = self.catalog_var.get()
        cat_txt = "ON" if use_cat else "OFF"

        self.lbl_status.config(
            text=f"[{state_txt}] Left: {self.left_ch.offset} ({st_l}) | Right: {self.right_ch.offset} ({st_r}) | Colour: {col_txt} | Catalog: {cat_txt}",
            fg="#a5d6a7"
        )
        self.lbl_stats.config(text=f"FFT Win: {self.left_ch.spec_nfft} spls | GUI: {self.fps:.1f} FPS")

        # Update Channel Header Information Banners
        if self.left_ch and hasattr(self, 'lbl_left_info'):
            if use_cat and self.left_ch.current_meta.get('name'):
                m_l = self.left_ch.current_meta
                src_l = f" — {m_l['source']}" if m_l.get('source') else ""
                c_tag_l = f"[{m_l['color']}]"
                self.lbl_left_info.config(text=f"{m_l['name']}{src_l} {c_tag_l}")
            else:
                self.lbl_left_info.config(text="")

        if self.right_ch and hasattr(self, 'lbl_right_info'):
            if use_cat and self.right_ch.current_meta.get('name'):
                m_r = self.right_ch.current_meta
                src_r = f" ({m_r['source']})" if m_r.get('source') else ""
                c_tag_r = f"[{m_r['color']}]"
                self.lbl_right_info.config(text=f"{m_r['name']}{src_r} {c_tag_r}")
            else:
                self.lbl_right_info.config(text="")

    def save_current_images(self, out_dir="golden_out"):
        if not self.audio_loaded:
            messagebox.showinfo("No Audio", "Please load an audio file first.")
            return

        os.makedirs(os.path.join(out_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "right"), exist_ok=True)
        t_stamp = int(time.time())
        saved = 0

        for ch, name in ((self.left_ch, "left"), (self.right_ch, "right")):
            meta = ch.current_meta
            c_name = meta.get('name') or f"frame_{name}"
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in c_name).strip().replace(" ", "_")
            c_tag = meta.get('color', 'Mono')
            p = os.path.join(out_dir, name, f"{safe_name}_{c_tag}_{ch.offset}_{t_stamp}.png")
            
            # Save the frame canvas (upright rotated if portrait)
            img = ch.frame_canvas.copy()
            if meta.get('portrait', False):
                if meta.get('anticlockwise', False):
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(p, img)
            saved += 1

        messagebox.showinfo("Saved", f"Decoded frames saved to folder:\n{os.path.abspath(out_dir)}")
