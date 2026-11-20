"""Signal decoder and renderers for one Voyager audio channel."""

import cv2
import numpy as np

from voyager_catalog import VOYAGER_CATALOG

# ---------------------------------------------------------------------------
# Channel Decoder Class
# ---------------------------------------------------------------------------

class ChannelDecoder:
    """
    Decodes one audio channel of the Voyager Golden Record.
    Implements scanline sync, live waterfall, live sweeping full frame, and
    GoldenDecoder Program.c multi-pass RGB color compositing.
    """
    def __init__(self, audio_data, sample_rate=44100, name="Left", initial_offset=0):
        self.audio = audio_data.astype(np.float32)
        self.total_samples = len(self.audio)
        self.sample_rate = sample_rate
        self.name = name
        self.offset = min(max(0, initial_offset), self.total_samples - 1)
        self.go = False

        # Scanline parameters (~734 samples per line at 44.1 kHz, 512 lines per frame)
        self.nominal_period = 734
        self.line_width = 700
        self.lines_per_frame = 512
        self.sync_locked = False
        self.last_period = self.nominal_period

        threshold_sample = self.audio[::max(1, self.total_samples // 1_000_000)]
        self.sync_threshold = (
            max(float(np.percentile(np.abs(threshold_sample), 99.5)) * 0.1, 1e-6)
            if threshold_sample.size else 1e-6
        )

        # NASA Catalog and Image Metadata
        ch_key = self.name.lower()
        self.catalog = VOYAGER_CATALOG.get(ch_key, [])
        self.use_catalog = True
        self.current_meta = self.get_image_meta(self.offset)

        # Live waterfall buffer (600 lines high x 700 wide, 3 channels BGR)
        self.waterfall_height = 600
        self.waterfall = np.full((self.waterfall_height, self.line_width, 3), 16, dtype=np.uint8)

        # Live Full Frame Canvas (700 rows x 512 columns x 3 channels BGR)
        # Matches the live scanning frame from GoldenDecoder Program.c!
        self.frame_canvas = np.zeros((self.line_width, self.lines_per_frame, 3), dtype=np.uint8)
        self.scan_line_index = 0
        self.completed_frames = []

        # Display settings
        self.enhance = True
        self.color_mode = 'rgb'   # 'rgb' or 'gray'
        self.view_mode = 'waterfall'  # 'waterfall' or 'frame'

        # Spectrogram parameters
        self.spec_nfft = 1024
        self.spec_bins = self.spec_nfft // 2 + 1
        self.spec_cols = 160
        self.spec_buffer = np.full((self.spec_bins, self.spec_cols), -80.0, dtype=np.float32)
        self.hanning_window = np.hanning(self.spec_nfft).astype(np.float32)
        self.spec_step_counter = 0

    def get_image_meta(self, offset):
        """Lookup current image metadata from the NASA catalog based on sample offset."""
        if not getattr(self, 'use_catalog', True) or not self.catalog:
            return {'name': '', 'offset': 0, 'color': 'Mono', 'source': '', 'desc': '', 'portrait': False, 'anticlockwise': False}
        if offset < self.catalog[0]['offset']:
            return {'name': '', 'offset': 0, 'color': 'Mono', 'source': '', 'desc': '', 'portrait': False, 'anticlockwise': False}
        for i in range(len(self.catalog)):
            cur = self.catalog[i]
            nxt = self.catalog[i + 1] if i + 1 < len(self.catalog) else None
            if offset >= cur['offset'] and (nxt is None or offset < nxt['offset']):
                return cur
        return self.catalog[-1]

    def set_fft_window(self, nfft):
        """Change FFT window size dynamically."""
        if nfft not in (256, 512, 1024, 2048, 4096):
            return
        self.spec_nfft = nfft
        self.spec_bins = nfft // 2 + 1
        self.spec_buffer = np.full((self.spec_bins, self.spec_cols), -80.0, dtype=np.float32)
        self.hanning_window = np.hanning(self.spec_nfft).astype(np.float32)

    def find_next_pulse(self):
        """Locates the next scanline sync pulse (bounded search + flywheel fallback)."""
        w_start = self.offset + self.nominal_period - 110
        w_end = self.offset + self.nominal_period + 110

        if w_end >= self.total_samples:
            self.offset = 0
            self.sync_locked = False
            return self.offset

        chunk = self.audio[w_start:w_end]
        max_val = np.max(chunk)

        if max_val > self.sync_threshold:
            local_peak = np.argmax(chunk)
            peak_idx = w_start + local_peak

            thresh = self.audio[peak_idx] * 0.20
            search_limit = min(self.total_samples, peak_idx + 45)
            window = self.audio[peak_idx:search_limit]
            drops = np.where(window < thresh)[0]
            falling_idx = peak_idx + drops[0] if len(drops) > 0 else peak_idx

            measured_period = falling_idx - self.offset
            if 650 <= measured_period <= 820:
                self.last_period = int(measured_period)
                self.sync_locked = True
                self.offset = int(falling_idx)
                return self.offset

        # Flywheel coasting fallback
        self.sync_locked = False
        self.offset += self.nominal_period
        self.last_period = self.nominal_period
        if self.offset >= self.total_samples - self.nominal_period:
            self.offset = 0
        return self.offset

    def decode_single_line(self):
        """
        Decodes a single scanline starting from current offset.
        Updates the live sweeping Full Frame canvas using the GoldenDecoder C program logic,
        and streams lines to the Live Waterfall buffer.
        """
        old_offset = self.offset
        self.find_next_pulse()

        if self.offset < old_offset:  # wrapped around
            self.frame_canvas.fill(0)
            self.scan_line_index = 0
            return None

        seg = self.audio[old_offset:self.offset]
        if len(seg) < 100:
            return None

        # Check for image transition in the NASA catalog
        meta = self.get_image_meta(self.offset)
        if getattr(self, 'use_catalog', True):
            if meta['offset'] != self.current_meta.get('offset', 0):
                previous_meta = self.current_meta
                same_rgb_image = (
                    meta['name'] == previous_meta.get('name')
                    and (
                        (previous_meta.get('color') == 'Red' and meta['color'] == 'Green')
                        or (previous_meta.get('color') == 'Green' and meta['color'] == 'Blue')
                    )
                )
                if not same_rgb_image:
                    self.frame_canvas.fill(0)
                self.scan_line_index = 0
                self.current_meta = meta
        else:
            self.current_meta = meta

        # Resample to line_width (700 px); reversed horizontally for correct scan orientation
        xs = np.linspace(0, len(seg) - 1, self.line_width)
        resampled = np.interp(xs, np.arange(len(seg)), seg[::-1])

        # Raw scan amplitude to brightness [0..255]
        raw_pixels = 108.0 - resampled * 2555.0
        line_uint8 = np.clip(raw_pixels, 0, 255).astype(np.uint8)

        # DC baseline correction & contrast enhancement
        if self.enhance:
            base = cv2.medianBlur(line_uint8.reshape(1, -1), 41).ravel()
            corrected = line_uint8.astype(np.float32) - (base.astype(np.float32) - 128.0) * 0.65
            line_uint8 = np.clip(corrected, 0, 255).astype(np.uint8)

        # -------------------------------------------------------------------
        # Full Frame Live Scanning Logic (from GoldenDecoder Program.c)
        # -------------------------------------------------------------------
        x = self.scan_line_index
        if 0 <= x < self.lines_per_frame:
            if not getattr(self, 'use_catalog', True) or self.color_mode == 'gray':
                # 1. Gray Scale mode or Catalog OFF: Always render in pure grayscale
                self.frame_canvas[:, x, 0] = line_uint8
                self.frame_canvas[:, x, 1] = line_uint8
                self.frame_canvas[:, x, 2] = line_uint8
            else:
                # 2. RGB mode with Catalog ON: Multi-pass overlay logic from Program.c
                c_chan = meta['color']
                if c_chan == 'Mono':
                    # Grayscale single pass
                    self.frame_canvas[:, x, 0] = line_uint8
                    self.frame_canvas[:, x, 1] = line_uint8
                    self.frame_canvas[:, x, 2] = line_uint8
                elif c_chan == 'Red':
                    # Pass 1: Red channel drawn; Blue & Green cleared
                    self.frame_canvas[:, x, 0] = 0
                    self.frame_canvas[:, x, 1] = 0
                    self.frame_canvas[:, x, 2] = line_uint8
                elif c_chan == 'Green':
                    # Pass 2: Green channel added on top of existing Red!
                    self.frame_canvas[:, x, 1] = line_uint8
                elif c_chan == 'Blue':
                    # Pass 3: Blue channel added on top of existing Red + Green!
                    self.frame_canvas[:, x, 0] = line_uint8

        self.scan_line_index = (self.scan_line_index + 1) % self.lines_per_frame

        # -------------------------------------------------------------------
        # Live Waterfall Buffer Streaming
        # -------------------------------------------------------------------
        if not getattr(self, 'use_catalog', True) or self.color_mode == 'gray' or meta['color'] == 'Mono':
            color_row = np.repeat(line_uint8[:, np.newaxis], 3, axis=1)
        else:
            color_row = np.zeros((self.line_width, 3), dtype=np.uint8)
            if meta['color'] == 'Red':
                color_row[:, 2] = line_uint8
            elif meta['color'] == 'Green':
                color_row[:, 1] = line_uint8
            elif meta['color'] == 'Blue':
                color_row[:, 0] = line_uint8
            else:
                color_row = np.repeat(line_uint8[:, np.newaxis], 3, axis=1)

        # Scroll waterfall upward by 2 px
        self.waterfall[0:-2, :, :] = self.waterfall[2:, :, :]
        self.waterfall[-2, :, :] = color_row
        self.waterfall[-1, :, :] = color_row

        return line_uint8

    def update_spectrogram(self):
        """Computes STFT slice at current offset and shifts rolling spectrogram."""
        half_n = self.spec_nfft // 2
        s_start = self.offset - half_n
        s_end = self.offset + half_n

        if s_start < 0:
            chunk = np.pad(self.audio[0:max(0, s_end)], (abs(s_start), 0))
        elif s_end >= self.total_samples:
            chunk = np.pad(self.audio[s_start:], (0, s_end - self.total_samples))
        else:
            chunk = self.audio[s_start:s_end]

        if len(chunk) < self.spec_nfft:
            chunk = np.pad(chunk, (0, self.spec_nfft - len(chunk)))
        elif len(chunk) > self.spec_nfft:
            chunk = chunk[:self.spec_nfft]

        windowed = chunk * self.hanning_window
        mag = np.abs(np.fft.rfft(windowed))
        db = 20.0 * np.log10(np.maximum(mag, 1e-4))

        self.spec_buffer[:, :-1] = self.spec_buffer[:, 1:]
        self.spec_buffer[:, -1] = db

    def step(self, num_lines=1):
        """Perform `num_lines` decoding iterations."""
        if not self.go:
            return
        for _ in range(num_lines):
            self.decode_single_line()
            self.spec_step_counter += 1
            if self.spec_step_counter % 2 == 0:
                self.update_spectrogram()

    def render_waveform(self, width, height):
        """Renders oscilloscope waveform matching web app style."""
        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        center_y = height // 2

        canvas[center_y, :, :] = (45, 45, 45)
        quarter = height // 4
        canvas[quarter, :, :] = (30, 30, 30)
        canvas[3 * quarter, :, :] = (30, 30, 30)
        for gx in range(0, width, max(20, width // 10)):
            canvas[:, gx, :] = (28, 28, 28)

        plot_len = 3000
        plot_start = self.offset - 140
        p_end = plot_start + plot_len

        if plot_start < 0:
            buf = np.pad(self.audio[0:max(0, p_end)], (abs(plot_start), 0))
        elif p_end > self.total_samples:
            buf = np.pad(self.audio[plot_start:], (0, p_end - self.total_samples))
        else:
            buf = self.audio[plot_start:p_end]

        if len(buf) >= plot_len:
            buf = buf[:plot_len]
            zoom = height * 2.2
            xs = np.linspace(0, width - 1, plot_len).astype(np.int32)
            ys = np.clip(center_y - (buf * zoom).astype(np.int32), 0, height - 1)
            pts = np.column_stack([xs, ys])
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 240, 160),
                          thickness=1, lineType=cv2.LINE_AA)

            trig_x = int(140 * (width / plot_len))
            if 0 <= trig_x < width:
                cv2.line(canvas, (trig_x, 0), (trig_x, height), (0, 120, 255), 1)

        cv2.putText(canvas, f"OSC: {self.name} Ch | Period: {self.last_period} spls", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        sync_txt = "SYNC LOCKED" if self.sync_locked else "SYNC RECOVERING"
        sync_col = (0, 220, 0) if self.sync_locked else (0, 140, 255)
        cv2.putText(canvas, sync_txt, (max(8, width - 130), 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, sync_col, 1, cv2.LINE_AA)
        return canvas

    def render_image(self, width, height):
        """
        Renders the decoded image: Live Waterfall or Full Frame view.
        Full Frame shows the live scanline cursor sweeping across and upright orientation,
        matching the GoldenDecoder C program GIF output.
        """
        meta = self.current_meta
        use_cat = getattr(self, 'use_catalog', True)

        if self.view_mode == 'frame':
            img_src = self.frame_canvas.copy()

            # Active scanline sweep cursor (like GoldenDecoder / GIF output!)
            cur_x = self.scan_line_index
            if 0 <= cur_x < self.lines_per_frame:
                img_src[:, cur_x, :] = np.clip(img_src[:, cur_x, :].astype(np.int16) + 80, 0, 255).astype(np.uint8)
                if cur_x + 1 < self.lines_per_frame:
                    img_src[:, cur_x + 1, :] = (0, 255, 255) # Bright sweep line

            # Portrait rotation for images stored sideways on the record (only when catalog is ON)
            if use_cat and meta.get('portrait', False):
                if meta.get('anticlockwise', False):
                    img_src = cv2.rotate(img_src, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    img_src = cv2.rotate(img_src, cv2.ROTATE_90_CLOCKWISE)

            scaled = cv2.resize(img_src, (width, height), interpolation=cv2.INTER_LINEAR)

            if use_cat and meta.get('name'):
                # Telemetry Banner: NASA Title, Source, and Color Badge
                name_txt = meta['name']
                src_txt = f" ({meta['source']})" if meta.get('source') else ""
                chan_badge = f"[{meta['color'].upper()}]" if meta['color'] != 'Mono' else "[MONO]"
                mode_badge = "RGB" if (self.color_mode == 'rgb' and meta['color'] != 'Mono') else "GRAYSCALE"

                # Top info card
                info_str = f"{name_txt}{src_txt}"
                cv2.rectangle(scaled, (4, 4), (min(width - 4, 14 + len(info_str) * 8), 26), (15, 15, 15), -1)
                cv2.putText(scaled, info_str, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 215, 0), 1, cv2.LINE_AA)

                # Status badge
                badge_txt = f"FULL FRAME | {mode_badge} {chan_badge}"
                b_w = len(badge_txt) * 7 + 14
                cv2.rectangle(scaled, (width - b_w - 6, 4), (width - 4, 26), (15, 15, 15), -1)
                badge_col = (0, 255, 255) if mode_badge == "RGB" else (180, 180, 180)
                cv2.putText(scaled, badge_txt, (width - b_w, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.36, badge_col, 1, cv2.LINE_AA)
            else:
                # No catalog metadata labels shown when catalog is OFF
                badge_txt = "FULL FRAME | RAW GRAYSCALE"
                b_w = len(badge_txt) * 7 + 14
                cv2.rectangle(scaled, (width - b_w - 6, 4), (width - 4, 26), (15, 15, 15), -1)
                cv2.putText(scaled, badge_txt, (width - b_w, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

        else:
            # Live Waterfall view
            scaled = cv2.resize(self.waterfall, (width, height), interpolation=cv2.INTER_LINEAR)
            if use_cat and meta.get('name'):
                chan_badge = f"[{meta['color'].upper()}]" if meta['color'] != 'Mono' else "[MONO]"
                mode_badge = "RGB" if (self.color_mode == 'rgb' and meta['color'] != 'Mono') else "GRAYSCALE"
                badge_txt = f"LIVE WATERFALL | {mode_badge} {chan_badge}"
                badge_col = (0, 220, 255) if mode_badge == "RGB" else (180, 180, 180)
            else:
                badge_txt = "LIVE WATERFALL | RAW GRAYSCALE"
                badge_col = (180, 180, 180)

            cv2.rectangle(scaled, (4, 4), (len(badge_txt) * 8 + 14, 26), (15, 15, 15), -1)
            cv2.putText(scaled, badge_txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_col, 1, cv2.LINE_AA)

        return scaled

    def render_spectrogram(self, width, height):
        """Rolling spectrogram with Inferno colormap."""
        norm = np.clip((self.spec_buffer + 75.0) * (255.0 / 75.0), 0, 255).astype(np.uint8)
        flipped = norm[::-1, :]
        colored = cv2.applyColorMap(flipped, cv2.COLORMAP_INFERNO)
        scaled = cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)

        for frac, khz in [(0.25, "16 kHz"), (0.50, "11 kHz"), (0.75, "5.5 kHz")]:
            gy = int(height * frac)
            scaled[gy, :, :] = (scaled[gy, :, :].astype(np.uint16) + 30).clip(0, 255).astype(np.uint8)
            cv2.putText(scaled, khz, (5, gy - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (200, 200, 200), 1, cv2.LINE_AA)

        win_ms = (self.spec_nfft / self.sample_rate) * 1000.0
        info_str = f"STFT Window: {self.spec_nfft} spls ({win_ms:.1f} ms) | 0-22.05 kHz"
        cv2.putText(scaled, info_str, (max(5, width - 280), height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)
        return scaled

    def seek(self, new_offset):
        """Seek channel to new offset, resetting frame state."""
        self.offset = min(max(0, int(new_offset)), self.total_samples - 1)
        self.current_meta = self.get_image_meta(self.offset)
        self.sync_locked = False
        self.scan_line_index = 0
        self.frame_canvas.fill(0)
        self.waterfall.fill(16)
        self.spec_buffer.fill(-80.0)
