"""Headless batch decoding for Voyager recordings."""

import os

import cv2
import numpy as np

from audio_utils import load_audio
from channel_decoder import ChannelDecoder


def _group_catalog_images(catalog):
    groups = []
    for entry in catalog:
        if groups:
            previous = groups[-1][-1]
            same_rgb_image = (
                entry['name'] == previous['name']
                and (
                    (previous['color'] == 'Red' and entry['color'] == 'Green')
                    or (previous['color'] == 'Green' and entry['color'] == 'Blue')
                )
            )
            if same_rgb_image:
                groups[-1].append(entry)
                continue
        groups.append([entry])
    return groups


def _image_color_label(group):
    if [entry['color'] for entry in group] == ['Red', 'Green', 'Blue']:
        return 'RGB'
    return group[0]['color']


def run_headless_batch(audio_path, out_dir="golden_out", duration=None, nfft=1024):
    """Decode and save each complete catalog image without resizing its pixels."""
    print(f"[HEADLESS] Loading audio: {audio_path}")
    audio, sr = load_audio(audio_path)
    if duration and duration > 0:
        audio = audio[:int(duration * sr)]

    os.makedirs(os.path.join(out_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "right"), exist_ok=True)

    def decode_channel_stream(channel_data, ch_name):
        print(f"\n--- Decoding {ch_name} Channel ---")
        decoder = ChannelDecoder(channel_data, sr, name=ch_name, initial_offset=0)
        decoder.set_fft_window(nfft)
        sub_dir = os.path.join(out_dir, ch_name.lower())
        image_groups = _group_catalog_images(decoder.catalog)
        groups_by_offset = {
            entry['offset']: group
            for group in image_groups
            for entry in group
        }
        pass_line_counts = {}
        output_counts = {}
        active_group = None
        image_canvas = np.zeros((decoder.line_width, 0, 3), dtype=np.uint8)
        saved = 0

        def save_complete_frame(group, frame, line_counts):
            nonlocal saved

            if not all(
                line_counts.get(entry['offset'], 0) >= decoder.lines_per_frame
                for entry in group
            ):
                return

            meta = group[-1]
            complete_width = min(line_counts[entry['offset']] for entry in group)
            frame = frame[:, :complete_width, :]
            name = meta.get('name') or f"frame_{meta['offset']:04d}"
            safe_name = "".join(
                char if char.isalnum() or char in "._- " else "_"
                for char in name
            ).strip().replace(" ", "_")
            label = _image_color_label(group)
            base_name = f"{safe_name}_{label}"
            occurrence = output_counts.get(base_name, 0)
            output_counts[base_name] = occurrence + 1
            suffix = f"_{occurrence + 1}" if occurrence else ""
            path = os.path.join(sub_dir, f"{base_name}{suffix}.png")

            if meta.get('portrait', False):
                rotation = (
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                    if meta.get('anticlockwise', False)
                    else cv2.ROTATE_90_CLOCKWISE
                )
                frame = cv2.rotate(frame, rotation)
            if not cv2.imwrite(path, frame):
                raise OSError(f"Could not save decoded frame: {path}")
            print(f"  [{ch_name}] Saved: {path}")
            saved += 1

        while decoder.offset + decoder.nominal_period + 110 < decoder.total_samples:
            previous_offset = decoder.offset
            decoded_line = decoder.decode_single_line()
            if decoder.offset < previous_offset:
                break

            meta = decoder.current_meta
            group = groups_by_offset.get(meta['offset'])
            if decoded_line is None or group is None:
                continue

            if active_group is None or group[0]['offset'] != active_group[0]['offset']:
                if active_group is not None:
                    save_complete_frame(active_group, image_canvas, pass_line_counts)
                active_group = group
                pass_line_counts = {}
                image_canvas = np.zeros((decoder.line_width, 0, 3), dtype=np.uint8)

            line_index = pass_line_counts.get(meta['offset'], 0)
            pass_line_counts[meta['offset']] = line_index + 1
            if line_index >= image_canvas.shape[1]:
                expanded_canvas = np.zeros(
                    (decoder.line_width, line_index + 1, 3), dtype=np.uint8
                )
                expanded_canvas[:, :image_canvas.shape[1], :] = image_canvas
                image_canvas = expanded_canvas

            if meta['color'] == 'Mono':
                image_canvas[:, line_index, :] = decoded_line[:, np.newaxis]
            elif meta['color'] == 'Red':
                image_canvas[:, line_index, :] = 0
                image_canvas[:, line_index, 2] = decoded_line
            elif meta['color'] == 'Green':
                image_canvas[:, line_index, 1] = decoded_line
            elif meta['color'] == 'Blue':
                image_canvas[:, line_index, 0] = decoded_line

        if active_group is not None:
            save_complete_frame(active_group, image_canvas, pass_line_counts)
        print(f"  [{ch_name}] Saved {saved} complete image(s).")

    decode_channel_stream(audio[:, 0], "Left")
    decode_channel_stream(audio[:, 1], "Right")

    print(f"\n[DONE] Batch decoding complete. Results saved in '{out_dir}'.")
