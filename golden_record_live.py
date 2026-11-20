"""Compatibility imports and launcher; use app.py as the application entry point."""

from app import main
from audio_utils import load_audio
from batch_decoder import run_headless_batch
from channel_decoder import ChannelDecoder
from voyager_app import (
    APP_VERSION,
    AUTHOR_EMAIL,
    AUTHOR_GITHUB,
    AUTHOR_LINKEDIN,
    AUTHOR_NAME,
    VoyagerApp,
    get_src_asset,
    load_photo_image,
)
from voyager_catalog import VOYAGER_CATALOG


if __name__ == "__main__":
    main()
