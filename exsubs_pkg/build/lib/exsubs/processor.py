#!/usr/bin/env python3

import logging
import subprocess
from pathlib import Path

from .config import Config, TranslationMode
from .env import verify_dependencies, verify_api_keys
from .mkv import (
    get_mkv_subtitle_tracks, 
    select_track_interactively, 
    select_best_track_with_fallback,
    extract_track_with_progress
)
from .subs_io import preprocess_timecodes, postprocess_timecodes, filter_subtitles
from .translate import build_translation_cmd, run_translation

logger = logging.getLogger("exsubs")


class SubtitleProcessor:
    def __init__(
        self, 
        config: Config, 
        mode: TranslationMode, 
        interactive: bool = False, 
        show_progress: bool = True
    ):
        self.config = config
        self.mode = mode
        self.interactive = interactive
        self.show_progress = show_progress
        self.verify_dependencies()
        self.verify_api_keys()

    def verify_dependencies(self):
        """Verify required dependencies are available"""
        verify_dependencies()

    def verify_api_keys(self):
        """Verify required API keys are set based on translation mode"""
        verify_api_keys(self.mode)

    def extract_subtitles(self, video_file: Path, subtitle_file: Path) -> bool:
        """Extract subtitles from MKV file using mkvextract"""
        if not video_file.suffix.lower() == ".mkv":
            logger.error(f"Only MKV files are supported, got: {video_file.suffix}")
            return False

        # Get subtitle tracks from MKV file
        subtitle_tracks = get_mkv_subtitle_tracks(video_file)

        if not subtitle_tracks:
            logger.error(f"No subtitle tracks found in {video_file}")
            return False

        # Select track based on mode (interactive or automatic)
        if self.interactive:
            selected_track = select_track_interactively(subtitle_tracks)
        else:
            selected_track = select_best_track_with_fallback(subtitle_tracks)

        if not selected_track:
            if self.interactive:
                logger.info("No subtitle track selected")
            else:
                logger.error(f"No suitable subtitle track found in {video_file} (tried English and French)")
                logger.info("Available tracks:")
                for track in subtitle_tracks:
                    title = track.get("tags", {}).get("title", "No title")
                    language = track.get("tags", {}).get("language", "Unknown")
                    logger.info(f"  Track {track['id']}: {title} ({language})")
            return False

        # Extract using mkvextract with progress bar
        return extract_track_with_progress(
            video_file, subtitle_file, selected_track["id"], self.show_progress
        )

    def process_video(self, video_file: Path):
        """Process a single video file"""
        logger.info(f"Processing {video_file}")

        # Only process MKV files
        if not video_file.suffix.lower() == ".mkv":
            logger.warning(f"Skipping {video_file} - only MKV files are supported")
            return

        subtitle_file = video_file.with_suffix(".srt")
        translated_file = video_file.with_suffix(
            f".{self.config.get_language_code(self.config.target_language)}.srt"
        )

        # Check if translated file already exists
        if translated_file.exists():
            logger.info(f"Skipping {video_file} - translated subtitle already exists")
            return

        # Extract subtitles if needed
        if not subtitle_file.exists():
            if not self.extract_subtitles(video_file, subtitle_file):
                return

        # Process subtitles: preprocess → filter → postprocess
        preprocess_timecodes(subtitle_file)
        filter_subtitles(subtitle_file)
        postprocess_timecodes(subtitle_file)

        # Translate subtitles
        cmd = build_translation_cmd(
            self.mode, self.config, subtitle_file, translated_file, 
            self.config.target_language
        )

        try:
            run_translation(cmd, translated_file, subtitle_file)
        except subprocess.CalledProcessError:
            pass  # Error already logged in run_translation
