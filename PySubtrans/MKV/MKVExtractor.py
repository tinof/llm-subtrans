import json
import logging
import os
import regex as re
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("exsubs")

def get_mkv_subtitle_tracks(video_file : Path) -> list[dict]:
    """Get subtitle tracks from MKV file using mkvmerge"""
    try:
        # Validate file exists and is readable
        if not video_file.exists():
            logger.error(f"File does not exist: {video_file}")
            return []

        if not video_file.is_file():
            logger.error(f"Path is not a file: {video_file}")
            return []

        cmd = ["mkvmerge", "-J", str(video_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse JSON and handle errors
        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse mkvmerge JSON output: {e}")
            logger.debug(f"mkvmerge stdout: {result.stdout[:500]}")
            return []

        # Check for errors in mkvmerge output
        if "errors" in info and info["errors"]:
            for error in info["errors"]:
                logger.error(f"mkvmerge error: {error}")
            return []

        subtitle_tracks = []
        for track in info.get("tracks", []):
            if track.get("type") == "subtitles":
                track_info = {
                    "id": track.get("id"),
                    "codec": track.get("codec"),
                    "properties": track.get("properties", {}),
                    "tags": {},
                }

                # Extract language info
                if "language" in track.get("properties", {}):
                    track_info["tags"]["language"] = track["properties"]["language"]

                # Extract track name if available
                if "track_name" in track.get("properties", {}):
                    track_info["tags"]["title"] = track["properties"]["track_name"]

                # Check for forced flag
                if (
                    "forced_track" in track.get("properties", {})
                    and track["properties"]["forced_track"]
                ):
                    track_info["disposition"] = {"forced": True}
                else:
                    track_info["disposition"] = {"forced": False}

                # Check for hearing impaired flag
                if (
                    "hearing_impaired" in track.get("properties", {})
                    and track["properties"]["hearing_impaired"]
                ):
                    track_info["disposition"]["hearing_impaired"] = True
                else:
                    track_info["disposition"]["hearing_impaired"] = False

                subtitle_tracks.append(track_info)

        return subtitle_tracks
    except subprocess.CalledProcessError as e:
        logger.error(f"mkvmerge command failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"Failed to get MKV subtitle tracks: {str(e)}")
        return []


def select_track_interactively(tracks : list[dict], console) -> dict|None:
    """Allow user to manually select a subtitle track"""
    if not tracks:
        console.print("[red]No subtitle tracks found.[/red]")
        return None

    console.print("\n[bold]Available subtitle tracks:[/bold]")
    console.print()

    # Display tracks in a table-like format
    for i, track in enumerate(tracks, 1):
        track_id = track.get("id", "Unknown")
        title = track.get("tags", {}).get("title", "No title")
        language = track.get("tags", {}).get("language", "Unknown")
        disposition = track.get("disposition", {})

        # Format flags
        flags = []
        if disposition.get("forced"):
            flags.append("Forced")
        if disposition.get("hearing_impaired"):
            flags.append("SDH")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        console.print(f"[cyan]{i:2d}.[/cyan] Track ID {track_id}: [bold]{title}[/bold] ({language}){flag_str}")

    console.print()

    while True:
        try:
            choice = console.input("Select track number (1-{}) or 'q' to quit: ".format(len(tracks)))

            if choice.lower() == 'q':
                console.print("[yellow]Cancelled by user.[/yellow]")
                return None

            track_num = int(choice)
            if 1 <= track_num <= len(tracks):
                selected_track = tracks[track_num - 1]
                title = selected_track.get("tags", {}).get("title", "No title")
                language = selected_track.get("tags", {}).get("language", "Unknown")
                track_id = selected_track.get("id")
                console.print(f"[green]✓ Selected track {track_id}: {title} ({language})[/green]")
                return selected_track
            else:
                console.print(f"[red]Please enter a number between 1 and {len(tracks)}[/red]")

        except ValueError:
            console.print("[red]Please enter a valid number or 'q' to quit[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user.[/yellow]")
            return None


def select_best_track_for_translation(tracks : list[dict], preferred_languages : list[str]|None = None) -> dict|None:
    """Select the best subtitle track optimized for translation purposes"""

    if preferred_languages is None:
        preferred_languages = ["eng", "en"]

    def is_text_track(track : dict) -> bool:
        """Return True if the codec represents a text-based subtitle"""
        codec = (track.get("codec") or "").lower()
        text_codecs = [
            "subrip", "srt", "s_text/utf8", "s_text/ass",
            "substationalpha", "ssa", "ass", "s_text/ssa",
        ]
        return any(tok in codec for tok in text_codecs)

    # Filter text-based tracks first
    text_tracks = [t for t in tracks if is_text_track(t)]
    candidate_tracks = text_tracks if text_tracks else tracks

    def get_track_score(track : dict, target_languages : list[str]) -> tuple:
        tags = track.get("tags", {})
        disposition = track.get("disposition", {})
        title = tags.get("title", "").lower()
        language = tags.get("language", "").lower()

        # Check if track matches any of the target languages
        is_target_language = (
            language in target_languages or
            any(lang.lower() in title for lang in target_languages if len(lang) > 2)
        )

        if not is_target_language:
            return (-1, track["id"])

        # Penalize image-based codecs heavily
        if not is_text_track(track):
            return (-5, track["id"])

        # Score based on track type and quality
        if any(keyword in title for keyword in ["dialogue", "dialog", "full", "complete"]):
            if not disposition.get("forced") and not disposition.get("hearing_impaired"):
                return (100, track["id"])  # Perfect match
            elif not disposition.get("forced"):
                return (95, track["id"])  # SDH dialogue is still very good

        # Regular subtitles (not forced, not SDH)
        if not disposition.get("forced") and not disposition.get("hearing_impaired"):
            if not any(keyword in title for keyword in ["forced", "signs", "songs", "commentary"]):
                return (90, track["id"])
            else:
                return (70, track["id"])

        # SDH subtitles (complete dialogue with descriptions)
        if not disposition.get("forced") and disposition.get("hearing_impaired"):
            return (60, track["id"])

        # Forced subtitles (incomplete - only foreign parts)
        if disposition.get("forced"):
            return (20, track["id"])

        return (10, track["id"])

    # Try each language in order of preference
    for lang_group in [preferred_languages[i:i+2] for i in range(0, len(preferred_languages), 2)]:
        scored_tracks = [(get_track_score(t, lang_group), t) for t in candidate_tracks]
        scored_tracks.sort(key=lambda item: item[0], reverse=True)

        if scored_tracks and scored_tracks[0][0][0] >= 0:
            best_track = scored_tracks[0][1]
            best_score = scored_tracks[0][0][0]

            title = best_track.get("tags", {}).get("title", "No title")
            language = best_track.get("tags", {}).get("language", "Unknown")
            disposition = best_track.get("disposition", {})

            logger.info(f"Selected subtitle track {best_track['id']} with score {best_score}")
            logger.info(
                f"Track details - Title: '{title}', Language: {language}, "
                f"Forced: {disposition.get('forced', False)}, "
                f"SDH: {disposition.get('hearing_impaired', False)}"
            )
            return best_track

    return None


def is_text_track_standalone(track : dict) -> bool:
    """Standalone version of is_text_track function"""
    codec = (track.get("codec") or "").lower()
    text_codecs = [
        "subrip", "srt", "s_text/utf8", "s_text/ass",
        "substationalpha", "ssa", "ass", "s_text/ssa",
    ]
    return any(tok in codec for tok in text_codecs)


def select_best_track_with_fallback(tracks : list[dict]) -> dict|None:
    """Select best track with English first, French as fallback, then any text track"""
    # Try English first
    english_track = select_best_track_for_translation(tracks, ["eng", "en"])
    if english_track:
        return english_track

    # Fall back to French
    logger.info("No English subtitle track found, trying French as fallback...")
    french_track = select_best_track_for_translation(tracks, ["fre", "fr", "fra", "french"])
    if french_track:
        logger.info("Using French subtitle track as fallback for translation")
        return french_track

    # If no preferred language found, try any text-based subtitle track
    logger.info("No English or French tracks found, looking for any text-based subtitle...")
    text_tracks = [t for t in tracks if is_text_track_standalone(t)]
    if text_tracks:
        # Sort by track ID to get consistent results
        best_text_track = sorted(text_tracks, key=lambda t: t.get("id", 0))[0]
        track_id = best_text_track.get("id")
        title = best_text_track.get("tags", {}).get("title", "No title")
        language = best_text_track.get("tags", {}).get("language", "Unknown")
        logger.info(f"Using track {track_id} as fallback: '{title}' ({language})")
        return best_text_track

    logger.error("No text-based subtitle tracks found in the file")
    return None


def extract_track_with_progress(video_file : Path, subtitle_file : Path, track_id : int, show_progress : bool = True, console=None) -> bool:
    """Extract subtitle from MKV file using mkvextract with optional progress bar"""
    try:
        file_size_mb = video_file.stat().st_size / (1024 * 1024)

        cmd = [
            "mkvextract",
            "--ui-language", "en_US",
            "tracks",
            str(video_file),
            f"{track_id}:{str(subtitle_file)}",
        ]

        if show_progress and console:
            console.print(f"[blue]Extracting subtitle track {track_id} from {video_file.name}...[/blue]")
            console.print(f"[dim]File size: {file_size_mb:.1f} MB[/dim]")

        # Set up command with nice/ionice if available
        ionice_available = shutil.which("ionice") is not None
        is_root = os.geteuid() == 0

        if ionice_available:
            if is_root:
                cmd = ["ionice", "-c", "1", "-n", "4"] + cmd
            else:
                cmd = ["ionice", "-c", "2", "-n", "4"] + cmd

        if is_root:
            cmd = ["nice", "-n", "-10"] + cmd
        else:
            cmd = ["nice", "-n", "10"] + cmd

        start_time = time.time()

        if show_progress and console:
            # Import here to avoid requiring rich if not using progress
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

            # Set up progress display
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )

            with progress:
                task = progress.add_task(f"Extracting subtitle from {video_file.name}", total=100)

                # Use simpler approach: read stderr only for progress
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,  # Ignore stdout
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True,
                )

                last_progress = 0
                stderr_data = []

                # Read stderr line by line for progress updates
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break

                    stderr_data.append(line)

                    # Parse progress from stderr
                    if "Progress:" in line:
                        progress_match = re.search(r"Progress:\s+(\d+)%", line)
                        if progress_match:
                            progress_value = int(progress_match.group(1))
                            if progress_value > last_progress:
                                last_progress = progress_value
                                elapsed = time.time() - start_time
                                rate = file_size_mb / elapsed if elapsed > 0 else 0
                                progress.update(
                                    task,
                                    completed=progress_value,
                                    description=f"Extracting subtitle from {video_file.name} ({rate:.1f} MB/s)"
                                )

                process.wait()
                stderr_text = "".join(stderr_data)

                if process.returncode == 0:
                    elapsed = time.time() - start_time
                    rate = file_size_mb / elapsed if elapsed > 0 else 0
                    progress.update(
                        task,
                        completed=100,
                        description=f"Completed {video_file.name} ({rate:.1f} MB/s avg)"
                    )
        else:
            # No progress display - just run the command
            process = subprocess.run(cmd, capture_output=True, text=True)
            stderr_text = process.stderr

        total_time = time.time() - start_time

        if process.returncode != 0:
            error_message = stderr_text if stderr_text else f"mkvextract exited with code {process.returncode}"
            logger.error(f"mkvextract failed: {error_message}")
            return False

        # Verify file was created and has content
        if not subtitle_file.exists() or subtitle_file.stat().st_size == 0:
            logger.error("Subtitle file was not created or is empty")
            return False

        if show_progress and console:
            avg_rate = file_size_mb / total_time if total_time > 0 else 0
            console.print(f"[green]✓ Subtitles extracted successfully in {total_time:.1f}s ({avg_rate:.1f} MB/s)[/green]")

            if avg_rate < 1.0 and file_size_mb > 100:
                console.print(
                    f"[yellow]⚠ Performance warning: {avg_rate:.1f} MB/s is quite slow. "
                    f"This may indicate I/O bottleneck.[/yellow]"
                )

        return True

    except Exception as e:
        logger.error(f"Failed to extract subtitles with mkvextract: {str(e)}")
        if show_progress and console:
            console.print_exception()
        return False
