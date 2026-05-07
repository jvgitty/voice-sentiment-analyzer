"""Time-windowed analyzer (Slice 7).

Tiles ``[0, audio_duration)`` into fixed-size windows, re-runs the
headline metrics per window, and emits a list of ``WindowMetrics`` for
the time-series view in the result JSON.

The windows tile the audio with no gaps and no overlaps. The last
window may be shorter when ``audio_duration % window_seconds != 0``.
For audio shorter than ``window_seconds``, a single window covering
``[0, audio_duration)`` is produced.

Per-window cost model
---------------------
Each window re-runs the (cheap) AcousticAnalyzer on its audio slice and,
when an EmotionAnalyzer is supplied, the dimensional regression head
only — categorical IEMOCAP classification is intentionally skipped per
window because (a) it is the slower of the two emotion models and (b)
it is not in the headline metrics. The whole-audio prosody result is
reused for every window because a 30s slice's transcript is too sparse
for a meaningful per-window WPM.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vsa.composites import ScoreInputs
from vsa.schema import WindowMetrics

if TYPE_CHECKING:
    from vsa.composites import CompositeScorer
    from vsa.features.acoustic import AcousticAnalyzer
    from vsa.features.emotion import EmotionAnalyzer
    from vsa.schema import ProsodyFeatures, Transcript


class WindowedAnalyzer:
    """Tile audio into fixed-size windows and score headline metrics per
    window for the time-series view.

    The analyzer is constructed cheaply (no model loads). Per-window
    inference re-uses caller-supplied analyzers so the heavy backbones
    are loaded at most once across all windows.
    """

    def __init__(self, window_seconds: float = 30.0) -> None:
        self._window_seconds = float(window_seconds)

    def _tile(self, audio_duration_sec: float) -> list[tuple[float, float]]:
        """Return ``[(start_sec, end_sec), ...]`` tiling
        ``[0, audio_duration_sec)`` with no gaps and no overlaps. The last
        tuple may have ``end_sec - start_sec < window_seconds`` when the
        duration does not divide evenly. For ``audio_duration_sec <=
        window_seconds`` a single window covering the full audio is
        returned."""
        if audio_duration_sec <= 0.0:
            return []
        # Single-window short-audio case: avoid spinning a degenerate
        # zero-length tail when duration <= window.
        if audio_duration_sec <= self._window_seconds:
            return [(0.0, float(audio_duration_sec))]
        # Full-length back-to-back windows for the bulk of the audio.
        tiles: list[tuple[float, float]] = []
        start = 0.0
        while start + self._window_seconds <= audio_duration_sec:
            tiles.append((start, start + self._window_seconds))
            start += self._window_seconds
        # Partial trailing window so coverage extends to the audio end.
        # ``start < audio_duration_sec`` is the only safe condition: float
        # rounding means the equality case may register either as a clean
        # multiple (already handled above) or a sub-microsecond tail; the
        # latter we accept rather than introducing an epsilon.
        if start < audio_duration_sec:
            tiles.append((start, float(audio_duration_sec)))
        return tiles

    @staticmethod
    def _slice_to_tempfile(
        audio_path: Path, start_sec: float, end_sec: float, tmp_dir: Path
    ) -> Path:
        """Write the audio slice ``[start_sec, end_sec)`` to a fresh WAV
        in ``tmp_dir`` and return its path. Uses stdlib ``wave`` so we
        don't pay a librosa load per window for the segmentation step
        itself; the analyzers downstream still load the slice with their
        own readers, but only over a window-sized chunk."""
        with wave.open(str(audio_path), "rb") as src:
            sample_rate = src.getframerate()
            sampwidth = src.getsampwidth()
            n_channels = src.getnchannels()
            start_frame = max(0, int(round(start_sec * sample_rate)))
            end_frame = min(
                src.getnframes(), int(round(end_sec * sample_rate))
            )
            src.setpos(start_frame)
            frames = src.readframes(max(0, end_frame - start_frame))
        # Use a deterministic-ish unique filename inside tmp_dir so
        # parselmouth's path-handling on Windows stays simple (no NamedTemp
        # delete-while-open semantics).
        out_path = tmp_dir / f"win_{int(round(start_sec * 1000)):08d}.wav"
        with wave.open(str(out_path), "wb") as dst:
            dst.setnchannels(n_channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(sample_rate)
            dst.writeframes(frames)
        return out_path

    @staticmethod
    def _audio_duration_sec(audio_path: Path) -> float:
        with wave.open(str(audio_path), "rb") as f:
            return f.getnframes() / float(f.getframerate())

    def analyze(
        self,
        audio_path: Path,
        transcript: "Optional[Transcript]",
        composite_scorer: "CompositeScorer",
        emotion_analyzer: "Optional[EmotionAnalyzer]" = None,
        acoustic_analyzer: "Optional[AcousticAnalyzer]" = None,
        prosody: "Optional[ProsodyFeatures]" = None,
    ) -> list[WindowMetrics]:
        """Tile the audio and produce one ``WindowMetrics`` per tile.

        ``prosody`` is the whole-audio prosody result; it is re-used for
        every window's composite scoring because per-window WPM over a
        30s slice is too noisy to be meaningful. ``transcript`` is unused
        for v1 (kept on the signature for symmetry with future
        word-level windowing). When ``acoustic_analyzer`` or
        ``emotion_analyzer`` is None the corresponding section of the
        WindowMetrics is left at None.
        """
        # Lazy imports here to avoid import-time cycles with vsa.pipeline
        # (which itself imports WindowedAnalyzer).
        from vsa.features.acoustic import AcousticAnalyzer  # noqa: F401

        duration_sec = self._audio_duration_sec(audio_path)
        tiles = self._tile(duration_sec)
        windows: list[WindowMetrics] = []
        # One temp dir for all per-window slice files; cleaned up on exit.
        with tempfile.TemporaryDirectory(prefix="vsa-windows-") as tmp:
            tmp_dir = Path(tmp)
            for start_sec, end_sec in tiles:
                slice_path = self._slice_to_tempfile(
                    audio_path, start_sec, end_sec, tmp_dir
                )

                pitch_mean_hz: Optional[float] = None
                loudness_mean_db: Optional[float] = None
                acoustic = None
                if acoustic_analyzer is not None:
                    try:
                        acoustic = acoustic_analyzer.analyze(slice_path)
                        pitch_mean_hz = float(acoustic.pitch.mean_hz)
                        loudness_mean_db = float(acoustic.loudness.mean_db)
                    except Exception:  # noqa: BLE001 -- partial success
                        acoustic = None

                arousal: Optional[float] = None
                valence: Optional[float] = None
                emotion = None
                if emotion_analyzer is not None:
                    # Run dimensional only — categorical IEMOCAP is too
                    # heavy per window and not in the headline metrics.
                    try:
                        dim = emotion_analyzer._run_dimensional(slice_path)
                        from vsa.schema import EmotionResult as _ER

                        emotion = _ER(dimensional=dim, categorical=None)
                        arousal = float(dim.arousal)
                        valence = float(dim.valence)
                    except Exception:  # noqa: BLE001 -- partial success
                        emotion = None

                # Per-window composite scoring: re-use whole-audio prosody
                # because per-window WPM is too noisy. A whole-scorer
                # crash on any one window leaves that window's composite
                # fields at None but does not abort the windows pass.
                confidence: Optional[float] = None
                engagement: Optional[float] = None
                calmness: Optional[float] = None
                try:
                    scores = composite_scorer.score(
                        ScoreInputs(
                            acoustic=acoustic,
                            prosody=prosody,
                            emotion=emotion,
                            audio_duration_seconds=end_sec - start_sec,
                        )
                    )
                    confidence = scores.confidence
                    engagement = scores.engagement
                    calmness = scores.calmness
                except Exception:  # noqa: BLE001 -- partial success
                    pass

                windows.append(
                    WindowMetrics(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        pitch_mean_hz=pitch_mean_hz,
                        loudness_mean_db=loudness_mean_db,
                        arousal=arousal,
                        valence=valence,
                        confidence=confidence,
                        engagement=engagement,
                        calmness=calmness,
                    )
                )
        return windows
