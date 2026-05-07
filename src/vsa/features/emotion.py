"""EmotionAnalyzer: dimensional + categorical emotion recognition.

Wraps two independent open-source models:

* **Dimensional**: ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``
  — predicts continuous arousal / valence / dominance via a custom
  regression head on Wav2Vec2. Loaded with a tiny custom subclass per the
  model card (transformers' standard heads don't fit a 3-output regression
  head); see :class:`_AudeeringEmotionModel`.

* **Categorical**: ``speechbrain/emotion-recognition-wav2vec2-IEMOCAP`` —
  classifies into IEMOCAP's 4-class label set ({neu, ang, hap, sad}, mapped
  here to {neutral, angry, happy, sad}). Loaded via SpeechBrain's
  ``foreign_class`` recipe per the model card.

Both models are loaded lazily on the first :meth:`analyze` call and cached
for the lifetime of the analyzer. Failures inside one model do not affect
the other — see :meth:`analyze` for the per-model try/except.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vsa.schema import (
    CategoricalEmotion,
    DimensionalEmotion,
    EmotionResult,
)

DIMENSIONAL_MODEL_NAME = (
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
)
CATEGORICAL_MODEL_NAME = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

# IEMOCAP's 4-class label encoder ships short codes; project-wide we expose
# the human-readable forms used in the schema (and that downstream
# composite formulas key off of).
_IEMOCAP_LABEL_MAP = {
    "neu": "neutral",
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
}


class EmotionAnalyzer:
    """Run dimensional and categorical emotion models on a wav file.

    Construction is cheap and side-effect free. Both backbones are loaded
    on the first :meth:`analyze` call and cached. The two models fail
    independently: if dimensional inference raises, ``result.dimensional``
    is ``None`` while ``result.categorical`` is still populated, and vice
    versa.
    """

    def __init__(self) -> None:
        # Dimensional regression model + its feature extractor.
        self._dimensional_model: Any | None = None
        self._dimensional_processor: Any | None = None
        # SpeechBrain's IEMOCAP classifier (loaded via foreign_class).
        self._categorical_classifier: Any | None = None
