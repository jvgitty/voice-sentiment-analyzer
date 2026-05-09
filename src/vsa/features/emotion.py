r"""EmotionAnalyzer: dimensional + categorical emotion recognition.

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

Windows + Python 3.13 quirk
---------------------------
SpeechBrain 1.1.0 ships a lazy-import shim
(:class:`speechbrain.utils.importutils.LazyModule`) whose "is the importer
inspect.py?" guard is hardcoded to ``"/inspect.py"`` and silently misses
the Windows ``\inspect.py`` path. CPython 3.13's ``inspect.getmodule`` then
chases ``__file__`` on every lazily-deferred sibling module
(``k2_fsa``, ``nlp``, ``numba.transducer_loss`` …), each of which lacks an
optional dep — the cascade surfaces as a misleading
"Please install transformers" error from the huggingface package init.
We patch ``LazyModule.ensure_module`` to normalise path separators before
the endswith check; see :func:`_patch_speechbrain_lazy_imports`. This is
called at most once, on the first categorical-model load attempt.
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
# Order matches the IEMOCAP label_encoder shipped with the model:
# neu=0, ang=1, hap=2, sad=3. Used to align softmax outputs to keys.
_IEMOCAP_INDEX_ORDER = ("neu", "ang", "hap", "sad")

# All audio is resampled to this rate by SpeechBrain's classify_file path.
# audeering's wav2vec2 backbone is also trained at 16 kHz.
_TARGET_SAMPLE_RATE = 16000


_speechbrain_patched = False


def _patch_speechbrain_lazy_imports() -> None:
    """Work around a Windows-only path-separator bug in
    ``speechbrain.utils.importutils.LazyModule.ensure_module``. Idempotent.

    The shipped check ``filename.endswith("/inspect.py")`` never matches on
    Windows, where ``inspect.getframeinfo`` reports backslash paths. Without
    this patch, importing any submodule of ``speechbrain.integrations``
    crashes when CPython 3.13's ``inspect.getmodule`` walks ``__file__`` on
    sibling LazyModules whose targets aren't installed.
    """
    global _speechbrain_patched
    if _speechbrain_patched:
        return

    import importlib
    import inspect
    import sys

    from speechbrain.utils import importutils as _iu

    _orig_ensure = _iu.LazyModule.ensure_module

    def _patched_ensure(self: Any, stacklevel: int = 0) -> Any:
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            pass
        if importer_frame is not None:
            normalized = importer_frame.filename.replace("\\", "/")
            if normalized.endswith("/inspect.py"):
                raise AttributeError()
        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(f"Lazy import of {self!r} failed") from e
        return self.lazy_module

    _iu.LazyModule.ensure_module = _patched_ensure  # type: ignore[method-assign]
    _speechbrain_patched = True


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

    # -- lazy loaders ------------------------------------------------------

    def _load_dimensional(self) -> tuple[Any, Any]:
        """Load the audeering regression model + processor on first use.

        Per the model card (``audeering/wav2vec2-large-robust-12-ft-emotion-
        msp-dim``), the architecture is a custom 3-output regression head
        on top of ``Wav2Vec2Model`` — transformers' built-in classification
        heads don't fit. We follow the model card recipe verbatim with a
        local subclass of ``Wav2Vec2PreTrainedModel``.

        ``id2label`` on this checkpoint is ``{0: arousal, 1: dominance,
        2: valence}`` — note dominance and valence are NOT in alphabetical
        or schema order, hence the explicit indexing in :meth:`analyze`.
        """
        if self._dimensional_model is None or self._dimensional_processor is None:
            import torch
            import torch.nn as nn
            from transformers import Wav2Vec2Processor
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2Model,
                Wav2Vec2PreTrainedModel,
            )

            class _RegressionHead(nn.Module):
                def __init__(self, config: Any) -> None:
                    super().__init__()
                    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                    self.dropout = nn.Dropout(config.final_dropout)
                    self.out_proj = nn.Linear(
                        config.hidden_size, config.num_labels
                    )

                def forward(self, features: Any, **kwargs: Any) -> Any:
                    x = features
                    x = self.dropout(x)
                    x = self.dense(x)
                    x = torch.tanh(x)
                    x = self.dropout(x)
                    x = self.out_proj(x)
                    return x

            class _AudeeringEmotionModel(Wav2Vec2PreTrainedModel):
                def __init__(self, config: Any) -> None:
                    super().__init__(config)
                    self.config = config
                    self.wav2vec2 = Wav2Vec2Model(config)
                    self.classifier = _RegressionHead(config)
                    self.init_weights()

                def forward(self, input_values: Any) -> Any:
                    outputs = self.wav2vec2(input_values)
                    hidden_states = outputs[0]
                    hidden_states = torch.mean(hidden_states, dim=1)
                    return hidden_states, self.classifier(hidden_states)

            self._dimensional_processor = Wav2Vec2Processor.from_pretrained(
                DIMENSIONAL_MODEL_NAME
            )
            self._dimensional_model = _AudeeringEmotionModel.from_pretrained(
                DIMENSIONAL_MODEL_NAME
            )
            self._dimensional_model.eval()
        return self._dimensional_model, self._dimensional_processor

    def _load_categorical(self) -> Any:
        """Load SpeechBrain's IEMOCAP classifier on first use.

        Uses ``foreign_class`` because the checkpoint ships a custom
        ``CustomEncoderWav2vec2Classifier`` interface in
        ``custom_interface.py``. We force ``LocalStrategy.COPY`` because
        Windows non-admin processes can't create the symlinks SpeechBrain
        uses by default. ``savedir`` lives next to the HF hub cache.
        """
        if self._categorical_classifier is None:
            _patch_speechbrain_lazy_imports()
            from huggingface_hub import constants as _hf_constants
            from speechbrain.inference.interfaces import foreign_class
            from speechbrain.utils.fetching import LocalStrategy

            savedir = (
                Path(_hf_constants.HF_HUB_CACHE) / "speechbrain-iemocap"
            )
            self._categorical_classifier = foreign_class(
                source=CATEGORICAL_MODEL_NAME,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=str(savedir),
                local_strategy=LocalStrategy.COPY,
            )
        return self._categorical_classifier

    # -- inference helpers -------------------------------------------------

    def _run_dimensional(self, audio_path: Path) -> DimensionalEmotion:
        import librosa
        import numpy as np
        import torch

        model, processor = self._load_dimensional()

        # Load mono and resample to the model's training rate.
        signal, _ = librosa.load(
            str(audio_path), sr=_TARGET_SAMPLE_RATE, mono=True
        )
        signal = np.asarray(signal, dtype=np.float32)

        proc_out = processor(signal, sampling_rate=_TARGET_SAMPLE_RATE)
        input_values = torch.from_numpy(
            np.asarray(proc_out["input_values"][0])
        ).reshape(1, -1)

        with torch.no_grad():
            logits = model(input_values)[1]
        # id2label on this checkpoint: 0=arousal, 1=dominance, 2=valence.
        values = logits[0].cpu().numpy().tolist()
        # Audeering's outputs are approximately in [0, 1]; clip to be safe
        # against tiny over/undershoots that would fail downstream
        # composite formulas expecting strict [0, 1].
        clipped = [max(0.0, min(1.0, float(v))) for v in values]
        return DimensionalEmotion(
            model=DIMENSIONAL_MODEL_NAME,
            arousal=clipped[0],
            dominance=clipped[1],
            valence=clipped[2],
        )

    def _run_categorical(self, audio_path: Path) -> CategoricalEmotion:
        import librosa
        import numpy as np
        import torch

        classifier = self._load_categorical()
        # We deliberately bypass classify_file: on Windows, SpeechBrain's
        # internal load_audio prepends a data_folder to absolute paths
        # ("D:\repo\C:\Users\...sine.wav") and crashes libsndfile. Loading
        # the waveform ourselves and feeding classify_batch sidesteps the
        # bug, costs us nothing (librosa is already a project dep), and
        # makes the resampling explicit at 16 kHz mono.
        signal, _ = librosa.load(
            str(audio_path), sr=_TARGET_SAMPLE_RATE, mono=True
        )
        wav = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0)
        out_prob, _score, _index, _text_lab = classifier.classify_batch(wav)
        # out_prob is a 1xN tensor of softmax probabilities aligned to
        # _IEMOCAP_INDEX_ORDER. Build the human-readable scores dict.
        probs = out_prob[0].detach().cpu().numpy().tolist()
        scores = {
            _IEMOCAP_LABEL_MAP[code]: float(probs[i])
            for i, code in enumerate(_IEMOCAP_INDEX_ORDER)
        }
        # Pick the argmax label rather than relying on classify_file's
        # text_lab so we keep label/scores consistent if anyone ever
        # post-processes the dict.
        label = max(scores, key=lambda k: scores[k])
        return CategoricalEmotion(
            model=CATEGORICAL_MODEL_NAME,
            label=label,
            scores=scores,
        )

    # -- memory-eviction helpers ------------------------------------------

    def release_categorical(self) -> None:
        """Drop the loaded SpeechBrain IEMOCAP classifier so its wav2vec2
        backbone (~1.3 GB) becomes eligible for GC.

        The windowed pass intentionally skips the categorical model (see
        :class:`vsa.windowed.WindowedAnalyzer`), so once whole-audio
        emotion has run, the classifier is dead weight for the rest of
        the request. The next :meth:`_run_categorical` call reloads
        lazily.
        """
        self._categorical_classifier = None

    # -- public API --------------------------------------------------------

    def analyze(self, audio_path: Path) -> EmotionResult:
        """Run both emotion models on ``audio_path`` and return their joined
        result. Each underlying model's failure is contained: if one
        raises, that section is ``None`` and the other is still populated.

        Errors are NOT recorded on the returned object — that's the
        Pipeline's job. The caller (Pipeline) wraps the entire ``analyze``
        call in another try/except for the analyzer-level partial-success
        contract.
        """
        try:
            dimensional: DimensionalEmotion | None = self._run_dimensional(
                audio_path
            )
        except Exception:  # noqa: BLE001 -- partial-success contract
            dimensional = None

        try:
            categorical: CategoricalEmotion | None = self._run_categorical(
                audio_path
            )
        except Exception:  # noqa: BLE001 -- partial-success contract
            categorical = None

        return EmotionResult(dimensional=dimensional, categorical=categorical)
