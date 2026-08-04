"""Which Indian languages this pipeline can read, and with which model.

Indian-language OCR is a *script* routing problem, not a language one. Hindi,
Marathi, Nepali and Sanskrit are four languages sharing one Devanagari
recogniser, so covering thirteen-odd languages takes six models, not thirteen.

Everything here is derived from the model registry actually shipped by the
installed PaddleOCR (`paddlex.inference.utils.official_models`), not from the
list of languages PaddleOCR's marketing mentions. The gap between those two is
the whole point of UNSUPPORTED_SCRIPTS below.
"""

from __future__ import annotations

# Script -> recognition model. Verified present in PaddleOCR 3.7's registry.
SCRIPT_MODELS: dict[str, str] = {
    "latin": "PP-OCRv6_medium_rec",              # English + romanised text
    "devanagari": "devanagari_PP-OCRv5_mobile_rec",
    "telugu": "te_PP-OCRv5_mobile_rec",
    "tamil": "ta_PP-OCRv5_mobile_rec",
    "kannada": "ka_PP-OCRv3_mobile_rec",         # v3 -- see KNOWN_WEAK below
    "arabic": "arabic_PP-OCRv5_mobile_rec",      # Urdu, Kashmiri, Sindhi
}

# Scripts whose only model is a generation behind the rest. Telugu gained ~8
# accuracy points moving v3 -> v5, so treat Kannada output as correspondingly
# weaker rather than assuming parity with the v5 scripts.
KNOWN_WEAK = {"kannada"}

# ISO-639 language code -> script. Multiple languages per script is the norm.
LANGUAGE_SCRIPTS: dict[str, str] = {
    # Devanagari
    "hi": "devanagari",     # Hindi
    "mr": "devanagari",     # Marathi
    "ne": "devanagari",     # Nepali
    "sa": "devanagari",     # Sanskrit
    "mai": "devanagari",    # Maithili
    "gom": "devanagari",    # Konkani
    "doi": "devanagari",    # Dogri
    "brx": "devanagari",    # Bodo
    "bho": "devanagari",    # Bhojpuri
    # Single-script languages
    "te": "telugu",
    "ta": "tamil",
    "kn": "kannada",
    # Perso-Arabic
    "ur": "arabic",         # Urdu
    "ks": "arabic",         # Kashmiri
    "sd": "arabic",         # Sindhi
    # Latin
    "en": "latin",
}

# Indian scripts with NO PaddleOCR recognition model at any version. These fail
# loudly rather than silently falling back to a model for a different script,
# which would return confident nonsense.
UNSUPPORTED_SCRIPTS: dict[str, str] = {
    "bn": "Bengali",
    "as": "Assamese",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi (Gurmukhi)",
    "or": "Odia",
    "sat": "Santali (Ol Chiki)",
    "mni": "Manipuri (Meitei Mayek)",
}

# Accept a few common aliases so callers are not forced to remember codes.
ALIASES: dict[str, str] = {
    "english": "en", "hindi": "hi", "marathi": "mr", "nepali": "ne",
    "sanskrit": "sa", "telugu": "te", "tamil": "ta", "kannada": "kn",
    "urdu": "ur", "konkani": "gom", "bengali": "bn", "malayalam": "ml",
    "gujarati": "gu", "punjabi": "pa", "odia": "or", "assamese": "as",
    "oriya": "or", "sindhi": "sd", "kashmiri": "ks",
}


class UnsupportedLanguage(ValueError):
    """Raised for an Indian language whose script PaddleOCR cannot read."""


def normalise(code: str) -> str:
    code = code.strip().lower()
    return ALIASES.get(code, code)


def resolve_script(code: str) -> str:
    """Map a language code (or script name) to a script with a model.

    Raises UnsupportedLanguage with an actionable message rather than guessing.
    Silently substituting a different script's model is the failure worth
    avoiding here: it does not error, it returns confident garbage.
    """
    code = normalise(code)

    if code in SCRIPT_MODELS:      # already a script name
        return code
    if code in LANGUAGE_SCRIPTS:
        return LANGUAGE_SCRIPTS[code]
    if code in UNSUPPORTED_SCRIPTS:
        raise UnsupportedLanguage(
            f"{UNSUPPORTED_SCRIPTS[code]} ({code}) has no PaddleOCR recognition "
            f"model at any version, so this pipeline cannot read it.\n"
            f"Unsupported: {', '.join(sorted(UNSUPPORTED_SCRIPTS.values()))}.\n"
            f"Supported: {', '.join(sorted(supported_languages()))}."
        )
    raise UnsupportedLanguage(
        f"Unknown language or script {code!r}. "
        f"Supported: {', '.join(sorted(supported_languages()))}."
    )


def resolve_models(codes: list[str]) -> list[tuple[str, str]]:
    """Language codes -> ordered unique (script, model) pairs.

    Deduplicates by script: `--lang hi+mr+ne` is three languages sharing one
    Devanagari model, so it costs one recognition pass, not three.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for code in codes:
        script = resolve_script(code)
        if script not in seen:
            seen.add(script)
            pairs.append((script, SCRIPT_MODELS[script]))
    return pairs


def supported_languages() -> set[str]:
    return set(LANGUAGE_SCRIPTS) | set(SCRIPT_MODELS)


# All scripts to probe during auto-detection, ordered by rough regional
# frequency so the log output reads naturally.  The order does not affect
# correctness — every script is always tried.
AUTO_DETECT_SCRIPTS: list[str] = list(SCRIPT_MODELS.keys())


