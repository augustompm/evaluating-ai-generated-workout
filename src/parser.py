"""SVM-based exercise plan parser with field extraction."""

import re
import pickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from scipy.sparse import hstack, csr_matrix


@dataclass
class Exercise:
    name: str
    intensity_label: str | None = None
    duration_min: float | None = None
    days_per_week: int | None = None
    raw_text: str = ""


@dataclass
class ParsedPlan:
    exercises: list[Exercise] = field(default_factory=list)
    mentions_clearance: bool = False
    mentions_warmup: bool = False
    mentions_cooldown: bool = False
    mentions_progression: bool = False
    total_days: int = 0
    parse_success: bool = True
    raw_text: str = ""
    strength_days: int = 0


# ─── ML Line Classification ────────────────────────────────────────────

LABEL_HEADER = 0
LABEL_EXERCISE = 1
LABEL_REST = 2
LABEL_SAFETY = 3
LABEL_NOTE = 4

_EXERCISE_VOCAB = {
    "walking", "running", "jogging", "cycling", "swimming", "stretching",
    "yoga", "pilates", "squat", "lunge", "push-up", "pushup", "plank",
    "burpee", "jumping", "hiking", "dancing", "rowing", "elliptical",
    "deadlift", "bench", "curl", "press", "pull-up", "pullup",
    "resistance", "dumbbell", "barbell", "kettlebell", "band",
    "tai chi", "aerobics", "sprint", "hiit", "tabata", "circuit",
    "step-up", "calf raise", "leg press", "leg curl", "lat pulldown",
    "cool-down", "warm-up", "cooldown", "warmup",
}
_DAY_NAMES = {"monday", "tuesday", "wednesday", "thursday", "friday",
              "saturday", "sunday"}
_INTENSITY_WORDS = {"light", "moderate", "vigorous", "easy", "hard",
                    "intense", "gentle", "brisk"}
_SAFETY_WORDS = {"doctor", "physician", "medical", "consult",
                 "blood pressure", "glucose", "stop if", "chest pain",
                 "dizziness", "medication", "clearance"}

_classifier_cache = None


def _extract_ml_features(line: str) -> dict:
    """Extract structural features for ML classification (22 features)."""
    t = line.strip()
    tl = t.lower()
    words = tl.split()
    return {
        "has_pipe": int("|" in t),
        "starts_number": int(bool(re.match(r"^\d+[\.\)]\s", t))),
        "starts_bullet": int(t.startswith(("-", "*", "\u2022"))),
        "starts_bold": int(t.startswith("**")),
        "starts_hash": int(t.startswith("#")),
        "has_colon": int(":" in t),
        "char_count": min(len(t), 300),
        "word_count": min(len(words), 50),
        "is_short": int(len(words) < 4),
        "is_long": int(len(words) > 25),
        "has_duration": int(bool(re.search(r"\d+\s*min|hour", tl))),
        "has_intensity": int(any(w in tl for w in _INTENSITY_WORDS)),
        "has_exercise_word": int(any(w in tl for w in _EXERCISE_VOCAB)),
        "has_day_name": int(any(d in tl for d in _DAY_NAMES)),
        "has_number": int(bool(re.search(r"\d", t))),
        "has_structured_format": int(bool(re.search(r"\|\s*intensity:", tl))),
        "has_sets_reps": int(bool(
            re.search(r"\d+\s*(?:sets?|reps?|x\d)", tl))),
        "has_safety_word": int(any(w in tl for w in _SAFETY_WORDS)),
        "starts_note_word": int(any(
            tl.startswith(w) for w in
            ["note", "important", "tip", "remember", "ensure",
             "always", "drink", "aim", "consider", "listen"]
        )),
        "is_header_like": int(bool(
            re.match(r"^(\*\*)?[A-Z][a-z]+day\s*:?\s*(\*\*)?$", t) or
            re.match(r"^(\*\*)?(day\s+\d+|week\s+\d+)\s*:?\s*(\*\*)?$",
                     t, re.I)
        )),
        "has_rest_word": int(any(
            w in tl for w in ["rest day", "rest", "recovery day"])),
        "pct_uppercase": sum(1 for c in t if c.isupper()) / max(len(t), 1),
    }


def _load_classifier():
    """Load trained SVM line classifier from disk."""
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache
    model_path = (Path(__file__).parent.parent
                  / "models" / "line_classifier.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"ML model not found: {model_path}\n"
            f"Run: python src/train_classifier.py")
    with open(model_path, "rb") as f:
        _classifier_cache = pickle.load(f)
    return _classifier_cache


def _classify_lines(lines: list[str]) -> list[int]:
    """Classify lines using trained SVM model.

    Returns list of labels: 0=HEADER, 1=EXERCISE, 2=REST, 3=SAFETY, 4=NOTE
    """
    data = _load_classifier()
    struct_dicts = [_extract_ml_features(line) for line in lines]
    feat_keys = list(struct_dicts[0].keys())
    struct_array = np.array(
        [[d[k] for k in feat_keys] for d in struct_dicts])
    struct_sparse = csr_matrix(struct_array)
    tfidf_w = data["tfidf_word"].transform(lines)
    tfidf_c = data["tfidf_char"].transform(lines)
    X = hstack([struct_sparse, tfidf_w, tfidf_c])
    predictions = data["classifier"].predict(X)
    return list(np.asarray(predictions).ravel())


# ─── Exercise Field Extraction ──────────────────────────────────────────

def _clean_name(raw: str) -> str:
    """Clean exercise name: strip markdown bold, extra whitespace, parentheticals."""
    name = raw.strip()
    name = name.replace("**", "")
    name = name.strip("*").strip()
    m = re.match(r"^([A-Za-z][A-Za-z0-9\s\-/:]+?)(?:\s*\(|\s+or\s+)", name)
    if m and len(m.group(1).strip()) >= 3:
        name = m.group(1).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.rstrip(":- ").strip()
    return name


def _resolve_generic_name(name: str, line: str) -> str:
    """If name is generic (e.g., 'Exercise 1'), try to find the real name."""
    m_inline = re.match(
        r"^(?:Exercise|Activity|Workout)\s*\d*\s*:\s*(.+)", name, re.I)
    if m_inline:
        candidate = m_inline.group(1).strip().rstrip(":- ")
        if len(candidate) >= 3 and candidate.lower() not in ("intensity", "duration"):
            return candidate

    if not re.match(r"^(Exercise|Activity|Workout)\s*\d*$", name, re.I):
        return name

    m = re.search(
        r"(?:Exercise|Activity|Workout)\s*\d*\s*[:*\-]+\s*\**"
        r"([A-Za-z][A-Za-z\s\-/()]{2,50}?)\s*(?:\(|Intensity|Duration|\||$)",
        line, re.I)
    if m:
        candidate = _clean_name(m.group(1))
        if len(candidate) >= 3 and candidate.lower() not in ("intensity", "duration"):
            return candidate

    return name


def _extract_name(line: str) -> str | None:
    """Extract exercise name from an exercise line."""
    text = line.strip()

    if "|" in text:
        m = re.match(r"\s*(?:\d+\.?|[-\u2022*])\s*(.+?)\s*\|", text)
        if m:
            name = _clean_name(m.group(1))
            if len(name) >= 2:
                return _resolve_generic_name(name, text)

    m = re.match(r"^\d+\.?\s*\**([A-Za-z][^|:\n]{2,50}?)\**\s*[:–-]", text)
    if m:
        name = _clean_name(m.group(1))
        return _resolve_generic_name(name, text)

    m = re.match(r"^[-\u2022*]\s*\**([A-Za-z][^|:\n]{2,50}?)\**\s*[:–-]",
                 text)
    if m:
        name = _clean_name(m.group(1))
        return _resolve_generic_name(name, text)

    m = re.match(r"^\*\*([A-Za-z][^*]{2,50}?)\*\*", text)
    if m:
        name = _clean_name(m.group(1))
        return _resolve_generic_name(name, text)

    m = re.match(r"^\d+\.?\s*\**([A-Za-z][A-Za-z\s\-/]{2,40}?)\**\s*\(",
                 text)
    if m:
        name = _clean_name(m.group(1))
        return _resolve_generic_name(name, text)

    return None


def _extract_intensity(text: str) -> str | None:
    """Extract intensity from exercise text."""
    m = re.search(r"[Ii]ntensity:\s*(.+?)(?:\s*\||$)", text)
    if m:
        raw = m.group(1).strip()
    else:
        tl = text.lower()
        for level in ["vigorous", "moderate", "light"]:
            if level in tl:
                return level
        return None

    raw_lower = raw.lower().strip()

    m2 = re.match(r"([\w\s]+?)\s+to\s+([\w\s]+)", raw_lower)
    if m2:
        p1 = m2.group(1).strip().replace(" ", "-")
        p2 = m2.group(2).strip().replace(" ", "-")
        return f"{p1}-to-{p2}"

    if raw_lower.startswith("very "):
        return raw_lower.replace(" ", "-")

    for level in ["vigorous", "moderate", "light"]:
        if level in raw_lower:
            return level

    return raw_lower if raw_lower else None


def _extract_duration(text: str) -> float | None:
    """Extract duration in minutes from exercise text."""
    m = re.search(
        r"[Dd]uration:\s*(\d+)\s*[-\u2013]\s*(\d+)\s*min(?:ute)?s?", text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2

    m = re.search(
        r"[Dd]uration:\s*(\d+)\s*min\w*\s+to\s+an?\s*hour", text)
    if m:
        return (float(m.group(1)) + 60) / 2

    m = re.search(
        r"[Dd]uration:\s*(\d+)\s*sets?\s+(?:of\s+)?"
        r"(\d+)(?:\s*[-\u2013]\s*(\d+))?\s*rep", text)
    if m:
        sets = int(m.group(1))
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        return round(sets * ((lo + hi) / 2 * 4 / 60 + 1.0), 1)

    m = re.search(
        r"[Dd]uration:\s*(\d+)\s*sets?\s*,?\s*hold\s+(?:for\s+)?"
        r"(\d+)\s*sec", text)
    if m:
        return round(int(m.group(1)) * int(m.group(2)) / 60, 1)

    m = re.search(
        r"[Dd]uration:\s*(\d+)\s*sets?\s+(?:of\s+)?(?:as many|AMRAP)",
        text, re.I)
    if m:
        return round(int(m.group(1)) * (12 * 4 / 60 + 1.0), 1)

    m = re.search(r"[Dd]uration:\s*(\d+)\s*min(?:ute)?s?", text)
    if m:
        return float(m.group(1))

    m = re.search(
        r"[Ss]ets:\s*(\d+)\s*,?\s*[Rr]eps?:\s*(\d+)"
        r"(?:\s*[-\u2013]\s*(\d+))?", text)
    if m:
        sets = int(m.group(1))
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        return round(sets * ((lo + hi) / 2 * 4 / 60 + 1.0), 1)

    m = re.search(r"[Ss]ets?/?[Rr]eps?:\s*(\d+)\s*[xX×]\s*(\d+)", text)
    if not m:
        m = re.search(r"(\d+)\s*[xX×]\s*(\d+)(?:\s*[-\u2013]\s*(\d+))?\s*rep", text)
    if m:
        sets = int(m.group(1))
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.lastindex >= 3 and m.group(3) else lo
        return round(sets * ((lo + hi) / 2 * 4 / 60 + 1.0), 1)

    m = re.search(
        r"(\d+)\s*sets?\s+(?:of\s+)?(\d+)(?:\s*[-\u2013]\s*(\d+))?\s*rep",
        text, re.I)
    if m:
        sets = int(m.group(1))
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        return round(sets * ((lo + hi) / 2 * 4 / 60 + 1.0), 1)

    m = re.search(r"[Hh]old\s+(?:for\s+)?(\d+)\s*sec", text)
    if m:
        return round(float(m.group(1)) / 60, 1)

    m = re.search(
        r"(\d+)\s*[-\u2013]\s*(\d+)\s*min(?:ute)?s?", text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2

    m = re.search(r"(\d+)\s*min(?:ute)?s?", text)
    if m:
        val = float(m.group(1))
        if 1 <= val <= 180:
            return val

    m = re.search(r"(\d+(?:\.\d+)?)\s*hours?", text, re.I)
    if m:
        return float(m.group(1)) * 60

    return None


def _extract_exercise(line: str) -> Exercise | None:
    """Extract exercise fields from a single EXERCISE-classified line."""
    text = line.strip()
    if len(text) < 5:
        return None

    cleaned = re.sub(r"[\d.*#\-\u2022\s]", "", text.lower())
    if cleaned in ("rest", "restday", "restdayactiverecovery"):
        return None

    name = _extract_name(text)
    if not name or len(name) < 2:
        return None
    name_lower = name.lower()
    if name_lower in ("rest", "rest day") or name_lower.startswith("rest day"):
        return None
    _NON_EXERCISE_PREFIXES = [
        "use a ", "use the ", "start with", "focus on", "include ",
        "aim for", "listen to", "drink ", "keep ", "note",
        "priorit", "adjust ", "ensure ", "maintain ", "incorporate ",
    ]
    if any(name_lower.startswith(p) for p in _NON_EXERCISE_PREFIXES):
        return None

    intensity = _extract_intensity(text)
    duration = _extract_duration(text)

    return Exercise(
        name=name,
        intensity_label=intensity,
        duration_min=duration,
        raw_text=text,
    )


# ─── Keyword Detection ──────────────────────────────────────────────────

CLEARANCE_KEYWORDS = [
    "medical clearance", "doctor", "physician", "healthcare provider",
    "consult", "check with", "medical professional", "medical advice",
    "seek medical", "talk to your doctor", "cleared by",
    "approval from", "before starting", "professional guidance",
]
WARMUP_KEYWORDS = ["warm-up", "warm up", "warmup", "warming up"]
COOLDOWN_KEYWORDS = ["cool-down", "cool down", "cooldown", "cooling down"]
PROGRESSION_KEYWORDS = [
    "gradually", "progress", "progressively", "increase gradually",
    "build up", "start slow", "start light", "work your way up",
    "begin with", "ease into", "incremental", "over time",
    "as you get stronger", "as fitness improves",
]
INTENSITY_KEYWORDS = {
    "light": ["light", "low intensity", "easy", "gentle"],
    "moderate": ["moderate", "medium intensity", "brisk"],
    "vigorous": ["vigorous", "high intensity", "intense", "hard",
                 "challenging"],
}
STRENGTH_KEYWORDS = [
    "strength", "resistance", "weight", "dumbbell", "barbell",
    "kettlebell", "bodyweight exercise", "push-up", "pushup", "push up",
    "pull-up", "pullup", "pull up", "squat", "lunge", "deadlift",
    "bench press", "chest press", "shoulder press", "bicep curl",
    "tricep", "lat pulldown", "plank", "core exercise",
]
HIGH_IMPACT_KEYWORDS = [
    "running", "jogging", "jumping", "high-impact", "plyometric",
    "jump rope", "skipping rope", "burpee", "box jump", "sprinting",
]
GLUCOSE_KEYWORDS = [
    "glucose", "blood sugar", "glycemia", "blood glucose",
    "sugar level", "glycemic", "hypoglycemia", "hyperglycemia",
]
BP_KEYWORDS = [
    "blood pressure", "bp monitoring", "hypertension",
    "monitor your blood pressure", "check your blood pressure",
]
G4_KEYWORDS = {
    "P3": ["avoid holding breath", "valsalva", "blood pressure"],
    "P4": ["glucose", "snack", "carbohydrate", "hypoglycemia",
            "blood sugar", "glycemia"],
    "P5": ["low impact", "low-impact", "avoid", "knee", "joint", "pain",
            "non-weight bearing"],
    "P6": ["heart rate", "chest pain", "shortness of breath", "stop if",
            "warning sign", "symptom", "discontinue"],
    "P8": ["blood pressure", "gradually", "start light", "start slow",
            "ease into"],
}


def _has_keyword(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _count_days_from_labels(lines: list[str], labels: list[int]) -> int:
    """Count distinct days from HEADER-classified lines."""
    days_found = set()
    day_names = ["monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"]
    for line, label in zip(lines, labels):
        if label != LABEL_HEADER:
            continue
        tl = line.lower()
        for d in day_names:
            if d in tl:
                days_found.add(d)
        for m in re.finditer(r"day\s+(\d+)", tl):
            days_found.add(f"day_{m.group(1)}")
    return len(days_found)


def _count_strength_days(text: str) -> int:
    """Count days that include strength/resistance training."""
    text_lower = text.lower()
    day_splits = re.split(
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|day\s+\d+)",
        text_lower, flags=re.IGNORECASE)
    count = 0
    for section in day_splits:
        if any(kw in section for kw in STRENGTH_KEYWORDS):
            count += 1
    return count


# ─── Main Parser ────────────────────────────────────────────────────────

def parse_plan(response_text: str) -> ParsedPlan:
    """Parse an LLM-generated exercise plan into structured data."""
    if not response_text or not response_text.strip():
        return ParsedPlan(parse_success=False, raw_text=response_text or "")

    text = response_text.strip()
    all_lines = [line.strip() for line in text.split("\n")
                 if line.strip()]
    if not all_lines:
        return ParsedPlan(parse_success=False, raw_text=text)

    labels = _classify_lines(all_lines)

    exercises = []
    for line, label in zip(all_lines, labels):
        if label != LABEL_EXERCISE:
            continue
        ex = _extract_exercise(line)
        if ex is not None:
            exercises.append(ex)

    total_days = _count_days_from_labels(all_lines, labels)
    if total_days == 0:
        day_names = ["monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday"]
        text_lower = text.lower()
        total_days = sum(1 for d in day_names if d in text_lower)
        if total_days == 0:
            total_days = len(set(re.findall(r"day\s+(\d+)", text_lower)))

    return ParsedPlan(
        exercises=exercises,
        mentions_clearance=_has_keyword(text, CLEARANCE_KEYWORDS),
        mentions_warmup=_has_keyword(text, WARMUP_KEYWORDS),
        mentions_cooldown=_has_keyword(text, COOLDOWN_KEYWORDS),
        mentions_progression=_has_keyword(text, PROGRESSION_KEYWORDS),
        total_days=total_days,
        parse_success=len(exercises) > 0,
        raw_text=text,
        strength_days=_count_strength_days(text),
    )


# ─── Helper functions used by rule_engine.py ────────────────────────────

def check_has_specific_exercises(parsed: ParsedPlan) -> bool:
    """Check if the plan names at least one specific exercise (C4)."""
    generic = {"exercise", "activity", "workout", "training", "movement"}
    for ex in parsed.exercises:
        if not set(ex.name.lower().split()).issubset(generic):
            return True
    return len(parsed.exercises) > 0


def check_high_impact(parsed: ParsedPlan) -> bool:
    """Check if the plan prescribes high-impact exercises (S3 for OA)."""
    for ex in parsed.exercises:
        name_lower = ex.name.lower()
        for kw in HIGH_IMPACT_KEYWORDS:
            if kw.lower() in name_lower:
                return True
    return False


def check_glucose_mention(parsed: ParsedPlan) -> bool:
    """Check if the plan mentions glucose monitoring (S4)."""
    return _has_keyword(parsed.raw_text, GLUCOSE_KEYWORDS)


def check_bp_mention(parsed: ParsedPlan) -> bool:
    """Check if the plan mentions blood pressure monitoring (S5)."""
    return _has_keyword(parsed.raw_text, BP_KEYWORDS)


def check_g4_condition_keywords(parsed: ParsedPlan, profile_id: str) -> bool:
    """Check G4: condition-specific precautions mentioned."""
    keywords = G4_KEYWORDS.get(profile_id)
    if keywords is None:
        return True
    return _has_keyword(parsed.raw_text, keywords)


def estimate_weekly_volume(parsed: ParsedPlan) -> dict:
    """Estimate total weekly exercise volume."""
    moderate_min = 0.0
    vigorous_min = 0.0
    total_min = 0.0
    for ex in parsed.exercises:
        dur = ex.duration_min or 0
        intensity = (ex.intensity_label or "moderate").lower()
        if "vigorous" in intensity:
            vigorous_min += dur
        else:
            moderate_min += dur
        total_min += dur
    return {
        "moderate_minutes": moderate_min,
        "vigorous_minutes": vigorous_min,
        "total_minutes": total_min,
    }


def intensity_matches(llm_label: str | None, acsm_class: str) -> bool:
    """Check if LLM intensity label matches ACSM classification."""
    if llm_label is None:
        return False
    llm = llm_label.lower().strip()
    acsm = acsm_class.lower().strip()
    if llm == acsm:
        return True
    if "-to-" in llm:
        parts = [p.strip() for p in llm.split("-to-")]
        return acsm in parts
    if llm == "very-light" and acsm == "light":
        return True
    return False
