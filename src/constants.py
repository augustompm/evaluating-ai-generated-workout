"""ACSM, WHO, and Tanaka constants for exercise prescription evaluation."""

# --- Tanaka HRmax formula ---
TANAKA_INTERCEPT = 208
TANAKA_SLOPE = 0.7

# --- ACSM Table 6.1: %HRmax zones ---
VERY_LIGHT_HRMAX_HIGH_PCT = 0.57
LIGHT_HRMAX_LOW_PCT = 0.57
LIGHT_HRMAX_HIGH_PCT = 0.64
MODERATE_ZONE_LOW_PCT = 0.64
MODERATE_ZONE_HIGH_PCT = 0.76
VIGOROUS_ZONE_LOW_PCT = 0.77
VIGOROUS_ZONE_HIGH_PCT = 0.95

# --- WHO 2020: Weekly volume targets ---
WEEKLY_GOAL_MODERATE_MINUTES = 150
WEEKLY_GOAL_VIGOROUS_MINUTES = 75
VIGOROUS_TO_MODERATE_RATIO = 2
WHO_STRENGTH_DAYS_PER_WEEK = 2

# --- ACSM Table 6.1: MET thresholds by age group ---
ACSM_MET_THRESHOLDS = {
    "young": {  # 20-39 years
        "age_range": (20, 39),
        "light": (2.4, 4.7),
        "moderate": (4.8, 7.1),
        "vigorous": (7.2, 10.1),
        "near_max": 10.2,
    },
    "middle": {  # 40-64 years
        "age_range": (40, 64),
        "light": (2.0, 3.9),
        "moderate": (4.0, 5.9),
        "vigorous": (6.0, 8.4),
        "near_max": 8.5,
    },
    "older": {  # >=65 years
        "age_range": (65, 120),
        "light": (1.6, 3.1),
        "moderate": (3.2, 4.7),
        "vigorous": (4.8, 6.7),
        "near_max": 6.8,
    },
}

# --- Intensity labels ---
INTENSITY_SEDENTARY = 0
INTENSITY_LIGHT = 1
INTENSITY_MODERATE = 2
INTENSITY_VIGOROUS = 3
INTENSITY_NEAR_MAX = 4

INTENSITY_LABELS = {
    INTENSITY_SEDENTARY: "sedentary",
    INTENSITY_LIGHT: "light",
    INTENSITY_MODERATE: "moderate",
    INTENSITY_VIGOROUS: "vigorous",
    INTENSITY_NEAR_MAX: "near_max",
}

# --- MET sedentary threshold (WHO 2020) ---
MET_SEDENTARY_MAX = 1.5

# --- Ollama models ---
OLLAMA_MODELS = [
    "llama3.1:8b",
    "qwen2.5:14b",
    "deepseek-r1:14b",
    "alibayram/medgemma:4b",
]

OLLAMA_MODEL_NAMES = {
    "llama3.1:8b": "Llama 3.1 8B",
    "qwen2.5:14b": "Qwen 2.5 14B",
    "deepseek-r1:14b": "DeepSeek-R1 14B",
    "alibayram/medgemma:4b": "MedGemma 4B",
}

# --- Generation parameters ---
GENERATION_TEMPERATURE = 0.7
GENERATION_RUNS = 3
GENERATION_SEED = 42


# --- Helper functions ---

def calculate_hrmax_tanaka(age: int) -> float:
    """HRmax = 208 - 0.7 * age (Tanaka et al. 2001)."""
    return TANAKA_INTERCEPT - TANAKA_SLOPE * age


def get_age_group(age: int) -> str:
    """Map age to ACSM Table 6.1 age group."""
    if age < 40:
        return "young"
    elif age < 65:
        return "middle"
    else:
        return "older"


def classify_met_by_age(met: float, age: int) -> str:
    """Classify a MET value into intensity category based on age group.

    Returns: 'sedentary', 'light', 'moderate', 'vigorous', or 'near_max'
    Source: ACSM 10th ed., Table 6.1, p.183
    """
    if met <= MET_SEDENTARY_MAX:
        return "sedentary"

    group = get_age_group(age)
    thresholds = ACSM_MET_THRESHOLDS[group]

    if met >= thresholds["near_max"]:
        return "near_max"
    elif met >= thresholds["vigorous"][0]:
        return "vigorous"
    elif met >= thresholds["moderate"][0]:
        return "moderate"
    elif met >= thresholds["light"][0]:
        return "light"
    else:
        return "sedentary"


def calculate_hr_zones(age: int) -> dict:
    """Calculate HR zones based on ACSM Table 6.1 %HRmax thresholds."""
    hrmax = calculate_hrmax_tanaka(age)
    return {
        "hrmax": hrmax,
        "very_light_max": hrmax * VERY_LIGHT_HRMAX_HIGH_PCT,
        "light_low": hrmax * LIGHT_HRMAX_LOW_PCT,
        "light_high": hrmax * LIGHT_HRMAX_HIGH_PCT,
        "moderate_low": hrmax * MODERATE_ZONE_LOW_PCT,
        "moderate_high": hrmax * MODERATE_ZONE_HIGH_PCT,
        "vigorous_low": hrmax * VIGOROUS_ZONE_LOW_PCT,
        "vigorous_high": hrmax * VIGOROUS_ZONE_HIGH_PCT,
    }


def check_weekly_goal(moderate_minutes: float, vigorous_minutes: float) -> bool:
    """Check if weekly volume meets WHO 2020 guidelines."""
    equivalent = moderate_minutes + (vigorous_minutes * VIGOROUS_TO_MODERATE_RATIO)
    return equivalent >= WEEKLY_GOAL_MODERATE_MINUTES
