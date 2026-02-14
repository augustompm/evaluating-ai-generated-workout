"""Exercise-to-MET mapping from the Compendium of Physical Activities 2024."""

_EXERCISE_DB = [
    # --- Sedentary / Very Light ---
    (["standing still", "stand"],
     1.3, "standing, general"),
    (["sitting", "seated"],
     1.3, "sitting, general"),
    (["lying", "resting"],
     1.0, "lying quietly"),

    # --- Light (2-3 METs) ---
    (["stretch", "flexibility", "yoga gentle", "cool-down stretch"],
     2.3, "stretching, mild"),
    (["tai chi"],
     2.5, "tai chi, general"),
    (["arm raise", "arm elevation", "arm circle"],
     2.8, "calisthenics, light effort"),
    (["balance exercise", "balance training"],
     2.5, "balance exercises"),
    (["breathing exercise", "deep breathing", "diaphragmatic breathing"],
     1.3, "breathing exercises, general"),
    (["seated marching", "chair marching", "marching in place"],
     2.5, "marching, in place, light effort"),

    # --- Light-Moderate (3-4 METs) ---
    (["walk", "brisk walk", "treadmill walk"],
     3.5, "walking, 3.0 mph, moderate pace"),
    (["pilates"],
     3.0, "Pilates, general"),
    (["yoga", "hatha yoga"],
     3.0, "yoga, hatha"),
    (["water aerobics", "aqua aerobics", "pool exercise", "aquatic"],
     3.5, "water aerobics"),
    (["bodyweight", "calisthenics moderate", "squat", "lunge", "push-up",
      "pushup", "push up", "plank", "glute bridge", "wall sit",
      "step-up", "step up", "bird dog", "dead bug"],
     3.5, "calisthenics, moderate effort (bodyweight)"),
    (["stair", "stair climbing", "step climbing"],
     4.0, "climbing stairs, slow pace"),
    (["resistance band", "elastic band", "band exercise"],
     3.5, "resistance training, light-moderate"),
    (["leg press", "leg extension", "leg curl", "calf raise"],
     3.5, "resistance training, light-moderate"),
    (["elliptical light", "elliptical low"],
     3.5, "elliptical trainer, light effort"),

    # --- Moderate (4-6 METs) ---
    (["strength training", "weight training", "weight lifting",
      "resistance training", "dumbbell", "barbell", "kettlebell",
      "chest press", "bench press", "shoulder press", "overhead press",
      "bicep curl", "tricep", "lat pulldown", "seated row",
      "deadlift", "pull-up", "pullup", "pull up", "chin-up", "chinup",
      "chin up", "cable", "machine"],
     5.0, "resistance training, moderate effort"),
    (["cycling moderate", "stationary bike moderate", "bike moderate"],
     5.5, "bicycling, stationary, moderate effort"),
    (["swim", "swimming moderate", "lap swimming"],
     5.8, "swimming laps, moderate effort"),
    (["dancing", "dance", "aerobic dance"],
     5.5, "dancing, aerobic, general"),
    (["hiking"],
     5.3, "hiking, general"),
    (["elliptical", "cross trainer", "elliptical moderate"],
     5.0, "elliptical trainer, moderate effort"),
    (["rowing moderate", "rowing machine"],
     4.8, "rowing, stationary, moderate effort"),
    (["tennis", "badminton", "racquetball"],
     5.0, "tennis, general"),
    (["basketball", "soccer", "football"],
     6.0, "basketball/soccer, general"),

    # --- Moderate-Vigorous (6-7 METs) ---
    (["cycling", "bicycling", "biking", "bike", "stationary bike",
      "spinning", "spin class"],
     6.8, "bicycling, 10-11.9 mph, moderate effort"),
    (["jog", "jogging", "light jog", "slow jog"],
     7.0, "jogging, general"),
    (["circuit training", "HIIT light", "boot camp"],
     7.0, "circuit training, moderate-vigorous"),

    # --- Vigorous (7+ METs) ---
    (["running", "run ", "treadmill run"],
     8.3, "running, 5 mph (12 min/mile)"),
    (["jumping", "jump rope", "skipping rope", "jumping jack",
      "burpee", "box jump", "plyometric"],
     8.0, "jumping, general/jump rope"),
    (["rowing vigorous"],
     8.5, "rowing, stationary, vigorous effort"),
    (["swimming vigorous", "swimming fast"],
     8.0, "swimming laps, vigorous effort"),
    (["HIIT", "high intensity interval", "tabata", "sprint interval"],
     8.0, "high intensity interval training"),
    (["sprint", "sprinting"],
     11.0, "running, 7 mph"),
    (["cycling vigorous", "bike vigorous", "cycling fast"],
     10.0, "bicycling, 14-15.9 mph, vigorous effort"),
]


def lookup_met(exercise_name: str) -> tuple[float | None, str | None]:
    """Look up the MET value for an exercise name.

    Uses longest-match-first strategy: among all matching patterns,
    picks the entry whose longest matching pattern is longest.
    """
    name_lower = exercise_name.lower().strip()

    best_match = None

    for patterns, met, description in _EXERCISE_DB:
        for pattern in patterns:
            if pattern.lower() in name_lower:
                plen = len(pattern)
                if best_match is None or plen > best_match[0]:
                    best_match = (plen, met, description)
                break

    if best_match:
        return best_match[1], best_match[2]
    return None, None


def lookup_met_value(exercise_name: str) -> float | None:
    """Simplified lookup returning only the MET value."""
    met, _ = lookup_met(exercise_name)
    return met
