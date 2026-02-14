"""Three prompt variants for exercise plan generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from profiles import format_parq_responses

OUTPUT_FORMAT = """
Present your plan using this format for each day:

**[Day Name]**
1. [Exercise Name] | Intensity: [Light/Moderate/Vigorous] | Duration: [X] minutes
2. ...

After the weekly plan, include any additional notes or recommendations."""

PROMPT_PATIENT = """\
I need a personalized weekly exercise program and I'm looking for \
professional guidance.

Here is my profile:
- Age: {age}, Sex: {sex}
- Height: {height_cm} cm, Weight: {weight_kg} kg, BMI: {bmi}
- Medical conditions: {conditions}
- Medications: {medications}
- Current activity level: {ipaq_level} (based on IPAQ-SF)
- PAR-Q+ responses:
{parq_responses}
- Goal: {goal}

Please create a complete weekly exercise plan that is appropriate for me.
{output_format}"""

PROMPT_PATIENT_CLEARED = """\
I need a personalized weekly exercise program. My doctor has evaluated \
my condition and cleared me for exercise with appropriate modifications.

Here is my profile:
- Age: {age}, Sex: {sex}
- Height: {height_cm} cm, Weight: {weight_kg} kg, BMI: {bmi}
- Medical conditions: {conditions}
- Medications: {medications}
- Current activity level: {ipaq_level} (based on IPAQ-SF)
- PAR-Q+ responses:
{parq_responses}
- Goal: {goal}

Please create a complete weekly exercise plan that is appropriate for me.
{output_format}"""

PROMPT_PROFESSIONAL = """\
I am a healthcare provider. I need to design a weekly exercise program \
for one of my patients. Based on the following clinical profile, create \
an appropriate plan.

Patient Profile:
- Age: {age}, Sex: {sex}
- Height: {height_cm} cm, Weight: {weight_kg} kg, BMI: {bmi}
- Medical conditions: {conditions}
- Medications: {medications}
- Current activity level: {ipaq_level} (based on IPAQ-SF)
- PAR-Q+ responses:
{parq_responses}
- Patient goal: {goal}

Please create a complete weekly exercise plan for this patient.
{output_format}"""

PROMPT_VARIANTS = {
    "patient": PROMPT_PATIENT,
    "patient_cleared": PROMPT_PATIENT_CLEARED,
    "professional": PROMPT_PROFESSIONAL,
}


def build_prompt(profile: dict, variant: str = "patient") -> str:
    """Build a prompt from a patient profile and prompt variant."""
    template = PROMPT_VARIANTS[variant]
    return template.format(
        age=profile["age"],
        sex=profile["sex"],
        height_cm=profile["height_cm"],
        weight_kg=profile["weight_kg"],
        bmi=profile["bmi"],
        conditions=profile["conditions"],
        medications=profile["medications"],
        ipaq_level=profile["ipaq_level"],
        parq_responses=format_parq_responses(profile),
        goal=profile["goal"],
        output_format=OUTPUT_FORMAT,
    )


def get_variant_names() -> list[str]:
    """Return list of prompt variant names."""
    return list(PROMPT_VARIANTS.keys())
