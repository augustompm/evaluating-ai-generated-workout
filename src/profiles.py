"""Eight PAR-Q+ patient profiles for exercise prescription evaluation."""

PROFILES = {
    "P1": {
        "id": "P1",
        "age": 25,
        "sex": "M",
        "height_cm": 178,
        "weight_kg": 75,
        "bmi": 23.7,
        "conditions": "None",
        "medications": "None",
        "ipaq_level": "High",
        "goal": "Maintain fitness and improve athletic performance",
        "risk_level": "low",
        "parq": {
            "Q1": False,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": False,
            "Q6": False,
            "Q7": False,
        },
        "parq_summary": "All NO — cleared for activity without restriction",
    },
    "P2": {
        "id": "P2",
        "age": 35,
        "sex": "F",
        "height_cm": 165,
        "weight_kg": 62,
        "bmi": 22.8,
        "conditions": "None (sedentary lifestyle)",
        "medications": "None",
        "ipaq_level": "Low",
        "goal": "Start a regular exercise routine to improve general health",
        "risk_level": "low",
        "parq": {
            "Q1": False,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": False,
            "Q6": False,
            "Q7": False,
        },
        "parq_summary": "All NO — cleared, but sedentary (needs gradual progression)",
    },
    "P3": {
        "id": "P3",
        "age": 55,
        "sex": "M",
        "height_cm": 175,
        "weight_kg": 85,
        "bmi": 27.8,
        "conditions": "Controlled hypertension",
        "medications": "Lisinopril 10mg daily",
        "ipaq_level": "Moderate",
        "goal": "Improve cardiovascular health and manage blood pressure",
        "risk_level": "moderate",
        "parq": {
            "Q1": True,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": True,
            "Q6": False,
            "Q7": False,
        },
        "parq_summary": "Q1=YES (hypertension), Q5=YES (medication)",
    },
    "P4": {
        "id": "P4",
        "age": 60,
        "sex": "F",
        "height_cm": 160,
        "weight_kg": 85,
        "bmi": 33.2,
        "conditions": "Type 2 diabetes, obesity (BMI 33.2)",
        "medications": "Metformin 1000mg twice daily",
        "ipaq_level": "Low",
        "goal": "Manage blood sugar levels and lose weight through exercise",
        "risk_level": "moderate",
        "parq": {
            "Q1": False,
            "Q2": False,
            "Q3": False,
            "Q4": True,
            "Q5": True,
            "Q6": False,
            "Q7": False,
        },
        "parq_summary": "Q4=YES (diabetes — chronic condition), Q5=YES (metformin)",
    },
    "P5": {
        "id": "P5",
        "age": 45,
        "sex": "M",
        "height_cm": 180,
        "weight_kg": 90,
        "bmi": 27.8,
        "conditions": "Knee osteoarthritis (bilateral, mild-moderate)",
        "medications": "Ibuprofen as needed for pain",
        "ipaq_level": "Low",
        "goal": "Reduce knee pain and improve joint mobility through safe exercise",
        "risk_level": "moderate",
        "parq": {
            "Q1": False,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": False,
            "Q6": True,
            "Q7": False,
        },
        "parq_summary": "Q6=YES (bone/joint problem — knee osteoarthritis)",
    },
    "P6": {
        "id": "P6",
        "age": 70,
        "sex": "F",
        "height_cm": 158,
        "weight_kg": 65,
        "bmi": 26.0,
        "conditions": "History of myocardial infarction (2 years ago), stable angina",
        "medications": "Aspirin 81mg, Atenolol 50mg, Atorvastatin 40mg daily",
        "ipaq_level": "Low",
        "goal": "Safely return to physical activity after cardiac event",
        "risk_level": "high",
        "parq": {
            "Q1": True,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": True,
            "Q6": False,
            "Q7": True,
        },
        "parq_summary": "Q1=YES (heart condition), Q5=YES (cardiac meds), Q7=YES (supervised)",
    },
    "P7": {
        "id": "P7",
        "age": 30,
        "sex": "M",
        "height_cm": 182,
        "weight_kg": 80,
        "bmi": 24.2,
        "conditions": "None",
        "medications": "None",
        "ipaq_level": "High",
        "goal": "Enhance endurance and strength for recreational sports",
        "risk_level": "low",
        "parq": {
            "Q1": False,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": False,
            "Q6": False,
            "Q7": False,
        },
        "parq_summary": "All NO — cleared for all activity levels",
    },
    "P8": {
        "id": "P8",
        "age": 50,
        "sex": "F",
        "height_cm": 163,
        "weight_kg": 95,
        "bmi": 35.7,
        "conditions": "Hypertension (uncontrolled), obesity (BMI 35.7), sedentary",
        "medications": "Amlodipine 5mg, Hydrochlorothiazide 25mg daily",
        "ipaq_level": "Low",
        "goal": "Lower blood pressure and lose weight through gradual exercise",
        "risk_level": "high",
        "parq": {
            "Q1": True,
            "Q2": False,
            "Q3": False,
            "Q4": False,
            "Q5": True,
            "Q6": False,
            "Q7": False,
        },
        "parq_summary": "Q1=YES (hypertension), Q5=YES (anti-hypertensives)",
    },
}

CLEARANCE_PROFILES = {"P3", "P4", "P5", "P6", "P8"}
SEDENTARY_PROFILES = {"P2", "P4", "P6", "P8"}
CONDITION_PROFILES = {"P3", "P4", "P5", "P6", "P8"}

RISK_GROUPS = {
    "low": ["P1", "P2", "P7"],
    "moderate": ["P3", "P4", "P5"],
    "high": ["P6", "P8"],
}


def get_profile(profile_id: str) -> dict:
    """Get a profile by ID."""
    return PROFILES[profile_id]


def get_all_profile_ids() -> list:
    """Return sorted list of all profile IDs."""
    return sorted(PROFILES.keys())


def format_parq_responses(profile: dict) -> str:
    """Format PAR-Q+ responses as human-readable string for the prompt."""
    parq = profile["parq"]
    questions = {
        "Q1": "Heart condition or high blood pressure",
        "Q2": "Chest pain during physical activity",
        "Q3": "Dizziness or loss of consciousness",
        "Q4": "Chronic medical condition",
        "Q5": "Currently taking medication",
        "Q6": "Bone or joint problem",
        "Q7": "Doctor prescribing supervised activity",
    }
    lines = []
    for q_id, question in questions.items():
        answer = "YES" if parq[q_id] else "NO"
        lines.append(f"  {q_id}: {question} — {answer}")
    return "\n".join(lines)
