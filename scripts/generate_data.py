"""Generate 10,000 synthetic healthcare providers with diversity enforcement."""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_US")
fake.seed_instance(SEED)

SPECIALTY_WEIGHTS = [
    ("Family Medicine", 0.20),
    ("Internal Medicine", 0.12),
    ("Pediatrics", 0.08),
    ("Obstetrics & Gynecology", 0.06),
    ("Psychiatry", 0.05),
    ("Cardiology", 0.05),
    ("Orthopedic Surgery", 0.04),
    ("Dermatology", 0.04),
    ("Neurology", 0.03),
    ("Emergency Medicine", 0.03),
    ("Ophthalmology", 0.02),
    ("Gastroenterology", 0.02),
    ("Urology", 0.02),
    ("Pulmonology", 0.02),
    ("Endocrinology", 0.02),
    ("Rheumatology", 0.02),
    ("Nephrology", 0.02),
    ("Hematology/Oncology", 0.02),
    ("Radiology", 0.02),
    ("Anesthesiology", 0.02),
    ("General Surgery", 0.02),
    ("Physical Medicine & Rehabilitation", 0.01),
    ("Infectious Disease", 0.01),
    ("Allergy & Immunology", 0.01),
    ("Geriatrics", 0.01),
    ("Sports Medicine", 0.01),
    ("Palliative Care", 0.01),
    ("Pathology", 0.01),
    ("Vascular Surgery", 0.01),
    ("Plastic Surgery", 0.01),
]

SPECIALTIES = [s for s, _ in SPECIALTY_WEIGHTS]
SPECIALTY_PROBS = [w for _, w in SPECIALTY_WEIGHTS]
SPECIALTY_PROBS[-1] += 1.0 - sum(SPECIALTY_PROBS)

INSURANCE_POOL = [
    "Medicare",
    "Medicaid",
    "Blue Cross Blue Shield",
    "Aetna",
    "Cigna",
    "Humana",
    "UnitedHealth",
    "Tricare",
    "Kaiser Permanente",
    "Anthem",
    "Molina Healthcare",
    "Centene",
    "WellCare",
    "Oscar Health",
]

US_STATES = {
    "CA": ("California", 39_538_223, (32.5, 42.0, -124.4, -114.1)),
    "TX": ("Texas", 29_145_505, (25.8, 36.5, -106.6, -93.5)),
    "FL": ("Florida", 21_538_187, (24.5, 31.0, -87.6, -80.0)),
    "NY": ("New York", 20_201_249, (40.5, 45.0, -79.8, -71.8)),
    "PA": ("Pennsylvania", 13_002_700, (39.7, 42.3, -80.5, -74.7)),
    "IL": ("Illinois", 12_812_508, (36.9, 42.5, -91.5, -87.0)),
    "OH": ("Ohio", 11_799_448, (38.4, 42.3, -84.8, -80.5)),
    "GA": ("Georgia", 10_711_908, (30.4, 35.0, -85.6, -80.8)),
    "NC": ("North Carolina", 10_439_388, (33.8, 36.6, -84.3, -75.5)),
    "MI": ("Michigan", 10_077_331, (41.7, 48.3, -90.4, -82.4)),
    "NJ": ("New Jersey", 9_288_994, (38.9, 41.4, -75.6, -73.9)),
    "VA": ("Virginia", 8_631_393, (36.5, 39.5, -83.7, -75.2)),
    "WA": ("Washington", 7_705_281, (45.5, 49.0, -124.7, -116.9)),
    "AZ": ("Arizona", 7_151_502, (31.3, 37.0, -114.8, -109.0)),
    "MA": ("Massachusetts", 7_029_917, (41.2, 42.9, -73.5, -69.9)),
    "TN": ("Tennessee", 6_910_840, (34.9, 36.7, -90.3, -81.6)),
    "IN": ("Indiana", 6_785_528, (37.8, 41.8, -88.1, -84.8)),
    "MO": ("Missouri", 6_154_913, (36.0, 40.6, -95.8, -89.1)),
    "MD": ("Maryland", 6_177_224, (37.9, 39.7, -79.5, -75.0)),
    "WI": ("Wisconsin", 5_893_718, (42.5, 47.1, -92.9, -86.8)),
    "CO": ("Colorado", 5_773_714, (36.9, 41.0, -109.1, -102.0)),
    "MN": ("Minnesota", 5_706_494, (43.5, 49.4, -97.2, -89.5)),
    "SC": ("South Carolina", 5_118_425, (32.0, 35.2, -83.4, -78.5)),
    "AL": ("Alabama", 5_024_279, (30.2, 35.0, -88.5, -84.9)),
    "LA": ("Louisiana", 4_657_757, (28.9, 33.0, -94.0, -89.0)),
    "KY": ("Kentucky", 4_505_836, (36.5, 39.1, -89.6, -81.9)),
    "OR": ("Oregon", 4_237_256, (42.0, 46.3, -124.6, -116.5)),
    "OK": ("Oklahoma", 3_959_353, (33.6, 37.0, -103.0, -94.4)),
    "CT": ("Connecticut", 3_605_944, (40.9, 42.1, -73.7, -71.8)),
    "UT": ("Utah", 3_271_616, (36.9, 42.0, -114.1, -109.0)),
    "IA": ("Iowa", 3_190_369, (40.4, 43.5, -96.6, -90.1)),
    "NV": ("Nevada", 3_104_614, (35.0, 42.0, -120.0, -114.0)),
    "AR": ("Arkansas", 3_011_524, (33.0, 36.5, -94.6, -89.6)),
    "MS": ("Mississippi", 2_961_279, (30.2, 35.0, -91.7, -88.1)),
    "KS": ("Kansas", 2_937_880, (36.9, 40.0, -102.1, -94.6)),
    "NM": ("New Mexico", 2_117_522, (31.3, 37.0, -109.1, -103.0)),
    "NE": ("Nebraska", 1_961_504, (40.0, 43.0, -104.1, -95.3)),
    "ID": ("Idaho", 1_839_106, (41.9, 49.0, -117.2, -111.0)),
    "WV": ("West Virginia", 1_793_716, (37.2, 40.6, -82.6, -77.7)),
    "HI": ("Hawaii", 1_455_271, (18.9, 22.2, -160.3, -154.8)),
    "NH": ("New Hampshire", 1_377_529, (42.7, 45.3, -72.6, -70.6)),
    "ME": ("Maine", 1_362_359, (43.1, 47.5, -71.1, -66.9)),
    "MT": ("Montana", 1_084_225, (44.4, 49.0, -116.1, -104.0)),
    "RI": ("Rhode Island", 1_097_379, (41.1, 42.0, -71.9, -71.1)),
    "DE": ("Delaware", 989_948, (38.4, 39.8, -75.8, -75.0)),
    "SD": ("South Dakota", 886_667, (42.5, 45.9, -104.1, -96.4)),
    "ND": ("North Dakota", 779_094, (45.9, 49.0, -104.0, -96.5)),
    "AK": ("Alaska", 733_391, (54.7, 71.4, -168.0, -130.0)),
    "VT": ("Vermont", 643_077, (42.7, 45.0, -73.4, -71.5)),
    "WY": ("Wyoming", 576_851, (41.0, 45.0, -111.1, -104.1)),
    "DC": ("District of Columbia", 689_545, (38.8, 38.9, -77.1, -76.9)),
}

STATE_CODES = list(US_STATES.keys())
STATE_POPS = [US_STATES[s][1] for s in STATE_CODES]
TOTAL_POP = sum(STATE_POPS)
STATE_PROBS = [p / TOTAL_POP for p in STATE_POPS]

BIO_TEMPLATES = [
    (
        "Dr. {name} is a board-certified {specialty} specialist with "
        "{years} years of clinical experience. Based in {city}, {state}, "
        "Dr. {last_name} works at {hospital} and is known for "
        "{attribute}. {insurance_sentence} "
        "{accepting_sentence}"
    ),
    (
        "{name} specialises in {specialty} and has been practising "
        "medicine for over {years} years in {city}, {state}. "
        "Dr. {last_name} completed residency at {hospital} and "
        "focuses on {attribute}. {insurance_sentence} "
        "{accepting_sentence}"
    ),
    (
        "A leading {specialty} physician, Dr. {name} brings "
        "{years} years of expertise to patients in {city}, {state}. "
        "Affiliated with {hospital}, Dr. {last_name} is recognised "
        "for {attribute}. {insurance_sentence} "
        "{accepting_sentence}"
    ),
]

HOSPITALS = [
    "General Hospital",
    "Regional Medical Center",
    "University Health System",
    "Community Medical Center",
    "Memorial Hospital",
    "St. Mary's Medical Center",
    "Children's Hospital",
    "County Health Center",
    "Physicians Medical Group",
    "Advanced Care Clinic",
]

ATTRIBUTES = {
    "Cardiology": [
        "expertise in interventional cardiology and heart failure management",
        "pioneering minimally invasive cardiac procedures",
        "comprehensive cardiac rehabilitation programmes",
    ],
    "Family Medicine": [
        "patient-centred preventive care across all age groups",
        "managing complex chronic conditions holistically",
        "building long-term patient-physician relationships",
    ],
    "Pediatrics": [
        "child development and adolescent health",
        "compassionate care for newborns through teenagers",
        "paediatric chronic disease management",
    ],
    "Psychiatry": [
        "evidence-based treatment of mood and anxiety disorders",
        "integrative approaches to mental health",
        "specialised care for complex psychiatric conditions",
    ],
    "default": [
        "delivering high-quality, patient-centred care",
        "applying the latest evidence-based treatments",
        "a commitment to compassionate, comprehensive healthcare",
    ],
}


def _random_state() -> str:
    return random.choices(STATE_CODES, weights=STATE_PROBS, k=1)[0]


def _random_lat_long(state: str) -> tuple[float, float]:
    _, _, (lat_min, lat_max, lon_min, lon_max) = US_STATES[state]
    lat = round(random.uniform(lat_min, lat_max), 6)
    lon = round(random.uniform(lon_min, lon_max), 6)
    return lat, lon


def _random_specialty() -> str:
    return random.choices(
        SPECIALTIES, weights=SPECIALTY_PROBS, k=1
    )[0]


def _random_insurances() -> List[str]:
    n = random.randint(1, 5)
    chosen = random.sample(INSURANCE_POOL, n)
    if random.random() < 0.40 and "Medicare" not in chosen:
        chosen[0] = "Medicare"
    return chosen


def _random_rating() -> float:
    r = np.random.normal(4.1, 0.6)
    return round(float(np.clip(r, 1.0, 5.0)), 1)


def _generate_npi(used: set) -> str:
    while True:
        npi = str(random.randint(1_000_000_000, 9_999_999_999))
        if npi not in used:
            used.add(npi)
            return npi


def _build_bio(
    name: str,
    last_name: str,
    specialty: str,
    city: str,
    state_name: str,
    insurances: List[str],
    accepting: bool,
) -> str:
    years = random.randint(3, 35)
    hospital = f"{city} {random.choice(HOSPITALS)}"
    attrs = ATTRIBUTES.get(specialty, ATTRIBUTES["default"])
    attribute = random.choice(attrs)

    if len(insurances) == 1:
        ins_str = insurances[0]
    elif len(insurances) == 2:
        ins_str = f"{insurances[0]} and {insurances[1]}"
    else:
        ins_str = (
            ", ".join(insurances[:-1]) + f", and {insurances[-1]}"
        )
    insurance_sentence = (
        f"Dr. {last_name} accepts {ins_str}."
    )
    accepting_sentence = (
        "Currently accepting new patients."
        if accepting
        else "Not currently accepting new patients."
    )

    template = random.choice(BIO_TEMPLATES)
    return template.format(
        name=name,
        last_name=last_name,
        specialty=specialty,
        years=years,
        city=city,
        state=state_name,
        hospital=hospital,
        attribute=attribute,
        insurance_sentence=insurance_sentence,
        accepting_sentence=accepting_sentence,
    )


def generate_providers(n: int = 10_000) -> List[Dict[str, Any]]:
    providers = []
    used_npis: set = set()

    female_target = int(n * 0.50)
    male_target = int(n * 0.48)
    nonbinary_target = n - female_target - male_target

    genders = (
        ["F"] * female_target
        + ["M"] * male_target
        + ["N"] * nonbinary_target
    )
    random.shuffle(genders)

    state_codes_list = STATE_CODES.copy()
    guaranteed_states = state_codes_list.copy()
    random.shuffle(guaranteed_states)

    for i in range(n):
        gender = genders[i]

        if gender == "F":
            first_name = fake.first_name_female()
        elif gender == "M":
            first_name = fake.first_name_male()
        else:
            first_name = fake.first_name()

        last_name = fake.last_name()
        full_name = f"Dr. {first_name} {last_name}"

        if i < len(guaranteed_states):
            state_code = guaranteed_states[i]
        else:
            state_code = _random_state()

        state_name, _, bounds = US_STATES[state_code]
        lat, lon = _random_lat_long(state_code)
        city = fake.city()
        address = fake.street_address()

        specialty = _random_specialty()
        insurances = _random_insurances()
        rating = _random_rating()
        accepting = random.random() < 0.70
        npi = _generate_npi(used_npis)

        bio = _build_bio(
            name=full_name,
            last_name=last_name,
            specialty=specialty,
            city=city,
            state_name=state_name,
            insurances=insurances,
            accepting=accepting,
        )

        providers.append(
            {
                "npi": npi,
                "name": full_name,
                "gender": gender,
                "specialties": [specialty],
                "state": state_code,
                "city": city,
                "address": f"{address}, {city}, {state_code}",
                "lat": lat,
                "long": lon,
                "insurances": insurances,
                "rating": rating,
                "accepting_new_patients": accepting,
                "bio": bio,
            }
        )

    return providers


def validate_diversity(providers: List[Dict[str, Any]]) -> None:
    n = len(providers)
    from collections import Counter

    genders = Counter(p["gender"] for p in providers)
    states = {p["state"] for p in providers}
    medicare_count = sum(
        1 for p in providers if "Medicare" in p["insurances"]
    )

    print(f"Total providers: {n}")
    print(
        f"Gender F: {genders['F']/n:.1%} "
        f"M: {genders['M']/n:.1%} "
        f"N: {genders['N']/n:.1%}"
    )
    print(f"States covered: {len(states)}/51")
    print(f"Medicare coverage: {medicare_count/n:.1%}")

    assert genders["F"] / n >= 0.48, "Female ratio below 48%"
    assert genders["M"] / n >= 0.46, "Male ratio below 46%"
    assert len(states) == 51, f"Not all states covered: {len(states)}"
    assert medicare_count / n >= 0.40, (
        "Medicare coverage below 40%"
    )
    print("Diversity validation passed.")


def main() -> None:
    output_path = Path(__file__).parent.parent / "data"
    output_path.mkdir(exist_ok=True)
    out_file = output_path / "providers.json"

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    print(f"Generating {n} synthetic providers...")

    providers = generate_providers(n)
    validate_diversity(providers)

    with open(out_file, "w") as f:
        json.dump(providers, f, indent=2)

    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
