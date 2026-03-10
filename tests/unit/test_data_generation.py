"""Unit tests for scripts.generate_data module."""

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_data import (
    _build_bio,
    _random_insurances,
    _random_rating,
    _random_specialty,
    _random_state,
    generate_providers,
    validate_diversity,
)


class TestRandomState:
    def test_returns_valid_state_code(self) -> None:
        state = _random_state()
        assert len(state) == 2
        assert state.isupper()

    def test_returns_known_state(self) -> None:
        from scripts.generate_data import US_STATES
        assert _random_state() in US_STATES


class TestRandomSpecialty:
    def test_returns_string(self) -> None:
        spec = _random_specialty()
        assert isinstance(spec, str)
        assert len(spec) > 0

    def test_returns_known_specialty(self) -> None:
        from scripts.generate_data import SPECIALTIES
        assert _random_specialty() in SPECIALTIES


class TestRandomInsurances:
    def test_returns_list(self) -> None:
        insurances = _random_insurances()
        assert isinstance(insurances, list)

    def test_length_between_1_and_5(self) -> None:
        for _ in range(20):
            ins = _random_insurances()
            assert 1 <= len(ins) <= 5

    def test_all_strings(self) -> None:
        ins = _random_insurances()
        assert all(isinstance(i, str) for i in ins)


class TestRandomRating:
    def test_within_bounds(self) -> None:
        for _ in range(50):
            r = _random_rating()
            assert 1.0 <= r <= 5.0

    def test_returns_float(self) -> None:
        assert isinstance(_random_rating(), float)


class TestBuildBio:
    def test_contains_provider_name(self) -> None:
        bio = _build_bio(
            name="Dr. Jane Doe",
            last_name="Doe",
            specialty="Cardiology",
            city="Los Angeles",
            state_name="California",
            insurances=["Medicare"],
            accepting=True,
        )
        assert "Doe" in bio

    def test_contains_specialty(self) -> None:
        bio = _build_bio(
            name="Dr. John Smith",
            last_name="Smith",
            specialty="Pediatrics",
            city="Austin",
            state_name="Texas",
            insurances=["Aetna", "Cigna"],
            accepting=False,
        )
        assert "Pediatrics" in bio

    def test_accepting_sentence_present(self) -> None:
        bio_yes = _build_bio(
            name="Dr. A B",
            last_name="B",
            specialty="Neurology",
            city="Chicago",
            state_name="Illinois",
            insurances=["Medicare"],
            accepting=True,
        )
        assert "accepting new patients" in bio_yes.lower()

        bio_no = _build_bio(
            name="Dr. C D",
            last_name="D",
            specialty="Neurology",
            city="Chicago",
            state_name="Illinois",
            insurances=["Medicare"],
            accepting=False,
        )
        assert "not currently accepting" in bio_no.lower()

    def test_insurance_in_bio(self) -> None:
        bio = _build_bio(
            name="Dr. E F",
            last_name="F",
            specialty="Dermatology",
            city="Miami",
            state_name="Florida",
            insurances=["Medicare", "Medicaid"],
            accepting=True,
        )
        assert "Medicare" in bio


class TestGenerateProviders:
    @pytest.fixture(scope="class")
    def small_dataset(self) -> List[Dict[str, Any]]:
        return generate_providers(n=200)

    def test_correct_count(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        assert len(small_dataset) == 200

    def test_required_fields_present(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        required = {
            "npi", "name", "gender", "specialties",
            "state", "city", "address", "lat", "long",
            "insurances", "rating", "accepting_new_patients",
            "bio",
        }
        for provider in small_dataset:
            assert required.issubset(provider.keys())

    def test_npis_unique(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        npis = [p["npi"] for p in small_dataset]
        assert len(npis) == len(set(npis))

    def test_npi_is_10_digits(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert len(p["npi"]) == 10
            assert p["npi"].isdigit()

    def test_rating_in_bounds(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert 1.0 <= p["rating"] <= 5.0

    def test_gender_values_valid(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert p["gender"] in ("M", "F", "N")

    def test_lat_long_are_numeric(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert isinstance(p["lat"], float)
            assert isinstance(p["long"], float)

    def test_specialties_is_list(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert isinstance(p["specialties"], list)
            assert len(p["specialties"]) >= 1

    def test_insurances_is_list(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert isinstance(p["insurances"], list)
            assert len(p["insurances"]) >= 1

    def test_bio_non_empty(
        self, small_dataset: List[Dict[str, Any]]
    ) -> None:
        for p in small_dataset:
            assert len(p["bio"]) > 20


class TestValidateDiversity:
    def test_passes_on_valid_dataset(self) -> None:
        providers = generate_providers(n=500)
        validate_diversity(providers)

    def test_fails_on_wrong_gender_ratio(self) -> None:
        providers = generate_providers(n=200)
        for p in providers:
            p["gender"] = "M"
        with pytest.raises(AssertionError):
            validate_diversity(providers)

    def test_fails_when_states_missing(self) -> None:
        providers = generate_providers(n=200)
        for p in providers:
            p["state"] = "CA"
        with pytest.raises(AssertionError):
            validate_diversity(providers)
