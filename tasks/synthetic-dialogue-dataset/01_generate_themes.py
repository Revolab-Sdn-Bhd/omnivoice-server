#!/usr/bin/env python3
"""
Stage 1 — Generate theme configs for M-CHAT synthetic dialogue dataset.

Outputs: stage1_themes.json
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


THEMES = [
    {
        "theme": "customer_support",
        "domains": ["banking", "telco", "insurance", "e-commerce"],
        "language_mix": "ms_en",
        "participants": {
            "agent": {"role": "customer_service", "voice_id": "anwar"},
            "human": {"role": "customer", "voice_id": "female_young_my"},
        },
        "constraints": {
            "min_turns": 6,
            "max_turns": 12,
            "emotions": ["neutral", "frustrated", "relieved", "polite", "confused", "satisfied"],
            "code_mix_ratio": 0.3,
        },
        "malaysian_entities": {
            "banks": ["Maybank", "CIMB", "Public Bank", "RHB", "Hong Leong", "Bank Islam"],
            "telcos": ["Maxis", "CelcomDigi", "U Mobile", "TM", "TIME"],
            "e_wallets": ["Touch n Go", "GrabPay", "Boost", "ShopeePay", "MAE"],
            "insurance": ["AIA", "Prudential", "Great Eastern", "Allianz", "Takaful"],
            "e_commerce": ["Shopee", "Lazada", "PG Mall"],
        },
    },
    {
        "theme": "podcast",
        "domains": ["tech_discussion", "lifestyle", "current_affairs", "food_review"],
        "language_mix": "ms_en",
        "participants": {
            "agent": {"role": "host", "voice_id": "anwar"},
            "human": {"role": "guest", "voice_id": "female_young_my"},
        },
        "constraints": {
            "min_turns": 8,
            "max_turns": 14,
            "emotions": ["enthusiastic", "neutral", "curious", "amused", "thoughtful"],
            "code_mix_ratio": 0.4,
        },
        "malaysian_entities": {
            "topics": ["AI trends", "startup ecosystem", "remote work", "social media", "gaming"],
            "brands": ["Grab", "AirAsia", "Petronas", "Proton", "Inovasi"],
            "places": ["KL", "Penang", "JB", "Cyberjaya", "Iskandar"],
        },
    },
    {
        "theme": "debate",
        "domains": ["education", "social_issues", "tech_ethics", "politics_light"],
        "language_mix": "ms_en",
        "participants": {
            "agent": {"role": "moderator", "voice_id": "anwar"},
            "human": {"role": "panelist", "voice_id": "female_young_my"},
        },
        "constraints": {
            "min_turns": 8,
            "max_turns": 14,
            "emotions": ["neutral", "passionate", "skeptical", "convinced", "diplomatic"],
            "code_mix_ratio": 0.25,
        },
        "malaysian_entities": {
            "issues": ["digital literacy", "public transport", "education reform", "cybersecurity"],
            "institutions": ["UKM", "UM", "UPM", "UIAM", "MDEC", "MCMC"],
        },
    },
    {
        "theme": "counselling",
        "domains": ["mental_health", "career_guidance", "academic", "relationship"],
        "language_mix": "ms_en",
        "participants": {
            "agent": {"role": "counselor", "voice_id": "anwar"},
            "human": {"role": "client", "voice_id": "female_young_my"},
        },
        "constraints": {
            "min_turns": 8,
            "max_turns": 12,
            "emotions": ["empathetic", "neutral", "anxious", "relieved", "hopeful"],
            "code_mix_ratio": 0.2,
        },
        "malaysian_entities": {
            "services": ["Befrienders", "MIASA", "HELP University", "Talian Kasih"],
            "workplaces": ["Khazanah", "TNB", "Sime Darby", "Tenaga Nasional"],
        },
    },
    {
        "theme": "medical",
        "domains": ["clinic_triage", "pharmacy", "follow_up", "specialist_referral"],
        "language_mix": "ms_en",
        "participants": {
            "agent": {"role": "healthcare_staff", "voice_id": "anwar"},
            "human": {"role": "patient", "voice_id": "female_young_my"},
        },
        "constraints": {
            "min_turns": 6,
            "max_turns": 10,
            "emotions": ["neutral", "concerned", "reassuring", "anxious", "grateful"],
            "code_mix_ratio": 0.2,
        },
        "malaysian_entities": {
            "hospitals": ["Hospital KL", "Hospital UKM", "KPJ", "Sunway Medical", "Pantai"],
            "insurances": ["Prudential", "AIA", "Great Eastern"],
            "medicines": ["Panadol", "Ubat batuk", "Antibiotik"],
        },
    },
    {
        "theme": "education",
        "domains": ["tutor_student", "parent_teacher", "university_advising", "training"],
        "language_mix": "ms_en",
        "participants": {
            "agent": {"role": "teacher", "voice_id": "anwar"},
            "human": {"role": "student", "voice_id": "female_young_my"},
        },
        "constraints": {
            "min_turns": 6,
            "max_turns": 12,
            "emotions": ["encouraging", "neutral", "frustrated", "curious", "proud"],
            "code_mix_ratio": 0.35,
        },
        "malaysian_entities": {
            "schools": ["SMK", "MRSM", "Sekolah Antarabangsa", "Chinese school"],
            "exams": ["SPM", "STPM", "UPSR", "IGCSE"],
            "subjects": ["Matematik", "Sains", "Sejarah", "Bahasa Melayu", "English"],
        },
    },
]


def generate_themes(config: dict) -> list[dict]:
    """Generate theme configs. Returns list of theme objects."""
    selected = config.get("pipeline", {}).get("themes")
    if selected:
        return [t for t in THEMES if t["theme"] in selected]
    return THEMES


def main() -> None:
    config_path = SCRIPT_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    themes = generate_themes(config)
    out_path = output_dir / "stage1_themes.json"
    with open(out_path, "w") as f:
        json.dump(themes, f, indent=2, ensure_ascii=False)

    logger.info("Stage 1 complete: %d themes → %s", len(themes), out_path)


if __name__ == "__main__":
    main()
