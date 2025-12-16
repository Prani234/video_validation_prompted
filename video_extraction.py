# -----------------------
# Hybrid Insights Extraction Module
# -----------------------

import re
import json
import numpy as np
from typing import Dict, Any, List
import librosa
from rapidfuzz import fuzz
# -----------------------
# CONSTANTS (Expandable)
# -----------------------

FILLERS = {
    "um", "uh", "like", "you know", "actually", "basically",
    "so", "i mean", "right", "okay", "well", "hmm", "sort of", "kind of"
}

CONFIDENT_WORDS = {
    "achieved", "led", "managed", "developed", "built", "initiated",
    "executed", "delivered", "implemented", "optimized", "designed",
    "oversaw", "mentored", "coordinated", "improved", "driven", "owned"
}

SOCIAL_SKILLS_BASE = {
    "communication", "teamwork", "leadership", "collaboration",
    "problem solving", "adaptability", "presentation", "interpersonal",
    "negotiation", "conflict resolution", "creativity",
    "critical thinking", "time management", "empathy", "mentoring"
}

SOCIAL_RULES = {
    "teamwork": [
        "collaborated", "worked with", "cross functional",
        "team", "stakeholders"
    ],
    "communication": [
        "communicated", "explained", "presented",
        "discussed", "interaction"
    ],
    "leadership": [
        "led", "managed", "mentored", "guided", "owned"
    ],
    "problem solving": [
        "solved", "debugged", "resolved", "troubleshoot"
    ],
    "adaptability": [
        "learned", "adapted", "quickly", "new technology"
    ]
}


STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "for", "is", "with",
    "on", "as", "by", "at", "an", "it", "that", "this"
}

YES_JOIN_PATTERNS = [
    "immediate joining", "join immediately", "available immediately",
    "ready to join", "no notice period", "can join now", "start immediately"
]

NO_JOIN_PATTERNS = [
    "notice period", "30 days", "60 days", "90 days",
    "one month", "two months", "currently serving"
]

CTC_CONTEXT = ["ctc", "salary", "package", "compensation", "expected", "preferred"]

# -----------------------
# TEXT METRICS
# -----------------------

def filler_rate(text: str) -> float:
    words = text.lower().split()
    return sum(w in FILLERS for w in words) / max(len(words), 1)

def lexical_diversity(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    return len(set(words)) / max(len(words), 1)

def confidence_word_ratio(text: str) -> float:
    words = text.lower().split()
    return sum(w in CONFIDENT_WORDS for w in words) / max(len(words), 1)

def words_per_minute(text: str, duration_sec: float) -> float:
    return (len(text.split()) / max(duration_sec, 1e-3)) * 60

# -----------------------
# AUDIO METRICS
# -----------------------

def count_long_pauses(segments: List[Dict[str, float]], threshold: float = 1.2) -> int:
    pauses = 0
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i-1]["end"]
        if gap > threshold:
            pauses += 1
    return pauses

def volume_variance(audio_path: str) -> float:
    y, _ = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    return float(np.std(rms))

# -----------------------
# SOCIAL SKILLS (HYBRID)
# -----------------------

def remove_similar_skills(skills: list, threshold: int = 85) -> list:
    cleaned = []
    for skill in skills:
        if not any(fuzz.ratio(skill, existing) > threshold for existing in cleaned):
            cleaned.append(skill)
    return cleaned

def extract_social_skills(transcript: str, resume_json: Dict[str, Any], embedder) -> List[str]:
    skills_found = set()

    full_text = (
        transcript.lower() + " " +
        json.dumps(resume_json).lower()
    )

    # ---- 1. Explicit resume soft skills ----
    soft_skills = resume_json.get("soft_skills", [])
    for s in soft_skills:
        skills_found.add(s.strip().lower())

    # ---- 2. Rule-based extraction ----
    for skill, patterns in SOCIAL_RULES.items():
        if any(p in full_text for p in patterns):
            skills_found.add(skill.strip().lower())

    # ---- 3. Semantic confirmation (LOW threshold) ----
    sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if len(s.strip()) > 5]

    if sentences:
        sent_vecs = embedder.encode(sentences, convert_to_numpy=True)
        skill_vecs = embedder.encode(list(SOCIAL_SKILLS_BASE), convert_to_numpy=True)

        sent_vecs /= np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-12
        skill_vecs /= np.linalg.norm(skill_vecs, axis=1, keepdims=True) + 1e-12

        sims = np.dot(sent_vecs, skill_vecs.T)

        for i, skill in enumerate(SOCIAL_SKILLS_BASE):
            if np.max(sims[:, i]) > 0.35:
                skills_found.add(skill.strip().lower())

    # Remove duplicates and sort
    skills_found = sorted(remove_similar_skills(list(skills_found)))

    return skills_found


# -----------------------
# KEYWORDS
# -----------------------

def extract_keywords(text: str, top_k: int = 15) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

# -----------------------
# HR SIGNALS
# -----------------------

def detect_immediate_joining(resume_json: Dict[str, Any], transcript: str) -> str:
    text = (json.dumps(resume_json) + " " + transcript).lower()

    if any(p in text for p in YES_JOIN_PATTERNS):
        return "Yes"
    if any(p in text for p in NO_JOIN_PATTERNS):
        return "No"
    return "Not Mentioned"

def extract_ctc(resume_json: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    text = json.dumps(resume_json).lower()
    numbers = re.findall(r'\b\d{1,3}\s*(?:lpa|lakhs|k)\b', text)

    expected = numbers[0] if numbers else None
    preferred = numbers[1] if len(numbers) > 1 else None

    return {
        "expected_ctc": expected,
        "preferred_ctc": preferred
    }

# -----------------------
# FINAL INSIGHTS
# -----------------------

def extract_insights(
    transcript: str,
    segments: List[Dict[str, float]],
    audio_path: str,
    audio_duration_sec: float,
    resume_json: Dict[str, Any],
    embedder
) -> Dict[str, Any]:

    # ---- Communication Metrics ----
    fr = filler_rate(transcript)
    ld = lexical_diversity(transcript)
    wpm = words_per_minute(transcript, audio_duration_sec)
    pauses = count_long_pauses(segments)

    fluency = 1 - fr
    pace = min(max((wpm - 110) / 50, 0), 1)
    clarity = min(ld * 1.5, 1)

    communication_score = round(
        (0.4 * fluency + 0.3 * pace + 0.3 * clarity) * 100, 2
    )

    if communication_score >= 80:
        level = "Excellent"
    elif communication_score >= 65:
        level = "Good"
    elif communication_score >= 45:
        level = "Average"
    else:
        level = "Poor"

    # ---- Confidence ----
    vol_var = volume_variance(audio_path)
    conf_words = confidence_word_ratio(transcript)

    confidence_score = round(
        0.5 * (1 - min(vol_var, 1)) +
        0.3 * conf_words +
        0.2 * (1 - fr),
        2
    )

    confidence_level = (
        "High" if confidence_score >= 0.7
        else "Moderate" if confidence_score >= 0.45
        else "Low"
    )

    # ---- Skills & Keywords ----
    social_skills = extract_social_skills(transcript, resume_json, embedder)

    video_keywords = extract_keywords(transcript)
    resume_keywords = extract_keywords(json.dumps(resume_json))

    # ---- HR Signals ----
    joining = detect_immediate_joining(resume_json, transcript)
    ctc = extract_ctc(resume_json, transcript)

    return {
        "communication": {
            "score": communication_score,
            "level": level,
            "details": {
                "wpm": round(wpm, 1),
                "filler_rate": round(fr, 3),
                "lexical_diversity": round(ld, 3),
                "long_pauses": pauses
            }
        },
        "overall_confidence": {
            "score": confidence_score,
            "level": confidence_level
        },
        "social_skills": social_skills,
        "main_keywords": {
            "video": video_keywords,
            "resume": resume_keywords
        },
        "hr_signals": {
            "expected_ctc": ctc["expected_ctc"],
            "preferred_ctc": ctc["preferred_ctc"],
            "immediate_joining": joining
        }
    }
