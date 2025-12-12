# video_resume_validator_streamlit.py
import os
import json
import tempfile
import time
import math
import re
from typing import List, Tuple, Dict, Any

import streamlit as st
import numpy as np
import cv2
import ffmpeg
import torch
from difflib import SequenceMatcher

from ultralytics import YOLO

# whisper (ensure installed)
import whisper

# embeddings & toxicity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------
# Configurable thresholds
# -----------------------
FRAME_SAMPLE_FPS = 1                  # sample rate for frames (frames per second)
MIN_PERSON_FRAMES_RATIO = 0.6         # person must appear in at least this ratio of sampled frames
CENTER_ZONE_RATIO = 0.35              # central region fraction (width & height)
MAX_AVG_CENTER_SHIFT = 0.12           # normalized (0..1) allowed average center shift between frames
MAX_AVG_BOX_SCALE_CHANGE = 0.20       # allowed average bounding-box relative scale change

# Semantic thresholds
SIMILARITY_THRESHOLD = 0.55           # aggregated similarity threshold for "related to resume"
SKILL_SIMILARITY_THRESHOLD = 0.55     # per-skill similarity threshold to mark a skill as mentioned
TOP_K_CHUNKS = 3                      # top-k transcript chunks to aggregate similarity

# Category thresholds (for per-category matching)
ROLE_SIM_THRESHOLD = 0.50
COMPANY_SIM_THRESHOLD = 0.55
PROJECT_SIM_THRESHOLD = 0.50

# Name matching thresholds
NAME_EXACT_REQUIRED = False           # set True if you want exact name required
FUZZY_NAME_THRESHOLD = 0.60

# Toxicity thresholds
TOXICITY_MODEL_PROB_THRESHOLD = 0.60  # individual model probability threshold to consider label strong
ENSEMBLE_VOTE_THRESHOLD = 2           # number of detectors that must agree to flag toxicity
MIN_TRANSCRIPT_WORDS = 3              # minimum words required in transcript to consider speech present

# fallback profanity list (if toxicity models fail or as strong signal)
PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap"}

# Models (change if you prefer different ones)
EMBEDDER_MODEL = "all-mpnet-base-v2"  # good balance for semantic tasks
TOXICITY_PRIMARY = "martin-ha/toxic-comment-model"  # primary toxic detector
TOXICITY_FALLBACK = None  # fallback detector

# -----------------------
# Utility functions
# -----------------------
def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def extract_audio(video_path: str, out_wav_path: str):
    """
    Extract audio as WAV (16k mono) using ffmpeg.
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(out_wav_path, ac=1, ar='16000', vn=None, loglevel='error')
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg error extracting audio: {e}")

def sample_frames(video_path: str, sample_fps: int = FRAME_SAMPLE_FPS) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for sampling.")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # step number of frames
    step = max(int(round(video_fps / max(1, sample_fps))), 1)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame.copy())
        idx += 1
    cap.release()
    return frames, video_fps

# -----------------------
# YOLO video analysis
# -----------------------
@st.cache_resource
def load_yolo(weights: str = "yolov8n.pt"):
    return YOLO(weights)

def analyze_video_yolo(video_path: str, model: YOLO) -> Dict[str, Any]:
    frames, _ = sample_frames(video_path, FRAME_SAMPLE_FPS)
    total = len(frames)
    if total == 0:
        return {"video_valid": False, "reasons": ["no_frames_extracted"], "person_frames": 0,
                "total_frames": 0}

    frame_h, frame_w = frames[0].shape[:2]
    centers = []
    areas = []

    person_frame_count = 0
    max_persons_in_frame = 0
    multi_person_flag = False

    for frame in frames:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        persons = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append((x1, y1, x2, y2))

        person_count = len(persons)
        max_persons_in_frame = max(max_persons_in_frame, person_count)

        # multi-person check
        if person_count > 1:
            multi_person_flag = True

        if person_count == 1:
            # track this one person
            (x1, y1, x2, y2) = persons[0]
            cx = (x1 + x2) / 2.0 / frame_w
            cy = (y1 + y2) / 2.0 / frame_h
            area = max(0, (x2 - x1) * (y2 - y1))
            centers.append((cx, cy))
            areas.append(area)
            person_frame_count += 1
        else:
            centers.append(None)
            areas.append(None)

    presence_ratio = person_frame_count / max(1, total)

    # movement metrics
    shifts = []
    scale_changes = []
    prev_center = None
    prev_area = None

    for c, a in zip(centers, areas):
        if c is not None and prev_center is not None:
            dx = abs(c[0] - prev_center[0])
            dy = abs(c[1] - prev_center[1])
            shifts.append(max(dx, dy))

        if a is not None and prev_area is not None and prev_area > 0:
            scale_changes.append(abs(a - prev_area) / (prev_area + 1e-9))

        if c is not None:
            prev_center = c
        if a is not None:
            prev_area = a

    avg_shift = float(np.mean(shifts)) if shifts else 0.0
    avg_scale_change = float(np.mean(scale_changes)) if scale_changes else 0.0

    # centered face ratio
    cx_min = 0.5 - CENTER_ZONE_RATIO / 2
    cx_max = 0.5 + CENTER_ZONE_RATIO / 2
    cy_min = 0.5 - CENTER_ZONE_RATIO / 2
    cy_max = 0.5 + CENTER_ZONE_RATIO / 2

    center_hits = 0
    for c in centers:
        if c is not None:
            if cx_min <= c[0] <= cx_max and cy_min <= c[1] <= cy_max:
                center_hits += 1

    centered_ratio = center_hits / max(1, person_frame_count)

    # Final decision
    reasons = []
    video_valid = True

    # multi-person invalid
    if multi_person_flag:
        reasons.append(f"multiple_persons_detected (max={max_persons_in_frame})")
        video_valid = False

    if presence_ratio < MIN_PERSON_FRAMES_RATIO:
        reasons.append(f"person_presence_low ({presence_ratio:.2f})")
        video_valid = False

    if avg_shift > MAX_AVG_CENTER_SHIFT:
        reasons.append(f"too_much_movement ({avg_shift:.3f})")
        video_valid = False

    if avg_scale_change > MAX_AVG_BOX_SCALE_CHANGE:
        reasons.append(f"bbox_size_fluctuation ({avg_scale_change:.3f})")
        video_valid = False

    if centered_ratio < 0.5:
        reasons.append(f"not_centered_enough ({centered_ratio:.2f})")
        video_valid = False
    if person_frame_count == 0:
        reasons.append("no_person_detected")
        video_valid = False

    return {
        "video_valid": video_valid,
        "reasons": reasons,
        "person_frames": person_frame_count,
        "total_frames": total,
        "presence_ratio": presence_ratio,
        "avg_center_shift": avg_shift,
        "avg_box_scale_change": avg_scale_change,
        "centered_ratio": centered_ratio,
        "max_persons_in_frame": max_persons_in_frame
    }

# -----------------------
# Whisper transcription
# -----------------------
@st.cache_resource
def load_whisper_model(model_size: str = "small"):
    """
    Load OpenAI Whisper ASR model (local).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    return model, device

def transcribe_whisper(audio_path: str, model, device: str = "cpu") -> Dict[str, Any]:
    """
    Transcribe audio using OpenAI Whisper.
    """
    result = model.transcribe(audio_path, language="en", fp16=(device=="cuda"))
    text = result.get("text", "").strip()
    segments = result.get("segments", [])  # start/end times for chunks
    return {"text": text, "segments": segments}

# -----------------------
# Semantic similarity & toxicity improvements
# -----------------------
@st.cache_resource
def load_embedding_model(model_name: str = EMBEDDER_MODEL):
    return SentenceTransformer(model_name)

@st.cache_resource
def load_toxicity_pipelines(primary: str = TOXICITY_PRIMARY, fallback: str = TOXICITY_FALLBACK):
    """
    Load two toxicity models to reduce single-model sensitivity.
    """
    device_idx = 0 if torch.cuda.is_available() else -1
    primary_pipe = None
    fallback_pipe = None
    try:
        primary_pipe = pipeline("text-classification", model=primary, return_all_scores=True, device=device_idx)
    except Exception:
        primary_pipe = None
    try:
        fallback_pipe = pipeline("text-classification", model=fallback, return_all_scores=True, device=device_idx)
    except Exception:
        fallback_pipe = None
    return primary_pipe, fallback_pipe

def embed_text(text: str, embedder: SentenceTransformer) -> np.ndarray:
    # producing normalized vector
    vec = embedder.encode([text], convert_to_numpy=True)[0]
    # normalize
    nrm = np.linalg.norm(vec) + 1e-12
    return vec / nrm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run_toxic_model(text: str, tox_pipe) -> Tuple[str, float]:
    """
    Run a toxicity pipeline and return (label, score) for the top predicted label.
    If tox_pipe is None, returns ('none', 0.0)
    """
    if tox_pipe is None:
        return ("none", 0.0)
    try:
        out = tox_pipe(text[:1000])  # shorten text for model
        if isinstance(out, list) and len(out) > 0:
            scores = out[0]
            top = max(scores, key=lambda x: x.get("score", 0.0))
            return (str(top.get("label", "")).lower(), float(top.get("score", 0.0)))
    except Exception:
        pass
    return ("none", 0.0)

def check_toxicity_ensemble(transcript: str, primary_pipe, fallback_pipe) -> Tuple[bool, float, List[Tuple[str,float]]]:
    """
    Hybrid toxicity detection:
    - run primary and fallback models
    - use profanity list fallback
    - require ensemble agreement or strong probability to flag toxicity.
    Returns (is_abusive, max_confidence, list_of_results)
    """
    results = []
    # primary
    p_lab, p_sc = run_toxic_model(transcript, primary_pipe)
    results.append((p_lab, p_sc))
    # fallback
    f_lab, f_sc = run_toxic_model(transcript, fallback_pipe)
    results.append((f_lab, f_sc))

    # profanity fallback
    text_low = transcript.lower()
    profanity_found = any(bad in text_low for bad in PROFANITY_LIST)
    if profanity_found:
        results.append(("profanity_fallback", 1.0))

    # Decide ensemble voting: count detectors that report offensive/toxic with >= threshold
    votes = 0
    max_conf = 0.0
    for lab, sc in results:
        max_conf = max(max_conf, sc)
        if sc >= TOXICITY_MODEL_PROB_THRESHOLD:
            # consider label toxic/offensive if common keywords present
            if any(k in lab for k in ("toxic", "offensive", "abuse", "insult", "attack", "hate", "off")) or lab == "profanity_fallback":
                votes += 1

    # If profanity fallback present without strong model scores, consider it abusive immediately
    if any(lab == "profanity_fallback" for lab, _ in results):
        return True, 1.0, results

    is_abusive = votes >= ENSEMBLE_VOTE_THRESHOLD
    return is_abusive, max_conf, results

# -----------------------
# Resume & Transcript text handling
# -----------------------
def prepare_resume_text(resume_json: Dict[str, Any]) -> str:
    """
    Extracts comprehensive text from resume JSON so that transcript comparison becomes accurate
    for any resume structure.
    """
    parts = []

    def add(v):
        if not v:
            return
        if isinstance(v, str):
            v = v.strip()
            if len(v) > 2:
                parts.append(v)
        elif isinstance(v, list):
            for it in v:
                add(it)
        elif isinstance(v, dict):
            for _, sub in v.items():
                add(sub)
        else:
            parts.append(str(v))

    # Basic fields
    for key in ["name", "headline", "title", "role", "summary", "about", "profile"]:
        add(resume_json.get(key))

    # Generic scan: collect most common resume keys if present
    keys_to_try = [
        "technical_skills", "skills", "soft_skills",
        "experience", "work_experience", "internship_details",
        "projects", "certifications", "education_details",
        "achievements", "responsibilities", "profile_links"
    ]
    for key in keys_to_try:
        add(resume_json.get(key))

    # Also include any string fields nested in the JSON (safe shallow traversal)
    def shallow_collect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    add(v)
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, str):
                            add(it)
                        elif isinstance(it, dict):
                            shallow_collect(it)
                elif isinstance(v, dict):
                    shallow_collect(v)
    shallow_collect(resume_json)

    # Remove duplicates while preserving order
    seen = set()
    final = []
    for s in parts:
        if s not in seen:
            final.append(s)
            seen.add(s)
    return " ".join(final)

def extract_skill_list(resume_json: Dict[str, Any]) -> List[str]:
    # try multiple common keys
    candidates = []
    for key in ("technical_skills", "skills", "technicalSkills"):
        val = resume_json.get(key)
        if isinstance(val, list):
            candidates.extend(val)
        elif isinstance(val, str):
            candidates.extend([v.strip() for v in val.split(",") if v.strip()])
    clean = []
    for s in candidates:
        s_norm = re.sub(r"[^a-zA-Z0-9+#+\.\- ]", "", str(s)).lower().strip()
        if len(s_norm) >= 2:
            clean.append(s_norm)
    return list(dict.fromkeys(clean))

def extract_resume_entities(resume_json: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract role titles, company names, project titles/descriptions from the resume JSON.
    """
    roles = []
    companies = []
    projects = []
    # headline/title
    for key in ["headline", "title", "role", "summary"]:
        v = resume_json.get(key)
        if isinstance(v, str):
            roles.append(v)
    # experience blocks
    exp_keys = ["experience", "work_experience", "internship_details"]
    for k in exp_keys:
        for e in resume_json.get(k, []) or []:
            if isinstance(e, dict):
                if "role" in e and e["role"]:
                    roles.append(e["role"])
                if "company" in e and e["company"]:
                    companies.append(e["company"])
                # techs
                if "technologies_used" in e and isinstance(e["technologies_used"], list):
                    projects.append(", ".join(e["technologies_used"]))
                # responsibilities/descriptions
                if "responsibilities" in e and e["responsibilities"]:
                    projects.append(e["responsibilities"])
                if "achievements" in e and e["achievements"]:
                    projects.append(e["achievements"])
    # projects
    for p in resume_json.get("projects", []) or []:
        if isinstance(p, dict):
            if "title" in p and p["title"]:
                projects.append(p["title"])
            if "description" in p and p["description"]:
                projects.append(p["description"])
            if "technologies_used" in p and isinstance(p["technologies_used"], list):
                projects.append(", ".join(p["technologies_used"]))
    # certifications & education (can contain names of technologies)
    for c in resume_json.get("certifications", []) or []:
        if isinstance(c, dict):
            if "name" in c and c["name"]:
                projects.append(c["name"])
            if "description" in c and c["description"]:
                projects.append(c["description"])
    # cleanup
    def clean_list(lst):
        out = []
        for x in lst:
            if not x:
                continue
            s = re.sub(r'\s+', ' ', str(x)).strip()
            if len(s) > 2:
                out.append(s)
        return list(dict.fromkeys(out))
    return {
        "roles": clean_list(roles),
        "companies": clean_list(companies),
        "projects": clean_list(projects)
    }

def clean_transcript_text(text: str) -> str:
    """
    Normalizes transcript: collapse whitespace, strip filler-only fragments.
    """
    text = text.strip()
    # collapse repeated spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text

def chunk_transcript(text: str) -> List[str]:
    text = clean_transcript_text(text)
    words = text.split()
    chunks = []
    if not words:
        return chunks
    window = 25  # target chunk size
    stride = 15
    for i in range(0, max(1, len(words)), stride):
        chunk = " ".join(words[i:i+window])
        if len(chunk.split()) > 3:
            chunks.append(chunk)
    return chunks

def compute_similarity(resume_vec: np.ndarray, chunk_vecs: np.ndarray) -> float:
    """
    Compute aggregated similarity between resume vector and all chunk vectors.
    Uses top-K chunk similarities for robustness.
    """
    if chunk_vecs.shape[0] == 0:
        return 0.0
    sims = np.dot(chunk_vecs, resume_vec)  # can be negative; do not clip
    # if less than TOP_K_CHUNKS available, take what we have
    k = min(TOP_K_CHUNKS, sims.shape[0])
    topk = np.sort(sims)[-k:]
    agg = float(0.6 * np.mean(topk) + 0.4 * np.max(sims))
    return agg

def max_entity_similarity(entity_texts: List[str], chunk_vecs: np.ndarray, embedder: SentenceTransformer) -> float:
    """
    For a list of short entity texts (roles, companies, projects) compute
    maximum similarity against transcript chunk vectors.
    """
    if not entity_texts or chunk_vecs.shape[0] == 0:
        return 0.0
    # embed entity_texts
    entity_vecs = embedder.encode(entity_texts, convert_to_numpy=True)
    entity_vecs /= (np.linalg.norm(entity_vecs, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(chunk_vecs, entity_vecs.T)  # shape: (chunks, entities)
    max_sim = float(np.max(sims))
    return max_sim

def fuzzy_name_match(resume_name: str, transcript: str) -> Tuple[bool, float]:
    """
    Improved name matching:
    - Splits names into tokens
    - Removes spaces
    - Normalizes spelling differences (dalip ≈ dilip)
    - Performs per-word loose matching
    - Returns (matched, score 0–1)
    """
    import re
    from difflib import SequenceMatcher

    def normalize_word(w: str) -> str:
        w = w.lower()
        w = re.sub(r'[^a-z]', '', w)   # remove non-letters
        return w

    def loose_match(a: str, b: str) -> bool:
        """
        Word-level similarity:
        - direct match
        - OR similarity >= 0.8
        Example: dalip ≈ dilip (0.9)
        """
        a = normalize_word(a)
        b = normalize_word(b)
        if not a or not b:
            return False
        if a == b:
            return True
        sim = SequenceMatcher(None, a, b).ratio()
        return sim >= 0.80

    # Normalize full names
    rn = resume_name.lower()
    t = transcript.lower()

    # Extract tokens
    rn_tokens = [normalize_word(x) for x in rn.split() if normalize_word(x)]
    t_tokens = [normalize_word(x) for x in t.split() if normalize_word(x)]

    if not rn_tokens or not t_tokens:
        return False, 0.0

    # Count matches token-by-token
    matches = 0
    for r in rn_tokens:
        for tt in t_tokens:
            if loose_match(r, tt):
                matches += 1
                break

    score = matches / len(rn_tokens)

    return (score >= 0.60), score



# -----------------------
# Main analysis: audio <-> resume
# -----------------------
def analyze_audio_and_compare(audio_wav_path: str, transcript: str, resume_json: Dict[str, Any],
                              embedder: SentenceTransformer, primary_tox, fallback_tox) -> Dict[str, Any]:
    words = [w for w in (transcript or "").split() if w.strip()]
    if len(words) < MIN_TRANSCRIPT_WORDS:
        return {"audio_valid": False, "reasons": ["no_or_too_short_speech"], "transcript": transcript}

    chunks = chunk_transcript(transcript)
    if not chunks:
        return {"audio_valid": False, "reasons": ["no_valid_chunks"], "transcript": transcript}

    # Prepare resume and entities
    resume_text = prepare_resume_text(resume_json)
    resume_skills = extract_skill_list(resume_json)
    resume_entities = extract_resume_entities(resume_json)
    resume_name = resume_json.get("name", "") or resume_json.get("full_name", "") or ""

    # embeddings
    resume_vec = embed_text(resume_text, embedder)
    chunk_vecs = embedder.encode(chunks, convert_to_numpy=True)
    chunk_vecs /= (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12)

    aggregated_sim = compute_similarity(resume_vec, chunk_vecs)

    # Skill matching (per-skill max similarity)
    skill_matches = []
    if resume_skills:
        skill_vecs = embedder.encode(resume_skills, convert_to_numpy=True)
        skill_vecs /= (np.linalg.norm(skill_vecs, axis=1, keepdims=True) + 1e-12)
        sims_matrix = np.dot(chunk_vecs, skill_vecs.T)
        for i, skill in enumerate(resume_skills):
            max_sim = float(np.max(sims_matrix[:, i]))
            skill_matches.append({"skill": skill, "max_sim": max_sim})
    skills_mentioned = [
    s for s in skill_matches 
    if s["max_sim"] >= SKILL_SIMILARITY_THRESHOLD
    ]


    # Entity (role/company/project) matching
    role_sim = max_entity_similarity(resume_entities.get("roles", []), chunk_vecs, embedder)
    company_sim = max_entity_similarity(resume_entities.get("companies", []), chunk_vecs, embedder)
    project_sim = max_entity_similarity(resume_entities.get("projects", []), chunk_vecs, embedder)

    # Toxicity
    is_abusive, tox_conf, tox_results = check_toxicity_ensemble(transcript, primary_tox, fallback_tox)

    # Name verification (exact or fuzzy)
    # -------- HYBRID NAME CHECK (fuzzy + embedding) --------
    name_matched_fuzzy, name_score_fuzzy = fuzzy_name_match(resume_name, transcript)

    # embedding similarity for name
    if resume_name:
        name_vec = embed_text(resume_name, embedder)
        name_sim = float(np.max(np.dot(chunk_vecs, name_vec)))
    else:
        name_sim = 0.0

    # final decision
    name_matched = (name_matched_fuzzy or name_sim >= 0.65)
    name_score = max(name_score_fuzzy, name_sim)
    # --------------------------------------------------------


    # Compose reasons
    reasons = []
    if is_abusive:
        reasons.append(f"toxic_or_abusive (conf={tox_conf:.2f})")
    # semantic not related AND few skills mentioned
    if aggregated_sim < SIMILARITY_THRESHOLD and len(skills_mentioned) < max(1, int(len(resume_skills) * 0.15)):
        reasons.append(f"semantic_not_related ({aggregated_sim:.3f})")

    # Decide audio_valid:
    # Rule:
    #  - name must match (fuzzy or exact) OR NAME_EXACT_REQUIRED False and name not present -> fail if name not matched
    #  - AND (aggregated_sim >= SIMILARITY_THRESHOLD OR at least one category matches strongly (role/company/project) OR enough skills matched)
    enough_skills = len(skills_mentioned) >= max(1, int(len(resume_skills) * 0.15))
    category_hit = (role_sim >= ROLE_SIM_THRESHOLD) or (company_sim >= COMPANY_SIM_THRESHOLD) or (project_sim >= PROJECT_SIM_THRESHOLD)

    semantic_hit = aggregated_sim >= SIMILARITY_THRESHOLD

    # If name check enabled strictly:
    if NAME_EXACT_REQUIRED:
        name_ok = name_matched
    else:
        # prefer name match, but allow passing if semantic + category are strong even without name (configurable)
        name_ok = name_matched

    # FINAL HYBRID DECISION:
    audio_valid = (
        not is_abusive
        and name_matched
        and (
            aggregated_sim >= SIMILARITY_THRESHOLD     # semantic summary match
            or category_hit                            # role/company/project match
            or enough_skills                           # skill match
        )
    )


    return {
        "audio_valid": audio_valid,
        "reasons": reasons,
        "similarity": aggregated_sim,
        "skill_matches": skill_matches,
        "skills_mentioned": skills_mentioned,
        "role_similarity": role_sim,
        "company_similarity": company_sim,
        "project_similarity": project_sim,
        "name_matched": name_matched,
        "name_match_score": name_score,
        "is_abusive": is_abusive,
        "tox_results": tox_results,
        "transcript": transcript,
        "resume_text_preview": resume_text[:4000]
    }

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Video + Resume Validator (Improved)", layout="wide")
st.title("Video + Resume Validator — YOLOv8 + Whisper + Semantic (Improved)")

with st.form("inputs"):
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "webm", "mkv"])
    resume_file = st.file_uploader("Upload resume JSON", type=["json"])
    submit = st.form_submit_button("Validate")

if submit:
    if video_file is None or resume_file is None:
        st.error("Both video and resume JSON are required.")
        st.stop()

    # save files
    video_path = save_uploaded_file(video_file)
    resume_path = save_uploaded_file(resume_file)
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_json = json.load(f)
    except Exception as e:
        st.error(f"Invalid resume JSON: {e}")
        st.stop()

    # Load models (with caching)
    st.info("Loading models (YOLOv8, Whisper, embeddings, toxicity). This may take time on first run.")
    # YOLO
    yolo = load_yolo()

    # Whisper
    whisper_model, device = load_whisper_model(model_size="small")

    # Embeddings and toxicity
    embedder = load_embedding_model(EMBEDDER_MODEL)
    primary_tox, fallback_tox = load_toxicity_pipelines()

    # 1) Video analysis
    st.info("Analyzing video (YOLOv8)...")
    t0 = time.time()
    try:
        video_metrics = analyze_video_yolo(video_path, yolo)
    except Exception as e:
        st.error(f"Video analysis failed: {e}")
        st.stop()
    t1 = time.time()
    st.success(f"Video analysis finished in {t1-t0:.1f}s")

    st.write("Video metrics:")
    st.json({
        "video_valid": video_metrics["video_valid"],
        "person_frames": video_metrics["person_frames"],
        "total_frames": video_metrics["total_frames"],
        "presence_ratio": video_metrics["presence_ratio"],
        "avg_center_shift": video_metrics["avg_center_shift"],
        "avg_box_scale_change": video_metrics["avg_box_scale_change"],
        "centered_ratio": video_metrics["centered_ratio"],
        "max_persons_in_frame": video_metrics["max_persons_in_frame"],
        "reasons": video_metrics["reasons"]
    })

    # 2) Audio: extract and transcribe using Whisper
    st.info("Extracting audio and transcribing with Whisper...")
    audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        extract_audio(video_path, audio_tmp)
    except Exception as e:
        st.error(f"Audio extraction failed: {e}")
        st.stop()

    t0 = time.time()
    try:
        trans_res = transcribe_whisper(audio_tmp, whisper_model, device=device)
        transcript_text = trans_res.get("text", "")
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        transcript_text = ""
    t1 = time.time()
    st.success(f"Transcription finished in {t1-t0:.1f}s")

    st.write("Transcript (first 5000 chars):")
    st.text_area("Transcript", value=(transcript_text[:5000] + ("..." if len(transcript_text) > 5000 else "")), height=250)

    # 3) Semantic similarity + toxicity checks
    st.info("Checking semantic similarity between transcript and resume and running toxicity detection...")
    audio_metrics = analyze_audio_and_compare(audio_tmp, transcript_text, resume_json, embedder, primary_tox, fallback_tox)
    st.write("Audio metrics:")
    st.json(audio_metrics)

    # 4) Final decision: AND combine video and audio checks
    final_accept = video_metrics.get("video_valid", False) and audio_metrics.get("audio_valid", False)

    st.write("## Final Decision")
    if final_accept:
        st.success("✅ VIDEO ACCEPTED — both video and audio checks passed.")
    else:
        st.error("❌ VIDEO REJECTED — video and/or audio checks failed.")
        st.write("Combined reasons:")
        st.json({"video_reasons": video_metrics.get("reasons", []), "audio_reasons": audio_metrics.get("reasons", [])})

    # show uploaded video (if supported)
    try:
        st.write("Uploaded video:")
        st.video(video_path)
    except Exception:
        st.write("Cannot display video inline.")
    # cleanup audio temp file
    try:
        os.remove(audio_tmp)
    except Exception:
        pass
