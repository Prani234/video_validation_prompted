# Video Resume Validator - Complete Flow Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VIDEO RESUME VALIDATOR SYSTEM                             │
│                                                                              │
│  Input: Video File + Resume JSON                                           │
│  Output: ACCEPTED ✅ or REJECTED ❌                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Main Processing Flow

```
                         START
                           │
                           ▼
               ┌───────────────────────┐
               │  Load User Inputs:    │
               │  - Video File         │
               │  - Resume JSON        │
               └───────┬───────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   ┌──────────────────┐    ┌─────────────────┐
   │  VIDEO ANALYSIS  │    │ AUDIO ANALYSIS  │
   │   (YOLOv8)       │    │ (Whisper + AI)  │
   └────────┬─────────┘    └────────┬────────┘
            │                       │
            ▼                       ▼
   ┌──────────────────┐    ┌──────────────────┐
   │  video_valid     │    │  audio_valid     │
   │  Boolean Result  │    │  Boolean Result  │
   └────────┬─────────┘    └─────────┬────────┘
            │                        │
            └────────────┬───────────┘
                         │
                    COMBINE RESULTS
                         │
                         ▼
            ┌──────────────────────────┐
            │  FINAL DECISION:         │
            │  video_valid AND         │
            │  audio_valid = ACCEPT    │
            └──────────────┬───────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                   YES           NO
                    │             │
                    ▼             ▼
              ✅ ACCEPTED    ❌ REJECTED
```

---

## Phase 1: VIDEO ANALYSIS (YOLOv8)

```
                    VIDEO FILE
                       │
                       ▼
        ┌──────────────────────────┐
        │ Load YOLOv8 Model        │
        │ (yolov8n.pt)             │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Sample Frames            │
        │ (at FRAME_SAMPLE_FPS=1)  │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ For Each Frame:          │
        │ - Run YOLOv8 detection   │
        │ - Identify "person" bbox │
        │ - Count detections       │
        └──────────┬───────────────┘
                   │
      ┌────────────┴────────────┐
      │                         │
      ▼                         ▼
┌────────────────┐      ┌──────────────────┐
│ NO PERSON      │      │ 1 PERSON         │
│ DETECTED       │      │ DETECTED         │
└────────┬───────┘      └────────┬─────────┘
         │                       │
         │              ┌────────┴────────┐
         │              │                 │
         │              ▼                 ▼
         │       ┌────────────────┐ ┌────────────────┐
         │       │ Track Center   │ │ Track Bounding │
         │       │ Coordinates:   │ │ Box Size       │
         │       │ (cx, cy)       │ │ (area)         │
         │       └────────┬───────┘ └────────┬───────┘
         │                │                   │
         │                └─────────┬─────────┘
         │                          │
         │                          ▼
         │            ┌──────────────────────────┐
         │            │ Calculate Movement       │
         │            │ Metrics per Frame:       │
         │            │ - Center shift (dx, dy)  │
         │            │ - Scale change (area)    │
         │            └──────────┬───────────────┘
         │                       │
         ▼                       ▼
    ┌─────────────────────────────────┐
    │ Compute Aggregate Metrics:      │
    │ • presence_ratio                │
    │ • avg_center_shift              │
    │ • avg_box_scale_change          │
    │ • centered_ratio                │
    │ • max_persons_in_frame          │
    └────────┬────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────┐
    │ VALIDATION CHECKS:               │
    │                                  │
    │ ❌ FAIL IF:                      │
    │ 1. Multiple persons detected     │
    │ 2. presence_ratio <              │
    │    MIN_PERSON_FRAMES_RATIO       │
    │    (default: 0.6)                │
    │ 3. avg_center_shift >            │
    │    MAX_AVG_CENTER_SHIFT          │
    │    (default: 0.12)               │
    │ 4. avg_box_scale_change >        │
    │    MAX_AVG_BOX_SCALE_CHANGE      │
    │    (default: 0.20)               │
    │ 5. centered_ratio < 0.5          │
    │                                  │
    │ ✅ PASS IF: All checks above OK  │
    └────────┬─────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │ video_valid:     │
    │ True/False       │
    │ + Failure Reasons│
    └──────────────────┘
```

---

## Phase 2: AUDIO ANALYSIS (Whisper + Semantic)

```
                    VIDEO FILE
                       │
                       ▼
        ┌──────────────────────────┐
        │ Extract Audio from Video │
        │ (ffmpeg → 16kHz mono WAV)│
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Load Whisper Model       │
        │ (openai/whisper - small) │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Transcribe Audio to Text │
        │ (English language)       │
        └──────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Clean & Prepare Text:    │
        │ - Collapse whitespace    │
        │ - Strip HTML/special chars│
        └──────────┬───────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
┌──────────────────┐  ┌──────────────────────┐
│ Check if Speech  │  │ Chunk Transcript:    │
│ Exists:          │  │ - Window: 25 words   │
│ words >=         │  │ - Stride: 15 words   │
│ MIN_TRANSCRIPT   │  │ - Min chunk: 3 words │
│ WORDS (3)        │  │                      │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
    ┌────┴────┐            ┌─────┴─────┐
    │          │            │           │
   NO         YES          │          NO
    │          │            │           │
    ▼          │            ▼           ▼
 FAIL         │        FAIL      Embed Each
              │                  Chunk
              │
              └────────┬──────────────┐
                       │              │
                       ▼              ▼
        ┌──────────────────────────────────────┐
        │ Load Embedding Model:                │
        │ (SentenceTransformer:                │
        │  all-mpnet-base-v2)                  │
        └──────────┬───────────────────────────┘
                   │
       ┌───────────┴──────────────────┐
       │                              │
       ▼                              ▼
┌─────────────────┐      ┌──────────────────────────┐
│ Prepare Resume: │      │ Embed Chunks to          │
│ Extract all     │      │ Vector Space:            │
│ text fields     │      │ - Normalize vectors      │
│ from JSON       │      │ - Shape: (N_chunks, 768)│
└────────┬────────┘      └──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Extract Resume Entities: │
│ • Skills list            │
│ • Roles/titles           │
│ • Company names          │
│ • Project descriptions   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Embed Resume Text        │
│ & Entities to Vectors    │
│ - Normalize              │
└────────┬─────────────────┘
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
┌──────────────────────┐  ┌─────────────────────────┐
│ SIMILARITY CHECK:    │  │ SKILL MATCHING:         │
│                      │  │                         │
│ Compare Resume Vec   │  │ For each skill:         │
│ with Chunk Vecs      │  │ - Find max similarity   │
│                      │  │   in transcript chunks  │
│ Aggregate using:     │  │ - Mark if >= SKILL_SIM_ │
│ 60% Mean TopK +      │  │   THRESHOLD (0.55)      │
│ 40% Max Similarity   │  │                         │
│                      │  │ Count: skills_mentioned │
│ Result:              │  │                         │
│ aggregated_sim       │  │                         │
└──────────────┬───────┘  └────────────┬────────────┘
               │                       │
               │                       ▼
               │              ┌─────────────────────┐
               │              │ Entity Similarity:  │
               │              │                     │
               │              │ • role_similarity   │
               │              │ • company_sim       │
               │              │ • project_sim       │
               │              │                     │
               │              │ Compare each entity │
               │              │ against chunk vecs  │
               │              │ → max similarity    │
               │              └────────┬────────────┘
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ Load Toxicity Detectors:         │
        │ • Primary: toxic-comment-model   │
        │ • Fallback: (configurable)       │
        │ • Profanity list (fallback)      │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ Ensemble Toxicity Detection:     │
        │                                  │
        │ 1. Run primary model             │
        │ 2. Run fallback model            │
        │ 3. Check profanity list          │
        │ 4. Voting:                       │
        │    - Count toxic votes >= 0.60   │
        │    - Need 2+ votes to flag       │
        │    - OR immediate flag if        │
        │      profanity found             │
        │                                  │
        │ Result: is_abusive (Bool)        │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ HYBRID Name Matching:            │
        │                                  │
        │ Method 1: Fuzzy Token Matching   │
        │ • Normalize name tokens          │
        │ • Remove non-letters             │
        │ • Token-to-token loose match     │
        │   (0.80 similarity threshold)    │
        │ • Count matches / name tokens    │
        │ • Pass if >= 0.60 ratio          │
        │                                  │
        │ Method 2: Embedding Similarity   │
        │ • Embed resume name as vector    │
        │ • Find max similarity vs chunks  │
        │ • Pass if >= 0.65 similarity     │
        │                                  │
        │ FINAL: name_matched = Method1   │
        │        OR Method2                │
        │ name_score = max(score1, score2)│
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────┐
        │ FINAL AUDIO VALIDATION RULES:    │
        │                                  │
        │ ✅ PASS IF ALL conditions:      │
        │ 1. NOT is_abusive                │
        │ 2. name_matched = True           │
        │ 3. AND at least ONE of:          │
        │    • aggregated_sim >=           │
        │      SIMILARITY_THRESHOLD (0.55) │
        │    OR                            │
        │    • category_hit (any of:       │
        │      role_sim >= 0.50 OR         │
        │      company_sim >= 0.55 OR      │
        │      project_sim >= 0.50)        │
        │    OR                            │
        │    • enough_skills               │
        │      (>= 15% of resume skills    │
        │       similarity matched)        │
        │                                  │
        │ ❌ FAIL IF ANY:                 │
        │ • is_abusive = True              │
        │ • name_matched = False           │
        │ • None of the 3 semantic         │
        │   conditions above met           │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ audio_valid:     │
        │ True/False       │
        │ + Details        │
        └──────────────────┘
```

---

## Phase 3: FINAL DECISION

```
        ┌─────────────────────────────────┐
        │ Combine Results:                │
        │                                 │
        │ final_accept =                  │
        │   video_valid AND audio_valid   │
        └─────────────┬───────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
           YES                 NO
            │                   │
            ▼                   ▼
        ┌──────────┐      ┌─────────────┐
        │ ✅       │      │ ❌ REJECTED │
        │ ACCEPTED │      │             │
        │          │      │ Show both:  │
        │          │      │ • Video     │
        │          │      │   reasons   │
        │          │      │ • Audio     │
        │          │      │   reasons   │
        └──────────┘      └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Cleanup:     │
                        │ Remove temp  │
                        │ files        │
                        └──────────────┘
```

---

## Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION MODELS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YOLOv8                  Whisper              SentenceTransformer│
│  └─ yolov8n.pt          └─ small              └─ all-mpnet-    │
│     (object detection)      (speech-to-text)      base-v2      │
│     • Detects persons       • Transcribes       (semantic      │
│     • Tracks movement       • English only      embeddings)    │
│     • Validates position                                       │
│                                                                 │
│  Toxicity Detectors                                            │
│  ├─ Primary: toxic-comment-model                              │
│  └─ Fallback: (optional)                                      │
│     • Ensemble voting                                          │
│     • Profanity list fallback                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Thresholds & Configuration

```
┌─────────────────────────────────────────────────────┐
│        VIDEO ANALYSIS THRESHOLDS                    │
├─────────────────────────────────────────────────────┤
│ MIN_PERSON_FRAMES_RATIO = 0.6                       │
│   (Person must be visible in 60% of frames)         │
│                                                     │
│ CENTER_ZONE_RATIO = 0.35                            │
│   (±35% from center is "in zone")                   │
│                                                     │
│ MAX_AVG_CENTER_SHIFT = 0.12                         │
│   (Max normalized movement in x/y direction)        │
│                                                     │
│ MAX_AVG_BOX_SCALE_CHANGE = 0.20                     │
│   (Max relative change in bounding box size)        │
│                                                     │
│ FRAME_SAMPLE_FPS = 1                                │
│   (Sample 1 frame per second)                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│      AUDIO ANALYSIS THRESHOLDS                      │
├─────────────────────────────────────────────────────┤
│ SIMILARITY_THRESHOLD = 0.55                         │
│   (Overall resume-transcript similarity)            │
│                                                     │
│ SKILL_SIMILARITY_THRESHOLD = 0.55                   │
│   (Per-skill match threshold)                       │
│                                                     │
│ TOP_K_CHUNKS = 3                                    │
│   (Use top 3 chunks for aggregation)                │
│                                                     │
│ ROLE_SIM_THRESHOLD = 0.50                           │
│ COMPANY_SIM_THRESHOLD = 0.55                        │
│ PROJECT_SIM_THRESHOLD = 0.50                        │
│   (Category-specific thresholds)                    │
│                                                     │
│ TOXICITY_MODEL_PROB_THRESHOLD = 0.60                │
│   (Individual model confidence cutoff)              │
│                                                     │
│ ENSEMBLE_VOTE_THRESHOLD = 2                         │
│   (Number of detectors that must agree)             │
│                                                     │
│ MIN_TRANSCRIPT_WORDS = 3                            │
│   (Minimum words needed to process)                 │
│                                                     │
│ FUZZY_NAME_THRESHOLD = 0.60                         │
│   (Sequence matching threshold for name)            │
└─────────────────────────────────────────────────────┘
```

---

## Error Handling & Edge Cases

```
┌──────────────────────────────────────────────────────┐
│ VIDEO FAILURES                                       │
├──────────────────────────────────────────────────────┤
│ • no_frames_extracted                               │
│ • multiple_persons_detected                         │
│ • person_presence_low                               │
│ • too_much_movement                                 │
│ • bbox_size_fluctuation                             │
│ • not_centered_enough                               │
│ • no_person_detected                                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ AUDIO FAILURES                                       │
├──────────────────────────────────────────────────────┤
│ • no_or_too_short_speech                            │
│ • no_valid_chunks                                   │
│ • toxic_or_abusive                                  │
│ • semantic_not_related                              │
│ • name_not_matched                                  │
│ • insufficient_skill_coverage                       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ GRACEFUL DEGRADATION                                │
├──────────────────────────────────────────────────────┤
│ • If toxicity model fails → profanity fallback      │
│ • If fallback also fails → pass toxic check         │
│ • If name not in resume → allow pass if semantic ok│
│ • If chunks not created → reject as no speech      │
│ • ffmpeg errors → fail audio extraction            │
│ • Invalid JSON → reject immediately                │
└──────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
INPUT                           PROCESSING                    OUTPUT
─────                           ──────────                    ──────

Video File ──┐
             ├─→ YOLOv8  ──→ Video Metrics ──┐
             │                               │
             └─→ ffmpeg  ──→ Audio WAV       ├─→ Final Decision
                                │             │   (Accept/Reject)
                                ▼             │
                        Whisper Model   ──→ Transcript ──┐
                                │                        │
                                ▼                        │
                        SentenceTransformer ─────────────┤
                                │                        │
                                ▼                        │
Resume JSON ────┐       Semantic Similarity             │
                ├─→ Entity Extraction & Embedding ──────┤
                │       Toxicity Detection ──────────────┘
                │       Name Matching
                └────→ Resume Metrics
```

---

## Performance Notes

```
┌────────────────────────────────────────────────────┐
│ TYPICAL EXECUTION TIMES                            │
├────────────────────────────────────────────────────┤
│ YOLOv8 frame sampling & analysis:      1-10 sec   │
│ Audio extraction (ffmpeg):              2-5 sec    │
│ Whisper transcription (small model):   10-60 sec   │
│ Embedding & similarity computation:     5-20 sec   │
│ Toxicity detection:                     3-10 sec   │
│                                                    │
│ Total: ~30-120 seconds per video                   │
│ (First run may be slower due to model loading)     │
└────────────────────────────────────────────────────┘
```

---

## Streamlit UI Flow

```
User Interface                  Backend Operations
──────────────                  ──────────────────

Render Upload Form ──┐
  (video + resume)   │
                     ├─→ Validate Inputs (both required)
                     │
                     ├─→ Save Temp Files
                     │
                     ├─→ Load & Parse Resume JSON
                     │
                     ├─→ Load All Models (with caching)
                     │   ├─ YOLOv8
                     │   ├─ Whisper (small)
                     │   ├─ SentenceTransformer
                     │   └─ Toxicity pipelines
                     │
        ┌────────────┘
        │
        ▼
  "Analyzing video (YOLOv8)..."
  │
  ├─→ Video Analysis ──→ Display Video Metrics JSON
  │   • person_frames
  │   • total_frames
  │   • presence_ratio
  │   • avg_center_shift
  │   • avg_box_scale_change
  │   • centered_ratio
  │   • max_persons_in_frame
  │   • reasons (failure list)
  │
  ▼
  "Extracting audio and transcribing with Whisper..."
  │
  ├─→ Extract Audio (ffmpeg)
  │   └─ 16kHz mono WAV
  │
  ├─→ Transcribe (Whisper small)
  │   └─ English language
  │
  ├─→ Display Transcript (first 5000 chars)
  │
  ▼
  "Checking semantic similarity & toxicity..."
  │
  ├─→ Audio & Resume Analysis ──→ Display Audio Metrics JSON
  │   • similarity (aggregated)
  │   • skill_matches (all skills with scores)
  │   • skills_mentioned (filtered >= threshold)
  │   • role_similarity
  │   • company_similarity
  │   • project_similarity
  │   • name_matched (bool)
  │   • name_match_score (float)
  │   • is_abusive (bool)
  │   • tox_results (list of detections)
  │   • reasons (failure list)
  │
  ▼
  "Final Decision"
  │
  ├─→ Combine video_valid AND audio_valid
  │
  ├─→ If ACCEPTED (✅):
  │   └─ Show success message
  │   └─ Display uploaded video
  │
  ├─→ If REJECTED (❌):
  │   └─ Show error message
  │   └─ Display combined failure reasons:
  │       • video_reasons
  │       • audio_reasons
  │   └─ Display uploaded video
  │
  ▼
  Cleanup temp files
  │
  └─→ Session ends

```

---

## Detailed Fuzzy Name Matching Algorithm

```
Input: resume_name (string), transcript (string)
Output: (matched: bool, score: float 0-1)

STEP 1: Define Helper Functions
├─ normalize_word(w: str) -> str
│  └─ Convert to lowercase
│  └─ Remove non-alphabetic characters
│  └─ Example: "David-123" → "david"
│
└─ loose_match(a: str, b: str) -> bool
   ├─ Normalize both words
   ├─ Check exact match: a == b
   ├─ Calculate SequenceMatcher similarity
   ├─ Return True if similarity >= 0.80
   └─ Example: "dilip" ≈ "dalip" (0.9 similarity) → True

STEP 2: Tokenize and Normalize
├─ Convert both to lowercase
├─ Split resume_name into tokens
│  └─ Remove empty tokens after normalization
│  └─ Example: "John Smith-Jr" → ["john", "smith", "jr"]
├─ Split transcript into tokens
│  └─ Example: "My name is John Smith" → ["my", "name", "is", "john", "smith"]

STEP 3: Token-by-Token Matching
├─ For each resume_name token:
│  ├─ Try to match against ALL transcript tokens
│  ├─ If loose_match(resume_token, transcript_token) == True:
│  │  └─ Increment match counter
│  │  └─ Break (move to next resume token)
│  └─ If no match found, move to next resume token
├─ Example:
│  ├─ "john" matches "john" in transcript ✓
│  ├─ "smith" matches "smith" in transcript ✓
│  ├─ "jr" cannot match anything ✗
│  └─ matches = 2

STEP 4: Calculate Score
├─ score = matches / len(resume_name_tokens)
├─ Example: score = 2 / 3 = 0.667

STEP 5: Final Decision
├─ If score >= 0.60: matched = True
├─ Else: matched = False
├─ Return (matched, score)
└─ Example: (True, 0.667)

```

---

## Hybrid Name Matching Decision

```
Combining Fuzzy Matching + Embedding Similarity:

╔════════════════════════════════════════════════════════════════╗
║ METHOD 1: FUZZY TOKEN MATCHING (fuzzy_name_match)            ║
╠════════════════════════════════════════════════════════════════╣
║ • Token normalization & loose matching (0.80 threshold)        ║
║ • Tolerance for spelling variations (dalip ≈ dilip)            ║
║ • Returns: (matched_fuzzy, score_fuzzy)                        ║
║ • Pass threshold: >= 0.60                                      ║
║                                                                ║
║ Example:                                                       ║
║  Resume: "Dilip Sharma"  Transcript: "My name is Dilip Sharma" ║
║  Tokens: ["dilip", "sharma"]  Matches: 2/2 = 1.0              ║
║  Result: matched_fuzzy = True, score_fuzzy = 1.0              ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║ METHOD 2: EMBEDDING SIMILARITY                                ║
╠════════════════════════════════════════════════════════════════╣
║ • Embed resume name to vector (768-dim)                        ║
║ • Compare against chunk vectors                                ║
║ • Find max cosine similarity                                   ║
║ • Pass threshold: >= 0.65                                      ║
║                                                                ║
║ Example:                                                       ║
║  Resume: "Dilip Sharma"  (embedded)                            ║
║  Chunks: ["My name is Dilip Sharma"]  (embedded)               ║
║  Similarity: 0.87 (high semantic match)                        ║
║  Result: name_sim = 0.87 >= 0.65 → True                       ║
╚════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════╗
║ FINAL HYBRID DECISION                                          ║
╠════════════════════════════════════════════════════════════════╣
║ name_matched = matched_fuzzy OR name_sim >= 0.65              ║
║ name_score = max(score_fuzzy, name_sim)                        ║
║                                                                ║
║ This approach:                                                 ║
║ • Catches exact/partial name mentions (fuzzy)                 ║
║ • Catches semantic name presence (embedding)                  ║
║ • Robust to spelling variations                               ║
║ • Tolerant of context variations                              ║
╚════════════════════════════════════════════════════════════════╝

```

---

## Skill Matching & "Enough Skills" Logic

```
Input: resume_skills (list of skills), chunk_vecs (embeddings), transcript
Output: skill_matches (list), skills_mentioned (filtered list)

STEP 1: Extract Skills from Resume
├─ Look for common keys:
│  ├─ "technical_skills"
│  ├─ "skills"
│  ├─ "technicalSkills"
├─ Normalize each skill:
│  ├─ Remove special characters (keep alphanumeric, +, #, ., -)
│  ├─ Convert to lowercase
│  ├─ Remove leading/trailing whitespace
│  └─ Keep only if length >= 2
├─ Remove duplicates while preserving order
└─ Example: ["Python", "C++", "Java", "Machine Learning"]

STEP 2: Embed All Skills
├─ Use SentenceTransformer to encode skills
├─ Shape: (num_skills, 768) dimensional vectors
├─ Normalize each skill vector to unit length
└─ Example: ["Python"] → [0.15, -0.23, 0.67, ..., 0.12] (768 dims)

STEP 3: Calculate Similarity Matrix
├─ Compute dot product: chunk_vecs @ skill_vecs.T
├─ Shape: (num_chunks, num_skills)
├─ Each cell = cosine similarity between chunk i and skill j
└─ Example:
   Chunks:    ["talking about python programming"]
   Skills:    ["Python", "C++", "Java"]
   Matrix:    [[0.78, 0.15, 0.22]]

STEP 4: Find Max Similarity per Skill
├─ For each skill column:
│  └─ Find maximum similarity across all chunks
├─ Result: skill_matches list
│  ├─ {"skill": "Python", "max_sim": 0.78}
│  ├─ {"skill": "C++", "max_sim": 0.15}
│  └─ {"skill": "Java", "max_sim": 0.22}

STEP 5: Filter Skills Meeting Threshold
├─ SKILL_SIMILARITY_THRESHOLD = 0.55
├─ skills_mentioned = [s for s in skill_matches 
│                      if s["max_sim"] >= 0.55]
├─ Result: [{"skill": "Python", "max_sim": 0.78}]
└─ Only "Python" mentioned, not C++ or Java

STEP 6: "Enough Skills" Decision
├─ Calculate minimum requirement:
│  └─ min_required = max(1, 15% of num_resume_skills)
│  └─ Example: 4 skills → require >= 1 mentioned
├─ Count: len(skills_mentioned) >= min_required
├─ Example: 1 >= 1 → True (enough_skills = True)
└─ This is one of three ways to pass audio validation

```

---

## Key Concepts

| Concept | Purpose |
|---------|---------|
| **Frame Sampling** | Reduce computation by analyzing every nth frame instead of all frames |
| **Semantic Embeddings** | Convert text to vector form for similarity comparison (768-dimensional) |
| **Ensemble Toxicity** | Combine multiple models + profanity list to reduce false positives |
| **Chunking** | Break transcript into overlapping windows for better granular matching |
| **Normalization** | Scale embeddings to unit vectors for consistent cosine similarity |
| **Top-K Aggregation** | Use average of top 3 similarities + max for robust score |
| **Fuzzy Name Matching** | Allow variations in name spelling/format |
| **AND Logic** | Both video AND audio must pass for final acceptance |

---

## Model Versions & Resources

```
┌──────────────────────────────────────────────────────────┐
│ REQUIRED FILES & MODELS                                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ yolov8n.pt (64 MB)                                       │
│ └─ YOLOv8 nano weights                                  │
│                                                          │
│ Whisper "small" (~466 MB)                                │
│ └─ Downloaded on first run                              │
│                                                          │
│ all-mpnet-base-v2 (~430 MB)                              │
│ └─ From SentenceTransformers (downloaded)                │
│                                                          │
│ Toxicity Models                                          │
│ └─ martin-ha/toxic-comment-model (~500 MB)               │
│                                                          │
│ vad_model.pth (optional)                                 │
│ └─ Not currently used in flow                            │
│                                                          │
│ Total disk space needed: ~2 GB (with all models)         │
│ RAM required: 8+ GB (for model inference)                │
│ GPU (recommended): CUDA-capable NVIDIA                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```


---

## Output Data Structures

### Video Analysis Result

```python
video_metrics = {
    "video_valid": bool,              # True if all checks pass
    "reasons": List[str],             # List of failure reasons (empty if valid)
    "person_frames": int,             # Number of frames with exactly 1 person
    "total_frames": int,              # Total sampled frames
    "presence_ratio": float,          # person_frames / total_frames (0-1)
    "avg_center_shift": float,        # Average movement in normalized coords (0-1)
    "avg_box_scale_change": float,    # Average relative size change (0-1)
    "centered_ratio": float,          # Frames where person in center zone (0-1)
    "max_persons_in_frame": int       # Maximum people detected in any frame
}
```

### Audio Analysis Result

```python
audio_metrics = {
    "audio_valid": bool,              # True if all checks pass
    "reasons": List[str],             # List of failure reasons (empty if valid)
    "similarity": float,              # Aggregated resume-transcript similarity (0-1)
    "skill_matches": List[Dict],      # All skills with their max similarities
                                      # [{"skill": str, "max_sim": float}, ...]
    "skills_mentioned": List[Dict],   # Filtered skills meeting threshold
                                      # [{"skill": str, "max_sim": float}, ...]
    "role_similarity": float,         # Max similarity of any role to transcript (0-1)
    "company_similarity": float,      # Max similarity of any company to transcript (0-1)
    "project_similarity": float,      # Max similarity of any project to transcript (0-1)
    "name_matched": bool,             # True if name found (fuzzy or embedding)
    "name_match_score": float,        # Max score from both methods (0-1)
    "is_abusive": bool,               # True if toxicity detected
    "tox_results": List[Tuple],       # Detailed toxicity detection results
                                      # [("label", score), ...]
    "transcript": str,                # Full transcribed text
    "resume_text_preview": str        # First 4000 chars of extracted resume
}
```

### Final Decision Result

```python
final_accept = video_valid AND audio_valid  # Boolean: True = Accept, False = Reject

# If REJECTED, show:
rejection_reasons = {
    "video_reasons": List[str],       # Why video failed
    "audio_reasons": List[str]        # Why audio failed
}
```

---

## Failure Reason Codes

### Video Failure Reasons
- `no_frames_extracted` - Video has 0 sampled frames
- `multiple_persons_detected` - More than 1 person in any frame
- `person_presence_low` - Less than 60% frames have 1 person
- `too_much_movement` - Average center shift > 0.12
- `bbox_size_fluctuation` - Average scale change > 0.20
- `not_centered_enough` - Less than 50% of frames have person in center zone
- `no_person_detected` - Zero persons detected in all frames

### Audio Failure Reasons
- `no_or_too_short_speech` - Fewer than 3 words in transcript
- `no_valid_chunks` - Transcript couldn't be chunked (too short after filtering)
- `toxic_or_abusive` - Profanity or toxicity detected with confidence
- `semantic_not_related` - Aggregated similarity < 0.55 AND too few skills mentioned

---

## Configuration Reference

All thresholds can be modified in the top of `app.py`:

```python
# Video Analysis Thresholds
FRAME_SAMPLE_FPS = 1                  # Frames per second to sample
MIN_PERSON_FRAMES_RATIO = 0.6         # Minimum presence ratio (default: 60%)
CENTER_ZONE_RATIO = 0.35              # Central region fraction (±35% from center)
MAX_AVG_CENTER_SHIFT = 0.12           # Max normalized movement (0-1 range)
MAX_AVG_BOX_SCALE_CHANGE = 0.20       # Max relative size change

# Semantic Thresholds
SIMILARITY_THRESHOLD = 0.55           # Overall resume-transcript similarity
SKILL_SIMILARITY_THRESHOLD = 0.55     # Per-skill threshold
TOP_K_CHUNKS = 3                      # Use top 3 chunks for aggregation
ROLE_SIM_THRESHOLD = 0.50             # Role category threshold
COMPANY_SIM_THRESHOLD = 0.55          # Company category threshold
PROJECT_SIM_THRESHOLD = 0.50          # Project category threshold

# Name Matching Thresholds
NAME_EXACT_REQUIRED = False           # If True, only exact name match
FUZZY_NAME_THRESHOLD = 0.60           # Fuzzy token match threshold (60% tokens)

# Toxicity Thresholds
TOXICITY_MODEL_PROB_THRESHOLD = 0.60  # Individual model confidence cutoff
ENSEMBLE_VOTE_THRESHOLD = 2           # Number of detectors to agree
MIN_TRANSCRIPT_WORDS = 3              # Minimum words for valid transcript

# Model Selection
EMBEDDER_MODEL = "all-mpnet-base-v2"  # Embedding model
TOXICITY_PRIMARY = "martin-ha/toxic-comment-model"  # Primary detector
TOXICITY_FALLBACK = None              # Fallback detector (optional)

# Profanity List
PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap"}
```

---

## Summary: What Passes and What Fails

### VIDEO PASSES if ALL conditions met:
✅ Exactly 1 person in >= 60% of frames
✅ Average movement <= 0.12 (normalized)
✅ Average scale change <= 0.20 (normalized)
✅ Person centered >= 50% of frames

### AUDIO PASSES if ALL conditions met:
✅ NOT toxic/abusive
✅ Name found in transcript (fuzzy match OR embedding match)
✅ AND at least ONE of:
  - Resume-transcript similarity >= 0.55, OR
  - Role/Company/Project match >= threshold, OR
  - Enough skills mentioned (>= 15% of resume skills)

### FINAL DECISION:
✅ **ACCEPTED** only if: video_valid AND audio_valid
❌ **REJECTED** if: video_valid = False OR audio_valid = False

---
