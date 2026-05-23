"""Chatbot engine for the Online Recruitment Fraud Detection app.

Primary engine: Google Gemini 2.5 Flash (via Google Gen AI SDK).
Fallback engine: Rule-based keyword/intent matching (when no API key is set).

Set the environment variable GEMINI_API_KEY or add it to a .env file in the
project root to enable the full Gemini-powered assistant.
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from string import punctuation

# Load .env file if present (optional, silent if not found)
ENV_PATH = Path(__file__).resolve().parent / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=True)
except ImportError:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


logger = logging.getLogger(__name__)
_gemini_client = None
_gemini_client_api_key: str | None = None
_gemini_sdk_name: str | None = None
_last_gemini_error: str | None = None

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

_PLACEHOLDER_API_KEYS = {
    "YOUR_GEMINI_API_KEY_HERE",
    "PASTE_YOUR_GEMINI_API_KEY_HERE",
    "REPLACE_WITH_YOUR_GEMINI_API_KEY",
}


def _read_api_key() -> str | None:
    """Read a real Gemini key from supported environment variable names."""
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(env_name, "").strip().strip("\"'")
        if value and value not in _PLACEHOLDER_API_KEYS:
            return value
    return None


GEMINI_API_KEY: str | None = _read_api_key()
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def _get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL_NAME", GEMINI_MODEL_NAME).strip() or "gemini-2.5-flash"

# System prompt for assistant context
SYSTEM_PROMPT = """You are Fraud Analysis Assistant, a highly knowledgeable and helpful assistant embedded inside the "Online Recruitment Fraud Detection" - a B.Tech Final Year Project web application built with Flask and Python.

## Your Role
You help users of this fraud detection web app by:
- Answering questions about job fraud, recruitment scams, and online safety
- Explaining AI/ML results produced by the app (confidence scores, Real vs Fraudulent labels)
- Guiding users through the app's features (Dashboard, Upload & Check, Assistant)
- Scanning job descriptions for red flags when users paste them
- Answering general questions clearly and helpfully

## About This Application
- **Purpose**: Detects fraudulent job postings using AI/ML
- **ML Pipeline**: BERT sentence embeddings (DistilBERT) -> 2D CNN classifier
- **Dataset**: Kaggle Fake Job Postings dataset (~18,000 records) + India-specific real and synthetic fraud jobs
- **Labels**: "Real Job" or "Fraudulent Job" with a confidence percentage
- **High-Risk**: Fraudulent predictions with 85% or higher confidence are flagged as High-Risk
- **Default demo file**: testData.csv (India-focused, offline-safe, uses cached BERT features)
- **Tech stack**: Flask (Python), NumPy CNN inference, SentenceTransformers, h5py for model weights

## App Navigation (sidebar)
1. **Dashboard** -> Shows dataset statistics (total jobs, real jobs, suspicious jobs)
2. **Upload & Check** -> Upload a CSV file (must have a `description` column) or type/paste a job manually; runs AI fraud detection
3. **Assistant** -> This chat interface

## Common Fraud Red Flags You Know About
- Unrealistic salary promises ("earn INR 50,000/week from home")
- No experience required for high-paying roles
- Upfront fees (registration, training, equipment)
- Vague, generic job descriptions with no company details
- Requests for personal/bank information early
- Poor grammar, spelling errors, generic email domains
- Work from home with guaranteed income
- Urgency tactics ("limited seats", "apply today only")
- Wire transfer or Western Union payment mentions
- Multi-level marketing or pyramid schemes

## Fraud Meter Output (MANDATORY for job analysis)
Whenever you analyze a job description (pasted text, uploaded file, or any job-related question where you assess risk), you MUST include EXACTLY ONE of these lines at the very end of your response on its own line:
- [FRAUD_METER:15:Likely Real] - for jobs that appear legitimate with no red flags
- [FRAUD_METER:35:Low Risk] - for jobs with minor concerns but mostly safe
- [FRAUD_METER:55:Moderate Risk] - for jobs with some suspicious patterns
- [FRAUD_METER:75:High Risk] - for jobs with multiple red flags
- [FRAUD_METER:90:Very High Risk] - for jobs that are almost certainly fraudulent
You may adjust the number (0-100) to reflect your confidence. The label should match the risk level.
Do NOT include this line for general questions, greetings, or non-job-analysis conversations.

## Personality
- Be friendly, professional, and concise
- Do not use emoji characters, decorative symbols, or mojibake text in replies
- Use plain text labels like "Alert:", "Tip:", and "Note:" instead of emoji
- Format responses with bullet points and bold text for clarity
- If asked something outside your knowledge, be honest and suggest checking official sources
- If a user pastes a job description (long text), proactively analyse it for red flags
- Always encourage users to use the Upload & Check page for full AI model analysis

## Important Notes
- You are NOT the ML fraud detection model itself - you are the conversational assistant
- The actual fraud prediction is done by the CNN model on the Upload & Check page
- You can give rule-based quick-scan assessments, but always clarify these are heuristic
- You can discuss India-specific job scam patterns, common in the project's dataset
"""

MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u00a2": "-",
    "\u00e2\u2020\u2019": "->",
    "\u00e2\u2030\u00a5": ">=",
    "\u00e2\u201a\u00b9": "INR",
}
MOJIBAKE_EMOJI_RE = re.compile(r"\s*" + "\u00f0\u0178" + r"[^\s]*")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def _sanitize_reply(text: str) -> str:
    """Remove encoding artifacts and emoji glyphs before rendering chat text."""
    cleaned = text or ""
    for bad, replacement in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, replacement)
    cleaned = MOJIBAKE_EMOJI_RE.sub("", cleaned)
    cleaned = EMOJI_RE.sub("", cleaned)
    cleaned = cleaned.replace("\ufffd", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    return cleaned.strip()


def _get_gemini_client():
    """Lazy-load Gemini and support both current and legacy SDK packages."""
    global _gemini_client, _gemini_client_api_key, _gemini_sdk_name, _last_gemini_error

    api_key = _read_api_key()
    if not api_key:
        _last_gemini_error = "No GEMINI_API_KEY or GOOGLE_API_KEY found in environment/.env."
        return None

    if _gemini_client is not None and _gemini_client_api_key == api_key:
        return _gemini_client

    _gemini_client = None
    _gemini_client_api_key = api_key
    _gemini_sdk_name = None
    _last_gemini_error = None

    try:
        from google import genai  # current package: google-genai

        _gemini_client = genai.Client(api_key=api_key)
        _gemini_sdk_name = "google-genai"
        logger.info("Gemini client loaded with google-genai for model: %s", _get_model_name())
        return _gemini_client
    except Exception as genai_exc:
        try:
            import google.generativeai as legacy_genai  # legacy package: google-generativeai

            legacy_genai.configure(api_key=api_key)
            _gemini_client = legacy_genai
            _gemini_sdk_name = "google-generativeai"
            logger.info("Gemini client loaded with google-generativeai for model: %s", _get_model_name())
            return _gemini_client
        except Exception as legacy_exc:
            _last_gemini_error = (
                "Could not load a Gemini SDK. "
                f"google-genai error: {genai_exc}; google-generativeai error: {legacy_exc}"
            )
            logger.warning(_last_gemini_error)
            return None


def _build_gemini_prompt(message: str, history: list[dict]) -> str:
    turns = []
    for entry in history[-8:]:
        role = "User" if entry.get("role") == "user" else "Assistant"
        content = str(entry.get("content", "")).strip()
        if content:
            turns.append(f"{role}: {content}")

    history_text = "\n".join(turns)
    
    # If the user message is long, remind Gemini to act as a scanner and output the meter.
    suffix = ""
    if len(message.split()) >= 20:
        suffix = "\n[SYSTEM NOTE: The user has pasted a job description. Analyze it for fraud red flags and YOU MUST end your response exactly with the tag [FRAUD_METER:score:label] (e.g. [FRAUD_METER:85:High Risk]).]\n"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Conversation history:\n"
        f"{history_text or 'No previous conversation.'}\n\n"
        f"User: {message}{suffix}\n"
        "Assistant:"
    )


def _extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    return ""


def _gemini_response(message: str, history: list[dict]) -> str | None:
    """Send message + conversation history to Gemini and return the reply text."""
    global _last_gemini_error

    client = _get_gemini_client()
    if client is None:
        return None

    prompt = _build_gemini_prompt(message, history)
    requested_model = _get_model_name()
    model_candidates = []
    for model_name in (requested_model, "gemini-2.0-flash", "gemini-1.5-flash"):
        if model_name and model_name not in model_candidates:
            model_candidates.append(model_name)

    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            if _gemini_sdk_name == "google-genai":
                response = client.models.generate_content(model=model_name, contents=prompt)
            else:
                model = client.GenerativeModel(model_name)
                response = model.generate_content(prompt)

            text = _extract_response_text(response)
            if text.strip():
                _last_gemini_error = None
                return _sanitize_reply(text)

            _last_gemini_error = f"Gemini returned an empty response for model {model_name}."
            logger.warning(_last_gemini_error)
        except Exception as exc:
            last_error = exc
            _last_gemini_error = f"Gemini API call failed for model {model_name}: {exc}"
            logger.warning(_last_gemini_error)

    if last_error is not None:
        logger.warning("All Gemini model attempts failed: %s", last_error)
    return None


# ---------------------------------------------------------------------------
# Rule-based fallback engine (used when Gemini API key is not set)
# ---------------------------------------------------------------------------

FRAUD_PATTERNS: list[dict] = [
    {"pattern": r"\bno experience (required|needed)\b", "label": "No experience required", "severity": "high"},
    {"pattern": r"\bwork from home\b", "label": "Work from home promise", "severity": "medium"},
    {"pattern": r"\beasy money\b", "label": "Easy money claim", "severity": "high"},
    {"pattern": r"\bguaranteed (income|salary|pay|earnings)\b", "label": "Guaranteed income", "severity": "high"},
    {"pattern": r"\bwire transfer\b", "label": "Wire transfer mention", "severity": "high"},
    {"pattern": r"\bwestern union\b", "label": "Western Union mention", "severity": "high"},
    {"pattern": r"\bunclaimed (funds|money|prize)\b", "label": "Unclaimed funds", "severity": "high"},
    {"pattern": r"\bupfront (fee|payment|deposit)\b", "label": "Upfront payment request", "severity": "high"},
    {"pattern": r"\bno interview\b", "label": "No interview required", "severity": "medium"},
    {"pattern": r"\bwork at home\b", "label": "Work at home promise", "severity": "medium"},
    {"pattern": r"\bearn \$\d+[,\d]* (per|a) (day|week|hour)\b", "label": "Specific earning claim", "severity": "high"},
    {"pattern": r"\b(unlimited|massive) earning(s)?\b", "label": "Unlimited earnings claim", "severity": "high"},
    {"pattern": r"\bpart.?time.*full.?time income\b", "label": "Part-time with full-time income", "severity": "high"},
    {"pattern": r"\bsend (us|me) your (bank|account) details\b", "label": "Requests bank details", "severity": "high"},
    {"pattern": r"\bjoin our team (immediately|today|now)\b", "label": "Immediate hiring urgency", "severity": "medium"},
    {"pattern": r"\bno (qualifications|degree|education) (needed|required)\b", "label": "No qualifications needed", "severity": "medium"},
    {"pattern": r"\b(government|lottery|prize) winner\b", "label": "Lottery/prize claim", "severity": "high"},
    {"pattern": r"\btraining (fee|cost|charge)\b", "label": "Charges for training", "severity": "high"},
    {"pattern": r"\bregistration fee\b", "label": "Registration fee required", "severity": "high"},
    {"pattern": r"\bwork online.{0,30}earn\b", "label": "Online work earning claim", "severity": "medium"},
    {"pattern": r"\bget rich\b", "label": "Get rich claim", "severity": "high"},
    {"pattern": r"\bpyramid\b", "label": "Pyramid scheme mention", "severity": "high"},
    {"pattern": r"\bmulti.?level marketing\b", "label": "MLM mention", "severity": "medium"},
    {"pattern": r"\bclicking (ads|links)\b", "label": "Paid-to-click scheme", "severity": "high"},
    {"pattern": r"\bdata entry.{0,20}home\b", "label": "Home data entry scheme", "severity": "medium"},
]

INTENTS: list[dict] = [
    {
        "name": "greet",
        "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy", "hiya"],
        "responses": [
            "Hello. I can help you understand fraud signs, explain results, or scan a job description. What would you like to know?",
            "Hi there! I'm here to help you spot fraudulent job postings. Ask me anything!",
        ],
    },
    {
        "name": "fraud_signs",
        "keywords": [
            "fraud signs", "fake job", "suspicious", "red flags", "warning signs", "how to spot",
            "identify fraud", "scam signs", "fraudulent", "what makes", "signs of",
        ],
        "responses": [
            (
                "**Common Red Flags in Fraudulent Job Postings:**\n\n"
                "- **Too good to be true** - unrealistic salaries like 'earn INR 50,000/week from home'\n"
                "- **No experience required** - real jobs have minimum requirements\n"
                "- **Upfront fees** - legitimate employers never ask you to pay to get hired\n"
                "- **Vague job descriptions** - real jobs describe roles clearly\n"
                "- **Pressure to act fast** - 'limited seats', 'apply today or miss out'\n"
                "- **Requests personal/bank info early** - a major red flag\n"
                "- **Poor grammar & spelling** - often a sign of overseas scammers\n"
                "- **Work from home with guaranteed income** - almost always a scam\n\n"
                "Tip: Use the Upload & Check page to run our AI model on any job CSV."
            ),
        ],
    },
    {
        "name": "confidence_explain",
        "keywords": ["confidence", "percentage", "accuracy", "score", "probability", "confidence score", "percent"],
        "responses": [
            (
                "**What does the Confidence % mean?**\n\n"
                "- **90-100%** -> Very high certainty\n"
                "- **75-89%** -> High confidence - likely correct\n"
                "- **60-74%** -> Moderate - review manually\n"
                "- **Below 60%** -> Low certainty - borderline case\n\n"
                "Jobs flagged as **Fraudulent** above **85% confidence** are marked *High-Risk*."
            ),
        ],
    },
    {
        "name": "how_to_use",
        "keywords": ["how to use", "guide", "steps", "instructions", "get started", "how does this work", "upload", "csv"],
        "responses": [
            (
                "**How to Use This App:**\n\n"
                "1. **Login** with your credentials\n"
                "2. **Upload & Check** - upload a CSV with a `description` column, or type a job manually\n"
                "3. **View Results** - jobs classified as Real or Fraudulent\n"
                "4. **Analyse** - filter, search, export CSV or print the report\n\n"
                "You can also paste any job description here for a quick scan."
            ),
        ],
    },
    {
        "name": "thanks",
        "keywords": ["thank", "thanks", "thank you", "great", "awesome", "helpful", "nice", "perfect"],
        "responses": [
            "You're welcome! Feel free to ask anything else.",
            "Happy to help. Let me know if you have more questions.",
        ],
    },
    {
        "name": "goodbye",
        "keywords": ["bye", "goodbye", "see you", "exit", "later", "take care"],
        "responses": [
            "Goodbye! Stay safe and always verify job postings before applying.",
            "Take care. Remember: if a job sounds too good to be true, it probably is.",
        ],
    },
]

FALLBACK_RESPONSES = [
    (
        "I'm not sure about that. Here's what I can help with:\n\n"
        "- **'What are fraud signs?'** - red flags in job postings\n"
        "- **'How do I use this app?'**\n"
        "- **'What does confidence mean?'**\n"
        "- **Paste a job description** - I'll scan it for red flags.\n\n"
        "Tip: Gemini is enabled for detailed answers."
    ),
]


def _normalize(text: str) -> str:
    table = str.maketrans("", "", punctuation)
    return text.lower().translate(table)


def _detect_intent(message: str) -> str | None:
    normalized = _normalize(message)
    for intent in INTENTS:
        for keyword in intent["keywords"]:
            if keyword in normalized:
                return intent["name"]
    return None


def _is_long_text(message: str) -> bool:
    return len(message.split()) >= 20


def _quick_scan(text: str) -> str:
    """Rule-based fraud pattern scan for pasted job descriptions."""
    text_lower = text.lower()
    found_flags: list[dict] = []

    for pattern_info in FRAUD_PATTERNS:
        if re.search(pattern_info["pattern"], text_lower):
            found_flags.append(pattern_info)

    if not found_flags:
        return (
            "**Quick Scan: No obvious red flags detected.**\n\n"
            "This description didn't trigger any rule-based fraud patterns. "
            "For a full AI analysis, upload it via **Upload & Check**.\n\n"
            "_Note: This quick scan checks common patterns only - the CNN model is more accurate._\n\n"
            "[FRAUD_METER:15:Likely Real]"
        )

    high = [f for f in found_flags if f["severity"] == "high"]
    medium = [f for f in found_flags if f["severity"] == "medium"]
    lines = ["**Quick Scan: Suspicious Patterns Detected!**\n"]

    if high:
        lines.append("**High-Severity Flags:**")
        for flag in high:
            lines.append(f"- {flag['label']}")

    if medium:
        lines.append("\n**Medium-Severity Flags:**")
        for flag in medium:
            lines.append(f"- {flag['label']}")

    # Calculate fraud score based on severity
    score = min(95, 30 + len(high) * 15 + len(medium) * 8)
    if score >= 75:
        risk_label = "High Risk"
        risk_text = "HIGH RISK"
    elif score >= 50:
        risk_label = "Moderate Risk"
        risk_text = "MODERATE RISK"
    else:
        risk_label = "Low Risk"
        risk_text = "LOW RISK"

    lines.append(
        f"\n**Overall: {risk_text}** - {len(found_flags)} pattern(s) matched.\n\n"
        "_Upload this via **Upload & Check** for a full AI model analysis with confidence scores._\n\n"
        f"[FRAUD_METER:{score}:{risk_label}]"
    )
    return "\n".join(lines)


def _rule_based_response(message: str) -> str:
    """Fallback rule-based engine."""
    if _is_long_text(message):
        return _quick_scan(message)

    intent = _detect_intent(message)
    if intent is not None:
        for definition in INTENTS:
            if definition["name"] == intent:
                return random.choice(definition["responses"])  # noqa: S311

    return random.choice(FALLBACK_RESPONSES)  # noqa: S311


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_gemini_enabled() -> bool:
    """Return True if a non-placeholder Gemini API key is configured."""
    return bool(_read_api_key())


def get_gemini_status() -> dict[str, object]:
    """Return chatbot configuration/runtime status without exposing the API key."""
    api_key = _read_api_key()
    configured = bool(api_key)
    return {
        "gemini_active": configured and _last_gemini_error is None,
        "gemini_configured": configured,
        "gemini_sdk": _gemini_sdk_name or "not loaded yet",
        "gemini_model": _get_model_name(),
        "gemini_error": _last_gemini_error or "",
    }



def get_response(message: str, history: list[dict] | None = None) -> str:
    """Return only the bot reply string for callers that do not need metadata."""
    reply, _engine = get_response_with_engine(message, history)
    return reply


def analyze_uploaded_file(file_bytes: bytes, filename: str, mime_type: str) -> tuple[str, str]:
    """Analyze an uploaded file (PDF or image) for job fraud indicators via Gemini.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename for context.
        mime_type:  MIME type of the file (e.g. 'application/pdf', 'image/jpeg').

    Returns:
        (reply_text, engine) — engine is 'gemini' or 'fallback'.
    """
    global _last_gemini_error

    client = _get_gemini_client()
    if client is None:
        return (
            "**File upload analysis requires Gemini AI.**\n\n"
            "The Gemini API key is not configured, so I cannot scan uploaded files. "
            "Please set your GEMINI_API_KEY in the .env file and restart the app.\n\n"
            "In the meantime, you can paste the job description text directly into "
            "the chat and I will run a rule-based quick scan.",
            "fallback",
        )

    import base64

    file_analysis_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "## Current Task: Uploaded File Analysis\n\n"
        f"The user has uploaded a file named '{filename}' (type: {mime_type}) "
        "which is a job description, job notification, or recruitment-related document.\n\n"
        "Please carefully analyze the contents of this file and:\n"
        "1. Extract and summarize the key job details (title, company, location, salary, requirements)\n"
        "2. Identify ALL potential fraud red flags and suspicious patterns\n"
        "3. Rate the overall risk level as LOW RISK, MODERATE RISK, or HIGH RISK\n"
        "4. Provide specific recommendations for the user\n\n"
        "Format your response clearly with bold headings and bullet points.\n"
        "If the file does not appear to be job/recruitment related, let the user know "
        "and suggest they upload a relevant document instead.\n\n"
        "CRITICAL: Because you are analyzing a file, you MUST end your response exactly with the tag [FRAUD_METER:score:label] on a new line. Example: [FRAUD_METER:85:High Risk]."
    )

    requested_model = _get_model_name()
    model_candidates = []
    for model_name in (requested_model, "gemini-2.0-flash", "gemini-1.5-flash"):
        if model_name and model_name not in model_candidates:
            model_candidates.append(model_name)

    last_error: Exception | None = None

    for model_name in model_candidates:
        try:
            if _gemini_sdk_name == "google-genai":
                from google.genai import types as genai_types

                file_part = genai_types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type,
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=[file_analysis_prompt, file_part],
                )
            else:
                # Legacy SDK
                b64_data = base64.b64encode(file_bytes).decode("utf-8")
                file_part = {"mime_type": mime_type, "data": b64_data}
                model = client.GenerativeModel(model_name)
                response = model.generate_content([file_analysis_prompt, file_part])

            text = _extract_response_text(response)
            if text.strip():
                _last_gemini_error = None
                return _sanitize_reply(text), "gemini"

            _last_gemini_error = f"Gemini returned an empty response for model {model_name}."
            logger.warning(_last_gemini_error)
        except Exception as exc:
            last_error = exc
            _last_gemini_error = f"Gemini file analysis failed for model {model_name}: {exc}"
            logger.warning(_last_gemini_error)

    if last_error is not None:
        logger.warning("All Gemini model attempts failed for file analysis: %s", last_error)

    return (
        "**File analysis could not be completed.**\n\n"
        "The AI model was unable to process this file. This may be due to the file "
        "format, size, or a temporary API issue.\n\n"
        "Please try again, or paste the job description text directly into the chat.",
        "fallback",
    )


def get_response_with_engine(message: str, history: list[dict] | None = None) -> tuple[str, str]:
    """
    Main entry point. Returns the bot reply and the engine that produced it.

    Args:
        message:  The latest user message.
        history:  List of previous turns, each a dict with keys
                  'role' ('user' or 'model') and 'content' (str).
                  Used for multi-turn context when Gemini is active.
    """
    message = message.strip()
    if not message:
        return "Please type a message or question!", "fallback"

    if history is None:
        history = []

    # --- Try Gemini first ---
    gemini_reply = _gemini_response(message, history)
    if gemini_reply:
        return _sanitize_reply(gemini_reply), "gemini"

    # --- Fall back to rule-based ---
    return _sanitize_reply(_rule_based_response(message)), "fallback"





