"""
Local vision labeling via Ollama (moondream) with JSON output.
Falls back gracefully if Ollama/model unavailable.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
import re
import json5
import os
from .ocr import extract_text_from_file

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

# Supabase Edge Function proxy: all OpenAI calls go through here so users
# don't have to bring their own key. Auth uses the logged-in user's
# Supabase access token; the OpenAI API key lives in Supabase secrets.
from .supabase_client import SUPABASE_URL
OPENAI_PROXY_URL = f"{SUPABASE_URL}/functions/v1/openai-proxy"

# Import settings for dynamic model configuration
from .settings import settings


def _get_auth_token() -> Optional[str]:
    """Get the current user's Supabase access token for proxy calls."""
    try:
        from .supabase_client import supabase_auth
        if supabase_auth.is_authenticated:
            return supabase_auth._access_token
        return None
    except Exception as e:
        logger.error(f"Failed to get auth token: {e}")
        return None


def _call_openai_proxy(endpoint: str, messages: List[Dict], max_tokens: int = 500, temperature: float = 0.2) -> Optional[Dict[str, Any]]:
    """Call OpenAI via the Supabase Edge Function proxy.

    Args:
        endpoint: 'chat' (text) or 'vision' (image).
        messages: OpenAI-format messages array.
        max_tokens: Response token cap.
        temperature: Sampling temperature.

    Returns:
        OpenAI response JSON (choices/message/content shape) or None on failure.
    """
    try:
        auth_token = _get_auth_token()
        if not auth_token:
            logger.warning("No Supabase auth token; cannot call OpenAI proxy. Sign in first.")
            return None

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "endpoint": endpoint,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        logger.info(f"Calling OpenAI proxy: endpoint={endpoint}")
        response = requests.post(OPENAI_PROXY_URL, json=payload, headers=headers, timeout=120)

        if response.status_code == 401:
            logger.error("OpenAI proxy: 401 (auth invalid — session may have expired; please sign in again)")
            return None
        if response.status_code == 403:
            logger.error("OpenAI proxy: 403 (no active subscription)")
            return None
        if response.status_code != 200:
            logger.error(f"OpenAI proxy error: {response.status_code} - {response.text[:300]}")
            return None

        return response.json()
    except requests.exceptions.Timeout:
        logger.error("OpenAI proxy call timed out")
        return None
    except Exception as e:
        logger.error(f"OpenAI proxy call failed: {e}")
        return None


def transcribe_audio_proxy(audio_path: str) -> Optional[str]:
    """Transcribe an audio file using Whisper through the Supabase proxy."""
    try:
        auth_token = _get_auth_token()
        if not auth_token:
            logger.warning("No Supabase auth token; cannot call Whisper proxy.")
            return None

        with open(audio_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "endpoint": "whisper",
            "audio_base64": audio_b64,
            "audio_filename": os.path.basename(audio_path),
        }

        response = requests.post(OPENAI_PROXY_URL, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            logger.error(f"Whisper proxy error: {response.status_code} - {response.text[:300]}")
            return None

        return response.json().get("text", "") or None
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return None


def get_local_model() -> str:
    """Get the current local model from settings (Qwen 2.5-VL handles both text and vision)."""
    return settings.local_model or os.environ.get("LOCAL_MODEL", "qwen2.5vl:3b")


# Aliases for backward compatibility - both point to the same unified model
def get_vision_model() -> str:
    """Get the vision model (same as local model since Qwen 2.5-VL handles both)."""
    return get_local_model()


def get_text_model() -> str:
    """Get the text model (same as local model since Qwen 2.5-VL handles both)."""
    return get_local_model()


def get_openai_vision_model() -> str:
    """Get the current OpenAI vision model from settings."""
    return settings.openai_vision_model or os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")


# Legacy constants for backward compatibility (use getters in new code)
DEFAULT_MODEL = get_vision_model()
TEXT_MODEL = get_text_model()
OPENAI_VISION_MODEL = get_openai_vision_model()

USER_INSTRUCTIONS_TEMPLATE = """

=== ADDITIONAL USER FOCUS ===
The user has provided specific guidance for what they want emphasized in the analysis and tags.
Their instructions: "{user_instructions}"

IMPORTANT RULES FOR HANDLING USER INSTRUCTIONS:
- You MUST still generate all standard tags (type, platform, subjects, objects, layout, style, colors, mood, branding, etc.)
- In ADDITION to standard tags, pay special attention to the user's focus areas mentioned above
- If the user mentions specific things to look for (e.g., "client names", "project codes", "invoice numbers", "brand names"), actively scan for these and include them as tags if they are visible in the content
- If the user's focus areas are NOT visible or NOT applicable to this content, simply ignore them and proceed with standard analysis
- The user's instructions should ENHANCE your tagging, not REPLACE your standard comprehensive analysis
- Aim for 25-45 tags total, blending thorough standard analysis with the user's specific focus areas
- If you find elements matching the user's focus, prioritize including those as tags
"""


def build_analysis_prompt(base_prompt: str, user_instructions: str = None) -> str:
    """Build the full analysis prompt, optionally including user instructions."""
    if user_instructions and user_instructions.strip():
        clean = user_instructions.strip()[:500]
        clean = clean.replace('"', "'")   # prevent prompt injection via quotes
        clean = clean.replace('\n', ' ')  # flatten to single line
        return base_prompt + USER_INSTRUCTIONS_TEMPLATE.format(user_instructions=clean)
    return base_prompt

SYSTEM_PROMPT = (
    "You are an on-device vision classifier. Output ONE JSON object ONLY (no markdown, no prose).\n"
    "Schema (strict):\n"
    "{\n"
    "  \"type\": <single high-level concept of what the image IS. Examples: youtube thumbnail, invoice, receipt, screenshot, meme, logo, poster, banner, flyer, slide, whiteboard photo, chart/graph, map, photograph, portrait, product image, app UI screenshot, code screenshot, document page, illustration, other>,\n"
    "  \"caption\": <<= 400 chars, 2-3 sentences describing the scene, visible text/labels (briefly), key subjects, layout, and purpose>,\n"
    "  \"tags\": [20..40 short lowercase tags capturing: concept (repeat type), platform/context (youtube, tiktok, twitter, web), subjects/roles (e.g., developer, presenter), key objects (e.g., laptop, microphone), UI elements (button, chart, code block, navbar), layout/composition (big headline, side-by-side, grid, centered subject), text_presence(none/low/medium/high), style/medium (photo, vector, 3d render, screenshot), mood/tone (playful, serious), color palette (e.g., blue/orange), branding/logos (brand names if visible), aspect ratio (16:9, 1:1, 9:16), background/env (studio, office, outdoors), actions/verbs (speaking, presenting), audience/intent (tutorial, ad, thumbnail, meme)],\n"
    "  \"confidence\": <float 0..1>\n"
    "}\n"
    "Rules:\n"
    "- \"type\" MUST be a conceptual category. Never copy overlay text.\n"
    "- Prefer the most specific applicable concept (e.g., youtube thumbnail over generic poster).\n"
    "- \"tags\" must be information-dense and include the concept itself, platform, salient elements, layout, style, mood, colors, branding cues, and aspect.\n"
    "- No trailing commas, no duplicates, JSON only."
)


# Detailed vision model/prompt for rich descriptions and explicit type
def get_detailed_vision_model() -> str:
    """Get the detailed vision model - uses same unified model for consistency."""
    return get_local_model()

DETAILED_VISION_MODEL = get_detailed_vision_model()

DETAILED_SYSTEM_PROMPT = (
    "You are a meticulous visual analyst. Output ONE JSON object ONLY (no markdown, no prose).\n"
    "Describe the attached image in a vivid, information-dense paragraph and classify what it is.\n"
    "Schema (strict):\n"
    "{\n"
    "  \"type\": <ONE of: Photograph | Screenshot | Scanned Document | Digital Document | Diagram/Chart | UI Mockup | Meme | Thumbnail | Poster | Logo | Other>,\n"
    "  \"description\": <a richly detailed paragraph covering: setting/location, subjects and obvious attributes, actions, relationships, notable objects, layout/composition (foreground/background, camera angle, lighting), color palette, mood/style (photo/illustration/3D, realistic/cartoon), and any logos/branding>,\n"
    "  \"detected_text\": <exact transcription of any legible text or \"none\">,\n"
    "  \"purpose\": <explicit, concrete intended use inferred strictly from what is visible; examples: \"YouTube thumbnail\", \"mobile app UI screenshot\", \"invoice\", \"event poster\", \"presentation slide\", \"meme\", \"product photo\">,\n"
    "  \"suggested_filename\": <5-8 words, lowercase kebab-case; summarize visible content; avoid generic words like 'image'/'photo'; include key objects/roles/topic; only include brand names, dates, or people if clearly visible>,\n"
    "  \"tags\": [10..25 short lowercase tags capturing concept, platform/context, salient elements/objects, UI elements if any, layout/composition, style/medium, mood/tone, colors, branding, aspect],\n"
    "  \"confidence\": <float 0..1>\n"
    "}\n"
    "Rules:\n"
    "- Be specific and concrete; avoid generic phrases.\n"
    "- Do not invent facts; if uncertain, write \"uncertain\".\n"
    "- For 'purpose', choose the single most likely end-use (e.g., 'YouTube thumbnail', 'invoice'); do not describe the scene again.\n"
    "- For 'suggested_filename', use only visible content; 5-8 words; kebab-case; no spaces; no private data; keep it descriptive and concise.\n"
    "- Preserve punctuation/casing for transcribed text.\n"
    "- JSON only. No markdown. No extra keys. No trailing commas."
)


def _pil_image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Return base64 string (no data URI prefix) for Ollama images field."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")


def _file_to_b64(file_path: Path) -> Optional[str]:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            # Render first page to image
            pages = convert_from_path(str(file_path), first_page=1, last_page=1)
            if not pages:
                return None
            img = pages[0].convert("RGB")
            # downscale large pages for speed
            img.thumbnail((1024, 1024))
            return _pil_image_to_b64(img)
        else:
            # load via PIL to optionally downscale
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                img.thumbnail((1024, 1024))
                return _pil_image_to_b64(img)
    except Exception as e:
        logger.error(f"Failed to build data URI for {file_path}: {e}")
        return None


def _ollama_is_alive() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False


def _normalize_model_name(name: str) -> tuple[str, str]:
    """Split model name into (base, tag). 'moondream:latest' -> ('moondream','latest')."""
    try:
        parts = (name or "").strip().lower().split(":", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""
    except Exception:
        return name.lower(), ""


def _names_match(installed: str, requested: str) -> bool:
    """Return True if installed model satisfies requested model.
    - Exact match OR
    - Installed startswith requested + ':' OR
    - Base names match when requested has no explicit tag
    """
    inst = (installed or "").strip().lower()
    req = (requested or "").strip().lower()
    if not inst or not req:
        return False
    if inst == req:
        return True
    if inst.startswith(req + ":"):
        return True
    inst_base, _ = _normalize_model_name(inst)
    req_base, req_tag = _normalize_model_name(req)
    if not req_tag and inst_base == req_base:
        return True
    return False


def _model_is_available(model: str) -> bool:
    """Return True if the requested model (with or without tag) exists locally in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if not r.ok:
            return False
        data = r.json() or {}
        models = data.get("models") or []
        names = [m.get("name", "") for m in models if isinstance(m, dict)]
        for n in names:
            if _names_match(n, model):
                return True
        return False
    except Exception:
        return False


def _ensure_model(model: str = None) -> None:
    """No-op placeholder retained for compatibility. We no longer auto-pull models
    to honor the requirement to use only locally available models.
    """
    if model is None:
        model = get_vision_model()
    return None


def analyze_image(image_path: Path, model: str = None, user_instructions: str = None) -> Optional[Dict[str, Any]]:
    """Return label/tags/caption/confidence using configured AI provider.

    Uses OpenAI by default (recommended). Falls back to local Ollama if configured.
    Returns None if no AI provider is available.
    """
    try:
        # Check which AI provider to use
        provider = settings.ai_provider

        # OpenAI is the primary/default provider
        if provider == 'openai':
            image_b64 = _file_to_b64(image_path)
            if image_b64:
                result = gpt_vision_fallback(image_b64, image_path.name, user_instructions=user_instructions)
                if result:
                    logger.info(f"OpenAI vision analysis successful for {image_path.name}")
                    return result
            logger.warning("OpenAI vision failed, no fallback configured")
            return None
        
        # Local (Ollama) provider
        if provider == 'local':
            if model is None:
                model = get_vision_model()
            if not _ollama_is_alive():
                logger.warning("Ollama is not running at localhost:11434")
                return None
            if not _model_is_available(model):
                logger.info(f"Model '{model}' not available locally. Skipping analyze_image.")
                return None
            
            # Build image b64 and gather context (dimensions, aspect)
            width = height = None
            try:
                with Image.open(image_path) as dim_img:
                    width, height = dim_img.size
            except Exception:
                width = height = None
            image_b64 = _file_to_b64(image_path)
            if not image_b64:
                return None
            aspect = round((width / height), 3) if width and height and height != 0 else None
            ctx = []
            if image_path.name:
                ctx.append(f"filename: {image_path.name}")
            if width and height:
                ctx.append(f"dimensions: {width}x{height}, aspect={aspect}")
            ctx_str = "\n".join(ctx)
            prompt = SYSTEM_PROMPT + "\n" + ctx_str + "\nReturn STRICT JSON only (no markdown)."
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "temperature": 0.2,
                "options": {"num_predict": 512}
            }
            # Retry up to 2 times on transient failures
            last_err: Optional[str] = None
            r = None
            for attempt in range(2):
                try:
                    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
                    if r.ok:
                        break
                    last_err = r.text
                except Exception as e:
                    last_err = str(e)
            if not r or not r.ok:
                logger.error("Ollama response not OK: %s", last_err)
                return None
            out = r.json()
            content = out.get("response") or ""
            logger.info(f"Ollama raw content (trunc): {content[:200]}")
            result = _parse_json_relaxed(content)
            if result is None:
                salvage = _salvage_from_content(content)
                if salvage:
                    logger.info("Used salvage parser for vision result")
                    return salvage
                logger.error("Failed to parse JSON from Ollama content after sanitation")
                return None

            # Validate fields
            label = str(result.get("type", "other")).strip().lower()
            tags = result.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).lower()[:64] for t in tags][:25]
            caption = str(result.get("caption", "")).strip()[:400]
            confidence = result.get("confidence")
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

            parsed = {
                "label": label,
                "tags": tags,
                "caption": caption,
                "vision_confidence": confidence,
            }
            logger.info(f"Parsed vision result: {parsed}")
            return parsed
        
        # Provider is 'none' - no AI analysis
        return None
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        return None


def _gpt_text_analysis(text: str, filename: Optional[str] = None, user_instructions: str = None) -> Optional[Dict[str, Any]]:
    """Analyze text content using OpenAI through the Supabase Edge Function proxy."""
    try:
        name_part = f"Filename: {filename}\n" if filename else ""
        snippet = (text or "").strip()
        if len(snippet) > 5000:
            snippet = snippet[:5000]

        user_prompt = (
            "Classify the following file content. Return STRICT JSON only.\n"
            + name_part
            + "Content snippet:\n" + snippet
        )

        system = build_analysis_prompt(SYSTEM_PROMPT, user_instructions)
        resp_data = _call_openai_proxy(
            "chat",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        if not resp_data:
            return None
        choices = resp_data.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "") or ""
        
        try:
            data = json.loads(content)
        except Exception:
            s = content.find("{")
            e = content.rfind("}")
            if s != -1 and e != -1 and e > s:
                data = json.loads(content[s:e+1])
            else:
                return None
        
        label = str(data.get("type", "other")).strip().lower()
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower()[:64] for t in tags][:25]
        caption = str(data.get("caption", "")).strip()[:400]
        try:
            conf = float(data.get("confidence", 0))
        except Exception:
            conf = 0.0
        
        return {
            "label": label,
            "tags": tags,
            "caption": caption,
            "vision_confidence": conf,
        }
    except Exception as e:
        logger.error(f"GPT text analysis error: {e}")
        return None


def analyze_text(text: str, filename: Optional[str] = None, model: str = None, user_instructions: str = None) -> Optional[Dict[str, Any]]:
    """Classify non-image files using configured AI provider.

    Uses OpenAI by default (recommended). Falls back to local Ollama if configured.
    Returns: dict with label, tags, caption, vision_confidence (score), or None on failure.
    """
    try:
        provider = settings.ai_provider

        # OpenAI is the primary/default provider
        if provider == 'openai':
            result = _gpt_text_analysis(text, filename, user_instructions=user_instructions)
            if result:
                logger.info(f"OpenAI text analysis successful for {filename or 'unknown'}")
                return result
            logger.warning("OpenAI text analysis failed")
            return None
        
        # Local (Ollama) provider
        if provider == 'local':
            if model is None:
                model = get_text_model()
            if not _ollama_is_alive():
                logger.warning("Ollama is not running at localhost:11434")
                return None
            if not _model_is_available(model):
                logger.info(f"Model '{model}' not available locally. Skipping text analyzer.")
                return None

            name_part = f"Filename: {filename}\n" if filename else ""
            snippet = (text or "").strip()
            if len(snippet) > 5000:
                snippet = snippet[:5000]
            prompt = (
                SYSTEM_PROMPT
                + "\nClassify the following file content. Return STRICT JSON only.\n"
                + name_part
                + "Content snippet:\n" + snippet
            )
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
                "options": {"num_predict": 512}
            }

            last_err: Optional[str] = None
            r = None
            for _ in range(3):
                try:
                    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
                    if r.ok:
                        break
                    last_err = r.text
                except Exception as e:
                    last_err = str(e)
            if not r or not r.ok:
                logger.error(f"Ollama text response not OK: {last_err}")
                return None
            out = r.json()
            content = out.get("response") or ""
            logger.info(f"Text LLM raw content (trunc): {content[:200]}")
            result = _parse_json_relaxed(content)
            if result is None:
                salvage = _salvage_from_content(content)
                if salvage:
                    logger.info("Used salvage parser for text result")
                    return salvage
                logger.error("Failed to parse JSON from text LLM content after sanitation")
                return None

            label = str(result.get("type", "other")).strip().lower()
            tags = result.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).lower()[:64] for t in tags][:25]
            caption = str(result.get("caption", "")).strip()[:400]
            confidence = result.get("confidence")
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

            parsed = {
                "label": label,
                "tags": tags,
                "caption": caption,
                "vision_confidence": confidence,
            }
            logger.info(f"Parsed text result: {parsed}")
            return parsed
        
        # Provider is 'none' - no AI analysis
        return None
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return None


def _parse_json_relaxed(content: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse possibly non-strict JSON (arrays, trailing commas, or prose).
    - Accepts a pure JSON array by returning its first object when present.
    - Otherwise extracts the outermost object braces and parses permissively.
    """
    try:
        if not content:
            return None
        lead = content.lstrip()
        if lead.startswith("["):
            try:
                arr = json5.loads(content)
                if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                    return arr[0]
            except Exception:
                pass
        s = content.find("{")
        e = content.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return None
        snippet = content[s:e+1].strip()
        return json5.loads(snippet)
    except Exception:
        try:
            snippet = re.sub(r",\s*(\])", r"\1", snippet)
            snippet = re.sub(r",\s*(\})", r"\1", snippet)
            snippet = re.sub(r",\s*,", ",", snippet)
            return json.loads(snippet)
        except Exception as e2:
            logger.error(f"Sanitized JSON parse failed: {e2}")
            return None


def _salvage_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Best-effort field extraction when JSON is malformed."""
    try:
        if not content:
            return None
        # Caption
        cap = None
        m = re.search(r'"caption"\s*:\s*"([^"]+)"', content)
        if m:
            cap = m.group(1)[:160]
        # Type/label
        lbl = None
        m = re.search(r'"type"\s*:\s*"([^"]+)"', content)
        if m:
            lbl = m.group(1)[:64]
        # Confidence
        conf = 0.0
        m = re.search(r'"confidence"\s*:\s*([0-9\.]+)', content)
        if m:
            try:
                conf = float(m.group(1))
            except Exception:
                conf = 0.0
        # Tags (very tolerant)
        tags: List[str] = []
        m = re.search(r'"tags"\s*:\s*\[(.*?)\]', content, re.S)
        if m:
            inner = m.group(1)
            for t in re.findall(r'"([^"]+)"', inner):
                tt = t.strip()
                if tt:
                    tags.append(tt)
        if not (lbl or cap or tags):
            return None
        return {
            "label": (lbl or "other"),
            "caption": (cap or ""),
            "tags": tags[:10],
            "vision_confidence": conf,
        }
    except Exception:
        return None


def gpt_vision_fallback(image_b64: str, filename: Optional[str] = None, user_instructions: str = None) -> Optional[Dict[str, Any]]:
    """Cloud vision analysis through the Supabase Edge Function proxy.
    image_b64 should be raw base64 without data URI prefix.
    """
    try:
        system = build_analysis_prompt(DETAILED_SYSTEM_PROMPT, user_instructions)
        # Build data URL for inline base64 image
        data_url = f"data:image/png;base64,{image_b64}"
        user_content: List[Dict[str, Any]] = []
        if filename:
            user_content.append({"type": "text", "text": f"filename: {filename}"})
        user_content.append({"type": "text", "text": "Return STRICT JSON only using the schema."})
        # detail:"high" sends ~1445 image tokens to the model instead of the
        # default ~765 (~2x input cost). On gpt-4o-mini specifically this is
        # the cheapest knob that meaningfully reduces look-alike misclass-
        # ifications — caught a tiger being labelled 'bear' on 2026-05-29
        # 02:12 because the model only had a low-detail thumbnail to work
        # with. Cost stays well under a cent per image.
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "high"},
        })

        resp_data = _call_openai_proxy(
            "vision",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        if not resp_data:
            return None
        choices = resp_data.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "") or ""
        try:
            data = json.loads(content)
        except Exception:
            s = content.find("{"); e = content.rfind("}")
            if s != -1 and e != -1 and e > s:
                data = json.loads(content[s:e+1])
            else:
                return None
        label = str(data.get("type", "other")).strip().lower()
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower()[:64] for t in tags][:25]
        description = str(data.get("description", "")).strip()
        caption = description[:1200] if description else str(data.get("caption", "")).strip()[:1200]
        try:
            conf = float(data.get("confidence", 0))
        except Exception:
            conf = 0.0
        # Include extended fields so the UI can display them from metadata
        return {
            "label": label,
            "tags": tags,
            "caption": caption,
            "vision_confidence": conf,
            "type": data.get("type"),
            "detected_text": data.get("detected_text"),
            "purpose": data.get("purpose"),
            "suggested_filename": data.get("suggested_filename"),
            "description": description,
        }
    except Exception as e:
        logger.error(f"GPT fallback error: {e}")
        return None



def describe_image_detailed(image_path: Path, model: str = None) -> Optional[Dict[str, Any]]:
    """Produce a rich paragraph, type classification, and extras via configured AI provider."""
    try:
        provider = settings.ai_provider
        
        # OpenAI is the primary/default provider - uses gpt_vision_fallback which has detailed prompt
        if provider == 'openai':
            image_b64 = _file_to_b64(image_path)
            if image_b64:
                result = gpt_vision_fallback(image_b64, image_path.name)
                if result:
                    logger.info(f"OpenAI detailed vision successful for {image_path.name}")
                    return result
            logger.warning("OpenAI detailed vision failed")
            return None
        
        # Local (Ollama) provider
        if provider == 'local':
            if model is None:
                model = get_detailed_vision_model()
            if not _ollama_is_alive():
                logger.warning("Ollama is not running at localhost:11434")
                return None
            if not _model_is_available(model):
                logger.info(f"Detailed model '{model}' not available locally. Skipping detailed analyzer.")
                return None

            image_b64 = _file_to_b64(image_path)
            if not image_b64:
                return None

            width = height = None
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                pass
            aspect = round((width / height), 3) if width and height and height != 0 else None

            ctx = []
            if image_path.name:
                ctx.append(f"filename: {image_path.name}")
            if width and height:
                ctx.append(f"dimensions: {width}x{height}, aspect={aspect}")
            ctx_str = "\n".join(ctx)
            prompt = DETAILED_SYSTEM_PROMPT + "\n" + ctx_str + "\nReturn STRICT JSON only."

            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "temperature": 0.2,
                "options": {"num_predict": 1024}
            }

            last_err: Optional[str] = None
            r = None
            for _ in range(2):
                try:
                    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
                    if r.ok:
                        break
                    last_err = r.text
                except Exception as e:
                    last_err = str(e)
            if not r or not r.ok:
                logger.info("Skipping detailed vision: %s", last_err)
                return None

            content = r.json().get("response") or ""
            logger.info(f"Detailed vision raw content (trunc): {content[:200]}")
            data = _parse_json_relaxed(content)
            if data is None:
                salvage = _salvage_from_content(content)
                if salvage:
                    logger.info("Used salvage parser for detailed vision result")
                    salvage.setdefault("label", salvage.get("type", "other"))
                    salvage.setdefault("caption", salvage.get("description", salvage.get("caption", "")))
                    salvage.setdefault("tags", salvage.get("tags", []))
                    return salvage
                return None

            label = str(data.get("type", "other")).strip().lower()
            description = str(data.get("description", "")).strip()
            caption = description[:1200]
            tags = data.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).lower()[:64] for t in tags][:25]
            try:
                confidence = float(data.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            result: Dict[str, Any] = {
                "label": label,
                "tags": tags,
                "caption": caption,
                "vision_confidence": confidence,
                "type": data.get("type"),
                "detected_text": data.get("detected_text"),
                "purpose": data.get("purpose"),
                "suggested_filename": data.get("suggested_filename"),
                "description": description,
            }
            logger.info(f"Detailed vision result: {{'label': {label}, 'len(caption)': {len(caption)}, 'tags': {len(tags)}}}")
            return result
        
        # Provider is 'none' - no AI analysis
        return None
    except Exception as e:
        logger.error(f"describe_image_detailed error: {e}")
        return None
