"""
AI-powered file organization planner.
LLM proposes → App validates → User approves → App executes

Core Principle: The LLM must never directly modify files.
- LLM plans
- App validates
- User approves
- App executes
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────

ORGANIZATION_SCHEMA = """{
  "folders": {
    "<folder-name>": [<file_id>, <file_id>, ...],
    ...
  }
}"""

SYSTEM_PROMPT = f"""You are a file organization assistant. Given a user's instruction and files with metadata, propose how to organize them into folders.

FILE INFORMATION PROVIDED:
- id: unique file identifier (use this in your response)
- name: the filename
- folder: the current subfolder the file lives in (or "." if at the root level)
- ext: the FILE EXTENSION (e.g., .mp4, .json, .png, .pdf) - USE THIS to identify file types!
- label/tags/caption: AI-generated descriptions

STRICT RULES:
1. Return ONLY valid JSON matching this schema:
{ORGANIZATION_SCHEMA}

2. folder-name: descriptive, lowercase, kebab-case (e.g., "screenshots", "videos", "documents")
3. Use ONLY file_ids from the provided list - NEVER invent IDs
4. USE THE FILE EXTENSION (ext:) to identify file types:
   - "videos" = .mp4, .mov, .avi, .mkv, .webm
   - "images/photos" = .jpg, .jpeg, .png, .gif, .webp
   - "screenshots" = .png files with "screenshot" in name OR tagged as screenshot
   - "JSON files" = .json
   - "PDFs" = .pdf
   - "documents" = .doc, .docx, .pdf, .txt
   - "audio" = .mp3, .wav, .m4a, .flac
5. Maximum 2 folder levels
6. Do NOT rename files - only organize into folders
7. NEVER return empty folders - every folder must have at least one file

PRESERVE INSTRUCTIONS (e.g., "keep X as is", "preserve folder X", "don't touch X", "leave X alone"):
- Files inside the named folder must be OMITTED from the plan entirely — do NOT include their file_ids
- Omitted files are left exactly where they are on disk — this is how preservation works
- You can identify which files belong to a folder using the "folder" field
- Example: "organize everything but preserve Work Projects" → include all files EXCEPT those with folder:"Work Projects"
- Only omit files for folders the user explicitly names — organize everything else normally

TWO MODES OF OPERATION:

MODE 1 - REGULAR ORGANIZE (instruction does NOT start with [AUTO-ORGANIZE]):
- ONLY include files that SPECIFICALLY MATCH the user's instruction
- Leave ALL other files OUT of the response
- It's OK to return fewer files than provided
- Do NOT create "misc" or "other" folders

MODE 2 - AUTO-ORGANIZE (instruction starts with [AUTO-ORGANIZE]):
- Follow user's specific instructions EXACTLY for mentioned file types
- ALSO organize ALL remaining files by their file type
- EVERY file MUST be included - no file left out (unless user explicitly asks to preserve a folder)
- Example: "screenshots to screenshots-folder" means:
  * Screenshots → "screenshots-folder" (as user specified)
  * Videos → "videos" (organized by type)
  * Documents → "documents" (organized by type)
  * etc.

JSON only. No markdown. No explanation. No prose."""


# ─────────────────────────────────────────────────────────────
# FILE SUMMARY FOR LLM
# ─────────────────────────────────────────────────────────────

def _infer_file_type_hints(file_name: str) -> List[str]:
    """
    Infer file type hints from filename patterns.
    This helps the AI identify files even without proper tags.
    """
    hints = []
    name_lower = file_name.lower()
    
    # Screenshot patterns
    if any(p in name_lower for p in ['screenshot', 'screen shot', 'screen_shot', 'snip', 'capture']):
        hints.append('screenshot')
    
    # Invoice/Receipt patterns
    if any(p in name_lower for p in ['invoice', 'receipt', 'bill', 'payment']):
        hints.append('invoice/receipt')
    
    # Document patterns
    if any(p in name_lower for p in ['document', 'doc', 'report', 'letter', 'contract']):
        hints.append('document')
    
    # Photo patterns
    if any(p in name_lower for p in ['img_', 'dsc_', 'photo', 'pic_', 'image']):
        hints.append('photo')
    
    # Video patterns  
    if any(p in name_lower for p in ['vid_', 'video', 'mov_', 'clip']):
        hints.append('video')
    
    # Download patterns
    if any(p in name_lower for p in ['download', 'downloaded']):
        hints.append('download')
    
    return hints


def build_file_summary(files: List[Dict[str, Any]], max_files: int = 300) -> str:
    """
    Create a compact summary of files for the LLM context.
    Limits tokens while preserving key metadata.
    Also includes file extension for accurate type matching.
    """
    lines = []
    for f in files[:max_files]:
        fid = f.get('id')
        name = f.get('file_name', 'unknown')[:50]
        label = f.get('label', '') or ''
        caption = (f.get('caption', '') or '')[:80]
        tags = f.get('tags', []) or []
        subfolder = f.get('subfolder', '.') or '.'

        # Extract file extension for accurate type matching
        ext = ''
        if '.' in name:
            ext = name[name.rfind('.'):].lower()

        # Add inferred hints from filename patterns
        hints = _infer_file_type_hints(name)
        all_tags = list(tags[:8]) + hints
        tags_str = ', '.join(all_tags) if all_tags else ''

        line = f"id:{fid} | {name} | folder:{subfolder} | ext:{ext} | label:{label} | tags:[{tags_str}]"
        if caption:
            line += f" | caption:{caption}"
        lines.append(line)
    
    summary = "\n".join(lines)
    
    if len(files) > max_files:
        summary += f"\n... and {len(files) - max_files} more files"
    
    return summary


# ─────────────────────────────────────────────────────────────
# LLM REQUEST
# ─────────────────────────────────────────────────────────────

def request_organization_plan(
    user_instruction: str,
    files: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Send user instruction + file metadata to LLM.
    Returns the proposed plan as a dict, or None on failure.
    
    The LLM acts only as a planner - it never executes anything.
    """
    from .settings import settings
    
    if not files:
        logger.warning("No files provided for organization")
        return None
    
    file_summary = build_file_summary(files)
    
    # Detect auto-organize mode vs specific instruction mode
    is_auto_organize = user_instruction.startswith("[AUTO-ORGANIZE]")
    is_existing_folders_only = "EXISTING FOLDERS ONLY" in user_instruction

    if is_auto_organize and is_existing_folders_only:
        # MODE A — ORGANIZE AS-IS: the AI MUST use only the folders already
        # passed in the instruction. No new folders, no misc, no catch-all.
        user_message = f"""User instruction: "{user_instruction}"

Files to organize ({len(files)} total):
{file_summary}

CRITICAL - EXISTING FOLDERS ONLY:
- You can ONLY use the folders listed in the instruction - DO NOT create new folders
- Put each file in the MOST APPROPRIATE existing folder based on file type/content
- You MUST include EVERY file_id in your response ({len(files)} total)
- Each file_id must appear in exactly ONE folder
- Use your best judgment to match files to the closest existing folder

Propose an organization plan. Return JSON only."""
    elif is_auto_organize:
        # Auto-organize: MUST include ALL files
        user_message = f"""User instruction: "{user_instruction}"

Files to organize ({len(files)} total):
{file_summary}

CRITICAL OVERRIDE FOR AUTO-ORGANIZE:
- You MUST include EVERY file_id in your response
- Each file_id must appear in exactly ONE folder
- Do NOT skip any files
- If a file doesn't fit a category, put it in 'misc' or 'other'
- Total files in your response must equal {len(files)}

Propose an organization plan. Return JSON only."""
    else:
        # Specific instruction: only organize matching files
        user_message = f"""User instruction: "{user_instruction}"

Files available ({len(files)} total):
{file_summary}

IMPORTANT - SPECIFIC INSTRUCTION MODE:
- ONLY include files that EXACTLY match what the user asked for
- If user says "move screenshots to X and JSON to Y", ONLY include screenshot files and JSON files
- Do NOT include any other files in your response
- The total number of files in your response should be MUCH LESS than {len(files)} 
- Leave all non-matching files OUT of the response entirely (they will stay in place)

Propose an organization plan. Return JSON only."""

    provider = settings.ai_provider
    
    if provider == 'openai':
        return _request_openai(user_message)
    elif provider == 'local':
        return _request_ollama(user_message)
    else:
        logger.warning("No AI provider configured")
        return None


def request_plan_refinement(
    original_instruction: str,
    current_plan: Dict[str, Any],
    feedback: str,
    files: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Refine an existing plan based on user feedback.
    Returns the updated plan as a dict, or None on failure.
    """
    from .settings import settings
    
    if not current_plan:
        logger.warning("No plan to refine")
        return None
    
    file_summary = build_file_summary(files)
    
    # Format current plan for context
    current_plan_json = json.dumps(current_plan, indent=2)
    
    user_message = f"""Original instruction: "{original_instruction}"

Current plan:
{current_plan_json}

User feedback: "{feedback}"

Files available ({len(files)} total):
{file_summary}

Based on the user feedback, provide an UPDATED organization plan.
Apply the user's requested changes to the current plan.
Return the complete updated plan as JSON only."""

    provider = settings.ai_provider
    
    if provider == 'openai':
        return _request_openai(user_message)
    elif provider == 'local':
        return _request_ollama(user_message)
    else:
        logger.warning("No AI provider configured")
        return None


def _request_openai(user_message: str) -> Optional[Dict[str, Any]]:
    """Request plan via OpenAI through the Supabase Edge Function proxy."""
    try:
        from .vision import _call_openai_proxy

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        resp_data = _call_openai_proxy("chat", messages, max_tokens=4000, temperature=0.1)
        if not resp_data:
            logger.error("OpenAI proxy returned no data for organization plan")
            return None

        choices = resp_data.get("choices", [])
        if not choices:
            logger.warning("OpenAI proxy response had no choices")
            return None
        content = choices[0].get("message", {}).get("content", "") or ""
        if not content:
            logger.warning("OpenAI proxy response had empty content")
            return None

        logger.info(f"OpenAI organization response (truncated): {content[:300]}")
        return _parse_json(content)
    except Exception as e:
        logger.error(f"OpenAI organization request failed: {e}")
        return None


def _request_ollama(user_message: str) -> Optional[Dict[str, Any]]:
    """Request plan via local Ollama."""
    import requests
    from .vision import OLLAMA_URL, get_local_model, _ollama_is_alive
    
    if not _ollama_is_alive():
        logger.warning("Ollama not running")
        return None
    
    payload = {
        "model": get_local_model(),
        "prompt": SYSTEM_PROMPT + "\n\n" + user_message,
        "stream": False,
        "temperature": 0.1,
    }
    
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
        if r.ok:
            content = r.json().get("response", "")
            logger.info(f"Ollama organization response (truncated): {content[:300]}")
            return _parse_json(content)
    except Exception as e:
        logger.error(f"Ollama organization request failed: {e}")
    return None


def _parse_json(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown wrapping."""
    # Try direct parse first
    try:
        return json.loads(content)
    except:
        pass
    
    # Try extracting JSON from markdown code block
    if "```" in content:
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
    
    # Try finding JSON object in the content
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end+1])
        except:
            pass
    
    logger.error("Failed to parse JSON from LLM response")
    return None


def deduplicate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove duplicate file_ids from the plan.
    If a file appears in multiple folders, keep only the first occurrence.
    This handles cases where the AI mistakenly puts the same file in multiple folders.
    """
    if not plan or "folders" not in plan:
        return plan
    
    seen_ids = set()
    duplicates_removed = 0
    cleaned_folders = {}
    
    for folder_name, file_ids in plan.get("folders", {}).items():
        if not isinstance(file_ids, list):
            continue
        
        cleaned_ids = []
        for fid in file_ids:
            try:
                fid_int = int(fid)
                if fid_int not in seen_ids:
                    seen_ids.add(fid_int)
                    cleaned_ids.append(fid_int)
                else:
                    duplicates_removed += 1
                    logger.debug(f"Removed duplicate file_id {fid_int} from folder '{folder_name}'")
            except (TypeError, ValueError):
                # Keep invalid IDs for validation to catch
                cleaned_ids.append(fid)
        
        if cleaned_ids:
            cleaned_folders[folder_name] = cleaned_ids
    
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate file_id(s) from AI plan")
    
    return {"folders": cleaned_folders}


def ensure_all_files_included(plan: Dict[str, Any], all_file_ids: set, files_info: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure all provided file IDs are included in the plan.

    If the AI missed any files, place them in the most relevant existing folder
    based on each file's metadata. This prevents files from being left
    unorganized and avoids inventing a generic 'misc' folder when real folders
    already exist in the plan.

    Args:
        plan: The organization plan from AI
        all_file_ids: Set of all file IDs that should be in the plan
        files_info: Optional list of file info dicts for better folder selection

    Returns:
        Updated plan with all files included
    """
    if not plan or "folders" not in plan:
        plan = {"folders": {}}

    # Collect all file IDs currently in the plan
    included_ids = set()
    for folder_name, file_ids in plan.get("folders", {}).items():
        for fid in file_ids:
            try:
                included_ids.add(int(fid))
            except (TypeError, ValueError):
                pass

    # Find missing file IDs
    missing_ids = all_file_ids - included_ids

    if not missing_ids:
        return plan  # All files are included

    logger.warning(f"AI plan missing {len(missing_ids)} file(s). Adding them to existing folders.")

    # Log which files are missing for debugging
    if files_info:
        missing_names = []
        for f in files_info:
            try:
                if int(f.get('id', 0)) in missing_ids:
                    missing_names.append(f.get('file_name', 'unknown'))
            except (TypeError, ValueError):
                pass
        if missing_names:
            logger.info(f"Missing files: {', '.join(missing_names[:10])}")

    folders = plan.get("folders", {})

    # If real folders already exist in the plan, route each missing file to the
    # MOST RELEVANT one based on its own metadata (name/ext/label/category/tags)
    # rather than inventing a generic 'misc' folder or dumping into the first
    # folder. A 'misc' folder is created only as a last resort when the plan
    # has no folders at all.
    if folders:
        info_by_id = {}
        if files_info:
            for f in files_info:
                try:
                    info_by_id[int(f.get('id', 0))] = f
                except (TypeError, ValueError):
                    pass
        folder_names = list(folders.keys())
        for missing_id in missing_ids:
            target = _best_folder_for_file(info_by_id.get(missing_id), folder_names)
            folders[target].append(missing_id)
            logger.info(f"Added missing file id {missing_id} to most relevant folder '{target}'")
    else:
        # No folders at all — create a misc folder as the last resort.
        folders['misc'] = list(missing_ids)
        logger.info(f"Added {len(missing_ids)} missing file(s) to 'misc' folder (no existing folders)")

    return {"folders": folders}


def _best_folder_for_file(file_info: Optional[Dict[str, Any]], folder_names: List[str]) -> str:
    """Pick the existing folder whose name best matches a file's metadata.

    Scoring strategy:
    1. Tags / filename / label tokens that overlap a folder name (substring
       either way) score 1.0 — the strongest signal.
    2. Otherwise use difflib similarity.
    3. If the best non-catch-all match is weak (no strong substring hit), and
       a catch-all-style folder exists (name contains 'everything', 'other',
       'misc', 'general', 'rest', 'else'), prefer the catch-all. This
       prevents a bear photo from being shoved into 'lions' just because of
       random character overlap when an 'everything-else' folder is sitting
       right there for exactly this case.
    4. Falls back to the first folder when no signal is available, and to
       'misc' when there are no folders at all.
    """
    if not folder_names:
        return 'misc'
    if not file_info:
        return folder_names[0]

    import difflib

    catchall_keywords = {'everything', 'other', 'misc', 'general', 'rest', 'else', 'miscellaneous', 'various', 'unsorted'}

    def _is_catchall(folder: str) -> bool:
        # Only the leaf folder name counts — 'everything-else/pigs' is NOT a
        # catch-all (the LEAF is 'pigs'), even though its parent name contains
        # 'else'. We use forward slash because the helper that builds nested
        # folder lists also uses forward slashes.
        leaf = folder.rsplit('/', 1)[-1].lower()
        return any(k in leaf for k in catchall_keywords)

    # Build descriptive terms for the file from its metadata
    terms = []
    name = file_info.get('file_name', '') or ''
    if name:
        stem = name.rsplit('.', 1)[0]
        ext = name.rsplit('.', 1)[1].lower() if '.' in name else ''
        terms.extend(part for part in stem.replace('_', ' ').replace('-', ' ').split() if part)
        if ext:
            terms.append(ext)
    if file_info.get('label'):
        terms.append(str(file_info['label']))
    if file_info.get('category'):
        terms.append(str(file_info['category']))
    for tag in (file_info.get('tags') or []):
        terms.append(str(tag))
    terms = [t.lower() for t in terms if t]

    best_folder = folder_names[0]
    best_score = 0.0
    had_strong_hit = False
    for folder in folder_names:
        folder_lc = folder.lower()
        # Don't let a catch-all folder win on fuzzy similarity alone — only via
        # an actual substring hit. Otherwise short generic names ("Other")
        # would be picked accidentally.
        if _is_catchall(folder):
            continue
        score = 0.0
        for term in terms:
            if term and (term in folder_lc or folder_lc in term):
                score = max(score, 1.0)
            else:
                score = max(score, difflib.SequenceMatcher(None, term, folder_lc).ratio())
        if score >= 0.85:
            had_strong_hit = True
        if score > best_score:
            best_score = score
            best_folder = folder

    # If no real signal pointed at a specific folder, prefer a catch-all if one
    # exists — that's the whole point of the catch-all folder.
    if not had_strong_hit:
        for folder in folder_names:
            if _is_catchall(folder):
                return folder

    return best_folder


# ─────────────────────────────────────────────────────────────
# VALIDATION (MANDATORY - App is the final authority)
# ─────────────────────────────────────────────────────────────

def validate_plan(
    plan: Dict[str, Any],
    valid_file_ids: set,
    max_depth: int = 2
) -> Tuple[bool, List[str]]:
    """
    Validate the organization plan for safety.
    
    This is the critical safety gate - the app validates everything
    before any file operation occurs.
    
    Checks:
    - All file_ids exist in our database
    - No duplicates across folders
    - Folder depth is limited
    - No system/root folders touched
    - No path traversal attacks
    
    Returns: (is_valid, list_of_errors)
    """
    errors = []
    
    if not plan:
        errors.append("Plan is empty")
        return False, errors
    
    folders = plan.get("folders")
    if not folders or not isinstance(folders, dict):
        errors.append("Plan must contain 'folders' dict")
        return False, errors
    
    seen_ids = set()
    
    for folder_name, file_ids in folders.items():
        # Safety checks on folder name
        if not folder_name or not isinstance(folder_name, str):
            errors.append(f"Invalid folder name: {folder_name}")
            continue
        
        # Prevent path traversal
        if ".." in folder_name:
            errors.append(f"Path traversal not allowed: {folder_name}")
            continue
        
        # Prevent absolute paths
        if folder_name.startswith("/") or folder_name.startswith("\\"):
            errors.append(f"Absolute paths not allowed: {folder_name}")
            continue
        
        # Windows drive letters
        if ":" in folder_name:
            errors.append(f"Drive letters not allowed: {folder_name}")
            continue
        
        # Check for system folder names
        dangerous_names = {'system32', 'windows', 'program files', 'programdata', '$recycle.bin'}
        if folder_name.lower() in dangerous_names:
            errors.append(f"System folder name not allowed: {folder_name}")
            continue
        
        # Check depth
        depth = folder_name.replace("\\", "/").count("/") + 1
        if depth > max_depth:
            errors.append(f"Folder too deep ({depth} > {max_depth}): {folder_name}")
        
        # Validate file IDs
        if not isinstance(file_ids, list):
            errors.append(f"Folder '{folder_name}' must have list of file IDs")
            continue
        
        for fid in file_ids:
            # Ensure fid is an integer
            try:
                fid_int = int(fid)
            except (TypeError, ValueError):
                errors.append(f"Invalid file_id type: {fid}")
                continue
            
            if fid_int not in valid_file_ids:
                errors.append(f"Unknown file_id: {fid_int}")
            elif fid_int in seen_ids:
                errors.append(f"Duplicate file_id: {fid_int} (appears in multiple folders)")
            seen_ids.add(fid_int)
    
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────
# CONVERT PLAN TO MOVE OPERATIONS
# ─────────────────────────────────────────────────────────────

def plan_to_moves(
    plan: Dict[str, Any],
    files_by_id: Dict[int, Dict[str, Any]],
    destination_root: Path
) -> List[Dict[str, Any]]:
    """
    Convert validated plan to concrete move operations.
    
    This is deterministic - no AI involved here.
    The app fully controls what actually happens.
    """
    moves = []
    skipped_not_found = 0
    skipped_no_info = 0
    skipped_already_in_dest = 0
    
    for folder_name, file_ids in plan.get("folders", {}).items():
        dest_folder = destination_root / folder_name
        
        for fid in file_ids:
            # Normalize fid to int
            try:
                fid_int = int(fid)
            except (TypeError, ValueError):
                logger.warning(f"Invalid file ID type: {fid}")
                continue
            
            file_info = files_by_id.get(fid_int)
            if not file_info:
                skipped_no_info += 1
                logger.debug(f"No file info for ID {fid_int}")
                continue
            
            source_path = Path(file_info['file_path'])
            if not source_path.exists():
                skipped_not_found += 1
                logger.debug(f"Source file doesn't exist: {source_path}")
                continue
            
            dest_path = dest_folder / source_path.name
            
            # Skip files that are already in the destination folder
            # This prevents "moving" files to where they already are
            if source_path.parent.resolve() == dest_folder.resolve():
                skipped_already_in_dest += 1
                logger.debug(f"Skipping {source_path.name} - already in destination folder {dest_folder}")
                continue
            
            # Also skip if the exact destination file already exists and is the same file
            if dest_path.exists() and source_path.resolve() == dest_path.resolve():
                skipped_already_in_dest += 1
                logger.debug(f"Skipping {source_path.name} - source and destination are the same file")
                continue
            
            # Handle collisions by adding numeric suffix (only for different files)
            counter = 1
            original_stem = source_path.stem
            original_suffix = source_path.suffix
            while dest_path.exists():
                dest_path = dest_folder / f"{original_stem} ({counter}){original_suffix}"
                counter += 1
            
            moves.append({
                "file_id": fid_int,
                "file_name": source_path.name,
                "source_path": str(source_path),
                "destination_path": str(dest_path),
                "destination_folder": folder_name,
                "size": file_info.get('file_size', 0),
            })
    
    # Log summary
    total_in_plan = sum(len(fids) for fids in plan.get("folders", {}).values())
    logger.info(f"plan_to_moves: {len(moves)} valid moves from {total_in_plan} files in plan. "
                f"Skipped: {skipped_not_found} not found, {skipped_no_info} no info, "
                f"{skipped_already_in_dest} already in destination")
    
    return moves


# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_plan_summary(plan: Dict[str, Any], files_by_id: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get a human-readable summary of the organization plan.
    """
    folders = plan.get("folders", {})
    
    total_files = sum(len(fids) for fids in folders.values())
    total_size = 0
    
    folder_summaries = []
    for folder_name, file_ids in folders.items():
        folder_size = 0
        for fid in file_ids:
            try:
                fid_int = int(fid)
                file_info = files_by_id.get(fid_int, {})
                folder_size += file_info.get('file_size', 0)
            except:
                pass
        total_size += folder_size
        
        folder_summaries.append({
            "name": folder_name,
            "file_count": len(file_ids),
            "size_bytes": folder_size,
            "size_mb": round(folder_size / (1024 * 1024), 2),
        })
    
    return {
        "total_folders": len(folders),
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "folders": folder_summaries,
    }
