"""
Search functionality for finding files using natural language queries.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .database import file_index
from .scan import scan_directory
from .categorize import get_file_metadata
from .vision import analyze_image, analyze_text, gpt_vision_fallback, describe_image_detailed
from .settings import settings
from .text_extract import extract_file_text, get_supported_text_formats
import os
from .embeddings import embed_text
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

# Parallel processing settings
MAX_CONCURRENT_AI_REQUESTS = 50  # Tier 2: 5,000 RPM allows 50-80 safely

# Media file extensions that count against the index limit
MEDIA_EXTENSIONS = {
    # Images
    '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp', '.tiff', '.tif',
    '.heic', '.heif', '.raw', '.cr2', '.nef', '.arw',
    # Videos
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
    # Audio
    '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.aiff', '.alac',
}


def is_media_file(file_path: Path) -> bool:
    """Check if a file is an image, video, or audio that counts against the index limit."""
    return file_path.suffix.lower() in MEDIA_EXTENSIONS


def count_media_files(files: list) -> int:
    """Count how many media files (images/videos/audio) are in the file list."""
    count = 0
    for f in files:
        # Check all possible path keys (scan uses 'source_path', others use 'path' or 'file_path')
        path = f.get('path') or f.get('file_path') or f.get('source_path')
        if path:
            if is_media_file(Path(path)):
                count += 1
    return count

class SearchService:
    """High-level search service for file discovery."""
    
    def __init__(self):
        self.index = file_index
        self._cancel_flag = threading.Event()
        self._pause_flag = threading.Event()
    
    def cancel_indexing(self):
        """Signal to cancel ongoing indexing operation."""
        self._cancel_flag.set()
        self._pause_flag.clear()  # Unpause so it can exit
        logger.info("Indexing cancellation requested")
    
    def pause_indexing(self):
        """Signal to pause ongoing indexing operation."""
        self._pause_flag.set()
        logger.info("Indexing paused")
    
    def resume_indexing(self):
        """Signal to resume a paused indexing operation."""
        self._pause_flag.clear()
        logger.info("Indexing resumed")
    
    def is_paused(self) -> bool:
        """Check if indexing is currently paused."""
        return self._pause_flag.is_set()
    
    def _check_index_limit(self, media_count: int) -> Dict[str, Any]:
        """
        Check if user can index the given number of media files.
        
        Args:
            media_count: Number of media files to index
            
        Returns:
            Dict with 'allowed', 'remaining', 'limit', 'plan', and optionally 'reason'
        """
        try:
            from app.core.supabase_client import supabase_auth
            
            if not supabase_auth.is_authenticated:
                # Not authenticated - allow indexing but warn
                logger.warning("User not authenticated, skipping index limit check")
                return {'allowed': True, 'remaining': 999999, 'limit': 999999, 'plan': 'unknown'}
            
            return supabase_auth.can_index_media(media_count)
            
        except Exception as e:
            logger.error(f"Error checking index limit: {e}")
            # On error, allow indexing to not block users
            return {'allowed': True, 'remaining': 999999, 'limit': 999999, 'plan': 'unknown', 'error': str(e)}
    
    def _update_index_usage(self, media_count: int) -> bool:
        """
        Update the index usage after successfully indexing media files.
        
        Args:
            media_count: Number of media files that were indexed
            
        Returns:
            True if successful
        """
        if media_count <= 0:
            return True
            
        try:
            from app.core.supabase_client import supabase_auth
            
            if not supabase_auth.is_authenticated:
                return True
            
            return supabase_auth.increment_index_usage(media_count)
            
        except Exception as e:
            logger.error(f"Error updating index usage: {e}")
            return False
    
    def _wait_if_paused(self):
        """Block while paused, checking every 100ms. Returns True if should continue, False if cancelled."""
        while self._pause_flag.is_set():
            if self._cancel_flag.is_set():
                return False
            import time
            time.sleep(0.1)
        return not self._cancel_flag.is_set()
    
    def _process_single_file(self, file_data: Dict, directory_path: Path, force_ai: bool = False, user_instructions: str = None) -> Dict[str, Any]:
        """
        Process a single file with AI analysis. Called in parallel.
        
        Returns:
            Dictionary with file metadata and AI analysis results
        """
        try:
            # Get file path
            if 'source_path' in file_data:
                file_path = Path(file_data['source_path'])
            else:
                file_path = directory_path / file_data['name']
            
            # Get basic metadata
            full_metadata = get_file_metadata(file_path)
            
            # Compute content hash
            try:
                h = hashlib.sha256()
                with open(file_path, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                        h.update(chunk)
                full_metadata['content_hash'] = h.hexdigest()
            except Exception:
                full_metadata['content_hash'] = None
            
            full_metadata['last_indexed_at'] = datetime.utcnow().isoformat()
            full_metadata['source_path'] = str(file_path)
            
            # Check if file already indexed with same content hash - skip AI analysis (unless forced)
            existing = self.index.get_file_by_path(str(file_path))
            if (not force_ai) and existing and existing.get('content_hash') == full_metadata.get('content_hash'):
                # File unchanged - skip expensive AI analysis, return existing data
                logger.debug(f"Skipping unchanged file: {file_path.name}")
                existing['_file_path'] = file_path
                existing['_skipped'] = True
                return existing
            if force_ai:
                logger.info(f"Forcing re-index (AI) for: {file_path}")
            
            # Check for pause/cancellation before AI call
            if not self._wait_if_paused():
                return {'_file_path': file_path, '_cancelled': True, **full_metadata}
            
            # AI Analysis - Vision for images/PDFs; Text LLM for others
            ext = file_path.suffix.lower()
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.avif', '.heic', '.heif', '.ico', '.raw', '.cr2', '.nef', '.arw', '.pdf'}
            
            if ext in image_extensions:
                # Image/PDF: Use vision model
                if settings.ai_provider == 'openai':
                    from .vision import _file_to_b64
                    image_b64 = _file_to_b64(file_path)
                    if image_b64:
                        gptv = gpt_vision_fallback(image_b64, filename=file_path.name, user_instructions=user_instructions)
                        if gptv:
                            full_metadata.update(gptv)
                            full_metadata['ai_source'] = f'openai:{settings.openai_vision_model}'
                else:
                    # Local models path
                    use_detailed = os.environ.get('USE_DETAILED_VISION', '1').strip() not in {'0', 'false', 'no'}
                    vision = None
                    if use_detailed:
                        vision = describe_image_detailed(file_path)
                    if not vision or not vision.get('caption'):
                        vision = analyze_image(file_path, user_instructions=user_instructions)
                    if not vision or not vision.get('caption'):
                        # Fallback to cloud
                        from .vision import _file_to_b64
                        image_b64 = _file_to_b64(file_path)
                        if image_b64:
                            gptv = gpt_vision_fallback(image_b64, filename=file_path.name, user_instructions=user_instructions)
                            if gptv:
                                vision = gptv
                                full_metadata['ai_source'] = f'openai:{settings.openai_vision_model}'
                    if vision:
                        full_metadata.update(vision)
                        if 'ai_source' not in full_metadata:
                            full_metadata['ai_source'] = 'ollama:local'
            else:
                # Non-image: Extract text and analyze
                snippet = ""
                try:
                    extracted = extract_file_text(file_path)
                    if extracted:
                        snippet = extracted
                        if ext == '.csv' or ext in {'.xlsx', '.xls'}:
                            full_metadata['ocr_text'] = extracted[:5000]
                            full_metadata['has_ocr'] = True
                    else:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                            snippet = fh.read(8000)
                except Exception:
                    snippet = ""

                if snippet:
                    tvision = analyze_text(snippet, filename=file_path.name, user_instructions=user_instructions)
                    if tvision:
                        full_metadata.update(tvision)
                        if settings.ai_provider == 'openai':
                            full_metadata['ai_source'] = 'openai:gpt-4o-mini'
                        else:
                            full_metadata['ai_source'] = 'ollama:qwen2.5vl'
            
            full_metadata['_file_path'] = file_path
            return full_metadata
            
        except Exception as e:
            logger.error(f"Error processing file {file_data.get('name', 'unknown')}: {e}")
            return {
                '_file_path': Path(file_data.get('source_path', file_data.get('name', 'unknown'))),
                '_error': str(e),
                'name': file_data.get('name', 'unknown'),
                'source_path': file_data.get('source_path', ''),
            }
    
    def index_single_file(self, file_path: Path, force_ai: bool = False, user_instructions: str = None) -> Dict[str, Any]:
        """
        Index a single file with AI analysis.

        Args:
            file_path: Path to the file to index
            force_ai: Force re-analysis even if file hasn't changed
            user_instructions: Optional focus instructions for AI tag generation

        Returns:
            Dictionary with result info (or 'error' key on failure)
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {'error': f'File not found: {file_path}'}
            
            if not file_path.is_file():
                return {'error': f'Not a file: {file_path}'}
            
            # Check if this is a media file (counts against index limit)
            file_is_media = is_media_file(file_path)
            
            # Check index limit before processing media files
            if file_is_media:
                limit_check = self._check_index_limit(1)
                if not limit_check.get('allowed'):
                    reason = limit_check.get('reason', 'Index limit reached')
                    logger.warning(f"Cannot index media file - limit reached: {reason}")
                    return {'error': reason, 'limit_reached': True}
            
            # Create file_data dict that _process_single_file expects
            file_data = {
                'source_path': str(file_path),
                'name': file_path.name
            }
            
            # Process the file with AI
            result = self._process_single_file(file_data, file_path.parent, force_ai=force_ai, user_instructions=user_instructions)
            
            # Handle errors
            file_path_result = result.pop('_file_path', None)
            error = result.pop('_error', None)
            was_cancelled = result.pop('_cancelled', False)
            was_skipped = result.pop('_skipped', False)
            
            if error:
                return {'error': error}
            
            if was_cancelled:
                return {'error': 'Indexing cancelled', 'cancelled': True}
            
            if was_skipped:
                return {'skipped': True, 'path': str(file_path)}
            
            # Add to index
            if self.index.add_file(result):
                # Create embedding
                try:
                    rec = self.index.get_file_by_path(str(file_path))
                    if rec:
                        text_parts = [rec.get('file_name') or '']
                        if rec.get('label'):
                            text_parts.append(rec['label'])
                        if rec.get('tags'):
                            text_parts.append(' '.join(rec['tags']))
                        if rec.get('caption'):
                            text_parts.append(rec['caption'])
                        if rec.get('ocr_text'):
                            text_parts.append(rec['ocr_text'])
                        text_blob = ' '.join([t for t in text_parts if t])[:5000]
                        if text_blob and rec.get('id'):
                            self.index.store_embedding(rec['id'], text_blob)
                except Exception as emb_err:
                    logger.warning(f"Embedding error for {file_path}: {emb_err}")
                
                # Update index usage for media files (images, videos, audio)
                if file_is_media:
                    self._update_index_usage(1)
                    logger.debug(f"Updated index usage: +1 media file ({file_path.name})")
                
                return {'success': True, 'path': str(file_path), 'is_media': file_is_media}
            else:
                return {'error': 'Failed to add file to index'}
                
        except Exception as e:
            logger.error(f"Error indexing single file {file_path}: {e}")
            return {'error': str(e)}
    
    def index_directory(
        self,
        directory_path: Path,
        recursive: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        user_instructions: str = None,
    ) -> Dict[str, Any]:
        """
        Index all files in a directory for search using parallel processing.
        
        Args:
            directory_path: Directory to index
            recursive: Whether to scan subdirectories
            progress_cb: Callback for progress updates (current, total, message)
            
        Returns:
            Dictionary with indexing statistics
        """
        try:
            logger.info(f"Starting to index directory: {directory_path}")
            self._cancel_flag.clear()
            self._pause_flag.clear()
            
            # Scan directory for files
            files = scan_directory(directory_path)
            total = len(files)
            
            if progress_cb:
                progress_cb(0, total, "Checking index limits...")
            
            if total == 0:
                return {
                    'total_files': 0,
                    'indexed_files': 0,
                    'files_with_ocr': 0,
                    'directory': str(directory_path)
                }
            
            # Check media file limit before indexing
            media_count = count_media_files(files)
            logger.info(f"Found {media_count} media files (images/videos/audio) out of {total} total files")
            limit_check = self._check_index_limit(media_count)
            
            if not limit_check['allowed']:
                logger.warning(f"Index limit exceeded: {limit_check.get('reason', 'Unknown')}")
                return {
                    'total_files': total,
                    'indexed_files': 0,
                    'files_with_ocr': 0,
                    'directory': str(directory_path),
                    'limit_exceeded': True,
                    'limit_info': limit_check
                }
            
            if progress_cb:
                progress_cb(0, total, "Starting parallel indexing...")
            
            # Process files in parallel
            indexed_count = 0
            ocr_count = 0
            skipped_count = 0
            completed = 0
            cancelled = False
            
            logger.info(f"Processing {total} files with {MAX_CONCURRENT_AI_REQUESTS} concurrent workers")
            
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_AI_REQUESTS) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(self._process_single_file, file_data, directory_path, False, user_instructions): idx
                    for idx, file_data in enumerate(files)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_idx):
                    # Check for pause - wait until resumed
                    if not self._wait_if_paused():
                        cancelled = True
                        for f in future_to_idx:
                            f.cancel()
                        break
                    
                    if self._cancel_flag.is_set():
                        cancelled = True
                        # Cancel remaining futures
                        for f in future_to_idx:
                            f.cancel()
                        break
                    
                    try:
                        result = future.result(timeout=120)  # 2 min timeout per file
                        file_path = result.pop('_file_path', None)
                        error = result.pop('_error', None)
                        was_cancelled = result.pop('_cancelled', False)
                        was_skipped = result.pop('_skipped', False)
                        
                        if was_cancelled:
                            cancelled = True
                            break
                        
                        if error:
                            logger.warning(f"File processing error: {error}")
                            completed += 1
                            continue
                        
                        # Handle skipped (unchanged) files
                        if was_skipped:
                            skipped_count += 1
                            completed += 1
                            if progress_cb:
                                name = file_path.name if file_path else "file"
                                try:
                                    progress_cb(completed, total, f"Skipped (unchanged): {name}")
                                except InterruptedError:
                                    self._cancel_flag.set()
                                    cancelled = True
                                    break
                            continue
                        
                        # Index the file
                        if self.index.add_file(result):
                            indexed_count += 1
                            if result.get('has_ocr', False):
                                ocr_count += 1
                            
                            # Create embedding (quick operation)
                            try:
                                rec = self.index.get_file_by_path(str(file_path))
                                if rec:
                                    text_parts = [rec.get('file_name') or '']
                                    if rec.get('label'):
                                        text_parts.append(rec['label'])
                                    if rec.get('tags'):
                                        text_parts.append(' '.join(rec['tags']))
                                    if rec.get('caption'):
                                        text_parts.append(rec['caption'])
                                    if rec.get('ocr_text'):
                                        text_parts.append(rec['ocr_text'])
                                    text_blob = ' '.join([t for t in text_parts if t])[:5000]
                                    vec = embed_text(text_blob)
                                    if vec:
                                        self.index.upsert_embedding(rec['id'], 'ollama:nomic-embed-text', vec)
                            except Exception:
                                pass
                        
                        completed += 1
                        
                        # Progress callback
                        if progress_cb:
                            name = file_path.name if file_path else "file"
                            try:
                                progress_cb(completed, total, f"Indexed: {name}")
                            except InterruptedError:
                                self._cancel_flag.set()
                                cancelled = True
                                break
                                
                    except Exception as e:
                        completed += 1
                        logger.error(f"Error in future result: {e}")
            
            if cancelled:
                logger.info(f"Indexing cancelled after {indexed_count} files ({skipped_count} skipped)")
                # Still update usage for files that WERE indexed before cancellation
                if indexed_count > 0 and media_count > 0:
                    media_indexed = min(media_count, indexed_count)
                    self._update_index_usage(media_indexed)
                    logger.info(f"Updated index usage: +{media_indexed} media files (before cancel)")
                return {
                    'total_files': total,
                    'indexed_files': indexed_count,
                    'skipped_files': skipped_count,
                    'files_with_ocr': ocr_count,
                    'directory': str(directory_path),
                    'cancelled': True,
                    'media_indexed': min(media_count, indexed_count) if indexed_count > 0 else 0
                }
            
            logger.info(f"Indexed {indexed_count} files, skipped {skipped_count} unchanged ({ocr_count} with OCR)")
            
            # Update index usage for media files
            if indexed_count > 0 and media_count > 0:
                # Calculate how many media files were actually indexed (not skipped)
                media_indexed = min(media_count, indexed_count)
                self._update_index_usage(media_indexed)
                logger.info(f"Updated index usage: +{media_indexed} media files")
            
            return {
                'total_files': total,
                'indexed_files': indexed_count,
                'skipped_files': skipped_count,
                'files_with_ocr': ocr_count,
                'directory': str(directory_path),
                'media_indexed': media_count if indexed_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {e}")
            return {'error': str(e)}
    
    def search_files(self, query: str, limit: int = 50, type_filter: str = None, 
                     date_start=None, date_end=None, extensions: list = None) -> List[Dict[str, Any]]:
        """
        Search for files using natural language queries.
        
        Args:
            query: Search query (can be natural language)
            limit: Maximum number of results
            type_filter: Filter by file type (e.g., 'images', 'documents')
            date_start: Filter by date - start of range (datetime)
            date_end: Filter by date - end of range (datetime)
            extensions: List of file extensions to filter by
            
        Returns:
            List of matching files with relevance scores
        """
        try:
            logger.info(f"[SEARCH_FILES] query='{query}', type_filter={type_filter}, date_start={date_start}, date_end={date_end}")
            
            # Check if this is a date-only search (no text query)
            is_date_only = not query.strip() and (date_start or date_end)
            
            # Parse and prepare query
            if query.strip():
                fts_terms, filters, debug_info = self._prepare_query(query)
            else:
                fts_terms, filters, debug_info = [], {}, "Date/filter-only search"
            self.last_debug_info = debug_info

            # Perform keyword search (FTS + LIKE fallback) - fetch more to allow for filtering
            # For date-only searches, fetch more to ensure we get all files
            if is_date_only:
                fetch_limit = 500  # Fetch many files for date-only filtering
                logger.info(f"[SEARCH_FILES] Date-only search, fetching up to {fetch_limit} files")
            else:
                fetch_limit = limit * 3 if (type_filter or date_start or extensions) else limit
            results = self.index.search_files_advanced(fts_terms, filters, fetch_limit)

            # Semantic search (local) or GPT rerank - skip for date-only searches
            sem_results: List[Dict[str, Any]] = []
            if not is_date_only:
                try:
                    if settings.use_openai_search_rerank and settings.openai_api_key:
                        sem_results = self._gpt_rerank_results(query, results[: min(20, len(results))])
                    else:
                        # Build a semantic query that includes name/label/tags/caption terms
                        qtext = query
                        if filters.get('label'):
                            qtext += f" {filters['label']}"
                        if filters.get('tags'):
                            qtext += " " + " ".join(filters['tags'])
                        qvec = embed_text(qtext)
                        if qvec:
                            # simple in-Python cosine over all embeddings
                            import math
                            embs = self.index.get_all_embeddings()
                            scored: List[tuple[float, int]] = []
                            qnorm = math.sqrt(sum(x*x for x in qvec)) or 1.0
                            for e in embs:
                                vec = e.get('vector') or []
                                if not vec or len(vec) != len(qvec):
                                    continue
                                dot = sum(a*b for a,b in zip(qvec, vec))
                                vnorm = math.sqrt(sum(x*x for x in vec)) or 1.0
                                cos = dot/(qnorm*vnorm)
                                scored.append((cos, e['file_id']))
                            scored.sort(reverse=True)
                            top_ids = [fid for _, fid in scored[:limit]]
                            sem_results = self.index.get_files_by_ids(top_ids)
                            # attach semantic score as rank
                            for (cos, fid) in scored[:limit]:
                                for r in sem_results:
                                    if r['id'] == fid:
                                        r['rank'] = cos*10
                except Exception:
                    pass

            # Merge keyword and semantic (simple union with max rank)
            by_id: Dict[int, Dict[str, Any]] = {}
            for r in results + sem_results:
                rid = r['id']
                if rid not in by_id:
                    by_id[rid] = r
                else:
                    by_id[rid]['rank'] = max(by_id[rid].get('rank',0), r.get('rank',0))
            merged = list(by_id.values())
            merged.sort(key=lambda x: x.get('rank',0), reverse=True)
            
            # Apply type filter (by file extension)
            if extensions:
                filtered = []
                for r in merged:
                    file_path = r.get('file_path', '') or r.get('name', '')
                    file_ext = Path(file_path).suffix.lower() if file_path else ''
                    if file_ext in extensions:
                        filtered.append(r)
                merged = filtered
            
            # Apply date filter - prioritize original_date (EXIF), then modified_date, then created_date
            if date_start or date_end:
                from datetime import datetime
                logger.info(f"Date filter active: start={date_start}, end={date_end}")
                filtered = []
                for r in merged:
                    # Priority: original_date (EXIF) > modified_date > created_date
                    # original_date is the actual creation date from EXIF metadata (for images)
                    # modified_date is more reliable than created_date on Windows (preserved on copy)
                    file_date_str = r.get('original_date') or r.get('modified_date') or r.get('created_date')
                    logger.debug(f"File: {r.get('file_name')}, original_date={r.get('original_date')}, modified={r.get('modified_date')}")
                    if file_date_str:
                        try:
                            # Parse ISO format date
                            file_date = datetime.fromisoformat(file_date_str.replace('Z', '+00:00'))
                            file_date = file_date.replace(tzinfo=None)
                            
                            # Check if within range
                            in_range = True
                            if date_start and file_date < date_start:
                                logger.debug(f"  Excluded: {file_date} < {date_start}")
                                in_range = False
                            if date_end and file_date > date_end:
                                logger.debug(f"  Excluded: {file_date} > {date_end}")
                                in_range = False
                            if in_range:
                                filtered.append(r)
                        except Exception as e:
                            logger.warning(f"Date parsing failed for {r.get('file_name')}: {e}")
                            # If date parsing fails, include the result
                            filtered.append(r)
                    else:
                        logger.debug(f"  No date info for {r.get('file_name')}, including by default")
                        # No date info, include by default
                        filtered.append(r)
                logger.info(f"Date filter: {len(merged)} -> {len(filtered)} results")
                merged = filtered
            
            merged = merged[:limit]
            
            # Enhance results with additional information
            enhanced_results = []
            for result in merged:
                enhanced_result = self._enhance_search_result(result)
                enhanced_results.append(enhanced_result)
            
            logger.info(f"Search for '{query}' returned {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return []
    
    def search_by_category(self, category: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for files by category.
        
        Args:
            category: Category to search for
            limit: Maximum number of results
            
        Returns:
            List of files in the specified category
        """
        try:
            # Use category as search query
            results = self.index.search_files(f"category:{category}", limit)
            
            enhanced_results = []
            for result in results:
                enhanced_result = self._enhance_search_result(result)
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error searching by category {category}: {e}")
            return []
    
    def search_by_date_range(self, start_date: str, end_date: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for files modified within a date range.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of results
            
        Returns:
            List of files modified in the date range
        """
        try:
            # This would require additional database queries
            # For now, return all files and filter in Python
            all_files = self.index.search_files("", limit=1000)
            
            filtered_files = []
            for file_data in all_files:
                modified_date = file_data.get('modified_date', '')
                if start_date <= modified_date <= end_date:
                    enhanced_result = self._enhance_search_result(file_data)
                    filtered_files.append(enhanced_result)
            
            return filtered_files[:limit]
            
        except Exception as e:
            logger.error(f"Error searching by date range: {e}")
            return []
    
    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of search suggestions
        """
        try:
            # Get search history
            history = self.index.get_search_history(limit=50)
            
            suggestions = []
            for entry in history:
                query = entry['query']
                if partial_query.lower() in query.lower():
                    suggestions.append(query)
            
            # Remove duplicates and limit results
            unique_suggestions = list(dict.fromkeys(suggestions))
            return unique_suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []
    
    def get_file_details(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detailed file information or None if not found
        """
        try:
            file_data = self.index.get_file_by_path(file_path)
            if file_data:
                return self._enhance_search_result(file_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting file details for {file_path}: {e}")
            return None
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed files.
        
        Returns:
            Dictionary with index statistics
        """
        return self.index.get_statistics()
    
    def _prepare_query(self, query: str) -> Tuple[List[str], Dict[str, Any], str]:
        """Parse query into FTS terms and filters.
        Supports operators: type:<label>, label:<label>, tag:<text>, has:ocr, has:vision.
        Returns (fts_terms, filters, debug_info).
        """
        original = query
        q = re.sub(r"\s+", " ", (query or "").strip())
        tokens = q.split()
        fts_terms: List[str] = []
        filters: Dict[str, Any] = {}
        for tok in tokens:
            t = tok.lower()
            if t.startswith("type:") or t.startswith("label:"):
                filters["label"] = tok.split(":", 1)[1]
            elif t.startswith("tag:"):
                filters.setdefault("tags", []).append(tok.split(":", 1)[1])
            elif t == "has:ocr":
                filters["has_ocr"] = True
            elif t == "has:vision":
                filters["has_vision"] = True
            else:
                fts_terms.append(tok)
        debug_info = f"fts_terms={fts_terms} filters={filters} original='{original}'"
        return fts_terms, filters, debug_info
    
    def _enhance_search_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance search result with additional information.
        
        Args:
            result: Raw search result
            
        Returns:
            Enhanced search result
        """
        enhanced = result.copy()
        
        # Add file path object
        file_path = Path(result.get('file_path', ''))
        enhanced['file_path_obj'] = file_path
        
        # Add file existence status
        enhanced['exists'] = file_path.exists()
        
        # Add file size in human-readable format
        size = result.get('file_size', 0)
        enhanced['size_formatted'] = self._format_file_size(size)
        
        # Add OCR text preview
        ocr_text = result.get('ocr_text', '')
        if ocr_text:
            enhanced['ocr_preview'] = ocr_text[:200] + '...' if len(ocr_text) > 200 else ocr_text
        else:
            enhanced['ocr_preview'] = None
        
        # Add relevance score
        rank = result.get('rank', 0)
        enhanced['relevance_score'] = min(rank / 10.0, 1.0) if rank > 0 else 0.0
        
        return enhanced

    def _gpt_rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank a small candidate set via the Supabase OpenAI proxy."""
        from .vision import _call_openai_proxy
        import json as _json
        items = []
        for c in candidates:
            items.append({
                "id": c.get('id'),
                "name": c.get('file_name'),
                "label": c.get('label'),
                "tags": c.get('tags'),
                "caption": (c.get('caption') or '')[:300],
                "ocr": (c.get('ocr_text') or '')[:200]
            })
        system = (
            "You are a reranker. Given a user query and a list of items (id, name, label, tags, caption, ocr), "
            "return a JSON array of item ids sorted from best to worst match. JSON only."
        )
        user = [{"type": "text", "text": f"Query: {query}\nItems: {_json.dumps(items)}\nReturn: [ids in best->worst order]"}]
        try:
            resp_data = _call_openai_proxy(
                "chat",
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            if not resp_data:
                return []
            choices = resp_data.get("choices", [])
            if not choices:
                return []
            content = choices[0].get("message", {}).get("content", "") or ""
            s = content.find('['); e = content.rfind(']')
            if s != -1 and e != -1 and e > s:
                import json
                order = json.loads(content[s:e+1])
                id_to_item = {c['id']: c for c in candidates}
                ranked = [id_to_item[i] for i in order if i in id_to_item]
                # assign a simple rank boost for UI sorting
                boost = len(ranked)
                for r in ranked:
                    boost -= 1
                    r['rank'] = 10 + boost
                return ranked
        except Exception:
            return []
        return []
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"


# Global search service instance
search_service = SearchService()
