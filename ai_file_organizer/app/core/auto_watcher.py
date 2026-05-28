"""
Auto-organize watcher module.

Watches folders for new files and automatically organizes them using AI.
Supports per-folder instructions and catch-up mode for files added while app was closed.
"""

import os
import shutil
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import defaultdict

from PySide6.QtCore import QObject, Signal, QTimer, QThread

logger = logging.getLogger(__name__)


class AutoWatcherWorker(QThread):
    """
    Background worker thread for auto-watcher file processing.
    Runs AI indexing and organization off the main thread to prevent UI lag.
    """
    
    # Signals to communicate back to main thread
    file_indexed = Signal(str)  # file_path that was indexed
    file_organized = Signal(str, str, str)  # source_path, dest_path, category
    status_changed = Signal(str)  # status message
    error_occurred = Signal(str, str)  # file_path, error_message
    finished_processing = Signal(list)  # Emitted when done, with list of all processed file paths
    limit_reached = Signal(dict)  # emitted once when the monthly index limit is hit
    
    def __init__(self, file_paths: List[str], folder: str, instruction: str, 
                 folder_instructions: Dict[str, str] = None,
                 existing_folders: List[str] = None):
        super().__init__()
        self.file_paths = file_paths
        self.folder = folder
        self.instruction = instruction
        self.folder_instructions = folder_instructions or {}
        self.existing_folders = existing_folders  # If set, ONLY use these folders (Organize As-Is mode)
        self._should_stop = False
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True
    
    def _fuzzy_match_folder(self, ai_folder: str) -> Optional[str]:
        """
        Fuzzy match an AI-suggested folder name to an existing folder.
        
        Returns the best matching existing folder name, or None if no good match.
        """
        if not self.existing_folders:
            return ai_folder  # No restriction, use as-is
        
        ai_lower = ai_folder.lower().strip()
        
        # First, try exact match (case insensitive)
        for existing in self.existing_folders:
            if existing.lower() == ai_lower:
                return existing
        
        # Try matching with common variations removed
        def normalize(s):
            return s.lower().replace(' ', '').replace('-', '').replace('_', '')
        
        ai_norm = normalize(ai_folder)
        for existing in self.existing_folders:
            if normalize(existing) == ai_norm:
                return existing
        
        # Try substring match (e.g., "documents 1" contains "document")
        for existing in self.existing_folders:
            if ai_lower in existing.lower() or existing.lower() in ai_lower:
                return existing
        
        # No good match found
        return None

    def _best_existing_folder_for_name(self, ai_folder: str) -> str:
        """Pick the best existing folder to absorb files the AI assigned to a
        non-existing folder. Prefers a catch-all folder, otherwise the existing
        folder whose name is most similar to the AI's suggested name.
        """
        import difflib
        # Prefer a catch-all style folder if one exists
        catchall_keywords = ['everything', 'other', 'misc', 'general', 'rest', 'else']
        for f in self.existing_folders or []:
            if any(k in f.lower() for k in catchall_keywords):
                return f
        # Otherwise pick the most similar existing folder name
        ai_lower = ai_folder.lower().strip()
        if not self.existing_folders:
            return ai_folder  # safety net; shouldn't reach here when called from As-Is mode
        best_folder = self.existing_folders[0]
        best_score = 0.0
        for f in self.existing_folders:
            score = difflib.SequenceMatcher(None, ai_lower, f.lower()).ratio()
            if score > best_score:
                best_score = score
                best_folder = f
        return best_folder

    def run(self):
        """Process files in background thread."""
        # Track all files we process (for marking as organized)
        all_processed_files = []
        
        if not self.file_paths:
            self.finished_processing.emit(all_processed_files)
            return
        
        from app.core.database import file_index
        from app.core.ai_organizer import request_organization_plan, validate_plan, deduplicate_plan, ensure_all_files_included
        from app.core.search import SearchService
        
        # First pass: identify files that need indexing
        # Simple logic: if a file (by filename) already has tags, don't re-index
        # Get all filenames that have tags upfront for fast lookup
        filenames_with_tags = file_index.get_filenames_with_tags()
        logger.info(f"[Worker] Found {len(filenames_with_tags)} files with tags in database")
        
        files_to_index = []
        files_skipped = 0
        
        for file_path in self.file_paths:
            if self._should_stop:
                self.finished_processing.emit(all_processed_files)
                return
            
            file_name = os.path.basename(file_path)
            
            # Simple check: does this filename have tags?
            if file_name in filenames_with_tags:
                # File already has tags - skip indexing
                files_skipped += 1
                logger.debug(f"[Worker] Skipping {file_name} - already has tags")
            else:
                # File doesn't have tags - needs indexing
                files_to_index.append(file_path)
        
        if files_skipped > 0:
            logger.info(f"[Worker] Skipped {files_skipped} file(s) that already have tags")
        
        # Index unindexed files first
        if files_to_index:
            logger.info(f"[Worker] Auto-indexing {len(files_to_index)} unindexed file(s)...")
            self.status_changed.emit(f"Indexing {len(files_to_index)} new file(s)...")
            
            search_service = SearchService()
            indexed_count = 0
            
            # Index files in PARALLEL (mirrors the manual-indexing path), so
            # when several new files appear at once the vision API calls run
            # concurrently instead of one-by-one. Cap by MAX_CONCURRENT_AI_REQUESTS.
            from app.core.search import MAX_CONCURRENT_AI_REQUESTS
            workers = max(1, min(MAX_CONCURRENT_AI_REQUESTS, len(files_to_index)))
            total = len(files_to_index)
            limit_reached = False
            limit_signal_emitted = False

            def _index_one(fp: str):
                try:
                    return fp, search_service.index_single_file(Path(fp), force_ai=False)
                except Exception as exc:
                    return fp, {'error': str(exc)}

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_index_one, fp): fp for fp in files_to_index}
                for fut in as_completed(futures):
                    if self._should_stop:
                        for f in futures:
                            f.cancel()
                        self.finished_processing.emit(all_processed_files)
                        return
                    try:
                        file_path, result = fut.result()
                    except Exception as e:
                        logger.error(f"[Worker] Indexing future failed: {e}")
                        continue

                    if result.get('success'):
                        indexed_count += 1
                        logger.info(f"[Worker] Auto-indexed: {os.path.basename(file_path)}")
                        self.file_indexed.emit(file_path)
                        self.status_changed.emit(f"Indexed {indexed_count}/{total} files...")
                    elif result.get('limit_reached'):
                        # Index limit hit — stop submitting more work, but let
                        # in-flight tasks finish naturally; emit the upgrade
                        # signal once per run.
                        limit_reached = True
                        logger.warning(f"[Worker] Index limit reached: {result.get('error')}")
                        self.status_changed.emit("Index limit reached - upgrade for more")
                        self.error_occurred.emit(file_path, result.get('error', 'Index limit reached'))
                        if not limit_signal_emitted:
                            limit_signal_emitted = True
                            self.limit_reached.emit({})
                        for f in futures:
                            f.cancel()
                        break
                    elif result.get('error'):
                        logger.warning(f"[Worker] Failed to index {file_path}: {result.get('error')}")
            
            if indexed_count > 0:
                self.status_changed.emit(f"Indexed {indexed_count} file(s), now organizing...")
                logger.info(f"[Worker] Auto-indexed {indexed_count} files successfully")
        
        if self._should_stop:
            self.finished_processing.emit(all_processed_files)
            return
        
        # Build file info for AI using SIMPLE SEQUENTIAL IDs
        files_info = []
        files_by_id = {}
        
        for idx, file_path in enumerate(self.file_paths, start=1):
            if self._should_stop:
                self.finished_processing.emit(all_processed_files)
                return
                
            try:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                file_id = idx
                
                indexed_info = file_index.get_file_by_path(file_path)
                
                files_info.append({
                    'id': file_id,
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_size': file_size,
                    'label': indexed_info.get('label') if indexed_info else None,
                    'caption': indexed_info.get('caption') if indexed_info else None,
                    'tags': indexed_info.get('tags', []) if indexed_info else [],
                    'category': indexed_info.get('category') if indexed_info else None,
                })
                files_by_id[file_id] = file_path
                
            except Exception as e:
                logger.warning(f"[Worker] Error getting file info for {file_path}: {e}")
        
        if not files_info:
            # All files were either processed or skipped
            all_processed_files = list(self.file_paths)
            self.finished_processing.emit(all_processed_files)
            return
        
        # Request organization plan from AI
        logger.info(f"[Worker] Requesting AI plan for {len(files_info)} files with instruction: {self.instruction[:50]}...")
        self.status_changed.emit(f"AI analyzing {len(files_info)} files...")
        
        try:
            plan = request_organization_plan(self.instruction, files_info)
            
            if not plan:
                logger.warning("[Worker] No plan returned from AI")
                all_processed_files = list(self.file_paths)
                self.finished_processing.emit(all_processed_files)
                return
            
            plan = deduplicate_plan(plan)
            
            # Safety net: route any file the AI omitted (not preserved) to the
            # most relevant EXISTING folder via _best_folder_for_file (substring
            # + difflib against folder names using each file's name/ext/label/
            # category/tags, with catch-all preference when no strong subject
            # match exists). Applies in BOTH modes — in As-Is the plan only
            # contains existing folders so "most relevant" picks among those.
            plan = ensure_all_files_included(plan, {f['id'] for f in files_info}, files_info)
            
            is_valid, errors = validate_plan(plan, {f['id'] for f in files_info})
            if not is_valid:
                logger.warning(f"[Worker] Plan validation failed: {errors}")
                all_processed_files = list(self.file_paths)
                self.finished_processing.emit(all_processed_files)
                return
            
            # Filter plan to only include files we're processing
            # Also apply fuzzy matching for folder names when in "Organize As-Is" mode
            valid_file_ids = set(files_by_id.keys())
            filtered_folders = {}
            # (Unmatched folders are no longer skipped; they're redirected via _best_existing_folder_for_name)
            
            for folder_name, file_ids in plan.get('folders', {}).items():
                valid_ids = [fid for fid in file_ids if fid in valid_file_ids]
                if not valid_ids:
                    continue
                
                # Apply fuzzy matching when in "Organize As-Is" mode
                if self.existing_folders:
                    matched_folder = self._fuzzy_match_folder(folder_name)
                    if matched_folder:
                        # Map to existing folder
                        if matched_folder != folder_name:
                            logger.info(f"[Worker] Mapped AI folder '{folder_name}' -> existing '{matched_folder}'")
                        if matched_folder in filtered_folders:
                            filtered_folders[matched_folder].extend(valid_ids)
                        else:
                            filtered_folders[matched_folder] = valid_ids
                    else:
                        # No fuzzy match — redirect to the best existing folder
                        # rather than leaving files loose. Prefer a catch-all
                        # style folder; otherwise pick the existing folder
                        # whose name is most similar to the AI's suggestion.
                        fallback = self._best_existing_folder_for_name(folder_name)
                        logger.info(f"[Worker] No match for '{folder_name}' - redirecting {len(valid_ids)} file(s) to '{fallback}'")
                        if fallback in filtered_folders:
                            filtered_folders[fallback].extend(valid_ids)
                        else:
                            filtered_folders[fallback] = list(valid_ids)
                else:
                    # Normal mode - use folder as-is
                    filtered_folders[folder_name] = valid_ids
            
            logger.info(f"[Worker] Filtered plan has {sum(len(ids) for ids in filtered_folders.values())} files in {len(filtered_folders)} folders")
            
            # Execute moves
            moved_count = 0
            skipped_count = 0
            
            for folder_name, file_ids in filtered_folders.items():
                for file_id in file_ids:
                    if self._should_stop:
                        self.finished_processing.emit(all_processed_files)
                        return
                        
                    file_path = files_by_id.get(file_id)
                    if not file_path or not os.path.exists(file_path):
                        continue
                    
                    dest_folder = os.path.join(self.folder, folder_name)
                    dest_path = os.path.join(dest_folder, os.path.basename(file_path))
                    
                    # CRITICAL: Check if file is ALREADY in the correct location
                    # This prevents the _1 suffix bug and unnecessary moves
                    source_normalized = os.path.normpath(file_path)
                    dest_normalized = os.path.normpath(dest_path)
                    
                    if source_normalized == dest_normalized:
                        # File is already exactly where it should be - skip move but track as processed
                        skipped_count += 1
                        all_processed_files.append(file_path)
                        logger.debug(f"[Worker] File already in place, skipping: {os.path.basename(file_path)}")
                        continue
                    
                    # Check if file is already in the destination folder (even if paths differ slightly)
                    source_folder = os.path.normpath(os.path.dirname(file_path))
                    dest_folder_normalized = os.path.normpath(dest_folder)
                    
                    if source_folder == dest_folder_normalized:
                        # File is already in the correct folder - skip move but track as processed
                        skipped_count += 1
                        all_processed_files.append(file_path)
                        logger.debug(f"[Worker] File already in correct folder, skipping: {os.path.basename(file_path)}")
                        continue
                    
                    # Create destination folder only if we're actually going to move something
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    # Handle duplicates - only if dest_path is a DIFFERENT file
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(os.path.basename(file_path))
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(dest_folder, f"{base}_{counter}{ext}")
                            counter += 1
                    
                    try:
                        shutil.move(file_path, dest_path)
                        moved_count += 1
                        all_processed_files.append(dest_path)  # Track new location
                        logger.info(f"[Worker] Organized: {file_path} -> {dest_path}")
                        self.file_organized.emit(file_path, dest_path, folder_name)
                        
                        # Update database path
                        try:
                            # Find the file in database by old path or filename
                            old_record = file_index.get_file_by_path(file_path)
                            if not old_record:
                                # Try by filename
                                old_record = file_index.get_file_by_name(os.path.basename(file_path))
                            
                            if old_record:
                                file_index.update_file_path(old_record['id'], dest_path)
                                logger.info(f"[Worker] Updated DB path for file {old_record['id']}")
                            else:
                                logger.warning(f"[Worker] Could not find file in DB to update path: {file_path}")
                        except Exception as e:
                            logger.warning(f"[Worker] Could not update index for moved file: {e}")
                            
                    except Exception as e:
                        logger.error(f"[Worker] Error moving {file_path}: {e}")
                        self.error_occurred.emit(file_path, str(e))
            
            if skipped_count > 0:
                logger.info(f"[Worker] Skipped {skipped_count} file(s) already in correct location")
            
            if moved_count > 0:
                self.status_changed.emit(f"Organized {moved_count} file(s)")
            else:
                self.status_changed.emit(f"All files already organized")
            
        except Exception as e:
            logger.error(f"[Worker] Error in organization: {e}")
            self.error_occurred.emit("", str(e))
        
        # Mark all files as processed (including those that weren't moved)
        if not all_processed_files:
            all_processed_files = list(self.file_paths)
        
        self.finished_processing.emit(all_processed_files)


class AutoOrganizeWatcher(QObject):
    """
    Watches folders for new files and auto-organizes them using AI.
    
    Features:
    - Per-folder instructions
    - Catch-up mode (organize files added while app was closed)
    - Flatten and re-organize existing folders
    - Background file system monitoring
    """
    
    # Signals
    file_organized = Signal(str, str, str)  # source_path, dest_path, category
    file_indexed = Signal(str)  # file_path that was auto-indexed
    error_occurred = Signal(str, str)  # file_path, error_message
    status_changed = Signal(str)  # status message
    limit_reached = Signal(dict)  # monthly index limit hit during auto-organize

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Watched folders list
        self.watched_folders: List[str] = []
        
        # Per-folder instructions: {folder_path: instruction_text}
        self.folder_instructions: Dict[str, str] = {}
        
        # For catch-up mode: only organize files modified after this time
        self.catch_up_since: Optional[datetime] = None
        
        # Internal state
        self._is_running = False
        self._pending_files: Dict[str, float] = {}  # path -> first_seen_time
        self._processed_files: Set[str] = set()
        self._file_check_timer: Optional[QTimer] = None
        self._debounce_seconds = 2.0  # Wait for file to stabilize
        
        # Counter for periodic cleanup (every N checks)
        self._check_count = 0
        self._cleanup_interval = 10  # Run cleanup every 10 checks (~30 seconds)
        
        # Files to ignore (system files, temp files, etc.)
        self._ignore_patterns = {
            '.DS_Store', 'Thumbs.db', 'desktop.ini', '.git', '.gitignore',
            '__pycache__', '.pyc', '.pyo', '.tmp', '.temp', '.swp', '.bak',
            '~$'  # Office temp files
        }
        self._ignore_extensions = {
            '.tmp', '.temp', '.crdownload', '.part', '.partial'
        }
        
        # Background worker for file processing (prevents UI lag)
        self._current_worker: Optional[AutoWatcherWorker] = None
        # Queue stores (folder, instruction, existing_folders) - we re-scan folder when processing
        self._worker_queue: List[tuple] = []
        
        # Track files that have been successfully organized
        # This prevents re-processing the same files and unnecessary AI calls
        self._organized_files: Set[str] = set()  # normalized paths of organized files

        # --- Organize New Only (action = 3) state ---
        # Per-folder set of SHA-256 hashes captured at start() time. Files
        # whose hash is in this set were present when watching began and must
        # be left untouched for the rest of the session — even if the user
        # later moves or renames them.
        self._baseline_hashes: Dict[str, Set[str]] = {}
        # Normalized lowercased paths the watcher has explicitly decided to
        # leave in place for this session. Checked at the top of the periodic
        # scan loop so we never re-hash or re-evaluate the same pre-existing
        # files again.
        self._left_in_place_paths: Set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._is_running
    
    def add_folder(self, folder_path: str) -> bool:
        """Add a folder to watch. Returns True if successful."""
        # Normalize path for consistent lookups
        folder_path = os.path.normpath(folder_path)
        
        if not os.path.isdir(folder_path):
            logger.warning(f"Not a valid directory: {folder_path}")
            return False
        
        if folder_path not in self.watched_folders:
            self.watched_folders.append(folder_path)
            logger.info(f"Added watch folder: {folder_path}")
        return True
    
    def remove_folder(self, folder_path: str) -> None:
        """Remove a folder from watch list."""
        folder_path = os.path.normpath(folder_path)
        if folder_path in self.watched_folders:
            self.watched_folders.remove(folder_path)
            logger.info(f"Removed watch folder: {folder_path}")
    
    def clear_folders(self) -> None:
        """Remove all watched folders."""
        self.watched_folders.clear()
        self.folder_instructions.clear()
        logger.info("Cleared all watch folders")
    
    def set_instruction(self, folder_path: str, instruction: str) -> None:
        """Set the organization instruction for a specific folder."""
        folder_path = os.path.normpath(folder_path)
        self.folder_instructions[folder_path] = instruction
        logger.info(f"Set instruction for {folder_path}: {instruction[:50]}...")
    
    def start(self, organize_existing: bool = True, flatten_first: bool = False) -> None:
        """
        Start watching folders.
        
        Args:
            organize_existing: If True, organize files already in the folders
            flatten_first: If True, flatten folder structure before organizing
        """
        if self._is_running:
            logger.warning("Watcher already running")
            return
        
        if not self.watched_folders:
            logger.warning("No folders to watch")
            self.status_changed.emit("No folders configured to watch")
            return
        
        self._is_running = True
        self._processed_files.clear()
        self._pending_files.clear()
        # Reset Organize-New-Only state every start() so the baseline reflects
        # the folder contents at THIS moment.
        self._baseline_hashes.clear()
        self._left_in_place_paths.clear()

        # Capture the Organize-New-Only baseline FIRST — before any flatten or
        # organize-existing pass — so files the app itself moves don't get
        # mistaken for "new" arrivals when the periodic scan starts.
        for folder in self.watched_folders:
            self._capture_baseline(os.path.normpath(folder))

        # Snapshot every file currently in the watched folders (root AND
        # subfolders). Files in this snapshot are treated as "already known"
        # so the periodic checker only fires on files added AFTER start.
        # Pre-existing files are either already organized (sitting in their
        # subfolders) or were left in place intentionally — either way the
        # watcher shouldn't keep re-processing them every cycle.
        for folder in self.watched_folders:
            folder = os.path.normpath(folder)
            if not os.path.isdir(folder):
                continue
            for root, dirs, files in os.walk(folder):
                # Skip hidden subdirs in-place
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fn in files:
                    if self._should_ignore(fn):
                        continue
                    self._processed_files.add(os.path.normpath(os.path.join(root, fn)))

        folder_count = len(self.watched_folders)
        self.status_changed.emit(f"Starting watch on {folder_count} folder(s)...")
        logger.info(f"Starting watcher for {folder_count} folders (snapshot has {len(self._processed_files)} pre-existing files)")
        
        # Flatten folders first if requested (for re-organize)
        if flatten_first:
            total_flattened = 0
            for folder in self.watched_folders:
                count = self.flatten_folder(folder, instruction=self.folder_instructions.get(folder, ''))
                total_flattened += count
            if total_flattened > 0:
                self.status_changed.emit(f"Flattened {total_flattened} files from subfolders")
        
        # Organize existing files if requested
        if organize_existing:
            self._organize_existing_files()
        
        # Start periodic file check
        self._file_check_timer = QTimer(self)
        self._file_check_timer.timeout.connect(self._check_for_new_files)
        self._file_check_timer.start(3000)  # Check every 3 seconds
        
        self.status_changed.emit(f"Watching {folder_count} folder(s) for new files...")
    
    def stop(self) -> None:
        """Stop watching folders."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._file_check_timer:
            self._file_check_timer.stop()
            self._file_check_timer = None
        
        # Stop any running worker
        if self._current_worker is not None and self._current_worker.isRunning():
            logger.info("Stopping background worker...")
            self._current_worker.stop()
            self._current_worker.wait(3000)  # Wait up to 3 seconds
        
        # Clear queues
        self._pending_files.clear()
        self._worker_queue.clear()
        
        self.status_changed.emit("Watcher stopped")
        logger.info("Watcher stopped")
    
    def flatten_folder(self, folder_path: str, instruction: str = None) -> int:
        """
        Flatten folder by moving all files from subfolders to root.

        This is used for the "Re-organize All" feature to reset folder structure
        before applying new organization instructions.

        Args:
            folder_path: Path to the folder to flatten
            instruction: Optional user instruction. When provided, folders the
                user explicitly named ("called X" / "named X") are auto-created
                up front and protected from the empty-folder cleanup that runs
                at the end of flattening.

        Returns:
            Number of files moved
        """
        folder_path = os.path.normpath(folder_path)
        if not os.path.isdir(folder_path):
            logger.warning(f"Cannot flatten - not a directory: {folder_path}")
            return 0

        # Auto-create user-named folders up front so they exist as destinations
        # for the AI step and survive the empty-folder cleanup below.
        protected: set = set()
        if instruction:
            named = self._extract_named_folders(instruction)
            if named:
                self._ensure_named_folders_exist(folder_path, instruction)
                protected = {n.lower() for n in named}

        moved_count = 0
        files_to_move = []
        
        # Collect files from subfolders (not the root level)
        for root, dirs, files in os.walk(folder_path):
            if root == folder_path:
                continue  # Skip root level files
            
            for file_name in files:
                if self._should_ignore(file_name):
                    continue
                files_to_move.append(os.path.join(root, file_name))
        
        if not files_to_move:
            logger.info(f"No files to flatten in {folder_path}")
            return 0
        
        self.status_changed.emit(f"Flattening {len(files_to_move)} files...")
        
        # Move files to root, handling name conflicts
        for file_path in files_to_move:
            try:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(folder_path, file_name)
                
                # Handle name conflicts by adding (1), (2), etc.
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(folder_path, f"{base} ({counter}){ext}")
                        counter += 1
                
                shutil.move(file_path, dest_path)
                moved_count += 1
                logger.info(f"Flattened: {file_path} -> {dest_path}")
                
            except Exception as e:
                logger.error(f"Error flattening {file_path}: {e}")
                self.error_occurred.emit(file_path, str(e))
        
        # Clean up empty subdirectories, preserving any folders the user
        # explicitly named in their instruction.
        self._cleanup_empty_folders(folder_path, protected_names=protected)

        logger.info(f"Flattened {moved_count} files in {folder_path}")
        return moved_count
    
    def _extract_named_folders(self, instruction: str) -> List[str]:
        """Extract folder names the user explicitly named in their instruction.

        Relies on the reliable "called X" / "named X" phrasing. The capture
        stops at conjunctions/prepositions (and, except, where, to, ...) so
        names like "everything else" are kept whole without bleeding into the
        rest of the sentence.
        """
        if not instruction:
            return []
        import re
        names = []
        seen = set()
        stopwords = {'a', 'an', 'the', 'it', 'them', 'this', 'that'}

        pattern = (
            r'(?:called|named)\s+["\']?([A-Za-z0-9][\w \-]*?)["\']?'
            r'(?=\s+(?:and|except|where|to|in|on|into|onto|or|but|then)\b|["\',.\n]|$)'
        )
        for m in re.finditer(pattern, instruction, flags=re.IGNORECASE):
            name = m.group(1).strip().strip('"\'').strip()
            key = name.lower()
            if not name or key in seen or key in stopwords or len(name) > 60:
                continue
            if 'folder' in key or ' move ' in key:
                continue
            seen.add(key)
            names.append(name)
        return names

    def _ensure_named_folders_exist(self, folder_path: str, instruction: str) -> None:
        """Create folders the user named in their instruction if they don't exist.

        Uses a case-insensitive check against current subfolders so existing
        folders are never duplicated.
        """
        named = self._extract_named_folders(instruction)
        if not named:
            return
        try:
            existing_lower = {
                item.lower() for item in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, item))
            }
        except OSError:
            existing_lower = set()
        for name in named:
            if name.lower() in existing_lower:
                continue
            try:
                os.makedirs(os.path.join(folder_path, name), exist_ok=True)
                logger.info(f"Created user-specified folder: {name}")
                existing_lower.add(name.lower())
            except OSError as e:
                logger.warning(f"Could not create folder '{name}': {e}")

    def _cleanup_empty_folders(self, root_folder: str, protected_names: set = None) -> int:
        """Remove empty subdirectories. Returns count of removed folders.

        Folders whose name is in ``protected_names`` (case-insensitive) are
        never deleted, even when empty — typically the folders the user
        explicitly named in their instruction.
        """
        removed_count = 0
        root_folder = os.path.normpath(root_folder)
        protected = {n.lower() for n in (protected_names or set())}

        # Walk bottom-up to remove nested empty folders
        for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
            # Skip the root folder itself (normalize for comparison)
            if os.path.normpath(dirpath) == root_folder:
                continue

            # Skip hidden folders
            if os.path.basename(dirpath).startswith('.'):
                continue

            # Skip folders the user explicitly named in their instruction
            if os.path.basename(dirpath).lower() in protected:
                continue

            try:
                # Check if folder is empty (no files or subdirs)
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
                    removed_count += 1
                    logger.info(f"Removed empty folder: {dirpath}")
            except OSError as e:
                logger.debug(f"Could not remove folder {dirpath}: {e}")

        return removed_count
    
    def _should_ignore(self, file_path: str) -> bool:
        """Check if a file should be ignored.
        
        Args:
            file_path: Full path or just filename to check
        """
        from app.core.settings import settings
        
        file_name = os.path.basename(file_path)
        
        # Check exact names
        if file_name in self._ignore_patterns:
            return True
        
        # Check patterns
        for pattern in self._ignore_patterns:
            if file_name.startswith(pattern):
                return True
        
        # Check extensions
        _, ext = os.path.splitext(file_name.lower())
        if ext in self._ignore_extensions:
            return True
        
        # Ignore hidden files (starting with .)
        if file_name.startswith('.'):
            return True
        
        # Check user-defined exclusions from settings
        if settings.should_exclude(file_path):
            logger.debug(f"Excluding {file_path} (matches user exclusion pattern)")
            return True
        
        return False
    
    def _get_instruction_for_folder(self, folder_path: str) -> str:
        """Get the CURRENT instruction for a specific folder.

        Reads live from settings (the source of truth, updated immediately
        when the user edits the instruction in the UI) so the watcher and the
        re-organize flow always honor the latest edits — without needing the
        watcher to be restarted or having to push updates through other paths.
        Falls back to the in-memory cache for legacy/test callers that pre-set
        instructions there.
        """
        folder_path = os.path.normpath(folder_path)
        try:
            from app.core.settings import settings
            for f in settings.auto_organize_folders:
                if os.path.normpath(f.get('path', '')) == folder_path:
                    instruction = f.get('instruction', '') or ''
                    logger.debug(f"Instruction (live) for {folder_path}: {instruction[:50] if instruction else '(none)'}")
                    return instruction
        except Exception as e:
            logger.warning(f"Could not read live instruction from settings: {e}")
        # Fallback to the cached dict
        instruction = self.folder_instructions.get(folder_path, '')
        logger.debug(f"Instruction (cache) for {folder_path}: {instruction[:50] if instruction else '(none)'}")
        return instruction

    # Folder names that look like build/system scaffolding rather than user
    # destinations; filtered out of the existing-folders list for As-Is mode.
    _SKIP_DIR_NAMES = {
        '__pycache__', 'node_modules', 'venv', '.venv',
        'dist', 'build', '.cache', '.next', '.nuxt',
    }

    def _collect_existing_folders(self, folder_path: str,
                                  max_depth: int = 3,
                                  max_total: int = 200) -> List[str]:
        """Collect existing subfolders inside ``folder_path`` for As-Is mode.

        Walks up to ``max_depth`` levels deep so nested folders like
        'Photos/Vacation' are also valid destinations — not just the immediate
        children. Hidden folders ('.git', '.idea', …) and obvious build
        artifacts are skipped. Returns paths RELATIVE to ``folder_path``,
        using forward slashes for stable consumption by the AI prompt.

        If the total exceeds ``max_total``, falls back to top-level only —
        deep listings produce noisy AI plans and the user is likely using
        nested folders as scaffolding, not destinations.
        """
        folder_path = os.path.normpath(folder_path)
        if not os.path.isdir(folder_path):
            return []
        out: List[str] = []
        try:
            for root, dirs, files in os.walk(folder_path):
                rel = os.path.relpath(root, folder_path)
                depth = 0 if rel == '.' else rel.count(os.sep) + 1
                # Children of this root are at depth+1. Stop descending past the cap.
                if depth + 1 > max_depth:
                    dirs[:] = []
                    continue
                # Prune hidden + known system/scaffolding folders in-place so we
                # neither include them nor descend into them.
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith('.') and d not in self._SKIP_DIR_NAMES
                ]
                for d in dirs:
                    sub_rel = os.path.relpath(os.path.join(root, d), folder_path)
                    out.append(sub_rel.replace(os.sep, '/'))
        except OSError as e:
            logger.warning(f"Could not walk {folder_path} for existing folders: {e}")
            return []

        if len(out) > max_total:
            logger.info(
                f"Existing-folders walk found {len(out)} folders (cap {max_total}); "
                "falling back to top-level only."
            )
            out = []
            try:
                for item in os.listdir(folder_path):
                    if item.startswith('.') or item in self._SKIP_DIR_NAMES:
                        continue
                    if os.path.isdir(os.path.join(folder_path, item)):
                        out.append(item)
            except OSError:
                pass
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Organize New Only (action = 3): hash-based pre-existing-file detection.
    # The watcher snapshots every file's SHA-256 at start() and uses that
    # baseline to recognize pre-existing files even if they're later moved,
    # renamed, or duplicated. Genuinely new files (no hash in the baseline,
    # not known in the DB) are the only ones that get organized.
    # ─────────────────────────────────────────────────────────────────────

    def _compute_file_hash(self, path: str) -> Optional[str]:
        """SHA-256 of the file's bytes (1 MiB chunks). MUST match the hash
        scheme the indexer uses (search.py uses hashlib.sha256 + 1 MiB) so a
        baseline hash equals the DB-stored content_hash for that file. Returns
        None on read error.
        """
        import hashlib
        try:
            h = hashlib.sha256()
            with open(path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            logger.warning(f"_compute_file_hash failed for {path}: {e}")
            return None

    def _capture_baseline(self, folder: str) -> None:
        """Snapshot the SHA-256 of every file currently in `folder` (recursively)
        — but ONLY when the folder's action is Organize New Only (3). Reuses
        the indexer's stored content_hash whenever possible so large media
        folders don't get re-hashed from disk. Stores the resulting set on
        ``self._baseline_hashes[normpath(folder)]``.
        """
        from app.core.settings import settings
        ORGANIZE_NEW_ONLY = 3
        if settings.get_auto_organize_action(folder) != ORGANIZE_NEW_ONLY:
            return

        try:
            from app.core.database import file_index
        except Exception:
            file_index = None

        folder = os.path.normpath(folder)
        baseline: Set[str] = set()
        hashed_from_disk = 0
        total = 0

        try:
            for root, dirs, files in os.walk(folder):
                # Skip hidden subdirs entirely
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fn in files:
                    if fn.startswith('.') or self._should_ignore(fn):
                        continue
                    path = os.path.join(root, fn)
                    total += 1
                    # Prefer the indexer's stored hash to avoid re-reading
                    # large media files from disk.
                    h: Optional[str] = None
                    if file_index is not None:
                        try:
                            row = file_index.get_file_by_path(path)
                            if row:
                                h = row.get('content_hash')
                        except Exception:
                            h = None
                    if not h:
                        h = self._compute_file_hash(path)
                        if h:
                            hashed_from_disk += 1
                    if h:
                        baseline.add(h)
        except OSError as e:
            logger.warning(f"[OrganizeNewOnly] Walk failed for baseline of {folder}: {e}")
            return

        self._baseline_hashes[folder] = baseline
        logger.info(
            f"[OrganizeNewOnly] Captured baseline of {total} existing file(s) "
            f"for {folder} ({hashed_from_disk} hashed from disk, "
            f"{total - hashed_from_disk} reused from DB)"
        )

    def _filter_genuinely_new_files(self, folder: str, candidate_paths: List[str]) -> List[str]:
        """For an Organize New Only folder: return only the candidate paths
        that are genuinely new (their hash is not in the baseline AND the
        file_index doesn't already know their content). Files that are
        recognized as pre-existing are added to ``_left_in_place_paths`` (and
        ``_processed_files`` / ``_organized_files``) so the watcher never
        re-evaluates them this session; if a pre-existing file's hash is
        already known to the DB at a different path, its DB record's path
        is also updated so the index stays consistent after a move/rename.
        For non-action-3 folders, returns ``candidate_paths`` unchanged.
        """
        from app.core.settings import settings
        ORGANIZE_NEW_ONLY = 3
        folder_n = os.path.normpath(folder)
        if settings.get_auto_organize_action(folder_n) != ORGANIZE_NEW_ONLY:
            return list(candidate_paths)

        baseline = self._baseline_hashes.get(folder_n, set())
        try:
            from app.core.database import file_index
        except Exception:
            file_index = None

        kept: List[str] = []
        for path in candidate_paths:
            try:
                h = self._compute_file_hash(path)
            except Exception:
                kept.append(path)  # be lenient — if we can't hash, treat as new
                continue
            is_baseline = bool(h) and h in baseline
            known = None
            if not is_baseline and h and file_index is not None:
                try:
                    known = file_index.get_file_by_hash(h)
                except Exception:
                    known = None

            np = os.path.normpath(path)
            # Baseline match → pre-existing file. Leave in place.
            if is_baseline:
                logger.info(
                    f"[OrganizeNewOnly] '{os.path.basename(path)}' pre-existing (baseline) - leaving in place"
                )
                self._left_in_place_paths.add(np.lower())
                self._processed_files.add(np)
                self._organized_files.add(np)
                continue

            # DB match. Only treat as pre-existing if the stored path is
            # DIFFERENT from the candidate path — that signals the user
            # moved/renamed a previously-indexed file into the watched
            # folder, which is the case we want to honor (rename detection).
            #
            # If the stored path EQUALS the candidate path, that's almost
            # always a race condition: the watcher just auto-indexed this
            # very file moments ago and the periodic scan came back around
            # to check it. Suppressing it here would lock a genuinely-new
            # file in place forever (seen in the 22:52:02 log for
            # animal-hero-tiger_0.jpg). Treat as new in that case so the
            # AI plan can still route it.
            if known is not None:
                stored = known.get('file_path') or ''
                if stored and os.path.normpath(stored) != np:
                    try:
                        file_index.update_file_path(known['id'], np)
                    except Exception as e:
                        logger.warning(f"[OrganizeNewOnly] Could not update DB path: {e}")
                    logger.info(
                        f"[OrganizeNewOnly] '{os.path.basename(path)}' already known (moved) - leaving in place"
                    )
                    self._left_in_place_paths.add(np.lower())
                    self._processed_files.add(np)
                    self._organized_files.add(np)
                    continue
                # stored == np → same path → just-indexed by this watcher.
                # Fall through to keep this file as a new candidate.

            kept.append(path)
        return kept

    def _get_existing_folders_if_as_is(self, folder_path: str) -> list:
        """Return existing subfolders (nested too) if folder is in ORGANIZE_AS_IS mode, else None."""
        from app.core.settings import settings
        ORGANIZE_AS_IS = 2
        if settings.get_auto_organize_action(folder_path) == ORGANIZE_AS_IS:
            return self._collect_existing_folders(folder_path)
        return None
    
    def _organize_existing_files(self) -> None:
        """Organize files already in the watched folders (including subfolders).

        Folders configured for Organize New Only (action=3) are deliberately
        skipped here — by contract those folders' pre-existing files must
        stay exactly where they are, and only files added AFTER the watcher
        starts get organized (the periodic scan in ``_check_for_new_files``
        handles those via the baseline filter).
        """
        from app.core.settings import settings
        ORGANIZE_NEW_ONLY = 3
        all_files = []

        for folder in self.watched_folders:
            folder = os.path.normpath(folder)
            if not os.path.isdir(folder):
                continue
            if settings.get_auto_organize_action(folder) == ORGANIZE_NEW_ONLY:
                logger.info(
                    f"[OrganizeNewOnly] Skipping existing-files pass for {folder} "
                    "(action=3, pre-existing files stay in place)"
                )
                continue

            # Get ALL files in this folder AND subfolders
            for root, dirs, files in os.walk(folder):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for item in files:
                    if self._should_ignore(item):
                        continue
                    
                    item_path = os.path.join(root, item)
                    
                    # Check catch-up filter
                    if self.catch_up_since:
                        try:
                            mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                            if mtime < self.catch_up_since:
                                continue  # Skip files older than catch-up time
                        except Exception:
                            pass
                    
                    all_files.append((item_path, folder))
        
        if not all_files:
            self.status_changed.emit("No existing files to organize")
            return
        
        self.status_changed.emit(f"Organizing {len(all_files)} existing files...")
        logger.info(f"Found {len(all_files)} existing files to organize")
        
        # Group files by their source folder
        files_by_folder: Dict[str, List[str]] = defaultdict(list)
        for file_path, folder in all_files:
            files_by_folder[folder].append(file_path)
        
        # Process each folder with its instruction
        for folder, files in files_by_folder.items():
            instruction = self._get_instruction_for_folder(folder)
            existing_folders = self._get_existing_folders_if_as_is(folder)
            self._process_files_with_ai(files, folder, instruction, existing_folders)
    
    def _organize_existing_files_with_options(self, flatten_first: bool = False) -> None:
        """
        Organize existing files with options.
        Called when instructions are changed while watching.
        
        Args:
            flatten_first: If True, flatten folder structure before organizing
        """
        # Clear the catch-up filter so ALL files are processed, not just recent ones
        self.catch_up_since = None
        
        if flatten_first:
            total_flattened = 0
            for folder in self.watched_folders:
                count = self.flatten_folder(folder, instruction=self.folder_instructions.get(folder, ''))
                total_flattened += count
            if total_flattened > 0:
                self.status_changed.emit(f"Flattened {total_flattened} files, now organizing...")
        
        # Now organize
        self._organize_existing_files()
    
    def organize_folders_with_per_folder_options(self, folder_choices: Dict[str, int]) -> None:
        """
        Organize folders with per-folder options.
        
        Args:
            folder_choices: Dict mapping folder_path -> choice where choice is:
                1 = REORGANIZE_ALL (flatten + organize)
                2 = ORGANIZE_AS_IS (organize without flatten)
                3 = CONTINUE_WATCHING (skip, just watch for new files)
        """
        # Clear the catch-up filter so ALL files are processed
        self.catch_up_since = None
        
        REORGANIZE_ALL = 1
        ORGANIZE_AS_IS = 2
        CONTINUE_WATCHING = 3
        
        # Separate folders by their chosen action
        folders_to_reorganize = []  # flatten + organize
        folders_to_organize = []     # organize as-is
        
        for folder_path, choice in folder_choices.items():
            folder_path = os.path.normpath(folder_path)
            if not os.path.isdir(folder_path):
                continue
                
            if choice == REORGANIZE_ALL:
                folders_to_reorganize.append(folder_path)
            elif choice == ORGANIZE_AS_IS:
                folders_to_organize.append(folder_path)
            # CONTINUE_WATCHING = do nothing for this folder
        
        # Step 1: Flatten folders that need it
        if folders_to_reorganize:
            total_flattened = 0
            for folder in folders_to_reorganize:
                count = self.flatten_folder(folder, instruction=self.folder_instructions.get(folder, ''))
                total_flattened += count
            if total_flattened > 0:
                self.status_changed.emit(f"Flattened {total_flattened} files from {len(folders_to_reorganize)} folder(s)...")
        
        # Combine all folders that need organizing
        all_folders_to_organize = folders_to_reorganize + folders_to_organize
        
        if not all_folders_to_organize:
            self.status_changed.emit("No folders selected for organizing")
            return
        
        # Step 2: Collect files from selected folders and track existing subfolders for "Organize As-Is"
        all_files = []
        existing_folders_by_parent: Dict[str, List[str]] = {}  # folder -> list of existing subfolders
        
        for folder in folders_to_organize:  # Only for "Organize As-Is" folders
            folder = os.path.normpath(folder)
            existing_subfolders = self._collect_existing_folders(folder)
            existing_folders_by_parent[folder] = existing_subfolders
            logger.info(f"Organize As-Is for {folder}: existing folders = {existing_subfolders}")
        
        for folder in all_folders_to_organize:
            folder = os.path.normpath(folder)
            if not os.path.isdir(folder):
                continue
            
            # Get ALL files in this folder AND subfolders
            for root, dirs, files in os.walk(folder):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for item in files:
                    if self._should_ignore(item):
                        continue
                    
                    item_path = os.path.join(root, item)
                    all_files.append((item_path, folder))
        
        if not all_files:
            self.status_changed.emit("No files to organize in selected folders")
            return
        
        self.status_changed.emit(f"Organizing {len(all_files)} files from {len(all_folders_to_organize)} folder(s)...")
        logger.info(f"Per-folder organize: {len(all_files)} files from {len(all_folders_to_organize)} folders")
        
        # Group files by their source folder
        files_by_folder: Dict[str, List[str]] = defaultdict(list)
        for file_path, folder in all_files:
            files_by_folder[folder].append(file_path)
        
        # Process each folder with its instruction
        for folder, files in files_by_folder.items():
            instruction = self._get_instruction_for_folder(folder)
            # Pass existing folders for "Organize As-Is" folders
            existing_folders = existing_folders_by_parent.get(folder)
            self._process_files_with_ai(files, folder, instruction, existing_folders)
    
    def organize_single_folder(self, folder_path: str, flatten_first: bool = False) -> None:
        """
        Organize a single folder with the given option.
        
        Args:
            folder_path: Path to the folder to organize
            flatten_first: If True, flatten folder structure before organizing.
                          If False (Organize As-Is), only use existing folders.
        """
        folder_path = os.path.normpath(folder_path)
        
        if not os.path.isdir(folder_path):
            logger.warning(f"Cannot organize - not a directory: {folder_path}")
            return
        
        # Clear the catch-up filter so ALL files are processed
        self.catch_up_since = None
        
        logger.info(f"Organizing single folder: {folder_path}, flatten={flatten_first}")
        
        # Step 1: Flatten if requested
        if flatten_first:
            count = self.flatten_folder(folder_path, instruction=self.folder_instructions.get(folder_path, ''))
            if count > 0:
                self.status_changed.emit(f"Flattened {count} files, now organizing...")
        
        # Step 2: Get existing folder structure (for "Organize As-Is" mode).
        # Includes nested subfolders (up to a sensible depth) so the AI can route
        # files directly into Photos/Vacation/ instead of just Photos/.
        existing_folders = []
        if not flatten_first:
            existing_folders = self._collect_existing_folders(folder_path)
            logger.info(f"Organize As-Is mode: existing folders = {existing_folders}")
        
        # Step 3: Collect files from this folder
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for item in files:
                if self._should_ignore(item):
                    continue
                
                item_path = os.path.join(root, item)
                all_files.append(item_path)
        
        if not all_files:
            self.status_changed.emit(f"No files to organize in {os.path.basename(folder_path)}")
            return
        
        self.status_changed.emit(f"Organizing {len(all_files)} files in {os.path.basename(folder_path)}...")
        logger.info(f"Single folder organize: {len(all_files)} files in {folder_path}")
        
        # Process files with AI
        instruction = self._get_instruction_for_folder(folder_path)
        use_existing_only = not flatten_first  # Organize As-Is = only use existing folders
        self._process_files_with_ai(all_files, folder_path, instruction, existing_folders if use_existing_only else None)
    
    def _check_for_new_files(self) -> None:
        """Periodic check for new files in watched folders."""
        if not self._is_running:
            return
        
        current_time = time.time()
        
        # Periodic cleanup of empty folders
        self._check_count += 1
        if self._check_count >= self._cleanup_interval:
            self._check_count = 0
            for folder in self.watched_folders:
                folder = os.path.normpath(folder)
                if os.path.isdir(folder):
                    deleted = self._cleanup_empty_folders(folder)
                    if deleted > 0:
                        logger.info(f"Periodic cleanup: removed {deleted} empty folder(s)")
        
        for folder in self.watched_folders:
            folder = os.path.normpath(folder)
            if not os.path.isdir(folder):
                continue

            try:
                # Walk the WHOLE tree (root + subfolders) so a file dropped
                # into a subfolder is also detected. Files that existed when
                # the watcher started were added to _processed_files in
                # start(); files we've organized are tracked in
                # _organized_files. Anything outside both sets is genuinely
                # new and should be routed by the current instruction.
                for root, dirs, files in os.walk(folder):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for item in files:
                        item_path = os.path.normpath(os.path.join(root, item))

                        if self._should_ignore(item):
                            continue
                        # Organize-New-Only loop suppression: once a file has
                        # been declared "leave in place" this session, skip it
                        # immediately — no hashing, no DB lookup, no nothing.
                        if item_path.lower() in self._left_in_place_paths:
                            continue
                        if item_path in self._processed_files:
                            continue
                        if item_path in self._organized_files:
                            continue

                        # Track pending files for debounce
                        if item_path not in self._pending_files:
                            self._pending_files[item_path] = current_time
                            logger.debug(f"New file detected: {item_path}")
                        else:
                            # Check if file has been stable long enough
                            first_seen = self._pending_files[item_path]
                            if current_time - first_seen >= self._debounce_seconds:
                                # Organize-New-Only filter (no-op for other
                                # modes): drop pre-existing files (baseline
                                # hash match OR known by content_hash in DB)
                                # and leave them in place forever.
                                to_process = self._filter_genuinely_new_files(folder, [item_path])
                                self._pending_files.pop(item_path, None)
                                if not to_process:
                                    continue
                                instruction = self._get_instruction_for_folder(folder)
                                existing_folders = self._get_existing_folders_if_as_is(folder)
                                self._process_files_with_ai(to_process, folder, instruction, existing_folders)
            except Exception as e:
                logger.error(f"Error checking folder {folder}: {e}")
    
    def _process_files_with_ai(self, file_paths: List[str], folder: str, instruction: str, 
                                existing_folders: List[str] = None) -> None:
        """
        Process files using AI to determine organization.
        
        Uses a background worker thread to prevent UI lag.
        
        Args:
            file_paths: List of file paths to organize
            folder: The destination folder (same as source folder for watch mode)
            instruction: User's organization instruction
            existing_folders: If provided, AI must ONLY use these folders (Organize As-Is mode)
        """
        if not file_paths:
            return
        
        # Build full instruction for AI
        parent_folder_name = os.path.basename(folder).lower()
        
        if existing_folders is not None and len(existing_folders) > 0:
            # ORGANIZE AS-IS MODE: Only use existing folders
            folders_list = ', '.join(f"'{f}'" for f in existing_folders)
            full_instruction = (
                f"[AUTO-ORGANIZE - EXISTING FOLDERS ONLY]\n"
                f"User's instructions: {instruction if instruction else 'Organize files into appropriate folders'}\n\n"
                f"EXISTING FOLDERS YOU CAN USE: {folders_list}\n\n"
                "GROUND RULES:\n"
                "• You can ONLY use the folders listed above — never create new folders.\n"
                "• Folder names use forward slashes for nested paths ('Photos/Vacation' = Vacation inside Photos).\n"
                "• When both a parent and a nested folder apply, PREFER the most-specific nested one (a vacation photo "
                "goes to 'Photos/Vacation', not just 'Photos').\n"
                "• Every file_id MUST appear exactly once in your plan. Omitting a file is FORBIDDEN.\n\n"
                "DECISION TREE — for EACH file, follow these steps IN ORDER:\n\n"
                "  Step 1: Compare the file's TAGS to the LEAF NAME of each existing folder. The LEAF NAME is the "
                "LAST segment after any slashes — for 'everything-else/pigs' the leaf is 'pigs' (not 'everything-else'). "
                "Does any tag (or its singular/plural form) match a leaf name? Tag 'pig' matches leaf 'pigs'; tag 'lion' "
                "matches leaf 'lions'; tag 'vacation' matches leaf 'Vacation'.\n"
                "    → YES: place the file in that folder. If MULTIPLE folders share a matching leaf (e.g. both 'pigs' "
                "and 'everything-else/pigs' would match a pig file), pick the most-specific (nested) one.\n"
                "    → NO: go to Step 2.\n\n"
                "  Step 2: Is there a CATCH-ALL folder among the existing folders? (A catch-all is one whose LEAF NAME "
                "contains 'other', 'everything', 'misc', 'general', 'rest', 'else', 'various', or 'unsorted'.)\n"
                "    → YES: place the file in that catch-all folder. STOP HERE. Do NOT put it in a subject folder it "
                "doesn't actually belong to just because that folder vaguely exists.\n"
                "    → NO: go to Step 3.\n\n"
                "  Step 3: No catch-all available — pick the existing folder whose name is most similar to the file's "
                "content type, using your best judgment.\n\n"
                "WORKED EXAMPLES:\n\n"
                "Folders = ['lions', 'everything-else', 'everything-else/pigs']  (NOTE: 'pigs' is NESTED inside "
                "everything-else):\n"
                "  • Lion photo  (tag 'lion') → 'lions'                  [Step 1: tag matches leaf 'lions']\n"
                "  • Pig photo   (tag 'pig')  → 'everything-else/pigs'   [Step 1: tag 'pig' matches LEAF 'pigs' of the "
                "nested folder — use the nested folder, NOT the parent 'everything-else']\n"
                "  • Bear photo  (tag 'bear') → 'everything-else'        [Step 2: no bear folder anywhere; catch-all]\n"
                "  • Sunset photo (no clear subject) → 'everything-else' [Step 2: catch-all]\n\n"
                "Folders = ['Photos', 'Photos/Vacation', 'Docs']:\n"
                "  • Vacation photo (tags 'photo','vacation') → 'Photos/Vacation'  [Step 1: 'vacation' matches the "
                "leaf of the nested folder; prefer nested over the parent 'Photos']\n"
                "  • Regular photo (tag 'photo' only) → 'Photos'  [Step 1: 'photo' matches leaf 'Photos'; nested "
                "'Photos/Vacation' does not apply because the file has no vacation tag]\n"
                "  • PDF document → 'Docs'  [Step 1: 'document' matches leaf 'Docs']\n\n"
                "FORBIDDEN:\n"
                "  • Putting a PIG photo in 'everything-else' when 'everything-else/pigs' exists. The nested 'pigs' "
                "folder is the correct destination for pig files — don't dump them in the parent.\n"
                "  • Putting a BEAR photo in 'lions' or 'pigs' just because they're animal-themed — they are not bear "
                "folders. Bears with no bear-named folder go to the catch-all.\n"
                "  • Omitting any file from your response (every file_id must appear exactly once).\n"
                "  • Creating a folder not in the listed existing folders."
            )
            logger.info(f"[Worker] Organize As-Is mode: restricting to folders: {existing_folders}")
        elif instruction:
            full_instruction = (
                f"[AUTO-ORGANIZE] User's specific instructions: {instruction}\n\n"
                f"PARENT FOLDER NAME: '{parent_folder_name}' - DO NOT create a folder with this name!\n\n"
                "RULES FOR AUTO-ORGANIZE MODE:\n"
                "1. CRITICAL — TAGS ARE AUTHORITATIVE: classify each file by its tags FIRST, the filename is only a fallback. "
                "If a file's tags contain a topic that matches a user-named folder (e.g. tags include 'lion' and the user "
                "named a 'lions' folder), the file MUST go to that folder, even if the filename does not obviously suggest it. "
                "A photo whose filename is 'hassan-pond.jpg' but whose tags include 'lion' is a LION photo and belongs in the "
                "lions folder.\n"
                "2. STRICT: Files of any TYPE/SUBJECT the user explicitly mentioned MUST go to the folder the user named for "
                "that subject. No exceptions, no alternative folders. If the user said 'videos to Clips', EVERY video file "
                "goes to 'Clips' — do NOT route any video to a different folder (no 'media', no 'videos', no 'Clips_2').\n"
                "3. ONLY for files whose tags/subject the user did NOT mention may you create a new, meaningful folder by "
                "content type (e.g. 'documents' for text/PDFs, 'audio' for music, 'code' for source files). This permission "
                "applies STRICTLY to subjects the user did not name in their instruction — it is NOT a license to invent extra "
                "folders for files the user already routed.\n"
                "4. FORBIDDEN folder names (catch-all junk drawers): 'misc', 'other', 'unsorted', 'miscellaneous', 'random', "
                "'stuff', 'various', 'etc'. Never use these names. If a file truly fits nowhere meaningful, place it in the "
                "closest existing folder by content type.\n"
                f"5. IMPORTANT: Do NOT create a folder named '{parent_folder_name}' - use a different name.\n"
                "6. Every file MUST be placed in exactly one folder."
            )
        else:
            full_instruction = (
                f"[AUTO-ORGANIZE] Organize ALL files into logical folders based on file type and content.\n"
                f"PARENT FOLDER NAME: '{parent_folder_name}' - DO NOT create a folder with this name!\n\n"
                "Use clear folder names like 'photos', 'docs', 'videos', 'audio', 'misc'.\n"
                f"IMPORTANT: Do NOT create a folder named '{parent_folder_name}' since that's the parent folder.\n"
                "EVERY file MUST be placed in a folder - NO files left out."
            )
        
        # If a worker is already running, queue this request
        # Store folder info instead of file paths - we'll re-scan when processing
        if self._current_worker is not None and self._current_worker.isRunning():
            # Don't add duplicate entries for the same folder
            folder_normalized = os.path.normpath(folder)
            already_queued = any(
                os.path.normpath(f) == folder_normalized 
                for f, _, _ in self._worker_queue
            )
            if not already_queued:
                logger.info(f"Worker busy, queuing folder {os.path.basename(folder)} for later processing")
                self._worker_queue.append((folder, full_instruction, existing_folders))
            else:
                logger.info(f"Folder {os.path.basename(folder)} already queued, skipping duplicate")
            return
        
        # Create and start worker thread
        self._start_worker(file_paths, folder, full_instruction, existing_folders)
    
    def _start_worker(self, file_paths: List[str], folder: str, instruction: str, 
                       existing_folders: List[str] = None) -> None:
        """Start a background worker to process files."""
        logger.info(f"Starting background worker for {len(file_paths)} files")
        
        self._current_worker = AutoWatcherWorker(
            file_paths, folder, instruction, self.folder_instructions, existing_folders
        )
        
        # Connect worker signals to our signals
        self._current_worker.file_indexed.connect(self._on_worker_file_indexed)
        self._current_worker.file_organized.connect(self._on_worker_file_organized)
        self._current_worker.status_changed.connect(self._on_worker_status)
        self._current_worker.error_occurred.connect(self._on_worker_error)
        self._current_worker.limit_reached.connect(self._on_worker_limit_reached)
        self._current_worker.finished_processing.connect(self._on_worker_finished_with_files)
        
        self._current_worker.start()
    
    def _on_worker_file_indexed(self, file_path: str):
        """Handle file indexed from worker."""
        self.file_indexed.emit(file_path)
    
    def _on_worker_file_organized(self, source: str, dest: str, category: str):
        """Handle file organized from worker."""
        self._processed_files.add(source)
        self._processed_files.add(dest)
        # Track the destination as an organized file (prevents re-processing)
        self._organized_files.add(os.path.normpath(dest))
        self.file_organized.emit(source, dest, category)
    
    def _on_worker_status(self, status: str):
        """Handle status update from worker."""
        self.status_changed.emit(status)
    
    def _on_worker_error(self, file_path: str, error: str):
        """Handle error from worker."""
        self.error_occurred.emit(file_path, error)

    def _on_worker_limit_reached(self, info: dict):
        """Forward the index-limit signal so the UI can show the upgrade popup."""
        self.limit_reached.emit(info or {})

    def _on_worker_finished_with_files(self, processed_files: list):
        """Handle worker finished with list of processed files."""
        # Add all processed files to _organized_files to prevent re-processing
        for file_path in processed_files:
            self._organized_files.add(os.path.normpath(file_path))
        
        logger.info(f"Worker finished, marked {len(processed_files)} files as organized")
        
        # Continue with regular finish handling
        self._on_worker_finished()
    
    def _on_worker_finished(self):
        """Handle worker finished - process queue if any."""
        logger.info("Worker finished processing")
        
        # Process next item in queue if any
        if self._worker_queue:
            folder, instruction, existing_folders = self._worker_queue.pop(0)

            # RE-SCAN the folder to get fresh file paths (not stale ones from before)
            # This also filters out files already in _organized_files, files
            # captured by the start()-time snapshot, and Organize-New-Only
            # leave-in-place paths.
            file_paths = self._scan_folder_for_files(folder, exclude_organized=True)

            # Organize-New-Only baseline + DB filter — same path the periodic
            # scan uses. No-op for non-action-3 folders.
            file_paths = self._filter_genuinely_new_files(folder, file_paths)

            if file_paths:
                logger.info(f"Processing queued request: {len(file_paths)} NEW files in {os.path.basename(folder)}")
                self._start_worker(file_paths, folder, instruction, existing_folders)
            else:
                logger.info(f"Queued folder {folder} has no NEW files to process (all already organized)")
                # Clear remaining queue for same folder to prevent loops
                self._worker_queue = [
                    (f, i, e) for f, i, e in self._worker_queue 
                    if os.path.normpath(f) != os.path.normpath(folder)
                ]
                # Process next in queue if any remain
                if self._worker_queue:
                    self._on_worker_finished()
                else:
                    self.status_changed.emit(f"Watching {len(self.watched_folders)} folder(s) for new files...")
        else:
            # All done - update status
            self.status_changed.emit(f"Watching {len(self.watched_folders)} folder(s) for new files...")
    
    def _scan_folder_for_files(self, folder: str, exclude_organized: bool = False) -> List[str]:
        """
        Scan a folder and return list of file paths.
        Used when processing queued requests to get fresh file paths.

        Args:
            folder: The folder to scan
            exclude_organized: If True, exclude files already in _organized_files

        IMPORTANT: This is the queue-dequeue path. It MUST apply the same
        "not new" filters as the periodic scan in ``_check_for_new_files``,
        otherwise pre-existing files in subfolders (captured by the
        start()-time ``_processed_files`` snapshot, or marked
        leave-in-place by Organize New Only) get re-walked here and end up
        in the next worker batch. That manifests as "the watcher organized
        files that were already sitting in subfolders before I clicked
        Start" — exactly the v12.2.2 regression seen in the log at
        22:40:54 where everything-else/* and tiggers/regal-bengal-tiger
        were reorganized despite being in the baseline.
        """
        folder = os.path.normpath(folder)
        file_paths = []

        if not os.path.isdir(folder):
            return file_paths

        for root, dirs, files in os.walk(folder):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for item in files:
                if self._should_ignore(item):
                    continue
                item_path = os.path.join(root, item)
                normalized_path = os.path.normpath(item_path)

                # Files captured by the start()-time snapshot were there
                # before watching began; never let them through the queue
                # path either.
                if normalized_path in self._processed_files:
                    continue
                # Organize-New-Only loop suppression: once a file has been
                # declared "leave in place" this session, skip it here too.
                if normalized_path.lower() in self._left_in_place_paths:
                    continue
                # Skip files that have already been organized (prevents loops).
                if exclude_organized and normalized_path in self._organized_files:
                    continue

                file_paths.append(item_path)

        return file_paths
    
    def _execute_plan(self, plan: Dict, files_by_id: Dict, dest_folder: str) -> None:
        """Execute the organization plan by moving files."""
        folders = plan.get('folders', {})
        
        if not folders:
            logger.info("Plan has no folders")
            return
        
        moved_count = 0
        error_count = 0
        dest_folder = os.path.normpath(dest_folder)
        
        for folder_name, file_ids in folders.items():
            # Create destination subfolder
            target_folder = os.path.join(dest_folder, folder_name)
            
            for file_id in file_ids:
                try:
                    # Handle string IDs from AI
                    file_id_int = int(file_id)
                    file_info = files_by_id.get(file_id_int)
                    
                    if not file_info:
                        logger.warning(f"File ID not found: {file_id}")
                        continue
                    
                    source_path = file_info['file_path']
                    file_name = file_info['file_name']
                    
                    if not os.path.exists(source_path):
                        logger.warning(f"Source file no longer exists: {source_path}")
                        continue
                    
                    # Create target folder if needed
                    os.makedirs(target_folder, exist_ok=True)
                    
                    dest_path = os.path.join(target_folder, file_name)
                    
                    # Handle name conflicts
                    if os.path.exists(dest_path) and os.path.normpath(source_path) != os.path.normpath(dest_path):
                        base, ext = os.path.splitext(file_name)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(target_folder, f"{base} ({counter}){ext}")
                            counter += 1
                    
                    # Skip if source and dest are the same
                    if os.path.normpath(source_path) == os.path.normpath(dest_path):
                        logger.debug(f"File already in place: {source_path}")
                        self._processed_files.add(source_path)
                        continue
                    
                    # Move the file
                    shutil.move(source_path, dest_path)
                    moved_count += 1
                    
                    # Track as processed
                    self._processed_files.add(source_path)
                    self._processed_files.add(dest_path)
                    
                    logger.info(f"Organized: {source_path} -> {dest_path}")
                    self.file_organized.emit(source_path, dest_path, folder_name)
                    
                    # Update database path using actual DB ID (not sequential ID)
                    from app.core.database import file_index
                    db_id = file_info.get('db_id')
                    if db_id:
                        file_index.update_file_path(db_id, dest_path)
                        logger.debug(f"Updated DB path for file {db_id}: {dest_path}")
                    else:
                        # File wasn't in DB yet - try to find by old path and update
                        old_record = file_index.get_file_by_path(source_path)
                        if old_record:
                            file_index.update_file_path(old_record['id'], dest_path)
                            logger.debug(f"Updated DB path for file {old_record['id']}: {dest_path}")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error moving file {file_id}: {e}")
                    self.error_occurred.emit(str(file_id), str(e))
        
        # Clean up empty folders
        if moved_count > 0:
            deleted_folders = self._cleanup_empty_folders(dest_folder)
            if deleted_folders > 0:
                logger.info(f"Deleted {deleted_folders} empty folder(s)")
            
            self.status_changed.emit(f"Organized {moved_count} file(s)" + 
                                     (f" ({error_count} errors)" if error_count else ""))
        elif error_count > 0:
            self.status_changed.emit(f"Organization failed: {error_count} error(s)")
    