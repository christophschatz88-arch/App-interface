"""
Auto-organize watcher module.

Watches folders for new files and automatically organizes them using AI.
Supports per-folder instructions and catch-up mode for files added while app was closed.
"""

import os
import shutil
import logging
import time
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
            
            limit_reached = False
            for i, file_path in enumerate(files_to_index):
                if self._should_stop:
                    self.finished_processing.emit(all_processed_files)
                    return
                
                # Stop indexing if limit was reached on a previous file
                if limit_reached:
                    logger.info(f"[Worker] Skipping remaining {len(files_to_index) - i} files due to index limit")
                    break
                    
                try:
                    result = search_service.index_single_file(Path(file_path), force_ai=False)
                    
                    if result.get('success'):
                        indexed_count += 1
                        logger.info(f"[Worker] Auto-indexed: {os.path.basename(file_path)}")
                        self.file_indexed.emit(file_path)
                        self.status_changed.emit(f"Indexed {indexed_count}/{len(files_to_index)} files...")
                    elif result.get('limit_reached'):
                        # Index limit reached - stop trying to index more media files
                        limit_reached = True
                        logger.warning(f"[Worker] Index limit reached: {result.get('error')}")
                        self.status_changed.emit("Index limit reached - upgrade for more")
                        self.error_occurred.emit(file_path, result.get('error', 'Index limit reached'))
                    elif result.get('error'):
                        logger.warning(f"[Worker] Failed to index {file_path}: {result.get('error')}")
                    
                    # Small delay between files
                    if i < len(files_to_index) - 1:
                        time.sleep(0.3)
                        
                except Exception as e:
                    logger.error(f"[Worker] Error auto-indexing {file_path}: {e}")
            
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
            
            # In "Organize As-Is" mode, DON'T add missing files to 'misc'
            # Instead, leave them where they are
            if not self.existing_folders:
                plan = ensure_all_files_included(plan, {f['id'] for f in files_info})
            
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
            skipped_folders = []  # Folders that didn't match any existing folder
            
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
                        # No match - don't create new folder, skip these files
                        skipped_folders.append(folder_name)
                        logger.info(f"[Worker] Skipping folder '{folder_name}' - not in existing folders, files will stay in place")
                else:
                    # Normal mode - use folder as-is
                    filtered_folders[folder_name] = valid_ids
            
            if skipped_folders:
                logger.info(f"[Worker] Skipped {len(skipped_folders)} non-existing folders: {skipped_folders}")
            
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
        
        folder_count = len(self.watched_folders)
        self.status_changed.emit(f"Starting watch on {folder_count} folder(s)...")
        logger.info(f"Starting watcher for {folder_count} folders")
        
        # Flatten folders first if requested (for re-organize)
        if flatten_first:
            total_flattened = 0
            for folder in self.watched_folders:
                count = self.flatten_folder(folder)
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
    
    def flatten_folder(self, folder_path: str) -> int:
        """
        Flatten folder by moving all files from subfolders to root.
        
        This is used for the "Re-organize All" feature to reset folder structure
        before applying new organization instructions.
        
        Args:
            folder_path: Path to the folder to flatten
            
        Returns:
            Number of files moved
        """
        folder_path = os.path.normpath(folder_path)
        if not os.path.isdir(folder_path):
            logger.warning(f"Cannot flatten - not a directory: {folder_path}")
            return 0
        
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
        
        # Clean up empty subdirectories
        self._cleanup_empty_folders(folder_path)
        
        logger.info(f"Flattened {moved_count} files in {folder_path}")
        return moved_count
    
    def _cleanup_empty_folders(self, root_folder: str) -> int:
        """Remove empty subdirectories. Returns count of removed folders."""
        removed_count = 0
        root_folder = os.path.normpath(root_folder)
        
        # Walk bottom-up to remove nested empty folders
        for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
            # Skip the root folder itself (normalize for comparison)
            if os.path.normpath(dirpath) == root_folder:
                continue
            
            # Skip hidden folders
            if os.path.basename(dirpath).startswith('.'):
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
        """Get the instruction for a specific folder."""
        folder_path = os.path.normpath(folder_path)
        instruction = self.folder_instructions.get(folder_path, '')
        logger.debug(f"Instruction for {folder_path}: {instruction[:50] if instruction else '(none)'}")
        return instruction

    def _get_existing_folders_if_as_is(self, folder_path: str) -> list:
        """Return existing subfolders if folder is in ORGANIZE_AS_IS mode, else None."""
        from app.core.settings import settings
        ORGANIZE_AS_IS = 2
        if settings.get_auto_organize_action(folder_path) == ORGANIZE_AS_IS:
            return [
                item for item in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, item)) and not item.startswith('.')
            ]
        return None
    
    def _organize_existing_files(self) -> None:
        """Organize files already in the watched folders (including subfolders)."""
        all_files = []
        
        for folder in self.watched_folders:
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
                count = self.flatten_folder(folder)
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
                count = self.flatten_folder(folder)
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
            existing_subfolders = []
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    existing_subfolders.append(item)
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
            count = self.flatten_folder(folder_path)
            if count > 0:
                self.status_changed.emit(f"Flattened {count} files, now organizing...")
        
        # Step 2: Get existing folder structure (for "Organize As-Is" mode)
        existing_folders = []
        if not flatten_first:
            # Collect existing subfolders - AI should only use these
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    existing_folders.append(item)
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
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    
                    if not os.path.isfile(item_path):
                        continue
                    
                    if self._should_ignore(item):
                        continue
                    
                    # Skip already processed files
                    if item_path in self._processed_files:
                        continue
                    
                    # Track pending files for debounce
                    if item_path not in self._pending_files:
                        self._pending_files[item_path] = current_time
                        logger.debug(f"New file detected: {item_path}")
                    else:
                        # Check if file has been stable long enough
                        first_seen = self._pending_files[item_path]
                        if current_time - first_seen >= self._debounce_seconds:
                            # File is stable, process it
                            instruction = self._get_instruction_for_folder(folder)
                            existing_folders = self._get_existing_folders_if_as_is(folder)
                            self._process_files_with_ai([item_path], folder, instruction, existing_folders)
                            self._pending_files.pop(item_path, None)
                            
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
                "CRITICAL RULES:\n"
                "1. You can ONLY use the folders listed above - DO NOT create any new folders\n"
                "2. Move EVERY file to the most appropriate EXISTING folder based on file type/content\n"
                "3. EVERY file MUST be included in your plan - put each file in the closest matching folder\n"
                "4. Do NOT leave any files out - use your best judgment for the closest match\n"
                "5. EVERY file_id in your response MUST go to one of the existing folders listed above"
            )
            logger.info(f"[Worker] Organize As-Is mode: restricting to folders: {existing_folders}")
        elif instruction:
            full_instruction = (
                f"[AUTO-ORGANIZE] User's specific instructions: {instruction}\n\n"
                f"PARENT FOLDER NAME: '{parent_folder_name}' - DO NOT create a folder with this name!\n\n"
                "RULES FOR AUTO-ORGANIZE MODE:\n"
                "1. FOLLOW the user's specific instructions EXACTLY for any files they mentioned\n"
                "2. For ALL REMAINING files not covered by user's instructions, organize them logically by file type\n"
                "3. EVERY file MUST be placed in a folder - NO files left out\n"
                "4. Use simple, clear folder names (e.g., 'photos', 'docs', 'videos', 'audio', 'misc')\n"
                f"5. IMPORTANT: Do NOT create a folder named '{parent_folder_name}' - use different names\n"
                "6. If user says 'screenshots to X' - put screenshots in X, organize everything else by type"
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
            # This also filters out files already in _organized_files
            file_paths = self._scan_folder_for_files(folder, exclude_organized=True)
            
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
                
                # Skip files that have already been organized (prevents loops)
                if exclude_organized:
                    normalized_path = os.path.normpath(item_path)
                    if normalized_path in self._organized_files:
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
    