"""
Move application and execution logic.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .settings import settings


logger = logging.getLogger(__name__)


def _get_unique_path(dest_path: Path) -> Path:
    """
    Generate a unique file path by adding (1), (2), etc. if file exists.
    
    Example: document.pdf → document (1).pdf → document (2).pdf
    
    Args:
        dest_path: The desired destination path
        
    Returns:
        A unique path that doesn't exist yet
    """
    if not dest_path.exists():
        return dest_path
    
    stem = dest_path.stem  # filename without extension
    suffix = dest_path.suffix  # .pdf, .jpg, etc.
    parent = dest_path.parent
    
    counter = 1
    while True:
        new_name = f"{stem} ({counter}){suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            logger.info(f"Duplicate detected: {dest_path.name} → {new_name}")
            return new_path
        counter += 1
        if counter > 1000:  # Safety limit
            raise ValueError(f"Could not find unique name for {dest_path.name} after 1000 attempts")


def apply_moves(move_plan: List[Dict[str, Any]]) -> Tuple[bool, List[str], str, int]:
    """
    Apply the move plan to actually move files.
    
    Handles duplicate files by auto-renaming (e.g., file.pdf → file (1).pdf).
    
    Args:
        move_plan: List of move plan dictionaries
        
    Returns:
        Tuple of (success, list_of_errors, log_file_path, renamed_count)
    """
    errors = []
    successful_moves = []
    
    # Create move log entry
    move_log = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(move_plan),
        "moves": [],
        "renamed_files": []  # Track files that were auto-renamed
    }
    renamed_count = 0
    
    try:
        for i, move in enumerate(move_plan):
            try:
                source_path = Path(move['source_path'])
                dest_path = Path(move['destination_path'])
                
                # Ensure source still exists
                if not source_path.exists():
                    if dest_path.exists():
                        # File already reached its destination — treat as success
                        successful_moves.append(move)
                        logger.info(f"Already at destination, counting as success: {source_path.name}")
                    else:
                        error_msg = f"Source file no longer exists: {source_path}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                    continue
                
                # Create destination directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle duplicate files - auto-rename if destination already exists
                original_dest = dest_path
                if dest_path.exists():
                    dest_path = _get_unique_path(dest_path)
                    # Update the move entry with new destination path
                    move['destination_path'] = str(dest_path)
                    renamed_count += 1
                    move_log["renamed_files"].append({
                        "original_name": original_dest.name,
                        "new_name": dest_path.name,
                        "folder": str(dest_path.parent)
                    })
                
                # Move the file
                shutil.move(str(source_path), str(dest_path))
                
                # Log successful move
                move_entry = {
                    "from": str(source_path.absolute()),
                    "to": str(dest_path.absolute()),
                    "timestamp": datetime.now().isoformat()
                }
                move_log["moves"].append(move_entry)
                successful_moves.append(move)
                
                logger.info(f"Moved {source_path.name} to {dest_path}")
                
            except Exception as e:
                error_msg = f"Error moving {move.get('file_name', 'unknown')}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
                continue
        
        # Save move log
        log_file_path = _save_move_log(move_log)
        
        success = len(errors) == 0
        renamed_msg = f", {renamed_count} renamed to avoid duplicates" if renamed_count > 0 else ""
        logger.info(f"Move operation completed. {len(successful_moves)} successful{renamed_msg}, {len(errors)} errors")
        
        return success, errors, log_file_path, renamed_count
        
    except Exception as e:
        error_msg = f"Critical error during move operation: {e}"
        errors.append(error_msg)
        logger.error(error_msg)
        return False, errors, "", 0


def _save_move_log(move_log: Dict[str, Any]) -> str:
    """
    Save move log to JSON file.
    
    Args:
        move_log: Move log dictionary
        
    Returns:
        Path to saved log file
    """
    try:
        moves_dir = settings.get_moves_dir()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_filename = f"moves-{timestamp}.json"
        log_file_path = moves_dir / log_filename
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(move_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Move log saved to: {log_file_path}")
        return str(log_file_path)
        
    except Exception as e:
        logger.error(f"Error saving move log: {e}")
        return ""


def get_move_history() -> List[Dict[str, Any]]:
    """
    Get history of move operations.
    
    Returns:
        List of move log summaries
    """
    history = []
    
    try:
        moves_dir = settings.get_moves_dir()
        
        for log_file in moves_dir.glob("moves-*.json"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                history.append({
                    "log_file": str(log_file),
                    "timestamp": log_data.get("timestamp", ""),
                    "total_files": log_data.get("total_files", 0),
                    "successful_moves": len(log_data.get("moves", []))
                })
                
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return history
        
    except Exception as e:
        logger.error(f"Error getting move history: {e}")
        return []


def validate_destination_space(move_plan: List[Dict[str, Any]], 
                             destination_root: Path) -> Tuple[bool, str]:
    """
    Validate that there's enough space in destination.
    
    Args:
        move_plan: List of move plan dictionaries
        destination_root: Destination directory root
        
    Returns:
        Tuple of (has_enough_space, error_message)
    """
    try:
        # Calculate required space
        required_space = sum(move.get('size', 0) for move in move_plan)
        
        # Get available space on destination drive
        total, used, free = shutil.disk_usage(destination_root)
        
        if required_space > free:
            required_mb = round(required_space / (1024 * 1024), 2)
            free_mb = round(free / (1024 * 1024), 2)
            return False, f"Insufficient space. Required: {required_mb}MB, Available: {free_mb}MB"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error checking disk space: {e}"


