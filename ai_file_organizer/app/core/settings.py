"""
Application settings and configuration management.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class Settings:
    """Application settings manager."""
    
    def __init__(self):
        self.app_name = "ai-file-organizer"
        self.category_map = self._load_default_categories()
        self.mime_fallbacks = self._get_mime_fallbacks()
        # AI Provider: 'openai' (default, recommended), 'local' (Ollama), or 'none'
        self.ai_provider: str = 'openai'  # OpenAI is now the default - no local setup needed!
        self.openai_api_key: str | None = os.environ.get('OPENAI_API_KEY')
        self.openai_vision_model: str = os.environ.get('OPENAI_VISION_MODEL', 'gpt-4o-mini')  # Cost-effective default
        # Search rerank option (ChatGPT)
        self.use_openai_search_rerank: bool = False
        self.openai_search_model: str = 'gpt-4o-mini'
        # Local AI model settings (Ollama)
        # Qwen 2.5-VL handles BOTH text AND vision in one model
        self.local_model: str = 'qwen2.5vl:3b'
        # Quick search overlay
        self.use_quick_search: bool = True
        self.quick_search_shortcut: str = 'ctrl+alt+h'
        self.quick_search_autopaste: bool = True
        self.quick_search_auto_confirm: bool = True
        self.quick_search_geometry: Dict[str, int] = {}
        # Theme: 'dark' or 'light'
        self.theme: str = 'light'
        self._theme_explicitly_set: bool = False
        # Auto-index downloads folder (legacy - kept for compatibility)
        self.auto_index_downloads: bool = False
        # Watch for new downloads - common folders (Downloads, Desktop, Documents, etc.)
        self.watch_common_folders: bool = False
        # Watch for new downloads - custom folders list
        self.watch_custom_folders: List[str] = []
        # OCR during indexing (slow - disable for faster indexing)
        self.enable_ocr_indexing: bool = False
        # Search enhancements
        # Single toggle: when enabled, we apply BOTH fuzzy keyword matching + spell correction
        self.enable_spell_check: bool = False
        # Auth tokens (stored securely)
        self.auth_access_token: str = ''
        self.auth_refresh_token: str = ''
        self.auth_user_email: str = ''
        
        # ======= AUTO-ORGANIZE WATCHER SETTINGS =======
        # List of folders with per-folder instructions: [{path: str, instruction: str}, ...]
        self.auto_organize_folders: List[Dict[str, str]] = []
        # Auto-start watcher when app opens (default True)
        self.auto_organize_auto_start: bool = True
        # User explicitly stopped the watcher and we should NOT auto-start it
        # back up — across tab switches, dialog Saves, or app restarts.
        # Cleared again when the user clicks Start.
        self.auto_organize_paused: bool = False
        # Last active timestamp (ISO format) for catch-up feature
        self.auto_organize_last_active: str = ''
        
        # ======= EXCLUSIONS SETTINGS =======
        # Patterns to exclude from organization (folders, files, wildcards)
        self.exclusion_patterns: List[str] = self._get_default_exclusions()
        
        # ======= PINNED FILES SETTINGS =======
        # Specific file/folder paths that are "pinned" and should never be organized
        self.pinned_paths: List[str] = []
        
        # ======= ONBOARDING SETTINGS =======
        # Whether the user has completed the onboarding tour
        self.has_completed_onboarding: bool = False
        # How many times user clicked "Remind Me Later" (stop after 3)
        self.onboarding_remind_count: int = 0
        # List of contextual tip IDs that have been seen/dismissed
        self.seen_tips: List[str] = []
        
        # Load persisted config if available
        try:
            self._load_config()
        except Exception:
            pass
    
    def _load_default_categories(self) -> Dict[str, List[str]]:
        """Load default category mappings from resources."""
        try:
            # Try to load from resources first
            resource_path = Path(__file__).parent.parent.parent / "resources" / "category_defaults.json"
            if resource_path.exists():
                with open(resource_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Fallback to hardcoded defaults
        return {
            "Documents/PDFs": [".pdf"],
            "Documents/Word": [".doc", ".docx", ".rtf"],
            "Documents/Text": [".txt", ".md"],
            "Spreadsheets": [".xls", ".xlsx", ".csv"],
            "Presentations": [".ppt", ".pptx"],
            "Images/Photos": [".jpg", ".jpeg"],
            "Images/Screenshots": [".png"],
            "Images/Graphics": [".gif", ".svg", ".webp"],
            "Videos": [".mp4", ".mov"],
            "Audio/Music": [".mp3"],
            "Audio/Recordings": [".wav", ".m4a"],
            "Archives": [".zip", ".rar", ".7z"],
            "Code": [".py", ".js", ".ts"],
            "Misc": []
        }
    
    def _get_mime_fallbacks(self) -> Dict[str, str]:
        """Get MIME type fallback mappings."""
        return {
            "image/": "Images/Photos",
            "video/": "Videos", 
            "audio/": "Audio/Recordings",
            "application/pdf": "Documents/PDFs"
        }
    
    def _get_default_exclusions(self) -> List[str]:
        """Get default exclusion patterns for files/folders that should not be organized."""
        return [
            # Version control
            ".git",
            ".svn",
            ".hg",
            # Dependencies & packages
            "node_modules",
            "vendor",
            "packages",
            # Python
            "__pycache__",
            "venv",
            ".venv",
            "*.pyc",
            ".eggs",
            "*.egg-info",
            # Build & cache
            "build",
            "dist",
            ".cache",
            ".tox",
            # Config files
            ".env",
            ".env.*",
            # IDE & editors
            ".vscode",
            ".idea",
            "*.sublime-*",
            # System files
            "Thumbs.db",
            ".DS_Store",
            "desktop.ini",
            # Temp files
            "*.tmp",
            "*.temp",
            "~$*",
        ]
    
    def get_app_data_dir(self) -> Path:
        """Get application data directory."""
        if os.name == 'nt':  # Windows
            app_data = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        else:  # macOS/Linux
            app_data = Path.home() / '.config'
        
        app_dir = app_data / self.app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir
    
    def get_moves_dir(self) -> Path:
        """Get moves log directory."""
        moves_dir = self.get_app_data_dir() / "moves"
        moves_dir.mkdir(parents=True, exist_ok=True)
        return moves_dir

    # Runtime updates from UI
    def set_openai_api_key(self, key: str | None) -> None:
        key = (key or '').strip()
        self.openai_api_key = key if key else None
        if self.openai_api_key:
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
        else:
            try:
                del os.environ['OPENAI_API_KEY']
            except Exception:
                pass
        self._save_config()

    def set_ai_provider(self, provider: str) -> None:
        """Set the AI provider: 'openai' (default), 'local' (Ollama), or 'none'."""
        if provider in ('openai', 'local', 'none'):
            self.ai_provider = provider
        else:
            self.ai_provider = 'openai'  # Default to OpenAI
        self._save_config()
    
    # Legacy compatibility - maps to ai_provider
    @property
    def use_openai_fallback(self) -> bool:
        """Legacy property - returns True if AI provider is OpenAI."""
        return self.ai_provider == 'openai'
    
    def set_use_openai_fallback(self, use: bool) -> None:
        self.ai_provider = 'openai' if use else 'local'
        self._save_config()

    def set_openai_vision_model(self, model: str) -> None:
        model = (model or '').strip() or 'gpt-4o-mini'
        self.openai_vision_model = model
        os.environ['OPENAI_VISION_MODEL'] = model
        self._save_config()

    def delete_openai_api_key(self) -> None:
        self.openai_api_key = None
        try:
            del os.environ['OPENAI_API_KEY']
        except Exception:
            pass
        self._save_config()

    # Local AI model setter
    def set_local_model(self, model: str) -> None:
        """Set the local model for Ollama (Qwen 2.5-VL handles both text and vision)."""
        self.local_model = (model or '').strip() or 'qwen2.5vl:3b'
        self._save_config()

    # Persistence helpers
    def _config_file(self) -> Path:
        return self.get_app_data_dir() / 'settings.json'

    def _load_config(self) -> None:
        cfg_file = self._config_file()
        if not cfg_file.exists():
            return
        with open(cfg_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # AI Provider (migrate from old use_openai_fallback)
        ai_prov = data.get('ai_provider')
        if ai_prov in ('openai', 'local', 'none'):
            self.ai_provider = ai_prov
        elif data.get('use_openai_fallback'):
            # Migrate old setting: if they had OpenAI fallback enabled, keep using OpenAI
            self.ai_provider = 'openai'
        # else keep default 'openai'
        self.use_openai_search_rerank = bool(data.get('use_openai_search_rerank', self.use_openai_search_rerank))
        self.use_quick_search = bool(data.get('use_quick_search', self.use_quick_search))
        k = data.get('openai_api_key')
        if isinstance(k, str) and k.strip():
            self.openai_api_key = k.strip()
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
        m = data.get('openai_vision_model')
        if isinstance(m, str) and m.strip():
            self.openai_vision_model = m.strip()
            os.environ['OPENAI_VISION_MODEL'] = self.openai_vision_model
        sm = data.get('openai_search_model')
        if isinstance(sm, str) and sm.strip():
            self.openai_search_model = sm.strip()
        # Local AI model (single model for both text and vision)
        lm = data.get('local_model')
        if isinstance(lm, str) and lm.strip():
            # Migrate from 7b to 3b (7b requires too much RAM for most systems)
            loaded_model = lm.strip()
            if loaded_model == 'qwen2.5vl:7b':
                loaded_model = 'qwen2.5vl:3b'  # Auto-migrate to lighter version
            self.local_model = loaded_model
        qs = data.get('quick_search_shortcut')
        if isinstance(qs, str) and qs.strip():
            self.quick_search_shortcut = qs.strip().lower()
        self.quick_search_autopaste = bool(data.get('quick_search_autopaste', self.quick_search_autopaste))
        self.quick_search_auto_confirm = bool(data.get('quick_search_auto_confirm', self.quick_search_auto_confirm))
        qsg = data.get('quick_search_geometry')
        if isinstance(qsg, dict):
            self.quick_search_geometry = {k: int(v) for k, v in qsg.items() if k in {'x','y','w','h'} and isinstance(v, (int, float, str))}
        # Theme — migrate existing users to light if they never explicitly chose dark
        theme = data.get('theme')
        if theme in ('dark', 'light'):
            if theme == 'dark' and not data.get('theme_explicitly_set', False):
                self.theme = 'light'  # one-time migration: default changed to light
            else:
                self.theme = theme
        # Auto-index downloads (legacy)
        self.auto_index_downloads = bool(data.get('auto_index_downloads', False))
        # Watch for new downloads - common folders
        self.watch_common_folders = bool(data.get('watch_common_folders', False))
        # Watch for new downloads - custom folders
        self.watch_custom_folders = list(data.get('watch_custom_folders', []))
        # OCR during indexing (disabled by default for speed)
        self.enable_ocr_indexing = bool(data.get('enable_ocr_indexing', False))
        # Search enhancements
        # Migration: previously we had separate enable_fuzzy_search and enable_spell_check.
        # Now a single toggle controls both; treat either legacy flag as enabling spell_check.
        legacy_fuzzy = data.get('enable_fuzzy_search')
        legacy_spell = data.get('enable_spell_check')
        if legacy_spell is not None:
            self.enable_spell_check = bool(legacy_spell)
        elif legacy_fuzzy is not None:
            self.enable_spell_check = bool(legacy_fuzzy)
        else:
            self.enable_spell_check = bool(data.get('enable_spell_check', self.enable_spell_check))
        # Auth tokens
        self.auth_access_token = data.get('auth_access_token', '')
        self.auth_refresh_token = data.get('auth_refresh_token', '')
        self.auth_user_email = data.get('auth_user_email', '')
        
        # Auto-organize watcher settings
        auto_folders = data.get('auto_organize_folders', [])
        if isinstance(auto_folders, list):
            self.auto_organize_folders = auto_folders
        self.auto_organize_auto_start = bool(data.get('auto_organize_auto_start', True))
        self.auto_organize_paused = bool(data.get('auto_organize_paused', False))
        self.auto_organize_last_active = data.get('auto_organize_last_active', '')
        
        # Exclusion patterns
        exclusions = data.get('exclusion_patterns')
        if isinstance(exclusions, list):
            self.exclusion_patterns = exclusions
        # If not in config, keep defaults (already set in __init__)
        
        # Pinned paths
        pinned = data.get('pinned_paths')
        if isinstance(pinned, list):
            self.pinned_paths = pinned
        
        # Onboarding
        self.has_completed_onboarding = bool(data.get('has_completed_onboarding', False))
        self.onboarding_remind_count = int(data.get('onboarding_remind_count', 0))
        self.seen_tips = list(data.get('seen_tips', []))

    def _save_config(self) -> None:
        cfg = {
            'ai_provider': self.ai_provider,
            'openai_api_key': self.openai_api_key or '',
            'openai_vision_model': self.openai_vision_model,
            'use_openai_search_rerank': self.use_openai_search_rerank,
            'openai_search_model': self.openai_search_model,
            'local_model': self.local_model,
            'use_quick_search': self.use_quick_search,
            'quick_search_shortcut': self.quick_search_shortcut,
            'quick_search_autopaste': self.quick_search_autopaste,
            'quick_search_auto_confirm': self.quick_search_auto_confirm,
            'quick_search_geometry': self.quick_search_geometry,
            'theme': self.theme,
            'theme_explicitly_set': getattr(self, '_theme_explicitly_set', False),
            'auto_index_downloads': self.auto_index_downloads,
            'watch_common_folders': self.watch_common_folders,
            'watch_custom_folders': self.watch_custom_folders,
            'enable_ocr_indexing': self.enable_ocr_indexing,
            'enable_spell_check': self.enable_spell_check,
            'auth_access_token': self.auth_access_token,
            'auth_refresh_token': self.auth_refresh_token,
            'auth_user_email': self.auth_user_email,
            # Auto-organize watcher settings
            'auto_organize_folders': self.auto_organize_folders,
            'auto_organize_auto_start': self.auto_organize_auto_start,
            'auto_organize_paused': self.auto_organize_paused,
            'auto_organize_last_active': self.auto_organize_last_active,
            # Exclusion patterns
            'exclusion_patterns': self.exclusion_patterns,
            # Pinned paths
            'pinned_paths': self.pinned_paths,
            # Onboarding
            'has_completed_onboarding': self.has_completed_onboarding,
            'onboarding_remind_count': self.onboarding_remind_count,
            'seen_tips': self.seen_tips,
        }
        try:
            with open(self._config_file(), 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
        except Exception:
            pass

    # Search rerank toggle
    def set_use_openai_search_rerank(self, use: bool) -> None:
        self.use_openai_search_rerank = bool(use)
        self._save_config()

    # Quick search setters
    def set_quick_search_shortcut(self, shortcut: str) -> None:
        sc = (shortcut or '').strip().lower() or 'ctrl+x'
        self.quick_search_shortcut = sc
        self._save_config()

    def set_quick_search_autopaste(self, use: bool) -> None:
        self.quick_search_autopaste = bool(use)
        self._save_config()

    def set_quick_search_auto_confirm(self, use: bool) -> None:
        self.quick_search_auto_confirm = bool(use)
        self._save_config()

    def set_theme(self, theme: str) -> None:
        """Set the application theme ('dark' or 'light')."""
        if theme in ('dark', 'light'):
            self.theme = theme
            self._theme_explicitly_set = True
            self._save_config()

    def set_auto_index_downloads(self, enabled: bool) -> None:
        """Enable or disable auto-indexing of Downloads folder."""
        self.auto_index_downloads = bool(enabled)
        self._save_config()
    
    def set_watch_common_folders(self, enabled: bool) -> None:
        """Enable or disable watching common folders for new downloads."""
        self.watch_common_folders = bool(enabled)
        self._save_config()
    
    def add_watch_custom_folder(self, folder_path: str) -> None:
        """Add a custom folder to watch for new downloads."""
        if folder_path and folder_path not in self.watch_custom_folders:
            self.watch_custom_folders.append(folder_path)
            self._save_config()
    
    def remove_watch_custom_folder(self, folder_path: str) -> None:
        """Remove a custom folder from the watch list."""
        if folder_path in self.watch_custom_folders:
            self.watch_custom_folders.remove(folder_path)
            self._save_config()

    def set_auth_tokens(self, access_token: str, refresh_token: str, email: str = '') -> None:
        """Store authentication tokens securely."""
        self.auth_access_token = access_token or ''
        self.auth_refresh_token = refresh_token or ''
        self.auth_user_email = email or ''
        self._save_config()

    def clear_auth_tokens(self) -> None:
        """Clear stored authentication tokens (logout)."""
        self.auth_access_token = ''
        self.auth_refresh_token = ''
        self.auth_user_email = ''
        self._save_config()

    def has_stored_session(self) -> bool:
        """Check if we have stored auth tokens."""
        return bool(self.auth_access_token and self.auth_refresh_token)

    def set_enable_spell_check(self, enabled: bool) -> None:
        """Enable or disable typo correction in search (fuzzy + spell check)."""
        self.enable_spell_check = bool(enabled)
        self._save_config()

    # ======= AUTO-ORGANIZE WATCHER METHODS =======
    
    def add_auto_organize_folder(self, folder_path: str, instruction: str = '') -> None:
        """Add a folder to auto-organize with its instruction."""
        # Check if folder already exists
        for folder in self.auto_organize_folders:
            if folder.get('path') == folder_path:
                folder['instruction'] = instruction
                self._save_config()
                return
        # Add new folder
        self.auto_organize_folders.append({
            'path': folder_path,
            'instruction': instruction
        })
        self._save_config()

    def remove_auto_organize_folder(self, folder_path: str) -> None:
        """Remove a folder from auto-organize."""
        self.auto_organize_folders = [
            f for f in self.auto_organize_folders 
            if f.get('path') != folder_path
        ]
        self._save_config()

    def update_auto_organize_instruction(self, folder_path: str, instruction: str) -> None:
        """Update instruction for a folder."""
        for folder in self.auto_organize_folders:
            if folder.get('path') == folder_path:
                folder['instruction'] = instruction
                self._save_config()
                return

    def update_auto_organize_action(self, folder_path: str, action: int) -> None:
        """Update (or insert) the organization action for a folder.

        If the folder isn't registered yet in auto_organize_folders, it is
        created on the spot. This matters when the user picks a per-folder
        action via the Options button *before* clicking the main Save — the
        choice would otherwise be silently dropped because the folder didn't
        yet exist in settings.

        Args:
            folder_path: Path to the folder
            action: 1=Re-organize All, 2=Organize As-Is, 3=Watch Only
        """
        for folder in self.auto_organize_folders:
            if folder.get('path') == folder_path:
                folder['action'] = action
                self._save_config()
                return
        # Folder not registered yet — create it with the chosen action so the
        # user's pick is persisted immediately. Instruction stays empty until
        # the main Save flow writes it.
        self.auto_organize_folders.append({
            'path': folder_path,
            'instruction': '',
            'action': action,
        })
        self._save_config()
    
    def get_auto_organize_action(self, folder_path: str) -> int:
        """Get the organization action for a folder.
        
        Returns:
            action: 1=Re-organize All, 2=Organize As-Is, 3=Watch Only (default)
        """
        for folder in self.auto_organize_folders:
            if folder.get('path') == folder_path:
                return folder.get('action', 3)  # Default to Watch Only
        return 3

    def set_auto_organize_paused(self, paused: bool) -> None:
        """Mark the watcher as user-paused (or un-pause it). Persisted across
        tab switches, dialog saves, and app restarts.
        """
        self.auto_organize_paused = bool(paused)
        self._save_config()

    def set_auto_organize_auto_start(self, enabled: bool) -> None:
        """Enable or disable auto-start of watcher on app open."""
        self.auto_organize_auto_start = bool(enabled)
        self._save_config()

    def update_auto_organize_last_active(self) -> None:
        """Update the last active timestamp to now."""
        self.auto_organize_last_active = datetime.now().isoformat()
        self._save_config()

    def get_auto_organize_last_active_time(self) -> Optional[datetime]:
        """Get the last active time as a datetime object, or None if not set."""
        if not self.auto_organize_last_active:
            return None
        try:
            return datetime.fromisoformat(self.auto_organize_last_active)
        except Exception:
            return None

    def clear_auto_organize_last_active(self) -> None:
        """Clear the last active timestamp."""
        self.auto_organize_last_active = ''
        self._save_config()

    # ======= EXCLUSION METHODS =======
    
    def add_exclusion_pattern(self, pattern: str) -> None:
        """Add an exclusion pattern."""
        pattern = pattern.strip()
        if pattern and pattern not in self.exclusion_patterns:
            self.exclusion_patterns.append(pattern)
            self._save_config()
    
    def remove_exclusion_pattern(self, pattern: str) -> None:
        """Remove an exclusion pattern."""
        if pattern in self.exclusion_patterns:
            self.exclusion_patterns.remove(pattern)
            self._save_config()
    
    def reset_exclusions_to_defaults(self) -> None:
        """Reset exclusion patterns to defaults."""
        self.exclusion_patterns = self._get_default_exclusions()
        self._save_config()
    
    def should_exclude(self, file_path: str) -> bool:
        """Check if a file/folder should be excluded based on patterns or if pinned.
        
        Supports:
        - Exact folder/file names: "node_modules", ".git"
        - Wildcards: "*.pyc", "*.tmp", "~$*"
        - Prefix wildcards: ".env.*"
        - File extensions: ".json" is automatically treated as "*.json"
        - Pinned paths: specific files/folders that are protected
        """
        import fnmatch
        import logging
        logger = logging.getLogger(__name__)
        
        # First check if the path is pinned (protected)
        if self.is_pinned(file_path):
            logger.debug(f"Excluding pinned path: {file_path}")
            return True
        
        # Get the file/folder name (case-insensitive matching)
        name = os.path.basename(file_path)
        name_lower = name.lower()
        file_path_lower = file_path.lower()
        
        for pattern in self.exclusion_patterns:
            pattern_lower = pattern.lower()
            
            # Check if pattern matches the name directly
            if fnmatch.fnmatch(name_lower, pattern_lower):
                logger.debug(f"Excluding {name} - matched pattern '{pattern}' directly")
                return True
            
            # Handle patterns like ".json" - treat as "*.json" for file extensions
            # This makes it user-friendly (no need to type the asterisk)
            if pattern_lower.startswith('.') and not pattern_lower.startswith('.*') and '*' not in pattern_lower:
                ext_pattern = '*' + pattern_lower
                if fnmatch.fnmatch(name_lower, ext_pattern):
                    logger.debug(f"Excluding {name} - matched extension pattern '{ext_pattern}'")
                    return True
        
        # Also check full path and path components for folder-based exclusions
        for pattern in self.exclusion_patterns:
            pattern_lower = pattern.lower()
            
            # Skip extension-only patterns for path matching (already handled above)
            if pattern_lower.startswith('.') and '*' not in pattern_lower and len(pattern_lower) <= 5:
                continue
            
            # Check if pattern matches the full path (for folder paths)
            if fnmatch.fnmatch(file_path_lower, f"*/{pattern_lower}") or fnmatch.fnmatch(file_path_lower, f"*\\{pattern_lower}"):
                logger.debug(f"Excluding {name} - path matched pattern '{pattern}'")
                return True
            
            # Check if any path component matches (for nested exclusions)
            path_parts = file_path.replace('\\', '/').split('/')
            for part in path_parts:
                if fnmatch.fnmatch(part.lower(), pattern_lower):
                    logger.debug(f"Excluding {name} - path component '{part}' matched pattern '{pattern}'")
                    return True
        
        return False
    
    # ======= PINNED PATHS METHODS =======
    
    def add_pinned_path(self, file_path: str) -> bool:
        """Pin a specific file or folder path so it's never organized.
        Returns True if successfully added, False if already pinned.
        """
        # Normalize the path for consistent comparison
        normalized = os.path.normpath(file_path)
        if normalized and normalized not in self.pinned_paths:
            self.pinned_paths.append(normalized)
            self._save_config()
            return True
        return False
    
    def remove_pinned_path(self, file_path: str) -> bool:
        """Unpin a file or folder path.
        Returns True if successfully removed, False if wasn't pinned.
        """
        normalized = os.path.normpath(file_path)
        if normalized in self.pinned_paths:
            self.pinned_paths.remove(normalized)
            self._save_config()
            return True
        return False
    
    def is_pinned(self, file_path: str) -> bool:
        """Check if a specific file or folder path is pinned."""
        normalized = os.path.normpath(file_path).lower()
        for pinned in self.pinned_paths:
            pinned_normalized = os.path.normpath(pinned).lower()
            # Check exact match
            if normalized == pinned_normalized:
                return True
            # Check if file is inside a pinned folder
            if normalized.startswith(pinned_normalized + os.sep):
                return True
        return False
    
    def get_pinned_paths(self) -> List[str]:
        """Get all pinned file/folder paths."""
        return self.pinned_paths.copy()
    
    def clear_all_pinned(self) -> None:
        """Remove all pinned paths."""
        self.pinned_paths = []
        self._save_config()
    
    # ======= ONBOARDING METHODS =======
    def complete_onboarding(self) -> None:
        """Mark the onboarding as completed."""
        self.has_completed_onboarding = True
        self._save_config()
    
    def reset_onboarding(self) -> None:
        """Reset onboarding so it shows again next time."""
        self.has_completed_onboarding = False
        self.onboarding_remind_count = 0
        self._save_config()
    
    def mark_tip_seen(self, tip_id: str) -> None:
        """Mark a contextual tip as seen"""
        if tip_id not in self.seen_tips:
            self.seen_tips.append(tip_id)
            self._save_config()
    
    def reset_tips(self) -> None:
        """Reset all contextual tips to show them again"""
        self.seen_tips = []
        self._save_config()


# Global settings instance
settings = Settings()


