#!/usr/bin/env python3
"""
Filect - File Search Assistant v1.0
A privacy-first desktop application for intelligent file search and quick path autofill.
Instantly find and autofill file paths in any application using global hotkeys.
"""

import sys
import os
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Handle PyInstaller bundled path vs running from source
if hasattr(sys, '_MEIPASS'):
    # Running as bundled exe - resources are in temp extraction folder
    project_root = Path(sys._MEIPASS)
    source_root = Path(sys._MEIPASS)
else:
    # Running from source
    project_root = Path(__file__).parent
    source_root = project_root

# Add the project root to Python path for consistent imports
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from app.ui.main_window import MainWindow
from app.ui.auth_dialog import AuthDialog
from app.core.logging_config import setup_logging
from app.core.supabase_client import supabase_auth, SUPABASE_AVAILABLE
from app.core.settings import settings

logger = logging.getLogger(__name__)


def register_url_scheme():
    """Register filect:// URL scheme in the Windows Registry (HKCU, no admin needed)."""
    try:
        import winreg
        exe_path = sys.executable

        # HKCU\Software\Classes\filect
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\filect")
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:Filect Protocol")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        winreg.CloseKey(key)

        # HKCU\Software\Classes\filect\shell\open\command
        cmd_key = winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\filect\shell\open\command"
        )
        winreg.SetValueEx(cmd_key, "", 0, winreg.REG_SZ, f'"{exe_path}" "%1"')
        winreg.CloseKey(cmd_key)

        logger.info("filect:// URL scheme registered in Windows Registry")
    except ImportError:
        pass  # Not on Windows
    except Exception as e:
        logger.warning(f"Could not register URL scheme: {e}")


def handle_deep_link(url: str, auth_dialog=None) -> bool:
    """
    Handle a filect:// deep link passed via sys.argv on Windows.

    filect://verify?token_hash=XXX&type=signup  — verify email and sign in
    filect://open                                — no-op at launch (window shows normally)

    Returns True if a verify action was attempted.
    """
    if not url.startswith('filect://'):
        return False

    if url.startswith('filect://verify'):
        params = parse_qs(urlparse(url).query)
        token_hash = params.get('token_hash', [None])[0]

        if not token_hash:
            logger.warning("filect://verify received with no token_hash")
            return False

        logger.info("Verifying email token from deep link")
        result = supabase_auth.verify_email_token(token_hash)

        if result.get('success'):
            tokens = supabase_auth.get_session_tokens()
            if tokens:
                settings.set_auth_tokens(
                    tokens['access_token'],
                    tokens['refresh_token'],
                    supabase_auth.user_email or ''
                )
            # If auth dialog is open, trigger subscription check to close it
            if auth_dialog and auth_dialog.isVisible():
                auth_dialog._check_subscription_silent()
            return True
        else:
            logger.error(f"Email verification failed: {result.get('error')}")
            return False

    return False


def check_existing_session():
    """
    Check if there's a valid stored session with active subscription.
    Returns True if user can skip login, False otherwise.
    """
    if not settings.has_stored_session():
        return False

    result = supabase_auth.restore_session(
        settings.auth_access_token,
        settings.auth_refresh_token
    )

    if not result.get('success'):
        settings.clear_auth_tokens()
        return False

    sub_result = supabase_auth.check_subscription()
    if sub_result.get('has_subscription'):
        return True

    return False


def main():
    """Main application entry point."""
    setup_logging()

    # Register filect:// URL scheme on Windows (safe to call every launch)
    register_url_scheme()

    # Check for deep link URL passed as command-line argument (Windows URL scheme)
    deep_link_url = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith('filect://') else None

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Filect")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Filect")

    # Set application icon
    icon_path = source_root / 'resources' / 'iconnn.ico'
    if not icon_path.exists():
        icon_path = source_root / 'resources' / 'icon.png'
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Apply saved theme (dark/light)
    try:
        from app.ui.theme_manager import theme_manager
        theme_manager.apply_theme()
    except Exception as e:
        print(f"Failed to apply theme: {e}")

    # Check if Supabase is available
    if not SUPABASE_AVAILABLE:
        QMessageBox.warning(
            None,
            "Missing Dependency",
            "The 'supabase' package is required.\n\nPlease run: pip install supabase"
        )
        sys.exit(1)

    # Check for existing valid session BEFORE showing auth dialog
    has_valid_session = check_existing_session()

    if not has_valid_session:
        auth_dialog = AuthDialog()

        # If a verify deep link was passed, handle it while the dialog is showing
        if deep_link_url and deep_link_url.startswith('filect://verify'):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: handle_deep_link(deep_link_url, auth_dialog))

        auth_result = auth_dialog.exec()

        if auth_result == 0:
            sub_check = supabase_auth.check_subscription()
            if not sub_check.get('has_subscription'):
                sys.exit(0)
    else:
        # Already logged in — handle filect://open (window shows normally) or verify
        if deep_link_url:
            handle_deep_link(deep_link_url)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
