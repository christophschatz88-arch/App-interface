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
    project_root = Path(sys._MEIPASS)
    source_root = Path(sys._MEIPASS)
else:
    project_root = Path(__file__).parent
    source_root = project_root

sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from app.ui.main_window import MainWindow
from app.ui.auth_dialog import AuthDialog
from app.core.logging_config import setup_logging
from app.core.supabase_client import supabase_auth, SUPABASE_AVAILABLE
from app.core.settings import settings

logger = logging.getLogger(__name__)

# Unique name for the single-instance IPC socket
_INSTANCE_KEY = "FilectApp-SingleInstance"


# ---------------------------------------------------------------------------
# Single-instance helpers
# ---------------------------------------------------------------------------

def _forward_to_existing_instance(message: str) -> bool:
    """
    Try to send a message to an already-running instance.
    Returns True if the message was delivered (an instance was running).
    """
    socket = QLocalSocket()
    socket.connectToServer(_INSTANCE_KEY)
    if socket.waitForConnected(500):
        socket.write(message.encode('utf-8'))
        socket.flush()
        socket.waitForBytesWritten(500)
        socket.disconnectFromServer()
        return True
    return False


def _setup_ipc_server(auth_dialog_holder: list, main_window_holder: list) -> QLocalServer:
    """
    Create the single-instance IPC server so future instances can forward
    URLs to this (primary) instance.
    """
    QLocalServer.removeServer(_INSTANCE_KEY)
    server = QLocalServer()
    server.listen(_INSTANCE_KEY)

    def _on_connection():
        conn = server.nextPendingConnection()
        if not conn:
            return
        conn.waitForReadyRead(500)
        url = conn.readAll().data().decode('utf-8').strip()
        conn.disconnectFromServer()
        logger.info(f"IPC received from secondary instance: {url}")

        if url.startswith('filect://verify'):
            # Verify the email token in this (primary) instance
            dlg = auth_dialog_holder[0]
            handle_deep_link(url, dlg)
        elif url.startswith('filect://'):
            # Bring existing window to front
            win = main_window_holder[0]
            dlg = auth_dialog_holder[0]
            target = win if (win and win.isVisible()) else dlg
            if target:
                target.show()
                target.raise_()
                target.activateWindow()
                # Force Windows to actually foreground the window
                try:
                    import ctypes
                    hwnd = int(target.winId())
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass

    server.newConnection.connect(_on_connection)
    return server


# ---------------------------------------------------------------------------
# URL scheme registration
# ---------------------------------------------------------------------------

def register_url_scheme():
    """Register filect:// URL scheme in the Windows Registry (HKCU, no admin needed)."""
    try:
        import winreg
        exe_path = sys.executable

        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\filect")
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "URL:Filect Protocol")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        winreg.CloseKey(key)

        cmd_key = winreg.CreateKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\filect\shell\open\command"
        )
        winreg.SetValueEx(cmd_key, "", 0, winreg.REG_SZ, f'"{exe_path}" "%1"')
        winreg.CloseKey(cmd_key)

        logger.info("filect:// URL scheme registered in Windows Registry")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not register URL scheme: {e}")


# ---------------------------------------------------------------------------
# Deep link handler
# ---------------------------------------------------------------------------

def handle_deep_link(url: str, auth_dialog=None) -> bool:
    """
    Handle a filect:// deep link.

    filect://verify?token_hash=XXX&type=signup  — verify email and sign in
    filect://open                                — bring window to front (handled by IPC)

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
            if auth_dialog and auth_dialog.isVisible():
                auth_dialog._check_subscription_silent()
            return True
        else:
            logger.error(f"Email verification failed: {result.get('error')}")
            return False

    return False


# ---------------------------------------------------------------------------
# Session check
# ---------------------------------------------------------------------------

def check_existing_session():
    """
    Check if there's a valid stored session with active subscription.
    Returns True if user can skip login, False otherwise.

    A user whose subscription has ``trial_blocked_reason`` set (the
    server-side abuse check rejected their trial — typically duplicate
    card) is NEVER allowed past this gate, even if a stale Stripe
    status briefly reads as 'trialing'. Without this guard a flagged
    user could bypass the auth dialog by closing + reopening the app.
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
    if sub_result.get('trial_blocked_reason'):
        return False
    return sub_result.get('has_subscription', False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""
    setup_logging()
    register_url_scheme()

    deep_link_url = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith('filect://') else None

    app = QApplication(sys.argv)
    app.setApplicationName("Filect")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Filect")

    # If there's a deep link and an instance is already running, forward and exit.
    # This is the key fix: "Open the App" brings the EXISTING window (with pre-filled
    # email) to front instead of opening a blank new instance.
    if deep_link_url and _forward_to_existing_instance(deep_link_url):
        logger.info(f"Forwarded '{deep_link_url}' to existing instance — exiting")
        sys.exit(0)

    # We are the primary instance — set up IPC server for future launches.
    # Use mutable holders so the server callback can reference objects created later.
    auth_dialog_holder = [None]
    main_window_holder = [None]
    ipc_server = _setup_ipc_server(auth_dialog_holder, main_window_holder)  # noqa: F841

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

    if not SUPABASE_AVAILABLE:
        QMessageBox.warning(
            None,
            "Missing Dependency",
            "The 'supabase' package is required.\n\nPlease run: pip install supabase"
        )
        sys.exit(1)

    has_valid_session = check_existing_session()

    if not has_valid_session:
        auth_dialog = AuthDialog()
        auth_dialog_holder[0] = auth_dialog

        if deep_link_url and deep_link_url.startswith('filect://verify'):
            QTimer.singleShot(500, lambda: handle_deep_link(deep_link_url, auth_dialog))

        auth_result = auth_dialog.exec()

        if auth_result == 0:
            sub_check = supabase_auth.check_subscription()
            if not sub_check.get('has_subscription'):
                sys.exit(0)
    else:
        if deep_link_url:
            handle_deep_link(deep_link_url)

    window = MainWindow()
    main_window_holder[0] = window
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
