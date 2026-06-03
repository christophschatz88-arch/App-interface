"""
Authentication dialog for login, signup, and subscription management.
Modern, clean design that adapts to dark/light theme.
"""

import json
import logging
import socket
import threading
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QStackedWidget, QWidget, QMessageBox, QFrame,
    QGraphicsDropShadowEffect, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, QByteArray, QSize
from PySide6.QtGui import QFont, QColor, QIcon, QPixmap, QPainter

from app.core.supabase_client import supabase_auth
from app.core.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Google OAuth loopback (RFC 8252) constants + helpers.
#
# Architecture (do NOT replace with a custom-scheme handler — we tried that
# on Mac for 5+ hours and it doesn't work reliably from packaged Python apps):
#
#   click → bind 127.0.0.1:53682 → open default browser to Supabase OAuth URL
#   with redirect_to=http://127.0.0.1:53682/callback → Google sign-in →
#   Supabase redirects to 127.0.0.1:53682/callback#tokens → we serve a tiny
#   HTML page with JS that reads window.location.hash and POSTs the tokens
#   to /tokens on the same loopback → polling QTimer sees the tokens,
#   installs the session, shuts the server down, brings Filect to the front.
# ---------------------------------------------------------------------------

GOOGLE_OAUTH_PORT = 53682
GOOGLE_OAUTH_CALLBACK_PATH = "/callback"
GOOGLE_OAUTH_TOKENS_PATH = "/tokens"
GOOGLE_OAUTH_PING_PATH = "/ping"
GOOGLE_OAUTH_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes total deadline

# Marker string returned by the /ping endpoint so a second Filect instance
# probing the port can recognise "the other end is also Filect" and show a
# friendlier error message than the generic "port in use".
FILECT_OAUTH_MARKER = "filect-oauth-loopback-v1"

# Inline Google "G" SVG (official branded mark) — rendered to a QIcon below.
_GOOGLE_G_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
<path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/>
<path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/>
<path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/>
<path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571.001-.001.002-.001.003-.002l6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/>
</svg>"""


def _make_google_icon(size: int = 22) -> QIcon:
    """Render the Google G SVG into a QIcon. Falls back to an empty icon if
    QtSvg is unavailable so the button is still functional text-only."""
    try:
        from PySide6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(QByteArray(_GOOGLE_G_SVG))
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        renderer.render(painter)
        painter.end()
        return QIcon(pix)
    except Exception as e:
        logger.debug(f"Google G icon render failed: {e}")
        return QIcon()


# Tiny HTML+JS page served at /callback. Tokens arrive in the URL fragment
# (Supabase OAuth uses fragment-mode for implicit-flow returns), so the
# server itself can't see them — only the browser can. The JS reads
# window.location.hash, posts to /tokens on the same loopback, then shows a
# friendly "you can close this tab" message.
_OAUTH_CALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Signing in to Filect</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, system-ui, sans-serif;
         background: #f6f5f7; color: #222; min-height: 100vh; margin: 0;
         display:flex; align-items:center; justify-content:center; text-align:center; }
  .card { background: white; border-radius: 16px; padding: 36px 48px;
          box-shadow: 0 4px 24px rgba(0,0,0,0.08); max-width: 360px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 8px; color: #2E7D32; }
  p  { font-size: 14px; color: #555; line-height: 1.5; margin: 0; }
</style></head><body>
<div class="card">
  <h1 id="title">Signing in…</h1>
  <p id="msg">Just a moment.</p>
</div>
<script>
(function() {
  var hash = window.location.hash || "";
  if (hash.charAt(0) === "#") hash = hash.substring(1);
  var params = {};
  hash.split("&").forEach(function(part) {
    if (!part) return;
    var idx = part.indexOf("=");
    var k = idx === -1 ? part : part.substring(0, idx);
    var v = idx === -1 ? ""   : part.substring(idx + 1);
    params[decodeURIComponent(k)] = decodeURIComponent(v.replace(/\\+/g, " "));
  });
  fetch("/tokens", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params)
  }).then(function() {
    document.getElementById("title").textContent = "✓ You're signed in";
    document.getElementById("msg").textContent = "You can close this tab and return to Filect.";
  }).catch(function() {
    document.getElementById("title").textContent = "Sign-in problem";
    document.getElementById("msg").textContent = "Please return to Filect and try again.";
  });
})();
</script></body></html>"""


class _ReuseHTTPServer(HTTPServer):
    """``HTTPServer`` with ``allow_reuse_address`` so back-to-back sign-ins
    don't fail to bind while the previous socket is still in ``TIME_WAIT``
    (~60s on Windows). Also needed on Linux/macOS."""
    allow_reuse_address = True


# Module-level flag: set to True when an OAuth flow has just completed
# successfully, so the next MainWindow.showEvent knows it needs to wrestle
# focus away from the browser. MainWindow consumes (reads + clears) the
# flag; AuthDialog only sets it. We use a class-attribute instead of a
# module-level global so the value lives next to the dialog code.

def force_window_to_foreground(widget):
    """Bring ``widget`` to the Windows foreground, bypassing Windows'
    focus-stealing protection.

    Windows blocks ``SetForegroundWindow`` from a process that doesn't
    currently own the foreground (e.g. our Python process while the
    browser is showing the OAuth result). The standard workaround is:

    1. Attach our thread's input queue to the foreground thread (the
       browser). This makes Windows treat us as if we were the
       foreground process for the duration of the attachment.
    2. Briefly flip to ``HWND_TOPMOST`` and back to ``HWND_NOTOPMOST`` —
       this physically hoists the window above the browser without
       leaving it permanently always-on-top.
    3. Then ``SetForegroundWindow`` lands.

    Detaches the input queue at the end so we don't keep stealing
    keystrokes from whichever app was foreground before.

    Pure no-op if any step throws (telemetry-style — focus is a nicety
    and must never crash the calling code).
    """
    try:
        from ctypes import windll, c_ulong, byref
        hwnd = int(widget.winId())

        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_SHOWWINDOW = 0x0040
        flags = SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW

        fg_hwnd = windll.user32.GetForegroundWindow()
        my_thread = windll.kernel32.GetCurrentThreadId()
        fg_thread = 0
        attached = False
        if fg_hwnd:
            pid = c_ulong(0)
            fg_thread = windll.user32.GetWindowThreadProcessId(fg_hwnd, byref(pid))

        try:
            if fg_thread and fg_thread != my_thread:
                attached = bool(windll.user32.AttachThreadInput(fg_thread, my_thread, True))

            # Briefly topmost → not-topmost flip hoists us above the
            # browser without keeping us always-on-top.
            windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
            windll.user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
            windll.user32.SetForegroundWindow(hwnd)
            try:
                widget.raise_()
                widget.activateWindow()
            except Exception:
                pass
        finally:
            if attached:
                windll.user32.AttachThreadInput(fg_thread, my_thread, False)
    except Exception as e:
        logger.debug(f"force_window_to_foreground failed: {e}")
        try:
            widget.raise_()
            widget.activateWindow()
        except Exception:
            pass


class EmailConfirmationDialog(QDialog):
    """
    Modern styled dialog for email confirmation notification.
    Shows after signup to tell user to check their email.
    Draggable and theme-aware.
    """
    
    def __init__(self, parent=None, email: str = ""):
        super().__init__(parent)
        self.email = email
        self._drag_pos = None
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedWidth(440)
        
        self._setup_ui()
    
    def mousePressEvent(self, event):
        """Enable dragging from anywhere on the dialog."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dialog dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        self._drag_pos = None
    
    def _create_step_row(self, number: str, text: str, c: dict) -> QWidget:
        """Create a step row with number badge and text."""
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 8, 0, 8)
        row_layout.setSpacing(14)
        
        # Number badge
        badge = QLabel(number)
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #7C4DFF, stop:1 #9575FF);
            color: white;
            font-size: 13px;
            font-weight: 700;
            border-radius: 14px;
        """)
        row_layout.addWidget(badge)
        
        # Step text
        step_text = QLabel(text)
        step_text.setStyleSheet(f"""
            font-size: 14px;
            color: {c['text']};
            background: transparent;
            font-weight: 500;
        """)
        row_layout.addWidget(step_text, 1)
        
        return row

    def _setup_ui(self):
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Container with shadow
        container = QFrame()
        container.setObjectName("emailConfirmContainer")
        container.setStyleSheet(f"""
            QFrame#emailConfirmContainer {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 20px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 80))
        container.setGraphicsEffect(shadow)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(36, 40, 36, 36)
        container_layout.setSpacing(0)
        
        # Email icon with gradient background
        icon_container = QFrame()
        icon_container.setFixedSize(80, 80)
        icon_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7C4DFF, stop:0.5 #9575FF, stop:1 #B39DFF);
                border-radius: 40px;
            }
        """)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel("✉️")
        icon_label.setStyleSheet("font-size: 36px; background: transparent;")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(icon_label)
        
        # Center the icon
        icon_wrapper = QHBoxLayout()
        icon_wrapper.addStretch()
        icon_wrapper.addWidget(icon_container)
        icon_wrapper.addStretch()
        container_layout.addLayout(icon_wrapper)
        
        container_layout.addSpacing(24)
        
        # Title
        title = QLabel("Check Your Inbox!")
        title.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 700;
            color: {c['text']};
            background: transparent;
        """)
        title.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title)
        
        container_layout.addSpacing(10)
        
        # Description
        desc1 = QLabel("We've sent a confirmation link to:")
        desc1.setStyleSheet(f"""
            font-size: 14px;
            color: {c['text_muted']};
            background: transparent;
        """)
        desc1.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(desc1)
        
        container_layout.addSpacing(4)
        
        # Email label with purple highlight
        email_label = QLabel(self.email)
        email_label.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 600;
            color: #7C4DFF;
            background: transparent;
        """)
        email_label.setAlignment(Qt.AlignCenter)
        email_label.setWordWrap(True)
        container_layout.addWidget(email_label)
        
        container_layout.addSpacing(24)
        
        # Steps card with clearer visual hierarchy
        steps_card = QFrame()
        steps_card.setStyleSheet(f"""
            QFrame {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 14px;
            }}
        """)
        steps_layout = QVBoxLayout(steps_card)
        steps_layout.setContentsMargins(20, 18, 20, 18)
        steps_layout.setSpacing(0)
        
        # Step 1
        step1_container = self._create_step_row("1", "Open your email inbox", c)
        steps_layout.addWidget(step1_container)
        
        # Divider
        divider1 = QFrame()
        divider1.setFixedHeight(1)
        divider1.setStyleSheet(f"background-color: {c['border']}; margin: 10px 0;")
        steps_layout.addWidget(divider1)
        
        # Step 2
        step2_container = self._create_step_row("2", "Click the verification link", c)
        steps_layout.addWidget(step2_container)
        
        # Divider
        divider2 = QFrame()
        divider2.setFixedHeight(1)
        divider2.setStyleSheet(f"background-color: {c['border']}; margin: 10px 0;")
        steps_layout.addWidget(divider2)
        
        # Step 3
        step3_container = self._create_step_row("3", "Come back and sign in", c)
        steps_layout.addWidget(step3_container)
        
        container_layout.addWidget(steps_card)
        
        container_layout.addSpacing(14)
        
        # Spam notice
        spam_notice = QLabel("💡 Don't see it? Check your spam folder")
        spam_notice.setStyleSheet(f"""
            font-size: 12px;
            color: {c['text_muted']};
            background: transparent;
        """)
        spam_notice.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(spam_notice)
        
        container_layout.addSpacing(24)
        
        # OK button
        ok_btn = QPushButton("Got it!")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setMinimumHeight(48)
        ok_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 600;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9575FF, stop:1 #B39DFF);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6A3DE8, stop:1 #7C4DFF);
            }
        """)
        ok_btn.clicked.connect(self.accept)
        container_layout.addWidget(ok_btn)
        
        layout.addWidget(container)
    
    @staticmethod
    def show_confirmation(parent, email: str):
        """Show the email confirmation dialog."""
        dialog = EmailConfirmationDialog(parent, email)
        dialog.exec()


class ForgotPasswordDialog(QDialog):
    """
    Modern styled dialog for password reset.
    Allows user to enter email and receive reset link.
    """
    
    def __init__(self, parent=None, prefill_email: str = ""):
        super().__init__(parent)
        self.prefill_email = prefill_email
        self._sent = False
        self._drag_pos = None
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedWidth(420)
        
        self._setup_ui()
    
    def mousePressEvent(self, event):
        """Enable dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        self._drag_pos = None
    
    def _setup_ui(self):
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Container
        self.container = QFrame()
        self.container.setObjectName("forgotPasswordContainer")
        self.container.setStyleSheet(f"""
            QFrame#forgotPasswordContainer {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 20px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.container.setGraphicsEffect(shadow)
        
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(36, 40, 36, 36)
        self.container_layout.setSpacing(0)
        
        # Icon
        icon_container = QFrame()
        icon_container.setFixedSize(72, 72)
        icon_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #7C4DFF, stop:1 #B39DFF);
                border-radius: 36px;
            }
        """)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel("🔑")
        icon_label.setStyleSheet("font-size: 32px; background: transparent;")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(icon_label)
        
        icon_wrapper = QHBoxLayout()
        icon_wrapper.addStretch()
        icon_wrapper.addWidget(icon_container)
        icon_wrapper.addStretch()
        self.container_layout.addLayout(icon_wrapper)
        
        self.container_layout.addSpacing(24)
        
        # Title
        self.title_label = QLabel("Reset Password")
        self.title_label.setStyleSheet(f"""
            font-size: 22px;
            font-weight: 700;
            color: {c['text']};
            background: transparent;
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.container_layout.addWidget(self.title_label)
        
        self.container_layout.addSpacing(8)
        
        # Description
        self.desc_label = QLabel("Enter your email address and we'll send you a link to reset your password.")
        self.desc_label.setStyleSheet(f"""
            font-size: 14px;
            color: {c['text_muted']};
            background: transparent;
            line-height: 1.4;
        """)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setWordWrap(True)
        self.container_layout.addWidget(self.desc_label)
        
        self.container_layout.addSpacing(24)
        
        # Email input
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter your email address")
        self.email_input.setText(self.prefill_email)
        self.email_input.setMinimumHeight(52)
        self.email_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 12px;
                padding: 12px 16px;
                font-size: 14px;
                color: {c['text']};
            }}
            QLineEdit:focus {{
                border: 2px solid #7C4DFF;
            }}
            QLineEdit::placeholder {{
                color: {c['text_muted']};
            }}
        """)
        self.container_layout.addWidget(self.email_input)
        
        self.container_layout.addSpacing(8)
        
        # Error/status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            font-size: 13px;
            color: #FF6B6B;
            background: transparent;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(20)
        self.container_layout.addWidget(self.status_label)
        
        self.container_layout.addSpacing(16)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setMinimumHeight(48)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {c['text_muted']};
                border: 1px solid {c['border']};
                border-radius: 12px;
                font-size: 14px;
                font-weight: 500;
                padding: 12px 24px;
            }}
            QPushButton:hover {{
                background: {c['card']};
                color: {c['text']};
            }}
        """)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        # Send button
        self.send_btn = QPushButton("Send Reset Link")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setMinimumHeight(48)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9575FF, stop:1 #B39DFF);
            }
            QPushButton:disabled {
                background: #555;
                color: #888;
            }
        """)
        self.send_btn.clicked.connect(self._send_reset)
        btn_layout.addWidget(self.send_btn)
        
        self.container_layout.addLayout(btn_layout)
        
        layout.addWidget(self.container)
        
        # Connect enter key
        self.email_input.returnPressed.connect(self._send_reset)
    
    def _send_reset(self):
        """Send password reset email."""
        email = self.email_input.text().strip()
        
        if not email:
            self.status_label.setText("Please enter your email address")
            self.status_label.setStyleSheet("font-size: 13px; color: #FF6B6B; background: transparent;")
            return
        
        if '@' not in email:
            self.status_label.setText("Please enter a valid email address")
            self.status_label.setStyleSheet("font-size: 13px; color: #FF6B6B; background: transparent;")
            return
        
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Sending...")
        self.status_label.setText("")
        
        # Send reset email via Supabase
        result = supabase_auth.reset_password(email)
        
        if result.get('success'):
            self._show_success(email)
        else:
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send Reset Link")
            self.status_label.setText(result.get('error', 'Failed to send reset email'))
            self.status_label.setStyleSheet("font-size: 13px; color: #FF6B6B; background: transparent;")
    
    def _show_success(self, email: str):
        """Transform dialog to show success state."""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        self._sent = True
        
        # Update title and description
        self.title_label.setText("Check Your Email!")
        self.desc_label.setText(
            f"We've sent a password reset link to:\n\n"
            f"{email}\n\n"
            f"Click the link in the email to create a new password."
        )
        self.desc_label.setStyleSheet(f"""
            font-size: 14px;
            color: {c['text_muted']};
            background: transparent;
            line-height: 1.5;
        """)
        
        # Update status to success
        self.status_label.setText("💡 Don't see it? Check your spam folder")
        self.status_label.setStyleSheet(f"font-size: 12px; color: {c['text_muted']}; background: transparent;")
        
        # Hide email input
        self.email_input.hide()
        
        # Update buttons
        self.cancel_btn.hide()
        self.send_btn.setText("Done")
        self.send_btn.setEnabled(True)
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.accept)
    
    @staticmethod
    def show_dialog(parent, prefill_email: str = ""):
        """Show the forgot password dialog."""
        dialog = ForgotPasswordDialog(parent, prefill_email)
        dialog.exec()


class AuthDialog(QDialog):
    """Authentication dialog for user login/signup and subscription."""
    
    # Signals
    auth_successful = Signal()
    
    # Cross-window flag: set to True when an OAuth flow has just completed
    # successfully. The next MainWindow.showEvent reads + clears it and
    # forces foreground if True. We need this because by the time the
    # auth dialog accepts and MainWindow is created+shown, Windows has
    # often handed foreground back to the browser, so MainWindow would
    # otherwise appear hidden behind the browser tab.
    _pending_foreground_after_oauth = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filect")
        # 720px fits the email + password + Sign In + or-divider + Continue
        # with Google + footer on the login page without clipping the
        # magnifying-glass logo at the top, even on smaller laptop screens.
        # We rely on the addStretch() in _create_login_page to absorb any
        # leftover vertical space cleanly.
        self.setFixedSize(460, 720)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setModal(True)
        self.setObjectName("authDialog")
        self._pending_email = ""

        self._setup_ui()
        self._setup_connections()
        
        # Subscription polling timer
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_subscription)
        self._poll_count = 0
        
        # Try to restore session on init
        QTimer.singleShot(100, self._try_restore_session)
    
    def showEvent(self, event):
        """Apply dark/light title bar when dialog is shown."""
        super().showEvent(event)
        from app.ui.theme_manager import apply_titlebar_theme
        apply_titlebar_theme(self)
        # Always reset the Google button when the dialog re-opens — without
        # this, signing out and re-showing the dialog leaves the button
        # stuck on "Waiting for Google sign-in…" if a previous attempt was
        # mid-flight when the dialog last closed.
        if hasattr(self, "google_button"):
            self._reset_google_button()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main container with card styling. Reduced vertical padding from 40
        # → 24 so the login page (which now has an extra row for Google)
        # fits in 720px without Qt squeezing the email/password input
        # containers below their min height.
        self.container = QFrame()
        self.container.setObjectName("authContainer")
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(40, 24, 40, 24)
        container_layout.setSpacing(0)
        
        # Header with logo
        header_layout = QVBoxLayout()
        header_layout.setSpacing(12)
        
        # Logo icon
        logo_label = QLabel("🔍")
        logo_label.setObjectName("authLogo")
        logo_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo_label)
        
        # Title
        self.title_label = QLabel("File Search Assistant")
        self.title_label.setObjectName("authTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.title_label)
        
        # Subtitle
        self.subtitle_label = QLabel("Sign in to your account")
        self.subtitle_label.setObjectName("authSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.subtitle_label)
        
        container_layout.addLayout(header_layout)
        container_layout.addSpacing(20)  # was 32 — tightened to fit Google row
        
        # Stacked widget for different views
        self.stack = QStackedWidget()
        self.stack.setObjectName("authStack")
        container_layout.addWidget(self.stack)
        
        # Create pages
        self._create_login_page()
        self._create_signup_page()
        self._create_subscribe_page()
        self._create_verify_page()

        layout.addWidget(self.container)
        
        # Start with login page
        self.stack.setCurrentIndex(0)
    
    def _create_input_field(self, label_text, placeholder, is_password=False):
        """Create a styled input field with label."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        label = QLabel(label_text)
        label.setObjectName("authLabel")
        layout.addWidget(label)
        
        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setObjectName("authInput")
        input_field.setMinimumHeight(52)
        if is_password:
            input_field.setEchoMode(QLineEdit.Password)
        
        layout.addWidget(input_field)
        
        return container, input_field
    
    def _create_login_page(self):
        """Create the login page."""
        page = QWidget()
        page.setObjectName("authPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        # 12px row spacing (down from 20) — with 10+ rows, 20px of spacing
        # between each adds up to ~200px and pushes the page beyond what a
        # 720px dialog can show without clipping the input containers.
        layout.setSpacing(12)
        
        # Email field
        email_container, self.login_email = self._create_input_field(
            "Email", "Enter your email address"
        )
        layout.addWidget(email_container)
        
        # Password field
        password_container, self.login_password = self._create_input_field(
            "Password", "Enter your password", is_password=True
        )
        layout.addWidget(password_container)
        
        layout.addSpacing(8)
        
        # Login button
        self.login_button = QPushButton("Sign In")
        self.login_button.setObjectName("primaryButton")
        self.login_button.setMinimumHeight(52)
        self.login_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.login_button)

        # "or" divider between the email/password form and the Google button.
        or_row = QHBoxLayout()
        or_row.setSpacing(12)
        or_line_left = QFrame()
        or_line_left.setFrameShape(QFrame.HLine)
        or_line_left.setObjectName("authDivider")
        or_line_left.setFixedHeight(1)
        or_line_right = QFrame()
        or_line_right.setFrameShape(QFrame.HLine)
        or_line_right.setObjectName("authDivider")
        or_line_right.setFixedHeight(1)
        or_label = QLabel("or")
        or_label.setObjectName("authSwitchLabel")
        or_label.setAlignment(Qt.AlignCenter)
        or_row.addWidget(or_line_left, 1)
        or_row.addWidget(or_label, 0)
        or_row.addWidget(or_line_right, 1)
        layout.addSpacing(6)
        layout.addLayout(or_row)
        layout.addSpacing(6)

        # Continue with Google button (loopback OAuth, see _sign_in_with_google).
        self.google_button = QPushButton("  Continue with Google")
        self.google_button.setObjectName("googleButton")
        self.google_button.setMinimumHeight(52)
        self.google_button.setCursor(Qt.PointingHandCursor)
        self.google_button.setIcon(_make_google_icon(22))
        self.google_button.setIconSize(QSize(22, 22))
        # Inline styling so we don't need to touch the global QSS — keeps
        # the white-bg / dark-text Google branding consistent in both
        # light and dark themes.
        self.google_button.setStyleSheet("""
            QPushButton#googleButton {
                background-color: white;
                color: #1F1F1F;
                border: 1px solid #DADCE0;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 500;
                padding: 0 16px;
                text-align: center;
            }
            QPushButton#googleButton:hover {
                background-color: #F8F9FA;
                border-color: #C8CDD3;
            }
            QPushButton#googleButton:pressed {
                background-color: #F1F3F4;
            }
            QPushButton#googleButton:disabled {
                background-color: #F5F5F5;
                color: #9E9E9E;
                border-color: #E0E0E0;
            }
        """)
        layout.addWidget(self.google_button)

        # Error label
        self.login_error = QLabel("")
        self.login_error.setObjectName("errorLabel")
        self.login_error.setAlignment(Qt.AlignCenter)
        self.login_error.setWordWrap(True)
        self.login_error.setMinimumHeight(24)
        layout.addWidget(self.login_error)

        # Forgot password link
        forgot_layout = QHBoxLayout()
        forgot_layout.addStretch()
        self.forgot_password_btn = QPushButton("Forgot password?")
        self.forgot_password_btn.setObjectName("linkButton")
        self.forgot_password_btn.setCursor(Qt.PointingHandCursor)
        forgot_layout.addWidget(self.forgot_password_btn)
        forgot_layout.addStretch()
        layout.addLayout(forgot_layout)
        
        layout.addStretch()
        
        # Divider
        divider = QFrame()
        divider.setObjectName("authDivider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        
        layout.addSpacing(16)
        
        # Switch to signup
        switch_layout = QHBoxLayout()
        switch_layout.setSpacing(4)
        switch_label = QLabel("Don't have an account?")
        switch_label.setObjectName("authSwitchLabel")
        self.to_signup_button = QPushButton("Create account")
        self.to_signup_button.setObjectName("linkButton")
        self.to_signup_button.setCursor(Qt.PointingHandCursor)
        switch_layout.addStretch()
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(self.to_signup_button)
        switch_layout.addStretch()
        layout.addLayout(switch_layout)
        
        self.stack.addWidget(page)
    
    def _create_signup_page(self):
        """Create the signup page."""
        page = QWidget()
        page.setObjectName("authPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Email field
        email_container, self.signup_email = self._create_input_field(
            "Email", "Enter your email address"
        )
        layout.addWidget(email_container)
        
        # Password field
        password_container, self.signup_password = self._create_input_field(
            "Password", "Create a password (min 6 chars)", is_password=True
        )
        layout.addWidget(password_container)
        
        # Confirm Password field
        confirm_container, self.signup_confirm = self._create_input_field(
            "Confirm Password", "Confirm your password", is_password=True
        )
        layout.addWidget(confirm_container)
        
        layout.addSpacing(4)
        
        # Signup button
        self.signup_button = QPushButton("Create Account")
        self.signup_button.setObjectName("primaryButton")
        self.signup_button.setMinimumHeight(52)
        self.signup_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.signup_button)
        
        # Error label
        self.signup_error = QLabel("")
        self.signup_error.setObjectName("errorLabel")
        self.signup_error.setAlignment(Qt.AlignCenter)
        self.signup_error.setWordWrap(True)
        self.signup_error.setMinimumHeight(24)
        layout.addWidget(self.signup_error)
        
        layout.addStretch()
        
        # Divider
        divider = QFrame()
        divider.setObjectName("authDivider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        layout.addWidget(divider)
        
        layout.addSpacing(16)
        
        # Switch to login
        switch_layout = QHBoxLayout()
        switch_layout.setSpacing(4)
        switch_label = QLabel("Already have an account?")
        switch_label.setObjectName("authSwitchLabel")
        self.to_login_button = QPushButton("Sign in")
        self.to_login_button.setObjectName("linkButton")
        self.to_login_button.setCursor(Qt.PointingHandCursor)
        switch_layout.addStretch()
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(self.to_login_button)
        switch_layout.addStretch()
        layout.addLayout(switch_layout)
        
        self.stack.addWidget(page)
    
    def _create_subscribe_page(self):
        """Create the subscription page."""
        page = QWidget()
        page.setObjectName("authPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Welcome message
        self.welcome_label = QLabel("Welcome!")
        self.welcome_label.setObjectName("authWelcome")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.welcome_label)
        
        layout.addSpacing(8)
        
        # Subscription card
        sub_card = QFrame()
        sub_card.setObjectName("subscriptionCard")
        card_layout = QVBoxLayout(sub_card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)
        
        # Plan badge
        plan_badge = QLabel("CHOOSE YOUR PLAN")
        plan_badge.setObjectName("planBadge")
        plan_badge.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(plan_badge)

        # Price
        price_layout = QHBoxLayout()
        price_layout.setAlignment(Qt.AlignCenter)
        price_layout.setSpacing(4)

        price_amount = QLabel("from $15")
        price_amount.setObjectName("priceAmount")
        price_period = QLabel("/ mo")
        price_period.setObjectName("pricePeriod")
        
        price_layout.addWidget(price_amount)
        price_layout.addWidget(price_period)
        card_layout.addLayout(price_layout)
        
        # Features - just 2 to fit the space
        features = [
            "✓  AI-powered search & organization",
            "✓  Auto-indexing, OCR & Vision AI",
        ]
        
        features_container = QWidget()
        features_container.setObjectName("featuresContainer")
        features_layout = QVBoxLayout(features_container)
        features_layout.setContentsMargins(8, 12, 8, 8)
        features_layout.setSpacing(8)
        
        for feature in features:
            feat_label = QLabel(feature)
            feat_label.setObjectName("featureLabel")
            feat_label.setMinimumHeight(24)
            features_layout.addWidget(feat_label)
        
        card_layout.addWidget(features_container)
        layout.addWidget(sub_card)
        
        layout.addSpacing(16)
        
        # Subscribe button
        self.subscribe_button = QPushButton("View plans && subscribe")
        self.subscribe_button.setObjectName("primaryButton")
        self.subscribe_button.setMinimumHeight(52)
        self.subscribe_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.subscribe_button)
        
        # Status label
        self.sub_status = QLabel("")
        self.sub_status.setObjectName("statusLabel")
        self.sub_status.setAlignment(Qt.AlignCenter)
        self.sub_status.setWordWrap(True)
        self.sub_status.setMinimumHeight(24)
        layout.addWidget(self.sub_status)
        
        layout.addStretch()
        
        # Logout
        self.logout_button = QPushButton("Sign Out")
        self.logout_button.setObjectName("linkButton")
        self.logout_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.logout_button, alignment=Qt.AlignCenter)

        self.stack.addWidget(page)

    def _create_verify_page(self):
        """Email code-verification page, shown after signup.

        The confirmation email contains BOTH a 6-digit code (entered here) and a
        "Verify account" button (filect:// deep link, handled in main.py). If the
        user clicks the email button instead, the deep-link handler verifies and
        calls _check_subscription_silent(), which advances off this page on its
        own — so the code is never required in that case.
        """
        page = QWidget()
        page.setObjectName("authPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        header = QLabel("Check your email")
        header.setObjectName("authWelcome")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        self.verify_message = QLabel(
            "We sent a 6-digit code to your email. Enter it below to verify your "
            "account — or just click the “Verify account” button in that email."
        )
        self.verify_message.setObjectName("authSwitchLabel")
        self.verify_message.setAlignment(Qt.AlignCenter)
        self.verify_message.setWordWrap(True)
        layout.addWidget(self.verify_message)

        layout.addSpacing(8)

        code_container, self.verify_code = self._create_input_field(
            "Verification code", "Enter 6-digit code"
        )
        layout.addWidget(code_container)

        self.verify_button = QPushButton("Verify && Continue")
        self.verify_button.setObjectName("primaryButton")
        self.verify_button.setMinimumHeight(52)
        self.verify_button.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.verify_button)

        self.verify_error = QLabel("")
        self.verify_error.setObjectName("errorLabel")
        self.verify_error.setAlignment(Qt.AlignCenter)
        self.verify_error.setWordWrap(True)
        self.verify_error.setMinimumHeight(24)
        layout.addWidget(self.verify_error)

        layout.addStretch()

        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        self.resend_code_button = QPushButton("Resend code")
        self.resend_code_button.setObjectName("linkButton")
        self.resend_code_button.setCursor(Qt.PointingHandCursor)
        self.verify_back_button = QPushButton("Back to sign in")
        self.verify_back_button.setObjectName("linkButton")
        self.verify_back_button.setCursor(Qt.PointingHandCursor)
        bottom.addStretch()
        bottom.addWidget(self.resend_code_button)
        bottom.addWidget(self.verify_back_button)
        bottom.addStretch()
        layout.addLayout(bottom)

        self.stack.addWidget(page)

    def _setup_connections(self):
        """Set up signal connections."""
        # Login page
        self.login_button.clicked.connect(self._do_login)
        self.login_password.returnPressed.connect(self._do_login)
        self.to_signup_button.clicked.connect(self._go_to_signup)
        self.forgot_password_btn.clicked.connect(self._show_forgot_password)
        self.google_button.clicked.connect(self._sign_in_with_google)

        # Signup page
        self.signup_button.clicked.connect(self._do_signup)
        self.signup_confirm.returnPressed.connect(self._do_signup)
        self.to_login_button.clicked.connect(self._go_to_login)

        # Subscribe page
        self.subscribe_button.clicked.connect(self._open_checkout)
        self.logout_button.clicked.connect(self._do_logout)

        # Verify page
        self.verify_button.clicked.connect(self._do_verify_code)
        self.verify_code.returnPressed.connect(self._do_verify_code)
        self.resend_code_button.clicked.connect(self._resend_code)
        self.verify_back_button.clicked.connect(self._go_to_login)
    
    def _show_forgot_password(self):
        """Show the forgot password dialog."""
        # Pre-fill with email from login field if available
        email = self.login_email.text().strip()
        ForgotPasswordDialog.show_dialog(self, email)
    
    def _go_to_signup(self):
        """Navigate to signup page."""
        self.title_label.setText("Create Account")
        self.subtitle_label.setText("Sign up to get started")
        self.stack.setCurrentIndex(1)
        self.setFixedSize(460, 760)

    def _go_to_login(self):
        """Navigate to login page."""
        self.title_label.setText("File Search Assistant")
        self.subtitle_label.setText("Sign in to your account")
        self.stack.setCurrentIndex(0)
        self.setFixedSize(460, 720)  # fits Continue with Google on laptop screens

    def _go_to_verify(self, email):
        """Show the code-verification page after signup."""
        self._pending_email = email
        self.title_label.setText("Verify your email")
        self.subtitle_label.setText("One quick step to finish")
        self.verify_message.setText(
            f"We sent a 6-digit code to {email}. Enter it below — or click the "
            f"“Verify account” button in that email and we’ll finish it for you."
        )
        self.verify_code.clear()
        self.verify_error.setText("")
        self.stack.setCurrentIndex(3)
        self.setFixedSize(460, 680)
        self.verify_code.setFocus()

    def _do_verify_code(self):
        """Verify the 6-digit signup code, then continue."""
        code = self.verify_code.text().strip()
        if not code:
            self.verify_error.setText("Enter the code from your email")
            return

        self.verify_button.setEnabled(False)
        self.verify_button.setText("Verifying...")

        result = supabase_auth.verify_signup_code(self._pending_email, code)

        self.verify_button.setEnabled(True)
        self.verify_button.setText("Verify && Continue")

        if result.get('success'):
            self.verify_error.setText("")
            tokens = supabase_auth.get_session_tokens()
            if tokens:
                settings.set_auth_tokens(
                    tokens['access_token'],
                    tokens['refresh_token'],
                    self._pending_email
                )
            self._check_subscription_silent()
        else:
            self.verify_error.setText("That code is incorrect or expired. Check your email and try again.")

    def _resend_code(self):
        """Resend the signup confirmation code."""
        result = supabase_auth.resend_signup_code(self._pending_email)
        if result.get('success'):
            self.verify_error.setText("")
            self.verify_message.setText(f"New code sent to {self._pending_email}. Enter it below.")
        else:
            self.verify_error.setText("Could not resend the code. Please try again.")

    def _try_restore_session(self):
        """Try to restore a previous session."""
        if settings.has_stored_session():
            result = supabase_auth.restore_session(
                settings.auth_access_token,
                settings.auth_refresh_token
            )
            
            if result.get('success'):
                logger.info("Session restored successfully")
                self._check_subscription_silent()
            else:
                settings.clear_auth_tokens()
    
    def _do_login(self):
        """Handle login button click."""
        email = self.login_email.text().strip()
        password = self.login_password.text()
        
        if not email or not password:
            self.login_error.setText("Please enter email and password")
            return
        
        self.login_button.setEnabled(False)
        self.login_button.setText("Signing in...")
        
        result = supabase_auth.sign_in(email, password)
        
        self.login_button.setEnabled(True)
        self.login_button.setText("Sign In")
        
        if result.get('success'):
            self.login_error.setText("")
            tokens = supabase_auth.get_session_tokens()
            if tokens:
                settings.set_auth_tokens(
                    tokens['access_token'],
                    tokens['refresh_token'],
                    email
                )
            self._check_subscription_silent()
        else:
            error = result.get('error', 'Login failed')
            self.login_error.setText(error)

    # ----- Google OAuth (loopback flow) ---------------------------------

    def _sign_in_with_google(self):
        """Entry point for the 'Continue with Google' button.

        Spins up a one-shot HTTP server on 127.0.0.1:53682, opens the
        default browser to Supabase's Google OAuth URL with redirect_to
        pointing at our loopback, then polls (via QTimer on the main
        thread) for the tokens that the browser will POST back via
        :data:`_OAUTH_CALLBACK_HTML`. See the module-level docstring for
        the full architecture.

        Click semantics:
        - Idle state → start the flow (show "✕ Cancel sign-in" label).
        - In-progress state → cancel the flow (close server, reset button).
          This is the recovery path when the user closes the browser tab
          before completing sign-in; without it they'd be stuck staring
          at "Waiting…" for the full 5-minute timeout.
        """
        from app.core.supabase_client import SUPABASE_URL

        # A click while a flow is in progress = the user wants to cancel.
        if getattr(self, "_google_oauth_in_progress", False):
            self._cancel_google_signin()
            return

        # Pre-flight bind check on a throwaway socket — gives us a clean
        # error message if the port is genuinely taken, without leaking
        # a half-started server.
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            test_sock.bind(("127.0.0.1", GOOGLE_OAUTH_PORT))
        except OSError as e:
            test_sock.close()
            # Distinguish "another Filect window" from "unrelated app" via
            # the /ping marker — the message is way friendlier when we can
            # tell the user *what* is holding the port.
            if self._port_is_held_by_filect():
                self.login_error.setText(
                    "Another Filect window is signing in with Google. "
                    "Finish or cancel that one first."
                )
                logger.warning(
                    "[GOOGLE] Pre-flight bind failed — another Filect "
                    "instance holds the port"
                )
            else:
                self.login_error.setText(
                    f"Port {GOOGLE_OAUTH_PORT} is in use by another app. "
                    "Try again in a minute, or close any other apps using it."
                )
                logger.error(f"[GOOGLE] Pre-flight bind failed: {e}")
            return
        finally:
            try:
                test_sock.close()
            except Exception:
                pass

        # Reset per-attempt state. ``_google_tokens`` is the shared mailbox
        # between the HTTP handler thread (writer) and the main-thread
        # QTimer (reader). Plain attribute assignment is fine here — we
        # only care about visibility of the final reference, and the GIL
        # gives us that.
        self._google_tokens = None

        callback_url = f"http://127.0.0.1:{GOOGLE_OAUTH_PORT}{GOOGLE_OAUTH_CALLBACK_PATH}"
        # prompt=select_account forces Google to always show the account
        # chooser, even when the browser is already signed into exactly one
        # Google account. Supabase /auth/v1/authorize forwards the ``prompt``
        # query param straight to the provider's authorize URL.
        oauth_url = (
            f"{SUPABASE_URL}/auth/v1/authorize?provider=google"
            f"&redirect_to={urllib.parse.quote(callback_url, safe='')}"
            f"&prompt=select_account"
        )
        logger.info(f"[GOOGLE] Starting loopback OAuth, callback={callback_url}")

        dialog_self = self  # captured by the inner Handler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                # Silence the noisy default access log — we have our own.
                pass

            def do_GET(self):
                if self.path == GOOGLE_OAUTH_PING_PATH:
                    # Identity probe — a second Filect instance hitting
                    # /ping recognises this marker and shows a friendlier
                    # "another Filect window is signing in" error instead
                    # of the generic "port in use" message.
                    body = FILECT_OAUTH_MARKER.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith(GOOGLE_OAUTH_CALLBACK_PATH):
                    body = _OAUTH_CALLBACK_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == GOOGLE_OAUTH_TOKENS_PATH:
                    length = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(length).decode("utf-8", errors="replace")
                    try:
                        payload = json.loads(raw) if raw else {}
                    except Exception:
                        payload = {}
                    # Hand off to the main-thread poller.
                    dialog_self._google_tokens = payload
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_response(404)
                    self.end_headers()

        try:
            self._google_server = _ReuseHTTPServer(
                ("127.0.0.1", GOOGLE_OAUTH_PORT), Handler
            )
        except OSError as e:
            self.login_error.setText(
                "Couldn't start sign-in server. Try again in a minute."
            )
            logger.error(f"[GOOGLE] Server bind failed after pre-flight: {e}")
            return

        self._google_server_thread = threading.Thread(
            target=self._google_server.serve_forever, daemon=True
        )
        self._google_server_thread.start()

        # Mark the flow in-progress and flip the button into "cancel
        # mode" — keep it ENABLED so a second click cancels rather than
        # being a dead click for 5 minutes.
        self._google_oauth_in_progress = True
        self.google_button.setEnabled(True)
        self.google_button.setText("  ✕ Cancel sign-in")
        self.login_error.setText("")

        try:
            webbrowser.open(oauth_url)
        except Exception as e:
            logger.error(f"[GOOGLE] Could not open browser: {e}")
            self.login_error.setText("Couldn't open your browser. Try again.")
            self._stop_google_server()
            self._reset_google_button()
            return

        # Poll for tokens at 200ms — gives the JS callback time to land
        # without a perceptible delay in unlocking the UI.
        if not hasattr(self, "_google_poll_timer") or self._google_poll_timer is None:
            self._google_poll_timer = QTimer(self)
            self._google_poll_timer.timeout.connect(self._google_check_tokens)
        self._google_poll_timer.start(200)

        # 5-minute total deadline. If the user abandons the browser tab,
        # we want the button back to a usable state.
        if not hasattr(self, "_google_timeout_timer") or self._google_timeout_timer is None:
            self._google_timeout_timer = QTimer(self)
            self._google_timeout_timer.setSingleShot(True)
            self._google_timeout_timer.timeout.connect(self._google_timeout)
        self._google_timeout_timer.start(GOOGLE_OAUTH_TIMEOUT_MS)

    def _google_check_tokens(self):
        """QTimer callback — runs on the main thread every 200ms.

        Inspects ``self._google_tokens`` (written by the loopback HTTP
        handler thread) and installs the session if tokens are present.
        """
        tokens = self._google_tokens
        if not tokens:
            return

        # Stop both timers + server before doing anything that could throw.
        if hasattr(self, "_google_poll_timer") and self._google_poll_timer is not None:
            self._google_poll_timer.stop()
        if hasattr(self, "_google_timeout_timer") and self._google_timeout_timer is not None:
            self._google_timeout_timer.stop()
        self._stop_google_server()

        access = tokens.get("access_token")
        refresh = tokens.get("refresh_token")
        if not access or not refresh:
            err = (
                tokens.get("error_description")
                or tokens.get("error")
                or "No tokens received from Google."
            )
            logger.error(f"[GOOGLE] Callback returned no tokens: {tokens}")
            self.login_error.setText(f"Google sign-in failed: {err}")
            self._reset_google_button()
            return

        result = supabase_auth.restore_session(access, refresh)
        if not result.get("success"):
            self.login_error.setText(
                f"Couldn't restore Google session: {result.get('error', 'unknown error')}"
            )
            logger.error(f"[GOOGLE] restore_session failed: {result}")
            self._reset_google_button()
            return

        email = supabase_auth.user_email or ""
        try:
            settings.set_auth_tokens(access, refresh, email)
        except Exception as e:
            logger.warning(f"[GOOGLE] Could not persist tokens to settings: {e}")

        logger.info(f"[GOOGLE] Sign-in complete: {email}")
        self._reset_google_button()

        # Subscription gate. The foreground push is deliberately INSIDE the
        # subscribed branch — for unsubscribed users we're about to open
        # the pricing page in their browser, and pulling Filect to the
        # front would steal focus from the page they need to interact
        # with.
        sub_result = supabase_auth.check_subscription()
        if sub_result.get("has_subscription"):
            logger.info("[GOOGLE] Active subscription found — closing auth dialog")
            # Subscribed user lands in the main app — yank Filect to the
            # front so they don't have to alt-tab away from the browser.
            self._bring_app_to_foreground()
            self.auth_successful.emit()
            self.accept()
        else:
            # No subscription → open the pricing page immediately (no
            # extra click) and keep the dialog open as a polling fallback
            # so it finishes itself when the purchase completes. We do
            # NOT call _bring_app_to_foreground here — the user just needs
            # to pick a plan in the browser, and stealing focus would
            # make them alt-tab back. The dialog re-shows itself
            # automatically when the user returns to it.
            logger.info("[GOOGLE] No subscription — opening pricing page (keeping browser in front)")
            self._show_subscribe_page()
            try:
                supabase_auth.open_web_pricing()
            except Exception as e:
                logger.error(f"[GOOGLE] open_web_pricing failed: {e}")
            self._poll_count = 0
            self._poll_timer.start(3000)

    def _google_timeout(self):
        """Five-minute deadline elapsed — clean up and tell the user."""
        logger.warning("[GOOGLE] OAuth flow timed out after 5 minutes")
        if hasattr(self, "_google_poll_timer") and self._google_poll_timer is not None:
            self._google_poll_timer.stop()
        self._stop_google_server()
        self._reset_google_button()
        self.login_error.setText("Google sign-in timed out. Try again.")

    def _port_is_held_by_filect(self) -> bool:
        """Probe ``127.0.0.1:53682/ping`` and check for the Filect marker.

        Used after a pre-flight bind failure to tell apart these two cases:

        - **Another Filect window** is mid-sign-in. Their server returns
          ``FILECT_OAUTH_MARKER`` from ``/ping``. We show "finish the
          other window first".
        - **An unrelated app** has the port. Either the connection
          refuses, times out, or returns different content. We show
          "try again in a minute".

        Returns False on any probe failure — the caller should treat
        that as "unrelated app" since we can't prove it's us.
        """
        try:
            import urllib.request
            url = f"http://127.0.0.1:{GOOGLE_OAUTH_PORT}{GOOGLE_OAUTH_PING_PATH}"
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip()
            return body == FILECT_OAUTH_MARKER
        except Exception as e:
            logger.debug(f"[GOOGLE] /ping probe failed (not a Filect server): {e}")
            return False

    def _stop_google_server(self):
        """Shut down the loopback HTTP server cleanly if it's running."""
        server = getattr(self, "_google_server", None)
        if server is None:
            return
        try:
            server.shutdown()
        except Exception as e:
            logger.debug(f"[GOOGLE] server.shutdown() raised: {e}")
        try:
            server.server_close()
        except Exception as e:
            logger.debug(f"[GOOGLE] server.server_close() raised: {e}")
        self._google_server = None

    def _reset_google_button(self):
        """Restore the Google button to its idle state. MUST be called
        from every exit path (success, timeout, error, cancel, dialog
        show) otherwise the button gets stuck in 'Cancel sign-in' mode
        and a fresh click would mis-route to ``_cancel_google_signin``.
        """
        self._google_oauth_in_progress = False
        if hasattr(self, "google_button"):
            self.google_button.setEnabled(True)
            self.google_button.setText("  Continue with Google")

    def _cancel_google_signin(self):
        """User-initiated cancel of an in-progress OAuth flow.

        Common trigger: the user closed the browser tab without finishing
        sign-in, came back to Filect, and clicked the button again. Tears
        down the loopback server, stops both timers, and resets the
        button so the next click starts a fresh flow.
        """
        logger.info("[GOOGLE] User canceled OAuth flow")
        if hasattr(self, "_google_poll_timer") and self._google_poll_timer is not None:
            self._google_poll_timer.stop()
        if hasattr(self, "_google_timeout_timer") and self._google_timeout_timer is not None:
            self._google_timeout_timer.stop()
        self._stop_google_server()
        self._reset_google_button()
        # Clear any error left from a previous attempt so the form looks
        # idle (we don't want a "Google sign-in timed out" message from a
        # previous click hanging around).
        if hasattr(self, "login_error"):
            self.login_error.setText("")

    def _bring_app_to_foreground(self):
        """Pull the current Filect window to the front after OAuth completes.

        Uses :func:`force_window_to_foreground` for the heavy lifting. Also
        sets the class-level ``_pending_foreground_after_oauth`` flag so
        that whatever window is shown next (typically MainWindow, created
        after this dialog accepts) can repeat the foreground push — by the
        time the dialog hides, the browser usually has reclaimed
        foreground and the new MainWindow would otherwise come up hidden
        behind it.
        """
        target = self.parent() if self.parent() is not None else self
        force_window_to_foreground(target)
        AuthDialog._pending_foreground_after_oauth = True

    def _do_signup(self):
        """Handle signup button click."""
        email = self.signup_email.text().strip()
        password = self.signup_password.text()
        confirm = self.signup_confirm.text()
        
        if not email or not password:
            self.signup_error.setText("Please fill all fields")
            return
        
        if password != confirm:
            self.signup_error.setText("Passwords don't match")
            return
        
        if len(password) < 6:
            self.signup_error.setText("Password must be at least 6 characters")
            return
        
        self.signup_button.setEnabled(False)
        self.signup_button.setText("Creating account...")
        
        result = supabase_auth.sign_up(email, password)
        
        self.signup_button.setEnabled(True)
        self.signup_button.setText("Create Account")
        
        if result.get('success'):
            self.signup_error.setText("")
            
            if result.get('needs_confirmation'):
                self._go_to_verify(email)
            else:
                # Account created and auto-logged in
                tokens = supabase_auth.get_session_tokens()
                if tokens:
                    settings.set_auth_tokens(
                        tokens['access_token'],
                        tokens['refresh_token'],
                        email
                    )
                # Check if they already have a subscription (e.g., from a previous signup)
                self._check_subscription_silent()
        else:
            error = result.get('error', 'Signup failed')
            self.signup_error.setText(error)
    
    def _show_subscribe_page(self):
        """Show the subscription page."""
        email = supabase_auth.user_email or settings.auth_user_email
        short_email = email.split('@')[0] if email else "there"
        self.welcome_label.setText(f"Hey {short_email}! 👋")
        self.title_label.setText("Unlock Filect")
        self.subtitle_label.setText("Choose a plan to access all features")
        self.stack.setCurrentIndex(2)
        # Taller than the other pages so the plan card isn't clipped on
        # high-DPI / scaled displays.
        self.setFixedSize(460, 730)
    
    def _open_checkout(self):
        """Open the web pricing page (plan selection + payment) and start polling."""
        self.sub_status.setObjectName("statusLabel")
        self.sub_status.setStyleSheet("")

        success = supabase_auth.open_web_pricing()

        if success:
            self.subscribe_button.setText("View plans again")
            self.sub_status.setText("Choose a plan in your browser — the app unlocks once you subscribe.")
            # Restart polling from zero each time the pricing page is opened
            self._poll_count = 0
            self._poll_timer.start(3000)
        else:
            self.subscribe_button.setText("View plans && subscribe")
            self.sub_status.setText("Couldn't open the plans page. Try again.")
            self.sub_status.setObjectName("errorLabel")

    def _poll_subscription(self):
        """Poll for subscription status after checkout."""
        self._poll_count += 1

        # Silent timeout after 30 minutes — reset button, stop polling
        if self._poll_count > 600:
            self._poll_timer.stop()
            self.subscribe_button.setText("View plans && subscribe")
            self.sub_status.setText("")
            return

        result = supabase_auth.check_subscription()

        if result.get('has_subscription'):
            self._poll_timer.stop()
            self.sub_status.setText("Payment confirmed! 🎉")
            logger.info("Subscription verified!")
            QTimer.singleShot(500, lambda: (self.auth_successful.emit(), self.accept()))
    
    def _check_subscription_silent(self):
        """Check subscription without UI updates."""
        result = supabase_auth.check_subscription()
        
        if result.get('has_subscription'):
            logger.info("Active subscription found")
            self.auth_successful.emit()
            self.accept()
        else:
            self._show_subscribe_page()
    
    def _do_logout(self):
        """Handle logout."""
        supabase_auth.sign_out()
        settings.clear_auth_tokens()
        
        self.login_email.clear()
        self.login_password.clear()
        self.signup_email.clear()
        self.signup_password.clear()
        self.signup_confirm.clear()
        
        self._go_to_login()
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self._poll_timer.isActive():
            self._poll_timer.stop()
        event.accept()
