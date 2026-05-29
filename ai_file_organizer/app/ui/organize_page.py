"""
AI-powered file organization page.

Flow:
1. User enters natural language instruction
2. App sends instruction + file metadata to LLM
3. LLM returns organization plan (folders + file assignments)
4. App validates the plan
5. User previews and approves
6. App executes moves deterministically
"""

import os
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QTreeWidget, QTreeWidgetItem,
    QProgressBar, QMessageBox, QFileDialog, QGroupBox,
    QSplitter, QFrame, QSizePolicy, QScrollArea,
    QDialog, QListWidget, QListWidgetItem, QCheckBox,
    QSpacerItem, QStackedWidget, QButtonGroup, QApplication,
    QRadioButton, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer

from app.core.settings import settings

from app.core.database import file_index
from app.core.ai_organizer import (
    request_organization_plan, validate_plan, plan_to_moves, get_plan_summary,
    deduplicate_plan, ensure_all_files_included
)
from app.core.apply import apply_moves

logger = logging.getLogger(__name__)


class PlanWorker(QThread):
    """Background worker for LLM planning - keeps UI responsive."""
    finished = Signal(object)  # plan dict or None
    error = Signal(str)
    
    def __init__(self, instruction: str, files: list):
        super().__init__()
        self.instruction = instruction
        self.files = files
    
    def run(self):
        try:
            plan = request_organization_plan(self.instruction, self.files)
            self.finished.emit(plan)
        except Exception as e:
            logger.error(f"Plan worker error: {e}")
            self.error.emit(str(e))


class RefineWorker(QThread):
    """Background worker for plan refinement."""
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, original_instruction: str, current_plan: dict, feedback: str, files: list):
        super().__init__()
        self.original_instruction = original_instruction
        self.current_plan = current_plan
        self.feedback = feedback
        self.files = files
    
    def run(self):
        try:
            from app.core.ai_organizer import request_plan_refinement
            plan = request_plan_refinement(
                self.original_instruction,
                self.current_plan,
                self.feedback,
                self.files
            )
            self.finished.emit(plan)
        except Exception as e:
            logger.error(f"Refine worker error: {e}")
            self.error.emit(str(e))


class VoiceRecordWorker(QThread):
    """Background worker for voice recording and transcription."""
    finished = Signal(str)  # transcribed text
    error = Signal(str)
    recording_stopped = Signal()  # emitted when recording stops
    
    def __init__(self, duration: int = 30, sample_rate: int = 16000):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_data = []
    
    def run(self):
        try:
            import sys, os
            # On Windows with PyInstaller --onefile, add the sounddevice PortAudio
            # binary directory to the DLL search path so its dependencies load correctly
            if sys.platform == "win32" and hasattr(sys, "_MEIPASS"):
                portaudio_dir = os.path.join(sys._MEIPASS, "_sounddevice_data", "portaudio-binaries")
                if os.path.isdir(portaudio_dir):
                    os.add_dll_directory(portaudio_dir)

            import sounddevice as sd
            import numpy as np
            from scipy.io import wavfile
            import tempfile
            import os
            from app.core.vision import transcribe_audio_proxy
            
            self.is_recording = True
            self.audio_data = []
            
            def audio_callback(indata, frames, time, status):
                if self.is_recording:
                    self.audio_data.append(indata.copy())
            
            # Start recording
            with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                              dtype='int16', callback=audio_callback):
                while self.is_recording:
                    sd.sleep(100)  # Check every 100ms
            
            self.recording_stopped.emit()
            
            if not self.audio_data:
                self.error.emit("No audio recorded")
                return
            
            # Combine audio chunks
            audio = np.concatenate(self.audio_data, axis=0)
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                wavfile.write(temp_path, self.sample_rate, audio)
            
            try:
                # Transcribe via the Supabase Edge Function Whisper proxy.
                text = transcribe_audio_proxy(temp_path)
                if text is None:
                    self.error.emit("Voice transcription failed — please sign in and try again.")
                else:
                    self.finished.emit(text)
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except ImportError as e:
            self.error.emit(f"Missing audio library: {e}\nRun: pip install sounddevice scipy")
        except Exception as e:
            logger.error(f"Voice recording error: {e}")
            self.error.emit(str(e))
    
    def stop_recording(self):
        """Stop the recording."""
        self.is_recording = False


class IndexBeforeOrganizeWorker(QThread):
    """Background worker for indexing files before organizing."""
    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(dict)  # stats dict
    error = Signal(str)
    cancelled = Signal()  # Emitted when user cancels
    
    def __init__(self, folder_path: Path):
        super().__init__()
        self.folder_path = folder_path
        self._cancelled = False
    
    def cancel(self):
        """Request cancellation of the indexing operation."""
        self._cancelled = True
    
    def is_cancelled(self):
        """Check if cancellation was requested."""
        return self._cancelled
    
    def run(self):
        try:
            from app.core.search import SearchService
            
            search_service = SearchService()
            
            def progress_callback(current, total, message):
                if self._cancelled:
                    raise InterruptedError("Indexing cancelled by user")
                self.progress.emit(current, total, message)
            
            # MUST be recursive: the "New Files Detected" detector
            # (_check_for_unindexed_files) uses os.walk to find unindexed
            # files at ANY depth in the destination folder tree. If we only
            # index the top level here, the detector keeps re-firing on the
            # same subfolder files every time the user clicks Generate Plan,
            # creating an infinite "Index Now" → popup → "Index Now" → popup
            # loop. Matching the detector's recursion fixes it.
            stats = search_service.index_directory(
                self.folder_path,
                recursive=True,
                progress_cb=progress_callback
            )
            
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit(stats)
            
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:
            if self._cancelled:
                self.cancelled.emit()
            else:
                logger.error(f"Index before organize error: {e}")
                self.error.emit(str(e))


class IndexProgressDialog(QDialog):
    """Modal progress dialog for indexing with Cancel/Skip options."""
    
    # Signals for the result
    skip_requested = Signal()  # User wants to skip indexing
    
    def __init__(self, folder_name: str, total_files: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Indexing Files")
        self.setMinimumSize(480, 280)
        self.setModal(True)
        self._drag_pos = None
        self.total_files = total_files
        self.current_file = 0
        self._result = None  # 'cancel', 'skip', or None (completed)
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._setup_ui(folder_name)
    
    def _setup_ui(self, folder_name: str):
        """Setup the dialog UI."""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Container with styled background
        container = QFrame()
        container.setObjectName("progressContainer")
        container.setStyleSheet(f"""
            QFrame#progressContainer {{
                background-color: {c['surface']};
                border-radius: 16px;
                border: 1px solid rgba(124, 77, 255, 0.25);
            }}
        """)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(28, 24, 28, 24)
        container_layout.setSpacing(16)
        
        # Header with icon
        header_layout = QHBoxLayout()
        
        icon_label = QLabel("🔍")
        icon_label.setStyleSheet("font-size: 32px;")
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Indexing Files")
        title_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        container_layout.addLayout(header_layout)
        
        # Folder info
        folder_label = QLabel(f"📁 {folder_name}")
        folder_label.setStyleSheet(f"""
            color: {c['text_secondary']};
            font-size: 14px;
            padding: 8px 12px;
            background: rgba(124, 77, 255, 0.06);
            border-radius: 8px;
        """)
        container_layout.addWidget(folder_label)
        
        # Progress info
        self.progress_label = QLabel(f"Preparing to index {self.total_files} files...")
        self.progress_label.setStyleSheet(f"""
            color: {c['text_secondary']};
            font-size: 14px;
        """)
        container_layout.addWidget(self.progress_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(1, self.total_files))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {c['border']};
                border: none;
                border-radius: 6px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #7C4DFF, stop:1 #9575FF);
                border-radius: 6px;
            }}
        """)
        container_layout.addWidget(self.progress_bar)
        
        # Current file label
        self.file_label = QLabel("")
        self.file_label.setStyleSheet(f"""
            color: {c['text_muted']};
            font-size: 12px;
        """)
        self.file_label.setWordWrap(True)
        container_layout.addWidget(self.file_label)
        
        container_layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        # Skip button - proceed without full indexing
        self.skip_btn = QPushButton("⏭️ Skip Indexing")
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.setMinimumHeight(40)
        self.skip_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {c['border_strong']};
                border-radius: 10px;
                color: {c['text_muted']};
                font-size: 13px;
                font-weight: 600;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                border-color: #FF9800;
                color: #FF9800;
                background-color: rgba(255, 152, 0, 0.06);
            }}
        """)
        self.skip_btn.setToolTip("Continue with already-indexed files only")
        self.skip_btn.clicked.connect(self._on_skip)
        button_layout.addWidget(self.skip_btn)
        
        button_layout.addStretch()
        
        # Cancel button
        self.cancel_btn = QPushButton("✕ Cancel")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5252;
                border: none;
                border-radius: 10px;
                color: white;
                font-size: 13px;
                font-weight: 600;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #FF6B6B;
            }
        """)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        container_layout.addLayout(button_layout)
        
        main_layout.addWidget(container)
    
    def update_progress(self, current: int, total: int, message: str):
        """Update the progress display."""
        self.current_file = current
        self.total_files = total
        
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        
        self.progress_label.setText(f"Indexing file {current} of {total}...")
        
        # Show truncated file name
        if message:
            display_msg = message
            if len(display_msg) > 60:
                display_msg = "..." + display_msg[-57:]
            self.file_label.setText(display_msg)
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self._result = 'cancel'
        self.reject()
    
    def _on_skip(self):
        """Handle skip button click."""
        self._result = 'skip'
        self.skip_requested.emit()
        self.accept()
    
    def get_result(self):
        """Get the dialog result: 'cancel', 'skip', or None (completed normally)."""
        return self._result
    
    def complete(self):
        """Called when indexing completes successfully."""
        self._result = None
        self.accept()
    
    # Dragging support
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()


class EmptyFolderDialog(QDialog):
    """Modern dialog to let user choose which empty folders to delete."""
    
    def __init__(self, empty_folders: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Empty Folders")
        self.setMinimumSize(520, 400)
        self.setModal(True)
        self._drag_pos = None
        
        self.empty_folders = empty_folders
        self.folders_to_delete = []
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self.container = QFrame(self)
        self.container.setObjectName("emptyFolderContainer")
        self.container.setStyleSheet(f"""
            QFrame#emptyFolderContainer {{
                background-color: {c['surface']};
                border-radius: 24px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add drop shadow
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)
        
        icon_label = QLabel("🗑️")
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(255, 152, 0, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(255, 152, 0, 0.20);
        """)
        icon_label.setFixedSize(48, 48)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Empty Folders Found")
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #E53935;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #C62828;
            }
        """)
        close_btn.clicked.connect(self.reject)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Description
        desc = QLabel("These folders are now empty. Select which ones to delete:")
        desc.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text_muted']};
        """)
        layout.addWidget(desc)
        
        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {c['border']};")
        layout.addWidget(divider)
        
        # Folder list with checkboxes
        self.folder_list = QListWidget()
        self.folder_list.setStyleSheet(f"""
            QListWidget {{
                border: 1px solid {c['border']};
                border-radius: 12px;
                background-color: {c['card']};
                padding: 8px;
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-radius: 8px;
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                color: {c['text_muted']};
            }}
            QListWidget::item:hover {{
                background-color: {c['input_bg']};
            }}
            QListWidget::item:selected {{
                background-color: rgba(124, 77, 255, 0.12);
                color: #B39DFF;
            }}
        """)
        
        for folder_path in empty_folders:
            # Show just the folder name with path hint
            from pathlib import Path
            folder = Path(folder_path)
            display_text = f"📁 {folder.name}"
            
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # Default to checked
            item.setText(display_text)
            item.setToolTip(folder_path)  # Full path on hover
            item.setData(Qt.UserRole, folder_path)
            self.folder_list.addItem(item)
        
        layout.addWidget(self.folder_list, 1)
        
        # Selection buttons row
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(10)
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.setMinimumHeight(36)
        select_all_btn.setCursor(Qt.PointingHandCursor)
        select_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 8px;
                font-size: 12px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
                color: #B39DFF;
            }}
        """)
        select_all_btn.clicked.connect(self._select_all)
        selection_layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.setMinimumHeight(36)
        deselect_all_btn.setCursor(Qt.PointingHandCursor)
        deselect_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 8px;
                font-size: 12px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: rgba(211, 47, 47, 0.08);
                border-color: #D32F2F;
                color: #FF6B6B;
            }}
        """)
        deselect_all_btn.clicked.connect(self._deselect_all)
        selection_layout.addWidget(deselect_all_btn)
        
        selection_layout.addStretch()
        layout.addLayout(selection_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        keep_all_btn = QPushButton("Keep All")
        keep_all_btn.setMinimumHeight(44)
        keep_all_btn.setCursor(Qt.PointingHandCursor)
        keep_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
                padding: 8px 24px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
                color: #B39DFF;
            }}
        """)
        keep_all_btn.clicked.connect(self.reject)
        button_layout.addWidget(keep_all_btn)
        
        button_layout.addStretch()
        
        delete_selected_btn = QPushButton("🗑️ Delete Selected")
        delete_selected_btn.setMinimumHeight(44)
        delete_selected_btn.setCursor(Qt.PointingHandCursor)
        delete_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        delete_selected_btn.clicked.connect(self._delete_selected)
        button_layout.addWidget(delete_selected_btn)
        
        layout.addLayout(button_layout)
    
    def _select_all(self):
        for i in range(self.folder_list.count()):
            self.folder_list.item(i).setCheckState(Qt.Checked)
    
    def _deselect_all(self):
        for i in range(self.folder_list.count()):
            self.folder_list.item(i).setCheckState(Qt.Unchecked)
    
    def _delete_selected(self):
        self.folders_to_delete = []
        for i in range(self.folder_list.count()):
            item = self.folder_list.item(i)
            if item.checkState() == Qt.Checked:
                self.folders_to_delete.append(item.data(Qt.UserRole))
        self.accept()
    
    def _delete_all(self):
        self.folders_to_delete = [item.data(Qt.UserRole) for i in range(self.folder_list.count())
                                   for item in [self.folder_list.item(i)]]
        self.accept()
    
    def get_folders_to_delete(self) -> list:
        return self.folders_to_delete
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()


class ModernConfirmDialog(QDialog):
    """
    Modern styled confirmation dialog matching the app's purple theme.
    Futuristic look with drop shadow, draggable window, and smooth styling.
    """
    
    def __init__(self, parent=None, title: str = "Confirm", message: str = "", 
                 details: list = None, highlight_text: str = "", info_text: str = "",
                 yes_text: str = "Yes", no_text: str = "No"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        self.setModal(True)
        self.result_accepted = False
        self._drag_pos = None
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container with rounded corners and shadow
        self.container = QFrame(self)
        self.container.setObjectName("modernDialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#modernDialogContainer {{
                background-color: {c['surface']};
                border-radius: 24px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add subtle drop shadow effect
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)  # Space for shadow
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)
        
        # Header with gradient icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        
        # Clean icon background
        icon_label = QLabel("✨")
        icon_label.setStyleSheet("""
            font-size: 26px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 22px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(52, 52)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
            letter-spacing: -0.3px;
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Close button - solid purple with white X (ALWAYS visible)
        close_btn = QPushButton("X")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self.reject)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Subtle divider line
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {c['border']};")
        layout.addWidget(divider)
        
        # Main message
        if message:
            msg_label = QLabel(message)
            msg_label.setWordWrap(True)
            msg_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 15px;
                color: {c['text_muted']};
                line-height: 1.6;
            """)
            layout.addWidget(msg_label)
        
        # Details list (bullet points) - clean solid style
        if details:
            details_frame = QFrame()
            details_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {c['surface']};
                    border-radius: 16px;
                    border: 1px solid {c['border']};
                }}
            """)
            details_layout = QVBoxLayout(details_frame)
            details_layout.setContentsMargins(20, 16, 20, 16)
            details_layout.setSpacing(12)
            
            for detail in details:
                detail_row = QHBoxLayout()
                detail_row.setSpacing(12)
                
                # Purple dot indicator
                dot = QLabel("•")
                dot.setStyleSheet("font-size: 18px; color: #7C4DFF;")
                dot.setFixedWidth(16)
                detail_row.addWidget(dot)
                
                detail_label = QLabel(detail)
                detail_label.setStyleSheet(f"""
                    font-family: "Segoe UI", sans-serif;
                    font-size: 14px;
                    color: {c['text']};
                    font-weight: 500;
                """)
                detail_row.addWidget(detail_label)
                detail_row.addStretch()
                
                details_layout.addLayout(detail_row)
            
            layout.addWidget(details_frame)
        
        # Highlighted text (purple with glow effect)
        if highlight_text:
            highlight_label = QLabel(highlight_text)
            highlight_label.setWordWrap(True)
            highlight_label.setStyleSheet("""
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                font-weight: 600;
                color: #7C4DFF;
                padding: 8px 0px;
            """)
            layout.addWidget(highlight_label)
        
        # Info text
        if info_text:
            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                color: {c['text_muted']};
                font-style: italic;
            """)
            layout.addWidget(info_label)
        
        layout.addSpacing(12)
        
        # Buttons with modern styling
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(14)
        btn_layout.addStretch()
        
        no_btn = QPushButton(no_text)
        no_btn.setMinimumHeight(46)
        no_btn.setMinimumWidth(120)
        no_btn.setCursor(Qt.PointingHandCursor)
        no_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 12px;
                font-family: "Segoe UI", sans-serif;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 28px;
            }}
            QPushButton:hover {{
                background-color: {c['card']};
                border-color: {c['text_muted']};
                color: {c['text']};
            }}
        """)
        no_btn.clicked.connect(self.reject)
        btn_layout.addWidget(no_btn)
        
        yes_btn = QPushButton(yes_text)
        yes_btn.setMinimumHeight(46)
        yes_btn.setMinimumWidth(120)
        yes_btn.setCursor(Qt.PointingHandCursor)
        yes_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 12px;
                font-family: "Segoe UI", sans-serif;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 28px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        yes_btn.clicked.connect(self.accept)
        btn_layout.addWidget(yes_btn)
        
        layout.addLayout(btn_layout)
    
    def mousePressEvent(self, event):
        """Enable dragging the dialog."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle drag movement."""
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """End dragging."""
        self._drag_pos = None
        event.accept()
    
    def accept(self):
        self.result_accepted = True
        super().accept()
    
    @staticmethod
    def ask(parent, title: str, message: str, details: list = None, 
            highlight_text: str = "", info_text: str = "",
            yes_text: str = "Yes", no_text: str = "No") -> bool:
        """Show dialog and return True if user clicked Yes."""
        dialog = ModernConfirmDialog(
            parent, title, message, details, highlight_text, info_text, yes_text, no_text
        )
        dialog.exec()
        return dialog.result_accepted


class ModernInfoDialog(QDialog):
    """
    Modern styled info/warning dialog with single OK button.
    Matches the app's purple theme with clean, minimal design.
    """
    
    def __init__(self, parent=None, title: str = "Information", message: str = "",
                 details: list = None, info_text: str = "", icon: str = "ℹ️",
                 ok_text: str = "OK", is_warning: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.setModal(True)
        self._drag_pos = None
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container with rounded corners and shadow
        self.container = QFrame(self)
        self.container.setObjectName("modernInfoContainer")
        self.container.setStyleSheet(f"""
            QFrame#modernInfoContainer {{
                background-color: {c['surface']};
                border-radius: 24px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add subtle drop shadow effect
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(18)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        
        # Icon with appropriate color
        icon_bg = "rgba(255, 152, 0, 0.08)" if is_warning else "rgba(124, 77, 255, 0.08)"
        icon_border = "rgba(255, 152, 0, 0.20)" if is_warning else "rgba(124, 77, 255, 0.20)"
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"""
            font-size: 24px;
            background-color: {icon_bg};
            border-radius: 22px;
            border: 2px solid {icon_border};
        """)
        icon_label.setFixedSize(52, 52)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
            letter-spacing: -0.3px;
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Close button - solid purple with white X (ALWAYS visible)
        close_btn = QPushButton("X")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self.accept)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Subtle divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {c['border']};")
        layout.addWidget(divider)
        
        # Main message
        if message:
            msg_label = QLabel(message)
            msg_label.setWordWrap(True)
            msg_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 15px;
                color: {c['text_muted']};
                line-height: 1.6;
            """)
            layout.addWidget(msg_label)
        
        # Details list (bullet points)
        if details:
            details_frame = QFrame()
            details_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {c['surface']};
                    border-radius: 14px;
                    border: 1px solid {c['border']};
                }}
            """)
            details_layout = QVBoxLayout(details_frame)
            details_layout.setContentsMargins(18, 14, 18, 14)
            details_layout.setSpacing(10)
            
            for detail in details:
                detail_row = QHBoxLayout()
                detail_row.setSpacing(10)
                
                dot = QLabel("•")
                dot.setStyleSheet("font-size: 16px; color: #7C4DFF;")
                dot.setFixedWidth(14)
                detail_row.addWidget(dot)
                
                detail_label = QLabel(detail)
                detail_label.setWordWrap(True)
                detail_label.setStyleSheet(f"""
                    font-family: "Segoe UI", sans-serif;
                    font-size: 14px;
                    color: {c['text_muted']};
                """)
                detail_row.addWidget(detail_label)
                detail_row.addStretch()
                
                details_layout.addLayout(detail_row)
            
            layout.addWidget(details_frame)
        
        # Info text
        if info_text:
            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            info_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                color: {c['text_muted']};
                font-style: italic;
            """)
            layout.addWidget(info_label)
        
        layout.addSpacing(8)
        
        # Single OK button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton(ok_text)
        ok_btn.setMinimumHeight(46)
        ok_btn.setMinimumWidth(140)
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 12px;
                font-family: "Segoe UI", sans-serif;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 32px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
    
    @staticmethod
    def show_warning(parent, title: str, message: str, details: list = None, 
                     info_text: str = "", ok_text: str = "OK"):
        """Show a warning dialog."""
        dialog = ModernInfoDialog(
            parent, title, message, details, info_text, 
            icon="⚠️", ok_text=ok_text, is_warning=True
        )
        dialog.exec()
    
    @staticmethod
    def show_info(parent, title: str, message: str, details: list = None,
                  info_text: str = "", ok_text: str = "OK"):
        """Show an info dialog."""
        dialog = ModernInfoDialog(
            parent, title, message, details, info_text,
            icon="ℹ️", ok_text=ok_text, is_warning=False
        )
        dialog.exec()


class ModernInputDialog(QDialog):
    """
    Modern styled input dialog for text input.
    Matches the app's theme with clean, minimal design.
    """
    
    def __init__(self, parent=None, title: str = "Input", message: str = "",
                 placeholder: str = "", icon: str = "✏️",
                 ok_text: str = "OK", cancel_text: str = "Cancel"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        self.setModal(True)
        self._drag_pos = None
        self.input_text = ""
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container with rounded corners and shadow
        self.container = QFrame(self)
        self.container.setObjectName("modernInputContainer")
        self.container.setStyleSheet(f"""
            QFrame#modernInputContainer {{
                background-color: {c['surface']};
                border-radius: 24px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add subtle drop shadow effect
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(18)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 22px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(52, 52)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
            letter-spacing: -0.3px;
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("X")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self.reject)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Message
        if message:
            msg_label = QLabel(message)
            msg_label.setWordWrap(True)
            msg_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                color: {c['text_muted']};
                line-height: 1.5;
            """)
            layout.addWidget(msg_label)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(placeholder)
        self.input_field.setMinimumHeight(48)
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: {c['input_bg']};
                border: 2px solid {c['border']};
                border-radius: 12px;
                padding: 12px 16px;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                color: {c['text']};
            }}
            QLineEdit:focus {{
                border-color: #7C4DFF;
            }}
            QLineEdit::placeholder {{
                color: {c['text_muted']};
            }}
        """)
        layout.addWidget(self.input_field)
        
        layout.addSpacing(8)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        
        cancel_btn = QPushButton(cancel_text)
        cancel_btn.setMinimumHeight(46)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 12px;
                font-family: "Segoe UI", sans-serif;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 24px;
            }}
            QPushButton:hover {{
                background-color: {c['card']};
                border-color: {c['text_muted']};
                color: {c['text']};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton(ok_text)
        ok_btn.setMinimumHeight(46)
        ok_btn.setMinimumWidth(100)
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 12px;
                font-family: "Segoe UI", sans-serif;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 24px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)
        
        layout.addLayout(btn_layout)
        
        # Focus the input field
        self.input_field.setFocus()
    
    def _on_ok(self):
        self.input_text = self.input_field.text()
        self.accept()
    
    def get_text(self) -> str:
        return self.input_text
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
    
    @staticmethod
    def get_input(parent, title: str, message: str, placeholder: str = "",
                  icon: str = "✏️", ok_text: str = "OK", cancel_text: str = "Cancel") -> tuple:
        """Show input dialog and return (text, accepted)."""
        dialog = ModernInputDialog(parent, title, message, placeholder, icon, ok_text, cancel_text)
        result = dialog.exec()
        return (dialog.get_text(), result == QDialog.Accepted)


class ModernProgressDialog(QDialog):
    """
    Modern styled progress dialog matching the app's theme.
    Features animated progress bar, clean design, and optional cancel button.
    """
    
    cancelled = Signal()
    
    def __init__(self, parent=None, title: str = "Processing", message: str = "",
                 icon: str = "⏳", can_cancel: bool = True, indeterminate: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setModal(True)
        self._drag_pos = None
        self._cancelled = False
        self._indeterminate = indeterminate
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._colors = c
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container with rounded corners and shadow
        self.container = QFrame(self)
        self.container.setObjectName("modernProgressContainer")
        self.container.setStyleSheet(f"""
            QFrame#modernProgressContainer {{
                background-color: {c['surface']};
                border-radius: 20px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add subtle drop shadow effect
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)
        
        # Animated icon
        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet("""
            font-size: 28px;
            background-color: rgba(124, 77, 255, 0.10);
            border-radius: 24px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        self.icon_label.setFixedSize(56, 56)
        self.icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.icon_label)
        
        # Title and message in vertical layout
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: {c['text']};
            letter-spacing: -0.2px;
        """)
        title_layout.addWidget(self.title_label)
        
        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text_muted']};
        """)
        title_layout.addWidget(self.message_label)
        
        header_layout.addLayout(title_layout, 1)
        layout.addLayout(header_layout)
        
        # Progress bar container
        progress_container = QFrame()
        progress_container.setStyleSheet(f"""
            QFrame {{
                background-color: {c['card']};
                border-radius: 12px;
                border: 1px solid {c['border']};
            }}
        """)
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(16, 14, 16, 14)
        progress_layout.setSpacing(10)
        
        # Status text
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text']};
            font-weight: 500;
        """)
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setMaximumHeight(8)
        
        if indeterminate:
            self.progress_bar.setRange(0, 0)  # Indeterminate mode
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
        
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {c['border']};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #7C4DFF, stop:0.5 #9575FF, stop:1 #7C4DFF);
                border-radius: 4px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Count label (e.g., "3 of 10 files")
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 12px;
            color: {c['text_muted']};
        """)
        self.count_label.setAlignment(Qt.AlignRight)
        progress_layout.addWidget(self.count_label)
        
        layout.addWidget(progress_container)
        
        # Cancel button (optional)
        if can_cancel:
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.setMinimumHeight(42)
            self.cancel_btn.setMinimumWidth(100)
            self.cancel_btn.setCursor(Qt.PointingHandCursor)
            self.cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #7C4DFF;
                    color: white;
                    border: none;
                    border-radius: 10px;
                    font-family: "Segoe UI", sans-serif;
                    font-weight: 600;
                    font-size: 14px;
                    padding: 10px 28px;
                }
                QPushButton:hover {
                    background-color: #9575FF;
                }
                QPushButton:pressed {
                    background-color: #6A3DE8;
                }
            """)
            self.cancel_btn.clicked.connect(self._on_cancel)
            btn_layout.addWidget(self.cancel_btn)
            
            layout.addLayout(btn_layout)
    
    def _on_cancel(self):
        self._cancelled = True
        self.cancelled.emit()
        self.close()
    
    def is_cancelled(self) -> bool:
        return self._cancelled
    
    def set_progress(self, current: int, total: int, status: str = ""):
        """Update progress bar and status text."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self.count_label.setText(f"{current} of {total}")
        if status:
            self.status_label.setText(status)
        QApplication.processEvents()
    
    def set_status(self, status: str):
        """Update only the status text."""
        self.status_label.setText(status)
        QApplication.processEvents()
    
    def set_message(self, message: str):
        """Update the message below the title."""
        self.message_label.setText(message)
        QApplication.processEvents()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
    
    @staticmethod
    def create(parent, title: str, message: str = "", icon: str = "⏳",
               can_cancel: bool = True, indeterminate: bool = False):
        """Create and show a modern progress dialog."""
        dialog = ModernProgressDialog(parent, title, message, icon, can_cancel, indeterminate)
        dialog.show()
        QApplication.processEvents()
        return dialog


class UpdateNotificationDialog(QDialog):
    """
    Modern styled dialog to notify users about available updates.
    Shows version info, release notes, and download button.
    """
    
    def __init__(self, parent=None, update_info: dict = None):
        super().__init__(parent)
        self.update_info = update_info or {}
        self._drag_pos = None  # For dragging
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(False)  # Non-blocking
        self._setup_ui()
    
    def mousePressEvent(self, event):
        """Start dragging when mouse is pressed."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Move dialog while dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging when mouse is released."""
        self._drag_pos = None
        event.accept()
    
    def _setup_ui(self):
        from app.ui.theme_manager import get_theme_colors
        from PySide6.QtGui import QColor
        c = get_theme_colors()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Container with shadow
        self.container = QFrame()
        self.container.setObjectName("updateContainer")
        self.container.setStyleSheet(f"""
            QFrame#updateContainer {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 16px;
            }}
        """)
        
        # Add shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.container.setGraphicsEffect(shadow)
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(24, 20, 24, 20)
        container_layout.setSpacing(16)
        
        # Header with icon and close button
        header = QHBoxLayout()
        
        # Sparkle icon
        icon = QLabel("✨")
        icon.setStyleSheet("font-size: 28px; background: transparent;")
        header.addWidget(icon)
        
        # Title
        title = QLabel("Update Available!")
        title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            color: {c['text']};
            background: transparent;
        """)
        header.addWidget(title)
        
        header.addStretch()
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 14px;
                font-size: 14px;
                color: {c['text_muted']};
            }}
            QPushButton:hover {{
                background: {c['card']};
                color: {c['text']};
            }}
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        
        container_layout.addLayout(header)
        
        # Version info
        current = self.update_info.get('current_version', '1.0.0')
        latest = self.update_info.get('latest_version', '1.0.1')
        
        version_label = QLabel(f"Version {latest} is ready")
        version_label.setStyleSheet(f"""
            font-size: 14px;
            color: {c['text_muted']};
            background: transparent;
        """)
        container_layout.addWidget(version_label)
        
        # Release notes (if any)
        notes = self.update_info.get('release_notes', '')
        if notes:
            notes_label = QLabel(notes)
            notes_label.setWordWrap(True)
            notes_label.setStyleSheet(f"""
                font-size: 13px;
                color: {c['text_muted']};
                background: {c['card']};
                border-radius: 8px;
                padding: 12px;
            """)
            container_layout.addWidget(notes_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        # Later button
        later_btn = QPushButton("Later")
        later_btn.setCursor(Qt.PointingHandCursor)
        later_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {c['text_muted']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {c['card']};
                color: {c['text']};
            }}
        """)
        later_btn.clicked.connect(self.close)
        btn_layout.addWidget(later_btn)
        
        # Download button
        download_btn = QPushButton("Download Update")
        download_btn.setCursor(Qt.PointingHandCursor)
        download_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9575FF, stop:1 #B39DFF);
            }
        """)
        download_btn.clicked.connect(self._on_download)
        btn_layout.addWidget(download_btn)
        
        container_layout.addLayout(btn_layout)
        
        layout.addWidget(self.container)
        
        # Set fixed width
        self.setFixedWidth(380)
    
    def _on_download(self):
        """Start download and install process."""
        url = self.update_info.get('download_url', '')
        if not url:
            return
        
        # Check if it's a direct download link (zip/exe) or a page
        if url.endswith('.zip') or url.endswith('.exe') or 'releases/download' in url:
            # Direct download - use auto-updater
            self._start_auto_update(url)
        else:
            # It's a page URL - open in browser
            from app.core.update_checker import open_download_page
            open_download_page(url)
            self.close()
    
    def _start_auto_update(self, download_url: str):
        """Download and install update automatically."""
        self.close()
        
        # Show download progress dialog
        parent = self.parent()
        if parent:
            dialog = UpdateDownloadDialog(parent, download_url, self.update_info)
            dialog.exec()
    
    @staticmethod
    def show_update(parent, update_info: dict):
        """Show update notification dialog."""
        dialog = UpdateNotificationDialog(parent, update_info)
        # Position in top-right corner
        if parent:
            parent_geo = parent.geometry()
            dialog.move(
                parent_geo.right() - dialog.width() - 20,
                parent_geo.top() + 60
            )
        dialog.show()
        return dialog


class UpdateDownloadDialog(QDialog):
    """
    Dialog showing download progress for auto-updates.
    Downloads, extracts, and applies updates automatically.
    """
    # Signals for thread-safe UI updates
    progress_signal = Signal(int, int)  # downloaded, total
    status_signal = Signal(str)  # status message
    download_complete_signal = Signal(object)  # installer_path or None
    
    def __init__(self, parent=None, download_url: str = "", update_info: dict = None):
        super().__init__(parent)
        self.download_url = download_url
        self.update_info = update_info or {}
        self.download_thread = None
        self.installer_path = None  # Path to downloaded installer
        self._drag_pos = None  # For dragging
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self._setup_ui()
        
        # Connect signals to slots (thread-safe UI updates)
        self.progress_signal.connect(self._handle_progress)
        self.status_signal.connect(self._handle_status)
        self.download_complete_signal.connect(self._handle_download_complete)
        
        # Start download after dialog is shown
        QTimer.singleShot(500, self._start_download)
    
    def mousePressEvent(self, event):
        """Start dragging when mouse is pressed."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Move dialog while dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Stop dragging when mouse is released."""
        self._drag_pos = None
        event.accept()
    
    def _setup_ui(self):
        from app.ui.theme_manager import get_theme_colors
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        c = get_theme_colors()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Container
        self.container = QFrame()
        self.container.setObjectName("downloadContainer")
        self.container.setStyleSheet(f"""
            QFrame#downloadContainer {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 16px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.container.setGraphicsEffect(shadow)
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(28, 24, 28, 24)
        container_layout.setSpacing(16)
        
        # Icon and title
        header = QHBoxLayout()
        icon = QLabel("⬇️")
        icon.setStyleSheet("font-size: 24px; background: transparent;")
        header.addWidget(icon)
        
        title = QLabel("Downloading Update...")
        title.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {c['text']};
            background: transparent;
        """)
        header.addWidget(title)
        header.addStretch()
        container_layout.addLayout(header)
        
        # Version info
        latest = self.update_info.get('latest_version', '')
        version_label = QLabel(f"Version {latest}")
        version_label.setStyleSheet(f"font-size: 13px; color: {c['text_muted']}; background: transparent;")
        container_layout.addWidget(version_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {c['card']};
                border: none;
                border-radius: 6px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7C4DFF, stop:1 #9575FF);
                border-radius: 6px;
            }}
        """)
        self.progress_bar.setTextVisible(False)
        container_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Connecting...")
        self.status_label.setStyleSheet(f"font-size: 12px; color: {c['text_muted']}; background: transparent;")
        container_layout.addWidget(self.status_label)
        
        # Buttons (hidden initially, shown when ready)
        self.btn_layout = QHBoxLayout()
        self.btn_layout.setSpacing(12)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {c['text_muted']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {c['card']};
            }}
        """)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.btn_layout.addWidget(self.cancel_btn)
        
        self.install_btn = QPushButton("Install & Restart")
        self.install_btn.setCursor(Qt.PointingHandCursor)
        self.install_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9575FF, stop:1 #B39DFF);
            }
        """)
        self.install_btn.clicked.connect(self._on_install)
        self.install_btn.setVisible(False)
        self.btn_layout.addWidget(self.install_btn)
        
        container_layout.addLayout(self.btn_layout)
        
        layout.addWidget(self.container)
        self.setFixedWidth(360)
    
    def _start_download(self):
        """Start download in background thread."""
        import threading
        
        def download_thread():
            from app.core.auto_updater import download_update
            
            # Download the installer with both progress and status callbacks
            # These callbacks emit signals which are thread-safe
            installer_path = download_update(
                self.download_url,
                progress_callback=self._emit_progress,
                status_callback=self._emit_status
            )
            
            # Emit completion signal (thread-safe)
            self.download_complete_signal.emit(installer_path)
        
        self.download_thread = threading.Thread(target=download_thread, daemon=True)
        self.download_thread.start()
    
    def _emit_status(self, status: str):
        """Emit status signal from background thread."""
        self.status_signal.emit(status)
    
    def _emit_progress(self, downloaded: int, total: int):
        """Emit progress signal from background thread."""
        self.progress_signal.emit(downloaded, total)
    
    def _handle_status(self, status: str):
        """Handle status update in main thread (connected to signal)."""
        self.status_label.setText(status)
    
    def _handle_progress(self, downloaded: int, total: int):
        """Handle progress update in main thread (connected to signal)."""
        if total > 0:
            percent = int((downloaded / total) * 100)
            self._update_progress(percent, downloaded, total)
        elif downloaded == 0:
            self.status_label.setText("Starting download...")
    
    def _handle_download_complete(self, installer_path):
        """Handle download completion in main thread (connected to signal)."""
        if installer_path:
            self.installer_path = installer_path
            self._on_ready()
        else:
            self._on_error("Download failed - check your internet connection")
    
    def _update_progress(self, percent: int, downloaded: int, total: int):
        """Update UI with progress."""
        self.progress_bar.setValue(percent)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        self.status_label.setText(f"Downloading... {mb_down:.1f} / {mb_total:.1f} MB")
    
    def _update_status(self, text: str):
        """Update status label."""
        self.status_label.setText(text)
    
    def _on_ready(self):
        """Download complete, ready to install."""
        self.progress_bar.setValue(100)
        self.status_label.setText("Ready to install!")
        self.cancel_btn.setText("Later")
        self.install_btn.setVisible(True)
    
    def _on_error(self, message: str):
        """Handle download error."""
        self.status_label.setText(f"Error: {message}")
        self.cancel_btn.setText("Close")
        self.progress_bar.setStyleSheet(self.progress_bar.styleSheet().replace("#7C4DFF", "#FF6B6B"))
    
    def _on_cancel(self):
        """Cancel download or close dialog."""
        self.close()
    
    def _on_install(self):
        """Apply update by running the installer."""
        if not self.installer_path:
            return
        
        self.status_label.setText("Launching installer...")
        self.install_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        
        from app.core.auto_updater import apply_update_and_restart
        
        if apply_update_and_restart(self.installer_path):
            # Close the app - installer will handle the rest
            self.status_label.setText("Installer started! App will close...")
            QTimer.singleShot(1000, QApplication.quit)
        else:
            self._on_error("Failed to launch installer")
            self.cancel_btn.setEnabled(True)


class HistoryDialog(QDialog):
    """
    Modern dialog showing organization history with undo capability.
    Displays recent file organization operations in a clean, scrollable list.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_page = parent
        self.setWindowTitle("Organization History")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        self._drag_pos = None
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._theme_colors = c  # Store for use in other methods
        self.container = QFrame(self)
        self.container.setObjectName("historyDialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#historyDialogContainer {{
                background-color: {c['surface']};
                border-radius: 24px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add drop shadow
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)
        
        icon_label = QLabel("📋")
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(48, 48)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Organization History")
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self.accept)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {c['border']};")
        layout.addWidget(divider)
        
        # History list container
        self.history_list = QWidget()
        self.history_layout = QVBoxLayout(self.history_list)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(10)
        
        # Scroll area for history items
        scroll = QScrollArea()
        scroll.setWidget(self.history_list)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['scrollbar_handle']};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #7C4DFF;
            }}
        """)
        layout.addWidget(scroll, 1)
        
        # Load history
        self._load_history()
        
        # Footer with clear button
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        
        clear_btn = QPushButton("Clear History")
        clear_btn.setMinimumHeight(40)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 10px;
                font-size: 13px;
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                background-color: rgba(211, 47, 47, 0.10);
                border-color: rgba(211, 47, 47, 0.30);
                color: #FF6B6B;
            }}
        """)
        clear_btn.clicked.connect(self._clear_history)
        footer_layout.addWidget(clear_btn)
        
        close_dialog_btn = QPushButton("Close")
        close_dialog_btn.setMinimumHeight(40)
        close_dialog_btn.setCursor(Qt.PointingHandCursor)
        close_dialog_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
                padding: 8px 28px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        close_dialog_btn.clicked.connect(self.accept)
        footer_layout.addWidget(close_dialog_btn)
        
        layout.addLayout(footer_layout)
    
    def _load_history(self):
        """Load move history from log files."""
        from app.core.apply import get_move_history
        
        history = get_move_history()
        c = self._theme_colors
        
        if not history:
            # Show empty state
            empty_label = QLabel("No organization history yet")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 15px;
                color: {c['text_muted']};
                padding: 40px;
            """)
            self.history_layout.addWidget(empty_label)
            self.history_layout.addStretch()
            return
        
        # Add history items (most recent first, limit to 20)
        for item in history[:20]:
            self._add_history_item(item)
        
        self.history_layout.addStretch()
    
    def _add_history_item(self, item: dict):
        """Add a single history item to the list."""
        from datetime import datetime
        c = self._theme_colors
        
        item_frame = QFrame()
        item_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {c['card']};
                border-radius: 12px;
                border: 1px solid {c['border']};
            }}
            QFrame:hover {{
                background-color: rgba(124, 77, 255, 0.06);
                border-color: rgba(124, 77, 255, 0.15);
            }}
        """)
        
        item_layout = QHBoxLayout(item_frame)
        item_layout.setContentsMargins(16, 14, 16, 14)
        item_layout.setSpacing(14)
        
        # Icon
        icon = QLabel("📁")
        icon.setStyleSheet("font-size: 20px; background: transparent; border: none;")
        icon.setFixedWidth(28)
        item_layout.addWidget(icon)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        
        # Parse timestamp
        timestamp_str = item.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp_str)
            formatted_date = dt.strftime("%b %d, %Y at %I:%M %p")
        except:
            formatted_date = timestamp_str[:19] if timestamp_str else "Unknown date"

        is_reverted = bool(item.get("reverted", False))

        # Date row carries an optional "Reverted" badge inline so the user
        # can spot already-undone operations at a glance.
        date_row = QHBoxLayout()
        date_row.setSpacing(8)

        date_label = QLabel(formatted_date)
        date_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: {c['text']};
            background: transparent;
            border: none;
        """)
        date_row.addWidget(date_label)

        if is_reverted:
            reverted_badge = QLabel("Reverted")
            reverted_badge.setStyleSheet("""
                font-family: "Segoe UI", sans-serif;
                font-size: 10px;
                font-weight: 600;
                color: white;
                background: #9E9E9E;
                border-radius: 4px;
                padding: 2px 6px;
            """)
            date_row.addWidget(reverted_badge)

        date_row.addStretch()
        info_layout.addLayout(date_row)

        files_count = item.get("successful_moves", item.get("total_files", 0))
        reverted_suffix = " (reverted)" if is_reverted else ""
        details_label = QLabel(f"{files_count} file(s) organized{reverted_suffix}")
        details_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 12px;
            color: {c['text_muted']};
            background: transparent;
            border: none;
        """)
        info_layout.addWidget(details_label)
        
        item_layout.addLayout(info_layout, 1)
        
        # View button
        view_btn = QPushButton("View")
        view_btn.setFixedSize(70, 32)
        view_btn.setCursor(Qt.PointingHandCursor)
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7C4DFF;
                border: 1px solid #7C4DFF;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #7C4DFF;
                color: white;
            }
        """)
        log_file = item.get("log_file", "")
        view_btn.clicked.connect(lambda checked, lf=log_file: self._view_details(lf))
        item_layout.addWidget(view_btn)
        
        self.history_layout.addWidget(item_frame)
    
    def _view_details(self, log_file: str):
        """Show details of a specific organization operation in a scrollable dialog."""
        import json
        from pathlib import Path
        c = self._theme_colors
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            moves = log_data.get("moves", [])
            renamed = log_data.get("renamed_files", [])
            is_reverted = bool(log_data.get("reverted", False))
            
            # Create a custom scrollable details dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Operation Details")
            dialog.setMinimumSize(550, 450)
            dialog.setModal(True)
            dialog._drag_pos = None
            
            # Remove frame for custom styling
            dialog.setWindowFlags(dialog.windowFlags() | Qt.FramelessWindowHint)
            dialog.setAttribute(Qt.WA_TranslucentBackground)
            
            # Container
            container = QFrame(dialog)
            container.setObjectName("detailsContainer")
            container.setStyleSheet(f"""
                QFrame#detailsContainer {{
                    background-color: {c['surface']};
                    border-radius: 20px;
                    border: 1px solid {c['border']};
                }}
            """)
            
            # Shadow
            from PySide6.QtWidgets import QGraphicsDropShadowEffect
            from PySide6.QtGui import QColor
            shadow = QGraphicsDropShadowEffect(dialog)
            shadow.setBlurRadius(25)
            shadow.setXOffset(0)
            shadow.setYOffset(4)
            shadow.setColor(QColor(0, 0, 0, 40))
            container.setGraphicsEffect(shadow)
            
            main_layout = QVBoxLayout(dialog)
            main_layout.setContentsMargins(15, 15, 15, 15)
            main_layout.addWidget(container)
            
            layout = QVBoxLayout(container)
            layout.setContentsMargins(24, 20, 24, 20)
            layout.setSpacing(14)
            
            # Header
            header = QHBoxLayout()
            header.setSpacing(12)
            
            icon = QLabel("📋")
            icon.setStyleSheet("font-size: 22px; background: rgba(124, 77, 255, 0.12); border-radius: 18px; border: 1px solid rgba(124, 77, 255, 0.20);")
            icon.setFixedSize(44, 44)
            icon.setAlignment(Qt.AlignCenter)
            header.addWidget(icon)
            
            title = QLabel(f"Organized {len(moves)} file(s)")
            title.setStyleSheet(f"font-family: 'Segoe UI'; font-size: 18px; font-weight: 700; color: {c['text']};")
            header.addWidget(title)
            header.addStretch()
            
            close_btn = QPushButton("X")
            close_btn.setFixedSize(32, 32)
            close_btn.setCursor(Qt.PointingHandCursor)
            close_btn.setStyleSheet("""
                QPushButton { background: #7C4DFF; border: none; color: white; font-size: 18px; font-weight: bold; font-family: Arial, Helvetica, sans-serif; border-radius: 16px; padding: 0px; margin: 0px; }
                QPushButton:hover { background: #5E35B1; }
            """)
            close_btn.clicked.connect(dialog.accept)
            header.addWidget(close_btn)
            
            layout.addLayout(header)
            
            # Divider
            divider = QFrame()
            divider.setFixedHeight(1)
            divider.setStyleSheet(f"background: {c['border']};")
            layout.addWidget(divider)
            
            # Scrollable file list
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(f"""
                QScrollArea {{ border: none; background: transparent; }}
                QScrollBar:vertical {{ border: none; background: transparent; width: 6px; border-radius: 3px; }}
                QScrollBar::handle:vertical {{ background: {c['scrollbar_handle']}; border-radius: 3px; min-height: 20px; }}
                QScrollBar::handle:vertical:hover {{ background: #7C4DFF; }}
            """)
            
            list_widget = QWidget()
            list_layout = QVBoxLayout(list_widget)
            list_layout.setContentsMargins(0, 8, 0, 8)
            list_layout.setSpacing(6)
            
            # Add ALL moves to the list
            for move in moves:
                from_name = Path(move.get("from", "")).name
                to_folder = Path(move.get("to", "")).parent.name
                
                item = QLabel(f"📄 {from_name}  →  {to_folder}/")
                item.setStyleSheet(f"""
                    font-family: 'Segoe UI'; font-size: 13px; color: {c['text_muted']};
                    padding: 8px 12px; background: {c['input_bg']}; border-radius: 8px;
                """)
                item.setWordWrap(True)
                list_layout.addWidget(item)
            
            # Show renamed files if any
            if renamed:
                spacer = QLabel("")
                spacer.setFixedHeight(10)
                list_layout.addWidget(spacer)
                
                renamed_header = QLabel(f"📝 {len(renamed)} file(s) renamed to avoid duplicates:")
                renamed_header.setStyleSheet("font-family: 'Segoe UI'; font-size: 13px; font-weight: 600; color: #7C4DFF; padding: 4px 0;")
                list_layout.addWidget(renamed_header)
                
                for r in renamed:
                    orig = r.get("original_name", "?")
                    new = r.get("new_name", "?")
                    rename_item = QLabel(f"  {orig}  →  {new}")
                    rename_item.setStyleSheet(f"font-family: 'Segoe UI'; font-size: 12px; color: {c['text_muted']}; padding: 4px 12px;")
                    list_layout.addWidget(rename_item)
            
            list_layout.addStretch()
            scroll.setWidget(list_widget)
            layout.addWidget(scroll, 1)
            
            # Footer
            footer = QHBoxLayout()

            # Revert button — moves files back to their original locations.
            # Disabled (replaced by a label) once a successful revert has
            # already been applied so the same operation can't be undone
            # twice from disk.
            if is_reverted:
                reverted_label = QLabel("✓ Already Reverted")
                reverted_label.setStyleSheet("""
                    font-family: "Segoe UI", sans-serif;
                    font-size: 14px;
                    font-weight: 600;
                    color: #9E9E9E;
                    padding: 8px 16px;
                """)
                footer.addWidget(reverted_label)
            else:
                revert_btn = QPushButton("↩ Revert")
                revert_btn.setMinimumHeight(40)
                revert_btn.setCursor(Qt.PointingHandCursor)
                revert_btn.setStyleSheet("""
                    QPushButton { background: transparent; color: #7C4DFF; border: 2px solid #7C4DFF; border-radius: 10px;
                                  font-weight: 600; font-size: 14px; padding: 8px 24px; }
                    QPushButton:hover { background: #7C4DFF; color: white; }
                """)
                revert_btn.setToolTip("Move all files back to their original locations")
                revert_btn.clicked.connect(lambda: self._revert_organization(log_file, moves, dialog))
                footer.addWidget(revert_btn)

            footer.addStretch()

            ok_btn = QPushButton("Close")
            ok_btn.setMinimumHeight(40)
            ok_btn.setCursor(Qt.PointingHandCursor)
            ok_btn.setStyleSheet("""
                QPushButton { background: #7C4DFF; color: white; border: none; border-radius: 10px;
                              font-weight: 600; font-size: 14px; padding: 8px 28px; }
                QPushButton:hover { background: #9575FF; }
            """)
            ok_btn.clicked.connect(dialog.accept)
            footer.addWidget(ok_btn)
            
            layout.addLayout(footer)
            
            # Make dialog draggable
            def mousePressEvent(event):
                if event.button() == Qt.LeftButton:
                    dialog._drag_pos = event.globalPosition().toPoint() - dialog.frameGeometry().topLeft()
                    event.accept()
            def mouseMoveEvent(event):
                if dialog._drag_pos and event.buttons() == Qt.LeftButton:
                    dialog.move(event.globalPosition().toPoint() - dialog._drag_pos)
                    event.accept()
            def mouseReleaseEvent(event):
                dialog._drag_pos = None
                event.accept()
            
            dialog.mousePressEvent = mousePressEvent
            dialog.mouseMoveEvent = mouseMoveEvent
            dialog.mouseReleaseEvent = mouseReleaseEvent
            
            dialog.exec()
            
        except Exception as e:
            ModernInfoDialog.show_warning(
                self,
                title="Error",
                message="Could not load operation details.",
                details=[str(e)]
            )
    
    def _revert_organization(self, log_file: str, moves: list, details_dialog: QDialog):
        """Move files back to their original (`from`) paths, undoing the
        organization recorded in ``log_file``.

        Behaviour matches the Mac implementation:
        1. Partition the recorded moves into can-revert / cannot-revert
           buckets — a move can only be reverted if the destination still
           exists AND the original source location is free.
        2. Confirm with the user, showing the partition counts.
        3. For each revertible move, ``shutil.move`` the file back and
           update the DB row's path via update_file_path_by_old_path.
        4. Mark the log file ``reverted: true`` so the history badge +
           "Already Reverted" label show next time.
        5. Refresh the history list.
        """
        from pathlib import Path
        import shutil

        logger.info(f"[REVERT] Starting revert for {len(moves)} moves from {log_file}")

        if not moves:
            ModernInfoDialog.show_info(
                self,
                title="Nothing to Revert",
                message="No file moves to revert."
            )
            return

        can_revert = []
        cannot_revert = []
        for move in moves:
            dest_path = Path(move.get("to", ""))
            source_path = Path(move.get("from", ""))
            if not dest_path.exists():
                cannot_revert.append(f"File not found: {dest_path.name}")
            elif source_path.exists():
                cannot_revert.append(f"Original location occupied: {source_path.name}")
            else:
                can_revert.append(move)

        if not can_revert:
            details_list = cannot_revert[:5]
            if len(cannot_revert) > 5:
                details_list.append(f"... and {len(cannot_revert) - 5} more issues")
            ModernInfoDialog.show_warning(
                self,
                title="Cannot Revert",
                message="Cannot revert this organization.",
                details=details_list
            )
            return

        # Confirmation
        warning_details = []
        if cannot_revert:
            warning_details.append(f"{len(cannot_revert)} file(s) cannot be reverted (moved or deleted)")
        warning_details.append(f"{len(can_revert)} file(s) will be moved back to original locations")

        confirmed = ModernConfirmDialog.ask(
            self,
            title="Confirm Revert",
            message=f"Move {len(can_revert)} file(s) back to original locations?",
            details=warning_details,
            yes_text="Revert",
            no_text="Cancel"
        )
        if not confirmed:
            return

        # Perform the revert
        reverted = 0
        errors = []
        for move in can_revert:
            dest_path = Path(move.get("to", ""))
            source_path = Path(move.get("from", ""))
            try:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest_path), str(source_path))
                reverted += 1
                logger.info(f"[REVERT] Moved back: {dest_path.name} -> {source_path.parent.name}/")
                # Best-effort DB sync — if the file isn't tracked, that's fine.
                try:
                    if file_index.update_file_path_by_old_path(str(dest_path), str(source_path)):
                        logger.info(f"[REVERT] DB path updated: {dest_path.name}")
                    else:
                        logger.debug(f"[REVERT] DB path update skipped (file not in DB): {dest_path.name}")
                except Exception as db_err:
                    logger.warning(f"[REVERT] DB path update failed for {dest_path.name}: {db_err}")
            except Exception as e:
                errors.append(f"{dest_path.name}: {str(e)}")

        # Close the details dialog before showing the result
        try:
            details_dialog.accept()
        except Exception:
            pass

        logger.info(
            f"[REVERT] Completed: {reverted} reverted, {len(errors)} errors, "
            f"{len(cannot_revert)} skipped"
        )

        if errors:
            tail = ([f"... and {len(errors) - 5} more errors"] if len(errors) > 5 else [])
            ModernInfoDialog.show_warning(
                self,
                title="Revert Partially Completed",
                message=f"Reverted {reverted} of {len(can_revert)} files.",
                details=errors[:5] + tail
            )
        else:
            ModernInfoDialog.show_info(
                self,
                title="Revert Complete",
                message=f"Successfully moved {reverted} file(s) back to original locations.",
                details=["Files have been restored to their original locations"]
            )

        # Mark the log file as reverted so the badge + "Already Reverted"
        # state shows the next time the dialog opens.
        if reverted > 0:
            try:
                import json as _json
                from datetime import datetime as _dt
                with open(log_file, 'r', encoding='utf-8') as fh:
                    log_data = _json.load(fh)
                log_data['reverted'] = True
                log_data['reverted_count'] = reverted
                log_data['reverted_at'] = _dt.now().isoformat()
                with open(log_file, 'w', encoding='utf-8') as fh:
                    _json.dump(log_data, fh, indent=2)
                logger.info(f"[REVERT] Marked log as reverted: {log_file}")
            except Exception as e:
                logger.warning(f"[REVERT] Could not mark log as reverted: {e}")

        self._refresh_history()

    def _refresh_history(self):
        """Reload the history list after a revert (or any external change)."""
        while self.history_layout.count() > 0:
            item = self.history_layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        self._load_history()

    def _clear_history(self):
        """Clear all history log files."""
        from app.core.settings import settings
        import shutil

        confirmed = ModernConfirmDialog.ask(
            self,
            title="Clear History",
            message="Delete all organization history?",
            details=["This cannot be undone", "Undo operations will no longer be possible for past moves"],
            yes_text="Clear All",
            no_text="Cancel"
        )
        
        if confirmed:
            try:
                moves_dir = settings.get_moves_dir()
                for log_file in moves_dir.glob("moves-*.json"):
                    log_file.unlink()
                
                # Refresh the list
                # Clear existing items
                while self.history_layout.count() > 0:
                    item = self.history_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Show empty state
                empty_label = QLabel("No organization history yet")
                empty_label.setAlignment(Qt.AlignCenter)
                empty_label.setStyleSheet("""
                    font-family: "Segoe UI", sans-serif;
                    font-size: 15px;
                    color: #7A7A90;
                    padding: 40px;
                """)
                self.history_layout.addWidget(empty_label)
                self.history_layout.addStretch()
                
            except Exception as e:
                ModernInfoDialog.show_warning(
                    self,
                    title="Error",
                    message="Could not clear history.",
                    details=[str(e)]
                )
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()


class PinnedDialog(QDialog):
    """
    Modern dialog for managing pinned files/folders that won't be organized.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_page = parent
        self.setWindowTitle("Pinned Items")
        self.setMinimumSize(550, 450)
        self.setModal(True)
        self._drag_pos = None
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._theme_colors = c  # Store for use in other methods
        self.container = QFrame(self)
        self.container.setObjectName("pinnedDialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#pinnedDialogContainer {{
                background-color: {c['surface']};
                border-radius: 24px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add drop shadow
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)
        
        icon_label = QLabel("📌")
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(48, 48)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Pinned Items")
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(36, 36)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 20px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self.accept)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Description
        desc = QLabel("Pinned files and folders will never be organized.")
        desc.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text_muted']};
        """)
        layout.addWidget(desc)
        
        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {c['border']};")
        layout.addWidget(divider)
        
        # Pinned items list container
        self.pinned_list = QWidget()
        self.pinned_layout = QVBoxLayout(self.pinned_list)
        self.pinned_layout.setContentsMargins(0, 0, 0, 0)
        self.pinned_layout.setSpacing(8)
        
        # Scroll area for pinned items
        scroll = QScrollArea()
        scroll.setWidget(self.pinned_list)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['scrollbar_handle']};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #7C4DFF;
            }}
        """)
        layout.addWidget(scroll, 1)
        
        # Load pinned items
        self._load_pinned_items()
        
        # Add new item section
        add_layout = QHBoxLayout()
        add_layout.setSpacing(10)
        
        add_file_btn = QPushButton("📄 Pin File")
        add_file_btn.setMinimumHeight(40)
        add_file_btn.setCursor(Qt.PointingHandCursor)
        add_file_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(124, 77, 255, 0.06);
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.15);
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.12);
                border-color: #7C4DFF;
            }
        """)
        add_file_btn.clicked.connect(self._add_pinned_file)
        add_layout.addWidget(add_file_btn)
        
        add_folder_btn = QPushButton("📁 Pin Folder")
        add_folder_btn.setMinimumHeight(40)
        add_folder_btn.setCursor(Qt.PointingHandCursor)
        add_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(124, 77, 255, 0.06);
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.15);
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.12);
                border-color: #7C4DFF;
            }
        """)
        add_folder_btn.clicked.connect(self._add_pinned_folder)
        add_layout.addWidget(add_folder_btn)
        
        add_layout.addStretch()
        layout.addLayout(add_layout)
        
        # Footer
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        
        clear_btn = QPushButton("Unpin All")
        clear_btn.setMinimumHeight(40)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['input_bg']};
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 10px;
                font-size: 13px;
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                background-color: rgba(211, 47, 47, 0.10);
                border-color: rgba(211, 47, 47, 0.30);
                color: #FF6B6B;
            }}
        """)
        clear_btn.clicked.connect(self._clear_all_pinned)
        footer_layout.addWidget(clear_btn)
        
        close_dialog_btn = QPushButton("Done")
        close_dialog_btn.setMinimumHeight(40)
        close_dialog_btn.setCursor(Qt.PointingHandCursor)
        close_dialog_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
                padding: 8px 28px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        close_dialog_btn.clicked.connect(self.accept)
        footer_layout.addWidget(close_dialog_btn)
        
        layout.addLayout(footer_layout)
    
    def _load_pinned_items(self):
        """Load pinned items from settings."""
        pinned_paths = settings.get_pinned_paths()
        c = self._theme_colors
        
        if not pinned_paths:
            # Show empty state
            empty_label = QLabel("No pinned items yet.\nPin files or folders to protect them from organization.")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet(f"""
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                color: {c['text_muted']};
                padding: 30px;
            """)
            self.pinned_layout.addWidget(empty_label)
            self.pinned_layout.addStretch()
            return
        
        # Add pinned items
        for path in pinned_paths:
            self._add_pinned_item_row(path)
        
        self.pinned_layout.addStretch()
    
    def _add_pinned_item_row(self, path: str):
        """Add a single pinned item row."""
        from pathlib import Path
        c = self._theme_colors
        
        item_frame = QFrame()
        item_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {c['card']};
                border-radius: 12px;
                border: 1px solid {c['border']};
            }}
            QFrame:hover {{
                background-color: rgba(124, 77, 255, 0.06);
                border-color: rgba(124, 77, 255, 0.15);
            }}
        """)
        
        item_layout = QHBoxLayout(item_frame)
        item_layout.setContentsMargins(14, 12, 14, 12)
        item_layout.setSpacing(12)
        
        # Icon - folder or file
        p = Path(path)
        is_folder = p.is_dir() if p.exists() else ('.' not in p.name)
        icon = QLabel("📁" if is_folder else "📄")
        icon.setStyleSheet("font-size: 18px; background: transparent; border: none;")
        icon.setFixedWidth(26)
        item_layout.addWidget(icon)
        
        # Path info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        name_label = QLabel(p.name)
        name_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: {c['text']};
            background: transparent;
            border: none;
        """)
        info_layout.addWidget(name_label)
        
        # Show parent folder
        parent_str = str(p.parent)
        if len(parent_str) > 45:
            parent_str = "..." + parent_str[-42:]
        parent_label = QLabel(parent_str)
        parent_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 11px;
            color: {c['text_muted']};
            background: transparent;
            border: none;
        """)
        info_layout.addWidget(parent_label)
        
        item_layout.addLayout(info_layout, 1)
        
        # Status indicator - exists or not
        if not p.exists():
            status = QLabel("⚠️")
            status.setToolTip("File/folder no longer exists")
            status.setStyleSheet("font-size: 14px; background: transparent; border: none;")
            item_layout.addWidget(status)
        
        # Unpin button - solid red with white X (ALWAYS visible)
        unpin_btn = QPushButton("X")
        unpin_btn.setFixedSize(28, 28)
        unpin_btn.setCursor(Qt.PointingHandCursor)
        unpin_btn.setToolTip("Unpin this item")
        unpin_btn.setStyleSheet("""
            QPushButton {
                background-color: #E53935;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 16px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #C62828;
            }
        """)
        unpin_btn.clicked.connect(lambda checked, p=path: self._unpin_item(p))
        item_layout.addWidget(unpin_btn)
        
        self.pinned_layout.addWidget(item_frame)
    
    def _add_pinned_file(self):
        """Add a file to pinned list via file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Pin",
            "",
            "All Files (*)"
        )
        
        if file_path:
            if settings.add_pinned_path(file_path):
                self._refresh_list()
    
    def _add_pinned_folder(self):
        """Add a folder to pinned list via folder dialog."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Pin",
            ""
        )
        
        if folder_path:
            if settings.add_pinned_path(folder_path):
                self._refresh_list()
    
    def _unpin_item(self, path: str):
        """Remove a path from pinned list."""
        settings.remove_pinned_path(path)
        self._refresh_list()
    
    def _clear_all_pinned(self):
        """Clear all pinned items."""
        if not settings.get_pinned_paths():
            return
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Unpin All Items",
            message="Remove all pinned items?",
            details=["These files/folders will be subject to organization again"],
            yes_text="Unpin All",
            no_text="Cancel"
        )
        
        if confirmed:
            settings.clear_all_pinned()
            self._refresh_list()
    
    def _refresh_list(self):
        """Refresh the pinned items list."""
        # Clear existing items
        while self.pinned_layout.count() > 0:
            item = self.pinned_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Reload
        self._load_pinned_items()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()


class ApplyInstructionsDialog(QDialog):
    """
    Modern dialog for choosing how to apply new instructions to existing files.
    Clean, minimal design with three options.
    """
    
    # Result constants
    REORGANIZE_ALL = 1
    ORGANIZE_AS_IS = 2
    CONTINUE_WATCHING = 3
    
    def __init__(self, parent=None, file_count: int = 0, subfolder_count: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Apply Instructions")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.result_choice = self.CONTINUE_WATCHING
        self._drag_pos = None
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._theme_colors = c
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        self.container = QFrame(self)
        self.container.setObjectName("applyDialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#applyDialogContainer {{
                background-color: {c['surface']};
                border-radius: 20px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add drop shadow
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        icon_label = QLabel("🔄")
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(44, 44)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Apply New Instructions")
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 18px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self._on_continue)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Subtitle with file count
        if subfolder_count > 0:
            subtitle = f"Found {file_count} files in {subfolder_count} subfolders"
        else:
            subtitle = f"Found {file_count} existing files"
        
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text_muted']};
        """)
        layout.addWidget(subtitle_label)
        
        # Options as styled buttons
        options_layout = QVBoxLayout()
        options_layout.setSpacing(10)
        
        # Option 1: Re-organize All
        reorganize_btn = self._create_option_button(
            "🔄 Re-organize All",
            "Flatten folders, then organize fresh",
            primary=True
        )
        reorganize_btn.clicked.connect(self._on_reorganize)
        options_layout.addWidget(reorganize_btn)
        
        # Option 2: Organize As-Is
        organize_btn = self._create_option_button(
            "📂 Organize As-Is",
            "Apply new instructions to current files"
        )
        organize_btn.clicked.connect(self._on_organize)
        options_layout.addWidget(organize_btn)
        
        # Option 3: Continue Watching
        continue_btn = self._create_option_button(
            "⏭️ Skip",
            "Only apply to new files going forward"
        )
        continue_btn.clicked.connect(self._on_continue)
        options_layout.addWidget(continue_btn)
        
        layout.addLayout(options_layout)
    
    def _create_option_button(self, title: str, subtitle: str, primary: bool = False) -> QPushButton:
        """Create a styled option button with title and subtitle."""
        c = self._theme_colors
        btn = QPushButton()
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(56)
        
        # Create layout for button content
        btn_layout = QVBoxLayout(btn)
        btn_layout.setContentsMargins(16, 10, 16, 10)
        btn_layout.setSpacing(2)
        
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: {"#FFFFFF" if primary else c['text']};
            background: transparent;
        """)
        title_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(title_lbl)
        
        subtitle_lbl = QLabel(subtitle)
        subtitle_lbl.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 11px;
            color: {"rgba(255,255,255,0.8)" if primary else c['text_muted']};
            background: transparent;
        """)
        subtitle_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(subtitle_lbl)
        
        if primary:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #7C4DFF;
                    border: none;
                    border-radius: 12px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #9575FF;
                }
                QPushButton:pressed {
                    background-color: #6A3DE8;
                }
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c['surface']};
                    border: 1px solid {c['border']};
                    border-radius: 12px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: rgba(124, 77, 255, 0.10);
                    border-color: #7C4DFF;
                }}
                QPushButton:pressed {{
                    background-color: {c['card']};
                }}
            """)
        
        return btn
    
    def _on_reorganize(self):
        self.result_choice = self.REORGANIZE_ALL
        self.accept()
    
    def _on_organize(self):
        self.result_choice = self.ORGANIZE_AS_IS
        self.accept()
    
    def _on_continue(self):
        self.result_choice = self.CONTINUE_WATCHING
        self.accept()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()


class ApplyInstructionsDialogPerFolder(QDialog):
    """
    Enhanced dialog for choosing how to apply new instructions PER FOLDER.
    Shows each folder with its own set of options (Re-organize All, Organize As-Is, Skip).
    """
    
    # Result constants (same as ApplyInstructionsDialog for compatibility)
    REORGANIZE_ALL = 1
    ORGANIZE_AS_IS = 2
    CONTINUE_WATCHING = 3
    
    def __init__(self, parent=None, folder_info: List[Dict] = None):
        """
        Args:
            parent: Parent widget
            folder_info: List of dicts with 'path', 'file_count', 'subfolder_count' keys
        """
        super().__init__(parent)
        self.setWindowTitle("Apply Instructions")
        self.setMinimumWidth(520)
        self.setMinimumHeight(300)
        self.setModal(True)
        self._drag_pos = None
        
        # Default to empty list
        self.folder_info = folder_info or []
        
        # Store per-folder choices: {folder_path: choice}
        self.folder_choices = {}
        self._radio_groups = {}  # {folder_path: QButtonGroup}
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._theme_colors = c
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        self.container = QFrame(self)
        self.container.setObjectName("applyDialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#applyDialogContainer {{
                background-color: {c['surface']};
                border-radius: 20px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add drop shadow
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        icon_label = QLabel("🔄")
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(44, 44)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Apply New Instructions")
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 18px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self._on_cancel)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Subtitle
        subtitle_label = QLabel("Choose how to handle each folder:")
        subtitle_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text_muted']};
        """)
        layout.addWidget(subtitle_label)
        
        # Scrollable area for folders
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: {c['surface']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['border']};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #7C4DFF;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 8, 0)
        scroll_layout.setSpacing(12)
        
        # Create row for each folder
        for folder_data in self.folder_info:
            folder_path = folder_data.get('path', '')
            file_count = folder_data.get('file_count', 0)
            subfolder_count = folder_data.get('subfolder_count', 0)
            
            folder_widget = self._create_folder_row(folder_path, file_count, subfolder_count)
            scroll_layout.addWidget(folder_widget)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        # Set max height for scroll area
        max_scroll_height = min(300, 80 * len(self.folder_info) + 20)
        scroll.setMaximumHeight(max_scroll_height)
        
        layout.addWidget(scroll)
        
        # Apply button
        apply_btn = QPushButton("Apply Changes")
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.setFixedHeight(44)
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                font-weight: 600;
                border: none;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
            QPushButton:pressed {
                background-color: #6A3DE8;
            }
        """)
        apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(apply_btn)
    
    def _create_folder_row(self, folder_path: str, file_count: int, subfolder_count: int) -> QFrame:
        """Create a styled row for a folder with radio button options."""
        c = self._theme_colors
        
        row = QFrame()
        row.setStyleSheet(f"""
            QFrame {{
                background-color: {c['card']};
                border-radius: 12px;
                border: 1px solid {c['border']};
            }}
        """)
        
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(16, 12, 16, 12)
        row_layout.setSpacing(8)
        
        # Folder info line
        info_layout = QHBoxLayout()
        info_layout.setSpacing(8)
        
        folder_icon = QLabel("📁")
        folder_icon.setStyleSheet("font-size: 16px; background: transparent;")
        info_layout.addWidget(folder_icon)
        
        # Truncate long paths
        display_path = folder_path
        if len(display_path) > 45:
            display_path = "..." + display_path[-42:]
        
        folder_label = QLabel(display_path)
        folder_label.setToolTip(folder_path)
        folder_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            font-weight: 500;
            color: {c['text']};
            background: transparent;
        """)
        info_layout.addWidget(folder_label, 1)
        
        # File count badge
        if subfolder_count > 0:
            count_text = f"{file_count} files, {subfolder_count} subfolders"
        else:
            count_text = f"{file_count} files"
        
        count_label = QLabel(count_text)
        count_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 11px;
            color: {c['text_muted']};
            background-color: rgba(124, 77, 255, 0.10);
            padding: 3px 8px;
            border-radius: 10px;
        """)
        info_layout.addWidget(count_label)
        
        row_layout.addLayout(info_layout)
        
        # Radio buttons for options
        options_layout = QHBoxLayout()
        options_layout.setSpacing(16)
        
        # Create button group for this folder
        btn_group = QButtonGroup(self)
        self._radio_groups[folder_path] = btn_group
        
        # Default to Skip
        self.folder_choices[folder_path] = self.CONTINUE_WATCHING
        
        radio_style = f"""
            QRadioButton {{
                font-family: "Segoe UI", sans-serif;
                font-size: 11px;
                color: {c['text']};
                background: transparent;
                spacing: 4px;
            }}
            QRadioButton::indicator {{
                width: 14px;
                height: 14px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 2px solid {c['border']};
                border-radius: 7px;
                background: transparent;
            }}
            QRadioButton::indicator:checked {{
                border: 2px solid #7C4DFF;
                border-radius: 7px;
                background: #7C4DFF;
            }}
        """
        
        radio_reorganize = QRadioButton("🔄 Re-organize All")
        radio_reorganize.setStyleSheet(radio_style)
        radio_reorganize.setToolTip("Flatten folders, then organize fresh")
        btn_group.addButton(radio_reorganize, self.REORGANIZE_ALL)
        options_layout.addWidget(radio_reorganize)
        
        radio_as_is = QRadioButton("📂 Organize As-Is")
        radio_as_is.setStyleSheet(radio_style)
        radio_as_is.setToolTip("Apply new instructions to current files")
        btn_group.addButton(radio_as_is, self.ORGANIZE_AS_IS)
        options_layout.addWidget(radio_as_is)
        
        radio_skip = QRadioButton("⏭️ Skip")
        radio_skip.setStyleSheet(radio_style)
        radio_skip.setToolTip("Only apply to new files going forward")
        radio_skip.setChecked(True)  # Default selection
        btn_group.addButton(radio_skip, self.CONTINUE_WATCHING)
        options_layout.addWidget(radio_skip)
        
        options_layout.addStretch()
        
        # Connect button group to update choices
        btn_group.idClicked.connect(lambda id, fp=folder_path: self._on_choice_changed(fp, id))
        
        row_layout.addLayout(options_layout)
        
        return row
    
    def _on_choice_changed(self, folder_path: str, choice_id: int):
        """Update folder choice when radio button is clicked."""
        self.folder_choices[folder_path] = choice_id
    
    def _on_apply(self):
        """Apply button clicked - accept dialog with current choices."""
        self.accept()
    
    def _on_cancel(self):
        """Cancel - set all to skip and reject."""
        for folder_path in self.folder_choices:
            self.folder_choices[folder_path] = self.CONTINUE_WATCHING
        self.reject()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()


class ApplyInstructionsDialogSingleFolder(QDialog):
    """
    Dialog for choosing how to apply instructions to a SINGLE folder.
    Shown after saving a folder's settings.
    """
    
    # Result constants
    REORGANIZE_ALL = 1
    ORGANIZE_AS_IS = 2
    SKIP = 3
    
    def __init__(self, parent=None, folder_path: str = "", file_count: int = 0, 
                 subfolder_count: int = 0, instruction: str = "", preselected_action: int = None):
        super().__init__(parent)
        self.setWindowTitle("Choose Organization Mode")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.result_choice = None  # Will be set when user clicks an option
        self._drag_pos = None
        
        self.folder_path = folder_path
        self.file_count = file_count
        self.subfolder_count = subfolder_count
        self.instruction = instruction
        self.preselected_action = preselected_action or self.SKIP
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._theme_colors = c
        
        # Remove default window frame for custom styling
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        self.container = QFrame(self)
        self.container.setObjectName("applyDialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#applyDialogContainer {{
                background-color: {c['surface']};
                border-radius: 20px;
                border: 1px solid {c['border']};
            }}
        """)
        
        # Add drop shadow
        from PySide6.QtWidgets import QGraphicsDropShadowEffect
        from PySide6.QtGui import QColor
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.container)
        
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        icon_label = QLabel("📁")
        icon_label.setStyleSheet("""
            font-size: 24px;
            background-color: rgba(124, 77, 255, 0.08);
            border-radius: 20px;
            border: 1px solid rgba(124, 77, 255, 0.20);
        """)
        icon_label.setFixedSize(44, 44)
        icon_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel("Organization Options")
        title_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: {c['text']};
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        close_btn = QPushButton("X")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 18px;
                font-weight: bold;
                font-family: Arial, Helvetica, sans-serif;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        close_btn.clicked.connect(self._on_skip)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Folder info
        display_path = folder_path
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
        
        folder_info = QLabel(f"📂 {display_path}")
        folder_info.setToolTip(folder_path)
        folder_info.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            font-weight: 500;
            color: {c['text']};
            background-color: {c['card']};
            padding: 10px 14px;
            border-radius: 8px;
        """)
        layout.addWidget(folder_info)
        
        # File count info
        if subfolder_count > 0:
            count_text = f"Found {file_count} files in {subfolder_count} subfolders"
        else:
            count_text = f"Found {file_count} existing files"
        
        subtitle_label = QLabel(count_text)
        subtitle_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text_muted']};
        """)
        layout.addWidget(subtitle_label)
        
        # Question
        question_label = QLabel("What would you like to do with these files?")
        question_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: {c['text']};
            margin-top: 8px;
        """)
        layout.addWidget(question_label)
        
        # Options as styled buttons
        options_layout = QVBoxLayout()
        options_layout.setSpacing(10)
        
        # Option 1: Re-organize All
        is_selected_1 = (self.preselected_action == self.REORGANIZE_ALL)
        reorganize_btn = self._create_option_button(
            "🔄 Re-organize All",
            "Flatten folders, then organize fresh",
            selected=is_selected_1
        )
        reorganize_btn.clicked.connect(self._on_reorganize)
        options_layout.addWidget(reorganize_btn)
        
        # Option 2: Organize As-Is
        is_selected_2 = (self.preselected_action == self.ORGANIZE_AS_IS)
        organize_btn = self._create_option_button(
            "📂 Organize As-Is",
            "Apply instruction to current files",
            selected=is_selected_2
        )
        organize_btn.clicked.connect(self._on_organize)
        options_layout.addWidget(organize_btn)
        
        # Option 3: Skip / Watch Only
        is_selected_3 = (self.preselected_action == self.SKIP)
        skip_btn = self._create_option_button(
            "⏭️ Watch Only",
            "Only apply to new files going forward",
            selected=is_selected_3
        )
        skip_btn.clicked.connect(self._on_skip)
        options_layout.addWidget(skip_btn)
        
        layout.addLayout(options_layout)
    
    def _create_option_button(self, title: str, subtitle: str, selected: bool = False) -> QPushButton:
        """Create a styled option button with title and subtitle."""
        c = self._theme_colors
        btn = QPushButton()
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(56)
        
        # Create layout for button content
        btn_layout = QVBoxLayout(btn)
        btn_layout.setContentsMargins(16, 10, 16, 10)
        btn_layout.setSpacing(2)
        
        # Add checkmark for selected option
        display_title = f"✓ {title}" if selected else title
        
        title_lbl = QLabel(display_title)
        title_lbl.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: {"#FFFFFF" if selected else c['text']};
            background: transparent;
        """)
        title_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(title_lbl)
        
        subtitle_lbl = QLabel(subtitle)
        subtitle_lbl.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 11px;
            color: {"rgba(255,255,255,0.8)" if selected else c['text_muted']};
            background: transparent;
        """)
        subtitle_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        btn_layout.addWidget(subtitle_lbl)
        
        if selected:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #7C4DFF;
                    border: 2px solid #9575FF;
                    border-radius: 12px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #9575FF;
                }
                QPushButton:pressed {
                    background-color: #6A3DE8;
                }
            """)
        else:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c['surface']};
                    border: 1px solid {c['border']};
                    border-radius: 12px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: rgba(124, 77, 255, 0.10);
                    border-color: #7C4DFF;
                }}
                QPushButton:pressed {{
                    background-color: {c['card']};
                }}
            """)
        
        return btn
    
    def _on_reorganize(self):
        self.result_choice = self.REORGANIZE_ALL
        self.accept()
    
    def _on_organize(self):
        self.result_choice = self.ORGANIZE_AS_IS
        self.accept()
    
    def _on_skip(self):
        self.result_choice = self.SKIP
        self.accept()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()


class WatchConfigDialog(QDialog):
    """
    Dialog for configuring Watch & Auto-Organize folders with per-folder instructions.
    Modern redesign to match the app's purple-bluish brand theme.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Watch & Auto-Organize")
        self.setMinimumWidth(650)
        self.setMinimumHeight(550)
        
        # Theme-aware styling
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {c['bg']};
                color: {c['text']};
            }}
            QLabel {{
                color: {c['text']};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {c['scrollbar_handle']};
                border-radius: 3px;
            }}
            QLineEdit {{
                background-color: {c['bg']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 8px 12px;
                color: {c['text']};
            }}
            QLineEdit:focus {{
                border: 1px solid #7C4DFF;
                background-color: {c['card']};
            }}
        """)
        
        # Track folder data: {path: instruction}
        self.folder_data: Dict[str, str] = {}
        # Track folder widgets for updates
        self.folder_widgets: Dict[str, Dict] = {}
        
        # Voice recording state
        self.voice_worker = None
        self.is_recording_voice = False
        self.current_recording_folder = None  # Track which folder's mic is recording
        
        self._setup_ui()
        self._load_from_settings()
    
    def showEvent(self, event):
        """Apply dark/light title bar when dialog is shown."""
        super().showEvent(event)
        from app.ui.theme_manager import apply_titlebar_theme
        apply_titlebar_theme(self)
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)
        
        from app.ui.theme_manager import get_theme_colors as _gtc
        _c = _gtc()
        header = QLabel("Watch & Auto-Organize Configuration")
        header.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {_c['text']};")
        header_layout.addWidget(header)
        
        subtitle = QLabel(
            "Add folders to watch for new files. Each folder can have its own organization instructions."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {_c['text_muted']}; font-size: 14px;")
        header_layout.addWidget(subtitle)
        
        layout.addLayout(header_layout)
        
        # Action Bar (Add Folder)
        action_row = QHBoxLayout()
        
        self.add_folder_btn = QPushButton("+ Add Folder")
        self.add_folder_btn.setMinimumHeight(40)
        self.add_folder_btn.setCursor(Qt.PointingHandCursor)
        self.add_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7C4DFF;
                border: 2px solid #7C4DFF;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.05);
            }
            QPushButton:pressed {
                background-color: rgba(124, 77, 255, 0.1);
            }
        """)
        self.add_folder_btn.clicked.connect(self._add_folder)
        action_row.addWidget(self.add_folder_btn)
        action_row.addStretch()
        
        layout.addLayout(action_row)
        
        # Scroll area for folder list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        self.folders_container = QWidget()
        self.folders_container.setStyleSheet("background-color: transparent;")
        self.folders_layout = QVBoxLayout(self.folders_container)
        self.folders_layout.setContentsMargins(0, 0, 5, 0)
        self.folders_layout.setSpacing(12)
        
        # Placeholder for when no folders
        self.no_folders_label = QLabel("No folders configured.\nClick '+ Add Folder' to start watching.")
        self.no_folders_label.setStyleSheet(f"""
            color: {_c['text_muted']};
            font-size: 14px;
            padding: 40px;
            background: {_c['surface']};
            border-radius: 12px;
            border: 2px dashed {_c['border_strong']};
        """)
        self.no_folders_label.setAlignment(Qt.AlignCenter)
        self.folders_layout.addWidget(self.no_folders_label)
        
        self.folders_layout.addStretch()
        
        scroll.setWidget(self.folders_container)
        layout.addWidget(scroll, 1)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"background-color: {_c['border']}; border: none; max-height: 1px;")
        layout.addWidget(line)
        
        # Bottom Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(44)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {_c['text_muted']};
                border: none;
                font-weight: 500;
                font-size: 14px;
            }}
            QPushButton:hover {{
                color: {_c['text']};
                background-color: {_c['input_bg']};
                border-radius: 8px;
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setMinimumHeight(44)
        self.save_btn.setMinimumWidth(100)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setToolTip("Save all folder settings and apply selected options")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
            QPushButton:pressed {
                background-color: #6A3DE8;
            }
        """)
        self.save_btn.clicked.connect(self._save_and_close)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
    
    def _load_from_settings(self):
        """Load saved folders from settings."""
        for folder_info in settings.auto_organize_folders:
            path = folder_info.get('path', '')
            instruction = folder_info.get('instruction', '')
            if path and os.path.isdir(path):
                self._create_folder_widget(path, instruction)
        
        self._update_no_folders_visibility()
    
    def _add_folder(self):
        """Add a new folder via file dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Watch", str(Path.home())
        )
        if folder:
            # Normalize path
            folder = os.path.normpath(folder)
            
            if folder in self.folder_data:
                # Use modern info dialog
                dialog = ModernInfoDialog(
                    self,
                    title="Folder Already Added",
                    message="This folder is already in your watch list.",
                    info_text=f"'{os.path.basename(folder)}' is already being monitored for auto-organization.",
                    icon="📂",
                    ok_text="Got it"
                )
                dialog.exec()
                return
            
            self._create_folder_widget(folder, '')
            self._update_no_folders_visibility()
    
    def _create_folder_widget(self, folder_path: str, instruction: str):
        """Create a widget card for a folder."""
        folder_path = os.path.normpath(folder_path)
        
        # Get theme colors
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        # Store in data
        self.folder_data[folder_path] = instruction
        
        # Create card frame
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {c['surface']};
                border: 1px solid {c['border_strong']};
                border-radius: 12px;
            }}
        """)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(12)
        frame_layout.setContentsMargins(16, 16, 16, 16)
        
        # Header row with path and remove button
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        
        folder_icon = QLabel("📂")
        folder_icon.setStyleSheet("font-size: 18px; border: none; background: transparent;")
        header_row.addWidget(folder_icon)
        
        path_label = QLabel(folder_path)
        path_label.setStyleSheet(f"font-weight: 600; font-size: 13px; color: {c['text']}; border: none; background: transparent;")
        path_label.setWordWrap(True)
        header_row.addWidget(path_label, 1)
        
        # Options button for this folder (choose organization mode)
        options_btn = QPushButton("Options")
        options_btn.setMinimumHeight(28)
        options_btn.setMinimumWidth(70)
        options_btn.setCursor(Qt.PointingHandCursor)
        options_btn.setToolTip("Choose how to organize this folder")
        options_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
            QPushButton:pressed {
                background-color: #6A3DE8;
            }
        """)
        options_btn.clicked.connect(lambda: self._show_folder_options(folder_path))
        header_row.addWidget(options_btn)
        
        remove_btn = QPushButton("Remove")
        remove_btn.setMinimumHeight(28)
        remove_btn.setMinimumWidth(70)
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.setToolTip("Remove this folder from auto-organize")
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 6px;
                font-weight: 500;
                font-size: 12px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background-color: rgba(211, 47, 47, 0.12);
                color: #D32F2F;
                border-color: #D32F2F;
            }}
        """)
        remove_btn.clicked.connect(lambda: self._remove_folder(folder_path))
        header_row.addWidget(remove_btn)
        
        frame_layout.addLayout(header_row)
        
        # Instruction input
        instruction_layout = QVBoxLayout()
        instruction_layout.setSpacing(6)
        
        instruction_label = QLabel("Organization Instruction (Optional)")
        instruction_label.setStyleSheet(f"color: {c['text_muted']}; font-size: 11px; font-weight: 600; text-transform: uppercase; border: none; background: transparent;")
        instruction_layout.addWidget(instruction_label)
        
        # Input row with text field and mic button
        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        
        instruction_input = QLineEdit()
        instruction_input.setPlaceholderText("e.g. Move screenshots to Images/Screenshots, organize others by type...")
        instruction_input.setText(instruction)
        instruction_input.setMinimumHeight(38)
        instruction_input.textChanged.connect(
            lambda text, fp=folder_path: self._on_instruction_changed(fp, text)
        )
        input_row.addWidget(instruction_input, 1)
        
        # Microphone button
        mic_button = QPushButton("Voice")
        mic_button.setMinimumSize(55, 38)
        mic_button.setMaximumSize(55, 38)
        mic_button.setCursor(Qt.PointingHandCursor)
        mic_button.setToolTip("Click to speak your instruction")
        mic_button.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                font-weight: bold;
                background-color: rgba(124, 77, 255, 0.06);
                border: 1px solid #7C4DFF;
                border-radius: 6px;
                color: #7C4DFF;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.10);
            }
        """)
        mic_button.clicked.connect(lambda checked, fp=folder_path: self._toggle_folder_voice(fp))
        input_row.addWidget(mic_button)
        
        instruction_layout.addLayout(input_row)
        
        frame_layout.addLayout(instruction_layout)
        
        # Status label to show selected organization mode (hidden by default)
        status_label = QLabel("")
        status_label.setStyleSheet(f"""
            QLabel {{
                color: #4CAF50;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 8px;
                background: rgba(76, 175, 80, 0.1);
                border-radius: 4px;
                border: none;
            }}
        """)
        frame_layout.addWidget(status_label)
        
        # Store widgets for later reference
        self.folder_widgets[folder_path] = {
            'widget': frame,
            'input': instruction_input,
            'mic_button': mic_button,
            'options_button': options_btn,
            'status_label': status_label
        }
        
        # Load and display saved action for this folder
        saved_action = settings.get_auto_organize_action(folder_path)
        self._update_folder_status_display(folder_path, saved_action)
        
        # Add to layout (before spacer)
        self.folders_layout.insertWidget(self.folders_layout.count() - 2, frame)
    
    def _remove_folder(self, folder_path: str):
        """Remove a folder from the list."""
        if folder_path in self.folder_widgets:
            # Remove widget
            widget = self.folder_widgets[folder_path]['widget']
            widget.deleteLater()
            del self.folder_widgets[folder_path]
            
            # Remove data
            if folder_path in self.folder_data:
                del self.folder_data[folder_path]
            
            self._update_no_folders_visibility()
    
    def _on_instruction_changed(self, folder_path: str, text: str):
        """Handle instruction text change."""
        if folder_path in self.folder_data:
            self.folder_data[folder_path] = text
            # Save instruction to settings immediately
            settings.update_auto_organize_instruction(folder_path, text)
    
    def _update_folder_status_display(self, folder_path: str, action: int):
        """Update the status label to show the current organization mode."""
        if folder_path not in self.folder_widgets:
            return
        
        status_label = self.folder_widgets[folder_path].get('status_label')
        if not status_label:
            return
        
        # Define action labels and styles
        action_configs = {
            1: ("🔄 Re-organize All", "#7C4DFF", "rgba(124, 77, 255, 0.15)"),  # Purple
            2: ("📂 Organize As-Is", "#2196F3", "rgba(33, 150, 243, 0.15)"),   # Blue
            3: ("⏭️ Watch Only", "#4CAF50", "rgba(76, 175, 80, 0.15)"),        # Green
        }
        
        config = action_configs.get(action, action_configs[3])
        label_text, text_color, bg_color = config
        
        status_label.setText(label_text)
        status_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 11px;
                font-weight: 600;
                padding: 6px 10px;
                background: {bg_color};
                border-radius: 6px;
                border: none;
            }}
        """)
        status_label.setVisible(True)
    
    def _show_folder_options(self, folder_path: str):
        """Show the options dialog for a folder."""
        folder_path = os.path.normpath(folder_path)
        instruction = self.folder_data.get(folder_path, '')
        
        # Save instruction first
        settings.update_auto_organize_instruction(folder_path, instruction)
        
        # Count files in this folder
        file_count = 0
        subfolder_count = 0
        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    file_count += 1
                elif os.path.isdir(item_path) and not item.startswith('.'):
                    subfolder_count += 1
                    for sub_item in os.listdir(item_path):
                        if os.path.isfile(os.path.join(item_path, sub_item)):
                            file_count += 1
        except Exception:
            pass
        
        # Get the previously selected action
        saved_action = settings.get_auto_organize_action(folder_path)
        
        # Show single-folder apply dialog with pre-selected option
        dialog = ApplyInstructionsDialogSingleFolder(
            self, 
            folder_path, 
            file_count, 
            subfolder_count,
            instruction,
            preselected_action=saved_action  # Pass the saved action
        )
        dialog.exec()
        
        if dialog.result_choice:
            # Save the selected action (but don't apply yet - wait for Save button)
            settings.update_auto_organize_action(folder_path, dialog.result_choice)
            
            # Update status display to show what will be applied
            self._update_folder_status_display(folder_path, dialog.result_choice)
            
            # Note: Action will be applied when user clicks "Save" button
    
    def _update_no_folders_visibility(self):
        """Show/hide placeholder based on folder count."""
        has_folders = len(self.folder_data) > 0
        self.no_folders_label.setVisible(not has_folders)
    
    def _save_and_close(self):
        """Save settings and apply selected actions for each folder."""
        try:
            logger.info(f"WatchConfigDialog._save_and_close called with {len(self.folder_data)} folders")
            
            # Collect folders with their actions
            folders_to_apply = []
            new_folders = []
            
            for path, instruction in self.folder_data.items():
                logger.info(f"  Saving folder: {path}, instruction: {instruction[:50] if instruction else '(empty)'}...")
                # Get existing action if any
                existing_action = settings.get_auto_organize_action(path)
                new_folders.append({
                    'path': path,
                    'instruction': instruction,
                    'action': existing_action  # Preserve the action setting
                })
                # Track folders that need action applied
                if existing_action in [1, 2]:  # 1=Re-organize, 2=As-Is
                    folders_to_apply.append((path, existing_action))
            
            # Save settings first
            settings.auto_organize_folders = new_folders
            settings._save_config()
            logger.info("Settings saved")
            
            # Apply actions regardless of whether watcher is currently running
            parent = self.parent()
            if parent and hasattr(parent, 'auto_watcher') and parent.auto_watcher:
                for folder_path, action in folders_to_apply:
                    logger.info(f"Applying action {action} to folder: {folder_path}")
                    if action == ApplyInstructionsDialogSingleFolder.REORGANIZE_ALL:
                        parent.auto_watcher.organize_single_folder(folder_path, flatten_first=True)
                    elif action == ApplyInstructionsDialogSingleFolder.ORGANIZE_AS_IS:
                        parent.auto_watcher.organize_single_folder(folder_path, flatten_first=False)
            
            logger.info("Actions applied, closing dialog")
            self.accept()
        except Exception as e:
            logger.error(f"Error in _save_and_close: {e}")
            import traceback
            traceback.print_exc()
    
    def _toggle_folder_voice(self, folder_path: str):
        """Toggle voice recording for a specific folder's instruction."""
        if self.is_recording_voice:
            # Stop current recording
            self._stop_folder_voice()
        else:
            # Start recording for this folder
            self._start_folder_voice(folder_path)
    
    def _start_folder_voice(self, folder_path: str):
        """Start recording voice for a folder's instruction."""
        self.is_recording_voice = True
        self.current_recording_folder = folder_path
        
        # Update the mic button to show recording state
        if folder_path in self.folder_widgets:
            mic_btn = self.folder_widgets[folder_path].get('mic_button')
            if mic_btn:
                mic_btn.setText("Stop")
                mic_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 11px;
                        font-weight: bold;
                        background-color: #EF5350;
                        border: 1px solid #D32F2F;
                        border-radius: 6px;
                        color: white;
                        padding: 0px;
                    }
                    QPushButton:hover {
                        background-color: #E53935;
                    }
                """)
                mic_btn.setToolTip("Recording... Click to stop")
        
        # Start voice worker
        self.voice_worker = VoiceRecordWorker()
        self.voice_worker.finished.connect(self._on_folder_voice_transcribed)
        self.voice_worker.error.connect(self._on_folder_voice_error)
        self.voice_worker.recording_stopped.connect(self._on_folder_recording_stopped)
        self.voice_worker.start()
    
    def _stop_folder_voice(self):
        """Stop recording voice input."""
        if self.voice_worker:
            self.voice_worker.stop_recording()
    
    def _on_folder_recording_stopped(self):
        """Called when recording has stopped, before transcription."""
        self.is_recording_voice = False
        self._reset_folder_mic_button()
    
    def _on_folder_voice_transcribed(self, text: str):
        """Handle transcribed text for a folder's instruction."""
        self.is_recording_voice = False
        self._reset_folder_mic_button()
        
        if text.strip() and self.current_recording_folder:
            folder_path = self.current_recording_folder
            if folder_path in self.folder_widgets:
                instruction_input = self.folder_widgets[folder_path].get('input')
                if instruction_input:
                    # Append to existing or replace
                    current = instruction_input.text().strip()
                    if current:
                        instruction_input.setText(f"{current}. {text}")
                    else:
                        instruction_input.setText(text)
        
        self.current_recording_folder = None
    
    def _on_folder_voice_error(self, error: str):
        """Handle voice recording errors."""
        self.is_recording_voice = False
        self._reset_folder_mic_button()
        self.current_recording_folder = None
        
        # Show error in a simple message
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Voice Error", f"Could not transcribe: {error}")
    
    def _reset_folder_mic_button(self):
        """Reset mic button to default state for the current recording folder."""
        if self.current_recording_folder and self.current_recording_folder in self.folder_widgets:
            mic_btn = self.folder_widgets[self.current_recording_folder].get('mic_button')
            if mic_btn:
                mic_btn.setText("Voice")
                mic_btn.setStyleSheet("""
                    QPushButton {
                        font-size: 11px;
                        font-weight: bold;
                        background-color: rgba(124, 77, 255, 0.06);
                        border: 1px solid #7C4DFF;
                        border-radius: 6px;
                        color: #7C4DFF;
                        padding: 0px;
                    }
                    QPushButton:hover {
                        background-color: rgba(124, 77, 255, 0.10);
                    }
                """)
                mic_btn.setToolTip("Click to speak your instruction")
    
    def get_folder_count(self) -> int:
        """Get the number of configured folders."""
        return len(self.folder_data)



class OrganizePage(QWidget):
    """
    AI Organization page widget.
    
    Implements the safe organization flow:
    - AI decides what should happen (proposes plan)
    - App decides what actually happens (validates + executes)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_plan = None
        self.current_moves = []
        self.files_by_id = {}
        self.destination_path = None
        self.plan_worker = None
        # Undo tracking - stores the last completed organization
        self.last_organization = None  # List of {source, destination, file_id}
        # Refinement tracking
        self.original_instruction = None
        
        # Watch & Auto-Organize
        self.auto_watcher = None
        self.watch_folders: List[str] = []
        self._init_auto_watcher()
        
        self.setup_ui()
        
        # Apply theme-aware styles and connect to theme changes
        from app.ui.theme_manager import theme_manager
        self._apply_theme_styles(theme_manager.current_theme)
        theme_manager.theme_changed.connect(self._apply_theme_styles)
        
        # Check auto-start after UI is fully ready (2 second delay to prevent lag)
        QTimer.singleShot(2000, self._check_auto_start)
    
    def _init_auto_watcher(self):
        """Initialize the auto-organize watcher."""
        from app.core.auto_watcher import AutoOrganizeWatcher
        
        self.auto_watcher = AutoOrganizeWatcher(self)
        self.auto_watcher.file_organized.connect(self._on_watch_file_organized)
        self.auto_watcher.file_indexed.connect(self._on_watch_file_indexed)
        self.auto_watcher.status_changed.connect(self._on_watch_status)
        self.auto_watcher.error_occurred.connect(self._on_watch_error)
        self.auto_watcher.limit_reached.connect(self._on_watch_limit_reached)
    
    def setup_ui(self):
        """Setup the organization page UI."""
        # Main layout for this widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area to handle overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container widget inside scroll area
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("Filect")
        header.setObjectName("heroHeading")
        layout.addWidget(header)
        
        self.subtitle = QLabel(
            "Describe how you want your files organized in plain English. "
            "AI will analyze your indexed files and propose an organization plan."
        )
        self.subtitle.setObjectName("heroSubtitle")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)
        
        # ========== SEGMENTED CONTROL (TAB SWITCHER) ==========
        tab_container = QHBoxLayout()
        tab_container.setSpacing(0)
        tab_container.addStretch()
        
        self.tab_organize_now = QPushButton("✨ Organize Now")
        self.tab_organize_now.setMinimumHeight(44)
        self.tab_organize_now.setMinimumWidth(160)
        self.tab_organize_now.setCursor(Qt.PointingHandCursor)
        self.tab_organize_now.setCheckable(True)
        self.tab_organize_now.setChecked(True)
        self.tab_organize_now.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                color: white;
                border: 1px solid #7C4DFF;
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
                border-top-right-radius: 0px;
                border-bottom-right-radius: 0px;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton:checked {
                background-color: #7C4DFF;
                color: white;
            }
            QPushButton:!checked {
                background-color: #16161F;
                color: #7A7A90;
                border: 1px solid #252535;
            }
            QPushButton:!checked:hover {
                background-color: rgba(124, 77, 255, 0.06);
                border-color: rgba(124, 77, 255, 0.30);
                color: #B39DFF;
            }
        """)
        tab_container.addWidget(self.tab_organize_now)
        
        self.tab_auto_organize = QPushButton("👁️ Auto-Organize")
        self.tab_auto_organize.setMinimumHeight(44)
        self.tab_auto_organize.setMinimumWidth(160)
        self.tab_auto_organize.setCursor(Qt.PointingHandCursor)
        self.tab_auto_organize.setCheckable(True)
        self.tab_auto_organize.setChecked(False)
        self.tab_auto_organize.setStyleSheet("""
            QPushButton {
                background-color: #16161F;
                color: #7A7A90;
                border: 1px solid #252535;
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton:checked {
                background-color: #7C4DFF;
                color: white;
                border-color: #7C4DFF;
            }
            QPushButton:!checked {
                background-color: #16161F;
                color: #7A7A90;
            }
            QPushButton:!checked:hover {
                background-color: rgba(124, 77, 255, 0.06);
                border-color: rgba(124, 77, 255, 0.30);
                color: #B39DFF;
            }
        """)
        tab_container.addWidget(self.tab_auto_organize)
        
        tab_container.addStretch()
        layout.addLayout(tab_container)
        
        layout.addSpacing(10)
        
        # ========== STACKED WIDGET FOR TAB CONTENT ==========
        self.content_stack = QStackedWidget()
        
        # ----- PAGE 0: Organize Now -----
        self.organize_now_page = QWidget()
        organize_now_page = self.organize_now_page
        organize_now_layout = QVBoxLayout(organize_now_page)
        organize_now_layout.setContentsMargins(0, 0, 0, 0)
        organize_now_layout.setSpacing(20)
        
        # Instruction Input Card
        self.instruction_card = QFrame()
        self.instruction_card.setObjectName("organizeCard")
        self.instruction_card.setStyleSheet("""
            QFrame#organizeCard {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 20px;
                padding: 24px;
            }
        """)
        instruction_layout = QVBoxLayout(self.instruction_card)
        instruction_layout.setContentsMargins(20, 20, 20, 20)
        instruction_layout.setSpacing(12)
        
        # Section title
        inst_title = QLabel("✨ Your Instruction")
        inst_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #7C4DFF; background: transparent;")
        instruction_layout.addWidget(inst_title)
        
        # Input row with text field and mic button
        input_row = QHBoxLayout()
        input_row.setSpacing(12)
        
        self.instruction_input = QLineEdit()
        self.instruction_input.setPlaceholderText(
            "e.g., Organize thumbnails by client name or Sort invoices by year"
        )
        self.instruction_input.setMinimumHeight(50)
        self.instruction_input.setStyleSheet("""
            QLineEdit {
                font-size: 15px;
                padding: 12px 16px;
                background-color: #0F0F1A;
                border: 1px solid #1C1C28;
                border-radius: 12px;
                color: #E8E8F0;
            }
            QLineEdit:focus {
                border: 1px solid #7C4DFF;
                background-color: #12121E;
            }
            QLineEdit::placeholder {
                color: #4A4A5A;
            }
        """)
        self.instruction_input.textChanged.connect(self._update_generate_button)
        self.instruction_input.returnPressed.connect(self.generate_plan)
        input_row.addWidget(self.instruction_input)
        
        # Microphone button for voice input
        self.mic_button = QPushButton("🎤")
        self.mic_button.setMinimumHeight(50)
        self.mic_button.setMinimumWidth(60)
        self.mic_button.setMaximumWidth(60)
        self.mic_button.setToolTip("Click to speak your instruction (click again to stop)")
        self.mic_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #16161F;
                border: 1px solid #1C1C28;
                border-radius: 12px;
                color: #E8E8F0;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
            QPushButton:pressed {
                background-color: rgba(124, 77, 255, 0.12);
            }
        """)
        self.mic_button.clicked.connect(self._toggle_voice_recording)
        input_row.addWidget(self.mic_button)
        
        instruction_layout.addLayout(input_row)
        
        # Voice recording state
        self.voice_worker = None
        self.is_recording_voice = False
        
        self._examples_label = QLabel(
            "💡 Examples: Organize by file type, Group photos by date, Sort by topic"
        )
        self._examples_label.setStyleSheet("color: #4A4A5A; font-size: 12px; background: transparent;")
        self._examples_label.setWordWrap(True)
        instruction_layout.addWidget(self._examples_label)
        
        organize_now_layout.addWidget(self.instruction_card)
        
        # Destination Folder Card
        self.dest_card = QFrame()
        self.dest_card.setObjectName("organizeCard")
        self.dest_card.setStyleSheet("""
            QFrame#organizeCard {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 20px;
            }
        """)
        dest_layout = QHBoxLayout(self.dest_card)
        dest_layout.setContentsMargins(20, 16, 20, 16)
        dest_layout.setSpacing(16)
        
        dest_icon = QLabel("📂")
        dest_icon.setStyleSheet("font-size: 24px; background: transparent;")
        dest_layout.addWidget(dest_icon)
        
        dest_info = QVBoxLayout()
        dest_info.setSpacing(4)
        
        dest_title = QLabel("Destination Folder")
        dest_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #7C4DFF; background: transparent;")
        dest_info.addWidget(dest_title)
        
        self.dest_label = QLabel("Select where organized files will be moved...")
        self.dest_label.setStyleSheet("color: #7A7A90; font-size: 13px; background: transparent;")
        self.dest_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        dest_info.addWidget(self.dest_label)
        
        dest_layout.addLayout(dest_info, 1)
        
        self.dest_button = QPushButton("Choose Folder")
        self.dest_button.setMinimumHeight(40)
        self.dest_button.setMinimumWidth(140)
        self.dest_button.setCursor(Qt.PointingHandCursor)
        self.dest_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
        """)
        self.dest_button.clicked.connect(self.select_destination)
        dest_layout.addWidget(self.dest_button)
        
        organize_now_layout.addWidget(self.dest_card)

        # Action Buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(12)
        
        self.generate_button = QPushButton("✨ Generate Plan")
        self.generate_button.setMinimumHeight(48)
        self.generate_button.setMinimumWidth(180)
        self.generate_button.setEnabled(False)
        self.generate_button.setCursor(Qt.PointingHandCursor)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(124, 77, 255, 0.08);
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.35);
                border-radius: 12px;
                font-weight: 700;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #7C4DFF;
                color: white;
                border-color: #7C4DFF;
            }
            QPushButton:disabled {
                background-color: #16161F;
                border: 1px solid #1C1C28;
                color: #4A4A5A;
            }
        """)
        self.generate_button.clicked.connect(self.generate_plan)
        action_layout.addWidget(self.generate_button)
        
        self.apply_button = QPushButton("✓ Apply Organization")
        self.apply_button.setMinimumHeight(48)
        self.apply_button.setMinimumWidth(200)
        self.apply_button.setEnabled(False)
        self.apply_button.setCursor(Qt.PointingHandCursor)
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(76, 175, 80, 0.08);
                color: #4CAF50;
                border: 1px solid rgba(76, 175, 80, 0.35);
                border-radius: 12px;
                font-weight: 700;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #4CAF50;
                color: white;
                border-color: #4CAF50;
            }
            QPushButton:disabled {
                background-color: #16161F;
                border-color: #1C1C28;
                color: #4A4A5A;
            }
        """)
        self.apply_button.clicked.connect(self.apply_organization)
        action_layout.addWidget(self.apply_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMinimumHeight(48)
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
            }
        """)
        self.clear_button.clicked.connect(self.clear_plan)
        action_layout.addWidget(self.clear_button)
        
        self.undo_button = QPushButton("↩ Undo Last")
        self.undo_button.setMinimumHeight(48)
        self.undo_button.setMinimumWidth(130)
        self.undo_button.setEnabled(False)
        self.undo_button.setCursor(Qt.PointingHandCursor)
        self.undo_button.setToolTip("Undo the last organization (move files back)")
        self.undo_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #9575FF;
                border: 1px solid rgba(149, 117, 255, 0.30);
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                color: #B39DFF;
                border-color: #7C4DFF;
            }
            QPushButton:disabled {
                background-color: #16161F;
                border-color: #1C1C28;
                color: #4A4A5A;
            }
        """)
        self.undo_button.clicked.connect(self.undo_last_organization)
        action_layout.addWidget(self.undo_button)
        
        # History button - shows past organization operations
        self.history_button = QPushButton("📋 History")
        self.history_button.setMinimumHeight(48)
        self.history_button.setMinimumWidth(130)
        self.history_button.setCursor(Qt.PointingHandCursor)
        self.history_button.setToolTip("View past organization operations")
        self.history_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7A7A90;
                border: 1px solid #252535;
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.06);
                border-color: #7C4DFF;
                color: #B39DFF;
            }
        """)
        self.history_button.clicked.connect(self._show_history_dialog)
        action_layout.addWidget(self.history_button)
        
        # Pinned button - manage pinned files/folders
        self.pinned_button = QPushButton("📌 Pinned")
        self.pinned_button.setMinimumHeight(48)
        self.pinned_button.setMinimumWidth(130)
        self.pinned_button.setCursor(Qt.PointingHandCursor)
        self.pinned_button.setToolTip("View and manage pinned files/folders that won't be organized")
        self.pinned_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7A7A90;
                border: 1px solid #252535;
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.06);
                border-color: #7C4DFF;
                color: #B39DFF;
            }
        """)
        self.pinned_button.clicked.connect(self._show_pinned_dialog)
        action_layout.addWidget(self.pinned_button)
        
        # Edit button (to show inputs again after plan generation)
        self.edit_inputs_button = QPushButton("✏️ Edit")
        self.edit_inputs_button.setMinimumHeight(48)
        self.edit_inputs_button.setCursor(Qt.PointingHandCursor)
        self.edit_inputs_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7A7A90;
                border: 1px solid #252535;
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
                padding: 0px 20px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.06);
                border-color: #7C4DFF;
                color: #B39DFF;
            }
        """)
        self.edit_inputs_button.clicked.connect(self._show_input_cards)
        self.edit_inputs_button.setVisible(False)
        action_layout.addWidget(self.edit_inputs_button)
        
        # No stretch - let buttons stay left-aligned
        
        # Hide these buttons initially - shown after plan is generated
        self.apply_button.setVisible(False)
        self.clear_button.setVisible(False)
        self.undo_button.setVisible(False)
        
        # Wrap action buttons in a scroll area to prevent cutoff on small windows
        action_widget = QWidget()
        action_widget.setLayout(action_layout)
        action_widget.setStyleSheet("background: transparent;")
        
        self._action_scroll = QScrollArea()
        self._action_scroll.setWidget(action_widget)
        self._action_scroll.setWidgetResizable(True)
        self._action_scroll.setFixedHeight(70)  # Fixed height for the button row + scrollbar space
        self._action_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._action_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._action_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:horizontal {
                border: none;
                background: #1C1C28;
                height: 8px;
                border-radius: 4px;
                margin-top: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #252535;
                border-radius: 4px;
                min-width: 40px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #7C4DFF;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        organize_now_layout.addWidget(self._action_scroll)
        
        # Progress and Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(8)
        organize_now_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #7A7A90; font-style: italic; font-size: 13px;")
        organize_now_layout.addWidget(self.status_label)

        # Results Area (Splitter: Tree + Details) - Hidden until plan is generated
        self.results_splitter = QSplitter(Qt.Horizontal)
        self.results_splitter.setChildrenCollapsible(False)
        
        # Plan Tree Card - Matching the clean input card style
        plan_card = QFrame()
        plan_card.setObjectName("planCard")
        plan_card.setStyleSheet("""
            QFrame#planCard {
                background-color: rgba(124, 77, 255, 0.06);
                border: 2px dashed rgba(124, 77, 255, 0.5);
                border-radius: 20px;
            }
        """)
        
        plan_layout = QVBoxLayout(plan_card)
        plan_layout.setContentsMargins(20, 20, 20, 20)
        plan_layout.setSpacing(12)
        
        # Simple title matching input card style
        plan_title = QLabel("📁 Proposed Organization")
        plan_title.setStyleSheet("""
            font-family: "Segoe UI", sans-serif;
            font-weight: 600;
            font-size: 16px;
            color: #7C4DFF;
            background: transparent;
        """)
        plan_layout.addWidget(plan_title)
        
        self.plan_tree = QTreeWidget()
        self.plan_tree.setHeaderHidden(True)
        self.plan_tree.setIndentation(20)
        self.plan_tree.setAlternatingRowColors(False)
        self.plan_tree.setStyleSheet("""
            QTreeWidget {
                background-color: transparent;
                border: none;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                padding: 4px;
                outline: none;
            }
            QTreeWidget::item {
                height: 38px;
                color: #E8E8F0;
                border-radius: 10px;
                padding-left: 8px;
                margin: 2px 0px;
            }
            QTreeWidget::item:hover {
                background-color: rgba(255, 255, 255, 0.6);
            }
            QTreeWidget::item:selected {
                background-color: rgba(255, 255, 255, 0.8);
                color: #7C4DFF;
                font-weight: 600;
            }
            /* Hide native branch indicators completely */
            QTreeView::branch {
                background: transparent;
                width: 0px;
                border: none;
                image: none;
            }
        """)
        self.plan_tree.setRootIsDecorated(False)  # Remove native expand buttons
        self.plan_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.plan_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plan_tree.customContextMenuRequested.connect(self._show_tree_context_menu)
        plan_layout.addWidget(self.plan_tree)
        
        self.results_splitter.addWidget(plan_card)
        
        self.results_splitter.setVisible(False)  # Hidden until plan is generated
        
        # Summary line (shown after plan generation) - subtle and clean
        self.plan_summary_label = QLabel("")
        self.plan_summary_label.setStyleSheet("""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
            font-weight: 500;
            color: #7A7A90;
            padding: 4px 0px;
        """)
        self.plan_summary_label.setVisible(False)
        organize_now_layout.addWidget(self.plan_summary_label)
        
        # Info note for existing folders (subtle, not alarming)
        self.existing_folders_note = QLabel("")
        self.existing_folders_note.setWordWrap(True)
        self.existing_folders_note.setStyleSheet("""
            font-family: "Segoe UI", sans-serif;
            font-size: 12px;
            color: #4A4A5A;
            font-style: italic;
            padding: 2px 0px;
        """)
        self.existing_folders_note.setVisible(False)
        organize_now_layout.addWidget(self.existing_folders_note)
        
        organize_now_layout.addWidget(self.results_splitter, 1)
        
        # Feedback/Refinement Section (hidden until plan is generated)
        self.feedback_group = QGroupBox("Refine Plan")
        self.feedback_group.setVisible(False)
        feedback_layout = QHBoxLayout(self.feedback_group)
        
        self.feedback_input = QLineEdit()
        self.feedback_input.setPlaceholderText(
            "e.g., 'Move the JSON files to a separate folder' or 'Don't include the screenshots'"
        )
        self.feedback_input.setMinimumHeight(36)
        self.feedback_input.returnPressed.connect(self.refine_plan)
        feedback_layout.addWidget(self.feedback_input, 1)
        
        self.refine_button = QPushButton("🔄 Refine")
        self.refine_button.setMinimumHeight(42)
        self.refine_button.setMinimumWidth(110)
        self.refine_button.setCursor(Qt.PointingHandCursor)
        self.refine_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9575FF, stop:1 #B39DFF);
            }
        """)
        self.refine_button.clicked.connect(self.refine_plan)
        feedback_layout.addWidget(self.refine_button)
        
        organize_now_layout.addWidget(self.feedback_group)
        organize_now_layout.addStretch()
        
        self.content_stack.addWidget(organize_now_page)
        
        # ----- PAGE 1: Auto-Organize -----
        self.auto_organize_page = QWidget()
        auto_organize_page = self.auto_organize_page
        auto_organize_layout = QVBoxLayout(auto_organize_page)
        auto_organize_layout.setContentsMargins(0, 0, 0, 0)
        auto_organize_layout.setSpacing(20)
        
        # ========== WATCH & AUTO-ORGANIZE SECTION ==========
        self._create_auto_organize_section(auto_organize_layout)
        
        auto_organize_layout.addStretch()
        self.content_stack.addWidget(auto_organize_page)
        
        layout.addWidget(self.content_stack, 1)
        
        # Connect tab buttons
        self.tab_organize_now.clicked.connect(lambda: self._switch_tab(0))
        self.tab_auto_organize.clicked.connect(lambda: self._switch_tab(1))
        
        # Finalize scroll area
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Load initial state
        self._update_file_count()
    
    def _create_auto_organize_section(self, parent_layout):
        """Create the Watch & Auto-Organize section matching app theme."""
        # Main card container
        self.watch_card = QFrame()
        self.watch_card.setObjectName("watchAutoCard")
        self.watch_card.setStyleSheet("""
            QFrame#watchAutoCard {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 20px;
            }
        """)
        watch_layout = QVBoxLayout(self.watch_card)
        watch_layout.setSpacing(12)
        watch_layout.setContentsMargins(24, 24, 24, 24)
        
        # Header with icon and title
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        
        watch_icon = QLabel("🔄")
        watch_icon.setStyleSheet("font-size: 28px; background: transparent;")
        header_row.addWidget(watch_icon)
        
        header_info = QVBoxLayout()
        header_info.setSpacing(2)
        
        watch_title = QLabel("Auto-Organize")
        watch_title.setStyleSheet("font-size: 18px; font-weight: 600; color: #7C4DFF; background: transparent;")
        header_info.addWidget(watch_title)
        
        self._watch_desc = QLabel("Monitor folders and organize new files automatically")
        self._watch_desc.setStyleSheet("color: #7A7A90; font-size: 13px; background: transparent;")
        header_info.addWidget(self._watch_desc)
        
        header_row.addLayout(header_info, 1)
        watch_layout.addLayout(header_row)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: rgba(124, 77, 255, 0.2); border: none; max-height: 1px;")
        watch_layout.addWidget(separator)
        
        # Combined folder count + status on one line
        self.watch_folder_label = QLabel("📁 No folders configured")
        self.watch_folder_label.setStyleSheet("""
            font-size: 13px; 
            color: #7A7A90; 
            background: transparent;
            padding: 4px 0;
        """)
        watch_layout.addWidget(self.watch_folder_label)
        
        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        # Edit button (previously Configure)
        self.watch_config_btn = QPushButton("✏️ Edit")
        self.watch_config_btn.setMinimumHeight(42)
        self.watch_config_btn.setMinimumWidth(100)
        self.watch_config_btn.setCursor(Qt.PointingHandCursor)
        self.watch_config_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
        """)
        self.watch_config_btn.clicked.connect(self._open_watch_config)
        btn_row.addWidget(self.watch_config_btn)
        
        # Start/Stop button (purple theme)
        self.watch_toggle_btn = QPushButton("▶ Start")
        self.watch_toggle_btn.setMinimumHeight(42)
        self.watch_toggle_btn.setMinimumWidth(120)
        self.watch_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.watch_toggle_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
                padding: 0 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9575FF, stop:1 #B39DFF);
            }
            QPushButton:disabled {
                background: rgba(124, 77, 255, 0.3);
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        self.watch_toggle_btn.clicked.connect(self._toggle_watch_mode)
        btn_row.addWidget(self.watch_toggle_btn)
        
        btn_row.addStretch()
        watch_layout.addLayout(btn_row)
        
        # Hidden summary label for compatibility
        self.watch_summary_label = QLabel("")
        self.watch_summary_label.setVisible(False)
        
        parent_layout.addWidget(self.watch_card)
        
        # Initial UI update
        self._update_watch_summary()
    
    def _apply_theme_styles(self, theme=None):
        """Re-apply all theme-dependent inline styles for the current theme."""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors(theme)

        # ---- Cards ----
        self.instruction_card.setStyleSheet(f"""
            QFrame#organizeCard {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 20px;
                padding: 24px;
            }}
        """)
        self.dest_card.setStyleSheet(f"""
            QFrame#organizeCard {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 20px;
            }}
        """)
        self.watch_card.setStyleSheet(f"""
            QFrame#watchAutoCard {{
                background-color: {c['surface']};
                border: 1px solid {c['border']};
                border-radius: 20px;
            }}
        """)

        # ---- Inputs ----
        self.instruction_input.setStyleSheet(f"""
            QLineEdit {{
                font-size: 15px;
                padding: 12px 16px;
                background-color: {c['bg']};
                border: 1px solid {c['border']};
                border-radius: 12px;
                color: {c['text']};
            }}
            QLineEdit:focus {{
                border: 1px solid #7C4DFF;
                background-color: {c['card']};
            }}
            QLineEdit::placeholder {{
                color: {c['text_disabled']};
            }}
        """)

        # ---- Mic button ----
        self.mic_button.setStyleSheet(f"""
            QPushButton {{
                font-size: 18px;
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 12px;
                color: {c['text']};
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }}
            QPushButton:pressed {{
                background-color: rgba(124, 77, 255, 0.12);
            }}
        """)

        # ---- Tab buttons ----
        self.tab_organize_now.setStyleSheet(f"""
            QPushButton {{
                background-color: #7C4DFF;
                color: white;
                border: 1px solid #7C4DFF;
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
                border-top-right-radius: 0px;
                border-bottom-right-radius: 0px;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 20px;
            }}
            QPushButton:checked {{
                background-color: #7C4DFF;
                color: white;
            }}
            QPushButton:!checked {{
                background-color: {c['tab_unchecked_bg']};
                color: {c['tab_unchecked_text']};
                border: 1px solid {c['tab_unchecked_border']};
            }}
            QPushButton:!checked:hover {{
                background-color: rgba(124, 77, 255, 0.06);
                border-color: rgba(124, 77, 255, 0.30);
                color: #B39DFF;
            }}
        """)
        self.tab_auto_organize.setStyleSheet(f"""
            QPushButton {{
                background-color: {c['tab_unchecked_bg']};
                color: {c['tab_unchecked_text']};
                border: 1px solid {c['tab_unchecked_border']};
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
                font-weight: 600;
                font-size: 14px;
                padding: 10px 20px;
            }}
            QPushButton:checked {{
                background-color: #7C4DFF;
                color: white;
                border-color: #7C4DFF;
            }}
            QPushButton:!checked {{
                background-color: {c['tab_unchecked_bg']};
                color: {c['tab_unchecked_text']};
            }}
            QPushButton:!checked:hover {{
                background-color: rgba(124, 77, 255, 0.06);
                border-color: rgba(124, 77, 255, 0.30);
                color: #B39DFF;
            }}
        """)

        # ---- Generate button ----
        self.generate_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(124, 77, 255, 0.08);
                color: #7C4DFF;
                border: 1px solid rgba(124, 77, 255, 0.35);
                border-radius: 12px;
                font-weight: 700;
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: #7C4DFF;
                color: white;
                border-color: #7C4DFF;
            }}
            QPushButton:disabled {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                color: {c['text_disabled']};
            }}
        """)

        # ---- Apply button ----
        self.apply_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(76, 175, 80, 0.08);
                color: #4CAF50;
                border: 1px solid rgba(76, 175, 80, 0.35);
                border-radius: 12px;
                font-weight: 700;
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: #4CAF50;
                color: white;
                border-color: #4CAF50;
            }}
            QPushButton:disabled {{
                background-color: {c['card']};
                border-color: {c['border']};
                color: {c['text_disabled']};
            }}
        """)

        # ---- Undo button ----
        self.undo_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: #9575FF;
                border: 1px solid rgba(149, 117, 255, 0.30);
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.08);
                color: #B39DFF;
                border-color: #7C4DFF;
            }}
            QPushButton:disabled {{
                background-color: {c['card']};
                border-color: {c['border']};
                color: {c['text_disabled']};
            }}
        """)

        # ---- History, Pinned, Edit buttons ----
        outline_btn_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {c['text_muted']};
                border: 1px solid {c['border_strong']};
                border-radius: 12px;
                font-weight: 600;
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.06);
                border-color: #7C4DFF;
                color: #B39DFF;
            }}
        """
        self.history_button.setStyleSheet(outline_btn_style)
        self.pinned_button.setStyleSheet(outline_btn_style)
        self.edit_inputs_button.setStyleSheet(outline_btn_style + """
            QPushButton { padding: 0px 20px; }
        """)

        # ---- Labels ----
        self.status_label.setStyleSheet(f"color: {c['text_muted']}; font-style: italic; font-size: 13px;")
        self.dest_label.setStyleSheet(f"color: {c['text_muted']}; font-size: 13px; background: transparent;")
        self._examples_label.setStyleSheet(f"color: {c['text_disabled']}; font-size: 12px; background: transparent;")
        self.plan_summary_label.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 13px; font-weight: 500;
            color: {c['text_muted']}; padding: 4px 0px;
        """)
        self.existing_folders_note.setStyleSheet(f"""
            font-family: "Segoe UI", sans-serif;
            font-size: 12px; color: {c['text_disabled']};
            font-style: italic; padding: 2px 0px;
        """)
        self.watch_folder_label.setStyleSheet(f"""
            font-size: 13px; color: {c['text_muted']};
            background: transparent; padding: 4px 0;
        """)
        self._watch_desc.setStyleSheet(f"color: {c['text_muted']}; font-size: 13px; background: transparent;")

        # ---- Action scroll area ----
        self._action_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none; background: transparent;
            }}
            QScrollBar:horizontal {{
                border: none; background: {c['scrollbar_bg']};
                height: 8px; border-radius: 4px; margin-top: 4px;
            }}
            QScrollBar::handle:horizontal {{
                background: {c['scrollbar_handle']};
                border-radius: 4px; min-width: 40px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: #7C4DFF;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

        # ---- Tree widget ----
        self.plan_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: transparent;
                border: none;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px; padding: 4px; outline: none;
            }}
            QTreeWidget::item {{
                height: 38px; color: {c['text']};
                border-radius: 10px; padding-left: 8px; margin: 2px 0px;
            }}
            QTreeWidget::item:hover {{
                background-color: rgba(124, 77, 255, 0.08);
            }}
            QTreeWidget::item:selected {{
                background-color: rgba(124, 77, 255, 0.15);
                color: #7C4DFF; font-weight: 600;
            }}
            QTreeView::branch {{
                background: transparent; width: 0px; border: none; image: none;
            }}
        """)

    def _open_watch_config(self):
        """Open the watch configuration dialog."""
        dialog = WatchConfigDialog(self)
        result = dialog.exec()
        
        if result == QDialog.Accepted:
            self._update_watch_summary()
            
            # If watcher is running, apply new settings without stopping
            if self.auto_watcher and self.auto_watcher.is_running:
                self._apply_config_changes()
    
    def _apply_config_changes(self):
        """Apply configuration changes while watcher is running."""
        # Update folder instructions from settings
        folder_instructions = {}
        for folder_data in settings.auto_organize_folders:
            folder_path = folder_data.get('path', '')
            instruction = folder_data.get('instruction', '')
            if folder_path:
                normalized_path = os.path.normpath(folder_path)
                folder_instructions[normalized_path] = instruction
        
        # Update watcher's instructions
        self.auto_watcher.folder_instructions = folder_instructions
        
        # NOTE: Per-folder organization choices are now handled by the individual "Save" 
        # buttons next to each folder, so we don't show a dialog here when clicking "Done"
        
        # Instructions updated - just update the summary
        self._update_watch_summary()
        logger.info("Applied configuration changes while watching")
    
    def _update_watch_summary(self):
        """Update the watch status display."""
        folder_count = len(settings.auto_organize_folders)
        is_watching = self.auto_watcher and self.auto_watcher.is_running
        
        if folder_count == 0:
            # No folders configured
            self.watch_folder_label.setText("📁 No folders configured")
            self.watch_folder_label.setStyleSheet("""
                font-size: 13px; 
                color: #7A7A90; 
                background: transparent;
                padding: 4px 0;
            """)
            self.watch_toggle_btn.setEnabled(False)
        else:
            # Show folder count + status on one line
            if is_watching:
                status_text = f"📁 {folder_count} folder{'s' if folder_count > 1 else ''} • ✅ Active"
                color = "#7C4DFF"
            else:
                status_text = f"📁 {folder_count} folder{'s' if folder_count > 1 else ''} configured"
                color = "#7A7A90"
            
            self.watch_folder_label.setText(status_text)
            self.watch_folder_label.setStyleSheet(f"""
                font-size: 13px; 
                color: {color}; 
                background: transparent;
                padding: 4px 0;
                font-weight: {'500' if is_watching else '400'};
            """)
            
            self.watch_toggle_btn.setEnabled(True)
        
        # Update button state (purple theme for both states)
        if is_watching:
            self.watch_toggle_btn.setText("⏹ Stop")
            self.watch_toggle_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #7C4DFF;
                    border: 2px solid #7C4DFF;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 0 20px;
                }
                QPushButton:hover {
                    background: rgba(124, 77, 255, 0.1);
                }
            """)
        else:
            self.watch_toggle_btn.setText("▶ Start")
            self.watch_toggle_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C4DFF, stop:1 #9575FF);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 0 20px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9575FF, stop:1 #B39DFF);
                }
                QPushButton:disabled {
                    background: rgba(124, 77, 255, 0.3);
                    color: rgba(255, 255, 255, 0.5);
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
                QPushButton:disabled {
                    background-color: #3a3a3a;
                    color: #4A4A5A;
                }
            """)
    
    def _update_watch_summary_as_watching(self):
        """Immediately update UI to show watching state (before watcher actually starts)."""
        # Update folder label to show active status
        folder_count = len(self.watch_folders) if self.watch_folders else len(settings.auto_organize_folders)
        self.watch_folder_label.setText(f"📁 {folder_count} folder{'s' if folder_count > 1 else ''} • ✅ Active")
        self.watch_folder_label.setStyleSheet("""
            font-size: 13px; 
            color: #7C4DFF; 
            background: transparent;
            padding: 4px 0;
            font-weight: 500;
        """)
        
        # Update toggle button to Stop state (purple outline)
        self.watch_toggle_btn.setText("⏹ Stop")
        self.watch_toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #7C4DFF;
                border: 2px solid #7C4DFF;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
                padding: 0 20px;
            }
            QPushButton:hover {
                background: rgba(124, 77, 255, 0.1);
            }
        """)
    
    def _toggle_watch_mode(self):
        """Toggle the watch mode on/off."""
        if self.auto_watcher and self.auto_watcher.is_running:
            self._stop_watch_mode()
            self._was_manually_stopped = True  # Track that user manually stopped
            # Persist the paused state so closing the app / switching tabs / dialog
            # Saves don't quietly resume auto-organize behind the user's back.
            settings.set_auto_organize_paused(True)
        else:
            # User explicitly clicked Start — clear the paused flag so future
            # auto-start checks honor it.
            settings.set_auto_organize_paused(False)

            # OPTIMISTIC UI UPDATE: disable the button and show a transient
            # "Starting…" label the instant the click is registered. The
            # actual start path (which can take a moment for Organize-New-Only
            # folders since it hashes every pre-existing file for the
            # baseline) is then deferred via QTimer.singleShot(0, …) so the
            # button visibly reacts before any blocking work begins.
            self.watch_toggle_btn.setEnabled(False)
            self.watch_toggle_btn.setText("⏳ Starting…")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

            # If resuming after manual stop, skip the popup (just restart watching)
            skip_popup = getattr(self, '_was_manually_stopped', False)
            self._was_manually_stopped = False  # Reset flag after starting
            QTimer.singleShot(0, lambda: self._start_watch_mode(skip_existing_popup=skip_popup))
    
    def _start_watch_mode(self, is_catch_up: bool = False, catch_up_since=None, skip_existing_popup: bool = False):
        """Start watching folders for new files.
        
        Args:
            is_catch_up: If True, organize files modified since catch_up_since
            catch_up_since: Datetime to filter files for catch-up mode
            skip_existing_popup: If True, skip the "Organize Existing Files?" popup (for auto-start)
        """
        if not settings.auto_organize_folders:
            QMessageBox.warning(
                self, "No Folders",
                "Please configure folders to watch first."
            )
            # Restore the toggle button so the user can try again.
            self.watch_toggle_btn.setEnabled(True)
            self._update_watch_summary()
            return

        # Clear and setup watcher
        self.auto_watcher.clear_folders()
        self.watch_folders.clear()
        
        # Build per-folder instructions dict from settings
        # CRITICAL: Use os.path.normpath to match watcher's path format
        folder_instructions = {}
        has_any_instruction = False
        
        for folder_data in settings.auto_organize_folders:
            folder_path = folder_data.get('path', '')
            instruction = folder_data.get('instruction', '')
            
            if folder_path:
                # Normalize the path to match how watcher stores folders
                normalized_path = os.path.normpath(folder_path)
                
                if os.path.isdir(normalized_path):
                    self.auto_watcher.add_folder(normalized_path)
                    self.watch_folders.append(normalized_path)
                    folder_instructions[normalized_path] = instruction
                    
                    if instruction:
                        has_any_instruction = True
                    
                    logger.info(f"Added watch folder: {normalized_path} with instruction: {instruction[:30] if instruction else '(none)'}...")
        
        if not self.watch_folders:
            QMessageBox.warning(
                self, "No Valid Folders",
                "None of the configured folders exist. Please reconfigure."
            )
            # Restore the toggle button so the user can try again.
            self.watch_toggle_btn.setEnabled(True)
            self._update_watch_summary()
            return

        # Set folder instructions
        self.auto_watcher.folder_instructions = folder_instructions
        
        # Set catch-up filter if provided
        if catch_up_since:
            self.auto_watcher.catch_up_since = catch_up_since
        
        # UPDATE UI IMMEDIATELY - show "watching" state right away
        self._update_watch_summary_as_watching()
        
        # Process events to update UI before dialog
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Count existing files (including files in subfolders)
        existing_count = 0
        subfolder_count = 0
        for folder in self.watch_folders:
            try:
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isfile(item_path):
                        existing_count += 1
                    elif os.path.isdir(item_path) and not item.startswith('.'):
                        # Count files in subfolders too
                        subfolder_count += 1
                        for sub_item in os.listdir(item_path):
                            if os.path.isfile(os.path.join(item_path, sub_item)):
                                existing_count += 1
            except Exception:
                pass
        
        # No popup on Start Watching. The per-folder action (Re-organize All /
        # As-Is / Watch Only) is set via each folder's Options button, and is
        # applied immediately by the configure dialog's Save (see
        # WatchConfigDialog._save_and_close → organize_single_folder). Starting
        # the watcher here is purely about kicking off the periodic
        # file-detection timer — never re-ask the user.
        #
        # The one exception is catch-up mode: if the app was previously running
        # and was closed, we re-organize existing files on startup so anything
        # added while the app was offline gets picked up.
        organize_existing = bool(is_catch_up)

        # Start the watcher (periodic timer; per-folder organize already
        # happened via Save when the user picked their action). Defer the
        # actual start() call one more event-loop tick so the "⏳ Starting…"
        # label set in _toggle_watch_mode has a chance to paint — start()
        # itself can take a moment for Organize-New-Only folders since it
        # hashes every pre-existing file to build the baseline.
        def _finish_start():
            try:
                self.auto_watcher.start(organize_existing=organize_existing, flatten_first=False)
            finally:
                # Re-enable the button only after the worker is actually
                # watching, regardless of whether start() succeeded. The
                # button is in the "Stop" visual state already (set in
                # _update_watch_summary_as_watching above).
                self.watch_toggle_btn.setEnabled(True)
        QTimer.singleShot(0, _finish_start)
        
    def _stop_watch_mode(self):
        """Stop watching folders."""
        if self.auto_watcher:
            self.auto_watcher.stop()
        
        # Save last active timestamp for catch-up feature
        settings.update_auto_organize_last_active()
        
        # Update UI using centralized method
        self._update_watch_summary()
    
    def _check_auto_start(self):
        """Check if we should auto-start the watcher on app open."""
        # Respect the user's explicit Stop. If they paused the watcher, leave it
        # paused — across app restarts, tab switches, and dialog Saves. They have
        # to click Start to resume.
        if settings.auto_organize_paused:
            logger.info("Auto-start skipped: watcher is paused (user previously clicked Stop)")
            return
        # Auto-start if there are configured folders (no toggle needed)
        if not settings.auto_organize_folders:
            return
        
        # Check for catch-up (files added while app was closed)
        last_active = settings.get_auto_organize_last_active_time()
        
        if last_active:
            # Calculate time difference
            from datetime import datetime
            now = datetime.now()
            diff = now - last_active
            hours = diff.total_seconds() / 3600
            
            if hours > 0.5:  # More than 30 min since last active - do catch-up silently
                logger.info(f"Auto-start catch-up: watcher was inactive for {hours:.1f} hours")
                self._start_watch_mode(is_catch_up=True, catch_up_since=last_active, skip_existing_popup=True)
                return
        
        # Normal auto-start - skip existing files popup, just watch for new files
        self._start_watch_mode(skip_existing_popup=True)
    
    def _on_watch_file_organized(self, source: str, dest: str, category: str):
        """Handle file organized signal from watcher."""
        # Just log it, don't show in UI per user request
        pass
        logger.info(f"Watch organized: {source} -> {dest}")
    
    def _on_watch_file_indexed(self, file_path: str):
        """Handle file indexed signal from watcher."""
        logger.info(f"Watch auto-indexed: {file_path}")
        # Update usage display in main window
        main_window = self.window()
        if main_window and hasattr(main_window, '_update_usage_labels'):
            main_window._update_usage_labels()
    
    def _on_watch_status(self, status: str):
        """Handle status updates from watcher."""
        logger.info(f"Watch status: {status}")
    
    def _on_watch_error(self, path: str, error: str):
        """Handle errors from watcher."""
        logger.error(f"Watch error for {path}: {error}")

    def _on_watch_limit_reached(self, info: dict):
        """Show the upgrade popup when auto-organize hits the monthly index limit.

        Debounced to once per session so repeated background batches don't spam
        the dialog. Reuses MainWindow._show_upgrade_dialog (the same popup the
        manual 'Index Files' path uses) for a single source of truth.
        """
        if getattr(self, '_auto_limit_popup_shown', False):
            return
        self._auto_limit_popup_shown = True
        main_window = self.window()
        if main_window is not None and hasattr(main_window, '_show_upgrade_dialog'):
            main_window._show_upgrade_dialog(info or {})
    
    def showEvent(self, event):
        """Refresh file count when page becomes visible."""
        super().showEvent(event)
        if not self.current_plan:
            self._update_file_count()
    
    def _update_file_count(self):
        """Show how many indexed files are available."""
        try:
            count = file_index.get_file_count()
            if count > 0:
                self.status_label.setText(f"{count} indexed files available for organization")
            else:
                self.status_label.setText("No files indexed yet. Go to Index Files to add files first.")
        except Exception as e:
            logger.error(f"Error getting file count: {e}")
            self.status_label.setText("Could not load file count")
    
    def _switch_tab(self, index: int):
        """Switch between Organize Now (0) and Auto-Organize (1) tabs."""
        # Hide tips IMMEDIATELY before switching
        main_window = self.window()
        if hasattr(main_window, 'tips_manager'):
            main_window.tips_manager.hide_all_tips()
        
        self.content_stack.setCurrentIndex(index)
        
        # Update button checked states
        self.tab_organize_now.setChecked(index == 0)
        self.tab_auto_organize.setChecked(index == 1)
        
        # Show tips for the new tab after a short delay
        if hasattr(main_window, 'tips_manager'):
            from PySide6.QtCore import QTimer
            QTimer.singleShot(150, main_window.tips_manager.show_tips_for_visible_widgets)
    def _hide_input_cards(self):
        """Hide instruction and destination cards after plan is generated."""
        self.instruction_card.setVisible(False)
        self.dest_card.setVisible(False)
        self.subtitle.setVisible(False)  # Hide subtitle to save space
        self.edit_inputs_button.setVisible(True)
    
    def _show_input_cards(self):
        """Show instruction and destination cards, hide plan (return to input view)."""
        self.instruction_card.setVisible(True)
        self.dest_card.setVisible(True)
        self.subtitle.setVisible(True)  # Show subtitle again
        self.edit_inputs_button.setVisible(False)
        
        # Hide the plan completely - return to clean input view
        self._hide_plan_ui()
        self.apply_button.setVisible(False)
        self.clear_button.setVisible(False)
        self.undo_button.setVisible(False)
        self.feedback_group.setVisible(False)
    
    def _show_plan_summary(self, folder_count: int, file_count: int, total_size_mb: float):
        """Show the plan summary line."""
        self.plan_summary_label.setText(f"📊 {folder_count} folders  •  {file_count} files  •  {total_size_mb:.2f} MB")
        self.plan_summary_label.setVisible(True)
    
    def _show_existing_folders_warning(self, folders: list):
        """Show subtle note for existing folders."""
        if folders:
            if len(folders) <= 3:
                folder_names = ", ".join(folders)
            else:
                folder_names = ", ".join(folders[:3]) + f" +{len(folders) - 3} more"
            self.existing_folders_note.setText(f"Note: {folder_names} already exist — files will be added to them.")
            self.existing_folders_note.setVisible(True)
        else:
            self.existing_folders_note.setVisible(False)
    
    def _hide_plan_ui(self):
        """Hide plan-related UI elements."""
        self.plan_summary_label.setVisible(False)
        self.existing_folders_note.setVisible(False)
        self.results_splitter.setVisible(False)
        self.edit_inputs_button.setVisible(False)

    
    def select_destination(self):
        """Open folder picker for destination."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Destination Folder", str(Path.home())
        )
        if folder:
            self.destination_path = Path(folder)
            self.dest_label.setText(str(self.destination_path))
            from app.ui.theme_manager import get_theme_colors
            c = get_theme_colors()
            self.dest_label.setStyleSheet(f"color: {c['text']}; font-weight: bold; font-size: 13px; background: transparent;")
            self._update_generate_button()
    
    def _update_generate_button(self):
        """Enable generate button when destination is set (instruction optional for auto-organize)."""
        has_destination = self.destination_path is not None
        self.generate_button.setEnabled(has_destination)
    
    def _toggle_voice_recording(self):
        """Toggle voice recording on/off."""
        if self.is_recording_voice:
            self._stop_voice_recording()
        else:
            self._start_voice_recording()
    
    def _start_voice_recording(self):
        """Start recording voice input."""
        self.is_recording_voice = True
        self.mic_button.setText("⏹️")
        self.mic_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #ff4444;
                border: 1px solid #cc0000;
                border-radius: 6px;
                color: white;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        self.mic_button.setToolTip("Recording... Click to stop")
        self.status_label.setText("🎤 Recording... Speak your instruction, then click to stop.")
        
        # Start voice worker
        self.voice_worker = VoiceRecordWorker()
        self.voice_worker.finished.connect(self._on_voice_transcribed)
        self.voice_worker.error.connect(self._on_voice_error)
        self.voice_worker.recording_stopped.connect(self._on_recording_stopped)
        self.voice_worker.start()
    
    def _stop_voice_recording(self):
        """Stop recording voice input."""
        if self.voice_worker:
            self.voice_worker.stop_recording()
        self.status_label.setText("⏳ Transcribing...")
    
    def _on_recording_stopped(self):
        """Called when recording has stopped, before transcription."""
        self.is_recording_voice = False
        self._reset_mic_button()
    
    def _on_voice_transcribed(self, text: str):
        """Handle transcribed text from voice input."""
        self.is_recording_voice = False
        self._reset_mic_button()
        
        if text.strip():
            # Append to existing text or replace
            current = self.instruction_input.text().strip()
            if current:
                self.instruction_input.setText(f"{current} {text}")
            else:
                self.instruction_input.setText(text)
            self.status_label.setText(f"✓ Voice transcribed: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            logger.info(f"Voice transcribed: {text}")
        else:
            self.status_label.setText("No speech detected. Try again.")
    
    def _on_voice_error(self, error: str):
        """Handle voice recording errors."""
        self.is_recording_voice = False
        self._reset_mic_button()
        self.status_label.setText(f"Voice error: {error}")
        logger.error(f"Voice recording error: {error}")
        
        # Show error if it's about missing libraries
        if "Missing audio library" in error:
            QMessageBox.warning(
                self, "Audio Library Missing",
                f"{error}\n\nThe voice input feature requires additional libraries.\n"
                "Please install them and restart the app."
            )
    
    def _reset_mic_button(self):
        """Reset mic button to default state."""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self.mic_button.setText("🎤")
        self.mic_button.setStyleSheet(f"""
            QPushButton {{
                font-size: 18px;
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 12px;
                color: {c['text']};
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }}
            QPushButton:pressed {{
                background-color: rgba(124, 77, 255, 0.12);
            }}
        """)
        self.mic_button.setToolTip("Click to speak your instruction (click again to stop)")

    def _load_files_from_db(self) -> List[Dict[str, Any]]:
        """Load indexed files from the database, filtering to destination folder only.
        
        CRITICAL: Only loads files that are WITHIN the destination folder.
        This prevents accidentally moving files from other locations.
        """
        files = []
        self.files_by_id = {}
        excluded_count = 0
        outside_folder_count = 0
        
        # Get destination path for filtering (normalized, case-insensitive on Windows)
        dest_path_str = None
        if self.destination_path:
            dest_path_str = os.path.normpath(str(self.destination_path)).lower()
        
        try:
            with sqlite3.connect(file_index.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM files")
                rows = cursor.fetchall()
            
            for row in rows:
                file_path = row["file_path"]
                
                # CRITICAL: Only include files within the destination folder
                # This prevents files from other indexed locations being moved
                if dest_path_str:
                    normalized_file_path = os.path.normpath(file_path).lower()
                    if not normalized_file_path.startswith(dest_path_str + os.sep) and normalized_file_path != dest_path_str:
                        outside_folder_count += 1
                        continue  # Skip files outside destination folder
                
                # Skip files matching exclusion patterns
                if settings.should_exclude(file_path):
                    excluded_count += 1
                    continue
                
                f = {
                    "id": row["id"],
                    "file_path": file_path,
                    "file_name": row["file_name"],
                    "file_size": row["file_size"] or 0,
                    "label": row["label"] if "label" in row.keys() else None,
                    "caption": row["caption"] if "caption" in row.keys() else None,
                    "tags": self._parse_tags(row["tags"] if "tags" in row.keys() else None),
                    "category": row["category"] if "category" in row.keys() else None,
                }
                files.append(f)
                self.files_by_id[row["id"]] = f
            
            if outside_folder_count > 0:
                logger.info(f"Filtered out {outside_folder_count} files outside destination folder")
            if excluded_count > 0:
                logger.info(f"Excluded {excluded_count} files based on exclusion patterns")
                
        except Exception as e:
            logger.error(f"Error loading files: {e}")
        
        return files
    
    def _check_instruction_for_exclusions(self, instruction: str) -> List[str]:
        """Check if the instruction mentions file types that are in the exclusion list.
        
        Returns a list of matched exclusion patterns, or empty list if none.
        """
        instruction_lower = instruction.lower()
        matched_exclusions = []
        
        # Common file type keywords to check
        file_type_keywords = {
            '.json': ['json', '.json'],
            '.py': ['python', '.py', 'py files'],
            '.js': ['javascript', '.js', 'js files'],
            '.ts': ['typescript', '.ts', 'ts files'],
            '.csv': ['csv', '.csv'],
            '.xml': ['xml', '.xml'],
            '.yaml': ['yaml', '.yaml', '.yml'],
            '.md': ['markdown', '.md', 'md files'],
            '.txt': ['text', '.txt', 'txt files'],
            '.log': ['log', '.log', 'log files'],
            '.env': ['env', '.env', 'environment'],
            '.git': ['git', '.git'],
            '.pyc': ['pyc', '.pyc', 'compiled python'],
        }
        
        for exclusion_pattern in settings.exclusion_patterns:
            pattern_lower = exclusion_pattern.lower()
            
            # Check if pattern is mentioned directly in instruction
            # Remove * from pattern for matching (*.json -> .json or json)
            clean_pattern = pattern_lower.replace('*', '').strip('.')
            
            if clean_pattern and len(clean_pattern) >= 2:
                # Check direct mention
                if clean_pattern in instruction_lower:
                    matched_exclusions.append(exclusion_pattern)
                    continue
                
                # Check if it's a known file type with alternative keywords
                for ext, keywords in file_type_keywords.items():
                    if clean_pattern in ext or ext.strip('.') == clean_pattern:
                        for keyword in keywords:
                            if keyword in instruction_lower:
                                matched_exclusions.append(exclusion_pattern)
                                break
                        break
        
        return list(set(matched_exclusions))  # Remove duplicates
    
    def _parse_tags(self, raw) -> List[str]:
        """Parse tags from DB storage format."""
        if not raw:
            return []
        if isinstance(raw, list):
            return raw
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return [str(t) for t in v]
        except:
            pass
        if isinstance(raw, str):
            return [t.strip() for t in raw.split(",") if t.strip()]
        return []
    
    def _verify_and_fix_paths(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify file paths and fix them if files have been moved.
        
        For each file:
        1. Check if it exists at the recorded path
        2. If not, search for it by name in the destination folder
        3. Try partial matching for renamed files (Windows adds (1), (2) etc)
        4. If found elsewhere, update the database path
        5. If not found anywhere, remove from the list
        
        Returns the list of files with verified/updated paths.
        """
        if not files or not self.destination_path:
            return files
        
        verified_files = []
        updated_count = 0
        updated_names = []
        removed_count = 0
        removed_names = []
        
        # Build a map of all files in the destination folder for quick lookup
        existing_files = {}  # exact filename -> [paths]
        all_files_list = []  # [(filename_lower, full_path), ...] for partial matching
        try:
            for root, dirs, filenames in os.walk(str(self.destination_path)):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    # Store by filename (lowercase for case-insensitive matching)
                    key = filename.lower()
                    if key not in existing_files:
                        existing_files[key] = []
                    existing_files[key].append(full_path)
                    all_files_list.append((key, full_path))
        except Exception as e:
            logger.warning(f"Error scanning destination folder: {e}")
        
        logger.info(f"Scanned {len(all_files_list)} files in destination folder")
        
        for f in files:
            file_path = f.get("file_path", "")
            file_name = f.get("file_name", "")
            file_id = f.get("id")
            
            # Check if file exists at recorded path
            if os.path.exists(file_path):
                verified_files.append(f)
                continue
            
            # File not at recorded path - try to find it
            logger.info(f"File not found at recorded path: {file_path}")
            
            # Search by exact filename in destination folder
            key = file_name.lower()
            candidates = existing_files.get(key, [])
            
            new_path = None
            
            if candidates:
                # Found file(s) with exact same name
                new_path = candidates[0]
                candidates.pop(0)
            else:
                # Try partial matching - look for files that start with same base name
                # This handles Windows renaming like "file.png" -> "file (1).png"
                base_name = os.path.splitext(file_name)[0].lower()
                extension = os.path.splitext(file_name)[1].lower()
                
                for existing_name, existing_path in all_files_list:
                    # Check if existing file starts with our base name and has same extension
                    if existing_name.startswith(base_name) and existing_name.endswith(extension):
                        # Make sure it's not already matched to another file
                        if existing_path not in [vf.get("file_path") for vf in verified_files]:
                            new_path = existing_path
                            logger.info(f"Partial match: {file_name} -> {os.path.basename(existing_path)}")
                            break
            
            if new_path:
                logger.info(f"Found moved file: {file_name} -> {new_path}")
                
                # Update database
                if file_index.update_file_path(file_id, new_path):
                    f["file_path"] = new_path
                    verified_files.append(f)
                    updated_count += 1
                    updated_names.append(f"{file_name} → {os.path.basename(new_path)}")
                else:
                    logger.warning(f"Failed to update path in database for {file_name}")
                    removed_count += 1
                    removed_names.append(file_name)
            else:
                # File not found anywhere - skip it
                logger.info(f"File no longer exists, skipping: {file_name}")
                removed_count += 1
                removed_names.append(file_name)
        
        # Show summary dialog if changes were made
        if updated_count > 0 or removed_count > 0:
            logger.info(f"Path verification: {updated_count} updated, {removed_count} removed, {len(verified_files)} verified")
            
            msg_parts = []
            if updated_count > 0:
                msg_parts.append(f"✓ Updated {updated_count} file path(s):")
                for name in updated_names[:5]:
                    msg_parts.append(f"   • {name}")
                if len(updated_names) > 5:
                    msg_parts.append(f"   ... and {len(updated_names) - 5} more")
            
            # Silent path verification - just update status bar, no popup
            status_msg = f"Path check: {updated_count} fixed"
            if removed_count > 0:
                status_msg += f", {removed_count} missing"
            status_msg += f". Sending {len(verified_files)} files to AI..."
            self.status_label.setText(status_msg)
        
        return verified_files
    
    def _check_for_unindexed_files(self) -> List[str]:
        """
        Scan destination folder for files not in the database.
        
        Returns list of unindexed file paths.
        """
        if not self.destination_path or not self.destination_path.exists():
            return []
        
        unindexed_files = []
        
        try:
            # Get all indexed file paths (normalized for comparison)
            indexed_paths = set()
            with sqlite3.connect(file_index.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM files")
                for row in cursor.fetchall():
                    indexed_paths.add(os.path.normpath(row[0]).lower())
            
            # Scan destination folder for files
            for root, dirs, filenames in os.walk(str(self.destination_path)):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    normalized_path = os.path.normpath(file_path).lower()
                    
                    # Skip if already indexed or matches exclusion
                    if normalized_path in indexed_paths:
                        continue
                    if settings.should_exclude(file_path):
                        continue
                    
                    unindexed_files.append(file_path)
            
        except Exception as e:
            logger.error(f"Error checking for unindexed files: {e}")
        
        return unindexed_files
    
    def generate_plan(self):
        """Request organization plan from LLM."""
        instruction = self.instruction_input.text().strip()
        
        if not self.destination_path:
            return
        
        # Check for unindexed files and offer to index them with full AI analysis
        unindexed_files = self._check_for_unindexed_files()
        if unindexed_files:
            # Ask user if they want to index new files first
            confirmed = ModernConfirmDialog.ask(
                self,
                title="New Files Detected",
                message=f"Found {len(unindexed_files)} new file(s) that haven't been analyzed yet.",
                details=[
                    "AI analysis provides better organization",
                    "Files will be tagged and categorized",
                    "This only takes a moment"
                ],
                info_text="Indexing ensures the AI can make smart organization decisions.",
                yes_text="Index Now",
                no_text="Skip"
            )
            
            if confirmed:
                # Trigger full AI indexing, then continue with plan
                self._index_folder_before_organize(self.destination_path)
                return  # Will call generate_plan again after indexing
        
        # Check if instruction mentions excluded file types
        if instruction:
            excluded_types_mentioned = self._check_instruction_for_exclusions(instruction)
            if excluded_types_mentioned:
                # Show warning and don't proceed
                excluded_list = ", ".join(excluded_types_mentioned)
                QMessageBox.warning(
                    self,
                    "Excluded File Types",
                    f"Your instruction mentions file types that are in your exclusion list:\n\n"
                    f"• {excluded_list}\n\n"
                    f"These files are protected from organization.\n\n"
                    f"To organize them, go to Settings → Exclusions and remove the pattern."
                )
                return
        
        # Auto-organize mode: no instruction provided
        if not instruction:
            confirmed = ModernConfirmDialog.ask(
                self,
                title="Auto-Organize Mode",
                message="AI will analyze your files and propose an organization structure based on:",
                details=[
                    "File types and categories",
                    "AI-generated tags and labels", 
                    "Content analysis"
                ],
                highlight_text="The structure will be kept simple with minimal nesting.",
                info_text="You will preview the plan before anything is moved.",
                yes_text="Continue",
                no_text="Cancel"
            )
            if not confirmed:
                return
            
            # Use auto-organize instruction - MUST organize ALL files
            instruction = (
                "[AUTO-ORGANIZE] Organize ALL of the provided files into a logical folder structure. "
                "CRITICAL: EVERY single file must be placed in a folder - do NOT leave any file out. "
                "Keep it simple - use only a few broad, clear folder names (e.g., screenshots, documents, images). "
                "Avoid deep nesting (no subfolders inside subfolders). "
                "Group similar files together based on their type, tags, and content. "
                "If some files don't fit any clear category, put them in a 'misc' or 'other' folder. "
                "EVERY file_id provided MUST appear in exactly one folder."
            )
        
        # Save the instruction for potential refinement
        self.original_instruction = instruction
        
        files = self._load_files_from_db()
        
        if not files:
            # No indexed files - check if destination folder has files to index
            if self.destination_path and self.destination_path.exists():
                # Count files in destination folder
                folder_files = []
                try:
                    for item in os.listdir(str(self.destination_path)):
                        item_path = os.path.join(str(self.destination_path), item)
                        if os.path.isfile(item_path):
                            folder_files.append(item_path)
                except Exception as e:
                    logger.error(f"Error scanning destination folder: {e}")
                
                if folder_files:
                    # Ask user if they want to index the folder first
                    reply = QMessageBox.question(
                        self,
                        "Index Files First?",
                        f"Found {len(folder_files)} file(s) in the destination folder that haven't been indexed.\n\n"
                        "Would you like to index them now? This is required before organizing.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    if reply == QMessageBox.Yes:
                        # Index the folder
                        self._index_folder_before_organize(self.destination_path)
                        return  # The indexing will call generate_plan again when done
                    else:
                        return
            
            QMessageBox.warning(
                self, "No Files",
                "No indexed files found. Please go to Index Files and index some files first."
            )
            return
        
        # Verify file paths and fix any that have been moved
        self.status_label.setText("Verifying file paths...")
        original_count = len(files)
        files = self._verify_and_fix_paths(files)
        
        # Re-filter exclusions after path verification (paths may have changed)
        excluded_after_verify = 0
        filtered_files = []
        for f in files:
            if settings.should_exclude(f["file_path"]):
                excluded_after_verify += 1
                logger.info(f"Excluding file after path verify: {f['file_name']} (matches exclusion pattern)")
            else:
                filtered_files.append(f)
        files = filtered_files
        if excluded_after_verify > 0:
            logger.info(f"Excluded {excluded_after_verify} files based on exclusion patterns (post-verification)")
        
        # Update files_by_id with verified files only
        self.files_by_id = {f["id"]: f for f in files}

        if not files:
            QMessageBox.warning(
                self, "No Valid Files",
                f"All {original_count} indexed files have been moved or deleted.\n\n"
                "Please re-index the folder to update the file list."
            )
            return

        # Attach subfolder context so the AI knows where each file currently lives.
        # This enables "preserve folder X" instructions to work correctly.
        for f in files:
            try:
                rel = Path(f["file_path"]).parent.relative_to(self.destination_path)
                f["subfolder"] = str(rel) if str(rel) != "." else "."
            except (ValueError, TypeError):
                f["subfolder"] = "."

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.generate_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status_label.setText(f"Asking AI to organize {len(files)} files...")
        self.plan_tree.clear()
        # Details panel removed

        self.plan_worker = PlanWorker(instruction, files)
        self.plan_worker.finished.connect(self._on_plan_received)
        self.plan_worker.error.connect(self._on_plan_error)
        self.plan_worker.start()
    
    def _index_folder_before_organize(self, folder_path: Path):
        """Index a folder before organizing, then continue with organization."""
        # Count files to index
        file_count = 0
        try:
            for item in os.listdir(str(folder_path)):
                item_path = os.path.join(str(folder_path), item)
                if os.path.isfile(item_path):
                    file_count += 1
        except Exception:
            file_count = 0
        
        # Create and show the progress dialog
        self._index_progress_dialog = IndexProgressDialog(
            folder_path.name, 
            file_count,
            parent=self
        )
        
        # Disable the generate button while indexing
        self.generate_button.setEnabled(False)
        self.status_label.setText(f"Indexing files in {folder_path.name}...")
        
        # Create a worker thread for indexing
        self._index_worker = IndexBeforeOrganizeWorker(folder_path)
        self._index_worker.progress.connect(self._on_index_progress)
        self._index_worker.finished.connect(self._on_index_before_organize_finished)
        self._index_worker.error.connect(self._on_index_error)
        self._index_worker.cancelled.connect(self._on_index_cancelled)
        
        # Connect dialog buttons to worker
        self._index_progress_dialog.rejected.connect(self._on_index_dialog_cancelled)
        self._index_progress_dialog.skip_requested.connect(self._on_index_skip_requested)
        
        # Start the worker and show dialog
        self._index_worker.start()
        self._index_progress_dialog.show()
    
    def _on_index_progress(self, current: int, total: int, message: str):
        """Handle indexing progress updates."""
        # Update the progress dialog if it exists
        if hasattr(self, '_index_progress_dialog') and self._index_progress_dialog:
            self._index_progress_dialog.update_progress(current, total, message)
        
        # Also update status label
        self.status_label.setText(f"Indexing: {current}/{total}")
    
    def _on_index_dialog_cancelled(self):
        """Handle user clicking Cancel in the progress dialog."""
        if hasattr(self, '_index_worker') and self._index_worker:
            self._index_worker.cancel()
        
        self.generate_button.setEnabled(True)
        self.status_label.setText("Indexing cancelled.")
        
        # Update usage labels in case some files were indexed before cancel
        main_window = self.window()
        if main_window and hasattr(main_window, '_update_usage_labels'):
            main_window._update_usage_labels()
    
    def _on_index_skip_requested(self):
        """Handle user clicking Skip in the progress dialog."""
        if hasattr(self, '_index_worker') and self._index_worker:
            self._index_worker.cancel()
        
        self.generate_button.setEnabled(True)
        self.status_label.setText("Skipped indexing. Using already-indexed files...")
        
        # Update usage labels in case some files were indexed before skip
        main_window = self.window()
        if main_window and hasattr(main_window, '_update_usage_labels'):
            main_window._update_usage_labels()
        
        # Continue with plan generation using existing indexed files
        QTimer.singleShot(100, self.generate_plan)
    
    def _on_index_cancelled(self):
        """Handle indexing being cancelled by the worker."""
        if hasattr(self, '_index_progress_dialog') and self._index_progress_dialog:
            self._index_progress_dialog.close()
            self._index_progress_dialog = None
        
        self.generate_button.setEnabled(True)
        self.status_label.setText("Indexing cancelled.")
        
        # Update usage labels in case some files were indexed before cancellation
        main_window = self.window()
        if main_window and hasattr(main_window, '_update_usage_labels'):
            main_window._update_usage_labels()
    
    def _on_index_before_organize_finished(self, stats: dict):
        """Handle indexing completion, then continue with organization."""
        # Close the progress dialog
        if hasattr(self, '_index_progress_dialog') and self._index_progress_dialog:
            self._index_progress_dialog.complete()
            self._index_progress_dialog = None
        
        indexed_count = stats.get('indexed_files', 0)
        self.generate_button.setEnabled(True)
        
        # Update usage labels to reflect the newly indexed files
        main_window = self.window()
        if main_window and hasattr(main_window, '_update_usage_labels'):
            main_window._update_usage_labels()
        
        if indexed_count > 0:
            self.status_label.setText(f"Indexed {indexed_count} files. Generating organization plan...")
            # Now call generate_plan again - files should be available
            QTimer.singleShot(100, self.generate_plan)
        else:
            self.status_label.setText("No new files to index.")
            # Still proceed with plan if there are already indexed files
            QTimer.singleShot(100, self.generate_plan)
    
    def _on_index_error(self, error: str):
        """Handle indexing errors."""
        # Close the progress dialog
        if hasattr(self, '_index_progress_dialog') and self._index_progress_dialog:
            self._index_progress_dialog.close()
            self._index_progress_dialog = None
        
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Indexing error: {error}")
        logger.error(f"Index before organize error: {error}")
        
        # Update usage labels in case some files were indexed before the error
        main_window = self.window()
        if main_window and hasattr(main_window, '_update_usage_labels'):
            main_window._update_usage_labels()

    def _on_plan_received(self, plan: Optional[Dict[str, Any]]):
        """Handle LLM plan response."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        
        if not plan:
            self.status_label.setText("Failed to generate plan. Check AI settings.")
            QMessageBox.warning(
                self, "Plan Failed",
                "Could not generate organization plan. Make sure your AI provider is configured in Settings."
            )
            return
        
        # Deduplicate file IDs (AI sometimes puts same file in multiple folders)
        plan = deduplicate_plan(plan)
        
        valid_ids = set(self.files_by_id.keys())
        
        # GRACEFUL RECOVERY: Filter out invalid file IDs from the plan
        # This prevents "Unknown file_id" errors if AI hallucinates IDs
        if "folders" in plan:
            cleaned_folders = {}
            for folder_name, file_ids in plan["folders"].items():
                valid_file_ids = []
                for fid in file_ids:
                    # Handle both int and string IDs
                    try:
                        fid_int = int(fid)
                        if fid_int in valid_ids:
                            valid_file_ids.append(fid_int)
                    except (ValueError, TypeError):
                        continue
                
                if valid_file_ids:
                    cleaned_folders[folder_name] = valid_file_ids
            
            plan["folders"] = cleaned_folders

        # Only ensure ALL files are included in AUTO-ORGANIZE mode
        # For specific instructions (e.g., "move screenshots to X"), we want to leave other files untouched
        is_auto_organize = self.original_instruction and self.original_instruction.startswith("[AUTO-ORGANIZE]")
        if is_auto_organize:
            files_list = list(self.files_by_id.values())
            plan = ensure_all_files_included(plan, valid_ids, files_list)
        
        is_valid, errors = validate_plan(plan, valid_ids)
        
        if not is_valid:
            error_text = "\n".join(errors[:10])
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more errors"
            
            # Log what the AI actually returned for debugging
            logger.warning(f"Invalid plan from AI: {plan}")
            
            self.status_label.setText("Plan validation failed")
            
            # Build detailed error display
            details = f"Validation Errors:\n{'='*40}\n\n{error_text}\n\n"
            details += f"{'='*40}\nAI Response (for debugging):\n"
            details += json.dumps(plan, indent=2, default=str)[:1000]  # Limit length
            # Details panel removed - summary shown in plan_summary_label
            
            # More helpful error message
            first_error = errors[0] if errors else "Unknown error"
            if "folders" in first_error.lower():
                msg = (
                    f"The AI returned an invalid response format.\n\n"
                    f"Error: {first_error}\n\n"
                    "Try rephrasing your instruction to be more specific about what you want to organize."
                )
            else:
                msg = (
                    f"The AI plan failed validation:\n\n{first_error}\n\n"
                    "This can happen if the AI invented file IDs or proposed invalid folders."
                )
            
            QMessageBox.warning(self, "Invalid Plan", msg)
            return
        
        self.current_plan = plan
        self.current_moves = plan_to_moves(plan, self.files_by_id, self.destination_path)
        
        # Calculate plan statistics
        folder_count = len(plan.get("folders", {}))
        file_count = sum(len(f_ids) for f_ids in plan.get("folders", {}).values())
        total_size = sum(
            self.files_by_id.get(fid, {}).get('file_size', 0) 
            for folder_files in plan.get("folders", {}).values() 
            for fid in folder_files
        )
        total_size_mb = total_size / (1024 * 1024)
        
        # Show plan summary
        self._show_plan_summary(folder_count, file_count, total_size_mb)
        
        # Check for existing folders and show warning
        existing_folders = []
        if self.destination_path and self.destination_path.exists():
            for folder_name in plan.get("folders", {}).keys():
                proposed_path = self.destination_path / folder_name
                if proposed_path.exists():
                    existing_folders.append(folder_name)
        self._show_existing_folders_warning(existing_folders)
        
        # Hide input cards to focus on the plan
        self._hide_input_cards()
        
        self._display_plan(plan)
        
        # Check for folders that already exist in destination
        existing_folders = []
        if self.destination_path and self.destination_path.exists():
            for folder_name in plan.get("folders", {}).keys():
                proposed_path = self.destination_path / folder_name
                if proposed_path.exists():
                    existing_folders.append(folder_name)
        
        if existing_folders:
            folder_list = ", ".join(existing_folders[:3])
            if len(existing_folders) > 3:
                folder_list += f" and {len(existing_folders) - 3} more"
            # self.details_text.append(f"\n" + "="*50 + f"\nNote: {len(existing_folders)} folder(s) already exist: {folder_list}\nFiles will be added to existing folders.")
        
        folder_count = len(plan.get("folders", {}))
        files_in_plan = sum(len(fids) for fids in plan.get("folders", {}).values())
        valid_moves = len(self.current_moves)
        
        logger.info(f"Plan has {files_in_plan} files, {valid_moves} valid moves possible")
        
        if valid_moves == 0 and files_in_plan > 0:
            self.status_label.setText(f"Plan has {files_in_plan} files but none can be moved")
            self.apply_button.setEnabled(False)
            ModernInfoDialog.show_warning(
                self,
                title="No Files to Move",
                message=f"The AI proposed organizing {files_in_plan} file(s), but none need to be moved.",
                details=[
                    "Files are already in the destination folder",
                    "Files were already moved or deleted",
                    "Files no longer exist at their indexed paths"
                ],
                info_text="If files have been moved, please re-index to update the database."
            )
        elif valid_moves < files_in_plan:
            self.status_label.setText(f"Plan ready: {valid_moves}/{files_in_plan} files can be moved to {folder_count} folders")
            self.apply_button.setEnabled(valid_moves > 0)
        else:
            self.status_label.setText(f"Plan ready: {valid_moves} files to {folder_count} folders")
            self.apply_button.setEnabled(valid_moves > 0)
        
        # Show refinement section and other elements if we have a valid plan
        if folder_count > 0 or files_in_plan > 0:
            self.feedback_group.setVisible(True)
            self.feedback_input.clear()
            # Show the results section and action buttons
            self.results_splitter.setVisible(True)
            self.apply_button.setVisible(True)
            self.clear_button.setVisible(True)
            self.undo_button.setVisible(True)
    
    def _on_plan_error(self, error: str):
        """Handle planning error."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Error: {error}")
        logger.error(f"Plan generation error: {error}")

    def _display_plan(self, plan: Dict[str, Any]):
        """Show the organization plan in the tree widget."""
        self.plan_tree.clear()
        
        # Connect expand/collapse signals to update arrows
        try:
            self.plan_tree.itemExpanded.disconnect()
            self.plan_tree.itemCollapsed.disconnect()
        except:
            pass
        self.plan_tree.itemExpanded.connect(self._on_folder_expanded)
        self.plan_tree.itemCollapsed.connect(self._on_folder_collapsed)
        
        folders = plan.get("folders", {})

        # Build a nested tree from slash-paths. A plan folder named
        # "everything else/cartoons" should be rendered as a child of
        # "everything else", not as a sibling. We also auto-create empty
        # intermediate parents when the AI returned a child folder without
        # the parent in the plan (e.g. only "Photos/Vacation" but no
        # "Photos") — that intermediate gets shown as a folder with no
        # files of its own, just the child folder below it.
        display_limit = 25

        # Folder path (forward-slash, lower-cased only for keying) -> QTreeWidgetItem
        path_to_item: dict = {}

        def _make_folder_item(full_path: str, own_file_ids: list) -> QTreeWidgetItem:
            leaf_name = full_path.rsplit('/', 1)[-1]
            label = f"▶  📁 {leaf_name}  ({len(own_file_ids)} files)"
            item = QTreeWidgetItem([label])
            item.setExpanded(False)
            item.setData(0, Qt.UserRole, {"type": "folder", "name": full_path})

            for fid in own_file_ids[:display_limit]:
                try:
                    fid_int = int(fid)
                    file_info = self.files_by_id.get(fid_int, {})
                    fname = file_info.get("file_name", f"id:{fid}")
                    file_item = QTreeWidgetItem([fname])
                    file_item.setData(0, Qt.UserRole, {"type": "file", "id": fid_int})
                    item.addChild(file_item)
                except Exception:
                    pass

            if len(own_file_ids) > display_limit:
                more_item = QTreeWidgetItem([f"+ {len(own_file_ids) - display_limit} more files..."])
                more_item.setDisabled(True)
                item.addChild(more_item)
            return item

        # Sort folder paths by depth so we always create the parent before
        # the child. Forward-slash is the canonical separator inside the
        # plan; back-slashes get normalised for safety.
        def _depth(p: str) -> int:
            return p.replace('\\', '/').count('/')

        sorted_folder_paths = sorted(folders.keys(), key=lambda p: (_depth(p), p.lower()))

        for folder_path in sorted_folder_paths:
            file_ids = folders.get(folder_path) or []
            norm_path = folder_path.replace('\\', '/').strip('/')
            if not norm_path:
                continue

            # Ensure every ancestor exists in the tree as a (possibly empty)
            # folder item.
            parts = norm_path.split('/')
            for i in range(1, len(parts)):
                ancestor_path = '/'.join(parts[:i])
                if ancestor_path not in path_to_item:
                    ancestor_item = _make_folder_item(ancestor_path, [])
                    path_to_item[ancestor_path] = ancestor_item
                    parent_path = '/'.join(parts[:i - 1]) if i > 1 else ''
                    if parent_path and parent_path in path_to_item:
                        path_to_item[parent_path].addChild(ancestor_item)
                    else:
                        self.plan_tree.addTopLevelItem(ancestor_item)

            if norm_path in path_to_item:
                # Pre-created as an empty ancestor; now backfill its files.
                existing = path_to_item[norm_path]
                # Replace the label with the real file count and attach the
                # file children.
                leaf_name = norm_path.rsplit('/', 1)[-1]
                existing.setText(0, f"▶  📁 {leaf_name}  ({len(file_ids)} files)")
                for fid in file_ids[:display_limit]:
                    try:
                        fid_int = int(fid)
                        file_info = self.files_by_id.get(fid_int, {})
                        fname = file_info.get("file_name", f"id:{fid}")
                        file_item = QTreeWidgetItem([fname])
                        file_item.setData(0, Qt.UserRole, {"type": "file", "id": fid_int})
                        existing.addChild(file_item)
                    except Exception:
                        pass
                if len(file_ids) > display_limit:
                    more_item = QTreeWidgetItem([f"+ {len(file_ids) - display_limit} more files..."])
                    more_item.setDisabled(True)
                    existing.addChild(more_item)
            else:
                new_item = _make_folder_item(norm_path, file_ids)
                path_to_item[norm_path] = new_item
                parent_path = '/'.join(parts[:-1])
                if parent_path and parent_path in path_to_item:
                    path_to_item[parent_path].addChild(new_item)
                else:
                    self.plan_tree.addTopLevelItem(new_item)
        
        summary = get_plan_summary(plan, self.files_by_id)
        
        details = f"""Organization Plan Summary
{'='*50}

Destination: {self.destination_path}
Total folders: {summary["total_folders"]}
Total files to move: {summary["total_files"]}
Total size: {summary["total_size_mb"]} MB

Folders:
{'-'*50}
"""
        for folder in summary["folders"]:
            details += f"{folder['name']}: {folder['file_count']} files ({folder['size_mb']} MB)\n"
        
        # Details panel removed - summary shown in plan_summary_label
    
    def _get_file_icon(self, filename: str) -> str:
        """Get appropriate emoji icon based on file type."""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Images
        if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg', 'ico', 'tiff']:
            return '🖼️'
        # Videos
        elif ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'm4v']:
            return '🎬'
        # Audio
        elif ext in ['mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a', 'wma']:
            return '🎵'
        # Documents
        elif ext in ['pdf']:
            return '📕'
        elif ext in ['doc', 'docx']:
            return '📘'
        elif ext in ['txt', 'md', 'rtf']:
            return '📝'
        # Spreadsheets
        elif ext in ['xlsx', 'xls', 'csv']:
            return '📊'
        # Presentations
        elif ext in ['ppt', 'pptx']:
            return '📙'
        # Code
        elif ext in ['py', 'js', 'ts', 'html', 'css', 'java', 'cpp', 'c', 'h', 'rb', 'go', 'rs']:
            return '💻'
        # Data
        elif ext in ['json', 'xml', 'yaml', 'yml']:
            return '📋'
        # Archives
        elif ext in ['zip', 'rar', '7z', 'tar', 'gz']:
            return '📦'
        # Executables
        elif ext in ['exe', 'msi', 'app', 'dmg']:
            return '⚙️'
        # Default
        else:
            return '📄'
    
    def _on_folder_expanded(self, item: QTreeWidgetItem):
        """Update folder arrow when expanded."""
        data = item.data(0, Qt.UserRole)
        if data and data.get("type") == "folder":
            text = item.text(0)
            if text.startswith("▶"):
                item.setText(0, "▼" + text[1:])
    
    def _on_folder_collapsed(self, item: QTreeWidgetItem):
        """Update folder arrow when collapsed."""
        data = item.data(0, Qt.UserRole)
        if data and data.get("type") == "folder":
            text = item.text(0)
            if text.startswith("▼"):
                item.setText(0, "▶" + text[1:])
    
    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item click - toggle expand/collapse for folders."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        
        # Toggle expand/collapse for folders
        if data.get("type") == "folder":
            item.setExpanded(not item.isExpanded())
            return
        
        if data.get("type") == "file":
            fid = data.get("id")
            file_info = self.files_by_id.get(fid, {})
            
            details = f"""File Details
{'='*50}

ID: {fid}
Name: {file_info.get('file_name', 'unknown')}
Path: {file_info.get('file_path', 'unknown')}
Size: {round(file_info.get('file_size', 0) / 1024, 2)} KB
Label: {file_info.get('label', 'none')}
Tags: {', '.join(file_info.get('tags', [])) or 'none'}
Caption: {file_info.get('caption', 'none')}
"""
            # Details panel removed - summary shown in plan_summary_label
    
    def _show_tree_context_menu(self, position):
        """Show right-click context menu for tree items with pin option."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction
        
        item = self.plan_tree.itemAt(position)
        if not item:
            return
        
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 8px;
                padding: 6px 4px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 6px;
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                color: #E8E8F0;
            }
            QMenu::item:selected {
                background-color: rgba(124, 77, 255, 0.10);
                color: #B39DFF;
            }
        """)
        
        if data.get("type") == "file":
            fid = data.get("id")
            file_info = self.files_by_id.get(fid, {})
            file_path = file_info.get("file_path", "")
            file_name = file_info.get("file_name", "Unknown")
            
            if file_path:
                is_pinned = settings.is_pinned(file_path)
                
                if is_pinned:
                    unpin_action = QAction(f"📌 Unpin '{file_name}'", self)
                    unpin_action.triggered.connect(lambda: self._unpin_from_tree(file_path, fid))
                    menu.addAction(unpin_action)
                else:
                    pin_action = QAction(f"📌 Pin '{file_name}' (never organize)", self)
                    pin_action.triggered.connect(lambda: self._pin_from_tree(file_path, fid))
                    menu.addAction(pin_action)
        
        elif data.get("type") == "folder":
            folder_name = data.get("name", "Unknown")
            if self.destination_path:
                folder_path = str(self.destination_path / folder_name)
                is_pinned = settings.is_pinned(folder_path)
                
                if is_pinned:
                    unpin_action = QAction(f"📌 Unpin folder '{folder_name}'", self)
                    unpin_action.triggered.connect(lambda: self._unpin_folder_from_tree(folder_path, item))
                    menu.addAction(unpin_action)
                else:
                    pin_action = QAction(f"📌 Pin folder '{folder_name}' (never organize contents)", self)
                    pin_action.triggered.connect(lambda: self._pin_folder_from_tree(folder_path, item))
                    menu.addAction(pin_action)
        
        if menu.actions():
            menu.exec(self.plan_tree.viewport().mapToGlobal(position))
    
    def _pin_from_tree(self, file_path: str, file_id: int):
        """Pin a file from the tree view and remove it from current plan."""
        if settings.add_pinned_path(file_path):
            # Remove from current moves
            self.current_moves = [m for m in self.current_moves if m.get("file_id") != file_id]
            
            # Regenerate tree display
            if self.current_plan:
                # Remove the file from the plan
                for folder_name, file_ids in self.current_plan.get("folders", {}).items():
                    if file_id in file_ids:
                        file_ids.remove(file_id)
                    if str(file_id) in file_ids:
                        file_ids.remove(str(file_id))
                
                # Redisplay the plan
                self._display_plan(self.current_plan)
            
            logger.info(f"Pinned file: {file_path}")
            self.status_label.setText(f"📌 Pinned '{Path(file_path).name}' - removed from plan")
    
    def _unpin_from_tree(self, file_path: str, file_id: int):
        """Unpin a file from the tree view."""
        settings.remove_pinned_path(file_path)
        logger.info(f"Unpinned file: {file_path}")
        self.status_label.setText(f"Unpinned '{Path(file_path).name}'")
    
    def _pin_folder_from_tree(self, folder_path: str, item: QTreeWidgetItem):
        """Pin a folder from the tree view."""
        if settings.add_pinned_path(folder_path):
            # Remove all files in this folder from current moves
            folder_name = Path(folder_path).name
            if folder_name in self.current_plan.get("folders", {}):
                file_ids = self.current_plan["folders"][folder_name]
                self.current_moves = [m for m in self.current_moves 
                                      if m.get("file_id") not in file_ids 
                                      and str(m.get("file_id")) not in file_ids]
                del self.current_plan["folders"][folder_name]
            
            # Redisplay the plan
            self._display_plan(self.current_plan)
            
            logger.info(f"Pinned folder: {folder_path}")
            self.status_label.setText(f"📌 Pinned folder '{folder_name}' - removed from plan")
    
    def _unpin_folder_from_tree(self, folder_path: str, item: QTreeWidgetItem):
        """Unpin a folder from the tree view."""
        settings.remove_pinned_path(folder_path)
        folder_name = Path(folder_path).name
        logger.info(f"Unpinned folder: {folder_path}")
        self.status_label.setText(f"Unpinned folder '{folder_name}'")

    def apply_organization(self):
        """Execute the organization plan after user confirmation."""
        logger.info(f"apply_organization called. current_moves count: {len(self.current_moves)}")
        
        if not self.current_moves:
            logger.warning("Apply clicked but current_moves is empty")
            ModernInfoDialog.show_warning(
                self,
                title="No Files to Move",
                message="No files can be moved.",
                details=[
                    "Files have already been moved or deleted",
                    "Files no longer exist at their indexed paths"
                ],
                info_text="Try re-indexing your files in Index Files first."
            )
            return
        
        folder_count = len(self.current_plan.get("folders", {}))
        file_count = len(self.current_moves)
        
        # Use modern styled dialog matching app theme
        dialog = ModernConfirmDialog(
            parent=self,
            title="Confirm Organization",
            message=f"Move {file_count} files into {folder_count} folders?",
            highlight_text=str(self.destination_path),
            info_text="This will physically move the files.\nA log will be saved for reference.",
            yes_text="Move Files",
            no_text="Cancel"
        )
        
        if not dialog.exec():
            return
        
        # Final safety check: filter out any excluded files before moving
        filtered_moves = []
        excluded_count = 0
        for m in self.current_moves:
            if settings.should_exclude(m["source_path"]):
                excluded_count += 1
                logger.info(f"Skipping excluded file in apply: {m['file_name']}")
            else:
                filtered_moves.append(m)
        
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} files from final move (matched exclusion patterns)")
        
        if not filtered_moves:
            ModernInfoDialog.show_info(
                self,
                title="No Files to Move",
                message=f"All {excluded_count} file(s) matched exclusion patterns and were skipped.",
                info_text="You can modify exclusion patterns in Settings."
            )
            return
        
        move_plan = []
        for m in filtered_moves:
            move_plan.append({
                "source_path": m["source_path"],
                "destination_path": m["destination_path"],
                "file_name": m["file_name"],
                "size": m["size"],
                "category": m["destination_folder"],
            })
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(move_plan))
        self.apply_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.status_label.setText("Moving files...")
        
        success, errors, log_file, renamed_count = apply_moves(move_plan)
        
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        
        if success:
            # Save undo information BEFORE updating database paths
            self.last_organization = []
            for m in self.current_moves:
                self.last_organization.append({
                    "source": m["source_path"],
                    "destination": m["destination_path"],
                    "file_id": m["file_id"],
                })
            self.undo_button.setEnabled(True)
            logger.info(f"Saved {len(self.last_organization)} moves for potential undo")
            
            paths_updated = 0
            for m in self.current_moves:
                if file_index.update_file_path(m["file_id"], m["destination_path"]):
                    paths_updated += 1
            
            logger.info(f"Updated {paths_updated}/{len(self.current_moves)} file paths in database")
            
            # Collect source folders (where files came from) and scan destination too
            source_folders = {Path(m["source_path"]).parent for m in self.current_moves}
            empty_from_sources = self._collect_empty_folders(source_folders)
            empty_from_dest = self._scan_all_empty_folders()
            all_empty = list({*empty_from_sources, *empty_from_dest})
            all_empty.sort(key=lambda p: len(Path(p).parts), reverse=True)

            cleanup_msg = ""
            if all_empty:
                removed_count = self._delete_folders(all_empty)

                # Second pass: parents that became empty after their children were deleted
                parent_candidates = {
                    str(Path(p).parent) for p in all_empty
                    if len(Path(p).parent.parts) > 2
                }
                parent_candidates -= set(all_empty)
                if parent_candidates:
                    removed_count += self._delete_folders(list(parent_candidates))

                if removed_count > 0:
                    cleanup_msg = f"\n\nDeleted {removed_count} empty folder(s)."

            # Build details list for success dialog
            details = [
                f"Organized {len(move_plan)} file(s)",
                "File paths updated in database"
            ]
            if renamed_count > 0:
                details.append(f"{renamed_count} file(s) renamed to avoid duplicates")
            if cleanup_msg:
                details.append(cleanup_msg.strip())
            
            ModernInfoDialog.show_info(
                self,
                title="Organization Complete",
                message="Your files have been organized successfully!",
                details=details,
                info_text=f"You can use Undo to reverse this. Log saved to:\n{log_file}",
                ok_text="Done"
            )
            self.status_label.setText("Organization complete! (Undo available)")
            self.clear_plan()
            self._update_file_count()
        else:
            paths_updated = 0
            for m in self.current_moves:
                dest_path = Path(m["destination_path"])
                if dest_path.exists():
                    if file_index.update_file_path(m["file_id"], m["destination_path"]):
                        paths_updated += 1
            
            logger.info(f"Partial success: Updated {paths_updated} file paths in database")
            
            # Build error details (first 5 errors)
            error_details = errors[:5]
            if len(errors) > 5:
                error_details.append(f"... and {len(errors) - 5} more errors")
            
            ModernInfoDialog.show_warning(
                self,
                title="Partial Failure",
                message=f"Some files could not be moved. {paths_updated} file(s) were successfully organized.",
                details=error_details,
                info_text="Check the log file for more details."
            )
            self.status_label.setText(f"Completed with {len(errors)} errors")
            self.apply_button.setEnabled(True)

    def clear_plan(self):
        """Clear the current plan and reset UI."""
        self.current_plan = None
        self.current_moves = []
        self.original_instruction = None
        self.plan_tree.clear()
        self.apply_button.setEnabled(False)
        self.feedback_group.setVisible(False)
        self.feedback_input.clear()
        
        # Hide plan UI elements
        self.apply_button.setVisible(False)
        self.clear_button.setVisible(False)
        self.undo_button.setVisible(False)
        self._hide_plan_ui()
        
        # Show input cards again
        self._show_input_cards()
        
        self._update_file_count()
    
    def refine_plan(self):
        """Refine the current plan based on user feedback."""
        feedback = self.feedback_input.text().strip()
        if not feedback:
            return
        
        if not self.current_plan or not self.original_instruction:
            QMessageBox.warning(
                self, "No Plan to Refine",
                "Generate a plan first before refining."
            )
            return
        
        # Build refinement prompt
        from app.core.ai_organizer import request_plan_refinement
        
        files = self._load_files_from_db()
        if not files:
            return
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.generate_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.refine_button.setEnabled(False)
        self.status_label.setText(f"Refining plan based on feedback...")
        
        # Run refinement in background
        self.plan_worker = RefineWorker(
            self.original_instruction,
            self.current_plan,
            feedback,
            files
        )
        self.plan_worker.finished.connect(self._on_plan_received)
        self.plan_worker.error.connect(self._on_plan_error)
        self.plan_worker.finished.connect(lambda _: self.refine_button.setEnabled(True))
        self.plan_worker.error.connect(lambda _: self.refine_button.setEnabled(True))
        self.plan_worker.start()
    
    def _show_history_dialog(self):
        """Show the organization history dialog."""
        dialog = HistoryDialog(self)
        dialog.exec()
    
    def _show_pinned_dialog(self):
        """Show the pinned items management dialog."""
        dialog = PinnedDialog(self)
        dialog.exec()
    
    def undo_last_organization(self):
        """Undo the last organization by moving files back to their original locations."""
        if not self.last_organization:
            QMessageBox.information(
                self, "Nothing to Undo",
                "There is no previous organization to undo."
            )
            return
        
        can_undo = []
        cannot_undo = []
        
        for move in self.last_organization:
            dest_path = Path(move["destination"])
            source_path = Path(move["source"])
            
            if not dest_path.exists():
                cannot_undo.append(f"File not found: {dest_path.name}")
            elif source_path.exists():
                cannot_undo.append(f"Original location occupied: {source_path.name}")
            else:
                can_undo.append(move)
        
        if not can_undo:
            QMessageBox.warning(
                self, "Cannot Undo",
                f"Cannot undo the last organization:\n\n" +
                "\n".join(cannot_undo[:5]) +
                (f"\n... and {len(cannot_undo) - 5} more issues" if len(cannot_undo) > 5 else "")
            )
            return
        
        warning_text = ""
        if cannot_undo:
            warning_text = f"\n\n{len(cannot_undo)} files cannot be undone (modified or moved)."
        
        reply = QMessageBox.question(
            self,
            "Confirm Undo",
            f"Move {len(can_undo)} files back to their original locations?{warning_text}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        import shutil
        success_count = 0
        errors = []
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(can_undo))
        self.status_label.setText("Undoing organization...")
        
        for i, move in enumerate(can_undo):
            self.progress_bar.setValue(i + 1)
            try:
                dest_path = Path(move["destination"])
                source_path = Path(move["source"])
                
                source_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.move(str(dest_path), str(source_path))
                
                file_index.update_file_path(move["file_id"], str(source_path))
                
                success_count += 1
                logger.info(f"Undo: {dest_path} -> {source_path}")
            except Exception as e:
                errors.append(f"{dest_path.name}: {e}")
                logger.error(f"Undo failed for {move['destination']}: {e}")
        
        self.progress_bar.setVisible(False)
        
        self._cleanup_empty_folders()
        
        self.last_organization = None
        self.undo_button.setEnabled(False)
        
        if errors:
            QMessageBox.warning(
                self, "Partial Undo",
                f"Restored {success_count} files.\n\n"
                f"{len(errors)} files could not be restored:\n" +
                "\n".join(errors[:3])
            )
            self.status_label.setText(f"Undo partial: {success_count} restored, {len(errors)} failed")
        else:
            QMessageBox.information(
                self, "Undo Complete",
                f"Successfully restored {success_count} files to their original locations!"
            )
            self.status_label.setText(f"Undo complete: {success_count} files restored")
        
        self._update_file_count()
    
    def _collect_empty_folders(self, source_folders: set) -> list:
        """
        Collect empty source folders after moving files out (does NOT delete).
        
        Safety rules:
        - Only checks folders from the provided set (where files came from)
        - Only includes if completely empty (no files, no subfolders)
        - Walks bottom-up (deepest folders first)
        - Never includes the destination path itself
        - Recursively checks parent folders up the tree (with depth limit)
        - Returns list of empty folder paths for user review
        """
        empty_folders = []
        already_checked = set()
        
        if not source_folders:
            logger.debug("No source folders to check for emptiness")
            return empty_folders
        
        logger.info(f"Checking {len(source_folders)} source folders for emptiness")
        
        # Sort by depth (deepest first) to handle nested empty folders
        sorted_folders = sorted(source_folders, key=lambda p: len(p.parts), reverse=True)
        
        # Track the minimum depth we should check (don't go too far up)
        min_depths = {}
        for folder in sorted_folders:
            # Allow checking up to 3 levels above the source folder
            min_depths[folder] = max(1, len(folder.parts) - 3)
        
        def check_folder_and_parents(folder: Path, min_depth: int):
            """Recursively check folder and its parents if empty."""
            if folder in already_checked:
                return
            already_checked.add(folder)
            
            # Safety: never include destination path
            if self.destination_path and folder.resolve() == self.destination_path.resolve():
                logger.debug(f"Skipping destination folder: {folder}")
                return
            
            # Safety: don't go above min depth (prevents deleting too far up the tree)
            if len(folder.parts) < min_depth:
                logger.debug(f"Reached min depth, stopping at: {folder}")
                return
            
            # Safety: don't delete drive roots or very short paths
            if len(folder.parts) <= 2:
                logger.debug(f"Too close to root, skipping: {folder}")
                return
            
            # Safety: must exist and be a directory
            if not folder.exists() or not folder.is_dir():
                logger.debug(f"Folder doesn't exist or not a dir: {folder}")
                return
            
            try:
                _META = {'.DS_Store', '.localized', 'Thumbs.db', 'desktop.ini'}
                real_contents = [p for p in folder.iterdir() if p.name not in _META]
                if not real_contents:
                    empty_folders.append(str(folder))
                    logger.info(f"Found empty source folder: {folder}")

                    # Recursively check parent
                    check_folder_and_parents(folder.parent, min_depth)
                else:
                    logger.debug(f"Folder not empty ({len(real_contents)} items): {folder}")
            except OSError as e:
                logger.debug(f"Could not check folder {folder}: {e}")
            except Exception as e:
                logger.warning(f"Error checking folder {folder}: {e}")
        
        for folder in sorted_folders:
            min_depth = min_depths.get(folder, 1)
            logger.debug(f"Checking folder: {folder} (min_depth={min_depth})")
            check_folder_and_parents(folder, min_depth)
        
        logger.info(f"Found {len(empty_folders)} empty folders total")
        return empty_folders
    
    def _scan_all_empty_folders(self) -> list:
        """
        Scan the entire destination folder for empty folders.
        
        This finds ALL empty folders, not just source folders of moved files.
        Returns a list of empty folder paths sorted by depth (deepest first).
        """
        if not self.destination_path or not self.destination_path.exists():
            logger.debug("No destination path set for empty folder scan")
            return []
        
        empty_folders = []
        
        logger.info(f"Scanning entire destination for empty folders: {self.destination_path}")
        
        try:
            # Walk bottom-up (topdown=False) to find empty folders
            for dirpath, dirnames, filenames in os.walk(str(self.destination_path), topdown=False):
                folder = Path(dirpath)
                
                # Skip the destination path itself
                if folder.resolve() == self.destination_path.resolve():
                    continue
                
                # Safety: don't check paths too close to root
                if len(folder.parts) <= 2:
                    continue
                
                try:
                    _META = {'.DS_Store', '.localized', 'Thumbs.db', 'desktop.ini'}
                    real_contents = [p for p in folder.iterdir() if p.name not in _META]
                    if not real_contents:
                        empty_folders.append(str(folder))
                        logger.info(f"Found empty folder: {folder}")
                except OSError as e:
                    logger.debug(f"Could not check folder {folder}: {e}")
                except Exception as e:
                    logger.warning(f"Error checking folder {folder}: {e}")
        
        except Exception as e:
            logger.error(f"Error scanning destination folder: {e}")
        
        # Sort by depth (deepest first) for proper deletion order
        empty_folders.sort(key=lambda p: len(Path(p).parts), reverse=True)
        
        logger.info(f"Scan complete: found {len(empty_folders)} empty folders")
        return empty_folders
    
    def _show_empty_folder_dialog(self, empty_folders: list) -> int:
        """
        Show dialog for user to choose which empty folders to delete.
        Returns the number of folders actually deleted.
        """
        if not empty_folders:
            return 0
        
        dialog = EmptyFolderDialog(empty_folders, self)
        result = dialog.exec()
        
        if result == QDialog.Accepted:
            folders_to_delete = dialog.get_folders_to_delete()
            if folders_to_delete:
                return self._delete_folders(folders_to_delete)
        
        return 0
    
    def _delete_folders(self, folder_paths: list) -> int:
        """
        Delete the specified folders. Returns count of successfully deleted folders.

        After deleting an empty folder, walk UP and delete any parent that just
        became empty as a result. Without this second step, an intermediate
        empty folder (e.g. `technology-photography/` when its only child
        `technology-photography/laptop/` was just removed) gets left behind
        because the initial scan ran before any deletion happened — so the
        parent wasn't empty at scan time.
        """
        deleted_count = 0
        deleted_set: set = set()

        sorted_paths = sorted(folder_paths, key=lambda p: len(Path(p).parts), reverse=True)

        _METADATA_FILES = {'.DS_Store', '.localized', 'Thumbs.db', 'desktop.ini'}

        # Safety guard: never walk above the destination folder (or above
        # depth 3 if we don't know the destination yet).
        try:
            dest_root = self.destination_path.resolve() if self.destination_path else None
        except Exception:
            dest_root = None

        def _try_delete(folder: Path) -> bool:
            try:
                if not folder.exists() or not folder.is_dir():
                    return False
                # Refuse to delete the destination root itself or anything
                # OUTSIDE it. We allow anything INSIDE the destination, where
                # dest_root appears in folder.parents.
                if dest_root is not None:
                    try:
                        resolved = folder.resolve()
                        if resolved == dest_root:
                            return False
                        if dest_root not in resolved.parents:
                            return False
                    except Exception:
                        # If resolution fails for any reason, fall through to
                        # the content check; rmdir is safe on non-empty dirs.
                        pass
                # Hidden meta files (.DS_Store etc.) shouldn't block deletion.
                for meta in folder.iterdir():
                    if meta.name in _METADATA_FILES and meta.is_file():
                        try:
                            meta.unlink(missing_ok=True)
                        except Exception:
                            pass
                real_contents = [p for p in folder.iterdir() if p.name not in _METADATA_FILES]
                if real_contents:
                    logger.debug(f"Folder not empty, skipping: {folder}")
                    return False
                folder.rmdir()
                logger.info(f"Deleted empty folder: {folder}")
                return True
            except OSError as e:
                logger.warning(f"Could not delete folder {folder}: {e}")
                return False
            except Exception as e:
                logger.error(f"Error deleting folder {folder}: {e}")
                return False

        for folder_path in sorted_paths:
            folder = Path(folder_path)
            if folder in deleted_set:
                continue
            if _try_delete(folder):
                deleted_count += 1
                deleted_set.add(folder)
                # Walk UP and clean any parent that just became empty.
                parent = folder.parent
                while True:
                    if parent in deleted_set:
                        break
                    if dest_root is not None:
                        try:
                            if parent.resolve() == dest_root:
                                break
                        except Exception:
                            break
                    if not _try_delete(parent):
                        break
                    deleted_count += 1
                    deleted_set.add(parent)
                    parent = parent.parent

        return deleted_count
    
    def _cleanup_empty_folders(self):
        """Remove empty folders left after undo (only in destination path)."""
        if not self.destination_path or not self.destination_path.exists():
            return
        
        try:
            for dirpath, dirnames, filenames in os.walk(str(self.destination_path), topdown=False):
                if not filenames and not dirnames:
                    try:
                        Path(dirpath).rmdir()
                        logger.info(f"Removed empty folder: {dirpath}")
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Error cleaning up folders: {e}")
