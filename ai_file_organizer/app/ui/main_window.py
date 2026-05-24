"""
Main application window using PySide6.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QProgressBar, QStatusBar,
    QHeaderView, QGroupBox, QTextEdit, QSplitter, QTabWidget,
    QLineEdit, QCompleter, QListWidget, QListWidgetItem, QComboBox,
    QApplication, QCheckBox, QProgressDialog, QInputDialog, QFrame,
    QSizePolicy, QStackedWidget, QButtonGroup, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QFont, QIcon, QDesktopServices, QShortcut, QKeySequence
import json
import os
import subprocess
import time

from app.core.scan import scan_directory, get_directory_stats
from app.core.plan import create_move_plan, validate_move_plan, get_plan_summary
from app.core.apply import apply_moves, validate_destination_space
from app.core.settings import settings
from app.core.search import search_service
from app.core.database import file_index
from app.core.supabase_client import supabase_auth
from app.core.query_parser import (
    parse_query, get_date_range, TYPE_EXTENSIONS,
    UI_DATE_MAPPING, UI_TYPE_MAPPING, FILTER_TO_UI_DATE, FILTER_TO_UI_TYPE
)
from app.ui.quick_search_overlay import QuickSearchOverlay
from app.ui.win_hotkey import register_global_hotkey, unregister_global_hotkey, get_foreground_hwnd, set_foreground_hwnd, set_foreground_hwnd_robust, get_window_rect
from app.ui.theme_manager import theme_manager
from app.ui.organize_page import OrganizePage
from app.ui.onboarding import OnboardingOverlay
from app.ui.contextual_tips import ContextualTipsManager


logger = logging.getLogger(__name__)

# QuickSearch heuristics: localized button/label names
CONFIRM_NAMES = [
    "Open", "Save", "OK", "Select", "Choose"
]
FILENAME_LABELS = [
    "File name:", "Filename:", "Name:", "Dateiname:", "Nom du fichier:", "Nombre de archivo:",
]


class ScanWorker(QThread):
    """Worker thread for directory scanning."""
    
    scan_completed = Signal(list)
    scan_error = Signal(str)
    progress_updated = Signal(str)
    
    def __init__(self, source_path: Path):
        super().__init__()
        self.source_path = source_path
    
    def run(self):
        try:
            self.progress_updated.emit("Scanning directory...")
            files = scan_directory(self.source_path)
            
            # Add source path to each file metadata
            for file_data in files:
                file_data['source_path'] = str(self.source_path / file_data['name'])
            
            self.scan_completed.emit(files)
        except Exception as e:
            self.scan_error.emit(str(e))


class IndexWorker(QThread):
    """Worker thread for directory indexing with pause/resume support."""
    
    index_completed = Signal(dict)
    index_error = Signal(str)
    progress_updated = Signal(str)
    progress_percent = Signal(int, int, int)  # current, total, percent
    progress_data = Signal(int, int, str)  # done, total, message - for UI updates
    
    def __init__(self, directory_path: Path):
        super().__init__()
        self.directory_path = directory_path
        self._paused = False
        self._cancelled = False
        self._pause_condition = None
    
    def pause(self):
        """Pause the indexing process."""
        self._paused = True
    
    def resume(self):
        """Resume the indexing process."""
        self._paused = False
    
    def cancel(self):
        """Cancel the indexing process."""
        self._cancelled = True
        self._paused = False  # Unpause so it can exit
    
    def is_paused(self) -> bool:
        """Check if indexing is paused."""
        return self._paused
    
    def is_cancelled(self) -> bool:
        """Check if indexing was cancelled."""
        return self._cancelled
    
    def wait_if_paused(self):
        """Block while paused (called from indexing loop)."""
        while self._paused and not self._cancelled:
            import time
            time.sleep(0.1)  # Check every 100ms
    
    def run(self):
        try:
            self.progress_updated.emit("Indexing directory...")
            result = search_service.index_directory(self.directory_path)
            self.index_completed.emit(result)
        except Exception as e:
            self.index_error.emit(str(e))


class BatchOperationWorker(QThread):
    """Worker thread for batch file operations to prevent UI freezing."""
    
    operation_completed = Signal(dict)  # Result stats
    operation_error = Signal(str)
    progress_updated = Signal(int, int, str)  # current, total, message
    
    def __init__(self, operation: str, file_ids: list = None, file_paths: list = None, extra_data: dict = None):
        super().__init__()
        self.operation = operation
        self.file_ids = file_ids or []
        self.file_paths = file_paths or []
        self.extra_data = extra_data or {}
        self._cancelled = False
    
    def cancel(self):
        """Request cancellation of the operation."""
        self._cancelled = True
    
    def run(self):
        """Execute the batch operation."""
        try:
            from app.core.file_operations import get_file_operations
            file_ops = get_file_operations()
            
            if self.operation == 'remove':
                result = file_ops.remove_from_index(self.file_ids)
            elif self.operation == 'reindex':
                def progress_cb(current, total):
                    if not self._cancelled:
                        self.progress_updated.emit(current, total, f"Re-indexing file {current}/{total}...")
                result = file_ops.reindex_files(self.file_paths, progress_callback=progress_cb)
            elif self.operation == 'add_tags':
                tags = self.extra_data.get('tags', [])
                result = file_ops.batch_add_tags(self.file_ids, tags)
            else:
                result = {'error': f'Unknown operation: {self.operation}'}
            
            self.operation_completed.emit(result)
            
        except Exception as e:
            self.operation_error.emit(str(e))


class AutoIndexWorker(QThread):
    """Background worker for auto-indexing individual files."""
    
    file_indexed = Signal(str, str)  # (filename, status: 'success'|'skipped'|'error')
    status_update = Signal(str)  # status message for UI
    
    def __init__(self):
        super().__init__()
        self._queue = []
        self._running = False
    
    def add_file(self, file_path: Path):
        """Add a file to the indexing queue."""
        self._queue.append(file_path)
        if not self._running:
            self.start()
    
    def run(self):
        """Process files in the queue."""
        import hashlib
        from datetime import datetime
        from app.core.categorize import get_file_metadata
        from app.core.database import file_index
        from app.core.vision import analyze_image, gpt_vision_fallback, _file_to_b64
        from app.core.settings import settings
        
        self._running = True
        
        while self._queue:
            file_path = self._queue.pop(0)
            
            try:
                # Check if file already exists in index with tags
                existing = file_index.get_file_by_path(str(file_path))
                if existing:
                    has_tags = existing.get('tags') and existing['tags'] not in ['[]', '', None]
                    has_label = existing.get('label') and existing['label'] not in ['', None]
                    has_caption = existing.get('caption') and existing['caption'] not in ['', None]
                    
                    if has_tags or has_label or has_caption:
                        logger.info(f"Skipping already indexed file: {file_path}")
                        self.file_indexed.emit(file_path.name, 'skipped')
                        continue
                
                # Get basic metadata
                metadata = get_file_metadata(file_path)
                metadata['source_path'] = str(file_path)
                
                # Compute content hash
                try:
                    h = hashlib.sha256()
                    with open(file_path, 'rb') as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                            h.update(chunk)
                    metadata['content_hash'] = h.hexdigest()
                except Exception:
                    metadata['content_hash'] = None
                
                metadata['last_indexed_at'] = datetime.utcnow().isoformat()
                
                # AI Vision analysis for images (uses configured AI provider)
                ext = file_path.suffix.lower()
                if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.avif', '.heic', '.heif', '.ico', '.raw', '.cr2', '.nef', '.arw', '.pdf'}:
                    self.status_update.emit(f"Analyzing: {file_path.name}")
                    
                    # analyze_image handles provider selection internally
                    vision = analyze_image(file_path)
                    if vision:
                        metadata.update(vision)
                        metadata['ai_source'] = settings.ai_provider
                
                # Add to index
                file_index.add_file(metadata)
                logger.info(f"Auto-indexed: {file_path}")
                self.file_indexed.emit(file_path.name, 'success')
                
            except Exception as e:
                logger.error(f"Error auto-indexing {file_path}: {e}")
                self.file_indexed.emit(file_path.name, 'error')
        
        self._running = False


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.source_path = None
        self.destination_path = None
        self.scanned_files = []
        self.move_plan = []
        
        # Indexing queue system
        self.index_queue = []  # List of Path objects to index
        self.is_indexing = False  # Whether indexing is in progress
        
        self.setup_ui()
        self.setup_connections()
        self.setup_quick_search()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Initialize auto-index if enabled
        if settings.auto_index_downloads:
            self._start_downloads_watcher()
        
        # Auto-load indexed files on startup
        QTimer.singleShot(100, self.refresh_debug_view)
        
        # Update usage labels on startup (after auth is ready)
        QTimer.singleShot(500, self._update_usage_labels)
        
        # Run database cleanup in background to remove stale entries
        # This prevents UNIQUE constraint errors from orphaned records
        QTimer.singleShot(2000, self._run_background_db_cleanup)
        
        # Flag to track if onboarding has been shown this session
        self._onboarding_shown_this_session = False
        
        # Contextual tips manager
        self.tips_manager = ContextualTipsManager(self, settings)
        
        logger.info("Main window initialized")
    
    def showEvent(self, event):
        """Handle window show event - show onboarding on first launch"""
        super().showEvent(event)
        
        # Apply dark/light title bar now that the window has a valid HWND
        theme_manager._apply_windows_titlebar(theme_manager.current_theme)
        
        # Only show onboarding once per session and if not completed
        # Also stop showing if user clicked "Remind Me Later" 3+ times
        remind_count = getattr(settings, 'onboarding_remind_count', 0)
        should_show = (
            not self._onboarding_shown_this_session 
            and not settings.has_completed_onboarding
            and remind_count < 3
        )
        if should_show:
            self._onboarding_shown_this_session = True
            # Delay slightly to ensure window is fully visible
            QTimer.singleShot(500, self._show_onboarding)
        else:
            # Show contextual tips if onboarding is done
            QTimer.singleShot(1000, self._init_contextual_tips)
        
        # Check for updates in background (delay to not slow startup)
        QTimer.singleShot(3000, self._check_for_updates)
    
    def _show_onboarding(self):
        """Display the onboarding overlay"""
        try:
            self.onboarding = OnboardingOverlay(self)
            self.onboarding.finished_onboarding.connect(self._on_onboarding_finished)
            self.onboarding.remind_later.connect(self._on_onboarding_remind_later)
            self.onboarding.show()
        except Exception as e:
            logger.error(f"Failed to show onboarding: {e}")
    
    def _on_onboarding_finished(self):
        """Handle onboarding completion"""
        settings.complete_onboarding()
        logger.info("Onboarding completed")
        # Now show contextual tips
        QTimer.singleShot(500, self._init_contextual_tips)
    
    def _on_onboarding_remind_later(self):
        """Handle 'Remind Me Later' - will show again next launch"""
        # Don't mark as completed - just increment remind count
        remind_count = getattr(settings, 'onboarding_remind_count', 0) + 1
        settings.onboarding_remind_count = remind_count
        settings._save_config()
        logger.info(f"Onboarding reminder set (count: {remind_count})")
    
    def _check_for_updates(self):
        """Check for app updates in background via Supabase."""
        try:
            from app.version import VERSION
            from app.core.update_checker import check_for_updates
            
            logger.info("[UPDATE] Starting update check...")
            
            # Check synchronously since it's already delayed by 3 seconds
            update_info = check_for_updates(VERSION)
            if update_info:
                logger.info(f"[UPDATE] Update found, showing notification...")
                self._show_update_notification(update_info)
            else:
                logger.info("[UPDATE] No update available or check failed")
            
        except Exception as e:
            logger.debug(f"Could not check for updates: {e}")
    
    def _show_update_notification(self, update_info: dict):
        """Show update notification dialog."""
        try:
            logger.info(f"[UPDATE] Showing update notification: {update_info}")
            from app.ui.organize_page import UpdateNotificationDialog
            dialog = UpdateNotificationDialog.show_update(self, update_info)
            logger.info(f"[UPDATE] Update dialog created: {dialog}")
        except Exception as e:
            logger.error(f"Could not show update notification: {e}", exc_info=True)
        # Still show contextual tips even if they skipped
        QTimer.singleShot(500, self._init_contextual_tips)
    
    def _init_contextual_tips(self):
        """Initialize contextual tips for various UI elements"""
        try:
            # Organize page tips
            if hasattr(self, 'organize_page'):
                op = self.organize_page
                
                # History button - force below
                if hasattr(op, 'history_button'):
                    self.tips_manager.add_tip("history_button", op.history_button, force_position="below")
                
                # Pinned button - force below
                if hasattr(op, 'pinned_button'):
                    self.tips_manager.add_tip("pinned_button", op.pinned_button, force_position="below")
                
                # Undo button
                if hasattr(op, 'undo_button'):
                    self.tips_manager.add_tip("undo_button", op.undo_button, force_position="below")
                
                # Apply button
                if hasattr(op, 'apply_button'):
                    self.tips_manager.add_tip("apply_button", op.apply_button, force_position="below")
                
                # Edit button
                if hasattr(op, 'edit_button'):
                    self.tips_manager.add_tip("edit_button", op.edit_button, force_position="below")
                
                # Voice button (mic) - force below
                if hasattr(op, 'mic_button'):
                    self.tips_manager.add_tip("voice_button", op.mic_button, force_position="below")
            
            # Search page tips
            if hasattr(self, 'search_input'):
                self.tips_manager.add_tip("search_input", self.search_input)
            
            # Settings page tips
            if hasattr(self, 'welcome_guide_btn'):
                self.tips_manager.add_tip("welcome_guide_button", self.welcome_guide_btn)
            
            # Exclusions section
            if hasattr(self, 'exclusions_toggle_btn'):
                self.tips_manager.add_tip("exclusions_section", self.exclusions_toggle_btn)
            
            # Now show tips for currently visible widgets only
            QTimer.singleShot(200, self.tips_manager.show_tips_for_visible_widgets)
            
            logger.info("Contextual tips initialized")
        except Exception as e:
            logger.error(f"Failed to initialize contextual tips: {e}")
    
    def setup_ui(self):
        """Setup the user interface with modern sidebar navigation."""
        self.setWindowTitle("Filect - File Search Assistant")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout: Sidebar | Content
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left Sidebar (fixed width ~220px)
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(220)
        self.setup_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Content area with stacked pages
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        main_layout.addWidget(self.page_stack)
        
        # Create pages (order matters - matches nav button indices)
        self.setup_search_page()      # Index 0
        self.setup_organize_page()    # Index 1
        self.setup_index_page()       # Index 2
        self.setup_settings_page()    # Index 3
        
        # Set default page to Search
        self.page_stack.setCurrentIndex(0)
        self.nav_buttons[0].setChecked(True)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("statusBar")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Refresh account info after all UI is built
        self._refresh_account_info()
    
    def setup_sidebar(self):
        """Setup the left sidebar with navigation and account section."""
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Logo/App Title section
        logo_container = QWidget()
        logo_container.setObjectName("logoContainer")
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(20, 20, 20, 16)
        
        logo_icon = QLabel("✦")
        logo_icon.setObjectName("logoIcon")
        logo_layout.addWidget(logo_icon)
        
        logo_text = QLabel("Filect")
        logo_text.setObjectName("logoText")
        logo_layout.addWidget(logo_text)
        logo_layout.addStretch()
        
        sidebar_layout.addWidget(logo_container)
        
        # Navigation buttons
        nav_container = QWidget()
        nav_container.setObjectName("navContainer")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(4)
        
        self.nav_buttons = []
        self.nav_button_group = QButtonGroup(self)
        self.nav_button_group.setExclusive(True)
        
        nav_items = [
            ("🔍", "Search", 0),
            ("🗂️", "Organize", 1),
            ("📁", "Index Files", 2),
            ("⚙️", "Settings", 3),
        ]
        
        for icon, text, idx in nav_items:
            btn = QPushButton(f"  {icon}  {text}")
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, i=idx: self._on_nav_clicked(i))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)
            self.nav_button_group.addButton(btn, idx)
        
        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_container, 1)  # Takes remaining space
        
        # Account section at bottom
        account_container = QWidget()
        account_container.setObjectName("accountContainer")
        account_layout = QHBoxLayout(account_container)
        account_layout.setContentsMargins(16, 12, 16, 16)
        account_layout.setSpacing(12)
        
        # Avatar circle with initials
        self.avatar_label = QLabel("?")
        self.avatar_label.setObjectName("avatarCircle")
        self.avatar_label.setFixedSize(40, 40)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        account_layout.addWidget(self.avatar_label)
        
        # Name and plan info
        info_container = QWidget()
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        
        self.account_name_label = QLabel("Not logged in")
        self.account_name_label.setObjectName("accountName")
        info_layout.addWidget(self.account_name_label)
        
        self.account_plan_label = QLabel("Free Plan")
        self.account_plan_label.setObjectName("accountPlan")
        info_layout.addWidget(self.account_plan_label)
        
        account_layout.addWidget(info_container, 1)
        
        # Make account section clickable to go to Settings
        account_container.setCursor(Qt.PointingHandCursor)
        account_container.mousePressEvent = lambda e: self._on_nav_clicked(3)  # Settings is index 3
        
        sidebar_layout.addWidget(account_container)
    
    def _on_nav_clicked(self, index: int):
        """Handle navigation button clicks."""
        # Hide tips IMMEDIATELY before page switch
        if hasattr(self, 'tips_manager'):
            self.tips_manager.hide_all_tips()
        
        self.page_stack.setCurrentIndex(index)
        
        # Show tips for new page after a tiny delay
        if hasattr(self, 'tips_manager'):
            QTimer.singleShot(150, self.tips_manager.show_tips_for_visible_widgets)
    
    def setup_organize_tab(self):
        """Setup the file organization tab."""
        organize_widget = QWidget()
        organize_layout = QVBoxLayout(organize_widget)
        
        # Folder selection group
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout(folder_group)
        
        # Source folder
        source_layout = QHBoxLayout()
        self.source_label = QLabel("Source folder: Not selected")
        self.source_label.setObjectName("secondaryLabel")
        self.source_button = QPushButton("Select Source Folder")
        source_layout.addWidget(self.source_label)
        source_layout.addWidget(self.source_button)
        folder_layout.addLayout(source_layout)
        
        # Destination folder
        dest_layout = QHBoxLayout()
        self.dest_label = QLabel("Destination folder: Not selected")
        self.dest_label.setObjectName("secondaryLabel")
        self.dest_button = QPushButton("Select Destination Folder")
        dest_layout.addWidget(self.dest_label)
        dest_layout.addWidget(self.dest_button)
        folder_layout.addLayout(dest_layout)
        
        organize_layout.addWidget(folder_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        self.scan_button = QPushButton("Scan & Plan (Dry Run)")
        self.scan_button.setEnabled(False)
        self.apply_button = QPushButton("Apply Moves")
        self.apply_button.setEnabled(False)
        action_layout.addWidget(self.scan_button)
        action_layout.addWidget(self.apply_button)
        organize_layout.addLayout(action_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        organize_layout.addWidget(self.progress_bar)
        
        # Results area
        results_splitter = QSplitter(Qt.Vertical)
        
        # File table
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels([
            "File Name", "Category", "Size", "Planned Destination"
        ])
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        results_splitter.addWidget(self.file_table)
        
        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setReadOnly(True)
        results_splitter.addWidget(self.summary_text)
        
        organize_layout.addWidget(results_splitter)
        
        # Add organize tab
        self.tab_widget.addTab(organize_widget, "Organize Files")
    
    def setup_search_page(self):
        """Setup the clean Search page with hero heading and modern search bar."""
        search_page = QWidget()
        search_page.setObjectName("searchPage")
        page_layout = QVBoxLayout(search_page)
        page_layout.setContentsMargins(40, 0, 40, 20)
        page_layout.setSpacing(0)
        
        # Top spacer for results mode (hidden in landing mode)
        self.search_top_spacer = QWidget()
        self.search_top_spacer.setFixedHeight(20)
        self.search_top_spacer.setVisible(False)
        page_layout.addWidget(self.search_top_spacer)
        
        # Hero section (landing state - hidden after search)
        self.hero_section = QWidget()
        hero_layout = QVBoxLayout(self.hero_section)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(12)
        
        # Hero heading
        self.hero_heading = QLabel("What are you looking for?")
        self.hero_heading.setObjectName("heroHeading")
        self.hero_heading.setAlignment(Qt.AlignCenter)
        hero_layout.addWidget(self.hero_heading)
        
        # Subtitle
        self.hero_subtitle = QLabel("Search across your entire local drive instantly.")
        self.hero_subtitle.setObjectName("heroSubtitle")
        self.hero_subtitle.setAlignment(Qt.AlignCenter)
        hero_layout.addWidget(self.hero_subtitle)
        
        hero_layout.addSpacing(32)
        
        # Top spacer to push hero content to ~35% from top
        self.hero_top_spacer = QWidget()
        self.hero_top_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        page_layout.addWidget(self.hero_top_spacer, 35)
        
        page_layout.addWidget(self.hero_section)
        
        # Modern Search Bar Container (large pill shape) - always visible
        self.search_container = QWidget()
        self.search_container.setObjectName("searchContainerLarge")
        self.search_container.setMinimumHeight(70)
        self.search_container.setMaximumHeight(74)
        self.search_container.setMinimumWidth(550)
        self.search_container.setMaximumWidth(750)
        search_bar_layout = QHBoxLayout(self.search_container)
        search_bar_layout.setContentsMargins(28, 12, 12, 12)
        search_bar_layout.setSpacing(16)
        
        # Search input - larger
        self.search_input = QLineEdit()
        self.search_input.setObjectName("heroSearchInputLarge")
        self.search_input.setPlaceholderText("Search files, folders, or describe what you need...")
        self.search_input.setMinimumHeight(50)
        search_bar_layout.addWidget(self.search_input, 1)
        
        # AI indicator
        self.ai_label = QLabel("✦ AI")
        self.ai_label.setObjectName("aiIndicatorLarge")
        search_bar_layout.addWidget(self.ai_label)
        
        # Round search button - larger
        self.search_button = QPushButton("→")
        self.search_button.setObjectName("searchSubmitBtnLarge")
        self.search_button.setFixedSize(50, 50)
        self.search_button.setCursor(Qt.PointingHandCursor)
        search_bar_layout.addWidget(self.search_button)
        
        # Center the search container
        search_row = QHBoxLayout()
        search_row.addStretch()
        search_row.addWidget(self.search_container)
        search_row.addStretch()
        page_layout.addLayout(search_row)
        
        page_layout.addSpacing(40)
        
        # Bottom spacer for landing mode (hidden after search)
        self.hero_bottom_spacer = QWidget()
        self.hero_bottom_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        page_layout.addWidget(self.hero_bottom_spacer, 50)
        
        # Search Results section (hidden until a search is made)
        self.results_container = QWidget()
        self.results_container.setObjectName("resultsContainer")
        results_layout = QVBoxLayout(self.results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        self.results_container.setVisible(False)  # Hide until search
        
        # Quick Actions bar (hidden by default, shown when files selected)
        self.quick_actions_widget = QWidget()
        self.quick_actions_widget.setObjectName("quickActionsBar")
        quick_actions_layout = QHBoxLayout(self.quick_actions_widget)
        quick_actions_layout.setContentsMargins(12, 8, 12, 8)
        quick_actions_layout.setSpacing(8)
        
        self.selection_count_label = QLabel("0 files selected")
        self.selection_count_label.setObjectName("selectionLabel")
        quick_actions_layout.addWidget(self.selection_count_label)
        
        quick_actions_layout.addWidget(self._create_separator())
        
        self.action_remove_btn = QPushButton("Remove")
        self.action_remove_btn.setObjectName("quickActionBtn")
        quick_actions_layout.addWidget(self.action_remove_btn)
        
        self.action_reindex_btn = QPushButton("Re-index")
        self.action_reindex_btn.setObjectName("quickActionBtn")
        quick_actions_layout.addWidget(self.action_reindex_btn)
        
        self.action_add_tags_btn = QPushButton("Add Tags")
        self.action_add_tags_btn.setObjectName("quickActionBtn")
        quick_actions_layout.addWidget(self.action_add_tags_btn)
        
        self.action_copy_paths_btn = QPushButton("Copy Paths")
        self.action_copy_paths_btn.setObjectName("quickActionBtn")
        quick_actions_layout.addWidget(self.action_copy_paths_btn)
        
        self.action_open_folders_btn = QPushButton("Open Folder")
        self.action_open_folders_btn.setObjectName("quickActionBtn")
        quick_actions_layout.addWidget(self.action_open_folders_btn)
        
        self.action_export_btn = QPushButton("Export")
        self.action_export_btn.setObjectName("quickActionBtn")
        quick_actions_layout.addWidget(self.action_export_btn)
        
        quick_actions_layout.addStretch()
        
        self.action_select_all_btn = QPushButton("Select All")
        self.action_select_all_btn.setObjectName("quickActionBtnSecondary")
        quick_actions_layout.addWidget(self.action_select_all_btn)
        
        self.action_clear_selection_btn = QPushButton("Clear")
        self.action_clear_selection_btn.setObjectName("quickActionBtnSecondary")
        quick_actions_layout.addWidget(self.action_clear_selection_btn)
        
        self.quick_actions_widget.setVisible(False)
        results_layout.addWidget(self.quick_actions_widget)
        
        # Search results table
        self.search_results_table = QTableWidget()
        self.search_results_table.setObjectName("searchResultsTable")
        self.search_results_table.setShowGrid(False)
        self.search_results_table.setAlternatingRowColors(True)
        self.search_results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.search_results_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.search_results_table.verticalHeader().setDefaultSectionSize(46)
        self.search_results_table.setColumnCount(5)
        self.search_results_table.setHorizontalHeaderLabels(["✓", "File Name", "Folder", "Size", "Actions"])
        
        search_header = self.search_results_table.horizontalHeader()
        search_header.setSectionResizeMode(QHeaderView.Interactive)
        search_header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.search_results_table.setColumnWidth(0, 40)
        search_header.setSectionResizeMode(1, QHeaderView.Stretch)
        search_header.setSectionResizeMode(2, QHeaderView.Stretch)
        search_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        search_header.setSectionResizeMode(4, QHeaderView.Interactive)
        self.search_results_table.setColumnWidth(4, 220)
        search_header.setStretchLastSection(True)
        self.search_results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.search_results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.search_results_table.cellDoubleClicked.connect(self._on_search_result_double_click)
        
        results_layout.addWidget(self.search_results_table)
        
        # Search statistics
        self.search_stats_label = QLabel("")
        self.search_stats_label.setObjectName("searchStatsLabel")
        results_layout.addWidget(self.search_stats_label)
        
        page_layout.addWidget(self.results_container, 1)
        
        # Hidden filter controls (used internally)
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All Types", "Images", "Documents", "PDFs", "Videos", "Audio", "Code"])
        self.type_filter.setCurrentIndex(0)
        self.type_filter.setVisible(False)
        
        self.date_filter = QComboBox()
        self.date_filter.addItems(["Any Time", "Today", "Yesterday", "This Week", "This Month", "This Year"])
        self.date_filter.setVisible(False)
        
        self.clear_filters_btn = QPushButton("Clear Filters")
        self.clear_filters_btn.setVisible(False)
        
        self.filter_status_label = QLabel("")
        self.filter_status_label.setVisible(False)
        
        self.search_debug_label = QLabel("")
        self.search_debug_label.setVisible(False)
        
        # Add page to stack
        self.page_stack.addWidget(search_page)
    
    def setup_index_page(self):
        """Setup the Index Management page with clean drop zone and View Files button."""
        from PySide6.QtWidgets import QScrollArea
        
        index_page = QWidget()
        index_page.setObjectName("indexPage")
        page_layout = QVBoxLayout(index_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        
        # Scroll area for the entire index page content
        scroll_area = QScrollArea()
        scroll_area.setObjectName("indexScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        # Scrollable content widget
        scroll_content = QWidget()
        scroll_content.setObjectName("indexScrollContent")
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(60, 50, 60, 40)
        scroll_content_layout.setSpacing(24)
        
        # Centered content container
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(32)
        
        # Drop zone for drag and drop - large centered card
        self.drop_zone = QWidget()
        self.drop_zone.setObjectName("dropZoneLarge")
        self.drop_zone.setMinimumHeight(320)
        self.drop_zone.setCursor(Qt.PointingHandCursor)
        self.drop_zone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignCenter)
        drop_layout.setSpacing(16)
        
        # Large icon
        self.drop_icon = QLabel("📁")
        self.drop_icon.setObjectName("dropIconLarge")
        self.drop_icon.setAlignment(Qt.AlignCenter)
        self.drop_icon.setStyleSheet("font-size: 64px; background: transparent;")
        drop_layout.addWidget(self.drop_icon, 0, Qt.AlignCenter)
        
        # Main text
        self.drop_title = QLabel("Add folder to index")
        self.drop_title.setObjectName("dropTitleLarge")
        self.drop_title.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(self.drop_title)
        
        # Subtitle
        self.drop_subtitle = QLabel("Drag and drop folders here, or click to browse")
        self.drop_subtitle.setObjectName("dropSubtitleLarge")
        self.drop_subtitle.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(self.drop_subtitle)
        
        # Make the entire widget clickable
        self.drop_zone.mousePressEvent = self._on_drop_zone_clicked
        content_layout.addWidget(self.drop_zone)
        
        # Progress section (hidden until indexing) - OUTSIDE the drop zone
        self.index_progress_container = QWidget()
        self.index_progress_container.setObjectName("progressContainer")
        progress_layout = QVBoxLayout(self.index_progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        
        # Progress bar (original format)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("indexProgressBar")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% complete (%v / %m files)")
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setProperty("paused", False)
        progress_layout.addWidget(self.progress_bar)
        
        # Progress label
        self.index_progress_label = QLabel("")
        self.index_progress_label.setObjectName("progressLabel")
        progress_layout.addWidget(self.index_progress_label)
        
        # Progress percentage
        self.index_percent_label = QLabel("0%")
        self.index_percent_label.setObjectName("percentLabel")
        self.index_percent_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.index_percent_label)
        
        # Pause/Cancel buttons
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        self.index_pause_btn = QPushButton("Pause")
        self.index_pause_btn.setObjectName("secondaryButton")
        btn_row.addWidget(self.index_pause_btn)
        
        self.index_cancel_btn = QPushButton("Cancel")
        self.index_cancel_btn.setObjectName("secondaryButton")
        btn_row.addWidget(self.index_cancel_btn)
        progress_layout.addLayout(btn_row)
        
        self.index_progress_container.setVisible(False)
        content_layout.addWidget(self.index_progress_container)
        
        # === More Options (collapsible) ===
        self.more_options_header = QPushButton("▶ More Options")
        self.more_options_header.setObjectName("moreOptionsHeader")
        self.more_options_header.setCursor(Qt.PointingHandCursor)
        self.more_options_header.setMinimumHeight(36)
        content_layout.addWidget(self.more_options_header, 0, Qt.AlignCenter)
        
        # More options content (hidden by default)
        self.more_options_content = QWidget()
        self.more_options_content.setObjectName("moreOptionsContent")
        self.more_options_content.setVisible(False)
        options_layout = QVBoxLayout(self.more_options_content)
        options_layout.setContentsMargins(0, 8, 0, 8)
        options_layout.setSpacing(12)
        
        # Watch for New Downloads - opens popup
        self.watch_header_btn = QPushButton("👁️ Watch for New Downloads")
        self.watch_header_btn.setObjectName("watchHeaderButton")
        self.watch_header_btn.setCursor(Qt.PointingHandCursor)
        self.watch_header_btn.setMinimumHeight(44)
        options_layout.addWidget(self.watch_header_btn, 0, Qt.AlignCenter)
        
        # Watch status indicator (shown inline)
        self.watch_status_label = QLabel("")
        self.watch_status_label.setObjectName("watchStatusLabel")
        self.watch_status_label.setAlignment(Qt.AlignCenter)
        options_layout.addWidget(self.watch_status_label)
        
        # Index Entire PC Now button
        self.index_pc_now_btn = QPushButton("⚡ Index Entire PC Now")
        self.index_pc_now_btn.setObjectName("indexPcButton")
        self.index_pc_now_btn.setCursor(Qt.PointingHandCursor)
        self.index_pc_now_btn.setMinimumHeight(44)
        self.index_pc_now_btn.setToolTip("Scan all files on your computer (one-time)")
        options_layout.addWidget(self.index_pc_now_btn, 0, Qt.AlignCenter)
        
        content_layout.addWidget(self.more_options_content)
        
        # Hidden widgets for compatibility
        self.watch_common_toggle = QPushButton()
        self.watch_common_toggle.setCheckable(True)
        self.watch_common_toggle.setVisible(False)
        self.watch_custom_toggle = QPushButton()
        self.watch_custom_toggle.setCheckable(True)
        self.watch_custom_toggle.setVisible(False)
        self.add_custom_folder_btn = QPushButton()
        self.add_custom_folder_btn.setVisible(False)
        self.custom_folders_list = QWidget()
        self.custom_folders_list_layout = QVBoxLayout(self.custom_folders_list)
        
        # View Indexed Files button - prominent and centered
        self.view_files_btn = QPushButton("View Indexed Files (0)")
        self.view_files_btn.setObjectName("viewFilesButton")
        self.view_files_btn.setCursor(Qt.PointingHandCursor)
        self.view_files_btn.setMinimumHeight(52)
        self.view_files_btn.setMinimumWidth(280)
        self.view_files_btn.clicked.connect(self._show_files_overlay)
        content_layout.addWidget(self.view_files_btn, 0, Qt.AlignCenter)
        
        # Usage indicator label - shows media files remaining
        self.usage_label = QLabel("")
        self.usage_label.setObjectName("usageLabel")
        self.usage_label.setAlignment(Qt.AlignCenter)
        self.usage_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #7A7A90;
                padding: 4px 0;
            }
        """)
        content_layout.addWidget(self.usage_label, 0, Qt.AlignCenter)
        
        # Add content container to scroll layout
        scroll_content_layout.addWidget(content_container, 1)
        
        # Set scroll content and add scroll area to page
        scroll_area.setWidget(scroll_content)
        page_layout.addWidget(scroll_area, 1)
        
        # Hidden compatibility stubs
        self.indexed_paths_list = None
        self.active_indicator = None
        self.clear_all_paths_btn = QPushButton()
        self.clear_all_paths_btn.setVisible(False)
        self.index_label = QLabel("")
        self.index_label.setVisible(False)
        self.index_button = QPushButton("")
        self.index_button.setVisible(False)
        self.index_button_action = QPushButton("")
        self.index_button_action.setVisible(False)
        self.queue_row = QFrame()
        self.queue_row.setVisible(False)
        self.queue_label = QLabel("")
        self.queue_items_label = QLabel("")
        self.clear_queue_btn = QPushButton("")
        self.queue_top_spacer = QWidget()
        self.queue_spacer = QWidget()
        self.quick_index_header = QPushButton("")
        self.quick_index_content = QWidget()
        self.index_pc_button = QPushButton("")
        self.auto_index_downloads_btn = QPushButton("")
        self.auto_index_status = QLabel("")
        self.advanced_header = QPushButton("")
        self.advanced_content = QWidget()
        self.refresh_debug_button = QPushButton("")
        self.refresh_debug_button.setVisible(False)
        self.clear_index_button = QPushButton("")
        self.clear_index_button.setVisible(False)
        
        # Info label (shown at bottom of page)
        self.debug_info_label = QLabel("")
        self.debug_info_label.setObjectName("infoLabel")
        self.debug_info_label.setAlignment(Qt.AlignCenter)
        page_layout.addWidget(self.debug_info_label)
        
        # Create the files table (will be shown in overlay)
        self._setup_files_table()
        
        # Add page to stack
        self.page_stack.addWidget(index_page)
    
    def _setup_files_table(self):
        """Setup the files table widget (used in overlay dialog)."""
        # Quick Actions bar for file management
        self.debug_quick_actions_widget = QWidget()
        self.debug_quick_actions_widget.setObjectName("quickActionsBar")
        debug_qa_layout = QHBoxLayout(self.debug_quick_actions_widget)
        debug_qa_layout.setContentsMargins(12, 8, 12, 8)
        debug_qa_layout.setSpacing(8)
        
        self.debug_selection_count_label = QLabel("0 files selected")
        self.debug_selection_count_label.setObjectName("selectionLabel")
        debug_qa_layout.addWidget(self.debug_selection_count_label)
        
        debug_qa_layout.addWidget(self._create_separator())
        
        self.debug_action_remove_btn = QPushButton("Remove")
        self.debug_action_remove_btn.setToolTip("Remove selected files from the index")
        self.debug_action_remove_btn.setObjectName("quickActionBtn")
        debug_qa_layout.addWidget(self.debug_action_remove_btn)
        
        self.debug_action_reindex_btn = QPushButton("Re-index")
        self.debug_action_reindex_btn.setObjectName("quickActionBtn")
        debug_qa_layout.addWidget(self.debug_action_reindex_btn)
        
        self.debug_action_add_tags_btn = QPushButton("Add Tags")
        self.debug_action_add_tags_btn.setObjectName("quickActionBtn")
        debug_qa_layout.addWidget(self.debug_action_add_tags_btn)
        
        self.debug_action_copy_paths_btn = QPushButton("Copy Paths")
        self.debug_action_copy_paths_btn.setObjectName("quickActionBtn")
        debug_qa_layout.addWidget(self.debug_action_copy_paths_btn)
        
        self.debug_action_open_folders_btn = QPushButton("Open Folder")
        self.debug_action_open_folders_btn.setObjectName("quickActionBtn")
        debug_qa_layout.addWidget(self.debug_action_open_folders_btn)
        
        self.debug_action_export_btn = QPushButton("Export")
        self.debug_action_export_btn.setObjectName("quickActionBtn")
        debug_qa_layout.addWidget(self.debug_action_export_btn)
        
        debug_qa_layout.addStretch()
        
        self.debug_action_select_all_btn = QPushButton("Select All")
        self.debug_action_select_all_btn.setObjectName("quickActionBtnSecondary")
        debug_qa_layout.addWidget(self.debug_action_select_all_btn)
        
        self.debug_action_clear_selection_btn = QPushButton("Clear")
        self.debug_action_clear_selection_btn.setObjectName("quickActionBtnSecondary")
        debug_qa_layout.addWidget(self.debug_action_clear_selection_btn)
        
        self.debug_quick_actions_widget.setVisible(False)
        
        # Simplified files table
        self.debug_table = QTableWidget()
        self.debug_table.setObjectName("filesTable")
        self.debug_table.setShowGrid(False)
        self.debug_table.setAlternatingRowColors(True)
        self.debug_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.debug_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.debug_table.setColumnCount(16)
        self.debug_table.verticalHeader().setDefaultSectionSize(48)
        self.debug_table.setHorizontalHeaderLabels([
            "✓", "File Name", "Category", "Size", "Has OCR", "Label", "Tags", "Caption", 
            "OCR Text Preview", "AI Source", "Vision Score", "Purpose", "Suggested Filename", 
            "Detected Text", "File Path", "Actions"
        ])
        
        debug_header = self.debug_table.horizontalHeader()
        debug_header.setSectionResizeMode(0, QHeaderView.Fixed)
        self.debug_table.setColumnWidth(0, 40)
        debug_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(7, QHeaderView.Interactive)
        self.debug_table.setColumnWidth(7, 200)
        debug_header.setSectionResizeMode(8, QHeaderView.Interactive)
        self.debug_table.setColumnWidth(8, 200)
        debug_header.setSectionResizeMode(9, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(10, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(11, QHeaderView.ResizeToContents)
        debug_header.setSectionResizeMode(12, QHeaderView.Interactive)
        self.debug_table.setColumnWidth(12, 200)
        debug_header.setSectionResizeMode(13, QHeaderView.Interactive)
        self.debug_table.setColumnWidth(13, 150)
        debug_header.setSectionResizeMode(14, QHeaderView.Interactive)
        self.debug_table.setColumnWidth(14, 300)
        debug_header.setSectionResizeMode(15, QHeaderView.Interactive)
        self.debug_table.setColumnWidth(15, 140)
        
        # Connect signals
        self.debug_table.itemChanged.connect(self.on_debug_cell_changed)
        self.debug_table.cellDoubleClicked.connect(self.on_debug_cell_double_clicked)
    
    def _show_files_overlay(self):
        """Show fullscreen overlay with indexed files table."""
        from PySide6.QtWidgets import QDialog, QFrame
        
        # Detect current theme
        is_dark = settings.theme == 'dark'
        
        # Theme-aware colors
        if is_dark:
            bg_color = "#1a1a2e"
            card_bg = "#252540"
            text_color = "#FFFFFF"
            subtitle_color = "#7A7A90"
            border_color = "#3A3A5A"
            close_bg = "rgba(255, 255, 255, 0.1)"
            close_hover = "rgba(124, 77, 255, 0.2)"
        else:
            bg_color = "#FFFFFF"
            card_bg = "#F8F6FF"
            text_color = "#1A1A1A"
            subtitle_color = "#666666"
            border_color = "#E8E8E8"
            close_bg = "rgba(0, 0, 0, 0.05)"
            close_hover = "rgba(124, 77, 255, 0.15)"
        
        # Create overlay dialog
        overlay = QDialog(self)
        overlay.setWindowTitle("Indexed Files")
        overlay.setObjectName("filesOverlay")
        overlay.setModal(True)
        overlay.resize(int(self.width() * 0.92), int(self.height() * 0.88))
        
        # Apply scrollbar styling at dialog level - this ensures ALL scrollbars in the dialog are styled
        overlay.setStyleSheet("""
            QScrollBar:vertical {
                background: #7A7A90;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #252535;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9575FF;
            }
            QScrollBar:horizontal {
                background: #7A7A90;
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #252535;
                min-width: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #9575FF;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
        """)
        
        # Center the dialog
        overlay.move(
            self.x() + (self.width() - overlay.width()) // 2,
            self.y() + (self.height() - overlay.height()) // 2
        )
        
        layout = QVBoxLayout(overlay)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(20)
        
        # Header with title and close button
        header = QHBoxLayout()
        header.setSpacing(12)
        
        # Folder icon to match the index page
        icon_label = QLabel("📁")
        icon_label.setStyleSheet(f"font-size: 28px; background: transparent;")
        header.addWidget(icon_label)
        
        # Title + usage indicator stacked vertically
        title_container = QVBoxLayout()
        title_container.setSpacing(2)
        title_container.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("Indexed Files")
        title.setStyleSheet(f"font-size: 24px; font-weight: 600; color: #7C4DFF; background: transparent;")
        title_container.addWidget(title)
        
        # Usage indicator in overlay
        self._overlay_usage_label = QLabel("")
        self._overlay_usage_label.setStyleSheet(f"""
            font-size: 11px; 
            color: {subtitle_color}; 
            background: transparent;
            padding-left: 2px;
        """)
        title_container.addWidget(self._overlay_usage_label)
        
        header.addLayout(title_container)
        
        header.addSpacing(20)
        
        # Search bar in header
        self._files_search_input = QLineEdit()
        self._files_search_input.setPlaceholderText("🔍 Search files...")
        self._files_search_input.setClearButtonEnabled(True)
        self._files_search_input.setFixedWidth(220)
        self._files_search_input.setFixedHeight(36)
        self._files_search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {card_bg};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                color: {text_color};
            }}
            QLineEdit:focus {{
                border-color: #7C4DFF;
            }}
            QLineEdit::placeholder {{
                color: {subtitle_color};
            }}
        """)
        self._files_search_input.textChanged.connect(self._filter_indexed_files)
        header.addWidget(self._files_search_input)
        
        header.addStretch()
        
        # Refresh button - styled to match theme
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setCursor(Qt.PointingHandCursor)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {card_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.1);
                border-color: #7C4DFF;
                color: #7C4DFF;
            }}
        """)
        refresh_btn.clicked.connect(self.refresh_debug_view)
        header.addWidget(refresh_btn)
        
        # Clear All button
        clear_btn = QPushButton("Clear All")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {card_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 100, 100, 0.1);
                border-color: #FF6B6B;
                color: #FF6B6B;
            }}
        """)
        clear_btn.clicked.connect(lambda: self._clear_all_indexed(overlay))
        header.addWidget(clear_btn)
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(40, 40)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {close_bg};
                border: none;
                border-radius: 20px;
                font-size: 18px;
                color: {subtitle_color};
            }}
            QPushButton:hover {{
                background: {close_hover};
                color: #7C4DFF;
            }}
        """)
        close_btn.clicked.connect(overlay.close)
        header.addWidget(close_btn)
        
        layout.addLayout(header)
        
        # Quick actions bar - apply theme-aware styling
        layout.addWidget(self.debug_quick_actions_widget)
        
        # Style the quick actions bar with clean ghost buttons
        ghost_btn_style = f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: #7C4DFF;
                font-size: 13px;
                font-weight: 500;
                padding: 8px 14px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background: rgba(124, 77, 255, 0.1);
            }}
            QPushButton:pressed {{
                background: rgba(124, 77, 255, 0.18);
            }}
        """
        
        secondary_btn_style = f"""
            QPushButton {{
                background: {card_bg};
                border: 1px solid {border_color};
                color: {subtitle_color};
                font-size: 12px;
                font-weight: 500;
                padding: 6px 12px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
                color: #7C4DFF;
            }}
        """
        
        selection_badge_style = f"""
            QLabel {{
                background: rgba(124, 77, 255, 0.12);
                color: #7C4DFF;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 14px;
                border-radius: 16px;
            }}
        """
        
        # Apply styles to action buttons
        self.debug_action_remove_btn.setStyleSheet(ghost_btn_style)
        self.debug_action_reindex_btn.setStyleSheet(ghost_btn_style)
        self.debug_action_add_tags_btn.setStyleSheet(ghost_btn_style)
        self.debug_action_copy_paths_btn.setStyleSheet(ghost_btn_style)
        self.debug_action_open_folders_btn.setStyleSheet(ghost_btn_style)
        self.debug_action_export_btn.setStyleSheet(ghost_btn_style)
        
        # Apply secondary styles
        self.debug_action_select_all_btn.setStyleSheet(secondary_btn_style)
        self.debug_action_clear_selection_btn.setStyleSheet(secondary_btn_style)
        
        # Apply selection badge style
        self.debug_selection_count_label.setStyleSheet(selection_badge_style)
        
        # Style the quick actions container
        self.debug_quick_actions_widget.setStyleSheet(f"""
            QWidget#quickActionsBar {{
                background: {card_bg};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 4px;
            }}
        """)
        
        # Table container with subtle styling
        table_container = QFrame()
        table_container.setObjectName("tableContainer")
        table_container.setStyleSheet(f"""
            QFrame#tableContainer {{
                background-color: {card_bg};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
        """)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        # Style the table for the current theme - use explicit colors, not theme variables
        # to ensure dark mode is always applied in this dialog
        if is_dark:
            tbl_bg = "#252540"
            tbl_alt_bg = "#1a1a2e"
            tbl_text = "#FFFFFF"
            tbl_border = "#3A3A5A"
            tbl_header_bg = "#1a1a2e"
            tbl_header_text = "#7A7A90"
        else:
            tbl_bg = "#FFFFFF"
            tbl_alt_bg = "#F8F6FF"
            tbl_text = "#1A1A1A"
            tbl_border = "#E8E8E8"
            tbl_header_bg = "#FFFFFF"
            tbl_header_text = "#666666"
        
        table_style = f"""
            QTableWidget, QTableWidget * {{
                background-color: {tbl_bg};
                color: {tbl_text};
            }}
            QTableWidget {{
                background-color: {tbl_bg};
                alternate-background-color: {tbl_alt_bg};
                color: {tbl_text};
                gridline-color: {tbl_border};
                border: none;
                selection-background-color: rgba(124, 77, 255, 0.25);
                selection-color: {tbl_text};
            }}
            QTableWidget QTableCornerButton::section {{
                background-color: {tbl_header_bg};
                border: none;
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {tbl_border};
                background-color: {tbl_bg};
                color: {tbl_text};
            }}
            QTableWidget::item:alternate {{
                background-color: {tbl_alt_bg};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(124, 77, 255, 0.25);
                color: {tbl_text};
            }}
            QHeaderView {{
                background-color: {tbl_header_bg};
            }}
            QHeaderView::section {{
                background-color: {tbl_header_bg};
                color: {tbl_header_text};
                padding: 10px 8px;
                border: none;
                border-bottom: 2px solid {tbl_border};
                font-weight: 600;
                font-size: 12px;
            }}
            QTableWidget QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid #5A5A70;
                border-radius: 4px;
                background: {tbl_bg};
            }}
            QTableWidget QCheckBox::indicator:checked {{
                background: #7C4DFF;
                border-color: #7C4DFF;
            }}
            #filesTable QScrollBar:vertical {{
                background: #7A7A90 !important;
                width: 8px;
                border-radius: 4px;
            }}
            #filesTable QScrollBar::handle:vertical {{
                background: #252535 !important;
                min-height: 30px;
                border-radius: 4px;
            }}
            #filesTable QScrollBar::handle:vertical:hover {{
                background: #9575FF !important;
            }}
            #filesTable QScrollBar:horizontal {{
                background: #7A7A90 !important;
                height: 8px;
                border-radius: 4px;
            }}
            #filesTable QScrollBar::handle:horizontal {{
                background: #252535 !important;
                min-width: 30px;
                border-radius: 4px;
            }}
            #filesTable QScrollBar::handle:horizontal:hover {{
                background: #9575FF !important;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                width: 0px;
                height: 0px;
            }}
        """
        self.debug_table.setStyleSheet(table_style)
        
        # Also style the viewport explicitly
        if self.debug_table.viewport():
            self.debug_table.viewport().setStyleSheet(f"background-color: {tbl_bg};")
        
        # Force scrollbar styling directly on scrollbar widgets
        scrollbar_style = """
            QScrollBar {
                background: #7A7A90;
                border-radius: 4px;
            }
            QScrollBar::handle {
                background: #252535;
                border-radius: 4px;
            }
            QScrollBar::handle:hover {
                background: #9575FF;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
        """
        if self.debug_table.verticalScrollBar():
            self.debug_table.verticalScrollBar().setStyleSheet(scrollbar_style)
        if self.debug_table.horizontalScrollBar():
            self.debug_table.horizontalScrollBar().setStyleSheet(scrollbar_style)
        
        table_layout.addWidget(self.debug_table)
        
        layout.addWidget(table_container, 1)
        
        # File count label
        count_label = QLabel(f"{self.debug_table.rowCount()} files indexed")
        count_label.setStyleSheet(f"color: {subtitle_color}; font-size: 13px; background: transparent;")
        count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(count_label)
        
        # Apply theme-aware styling to dialog
        overlay.setStyleSheet(f"""
            QDialog#filesOverlay {{
                background-color: {bg_color};
            }}
        """)
        
        # Apply dark/light title bar
        from app.ui.theme_manager import apply_titlebar_theme
        from PySide6.QtWidgets import QApplication
        
        # Populate table data BEFORE showing dialog
        self.refresh_debug_view()
        
        # Process events to ensure table layout is calculated with data
        QApplication.processEvents()
        
        # Force column resize after data is loaded
        for col in range(1, self.debug_table.columnCount()):
            self.debug_table.resizeColumnToContents(col)
        
        # Ensure minimum width for key columns
        if self.debug_table.columnWidth(1) < 200:  # File Name
            self.debug_table.setColumnWidth(1, 200)
        if self.debug_table.columnWidth(2) < 100:  # Category
            self.debug_table.setColumnWidth(2, 100)
        
        # Update usage labels before showing
        self._update_usage_labels()
        
        # Process events again to finalize layout before showing
        QApplication.processEvents()
        
        # Ensure overlay is properly sized and centered
        overlay.resize(int(self.width() * 0.92), int(self.height() * 0.88))
        overlay.move(
            self.x() + (self.width() - overlay.width()) // 2,
            self.y() + (self.height() - overlay.height()) // 2
        )
        
        # Show first to create HWND, then apply titlebar theme
        overlay.show()
        apply_titlebar_theme(overlay)
        
        # Execute the dialog (blocking)
        overlay.exec()
        
        # After closing, restore widgets to main window (for next time)
        # The widgets are still valid, just not displayed
    
    def _clear_all_indexed(self, dialog=None):
        """Clear all indexed files from the database."""
        from app.core.database import file_index
        from app.ui.organize_page import ModernConfirmDialog, ModernInfoDialog
        
        confirmed = ModernConfirmDialog.ask(
            self, 
            title="Clear Index",
            message="Are you sure you want to remove all indexed files?",
            info_text="This will not delete your actual files, only the search index.",
            yes_text="Clear",
            no_text="Cancel"
        )
        if confirmed:
            try:
                # Use the proper file_index to clear
                file_index.clear_index()
                
                # Refresh UI
                self.refresh_debug_view()
                self._update_view_files_button_count()
                
                # Close dialog if provided
                if dialog:
                    dialog.close()
                
                self.status_bar.showMessage("Index cleared successfully", 3000)
                logger.info("Index cleared by user")
                
            except Exception as e:
                logger.error(f"Error clearing index: {e}")
                ModernInfoDialog.show_warning(self, "Error", f"Failed to clear index: {e}")
    
    def _offer_reindex_watched_files(self):
        """
        If the auto-watcher is running, offer to re-index existing files in watched folders.
        Called after clearing the index.
        """
        try:
            # Check if organize_page and watcher exist and are running
            if not hasattr(self, 'organize_page'):
                return
            
            organize_page = self.organize_page
            if not organize_page.auto_watcher or not organize_page.auto_watcher.is_running:
                return
            
            # Build per-folder info with file counts
            folder_info = []
            total_files = 0
            
            for folder in organize_page.watch_folders:
                folder_file_count = 0
                folder_subfolder_count = 0
                try:
                    for item in os.listdir(folder):
                        item_path = os.path.join(folder, item)
                        if os.path.isfile(item_path):
                            folder_file_count += 1
                        elif os.path.isdir(item_path) and not item.startswith('.'):
                            folder_subfolder_count += 1
                            # Count files in subfolders too
                            for sub_item in os.listdir(item_path):
                                if os.path.isfile(os.path.join(item_path, sub_item)):
                                    folder_file_count += 1
                except Exception:
                    pass
                
                if folder_file_count > 0 or folder_subfolder_count > 0:
                    folder_info.append({
                        'path': folder,
                        'file_count': folder_file_count,
                        'subfolder_count': folder_subfolder_count
                    })
                    total_files += folder_file_count
            
            # If there are files, show the per-folder dialog
            if total_files > 0 and folder_info:
                from app.ui.organize_page import ApplyInstructionsDialogPerFolder
                from PySide6.QtWidgets import QDialog
                
                dialog = ApplyInstructionsDialogPerFolder(self, folder_info)
                result = dialog.exec()
                
                if result == QDialog.Accepted:
                    # Apply per-folder choices
                    logger.info(f"User chose per-folder options: {dialog.folder_choices}")
                    organize_page.auto_watcher.organize_folders_with_per_folder_options(dialog.folder_choices)
                else:
                    # Skip - just continue watching for new files
                    logger.info("User cancelled re-indexing dialog")
                    
        except Exception as e:
            logger.error(f"Error offering reindex for watched files: {e}")
    
    def _update_view_files_button_count(self):
        """Update the View Files button with the current file count."""
        try:
            from app.core.indexer import FileIndexer
            indexer = FileIndexer()
            files = indexer.get_all_indexed_files()
            count = len(files)
            self.view_files_btn.setText(f"View Indexed Files ({count})")
        except Exception as e:
            logger.error(f"Error updating file count: {e}")
            self.view_files_btn.setText("View Indexed Files (0)")

    def setup_organize_page(self):
        """Setup the AI-powered file organization page."""
        self.organize_page = OrganizePage(self)
        self.page_stack.addWidget(self.organize_page)

    def setup_settings_page(self):
        """Setup the Settings page with AI and app configuration."""
        # Create scroll area for settings
        from PySide6.QtWidgets import QScrollArea
        
        scroll_area = QScrollArea()
        scroll_area.setObjectName("settingsPage")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        settings_widget = QWidget()
        settings_widget.setObjectName("settingsContent")
        layout = QVBoxLayout(settings_widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(16)

        # Style variables for settings (no cascading - applied individually)
        settings_title_style = "font-size: 15px; font-weight: 600; color: #7C4DFF; background: transparent; border: none;"
        settings_label_style = "color: #7A7A90; font-size: 13px; background: transparent; border: none;"
        settings_hint_style = "color: #4A4A5A; font-size: 11px; background: transparent; border: none;"

        # ======= APPEARANCE CARD =======
        appearance_card = QFrame()
        appearance_card.setObjectName("settingsCard")
        appearance_card.setStyleSheet("""
            QFrame#settingsCard {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCard QLabel {
                border: none;
                background: transparent;
            }
        """)
        appearance_layout = QVBoxLayout(appearance_card)
        appearance_layout.setContentsMargins(20, 20, 20, 20)
        appearance_layout.setSpacing(12)
        
        appearance_title = QLabel("🎨 Appearance")
        appearance_title.setStyleSheet(settings_title_style)
        appearance_layout.addWidget(appearance_title)
        
        theme_row = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet(settings_label_style)
        theme_row.addWidget(theme_label)
        self.theme_toggle_btn = QPushButton()
        self.theme_toggle_btn.setCheckable(True)
        self.theme_toggle_btn.setMinimumHeight(36)
        self.theme_toggle_btn.setMinimumWidth(120)
        self.theme_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.theme_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #16161F;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 10px;
                color: #7C4DFF;
                font-weight: 600;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
            QPushButton:checked {
                background-color: #7C4DFF;
                color: white;
            }
        """)
        self._update_theme_button()
        self.theme_toggle_btn.clicked.connect(self._on_theme_toggle)
        theme_row.addWidget(self.theme_toggle_btn)
        theme_row.addStretch()
        appearance_layout.addLayout(theme_row)
        
        layout.addWidget(appearance_card)

        # ======= HELP & ONBOARDING CARD =======
        help_card = QFrame()
        help_card.setObjectName("settingsCardHelp")
        help_card.setStyleSheet("""
            QFrame#settingsCardHelp {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCardHelp > QLabel {
                border: none;
                background: transparent;
            }
        """)
        help_layout = QVBoxLayout(help_card)
        help_layout.setContentsMargins(20, 20, 20, 20)
        help_layout.setSpacing(12)
        
        help_title = QLabel("🎓 Help & Guidance")
        help_title.setStyleSheet(settings_title_style)
        help_layout.addWidget(help_title)
        
        help_desc = QLabel("New to the app? Take a quick tour to learn the basics.")
        help_desc.setStyleSheet(settings_hint_style)
        help_layout.addWidget(help_desc)
        
        show_guide_btn = QPushButton("📖 Show Welcome Guide")
        show_guide_btn.setMinimumHeight(40)
        show_guide_btn.setCursor(Qt.PointingHandCursor)
        show_guide_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 10px;
                color: #7C4DFF;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
        """)
        show_guide_btn.clicked.connect(self._show_onboarding)
        help_layout.addWidget(show_guide_btn)
        
        layout.addWidget(help_card)

        # ======= SUPPORT CARD =======
        support_card = QFrame()
        support_card.setObjectName("settingsCardSupport")
        support_card.setStyleSheet("""
            QFrame#settingsCardSupport {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCardSupport > QLabel {
                border: none;
                background: transparent;
            }
        """)
        support_layout = QVBoxLayout(support_card)
        support_layout.setContentsMargins(20, 20, 20, 20)
        support_layout.setSpacing(12)

        support_title = QLabel("💬 Support")
        support_title.setStyleSheet(settings_title_style)
        support_layout.addWidget(support_title)

        support_desc = QLabel("Experiencing an issue? Reach out and we'll help you out.")
        support_desc.setStyleSheet(settings_hint_style)
        support_desc.setWordWrap(True)
        support_layout.addWidget(support_desc)

        contact_btn = QPushButton("✉️ Contact Support")
        contact_btn.setMinimumHeight(40)
        contact_btn.setCursor(Qt.PointingHandCursor)
        contact_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 10px;
                color: #7C4DFF;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
        """)
        contact_btn.clicked.connect(lambda: __import__('webbrowser').open('mailto:softwaregentofficial@gmail.com?subject=Filect Support'))
        support_layout.addWidget(contact_btn)

        layout.addWidget(support_card)

        # AI Providers section removed - app covers AI costs for users
        # (Hidden placeholder widgets to prevent AttributeError in event handlers)
        self.local_ai_group = QFrame()
        self.local_ai_group.setVisible(False)
        self.openai_group = QFrame()
        self.openai_group.setVisible(False)

        # ======= QUICK SEARCH CARD =======
        qs_card = QFrame()
        qs_card.setObjectName("settingsCardQS")
        qs_card.setStyleSheet("""
            QFrame#settingsCardQS {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCardQS > QLabel {
                border: none;
                background: transparent;
            }
        """)
        qs_layout = QVBoxLayout(qs_card)
        qs_layout.setContentsMargins(20, 20, 20, 20)
        qs_layout.setSpacing(12)
        
        qs_title = QLabel("🔍 Quick Search")
        qs_title.setStyleSheet(settings_title_style)
        qs_layout.addWidget(qs_title)
        
        toggle_btn_style = """
            QPushButton {
                background-color: #16161F;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 10px;
                color: #7C4DFF;
                font-weight: 600;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
            QPushButton:checked {
                background-color: #7C4DFF;
                color: white;
                border-color: #7C4DFF;
            }
        """

        qs_row1 = QHBoxLayout()
        self.qs_autopaste_btn = QPushButton("Auto-Paste: ON" if settings.quick_search_autopaste else "Auto-Paste: OFF")
        self.qs_autopaste_btn.setCheckable(True)
        self.qs_autopaste_btn.setChecked(settings.quick_search_autopaste)
        self.qs_autopaste_btn.setMinimumHeight(36)
        self.qs_autopaste_btn.setMinimumWidth(140)
        self.qs_autopaste_btn.setCursor(Qt.PointingHandCursor)
        self.qs_autopaste_btn.setStyleSheet(toggle_btn_style)
        qs_row1.addWidget(self.qs_autopaste_btn)
        
        self.qs_autoconfirm_btn = QPushButton("Auto-Confirm: ON" if settings.quick_search_auto_confirm else "Auto-Confirm: OFF")
        self.qs_autoconfirm_btn.setCheckable(True)
        self.qs_autoconfirm_btn.setChecked(settings.quick_search_auto_confirm)
        self.qs_autoconfirm_btn.setMinimumHeight(36)
        self.qs_autoconfirm_btn.setMinimumWidth(150)
        self.qs_autoconfirm_btn.setCursor(Qt.PointingHandCursor)
        self.qs_autoconfirm_btn.setStyleSheet(toggle_btn_style)
        qs_row1.addWidget(self.qs_autoconfirm_btn)
        qs_row1.addStretch()
        qs_layout.addLayout(qs_row1)

        qs_row2 = QHBoxLayout()
        qs_row2.setSpacing(10)
        shortcut_label = QLabel("Shortcut:")
        shortcut_label.setStyleSheet(settings_label_style)
        qs_row2.addWidget(shortcut_label)
        self.qs_shortcut_input = QLineEdit(settings.quick_search_shortcut)
        self.qs_shortcut_input.setMinimumHeight(36)
        self.qs_shortcut_input.setMaximumWidth(180)
        self.qs_shortcut_input.setStyleSheet("""
            QLineEdit {
                background-color: #0F0F1A;
                border: 1px solid #1C1C28;
                border-radius: 8px;
                padding: 6px 12px;
                color: #E8E8F0;
            }
            QLineEdit:focus {
                border-color: #7C4DFF;
                background-color: #12121E;
            }
        """)
        qs_row2.addWidget(self.qs_shortcut_input)
        self.qs_shortcut_save = QPushButton("Save")
        self.qs_shortcut_save.setMinimumHeight(36)
        self.qs_shortcut_save.setMinimumWidth(80)
        self.qs_shortcut_save.setCursor(Qt.PointingHandCursor)
        self.qs_shortcut_save.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        qs_row2.addWidget(self.qs_shortcut_save)
        qs_row2.addStretch()
        qs_layout.addLayout(qs_row2)

        layout.addWidget(qs_card)
        
        # ======= SEARCH ENHANCEMENTS CARD =======
        search_card = QFrame()
        search_card.setObjectName("settingsCardSearch")
        search_card.setStyleSheet("""
            QFrame#settingsCardSearch {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCardSearch > QLabel {
                border: none;
                background: transparent;
            }
        """)
        search_layout = QVBoxLayout(search_card)
        search_layout.setContentsMargins(20, 20, 20, 20)
        search_layout.setSpacing(12)
        
        search_title = QLabel("✨ Search Enhancements")
        search_title.setStyleSheet(settings_title_style)
        search_layout.addWidget(search_title)
        
        # Smart Rerank toggle
        gpt_row = QHBoxLayout()
        self.gpt_rerank_button = QPushButton(
            "Smart Rerank: ON" if settings.use_openai_search_rerank else "Smart Rerank: OFF"
        )
        self.gpt_rerank_button.setCheckable(True)
        self.gpt_rerank_button.setChecked(settings.use_openai_search_rerank)
        self.gpt_rerank_button.setMinimumHeight(36)
        self.gpt_rerank_button.setMinimumWidth(160)
        self.gpt_rerank_button.setCursor(Qt.PointingHandCursor)
        self.gpt_rerank_button.setStyleSheet(toggle_btn_style)
        self.gpt_rerank_button.setToolTip("Uses GPT to re-rank search results for better relevance")
        gpt_row.addWidget(self.gpt_rerank_button)
        
        # Spell Check toggle
        self.spell_check_btn = QPushButton(
            "Spell Check: ON" if settings.enable_spell_check else "Spell Check: OFF"
        )
        self.spell_check_btn.setCheckable(True)
        self.spell_check_btn.setChecked(settings.enable_spell_check)
        self.spell_check_btn.setMinimumHeight(36)
        self.spell_check_btn.setMinimumWidth(150)
        self.spell_check_btn.setCursor(Qt.PointingHandCursor)
        self.spell_check_btn.setStyleSheet(toggle_btn_style)
        self.spell_check_btn.setToolTip("Fixes typos automatically in search queries")
        self.spell_check_btn.clicked.connect(self.on_spell_check_toggle)
        gpt_row.addWidget(self.spell_check_btn)
        gpt_row.addStretch()
        search_layout.addLayout(gpt_row)
        
        search_info = QLabel("💡 Smart Rerank uses AI. Spell Check fixes typos in queries.")
        search_info.setStyleSheet(settings_hint_style)
        search_layout.addWidget(search_info)
        
        layout.addWidget(search_card)
        
        # ======= ACCOUNT CARD =======
        account_card = QFrame()
        account_card.setObjectName("settingsCardAccount")
        account_card.setStyleSheet("""
            QFrame#settingsCardAccount {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCardAccount > QLabel {
                border: none;
                background: transparent;
            }
        """)
        account_layout = QVBoxLayout(account_card)
        account_layout.setContentsMargins(20, 20, 20, 20)
        account_layout.setSpacing(12)
        
        account_title = QLabel("👤 Account")
        account_title.setStyleSheet(settings_title_style)
        account_layout.addWidget(account_title)
        
        # Email display
        email_row = QHBoxLayout()
        email_label = QLabel("Email:")
        email_label.setStyleSheet(settings_label_style)
        email_row.addWidget(email_label)
        self.account_email_label = QLabel("Not logged in")
        self.account_email_label.setStyleSheet("color: #7C4DFF; font-weight: 500; font-size: 13px; background: transparent;")
        email_row.addWidget(self.account_email_label)
        email_row.addStretch()
        account_layout.addLayout(email_row)
        
        # Subscription status
        sub_row = QHBoxLayout()
        sub_label = QLabel("Subscription:")
        sub_label.setStyleSheet(settings_label_style)
        sub_row.addWidget(sub_label)
        self.account_sub_label = QLabel("No subscription")
        self.account_sub_label.setStyleSheet("color: #7A7A90; font-size: 13px; background: transparent;")
        sub_row.addWidget(self.account_sub_label)
        sub_row.addStretch()
        account_layout.addLayout(sub_row)
        
        # Buttons row
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.manage_sub_btn = QPushButton("Manage Subscription")
        self.manage_sub_btn.setMinimumHeight(36)
        self.manage_sub_btn.setMinimumWidth(150)
        self.manage_sub_btn.setCursor(Qt.PointingHandCursor)
        self.manage_sub_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 8px;
                color: #7C4DFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
        """)
        self.manage_sub_btn.clicked.connect(self._open_billing_portal)
        button_row.addWidget(self.manage_sub_btn)
        
        self.signout_btn = QPushButton("Sign Out")
        self.signout_btn.setMinimumHeight(36)
        self.signout_btn.setMinimumWidth(100)
        self.signout_btn.setCursor(Qt.PointingHandCursor)
        self.signout_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #252535;
                border-radius: 8px;
                color: #7A7A90;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(211, 47, 47, 0.08);
                border-color: #D32F2F;
                color: #FF6B6B;
            }
        """)
        self.signout_btn.clicked.connect(self._sign_out)
        button_row.addWidget(self.signout_btn)
        button_row.addStretch()
        account_layout.addLayout(button_row)
        
        layout.addWidget(account_card)
        
        # ======= EXCLUSIONS SECTION (Collapsible) =======
        exclusions_container = QFrame()
        exclusions_container.setObjectName("settingsCardExclusions")
        exclusions_container.setStyleSheet("""
            QFrame#settingsCardExclusions {
                background-color: #111119;
                border: 1px solid #1C1C28;
                border-radius: 16px;
            }
            QFrame#settingsCardExclusions > QLabel {
                border: none;
                background: transparent;
            }
        """)
        exclusions_container_layout = QVBoxLayout(exclusions_container)
        exclusions_container_layout.setContentsMargins(0, 0, 0, 0)
        exclusions_container_layout.setSpacing(0)
        
        # Collapsible header button
        self.exclusions_toggle_btn = QPushButton("▶ 🛡️ Exclusions (Advanced)")
        self.exclusions_toggle_btn.setCheckable(True)
        self.exclusions_toggle_btn.setChecked(False)
        self.exclusions_toggle_btn.setMinimumHeight(50)
        self.exclusions_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.exclusions_toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 20px;
                font-size: 15px;
                font-weight: 600;
                background-color: transparent;
                border: none;
                border-radius: 20px;
                color: #7C4DFF;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
            }
        """)
        self.exclusions_toggle_btn.clicked.connect(self._toggle_exclusions_section)
        exclusions_container_layout.addWidget(self.exclusions_toggle_btn)
        
        # Collapsible content
        self.exclusions_content = QWidget()
        self.exclusions_content.setVisible(False)
        exclusions_layout = QVBoxLayout(self.exclusions_content)
        exclusions_layout.setContentsMargins(20, 0, 20, 20)
        exclusions_layout.setSpacing(12)
        
        # Description
        exclusions_desc = QLabel("Files and folders matching these patterns will be skipped during organization.")
        exclusions_desc.setStyleSheet(settings_hint_style)
        exclusions_desc.setWordWrap(True)
        exclusions_layout.addWidget(exclusions_desc)
        
        # List widget for patterns
        self.exclusions_list = QListWidget()
        self.exclusions_list.setMinimumHeight(150)
        self.exclusions_list.setMaximumHeight(200)
        self.exclusions_list.setStyleSheet("""
            QListWidget {
                background-color: #0F0F1A;
                border: 1px solid #1C1C28;
                border-radius: 10px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-radius: 6px;
                color: #B0B0C0;
            }
            QListWidget::item:hover {
                background-color: #16161F;
            }
            QListWidget::item:selected {
                background-color: rgba(124, 77, 255, 0.12);
                color: #B39DFF;
            }
        """)
        self._refresh_exclusions_list()
        exclusions_layout.addWidget(self.exclusions_list)
        
        # Add/Remove row
        exclusions_btn_row = QHBoxLayout()
        exclusions_btn_row.setSpacing(8)
        
        self.exclusion_input = QLineEdit()
        self.exclusion_input.setPlaceholderText("Enter pattern (e.g., node_modules, *.tmp, .env)")
        self.exclusion_input.setMinimumHeight(36)
        self.exclusion_input.setStyleSheet("""
            QLineEdit {
                background-color: #0F0F1A;
                border: 1px solid #1C1C28;
                border-radius: 8px;
                padding: 6px 12px;
                color: #E8E8F0;
            }
            QLineEdit:focus {
                border-color: #7C4DFF;
                background-color: #12121E;
            }
        """)
        self.exclusion_input.returnPressed.connect(self._add_exclusion_pattern)
        exclusions_btn_row.addWidget(self.exclusion_input, 1)
        
        add_exclusion_btn = QPushButton("+ Add")
        add_exclusion_btn.setMinimumHeight(36)
        add_exclusion_btn.setMinimumWidth(70)
        add_exclusion_btn.setCursor(Qt.PointingHandCursor)
        add_exclusion_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C4DFF;
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #9575FF;
            }
        """)
        add_exclusion_btn.clicked.connect(self._add_exclusion_pattern)
        exclusions_btn_row.addWidget(add_exclusion_btn)
        
        self._remove_exclusion_btn = QPushButton("Remove")
        self._remove_exclusion_btn.setMinimumHeight(36)
        self._remove_exclusion_btn.setMinimumWidth(80)
        self._remove_exclusion_btn.setCursor(Qt.PointingHandCursor)
        self._remove_exclusion_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #252535;
                border-radius: 8px;
                color: #7A7A90;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(211, 47, 47, 0.08);
                border-color: #D32F2F;
                color: #FF6B6B;
            }
        """)
        self._remove_exclusion_btn.clicked.connect(self._remove_exclusion_pattern)
        exclusions_btn_row.addWidget(self._remove_exclusion_btn)
        
        reset_exclusions_btn = QPushButton("Reset")
        reset_exclusions_btn.setMinimumHeight(36)
        reset_exclusions_btn.setMinimumWidth(80)
        reset_exclusions_btn.setCursor(Qt.PointingHandCursor)
        reset_exclusions_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 8px;
                color: #7C4DFF;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }
        """)
        reset_exclusions_btn.clicked.connect(self._reset_exclusions)
        exclusions_btn_row.addWidget(reset_exclusions_btn)
        
        exclusions_layout.addLayout(exclusions_btn_row)
        
        # Help text
        exclusions_help = QLabel("💡 Use * as wildcard. Examples: *.log, backup_*, .env.*")
        exclusions_help.setStyleSheet(settings_hint_style)
        exclusions_layout.addWidget(exclusions_help)
        
        exclusions_container_layout.addWidget(self.exclusions_content)
        layout.addWidget(exclusions_container)
        
        # Load account info on startup
        self._refresh_account_info()
        
        layout.addStretch()

        # Set widget in scroll area and add to stack
        scroll_area.setWidget(settings_widget)
        self.page_stack.addWidget(scroll_area)
        
        # Apply correct theme styles on startup
        self._apply_settings_theme_styles(theme_manager.current_theme)
    
    def _update_theme_button(self):
        """Update the theme toggle button text and state."""
        current = theme_manager.current_theme
        if current == 'dark':
            self.theme_toggle_btn.setText("🌙 Dark Mode")
            self.theme_toggle_btn.setChecked(True)
        else:
            self.theme_toggle_btn.setText("☀️ Light Mode")
            self.theme_toggle_btn.setChecked(False)
    
    def _on_theme_toggle(self):
        """Handle theme toggle button click."""
        new_theme = theme_manager.toggle_theme()
        self._update_theme_button()
        # Re-apply inline styles for the new theme
        self._apply_settings_theme_styles(new_theme)
        if hasattr(self, 'organize_page') and hasattr(self.organize_page, '_apply_theme_styles'):
            self.organize_page._apply_theme_styles(new_theme)
        self.status_bar.showMessage(f"Switched to {new_theme} mode", 3000)
    
    def _apply_settings_theme_styles(self, theme=None):
        """Re-apply all theme-dependent inline styles on the settings page."""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors(theme)

        # Style variables
        settings_label_style = f"color: {c['text_muted']}; font-size: 13px; background: transparent; border: none;"
        settings_hint_style = f"color: {c['text_disabled']}; font-size: 11px; background: transparent; border: none;"

        # ---- Card backgrounds (find by object name) ----
        from PySide6.QtWidgets import QFrame
        card_names = ['settingsCard', 'settingsCardHelp', 'settingsCardQS',
                      'settingsCardSearch', 'settingsCardAccount', 'settingsCardExclusions']
        for name in card_names:
            card = self.findChild(QFrame, name)
            if card:
                card.setStyleSheet(f"""
                    QFrame#{name} {{
                        background-color: {c['surface']};
                        border: 1px solid {c['border']};
                        border-radius: 16px;
                    }}
                    QFrame#{name} QLabel {{
                        border: none;
                        background: transparent;
                    }}
                """)

        # ---- Toggle buttons ----
        toggle_btn_style = f"""
            QPushButton {{
                background-color: {c['card']};
                border: 1px solid rgba(124, 77, 255, 0.30);
                border-radius: 10px;
                color: #7C4DFF;
                font-weight: 600;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: rgba(124, 77, 255, 0.08);
                border-color: #7C4DFF;
            }}
            QPushButton:checked {{
                background-color: #7C4DFF;
                color: white;
                border-color: #7C4DFF;
            }}
        """
        for btn in [self.theme_toggle_btn, self.qs_autopaste_btn, self.qs_autoconfirm_btn,
                    self.gpt_rerank_button, self.spell_check_btn]:
            btn.setStyleSheet(toggle_btn_style)

        # ---- Input fields ----
        input_style = f"""
            QLineEdit {{
                background-color: {c['bg']};
                border: 1px solid {c['border']};
                border-radius: 8px;
                padding: 6px 12px;
                color: {c['text']};
            }}
            QLineEdit:focus {{
                border-color: #7C4DFF;
                background-color: {c['card']};
            }}
        """
        self.qs_shortcut_input.setStyleSheet(input_style)
        self.exclusion_input.setStyleSheet(input_style)

        # ---- Exclusions list ----
        self.exclusions_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {c['bg']};
                border: 1px solid {c['border']};
                border-radius: 10px;
                padding: 6px;
            }}
            QListWidget::item {{
                padding: 8px 12px;
                border-radius: 6px;
                color: {c['text_secondary']};
            }}
            QListWidget::item:hover {{
                background-color: {c['card']};
            }}
            QListWidget::item:selected {{
                background-color: rgba(124, 77, 255, 0.12);
                color: #B39DFF;
            }}
        """)

        # ---- Sign out button ----
        self.signout_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {c['border_strong']};
                border-radius: 8px;
                color: {c['text_muted']};
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: rgba(211, 47, 47, 0.08);
                border-color: #D32F2F;
                color: #FF6B6B;
            }}
        """)

        # ---- Remove exclusion button ----
        self._remove_exclusion_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {c['border_strong']};
                border-radius: 8px;
                color: {c['text_muted']};
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: rgba(211, 47, 47, 0.08);
                border-color: #D32F2F;
                color: #FF6B6B;
            }}
        """)

        # ---- Labels inside cards ----
        # Update labels that use muted/hint colors
        # Find labels by checking their current style properties
        settings_page_idx = 3  # settings is the 4th page in page_stack
        if self.page_stack.count() > settings_page_idx:
            settings_page = self.page_stack.widget(settings_page_idx)
            if settings_page:
                for label in settings_page.findChildren(QLabel):
                    style = label.styleSheet()
                    if not style:
                        continue
                    # Update muted color labels (font-size: 13px)
                    if 'font-size: 13px' in style and ('color: #7A7A90' in style or 'color: #888888' in style):
                        label.setStyleSheet(settings_label_style)
                    # Update hint color labels (font-size: 11px)
                    elif 'font-size: 11px' in style and ('color: #4A4A5A' in style or 'color: #999999' in style):
                        label.setStyleSheet(settings_hint_style)
    
    def _refresh_account_info(self):
        """Refresh and display account information in sidebar and settings."""
        try:
            logger.info(f"[ACCOUNT] Refreshing account info. is_authenticated={supabase_auth.is_authenticated}")
            
            if supabase_auth.is_authenticated:
                # Display email
                email = supabase_auth.user_email or "Unknown"
                logger.info(f"[ACCOUNT] User email: {email}")
                self.account_email_label.setText(email)
                
                # Update sidebar
                name = email.split('@')[0].title() if '@' in email else email
                self.account_name_label.setText(name)
                initials = name[:2].upper() if name else "?"
                self.avatar_label.setText(initials)
                
                # Check subscription status
                result = supabase_auth.check_subscription()
                logger.info(f"[ACCOUNT] Subscription check result: {result}")
                
                if result.get('has_subscription'):
                    status = result.get('status', 'active')
                    expires_at = result.get('expires_at', 'N/A')
                    if expires_at and expires_at != 'N/A':
                        try:
                            # Format the date nicely
                            dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                            expires_str = dt.strftime('%Y-%m-%d')
                            self.account_sub_label.setText(f"✓ Active (until {expires_str})")
                        except Exception:
                            self.account_sub_label.setText(f"✓ Active ({status})")
                    else:
                        self.account_sub_label.setText(f"✓ Active ({status})")
                    self.account_sub_label.setStyleSheet("color: #7C4DFF;")
                    self.account_plan_label.setText("Pro Plan ✓")
                    self.account_plan_label.setStyleSheet("color: #7C4DFF; font-weight: 500;")
                else:
                    status = result.get('status')
                    if status:
                        self.account_sub_label.setText(f"⚠ {status}")
                    else:
                        self.account_sub_label.setText("No subscription")
                    self.account_sub_label.setStyleSheet("color: #FF6B6B;")
                    self.account_plan_label.setText("Free Plan")
                    self.account_plan_label.setStyleSheet("")
            else:
                logger.info("[ACCOUNT] User not authenticated")
                self.account_email_label.setText("Not logged in")
                self.account_sub_label.setText("No subscription")
                self.account_sub_label.setStyleSheet("")
                # Update sidebar
                self.account_name_label.setText("Guest")
                self.account_plan_label.setText("Sign in to continue")
                self.account_plan_label.setStyleSheet("color: #7A7A90;")
                self.avatar_label.setText("?")
        except Exception as e:
            logger.error(f"[ACCOUNT] Error refreshing account info: {e}")
        
        # Always update usage labels when account info is refreshed
        # This ensures the correct user's usage is displayed after login/logout
        try:
            self._update_usage_labels()
        except Exception as e:
            logger.debug(f"Could not update usage labels: {e}")
    
    def _open_billing_portal(self):
        """Open Stripe billing portal for subscription management."""
        import webbrowser
        from urllib.parse import quote
        from app.core.supabase_client import supabase_auth, SUPABASE_URL
        from app.ui.organize_page import ModernInfoDialog
        
        if not supabase_auth.is_authenticated:
            ModernInfoDialog.show_warning(self, "Not Logged In", "Please sign in to manage your subscription.")
            return
        
        # Get user info from the current_user dict
        current_user = supabase_auth.current_user
        if not current_user:
            ModernInfoDialog.show_warning(self, "Error", "Could not retrieve account info. Please try signing out and back in.")
            return
        
        user_id = current_user.get('id', '')
        email = current_user.get('email', '')
        
        if not email or not user_id:
            ModernInfoDialog.show_warning(self, "Error", "Could not retrieve account email. Please try signing out and back in.")
            return
        
        # Open billing portal via Supabase Edge Function
        portal_url = f"{SUPABASE_URL}/functions/v1/create-portal-session?user_id={user_id}&email={quote(email)}"
        
        try:
            webbrowser.open(portal_url)
            self.status_bar.showMessage("Opening subscription management...", 3000)
        except Exception as e:
            logger.error(f"Error opening billing portal: {e}")
            ModernInfoDialog.show_warning(self, "Error", "Could not open billing portal. Please try again.")
    
    def _rebuild_fts_index(self):
        """Rebuild the FTS search index to fix corruption."""
        from app.ui.organize_page import ModernConfirmDialog, ModernInfoDialog
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Rebuild Search Index",
            message="This will rebuild the full-text search index.",
            info_text="Your indexed files and metadata will be preserved. This may take a moment for large databases.",
            yes_text="Rebuild",
            no_text="Cancel"
        )
        
        if not confirmed:
            return
        
        try:
            self.rebuild_fts_btn.setEnabled(False)
            self.rebuild_fts_btn.setText("Rebuilding...")
            QApplication.processEvents()
            
            # Run rebuild with progress
            def progress_callback(current, total):
                self.status_bar.showMessage(f"Rebuilding search index: {current}/{total}")
                QApplication.processEvents()
            
            stats = file_index.rebuild_fts_index(progress_callback)
            
            self.rebuild_fts_btn.setEnabled(True)
            self.rebuild_fts_btn.setText("Rebuild Search Index")
            
            if stats['errors'] == 0:
                ModernInfoDialog.show_info(
                    self,
                    title="Success",
                    message="Search index rebuilt successfully!",
                    details=[f"Indexed: {stats['indexed']} files"]
                )
                self.status_bar.showMessage("Search index rebuilt successfully", 5000)
            else:
                ModernInfoDialog.show_warning(
                    self,
                    title="Partial Success",
                    message="Search index rebuilt with some errors.",
                    details=[
                        f"Indexed: {stats['indexed']} files",
                        f"Errors: {stats['errors']}"
                    ]
                )
        except Exception as e:
            self.rebuild_fts_btn.setEnabled(True)
            self.rebuild_fts_btn.setText("Rebuild Search Index")
            logger.error(f"Error rebuilding FTS index: {e}")
            ModernInfoDialog.show_warning(
                self,
                title="Error",
                message=f"Failed to rebuild search index:\n\n{e}"
            )
    
    def _refresh_exclusions_list(self):
        """Refresh the exclusions list widget with current patterns."""
        self.exclusions_list.clear()
        for pattern in settings.exclusion_patterns:
            self.exclusions_list.addItem(pattern)
    
    def _add_exclusion_pattern(self):
        """Add a new exclusion pattern."""
        pattern = self.exclusion_input.text().strip()
        if pattern:
            settings.add_exclusion_pattern(pattern)
            self._refresh_exclusions_list()
            self.exclusion_input.clear()
            self.status_bar.showMessage(f"Added exclusion: {pattern}", 3000)
    
    def _remove_exclusion_pattern(self):
        """Remove the selected exclusion pattern."""
        current_item = self.exclusions_list.currentItem()
        if current_item:
            pattern = current_item.text()
            settings.remove_exclusion_pattern(pattern)
            self._refresh_exclusions_list()
            self.status_bar.showMessage(f"Removed exclusion: {pattern}", 3000)
    
    def _reset_exclusions(self):
        """Reset exclusions to default patterns."""
        from app.ui.organize_page import ModernConfirmDialog
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Reset Exclusions",
            message="Reset all exclusion patterns to defaults?",
            info_text="This will remove any custom patterns you've added.",
            yes_text="Reset",
            no_text="Cancel"
        )
        if confirmed:
            settings.reset_exclusions_to_defaults()
            self._refresh_exclusions_list()
            self.status_bar.showMessage("Exclusions reset to defaults", 3000)
    
    def _toggle_exclusions_section(self):
        """Toggle the collapsible exclusions section."""
        is_expanded = self.exclusions_toggle_btn.isChecked()
        self.exclusions_content.setVisible(is_expanded)
        if is_expanded:
            self.exclusions_toggle_btn.setText("▼ 🛡️ Exclusions (Advanced)")
        else:
            self.exclusions_toggle_btn.setText("▶ 🛡️ Exclusions (Advanced)")
    
    def _sign_out(self):
        """Sign out the current user and show login dialog."""
        from app.ui.organize_page import ModernConfirmDialog
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Sign Out",
            message="Are you sure you want to sign out?",
            info_text="You'll need to sign in again to use the app.",
            yes_text="Sign Out",
            no_text="Cancel"
        )
        
        if confirmed:
            # Sign out from Supabase
            supabase_auth.sign_out()
            settings.clear_auth_tokens()
            
            # Hide main window
            self.hide()
            
            # Show auth dialog again
            from app.ui.auth_dialog import AuthDialog
            auth_dialog = AuthDialog()
            
            if auth_dialog.exec():
                # User logged in successfully, refresh account info and show window
                self._refresh_account_info()
                self._update_usage_labels()  # Update usage for new user
                self.show()
                self.status_bar.showMessage("Welcome back!", 3000)
            else:
                # User cancelled login, close app
                QApplication.quit()
    
    def _resync_file_dates(self):
        """Resync file dates from Windows filesystem."""
        from app.ui.organize_page import ModernConfirmDialog, ModernInfoDialog
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Resync File Dates",
            message="Re-read file creation and modification dates from Windows for all indexed files?",
            info_text="This may take a moment for large databases.",
            yes_text="Resync",
            no_text="Cancel"
        )
        
        if not confirmed:
            return
        
        self.resync_dates_btn.setEnabled(False)
        self.resync_status_label.setText("Resyncing...")
        QApplication.processEvents()
        
        try:
            # Perform the resync
            from app.core.database import file_index
            
            def progress_callback(current, total):
                self.resync_status_label.setText(f"Processing {current}/{total}...")
                QApplication.processEvents()
            
            stats = file_index.resync_file_dates(progress_callback)
            
            # Show result
            self.resync_status_label.setText(
                f"Done: {stats['updated']} updated, {stats['not_found']} not found"
            )
            
            metadata_count = stats.get('exif_found', 0)
            ModernInfoDialog.show_info(
                self,
                title="Resync Complete",
                message="File dates resynced successfully!",
                details=[
                    f"Updated: {stats['updated']} files",
                    f"Metadata dates extracted: {metadata_count} files",
                    f"Not found: {stats['not_found']} files",
                    f"Errors: {stats['errors']} files"
                ],
                info_text="Date filters now use the best available date for each file type."
            )
            
        except Exception as e:
            logger.error(f"Error resyncing file dates: {e}")
            self.resync_status_label.setText("Error!")
            ModernInfoDialog.show_warning(self, "Error", f"Failed to resync file dates:\n{e}")
        
        finally:
            self.resync_dates_btn.setEnabled(True)
    
    def setup_connections(self):
        """Setup signal connections."""
        # Organize tab connections - Hidden for MVP (search-only mode)
        # self.source_button.clicked.connect(self.select_source_folder)
        # self.dest_button.clicked.connect(self.select_destination_folder)
        # self.scan_button.clicked.connect(self.scan_and_plan)
        # self.apply_button.clicked.connect(self.apply_moves)
        
        # Search tab connections
        self.index_button.clicked.connect(self.select_index_folder)
        self.index_button_action.clicked.connect(self.index_directory)
        self.index_pause_btn.clicked.connect(self._toggle_index_pause)
        self.index_cancel_btn.clicked.connect(self._cancel_indexing)
        self.search_button.clicked.connect(self.search_files)
        self.search_input.returnPressed.connect(self.search_files)
        
        # Filter connections
        self.type_filter.currentIndexChanged.connect(self._on_filter_changed)
        self.date_filter.currentIndexChanged.connect(self._on_filter_changed)
        self.clear_filters_btn.clicked.connect(self._clear_filters)
        
        # Quick Actions connections (itemChanged fires when checkbox is clicked)
        self.search_results_table.itemChanged.connect(self._on_selection_changed)
        self.search_results_table.itemChanged.connect(self.on_search_cell_changed)
        self.action_remove_btn.clicked.connect(self._action_remove_from_index)
        self.action_reindex_btn.clicked.connect(self._action_reindex_selected)
        self.action_add_tags_btn.clicked.connect(self._action_add_tags)
        self.action_copy_paths_btn.clicked.connect(self._action_copy_paths)
        self.action_open_folders_btn.clicked.connect(self._action_open_folders)
        self.action_export_btn.clicked.connect(self._action_export_list)
        self.action_select_all_btn.clicked.connect(self._action_select_all)
        self.action_clear_selection_btn.clicked.connect(self._action_clear_selection)
        
        # Debug tab Quick Actions connections (itemChanged fires when checkbox is clicked)
        self.debug_table.itemChanged.connect(self._on_debug_selection_changed)
        self.debug_action_remove_btn.clicked.connect(lambda: self._action_remove_from_index(source='debug'))
        self.debug_action_reindex_btn.clicked.connect(lambda: self._action_reindex_selected(source='debug'))
        self.debug_action_add_tags_btn.clicked.connect(lambda: self._action_add_tags(source='debug'))
        self.debug_action_copy_paths_btn.clicked.connect(lambda: self._action_copy_paths(source='debug'))
        self.debug_action_open_folders_btn.clicked.connect(lambda: self._action_open_folders(source='debug'))
        self.debug_action_export_btn.clicked.connect(lambda: self._action_export_list(source='debug'))
        self.debug_action_select_all_btn.clicked.connect(lambda: self._action_select_all(source='debug'))
        self.debug_action_clear_selection_btn.clicked.connect(lambda: self._action_clear_selection(source='debug'))
        
        # Quick index options (legacy stubs)
        self.index_pc_button.clicked.connect(self.on_index_entire_pc)
        self.auto_index_downloads_btn.toggled.connect(self.on_toggle_auto_index_downloads)
        
        # New indexing options
        self.more_options_header.clicked.connect(self._toggle_more_options)
        self.index_pc_now_btn.clicked.connect(self.on_index_entire_pc)
        self.watch_header_btn.clicked.connect(self._toggle_watch_options)
        self.watch_common_toggle.toggled.connect(self._on_watch_common_toggled)
        self.watch_custom_toggle.toggled.connect(self._on_watch_custom_toggled)
        self.add_custom_folder_btn.clicked.connect(self._on_add_custom_folder)
        
        # Initialize custom folders list
        self._refresh_custom_folders_list()
        self._update_watch_status()
        
        # Start watching if enabled
        if settings.watch_common_folders or settings.watch_custom_folders:
            self._start_folder_watching()
        
        # Index page connections
        self.refresh_debug_button.clicked.connect(self.refresh_debug_view)
        self.clear_index_button.clicked.connect(self.clear_index)
        self.clear_all_paths_btn.clicked.connect(self._clear_all_indexed_paths)
        
        # Settings tab connections
        # AI Provider selection
        if hasattr(self, 'ai_provider_combo'):
            self.ai_provider_combo.currentIndexChanged.connect(self._update_ai_provider_visibility)
        if hasattr(self, 'check_ollama_btn'):
            self.check_ollama_btn.clicked.connect(self._check_ollama_status)
        if hasattr(self, 'save_local_ai_btn'):
            self.save_local_ai_btn.clicked.connect(self.on_save_local_ai)
        if hasattr(self, 'save_ai_settings_button'):
            self.save_ai_settings_button.clicked.connect(self.on_save_openai)
        if hasattr(self, 'delete_ai_key_button'):
            self.delete_ai_key_button.clicked.connect(self.on_delete_openai_key)
        if hasattr(self, 'gpt_rerank_button'):
            self.gpt_rerank_button.toggled.connect(self.on_toggle_gpt_rerank)
        # Quick search settings connections
        if hasattr(self, 'qs_autopaste_btn'):
            self.qs_autopaste_btn.toggled.connect(self.on_qs_autopaste)
        if hasattr(self, 'qs_autoconfirm_btn'):
            self.qs_autoconfirm_btn.toggled.connect(self.on_qs_autoconfirm)
        if hasattr(self, 'qs_shortcut_save'):
            self.qs_shortcut_save.clicked.connect(self.on_qs_save_shortcut)
        
        # Update search button state when text changes
        self.search_input.textChanged.connect(self.update_search_button_state)

    def setup_quick_search(self):
        """Register global hotkey and prepare overlay."""
        # Use None as parent so the popup doesn't bring up the main window
        self.quick_overlay = QuickSearchOverlay(None)
        self.quick_overlay.pathSelected.connect(self.on_quick_path_selected)
        logger.info("[QS] *** Signal connection established: pathSelected -> on_quick_path_selected")

        # Wrapper to show overlay and remember previously focused window
        def show_quick_overlay():
            try:
                self._prev_foreground_hwnd = get_foreground_hwnd()
                # Save mouse position relative to the dialog window
                self._rel_click_point = None
                try:
                    rect = get_window_rect(self._prev_foreground_hwnd)
                    if rect:
                        l, t, r, b = rect
                        # Get current cursor pos
                        pt = ctypes.wintypes.POINT()
                        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                        self._rel_click_point = (pt.x - l, pt.y - t)
                except Exception:
                    self._rel_click_point = None
            except Exception:
                self._prev_foreground_hwnd = 0
                self._rel_click_point = None
            self.quick_overlay.show_centered_bottom()

        # Register global hotkey via QHotkey → keyboard → WinAPI
        self._qhotkey = None
        self._win_hotkey = None
        try:
            from qhotkey import QHotkey  # type: ignore
            ks = settings.quick_search_shortcut or 'ctrl+alt+h'
            self._qhotkey = QHotkey(QKeySequence(ks), True, self)
            self._qhotkey.activated.connect(show_quick_overlay)
            logger.info(f"Registered global hotkey (QHotkey): {ks}")
        except Exception as e:
            logger.warning(f"QHotkey failed: {e}")
            # Skip keyboard library and go directly to WinAPI (more reliable)
            logger.warning("Skipping keyboard library, using WinAPI directly")
            # Use raw WinAPI RegisterHotKey for maximum reliability
            hk = register_global_hotkey(self,
                                        settings.quick_search_shortcut or 'ctrl+alt+h',
                                        lambda: QTimer.singleShot(0, show_quick_overlay))
            if hk:
                self._win_hotkey = hk
                logger.info("Registered global hotkey (WinAPI)")
            else:
                logger.error("Global hotkey not available; quick search disabled")
                # Final fallback: try keyboard library
                try:
                    import keyboard  # type: ignore
                    hotkey = settings.quick_search_shortcut or 'ctrl+alt+h'
                    keyboard.add_hotkey(hotkey, lambda: QTimer.singleShot(0, show_quick_overlay))
                    logger.info(f"Registered global hotkey (keyboard fallback): {hotkey}")
                except Exception as e2:
                    logger.warning(f"Keyboard hook also failed: {e2}")
        # App-focus fallback using QShortcut so it works when the app is focused
        try:
            ks = settings.quick_search_shortcut.replace('ctrl', 'Ctrl').replace('alt', 'Alt').replace('shift', 'Shift')
            self._focus_quick_shortcut = QShortcut(QKeySequence(ks or 'Ctrl+Alt+Space'), self)
            self._focus_quick_shortcut.setContext(Qt.ApplicationShortcut)
            self._focus_quick_shortcut.activated.connect(show_quick_overlay)
        except Exception:
            pass
        # Debug: dump active dialog tree (Ctrl+Alt+D)
        try:
            self._dump_tree_shortcut = QShortcut(QKeySequence('Ctrl+Alt+D'), self)
            self._dump_tree_shortcut.setContext(Qt.ApplicationShortcut)
            self._dump_tree_shortcut.activated.connect(self.dump_active_dialog_tree)
        except Exception:
            pass
        # Debug: comprehensive system state (Ctrl+Alt+S)
        try:
            self._debug_state_shortcut = QShortcut(QKeySequence('Ctrl+Alt+S'), self)
            self._debug_state_shortcut.setContext(Qt.ApplicationShortcut)
            self._debug_state_shortcut.activated.connect(self.debug_comprehensive_state)
        except Exception:
            pass
        # Quick overlay focus-mode toggle (Ctrl+Alt+F)
        try:
            self._focus_mode_shortcut = QShortcut(QKeySequence('Ctrl+Alt+F'), self)
            self._focus_mode_shortcut.setContext(Qt.ApplicationShortcut)
            self._focus_mode_shortcut.activated.connect(self.quick_overlay.enable_focus_mode)
        except Exception:
            pass

    

    def on_quick_path_selected(self, payload: str):
        logger.info(f"[QS] *** on_quick_path_selected CALLED with payload: {payload}")
        
        # payload may be 'path' or 'path||OPEN'
        path = payload
        do_open = False
        if payload.endswith('||OPEN'):
            path = payload[:-6]
            do_open = True
        
        # If opening file, just open it - don't copy to clipboard or autofill
        if do_open:
            logger.info(f"[QS] Opening file: {path}")
            self.open_file_in_os(path)
            # Re-activate popup so it stays focused and on top for rapid multi-file opening
            # User can click outside the popup to focus on opened files
            # Use delayed re-activation to combat apps that aggressively steal focus
            if hasattr(self, 'quick_overlay') and self.quick_overlay:
                overlay = self.quick_overlay
                overlay._allow_reactivation = True  # Enable reactivation for this open
                def reactivate():
                    # Only reactivate if user hasn't clicked outside the popup
                    if overlay.isVisible() and overlay._allow_reactivation:
                        overlay.raise_()
                        overlay.activateWindow()
                # Immediate activation
                overlay.raise_()
                overlay.activateWindow()
                # Delayed re-activation to reclaim focus if an app steals it
                QTimer.singleShot(100, reactivate)
                QTimer.singleShot(300, reactivate)
            return
        
        # Copy to clipboard
        try:
            cb = QApplication.clipboard()
            cb.setText(path)
            self.status_bar.showMessage("Copied path to clipboard")
        except Exception:
            pass
        
        # Auto-fill using our enhanced Phase 1-3 system
        logger.info(f"[QS] Autopaste setting: {settings.quick_search_autopaste}")
        if settings.quick_search_autopaste:
            logger.info("[QS] === STARTING ENHANCED AUTOFILL ===")
            # Use a short delay to let the dialog settle after focus restoration
            def _run_enhanced_autofill(p=path):
                logger.info(f"[QS] Running enhanced autofill for: {p}")
                self.try_autofill_file_dialog(p)
            
            # Short delay since focus restoration already happened in Phase 2
            QTimer.singleShot(200, _run_enhanced_autofill)
        else:
            logger.info("[QS] Autopaste is DISABLED - skipping autofill")

    def try_autofill_file_dialog(self, path: str) -> None:
        """
        Phase 3: Enhanced autofill pipeline with state-aware dialog targeting.
        Uses saved state from quick search overlay if available.
        """
        logger.info("[QS] Phase 3: Starting enhanced autofill pipeline")
        
        # Check if we have saved state from the quick search overlay
        overlay = getattr(self, 'quick_overlay', None)
        if overlay and overlay.has_valid_saved_state():
            logger.info("[QS] Using saved state from quick search overlay")
            success = self._autofill_with_saved_state(path, overlay)
            # IMPORTANT: Don't fall back - saved state method handles everything
            # The old code would run fallback even if path was partially filled, causing double fill
            if success:
                logger.info("[QS] Saved state autofill succeeded")
            else:
                logger.warning("[QS] Saved state autofill failed - NOT running fallback to avoid double fill")
                self.status_bar.showMessage("QuickSearch: Autofill failed - path copied to clipboard")
            return  # Stop here, don't run fallback pipelines
        
        # Only use discovery-based autofill if we have NO saved state
        logger.info("[QS] No saved state - using discovery-based autofill")
        ok = self._autofill_uia_pipeline(path)
        if not ok:
            logger.info("[QS] UIA pipeline failed; trying win32 pipeline")
            self._autofill_win32_pipeline(path)

    def _autofill_with_saved_state(self, path: str, overlay) -> bool:
        """
        Phase 3: Autofill using saved state from the quick search overlay.
        This is more reliable than discovery because we know the exact dialog.
        """
        try:
            logger.info("[QS] Phase 3: Autofill with saved state")
            
            # Get saved state
            hwnd = overlay._saved_window_hwnd
            window_title = overlay._saved_window_title
            window_class = overlay._saved_window_class
            is_verified_dialog = overlay._is_dialog_verified
            
            logger.info(f"[QS] Target dialog: hwnd={hwnd}, title='{window_title}', class='{window_class}', verified={is_verified_dialog}")
            
            # Phase 4: Create debug report before attempting autofill
            from app.ui.win_hotkey import create_autofill_debug_report
            create_autofill_debug_report(hwnd, overlay._saved_cursor_pos, overlay._saved_window_rect, logger, "[QS]")
            
            # Verify the window still exists and is the same dialog
            from app.ui.win_hotkey import window_still_exists, get_window_title, get_window_class
            if not window_still_exists(hwnd):
                logger.warning("[QS] Target dialog no longer exists")
                return False
            
            current_title = get_window_title(hwnd)
            current_class = get_window_class(hwnd)
            
            if current_title != window_title or current_class != window_class:
                logger.warning(f"[QS] Dialog changed: was '{window_title}'/'{window_class}', now '{current_title}'/'{current_class}'")
                return False
            
            logger.info("[QS] Dialog verified, attempting targeted autofill")
            
            # Try multiple autofill strategies with increasing robustness
            # Note: Removed cursor-click strategies (modern_directui, stealth_click_paste) 
            # because clicking at saved cursor position can select wrong files if cursor was over file list
            strategies = [
                ("targeted_uia", self._autofill_targeted_uia),
                ("targeted_win32", self._autofill_targeted_win32),
                ("keyboard_altn", self._autofill_keyboard_altn),  # Uses Alt+N to focus filename field
            ]
            
            for i, (strategy_name, strategy_func) in enumerate(strategies):
                logger.info(f"[QS] === STRATEGY {i+1}/{len(strategies)}: {strategy_name.upper()} ===")
                try:
                    success = strategy_func(path, hwnd, overlay)
                    if success:
                        logger.info(f"[QS] ✅ Strategy {strategy_name} SUCCESS!")
                        self.status_bar.showMessage(f"QuickSearch: Autofilled via {strategy_name}")
                        return True
                    else:
                        logger.warning(f"[QS] ❌ Strategy {strategy_name} failed")
                except Exception as e:
                    logger.error(f"[QS] ❌ Strategy {strategy_name} exception: {e}")
                
                # Brief pause between strategies
                if i < len(strategies) - 1:
                    import time
                    time.sleep(0.2)
            
            logger.error("[QS] ❌ ALL AUTOFILL STRATEGIES FAILED")
            self.status_bar.showMessage("QuickSearch: All autofill methods failed")
            return False
            
        except Exception as e:
            logger.error(f"[QS] Exception in _autofill_with_saved_state: {e}")
            return False

    def _autofill_targeted_uia(self, path: str, hwnd: int, overlay) -> bool:
        """Strategy 1: Targeted UIA autofill using the specific window handle."""
        try:
            import time
            from pywinauto import Application
            
            start_time = time.time()
            logger.info("[QS] UIA Strategy: Starting targeted UIA autofill")
            
            # Connect directly to the specific window
            app = Application(backend="uia").connect(handle=hwnd)
            win = app.window(handle=hwnd)
            
            logger.info("[QS] Connected to target window via UIA")
            
            # Ensure window is focused
            try:
                win.set_focus()
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"[QS] Failed to set focus: {e}")
            
            # Find filename field using multiple strategies
            target = None
            
            # Strategy A: FileNameControlHost (modern dialogs)
            try:
                host = win.child_window(auto_id="FileNameControlHost", control_type="Pane")
                if host.exists():
                    eds = host.descendants(control_type='Edit')
                    if eds:
                        target = eds[0]
                        logger.info("[QS] Found filename field via FileNameControlHost")
            except Exception:
                pass
            
            # Strategy B: By label proximity
            if target is None:
                try:
                    from app.core.vision import FILENAME_LABELS
                    texts = win.descendants(control_type='Text')
                    edits = win.descendants(control_type='Edit')
                    
                    label_rects = []
                    for t in texts:
                        try:
                            name = (t.window_text() or '').strip()
                            if any(name.lower().startswith(lbl.lower().rstrip(':')) for lbl in FILENAME_LABELS):
                                label_rects.append(t.rectangle())
                        except Exception:
                            continue
                    
                    best = None
                    best_dx = 10**9
                    for e in edits:
                        try:
                            er = e.rectangle()
                            for lr in label_rects:
                                if er.left >= lr.right - 4 and (min(er.bottom, lr.bottom) - max(er.top, lr.top)) > 6:
                                    dx = er.left - lr.right
                                    if er.width() > 150 and er.height() < 60 and dx < best_dx:
                                        best = e
                                        best_dx = dx
                        except Exception:
                            continue
                    
                    if best:
                        target = best
                        logger.info("[QS] Found filename field via label proximity")
                except Exception:
                    pass
            
            # Strategy C: Bottom-most edit heuristic
            if target is None:
                try:
                    edits = win.descendants(control_type='Edit')
                    best = None
                    best_y = -1
                    for e in edits:
                        try:
                            rect = e.rectangle()
                            if rect.width() > 150 and rect.height() < 60 and rect.top > best_y:
                                best = e
                                best_y = rect.top
                        except Exception:
                            continue
                    if best:
                        target = best
                        logger.info("[QS] Found filename field via bottom-most heuristic")
                except Exception:
                    pass
            
            if not target:
                logger.warning("[QS] No filename field found in UIA")
                return False
            
            # Insert the path using multiple methods
            success = self._insert_path_uia(target, path, win)
            
            elapsed = time.time() - start_time
            if success:
                logger.info(f"[QS] UIA Strategy: SUCCESS in {elapsed:.2f}s")
            else:
                logger.warning(f"[QS] UIA Strategy: FAILED after {elapsed:.2f}s")
            
            return success
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[QS] UIA Strategy: EXCEPTION after {elapsed:.2f}s: {e}")
            return False
    
    def _autofill_targeted_win32(self, path: str, hwnd: int, overlay) -> bool:
        """Strategy 2: Targeted Win32 autofill using the specific window handle."""
        try:
            import time
            from pywinauto import Application
            
            start_time = time.time()
            logger.info("[QS] Win32 Strategy: Starting targeted Win32 autofill")
            
            # Connect directly to the specific window
            app = Application(backend="win32").connect(handle=hwnd)
            win = app.window(handle=hwnd)
            
            logger.info("[QS] Connected to target window via Win32")
            
            # Ensure window is focused
            try:
                win.set_focus()
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"[QS] Failed to set focus: {e}")
            
            # Find filename field
            target = None
            
            # Strategy A: ComboBoxEx32 with Edit child (common in file dialogs)
            try:
                combo_hosts = win.descendants(class_name='ComboBoxEx32')
                for host in combo_hosts:
                    eds = host.descendants(class_name='Edit')
                    if eds:
                        target = eds[0]
                        logger.info("[QS] Found filename field via ComboBoxEx32")
                        break
            except Exception:
                pass
            
            # Strategy B: Last Edit control (fallback)
            if target is None:
                try:
                    edits = win.descendants(class_name='Edit')
                    if edits:
                        target = edits[-1]
                        logger.info("[QS] Found filename field via last Edit")
                except Exception:
                    pass
            
            if not target:
                logger.warning("[QS] No filename field found in Win32")
                return False
            
            # Insert the path
            success = self._insert_path_win32(target, path, win)
            
            elapsed = time.time() - start_time
            if success:
                logger.info(f"[QS] Win32 Strategy: SUCCESS in {elapsed:.2f}s")
            else:
                logger.warning(f"[QS] Win32 Strategy: FAILED after {elapsed:.2f}s")
            
            return success
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[QS] Win32 Strategy: EXCEPTION after {elapsed:.2f}s: {e}")
            return False
    
    def _autofill_modern_directui(self, path: str, hwnd: int, overlay) -> bool:
        """Strategy 3: Modern DirectUI dialog autofill for Windows 10/11 file pickers."""
        try:
            import time
            from app.ui.win_hotkey import click_at_position, set_foreground_hwnd_robust
            
            start_time = time.time()
            logger.info("[QS] DirectUI Strategy: Starting modern DirectUI autofill")
            
            # Ensure the dialog is focused
            if not set_foreground_hwnd_robust(hwnd):
                logger.warning("[QS] DirectUI: Failed to focus dialog")
                return False
            
            time.sleep(0.3)  # Let focus settle
            
            # For modern DirectUI dialogs, we need to:
            # 1. Click at the saved cursor position (filename field)
            # 2. Use keyboard shortcuts to paste
            
            cursor_pos = overlay._saved_cursor_pos
            if not cursor_pos:
                logger.warning("[QS] DirectUI: No saved cursor position")
                return False
            
            logger.info(f"[QS] DirectUI: Clicking at saved position {cursor_pos}")
            
            # Click at the saved position (should be the filename field)
            if not click_at_position(cursor_pos[0], cursor_pos[1]):
                logger.warning("[QS] DirectUI: Failed to click at saved position")
                return False
            
            time.sleep(0.3)  # Let click register and focus filename field
            
            # Clear any existing text and paste the path
            try:
                cb = QApplication.clipboard()
                cb.setText(path)
                
                import keyboard
                
                # Clear existing text
                keyboard.send('ctrl+a')
                time.sleep(0.1)
                keyboard.send('delete')
                time.sleep(0.1)
                
                # Paste the path
                keyboard.send('ctrl+v')
                time.sleep(0.2)
                
                logger.info("[QS] DirectUI: Path pasted successfully")
                
                # Auto-confirm if enabled
                if settings.quick_search_auto_confirm:
                    time.sleep(0.3)  # Give time for path to register
                    keyboard.send('enter')
                    logger.info("[QS] DirectUI: Auto-confirmed via Enter")
                
                elapsed = time.time() - start_time
                logger.info(f"[QS] DirectUI Strategy: SUCCESS in {elapsed:.2f}s")
                self.status_bar.showMessage("QuickSearch: path filled via DirectUI" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                return True
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[QS] DirectUI Strategy: Failed to paste after {elapsed:.2f}s: {e}")
                return False
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[QS] DirectUI Strategy: EXCEPTION after {elapsed:.2f}s: {e}")
            return False
    
    def _autofill_stealth_click_paste(self, path: str, hwnd: int, overlay) -> bool:
        """Strategy 3: Stealth click at saved cursor position + paste."""
        try:
            import time
            from app.ui.win_hotkey import click_at_position, set_foreground_hwnd_robust
            
            start_time = time.time()
            logger.info("[QS] Stealth Strategy: Starting stealth click + paste")
            
            # Ensure the dialog is focused
            if not set_foreground_hwnd_robust(hwnd):
                logger.warning("[QS] Failed to focus dialog for stealth click")
                return False
            
            time.sleep(0.3)  # Let focus settle
            
            # Get saved cursor position
            cursor_pos = overlay._saved_cursor_pos
            if not cursor_pos:
                logger.warning("[QS] No saved cursor position for stealth click")
                return False
            
            logger.info(f"[QS] Stealth clicking at saved position: {cursor_pos}")
            
            # Click at the saved position (should be the filename field)
            if not click_at_position(cursor_pos[0], cursor_pos[1]):
                logger.warning("[QS] Failed to click at saved position")
                return False
            
            time.sleep(0.2)  # Let click register
            
            # Clear existing text and paste new path
            try:
                cb = QApplication.clipboard()
                cb.setText(path)
                
                import keyboard
                keyboard.send('ctrl+a')  # Select all
                time.sleep(0.05)
                keyboard.send('ctrl+v')  # Paste
                time.sleep(0.1)
                
                logger.info("[QS] Stealth click + paste completed")
                
                # Auto-confirm if enabled
                if settings.quick_search_auto_confirm:
                    time.sleep(0.2)
                    keyboard.send('enter')
                    logger.info("[QS] Auto-confirmed via Enter")
                
                elapsed = time.time() - start_time
                logger.info(f"[QS] Stealth Strategy: SUCCESS in {elapsed:.2f}s")
                self.status_bar.showMessage("QuickSearch: path filled via stealth click" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                return True
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[QS] Stealth Strategy: Failed to paste after {elapsed:.2f}s: {e}")
                return False
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[QS] Stealth Strategy: EXCEPTION after {elapsed:.2f}s: {e}")
            return False

    def _autofill_keyboard_altn(self, path: str, hwnd: int, overlay) -> bool:
        """Strategy 3: Use Alt+N keyboard shortcut to focus filename field (no clicking).
        
        Alt+N is a standard Windows shortcut that focuses the filename field in file dialogs.
        This is safer than clicking at cursor position because it doesn't depend on where
        the cursor was when the popup opened.
        """
        try:
            import time
            from app.ui.win_hotkey import set_foreground_hwnd_robust
            
            start_time = time.time()
            logger.info("[QS] Alt+N Strategy: Starting keyboard-based autofill")
            
            # Ensure the dialog is focused
            if not set_foreground_hwnd_robust(hwnd):
                logger.warning("[QS] Alt+N: Failed to focus dialog")
                return False
            
            time.sleep(0.3)  # Let focus settle
            
            # Set clipboard content first
            try:
                cb = QApplication.clipboard()
                cb.setText(path)
            except Exception as e:
                logger.error(f"[QS] Alt+N: Failed to set clipboard: {e}")
                return False
            
            try:
                import keyboard
                
                # Use Alt+N to focus filename field (standard Windows shortcut)
                logger.info("[QS] Alt+N: Sending Alt+N to focus filename field")
                keyboard.send('alt+n')
                time.sleep(0.2)
                
                # Select ALL text reliably and paste (ctrl+a is more reliable than home+shift+end)
                keyboard.send('ctrl+a')
                time.sleep(0.05)
                keyboard.send('ctrl+v')
                time.sleep(0.15)
                
                logger.info("[QS] Alt+N: Path pasted successfully")
                
                # Auto-confirm if enabled
                if settings.quick_search_auto_confirm:
                    time.sleep(0.2)
                    keyboard.send('enter')
                    logger.info("[QS] Alt+N: Auto-confirmed via Enter")
                
                elapsed = time.time() - start_time
                logger.info(f"[QS] Alt+N Strategy: SUCCESS in {elapsed:.2f}s")
                self.status_bar.showMessage("QuickSearch: path filled via Alt+N" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                return True
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"[QS] Alt+N Strategy: Failed after {elapsed:.2f}s: {e}")
                return False
            
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"[QS] Alt+N Strategy: EXCEPTION after {elapsed:.2f}s: {e}")
            return False

    def _insert_path_uia(self, target, path: str, win) -> bool:
        """Insert path into UIA Edit control using multiple methods with verification."""
        try:
            import time
            from app.core.vision import CONFIRM_NAMES
            
            def _get_text_safe(ctrl):
                try:
                    return ctrl.get_value()
                except Exception:
                    try:
                        return ctrl.window_text()
                    except Exception:
                        return None
            
            # Try insertion methods - only ONE attempt to avoid double-filling
            logger.info("[QS] UIA insertion attempt")
            
            try:
                target.set_focus()
                time.sleep(0.15)
            except Exception:
                pass
            
            filled = False
            
            # Method 1: ValuePattern.SetValue (most reliable)
            try:
                target.set_value(path)
                filled = True
                logger.info("[QS] UIA: Set via ValuePattern")
            except Exception:
                pass
            
            # Method 2: type_keys with clear (only if method 1 failed)
            if not filled:
                try:
                    target.type_keys('^a{BACKSPACE}', set_foreground=True)
                    time.sleep(0.05)
                    target.type_keys(path, with_spaces=True, set_foreground=True)
                    filled = True
                    logger.info("[QS] UIA: Set via type_keys")
                except Exception:
                    pass
            
            # Method 3: Clipboard paste fallback (only if methods 1 & 2 failed)
            if not filled:
                try:
                    cb = QApplication.clipboard()
                    cb.setText(path)
                    
                    import keyboard
                    keyboard.send('ctrl+a')
                    time.sleep(0.05)
                    keyboard.send('ctrl+v')
                    filled = True
                    logger.info("[QS] UIA: Set via clipboard paste")
                except Exception:
                    pass
            
            # Verify the text was inserted (run after all methods)
            time.sleep(0.15)
            current_text = _get_text_safe(target)
            if current_text and current_text.strip() == path.strip():
                logger.info("[QS] UIA: Path insertion verified")
                
                # Auto-confirm if enabled
                if settings.quick_search_auto_confirm:
                    time.sleep(0.2)
                    confirmed = False
                    
                    # Try to find and click Open/Save button
                    try:
                        for name in CONFIRM_NAMES:
                            try:
                                btn = win.child_window(title=name, control_type='Button')
                                if btn.exists():
                                    btn.invoke()
                                    confirmed = True
                                    logger.info(f"[QS] UIA: Confirmed via {name} button")
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass
                    
                    # Fallback: Send Enter
                    if not confirmed:
                        try:
                            target.type_keys('{ENTER}', set_foreground=True)
                            logger.info("[QS] UIA: Confirmed via Enter")
                        except Exception:
                            try:
                                win.type_keys('{ENTER}')
                            except Exception:
                                pass
                
                self.status_bar.showMessage("QuickSearch: path filled via UIA" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                return True
            else:
                logger.warning(f"[QS] UIA: Text verification failed. Expected: '{path}', Got: '{current_text}'")
            
            return False
            
        except Exception as e:
            logger.error(f"[QS] Exception in _insert_path_uia: {e}")
            return False
    
    def _insert_path_win32(self, target, path: str, win) -> bool:
        """Insert path into Win32 Edit control using multiple methods with verification."""
        try:
            import time
            
            # Single attempt to avoid double-filling
            logger.info("[QS] Win32 insertion attempt")
            
            try:
                target.set_focus()
                time.sleep(0.15)
            except Exception:
                pass
            
            filled = False
            
            # Method 1: type_keys with clear
            try:
                target.type_keys('^a{BACKSPACE}')
                time.sleep(0.05)
                target.type_keys(path, with_spaces=True)
                filled = True
                logger.info("[QS] Win32: Set via type_keys")
            except Exception:
                pass
            
            # Method 2: Clipboard paste fallback (only if method 1 failed)
            if not filled:
                try:
                    cb = QApplication.clipboard()
                    cb.setText(path)
                    
                    import keyboard
                    keyboard.send('ctrl+a')
                    time.sleep(0.05)
                    keyboard.send('ctrl+v')
                    filled = True
                    logger.info("[QS] Win32: Set via clipboard paste")
                except Exception:
                    pass
            
            # Verify the text was inserted
            time.sleep(0.15)
            try:
                current_text = target.window_text()
                if current_text and current_text.strip() == path.strip():
                    logger.info("[QS] Win32: Path insertion verified")
                    
                    # Auto-confirm if enabled
                    if settings.quick_search_auto_confirm:
                        time.sleep(0.2)
                        confirmed = False
                        
                        # Try to find and click Open/Save button
                        try:
                            from app.core.vision import CONFIRM_NAMES
                            for name in CONFIRM_NAMES:
                                try:
                                    btn = win.child_window(title=name, class_name='Button')
                                    if btn.exists():
                                        btn.click()
                                        confirmed = True
                                        logger.info(f"[QS] Win32: Confirmed via {name} button")
                                        break
                                except Exception:
                                    continue
                        except Exception:
                            pass
                        
                        # Fallback: Send Enter
                        if not confirmed:
                            try:
                                target.type_keys('{ENTER}')
                                logger.info("[QS] Win32: Confirmed via Enter")
                            except Exception:
                                try:
                                    win.type_keys('{ENTER}')
                                except Exception:
                                    pass
                    
                    self.status_bar.showMessage("QuickSearch: path filled via Win32" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                    return True
                else:
                    logger.warning(f"[QS] Win32: Text verification failed. Expected: '{path}', Got: '{current_text}'")
            except Exception as e:
                logger.warning(f"[QS] Win32: Could not verify text insertion: {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"[QS] Exception in _insert_path_win32: {e}")
            return False

    def _relative_click_into_filename(self, hwnd: int) -> bool:
        """If we saved a relative mouse point for this window, click it stealthily.
        Returns True if we clicked, False otherwise.
        """
        try:
            pt = getattr(self, '_rel_click_point', None)
            if not (hwnd and pt):
                return False
            rect = get_window_rect(hwnd)
            if not rect:
                return False
            l, t, r, b = rect
            x = l + max(0, pt[0])
            y = t + max(0, pt[1])
            # Stealth click using WinAPI: save cursor, click, restore
            user32 = ctypes.windll.user32
            cur = ctypes.wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(cur)):
                return False
            oldx, oldy = cur.x, cur.y
            user32.SetCursorPos(int(x), int(y))
            time.sleep(0.05)
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(0.02)
            user32.SetCursorPos(int(oldx), int(oldy))
            logger.info("[QS] Stealth clicked at %s,%s (relative fallback)", x, y)
            return True
        except Exception:
            return False

    def _paste_and_confirm(self, path: str) -> None:
        try:
            # Paste path and confirm
            try:
                cb = QApplication.clipboard(); cb.setText(path)
            except Exception:
                pass
            try:
                import keyboard  # type: ignore
                keyboard.send('ctrl+a')
                time.sleep(0.05)
                keyboard.send('ctrl+v')
                time.sleep(0.12)
                if settings.quick_search_auto_confirm:
                    keyboard.send('enter')
            except Exception:
                pass
            self.status_bar.showMessage("QuickSearch: path filled" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
        except Exception:
            pass

    def _autofill_uia_pipeline(self, path: str) -> bool:
        try:
            from pywinauto import Desktop
            desktop = Desktop(backend="uia")
            self.status_bar.showMessage("QuickSearch: locating file dialog…")
            logger.info("[QS] Autofill(UIA) start for path: %s", path)
            win = desktop.get_active()
            try:
                if win:
                    logger.info("[QS] Active window(UIA): title='%s' class='%s'", win.window_text(), getattr(win.element_info, 'class_name', '?'))
            except Exception:
                pass
            if not win:
                wins = desktop.windows()
                for w in reversed(wins):
                    try:
                        if not w.is_visible():
                            continue
                        btns = w.descendants(control_type='Button')
                        names = {b.window_text().lower() for b in btns}
                        if any(n in names for n in {'open', 'save', 'cancel'}):
                            edits = w.descendants(control_type='Edit')
                            if edits:
                                win = w
                                break
                    except Exception:
                        continue
            if not win:
                logger.info("[QS] No candidate file dialog found (UIA)")
                return False
            try:
                win.set_focus(); time.sleep(0.2)
            except Exception:
                pass
            target = None
            # A) FileNameControlHost
            try:
                host = win.child_window(auto_id="FileNameControlHost", control_type="Pane")
                eds = host.descendants(control_type='Edit') if host else []
                if eds:
                    target = eds[0]; logger.info("[QS] Using FileNameControlHost Edit")
            except Exception:
                pass
            # B) By label proximity
            if target is None:
                try:
                    texts = win.descendants(control_type='Text')
                except Exception:
                    texts = []
                try:
                    edits = win.descendants(control_type='Edit')
                except Exception:
                    edits = []
                label_rects = []
                for t in texts:
                    try:
                        name = (t.window_text() or '').strip()
                        if any(name.lower().startswith(lbl.lower().rstrip(':')) for lbl in FILENAME_LABELS):
                            label_rects.append(t.rectangle())
                    except Exception:
                        continue
                best = None
                best_dx = 10**9
                for e in edits:
                    try:
                        er = e.rectangle()
                        for lr in label_rects:
                            if er.left >= lr.right - 4 and (min(er.bottom, lr.bottom) - max(er.top, lr.top)) > 6:
                                dx = er.left - lr.right
                                if er.width() > 150 and er.height() < 60 and dx < best_dx:
                                    best = e; best_dx = dx
                    except Exception:
                        continue
                if best:
                    target = best; logger.info("[QS] Using Edit next to filename label")
            # C) Bottom-most edit heuristic
            if target is None:
                try:
                    edits = win.descendants(control_type='Edit')
                except Exception:
                    edits = []
                best = None; best_y = -1
                for e in edits:
                    try:
                        rect = e.rectangle()
                        if rect.width() > 150 and rect.height() < 60 and rect.top > best_y:
                            best = e; best_y = rect.top
                    except Exception:
                        continue
                target = best
            if not target:
                logger.info("[QS] No filename Edit found (UIA)")
                return False

            def _get_text_safe(ctrl):
                try:
                    return ctrl.get_value()
                except Exception:
                    try:
                        return ctrl.window_text()
                    except Exception:
                        return None

            for attempt in range(2):
                try:
                    target.set_focus(); time.sleep(0.12)
                    filled = False
                    if attempt == 0:
                        try:
                            target.set_value(path); filled = True; logger.info("[QS] Set via ValuePattern")
                        except Exception:
                            pass
                        if not filled:
                            try:
                                target.type_keys('^a{BACKSPACE}', set_foreground=True)
                                target.type_keys(path, with_spaces=True, set_foreground=True); filled = True; logger.info("[QS] Set via type_keys")
                            except Exception:
                                pass
                    else:
                        try:
                            target.type_keys('^a{BACKSPACE}', set_foreground=True)
                            target.type_keys(path, with_spaces=True, set_foreground=True); filled = True; logger.info("[QS] Retry set via type_keys")
                        except Exception:
                            pass
                        if not filled:
                            try:
                                cb = QApplication.clipboard(); cb.setText(path)
                                import keyboard  # type: ignore
                                keyboard.send('ctrl+v'); filled = True; logger.info("[QS] Retry set via clipboard paste")
                            except Exception:
                                pass
                    if not filled:
                        try:
                            cb = QApplication.clipboard(); cb.setText(path)
                            import keyboard  # type: ignore
                            keyboard.send('ctrl+v'); filled = True; logger.info("[QS] Set via clipboard paste (fallback)")
                        except Exception:
                            pass
                    time.sleep(0.12)
                    cur = _get_text_safe(target)
                    if cur and (cur.strip() == path):
                        if settings.quick_search_auto_confirm:
                            time.sleep(0.15)
                            confirmed = False
                            try:
                                for name in CONFIRM_NAMES:
                                    try:
                                        btn = win.child_window(title=name, control_type='Button')
                                        if btn:
                                            btn.invoke(); confirmed = True; logger.info("[QS] Confirmed via %s button", name); break
                                    except Exception:
                                        continue
                            except Exception:
                                pass
                            if not confirmed:
                                try:
                                    target.type_keys('{ENTER}', set_foreground=True); logger.info("[QS] Confirmed via Enter")
                                except Exception:
                                    try:
                                        win.type_keys('{ENTER}')
                                    except Exception:
                                        pass
                        self.status_bar.showMessage("QuickSearch: path filled" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                        return True
                except Exception:
                    logger.info("[QS] Exception in UIA attempt %d", attempt+1, exc_info=True)
                    continue
            return False
        except Exception:
            return False

    def _autofill_win32_pipeline(self, path: str) -> bool:
        try:
            from pywinauto import Desktop
            desktop = Desktop(backend="win32")
            logger.info("[QS] Autofill(win32) start for path: %s", path)
            win = desktop.get_active()
            try:
                if win:
                    logger.info("[QS] Active window(win32): title='%s' class='%s'", win.window_text(), getattr(win.element_info, 'class_name', '?'))
            except Exception:
                pass
            if not win:
                wins = desktop.windows()
                for w in reversed(wins):
                    try:
                        if not w.is_visible():
                            continue
                        btns = w.descendants(class_name='Button')
                        names = {b.window_text().lower() for b in btns}
                        if any(n in names for n in {'open', 'save', 'cancel'}):
                            edits = w.descendants(class_name='Edit')
                            if edits:
                                win = w; break
                    except Exception:
                        continue
            if not win:
                logger.info("[QS] No candidate file dialog found (win32)")
                return False
            try:
                win.set_focus(); time.sleep(0.2)
            except Exception:
                pass
            target = None
            try:
                combo_hosts = win.descendants(class_name='ComboBoxEx32')
                for host in combo_hosts:
                    eds = host.descendants(class_name='Edit')
                    if eds:
                        target = eds[0]; break
            except Exception:
                pass
            if target is None:
                try:
                    edits = win.descendants(class_name='Edit')
                except Exception:
                    edits = []
                if edits:
                    target = edits[-1]
            if not target:
                logger.info("[QS] No filename Edit found (win32)")
                return False
            for attempt in range(2):
                try:
                    target.set_focus(); time.sleep(0.12)
                    done = False
                    if attempt == 0:
                        try:
                            target.type_keys('^a{BACKSPACE}')
                            target.type_keys(path, with_spaces=True); done = True
                        except Exception:
                            pass
                    if not done:
                        try:
                            cb = QApplication.clipboard(); cb.setText(path)
                            import keyboard  # type: ignore
                            keyboard.send('ctrl+v'); done = True
                        except Exception:
                            pass
                    if not done:
                        continue
                    if settings.quick_search_auto_confirm:
                        time.sleep(0.15)
                        try:
                            for name in CONFIRM_NAMES:
                                btn = win.child_window(title=name, class_name='Button')
                                if btn:
                                    btn.click_input(); logger.info("[QS] Confirmed via %s button (win32)", name); break
                        except Exception:
                            pass
                        try:
                            win.type_keys('{ENTER}')
                        except Exception:
                            try:
                                target.type_keys('{ENTER}')
                            except Exception:
                                pass
                    self.status_bar.showMessage("QuickSearch: path filled" + (" and confirmed" if settings.quick_search_auto_confirm else ""))
                    return True
                except Exception:
                    logger.info("[QS] Exception in win32 attempt %d", attempt+1, exc_info=True)
                    continue
            return False
        except Exception:
            return False

    def on_save_openai(self):
        key = self.openai_key_input.text().strip()
        settings.set_openai_api_key(key)
        model = self.openai_model_combo.currentText().strip() or settings.openai_vision_model
        settings.set_openai_vision_model(model)
        self.status_bar.showMessage("OpenAI settings saved")

    def on_delete_openai_key(self):
        settings.delete_openai_api_key()
        self.openai_key_input.clear()
        self.status_bar.showMessage("OpenAI API key deleted")

    def _update_ai_provider_visibility(self):
        """Show/hide Local vs OpenAI settings based on provider selection."""
        idx = self.ai_provider_combo.currentIndex()
        # 0 = OpenAI, 1 = Local, 2 = None
        self.openai_group.setVisible(idx == 0)
        self.local_ai_group.setVisible(idx == 1)
        
        # Save the provider selection
        provider_map = {0: 'openai', 1: 'local', 2: 'none'}
        provider_names = {0: 'OpenAI (Recommended)', 1: 'Local (Ollama)', 2: 'None'}
        settings.set_ai_provider(provider_map.get(idx, 'openai'))
        
        # Show status message
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"AI Provider: {provider_names.get(idx, 'OpenAI')}")

    def _toggle_quick_index_options(self):
        """Toggle visibility of More Options content."""
        visible = not self.quick_index_content.isVisible()
        self.quick_index_content.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.quick_index_header.setText(f"{arrow} More Options")

    def _toggle_advanced_filters(self):
        """Toggle visibility of Advanced filters content."""
        visible = not self.advanced_content.isVisible()
        self.advanced_content.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.advanced_header.setText(f"{arrow} Advanced")

    def on_save_local_ai(self):
        """Save local AI model settings."""
        local_model = self.local_model_combo.currentText().strip() or settings.local_model
        settings.set_local_model(local_model)
        self.status_bar.showMessage(f"Local AI model saved: {local_model}")

    def _check_ollama_status(self):
        """Check if Ollama is running and list available models."""
        import requests
        from app.ui.organize_page import ModernInfoDialog
        
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.ok:
                data = r.json()
                models = [m.get('name', 'unknown') for m in data.get('models', [])]
                if models:
                    ModernInfoDialog.show_info(
                        self, 
                        title="Ollama Status", 
                        message="✓ Ollama is running!",
                        details=[f"• {model}" for model in models],
                        info_text="Tip: To install new models, run: ollama pull <model-name>"
                    )
                else:
                    ModernInfoDialog.show_info(
                        self, 
                        title="Ollama Status", 
                        message="✓ Ollama is running, but no models are installed.",
                        info_text="Install the recommended model:\n  ollama pull qwen2.5vl:3b"
                    )
            else:
                ModernInfoDialog.show_warning(
                    self, 
                    title="Ollama Status", 
                    message="Ollama is not responding.",
                    info_text="Make sure Ollama is running."
                )
        except requests.exceptions.ConnectionError:
            ModernInfoDialog.show_warning(
                self, 
                title="Ollama Status", 
                message="Ollama is not running.",
                info_text="Start it by:\n1. Open a terminal\n2. Run: ollama serve\n\nOr download from: https://ollama.com"
            )
        except Exception as e:
            ModernInfoDialog.show_warning(
                self, 
                title="Ollama Status", 
                message=f"Error checking Ollama status:\n{str(e)}"
            )

    def on_toggle_gpt_rerank(self, checked: bool):
        settings.set_use_openai_search_rerank(bool(checked))
        self.gpt_rerank_button.setText("Smart Rerank: ON" if checked else "Smart Rerank: OFF")
        self.status_bar.showMessage("GPT rerank " + ("enabled" if checked else "disabled"))

    def on_spell_check_toggle(self, checked: bool):
        settings.set_enable_spell_check(bool(checked))
        self.spell_check_btn.setText("Spell Check: ON" if checked else "Spell Check: OFF")
        self.status_bar.showMessage("Spell check " + ("enabled" if checked else "disabled"))

    # Quick Search settings handlers
    def on_qs_autopaste(self, checked: bool):
        settings.set_quick_search_autopaste(bool(checked))
        self.qs_autopaste_btn.setText("Auto-Paste: ON" if checked else "Auto-Paste: OFF")
        self.status_bar.showMessage("Quick Search auto-paste " + ("enabled" if checked else "disabled"))

    def on_qs_autoconfirm(self, checked: bool):
        settings.set_quick_search_auto_confirm(bool(checked))
        self.qs_autoconfirm_btn.setText("Auto-Confirm: ON" if checked else "Auto-Confirm: OFF")
        self.status_bar.showMessage("Quick Search auto-confirm " + ("enabled" if checked else "disabled"))

    def on_qs_save_shortcut(self):
        sc = (self.qs_shortcut_input.text() or '').strip()
        if not sc:
            from app.ui.organize_page import ModernInfoDialog
            ModernInfoDialog.show_warning(self, "Shortcut", "Please enter a shortcut (e.g., ctrl+alt+h)")
            return
        settings.set_quick_search_shortcut(sc)
        self.status_bar.showMessage(f"Quick Search shortcut saved: {sc}")
        # Hotkey will take effect on next app start; to apply now, restart the app

    def on_debug_cell_changed(self, item: QTableWidgetItem) -> None:
        # DIAGNOSTIC: Log every call to this handler
        logger.warning(f"[HANDLER] on_debug_cell_changed FIRED: row={item.row()}, col={item.column()}, text='{item.text()[:50] if item.text() else ''}'")
        
        # Avoid handling during table population
        if getattr(self, '_populating_debug_table', False):
            logger.warning("[HANDLER] on_debug_cell_changed BLOCKED by _populating_debug_table flag")
            return
        try:
            row = item.row()
            col = item.column()
            # file id is stored in column 0's user data
            name_item = self.debug_table.item(row, 0)
            file_id = name_item.data(Qt.UserRole) if name_item else None
            if not file_id:
                return
            text = item.text()
            if col == 4:  # Label
                ok = file_index.update_file_field(file_id, 'label', text)
            elif col == 5:  # Tags
                tags = [t.strip() for t in (text or '').split(',') if t.strip()]
                ok = file_index.update_file_field(file_id, 'tags', tags)
            elif col == 6:  # Caption
                ok = file_index.update_file_field(file_id, 'caption', text)
            elif col == 10:  # Purpose
                # update metadata JSON
                # read existing metadata from current table row if possible
                meta_text = self.debug_table.item(row, 12)  # detected text col; not metadata
                # fallback: fetch from db if needed is overkill; we set only one key
                meta = {}
                try:
                    rec = file_index.get_file_by_path(self.debug_table.item(row, 8).text())  # unlikely path in col8; ignore if fails
                except Exception:
                    rec = None
                meta = (rec or {}).get('metadata', {}) if rec else {}
                meta['purpose'] = text
                ok = file_index.update_file_field(file_id, 'metadata', meta)
            elif col == 11:  # Suggested filename
                meta = {}
                try:
                    rec = file_index.get_file_by_path(self.debug_table.item(row, 8).text())
                except Exception:
                    rec = None
                meta = (rec or {}).get('metadata', {}) if rec else {}
                meta['suggested_filename'] = text
                ok = file_index.update_file_field(file_id, 'metadata', meta)
            else:
                return
            if ok:
                self.status_bar.showMessage("Saved edit")
                # Clear selection to remove the persistent highlight
                self.debug_table.clearSelection()
                self.debug_table.setCurrentCell(-1, -1)
            else:
                from app.ui.organize_page import ModernInfoDialog
                ModernInfoDialog.show_warning(self, "Save Error", "Failed to save your edit.")
        except Exception as e:
            from app.ui.organize_page import ModernInfoDialog
            ModernInfoDialog.show_warning(self, "Edit Error", f"Failed to apply edit:\n{e}")
    
    def on_debug_cell_double_clicked(self, row: int, col: int):
        """Show full cell content in a popup when double-clicked, with edit option."""
        item = self.debug_table.item(row, col)
        if not item:
            return
        
        original_text = item.text()
        
        # Get column name
        header = self.debug_table.horizontalHeaderItem(col)
        column_name = header.text() if header else f"Column {col}"
        
        # Editable columns (others are read-only)
        editable_columns = {4: 'label', 5: 'tags', 6: 'caption', 10: 'purpose', 11: 'suggested_filename'}
        is_editable = col in editable_columns
        
        # Create a styled dialog
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"📄 {column_name}")
        dialog.setMinimumSize(600, 400)
        dialog.setModal(True)  # Ensure dialog blocks parent and stays on top
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # Style based on current theme
        is_dark = settings.theme == 'dark'
        
        if is_dark:
            dialog_style = """
                QDialog {
                    background-color: #1E1E1E;
                }
                QLabel {
                    color: #7C4DFF;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px 0;
                    background-color: transparent;
                }
                QTextEdit {
                    background-color: #2A2A2A;
                    color: #FFFFFF;
                    border: 2px solid #7C4DFF;
                    border-radius: 8px;
                    font-size: 14px;
                    padding: 15px;
                }
                QTextEdit:focus {
                    border: 2px solid #9575FF;
                }
                QPushButton {
                    background-color: #7C4DFF;
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 25px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #9575FF;
                }
            """
        else:
            # Light mode styling
            dialog_style = """
                QDialog {
                    background-color: #FFFFFF;
                }
                QLabel {
                    color: #7C4DFF;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px 0;
                    background-color: transparent;
                }
                QTextEdit {
                    background-color: #F5F5F8;
                    color: #1A1A2E;
                    border: 1px solid rgba(124, 77, 255, 0.30);
                    border-radius: 8px;
                    font-size: 14px;
                    padding: 15px;
                }
                QTextEdit:focus {
                    border: 1px solid #7C4DFF;
                }
                QPushButton {
                    background-color: #7C4DFF;
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 25px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #9575FF;
                }
            """
        
        dialog.setStyleSheet(dialog_style)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header label
        header_label = QLabel(f"{column_name}")
        if is_editable:
            header_label.setText(f"{column_name} (editable)")
        layout.addWidget(header_label)
        
        # Text edit area
        text_edit = QTextEdit()
        text_edit.setPlainText(original_text or "")
        text_edit.setReadOnly(not is_editable)
        layout.addWidget(text_edit)
        
        # Buttons
        if is_editable:
            button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
        else:
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        
        # Apply dark/light title bar
        from app.ui.theme_manager import apply_titlebar_theme
        dialog.show()
        apply_titlebar_theme(dialog)
        
        # Ensure dialog is focused and on top
        dialog.raise_()
        dialog.activateWindow()
        text_edit.setFocus()
        
        result = dialog.exec()
        
        # Save changes if user clicked Save and content changed
        if is_editable and result == QDialog.Accepted:
            new_text = text_edit.toPlainText()
            if new_text != original_text:
                # Update the table cell
                item.setText(new_text)
                self.status_bar.showMessage(f"Updated {column_name}")
    
    def select_source_folder(self):
        """Select source folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", str(Path.home())
        )
        
        if folder:
            self.source_path = Path(folder)
            self.source_label.setText(f"Source folder: {self.source_path}")
            self.source_label.setStyleSheet("")
            self.update_scan_button_state()
            self.status_bar.showMessage(f"Source folder selected: {self.source_path}")
    
    def select_destination_folder(self):
        """Select destination folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder", str(Path.home())
        )
        
        if folder:
            self.destination_path = Path(folder)
            self.dest_label.setText(f"Destination folder: {self.destination_path}")
            self.dest_label.setStyleSheet("")
            self.update_scan_button_state()
            self.status_bar.showMessage(f"Destination folder selected: {self.destination_path}")
    
    def update_scan_button_state(self):
        """Update scan button enabled state."""
        self.scan_button.setEnabled(
            self.source_path is not None and self.destination_path is not None
        )
    
    def scan_and_plan(self):
        """Scan source folder and create move plan."""
        if not self.source_path or not self.destination_path:
            return
        
        # Clear previous results
        self.file_table.setRowCount(0)
        self.summary_text.clear()
        self.scanned_files = []
        self.move_plan = []
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.scan_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        
        # Start scan worker
        self.scan_worker = ScanWorker(self.source_path)
        self.scan_worker.scan_completed.connect(self.on_scan_completed)
        self.scan_worker.scan_error.connect(self.on_scan_error)
        self.scan_worker.progress_updated.connect(self.status_bar.showMessage)
        self.scan_worker.start()
    
    def on_scan_completed(self, files: List[Dict[str, Any]]):
        """Handle scan completion."""
        self.scanned_files = files
        
        if not files:
            self.status_bar.showMessage("No files found in source directory")
            self.progress_bar.setVisible(False)
            self.scan_button.setEnabled(True)
            return
        
        # Create move plan
        self.status_bar.showMessage("Creating move plan...")
        self.move_plan = create_move_plan(files, self.source_path, self.destination_path)
        
        # Display results
        self.display_results()
        
        # Update UI
        self.progress_bar.setVisible(False)
        self.scan_button.setEnabled(True)
        self.apply_button.setEnabled(len(self.move_plan) > 0)
        
        self.status_bar.showMessage(f"Scan completed. Found {len(files)} files.")
    
    def on_scan_error(self, error: str):
        """Handle scan error."""
        from app.ui.organize_page import ModernInfoDialog
        ModernInfoDialog.show_warning(self, "Scan Error", f"Error scanning directory:\n{error}")
        self.progress_bar.setVisible(False)
        self.scan_button.setEnabled(True)
        self.status_bar.showMessage("Scan failed")
    
    def display_results(self):
        """Display scan and plan results."""
        # Populate table
        self.file_table.setRowCount(len(self.move_plan))
        
        for row, move in enumerate(self.move_plan):
            # File name
            name_item = QTableWidgetItem(move['file_name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.file_table.setItem(row, 0, name_item)
            
            # Category
            category_item = QTableWidgetItem(move['category'])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsEditable)
            self.file_table.setItem(row, 1, category_item)
            
            # Size
            size_mb = round(move['size'] / (1024 * 1024), 2)
            size_item = QTableWidgetItem(f"{size_mb} MB")
            size_item.setFlags(size_item.flags() & ~Qt.ItemIsEditable)
            self.file_table.setItem(row, 2, size_item)
            
            # Planned destination
            dest_item = QTableWidgetItem(move['relative_destination'])
            dest_item.setFlags(dest_item.flags() & ~Qt.ItemIsEditable)
            self.file_table.setItem(row, 3, dest_item)
        
        # Display summary
        summary = get_plan_summary(self.move_plan)
        summary_text = f"""
Move Plan Summary:
• Total files: {summary['total_files']}
• Total size: {summary['total_size_mb']} MB
• Categories:
"""
        
        for category, info in summary['categories'].items():
            count = info['count']
            size_mb = round(info['size'] / (1024 * 1024), 2)
            summary_text += f"  - {category}: {count} files ({size_mb} MB)\n"
        
        self.summary_text.setPlainText(summary_text)
    
    def apply_moves(self):
        """Apply the move plan."""
        if not self.move_plan:
            return
        
        # Validate plan
        is_valid, errors = validate_move_plan(
            self.move_plan, self.source_path, self.destination_path
        )
        
        if not is_valid:
            error_text = "\n".join(errors)
            from app.ui.organize_page import ModernInfoDialog
            ModernInfoDialog.show_warning(self, "Validation Error", f"Move plan validation failed:\n{error_text}")
            return
        
        # Check disk space
        has_space, space_error = validate_destination_space(
            self.move_plan, self.destination_path
        )
        
        if not has_space:
            from app.ui.organize_page import ModernInfoDialog
            ModernInfoDialog.show_warning(self, "Insufficient Space", space_error)
            return
        
        # Confirm action
        from app.ui.organize_page import ModernConfirmDialog
        
        confirmed = ModernConfirmDialog.ask(
            self, 
            title="Confirm Moves",
            message=f"Are you sure you want to move {len(self.move_plan)} files?",
            info_text="This action cannot be undone in this version.",
            yes_text="Move Files",
            no_text="Cancel"
        )
        
        if not confirmed:
            return
        
        # Apply moves
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.move_plan))
        self.apply_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        
        success, errors, log_file, renamed_count = apply_moves(self.move_plan)
        
        self.progress_bar.setVisible(False)
        self.scan_button.setEnabled(True)
        
        if success:
            renamed_msg = ""
            if renamed_count > 0:
                renamed_msg = f"\n\n{renamed_count} file(s) renamed to avoid duplicates."
            QMessageBox.information(
                self, "Success",
                f"Successfully moved {len(self.move_plan)} files!{renamed_msg}\n\n"
                f"Move log saved to: {log_file}"
            )
            self.status_bar.showMessage("Moves completed successfully")
            
            # Clear results
            self.file_table.setRowCount(0)
            self.summary_text.clear()
            self.scanned_files = []
            self.move_plan = []
            self.apply_button.setEnabled(False)
        else:
            error_text = "\n".join(errors[:10])  # Show first 10 errors
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more errors"
            
            QMessageBox.critical(
                self, "Move Errors",
                f"Some files could not be moved:\n{error_text}"
            )
            self.status_bar.showMessage("Moves completed with errors")
            self.apply_button.setEnabled(True)

    # Search functionality methods
    def select_index_folder(self):
        """Select folder to index for search."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Folder", str(Path.home())
        )
        
        if folder:
            self.index_path = Path(folder)
            self.index_label.setText(f"Index folder: {self.index_path}")
            self.index_label.setStyleSheet("")
            self.index_button_action.setEnabled(True)
            self.status_bar.showMessage(f"Index folder selected: {self.index_path}")
    
    def index_directory(self):
        """Index the selected directory for search."""
        if not hasattr(self, 'index_path') or not self.index_path:
            return
        
        # If already indexing, add to queue instead of replacing
        if self.is_indexing:
            self._add_to_index_queue(self.index_path)
            return
        
        # Start indexing
        self._start_indexing_path(self.index_path)
    
    def _add_to_index_queue(self, path: Path):
        """Add a path to the indexing queue."""
        logger.info(f"_add_to_index_queue called with path: {path}")
        if path not in self.index_queue:
            self.index_queue.append(path)
            logger.info(f"Queue now has {len(self.index_queue)} items: {[p.name for p in self.index_queue]}")
            self._update_queue_ui()
            self.status_bar.showMessage(f"Added to queue: {path.name} ({len(self.index_queue)} pending)")
    
    def _update_queue_ui(self):
        """Update the compact queue indicator and expand section when queue is visible."""
        logger.info(f"_update_queue_ui called. Queue has {len(self.index_queue)} items")
        if self.index_queue:
            logger.info("Queue not empty, showing queue_row")
            # Add spacing above the queue row
            self.queue_top_spacer.setVisible(True)
            self.queue_top_spacer.setFixedHeight(25)  # Space above queue
            
            self.queue_row.setVisible(True)
            self.queue_row.show()  # Force show
            self.queue_spacer.setVisible(True)
            self.queue_spacer.setFixedHeight(15)  # Space below queue
            logger.info(f"queue_row.isVisible() = {self.queue_row.isVisible()}")
            
            count = len(self.index_queue)
            self.queue_label.setText(f"📋 Queue: {count} pending")
            # Show first 2 item names inline
            names = [p.name for p in self.index_queue[:2]]
            if count > 2:
                names.append(f"+{count - 2} more")
            self.queue_items_label.setText(" • ".join(names))
            # Expand the index group to accommodate queue
            self.index_group.setMinimumHeight(self.index_group.sizeHint().height() + 80)
        else:
            self.queue_top_spacer.setVisible(False)
            self.queue_top_spacer.setFixedHeight(0)
            self.queue_row.setVisible(False)
            self.queue_spacer.setVisible(False)
            self.queue_spacer.setFixedHeight(0)
            # Reset minimum height
            self.index_group.setMinimumHeight(0)
    
    def _clear_index_queue(self):
        """Clear the indexing queue."""
        self.index_queue.clear()
        self._update_queue_ui()
        self.status_bar.showMessage("Queue cleared")
    
    def _start_indexing_path(self, path: Path):
        """Start indexing a specific path."""
        self.is_indexing = True
        self.index_path = path
        self.index_label.setText(f"Indexing: {path.name}")
        self.index_button_action.setText("➕ Add to Queue")
        self.index_button_action.setEnabled(True)  # Allow adding more
        
        # Show progress container (contains progress bar, labels, buttons)
        if hasattr(self, 'index_progress_container'):
            self.index_progress_container.setVisible(True)
        
        # Show progress controls with explicit text settings
        self.progress_bar.setVisible(True)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v / %m files)")
        self.progress_bar.setRange(0, 0)  # Start indeterminate
        self.progress_bar.setProperty("paused", False)
        self.progress_bar.style().unpolish(self.progress_bar)
        self.progress_bar.style().polish(self.progress_bar)
        
        # Update drop zone text instead of hiding it completely
        self._update_drop_zone(f"Indexing: {path.name}", "Drop more folders to add to queue")
        
        self.index_percent_label.setVisible(True)
        self.index_percent_label.setText("0%")
        self.index_pause_btn.setVisible(True)
        self.index_pause_btn.setText("Pause")
        self.index_cancel_btn.setVisible(True)
        self.index_progress_label.setVisible(True)
        # Keep button enabled so user can add more files to queue
        # self.index_button_action.setEnabled(False)  # Removed - allow adding to queue
        self.status_bar.showMessage("Indexing directory...")
        
        # Create the worker
        self.index_worker = IndexWorker(self.index_path)
        self.index_worker.index_completed.connect(self.on_index_completed)
        self.index_worker.index_error.connect(self.on_index_error)
        self.index_worker.progress_updated.connect(self.status_bar.showMessage)
        # Connect progress_data signal to slot for thread-safe UI updates
        self.index_worker.progress_data.connect(self._on_index_progress)
        
        # Progress callback that emits signal instead of using QTimer
        def progress_cb(done: int, total: int, message: str):
            # Check for cancel first
            if hasattr(self, 'index_worker') and self.index_worker:
                if self.index_worker.is_cancelled():
                    raise InterruptedError("Indexing cancelled by user")
            
            # Emit signal - this will be received on the main thread
            self.index_worker.progress_data.emit(done, total, message)
            
            # Now wait if paused
            if hasattr(self, 'index_worker') and self.index_worker:
                self.index_worker.wait_if_paused()
                if self.index_worker.is_cancelled():
                    raise InterruptedError("Indexing cancelled by user")

        # Monkey-patch run to inject callback without refactor
        def run_with_progress():
            try:
                result = search_service.index_directory(self.index_path, progress_cb=progress_cb)
                self.index_worker.index_completed.emit(result)
            except Exception as e:
                self.index_worker.index_error.emit(str(e))
        self.index_worker.run = run_with_progress  # type: ignore
        self.index_worker.start()
    
    def _toggle_index_pause(self):
        """Toggle pause/resume for indexing - IMMEDIATE UI response."""
        # Use search_service pause directly for real control
        if search_service.is_paused():
            # Resume
            search_service.resume_indexing()
            self.index_pause_btn.setText("⏸ Pause")
            self.status_bar.showMessage("Indexing resumed...")
            self.index_progress_label.setStyleSheet("")  # Normal color
            self.index_percent_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7C4DFF;")
            # Reset progress bar to normal color using property
            self.progress_bar.setProperty("paused", False)
            self.progress_bar.style().unpolish(self.progress_bar)
            self.progress_bar.style().polish(self.progress_bar)
        else:
            # Pause - IMMEDIATELY freeze the indexing
            search_service.pause_indexing()
            self.index_pause_btn.setText("▶ Resume")
            self.status_bar.showMessage("⏸ PAUSED - Click Resume to continue")
            self.index_progress_label.setText("⏸ PAUSED")
            self.index_progress_label.setStyleSheet("color: #FFA500; font-weight: bold;")  # Orange
            self.index_percent_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFA500;")
            # Change progress bar to orange when paused using property
            self.progress_bar.setProperty("paused", True)
            self.progress_bar.style().unpolish(self.progress_bar)
            self.progress_bar.style().polish(self.progress_bar)
    
    def _cancel_indexing(self):
        """Cancel the current indexing operation - IMMEDIATE UI response."""
        reply = QMessageBox.question(
            self,
            "Cancel Indexing",
            "Are you sure you want to cancel indexing?\n\nFiles already indexed will be kept.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Cancel via search_service for REAL cancellation
            search_service.cancel_indexing()
            
            # Also cancel worker if it exists
            if hasattr(self, 'index_worker') and self.index_worker:
                self.index_worker.cancel()
            
            # Clear PC indexing queue if active
            if hasattr(self, '_pc_index_queue'):
                self._pc_index_queue = []
            
            self._hide_index_controls()
            self.status_bar.showMessage("Indexing cancelled. Files already indexed have been saved.")
            
            # Refresh views with what was indexed
            stats = search_service.get_index_statistics()
            self.update_search_statistics(stats)
            self.search_button.setEnabled(True)
            self.refresh_debug_view()
    
    def _on_index_progress(self, done: int, total: int, message: str):
        """Slot to handle progress updates from the worker thread (runs on main thread)."""
        try:
            # Don't update UI if paused (keep showing PAUSED message)
            if hasattr(self, 'index_worker') and self.index_worker and self.index_worker.is_paused():
                return
            
            # Don't update if cancelled
            if hasattr(self, 'index_worker') and self.index_worker and self.index_worker.is_cancelled():
                return
            
            self.progress_bar.setVisible(True)
            if total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(done)
                percent = int((done / total) * 100)
                # Update the prominent percentage label
                self.index_percent_label.setText(f"{percent}%")
                self.index_percent_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7C4DFF;")
                self.index_progress_label.setText(f"Processing file {done} of {total}")
                self.index_progress_label.setStyleSheet("")  # Reset style
            else:
                self.progress_bar.setRange(0, 0)
                self.index_percent_label.setText("...")
                self.index_progress_label.setText("Scanning files...")
            self.status_bar.showMessage(message)
        except Exception:
            pass  # UI might be closed
    
    def _hide_index_controls(self):
        """Hide indexing controls after completion."""
        # Hide progress container
        if hasattr(self, 'index_progress_container'):
            self.index_progress_container.setVisible(False)
        
        self.progress_bar.setVisible(False)
        self.index_pause_btn.setVisible(False)
        self.index_cancel_btn.setVisible(False)
        self.index_progress_label.setVisible(False)
        self.index_percent_label.setVisible(False)
        
        # Restore drop zone and options
        self.drop_zone.setVisible(True)
        if hasattr(self, 'more_options_header'):
            self.more_options_header.setVisible(True)
        
        # Reset drop zone text
        self._update_drop_zone("Add folder to index", "Drag and drop or click to browse")
        self.index_button_action.setEnabled(True)
        
        # Update indexed paths list
        self._update_indexed_paths_list()
    
    def on_index_completed(self, result: Dict[str, Any]):
        """Handle index completion."""
        # Check if there are more items in the queue
        if self.index_queue:
            next_path = self.index_queue.pop(0)
            self._update_queue_ui()
            self.status_bar.showMessage(f"Starting next: {next_path.name}")
            QTimer.singleShot(300, lambda: self._start_indexing_path(next_path))
            return
        
        # No more items - finish up
        self.is_indexing = False
        self.index_button_action.setText("Add Folder")
        self._hide_index_controls()
        
        if 'error' in result:
            QMessageBox.critical(self, "Index Error", f"Error indexing directory:\n{result['error']}")
            self.status_bar.showMessage("Indexing failed")
            return
        
        # Check if index limit was exceeded
        if result.get('limit_exceeded'):
            limit_info = result.get('limit_info', {})
            self._show_upgrade_dialog(limit_info)
            self.status_bar.showMessage("Index limit reached - upgrade for more")
            return
        
        if result.get('cancelled'):
            self.status_bar.showMessage(
                f"Indexing cancelled. Indexed {result.get('indexed_files', 0)} files before cancellation."
            )
            # Still update stats for what was indexed
            stats = search_service.get_index_statistics()
            self.update_search_statistics(stats)
            self.search_button.setEnabled(True)
            self.refresh_debug_view()
            # Update usage labels to reflect any files indexed before cancel
            self._update_usage_labels()
            return
        
        # Update search statistics
        stats = search_service.get_index_statistics()
        self.update_search_statistics(stats)
        
        # Enable search
        self.search_button.setEnabled(True)
        
        # Refresh debug view and force UI update
        self.refresh_debug_view()
        
        # Force table to update visually
        if hasattr(self, 'debug_table'):
            self.debug_table.viewport().update()
            QApplication.processEvents()
        
        # Show success message
        indexed_count = result.get('indexed_files', 0)
        self.status_bar.showMessage(
            f"✓ Indexed {indexed_count} files ({result.get('files_with_ocr', 0)} with OCR)"
        )
        
        # Update info label
        if hasattr(self, 'debug_info_label'):
            self.debug_info_label.setText(f"Showing {indexed_count} indexed files")
        
        # Update usage labels to reflect new count
        self._update_usage_labels()
    
    def _show_upgrade_dialog(self, limit_info: dict):
        """Show upgrade dialog when index limit is reached."""
        from app.ui.organize_page import ModernConfirmDialog, ModernInfoDialog
        from app.core.supabase_client import supabase_auth, INDEX_LIMIT_STARTER, INDEX_LIMIT_ULTRA
        
        remaining = limit_info.get('remaining', 0)
        limit = limit_info.get('limit', INDEX_LIMIT_STARTER)
        plan = limit_info.get('plan', 'starter')
        
        if plan == 'ultra':
            # Ultra users hit their 5000 limit - no upgrade available
            ModernInfoDialog.show_info(
                self,
                title="Index Limit Reached",
                message=f"You've reached your Ultra plan limit of {INDEX_LIMIT_ULTRA} media files this month.",
                details=[
                    "Your limit will reset at the start of your next billing cycle.",
                    "Text files are unlimited and can still be indexed."
                ]
            )
        else:
            # Starter users - offer upgrade to Ultra
            confirmed = ModernConfirmDialog.ask(
                self,
                title="Index Limit Reached",
                message=f"You've reached your monthly limit of {limit} media files.",
                info_text=f"Upgrade to Ultra for {INDEX_LIMIT_ULTRA} media files per month!",
                highlight_text=f"Current plan: {plan.title()} | Used: {limit - remaining}/{limit}",
                yes_text="Upgrade to Ultra ($49/mo)",
                no_text="Maybe Later"
            )
            
            if confirmed:
                # Open upgrade checkout
                supabase_auth.open_upgrade_checkout()
                self.status_bar.showMessage("Opening upgrade checkout...", 5000)
    
    def on_index_error(self, error: str):
        """Handle index error."""
        self._hide_index_controls()
        
        # Check if it was a cancellation
        if "cancelled" in error.lower() or "interrupted" in error.lower():
            self.status_bar.showMessage("Indexing cancelled")
            return
        
        QMessageBox.critical(self, "Index Error", f"Error indexing directory:\n{error}")
        self.status_bar.showMessage("Indexing failed")
    
    def on_index_entire_pc(self):
        """Handle 'Search Entire PC' button click with styled warning dialog."""
        from PySide6.QtWidgets import QDialog, QFrame, QScrollArea
        import os
        
        # Get folders that will be indexed
        home = Path.home()
        user_folders = []
        
        if os.name == 'nt':
            for folder_name in ['Desktop', 'Documents', 'Downloads', 'Pictures', 'Videos', 'Music']:
                folder = home / folder_name
                if folder.exists():
                    user_folders.append(folder)
            for item in home.iterdir():
                if item.is_dir() and 'OneDrive' in item.name:
                    user_folders.append(item)
        else:
            for folder_name in ['Desktop', 'Documents', 'Downloads', 'Pictures', 'Videos', 'Music']:
                folder = home / folder_name
                if folder.exists():
                    user_folders.append(folder)
        
        if not user_folders:
            QMessageBox.information(self, "No Folders Found", "Could not find any standard user folders to index.")
            return
        
        # Create custom styled dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Index Entire PC")
        dialog.setObjectName("styledWarningDialog")
        dialog.setFixedSize(460, 420)
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dialog.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container with styling
        container = QFrame(dialog)
        container.setObjectName("warningDialogContainer")
        container.setGeometry(0, 0, 460, 420)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        
        # Header with icon
        header = QLabel("⚡ Index Entire PC")
        header.setObjectName("warningDialogTitle")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Subtitle
        subtitle = QLabel("Scan all your files for AI-powered search")
        subtitle.setObjectName("warningDialogSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Folders list
        folders_label = QLabel("Folders to index:")
        folders_label.setObjectName("warningDialogSectionLabel")
        layout.addWidget(folders_label)
        
        folder_list_text = "\n".join([f"📁 {f.name}" for f in user_folders])
        folders_content = QLabel(folder_list_text)
        folders_content.setObjectName("warningDialogFolderList")
        folders_content.setWordWrap(True)
        layout.addWidget(folders_content)
        
        # Info note
        info_note = QLabel(
            "ℹ️ This may take a while depending on the number of files.\n"
            "You can pause or cancel at any time."
        )
        info_note.setObjectName("warningDialogNote")
        info_note.setWordWrap(True)
        layout.addWidget(info_note)
        
        layout.addStretch()
        
        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("warningDialogCancelBtn")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setMinimumHeight(44)
        cancel_btn.setMinimumWidth(130)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("Start Indexing")
        confirm_btn.setObjectName("warningDialogConfirmBtn")
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.setMinimumHeight(44)
        confirm_btn.setMinimumWidth(150)
        confirm_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(confirm_btn)
        
        layout.addLayout(btn_row)
        
        # Center on parent
        dialog.move(
            self.x() + (self.width() - dialog.width()) // 2,
            self.y() + (self.height() - dialog.height()) // 2
        )
        
        # Apply dark/light title bar
        from app.ui.theme_manager import apply_titlebar_theme
        dialog.show()
        apply_titlebar_theme(dialog)
        
        if dialog.exec() == QDialog.Accepted:
            # Start indexing with the existing progress UI
            self._pc_index_queue = list(user_folders)
            self._pc_total_indexed = 0  # Reset counter
            self._index_next_pc_folder()
    
    
    def _index_next_pc_folder(self):
        """Index the next folder in the PC indexing queue."""
        if not hasattr(self, '_pc_index_queue') or not self._pc_index_queue:
            # All folders done - hide progress UI and show drop zone
            if hasattr(self, 'index_progress_container'):
                self.index_progress_container.setVisible(False)
            self.progress_bar.setVisible(False)
            self.index_progress_label.setVisible(False)
            self.index_percent_label.setVisible(False)
            
            # Restore drop zone and options
            self.drop_zone.setVisible(True)
            self.more_options_header.setVisible(True)
            self._update_drop_zone("Add folder to index", "Drag and drop or click to browse")
            self.status_bar.showMessage("PC indexing complete!", 5000)
            
            # Refresh file count
            self.refresh_debug_view()
            
            # Show completion dialog
            total_indexed = getattr(self, '_pc_total_indexed', 0)
            QMessageBox.information(
                self, "Indexing Complete",
                f"Successfully indexed {total_indexed} files from your PC."
            )
            return
        
        folder = self._pc_index_queue.pop(0)
        remaining = len(self._pc_index_queue)
        self.index_path = folder
        
        # Hide drop zone and show progress container
        self.drop_zone.setVisible(False)
        self.more_options_header.setVisible(False)
        self.more_options_content.setVisible(False)
        
        self.status_bar.showMessage(f"Indexing: {folder}")
        
        # Show progress container with pause/cancel buttons
        if hasattr(self, 'index_progress_container'):
            self.index_progress_container.setVisible(True)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.index_progress_label.setVisible(True)
        self.index_progress_label.setText(f"Indexing: {folder.name} ({remaining} folder(s) remaining)")
        self.index_percent_label.setVisible(True)
        self.index_percent_label.setText("0%")
        self.index_pause_btn.setVisible(True)
        self.index_cancel_btn.setVisible(True)
        
        def progress_cb(done: int, total: int, message: str):
            def update_ui():
                try:
                    if total > 0:
                        self.progress_bar.setRange(0, total)
                        self.progress_bar.setValue(done)
                        percent = int((done / total) * 100)
                        self.index_percent_label.setText(f"{percent}%")
                    else:
                        self.progress_bar.setRange(0, 0)
                    self.index_progress_label.setText(message)
                    # Force immediate UI refresh
                    QApplication.processEvents()
                except Exception:
                    pass
            QTimer.singleShot(0, update_ui)
        
        self.index_worker = IndexWorker(folder)
        
        def on_complete(result):
            # Track total indexed files
            if not hasattr(self, '_pc_total_indexed'):
                self._pc_total_indexed = 0
            self._pc_total_indexed += result.get('indexed_files', 0)
            
            stats = search_service.get_index_statistics()
            self.update_search_statistics(stats)
            self.refresh_debug_view()
            
            # Continue with next folder
            QTimer.singleShot(100, self._index_next_pc_folder)
        
        def on_error(error):
            logger.error(f"Error indexing {folder}: {error}")
            # Continue with next folder despite error
            QTimer.singleShot(100, self._index_next_pc_folder)
        
        self.index_worker.index_completed.connect(on_complete)
        self.index_worker.index_error.connect(on_error)
        
        def run_with_progress():
            try:
                result = search_service.index_directory(folder, progress_cb=progress_cb)
                self.index_worker.index_completed.emit(result)
            except Exception as e:
                self.index_worker.index_error.emit(str(e))
        
        self.index_worker.run = run_with_progress
        self.index_worker.start()
    
    def on_toggle_auto_index_downloads(self, checked: bool):
        """Handle auto-add toggle with warning."""
        if checked:
            # Show warning before enabling
            reply = QMessageBox.warning(
                self,
                "Enable Auto-Add New Files",
                "📥 This will automatically add any NEW files from common folders:\n\n"
                "• Downloads\n"
                "• Desktop\n"
                "• Documents\n"
                "• Pictures\n"
                "• Videos\n"
                "• Music\n"
                "• OneDrive (if present)\n\n"
                "Only files added AFTER enabling will be included.\n"
                "Existing files will NOT be touched.\n\n"
                "Do you want to enable this feature?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                settings.set_auto_index_downloads(True)
                self.auto_index_downloads_btn.setText("📥 Auto-Add New Files: ON")
                self._start_downloads_watcher()
                self.auto_index_status.setText("Monitoring folders for new files...")
            else:
                # User cancelled - uncheck the button
                self.auto_index_downloads_btn.setChecked(False)
        else:
            settings.set_auto_index_downloads(False)
            self.auto_index_downloads_btn.setText("📥 Auto-Add New Files: OFF")
            self._stop_downloads_watcher()
            self.auto_index_status.setText("")
    
    def _toggle_more_options(self):
        """Toggle visibility of More Options section."""
        visible = not self.more_options_content.isVisible()
        self.more_options_content.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.more_options_header.setText(f"{arrow} More Options")
    
    def _toggle_watch_options(self):
        """Open the Watch for New Downloads popup dialog."""
        self._show_watch_popup()
    
    def _show_watch_popup(self):
        """Show the Watch for New Downloads popup dialog."""
        from PySide6.QtWidgets import QDialog, QFrame, QScrollArea
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Watch for New Downloads")
        dialog.setObjectName("watchPopupDialog")
        dialog.setFixedSize(480, 520)
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dialog.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main container
        container = QFrame(dialog)
        container.setObjectName("warningDialogContainer")
        container.setGeometry(0, 0, 480, 520)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        
        # Header row with title and close button
        header_row = QHBoxLayout()
        header = QLabel("👁️ Watch for New Downloads")
        header.setObjectName("warningDialogTitle")
        header_row.addWidget(header)
        header_row.addStretch()
        
        close_btn = QPushButton("✕")
        close_btn.setObjectName("watchPopupCloseBtn")
        close_btn.setFixedSize(32, 32)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(dialog.accept)
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)
        
        # Subtitle
        subtitle = QLabel("Automatically index new files as they're added")
        subtitle.setObjectName("warningDialogSubtitle")
        layout.addWidget(subtitle)
        
        # === Common Folders Section ===
        common_section = QFrame()
        common_section.setObjectName("watchSectionFrame")
        common_layout = QVBoxLayout(common_section)
        common_layout.setContentsMargins(16, 12, 16, 12)
        common_layout.setSpacing(8)
        
        common_header = QHBoxLayout()
        common_label = QLabel("📁 Common Folders")
        common_label.setObjectName("watchSectionTitle")
        common_header.addWidget(common_label)
        common_header.addStretch()
        
        common_toggle = QPushButton("OFF")
        common_toggle.setObjectName("watchTogglePill")
        common_toggle.setCheckable(True)
        common_toggle.setChecked(settings.watch_common_folders)
        common_toggle.setFixedSize(60, 28)
        common_toggle.setCursor(Qt.PointingHandCursor)
        if settings.watch_common_folders:
            common_toggle.setText("ON")
        common_header.addWidget(common_toggle)
        common_layout.addLayout(common_header)
        
        common_desc = QLabel("Downloads, Desktop, Documents, Pictures, Videos, Music")
        common_desc.setObjectName("watchSectionDesc")
        common_desc.setWordWrap(True)
        common_layout.addWidget(common_desc)
        
        layout.addWidget(common_section)
        
        # === Custom Folders Section ===
        custom_section = QFrame()
        custom_section.setObjectName("watchSectionFrame")
        custom_layout = QVBoxLayout(custom_section)
        custom_layout.setContentsMargins(16, 12, 16, 12)
        custom_layout.setSpacing(8)
        
        custom_header = QHBoxLayout()
        custom_label = QLabel("📂 Custom Folders")
        custom_label.setObjectName("watchSectionTitle")
        custom_header.addWidget(custom_label)
        custom_header.addStretch()
        
        add_folder_btn = QPushButton("+ Add")
        add_folder_btn.setObjectName("watchAddFolderBtn")
        add_folder_btn.setCursor(Qt.PointingHandCursor)
        add_folder_btn.setFixedHeight(28)
        custom_header.addWidget(add_folder_btn)
        custom_layout.addLayout(custom_header)
        
        # Custom folders list
        folders_list_widget = QWidget()
        folders_list_layout = QVBoxLayout(folders_list_widget)
        folders_list_layout.setContentsMargins(0, 4, 0, 0)
        folders_list_layout.setSpacing(4)
        
        def refresh_folder_list():
            # Clear existing
            while folders_list_layout.count():
                item = folders_list_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            if not settings.watch_custom_folders:
                empty_label = QLabel("No custom folders added")
                empty_label.setObjectName("watchEmptyLabel")
                folders_list_layout.addWidget(empty_label)
            else:
                for folder_path in settings.watch_custom_folders:
                    folder_row = QWidget()
                    row_layout = QHBoxLayout(folder_row)
                    row_layout.setContentsMargins(0, 2, 0, 2)
                    row_layout.setSpacing(8)
                    
                    # Folder icon and name
                    folder_name = Path(folder_path).name
                    folder_label = QLabel(f"📁 {folder_name}")
                    folder_label.setObjectName("watchFolderItem")
                    folder_label.setToolTip(folder_path)
                    row_layout.addWidget(folder_label, 1)
                    
                    # Remove button
                    remove_btn = QPushButton("✕")
                    remove_btn.setObjectName("watchRemoveFolderBtn")
                    remove_btn.setFixedSize(22, 22)
                    remove_btn.setCursor(Qt.PointingHandCursor)
                    remove_btn.setToolTip("Remove")
                    remove_btn.clicked.connect(lambda checked, fp=folder_path: remove_folder(fp))
                    row_layout.addWidget(remove_btn)
                    
                    folders_list_layout.addWidget(folder_row)
        
        def remove_folder(folder_path):
            settings.remove_watch_custom_folder(folder_path)
            refresh_folder_list()
            self._restart_folder_watching()
            self._update_watch_status()
        
        def add_folder():
            folder = QFileDialog.getExistingDirectory(dialog, "Select Folder to Watch", str(Path.home()))
            if folder:
                settings.add_watch_custom_folder(folder)
                refresh_folder_list()
                self._restart_folder_watching()
                self._update_watch_status()
        
        add_folder_btn.clicked.connect(add_folder)
        refresh_folder_list()
        
        custom_layout.addWidget(folders_list_widget)
        layout.addWidget(custom_section)
        
        layout.addStretch()
        
        # Status
        status_label = QLabel("")
        status_label.setObjectName("watchPopupStatus")
        status_label.setAlignment(Qt.AlignCenter)
        
        def update_status():
            watching = []
            if settings.watch_common_folders:
                watching.append("common folders")
            if settings.watch_custom_folders:
                watching.append(f"{len(settings.watch_custom_folders)} custom folder(s)")
            if watching:
                status_label.setText(f"✓ Watching: {', '.join(watching)}")
                status_label.setStyleSheet("color: #4CAF50; font-size: 13px;")
            else:
                status_label.setText("Not watching any folders")
                status_label.setStyleSheet("color: #7A7A90; font-size: 13px;")
        
        update_status()
        layout.addWidget(status_label)
        
        # Done button
        done_btn = QPushButton("Done")
        done_btn.setObjectName("warningDialogConfirmBtn")
        done_btn.setCursor(Qt.PointingHandCursor)
        done_btn.setMinimumHeight(44)
        done_btn.setMinimumWidth(120)
        done_btn.clicked.connect(dialog.accept)
        layout.addWidget(done_btn, 0, Qt.AlignCenter)
        
        # Connect toggle
        def on_common_toggle(checked):
            if checked:
                settings.set_watch_common_folders(True)
                common_toggle.setText("ON")
            else:
                settings.set_watch_common_folders(False)
                common_toggle.setText("OFF")
            self._restart_folder_watching()
            self._update_watch_status()
            update_status()
        
        common_toggle.toggled.connect(on_common_toggle)
        
        # Center on parent
        dialog.move(
            self.x() + (self.width() - dialog.width()) // 2,
            self.y() + (self.height() - dialog.height()) // 2
        )
        
        # Apply dark/light title bar
        from app.ui.theme_manager import apply_titlebar_theme
        dialog.show()
        apply_titlebar_theme(dialog)
        
        dialog.exec()
    
    def _on_watch_common_toggled(self, checked: bool):
        """Handle common folders watch toggle."""
        if checked:
            reply = QMessageBox.question(
                self,
                "Watch Common Folders",
                "This will monitor these folders for new files:\n\n"
                "• Downloads\n"
                "• Desktop\n"
                "• Documents\n"
                "• Pictures\n"
                "• Videos\n"
                "• Music\n\n"
                "Only NEW files will be indexed automatically.\n"
                "Enable watching?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                settings.set_watch_common_folders(True)
                self._start_folder_watching()
                self._update_watch_status()
            else:
                self.watch_common_toggle.setChecked(False)
        else:
            settings.set_watch_common_folders(False)
            self._restart_folder_watching()
            self._update_watch_status()
    
    def _on_watch_custom_toggled(self, checked: bool):
        """Handle custom folders watch toggle."""
        if checked and not settings.watch_custom_folders:
            # No custom folders yet - prompt to add one
            self._on_add_custom_folder()
            if not settings.watch_custom_folders:
                # User cancelled - uncheck
                self.watch_custom_toggle.setChecked(False)
        elif not checked:
            # Disable custom watching but keep the folder list
            self._restart_folder_watching()
            self._update_watch_status()
    
    def _on_add_custom_folder(self):
        """Add a custom folder to watch."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Watch",
            str(Path.home())
        )
        if folder:
            settings.add_watch_custom_folder(folder)
            self._refresh_custom_folders_list()
            self.watch_custom_toggle.setChecked(True)
            self._restart_folder_watching()
            self._update_watch_status()
    
    def _remove_custom_folder(self, folder_path: str):
        """Remove a custom folder from the watch list."""
        settings.remove_watch_custom_folder(folder_path)
        self._refresh_custom_folders_list()
        if not settings.watch_custom_folders:
            self.watch_custom_toggle.setChecked(False)
        self._restart_folder_watching()
        self._update_watch_status()
    
    def _refresh_custom_folders_list(self):
        """Refresh the custom folders list UI."""
        # Clear existing items
        while self.custom_folders_list_layout.count():
            item = self.custom_folders_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add folder items
        for folder_path in settings.watch_custom_folders:
            folder_row = QWidget()
            row_layout = QHBoxLayout(folder_row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(8)
            
            folder_label = QLabel(f"📁 {folder_path}")
            folder_label.setObjectName("customFolderLabel")
            row_layout.addWidget(folder_label, 1)
            
            remove_btn = QPushButton("✕")
            remove_btn.setObjectName("removeFolderBtn")
            remove_btn.setFixedSize(24, 24)
            remove_btn.setCursor(Qt.PointingHandCursor)
            remove_btn.setToolTip("Remove this folder")
            remove_btn.clicked.connect(lambda checked, fp=folder_path: self._remove_custom_folder(fp))
            row_layout.addWidget(remove_btn)
            
            self.custom_folders_list_layout.addWidget(folder_row)
    
    def _update_watch_status(self):
        """Update the watch status label."""
        watching = []
        if settings.watch_common_folders:
            watching.append("common folders")
        if settings.watch_custom_folders and self.watch_custom_toggle.isChecked():
            watching.append(f"{len(settings.watch_custom_folders)} custom folder(s)")
        
        if watching:
            self.watch_status_label.setText(f"✓ Watching: {', '.join(watching)}")
            self.watch_status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.watch_status_label.setText("Not watching any folders")
            self.watch_status_label.setStyleSheet("color: #7A7A90;")
    
    def _start_folder_watching(self):
        """Start watching folders for new files."""
        from PySide6.QtCore import QFileSystemWatcher
        import os
        
        # Initialize the background worker for auto-indexing
        if not hasattr(self, '_auto_index_worker'):
            self._auto_index_worker = AutoIndexWorker()
            self._auto_index_worker.file_indexed.connect(self._on_file_indexed)
            self._auto_index_worker.status_update.connect(self._on_auto_index_status)
        
        if not hasattr(self, '_folder_watcher'):
            self._folder_watcher = QFileSystemWatcher()
            self._folder_watcher.directoryChanged.connect(self._on_watched_folder_changed)
        
        if not hasattr(self, '_watched_folders'):
            self._watched_folders = {}
        
        folders_to_watch = []
        
        # Common folders
        if settings.watch_common_folders:
            home = Path.home()
            common_paths = [
                home / "Downloads",
                home / "Desktop",
                home / "Documents",
                home / "Pictures",
                home / "Videos",
                home / "Music",
            ]
            # OneDrive
            onedrive = home / "OneDrive"
            if onedrive.exists():
                common_paths.append(onedrive)
            
            for p in common_paths:
                if p.exists():
                    folders_to_watch.append(str(p))
        
        # Custom folders
        if self.watch_custom_toggle.isChecked():
            for folder in settings.watch_custom_folders:
                if Path(folder).exists():
                    folders_to_watch.append(folder)
        
        # Add to watcher
        for folder in folders_to_watch:
            if folder not in self._folder_watcher.directories():
                self._folder_watcher.addPath(folder)
                # Track current files
                try:
                    self._watched_folders[folder] = set(Path(folder).iterdir())
                except Exception:
                    self._watched_folders[folder] = set()
        
        logger.info(f"Started watching {len(folders_to_watch)} folders")
    
    def _restart_folder_watching(self):
        """Restart folder watching with updated settings."""
        self._stop_folder_watching()
        if settings.watch_common_folders or (self.watch_custom_toggle.isChecked() and settings.watch_custom_folders):
            self._start_folder_watching()
    
    def _stop_folder_watching(self):
        """Stop watching all folders."""
        if hasattr(self, '_folder_watcher'):
            dirs = self._folder_watcher.directories()
            if dirs:
                self._folder_watcher.removePaths(dirs)
        if hasattr(self, '_watched_folders'):
            self._watched_folders = {}
        logger.info("Stopped watching folders")
    
    def _start_downloads_watcher(self):
        """Start watching common folders for new files."""
        from PySide6.QtCore import QFileSystemWatcher
        from pathlib import Path
        import os
        
        home = Path.home()
        
        # Common folders to watch
        folder_names = ['Downloads', 'Desktop', 'Documents', 'Pictures', 'Videos', 'Music']
        folders_to_watch = []
        
        for name in folder_names:
            folder = home / name
            if folder.exists() and folder.is_dir():
                folders_to_watch.append(folder)
        
        # Also check for OneDrive folders
        for item in home.iterdir():
            if item.is_dir() and 'OneDrive' in item.name:
                folders_to_watch.append(item)
                # Also add common subfolders in OneDrive
                for name in ['Desktop', 'Documents', 'Pictures']:
                    subfolder = item / name
                    if subfolder.exists() and subfolder.is_dir():
                        folders_to_watch.append(subfolder)
        
        if not folders_to_watch:
            self.auto_index_status.setText("No common folders found to watch")
            return
        
        # Initialize the background worker for auto-indexing
        if not hasattr(self, '_auto_index_worker'):
            self._auto_index_worker = AutoIndexWorker()
            self._auto_index_worker.file_indexed.connect(self._on_file_indexed)
            self._auto_index_worker.status_update.connect(self._on_auto_index_status)
        
        if not hasattr(self, '_folder_watcher'):
            self._folder_watcher = QFileSystemWatcher(self)
            self._folder_watcher.directoryChanged.connect(self._on_watched_folder_changed)
        
        # Track known files per folder (so we only index NEW files)
        self._watched_folders = {}
        for folder in folders_to_watch:
            self._folder_watcher.addPath(str(folder))
            # Capture current files as "known" - these won't be indexed
            try:
                self._watched_folders[str(folder)] = set(folder.iterdir())
            except Exception:
                self._watched_folders[str(folder)] = set()
        
        logger.info(f"Started watching {len(folders_to_watch)} folders for new files")
        self.auto_index_status.setText(f"Monitoring {len(folders_to_watch)} folders...")
    
    def _stop_downloads_watcher(self):
        """Stop watching folders."""
        if hasattr(self, '_folder_watcher'):
            self._folder_watcher.removePaths(self._folder_watcher.directories())
            self._watched_folders = {}
            logger.info("Stopped watching folders")
    
    def _on_watched_folder_changed(self, path: str):
        """Handle changes in any watched folder."""
        from pathlib import Path
        
        if not hasattr(self, '_watched_folders'):
            return
        
        folder_path = Path(path)
        if str(folder_path) not in self._watched_folders:
            return
        
        try:
            current_files = set(folder_path.iterdir())
        except Exception:
            return
        
        known_files = self._watched_folders.get(str(folder_path), set())
        
        # Find new files (only files added AFTER we started watching)
        new_files = current_files - known_files
        
        # Update known files
        self._watched_folders[str(folder_path)] = current_files
        
        for new_file in new_files:
            if new_file.is_file():
                # Skip temporary/partial download files
                skip_extensions = {'.tmp', '.crdownload', '.part', '.partial', '.download'}
                if new_file.suffix.lower() in skip_extensions:
                    logger.debug(f"Skipping temporary file: {new_file.name}")
                    continue
                
                # Skip hidden files
                if new_file.name.startswith('.'):
                    continue
                
                logger.info(f"New file detected in {folder_path.name}: {new_file.name}")
                self.watch_status_label.setText(f"📥 Indexing: {new_file.name}")
                
                # Add to background worker queue (non-blocking)
                if hasattr(self, '_auto_index_worker'):
                    self._auto_index_worker.add_file(new_file)
    
    def _on_file_indexed(self, filename: str, status: str):
        """Handle file indexed signal from background worker."""
        if status == 'success':
            self.watch_status_label.setText(f"✓ Indexed: {filename}")
            self.watch_status_label.setStyleSheet("color: #4CAF50; font-size: 13px;")
            # Refresh file count
            self.refresh_debug_view()
        elif status == 'skipped':
            self.watch_status_label.setText(f"○ Already indexed: {filename}")
            self.watch_status_label.setStyleSheet("color: #7A7A90; font-size: 13px;")
        else:
            self.watch_status_label.setText(f"✗ Error: {filename}")
            self.watch_status_label.setStyleSheet("color: #ff5555; font-size: 13px;")
        
        # Reset status after delay
        def reset_status():
            self._update_watch_status()
        QTimer.singleShot(3000, reset_status)
    
    def _on_auto_index_status(self, message: str):
        """Handle status update from background worker."""
        self.auto_index_status.setText(message)
    
    def update_search_button_state(self):
        """Update search button enabled state."""
        # Enable if there's a query AND (index_path is set OR there are indexed files in DB)
        has_index_path = hasattr(self, 'index_path') and self.index_path is not None
        has_query = bool(self.search_input.text().strip())
        
        # Also check if database has any indexed files (for cases where user opens app with existing data)
        has_indexed_files = False
        try:
            from app.core.database import file_index
            has_indexed_files = file_index.get_file_count() > 0
        except Exception:
            pass
        
        self.search_button.setEnabled(has_query and (has_index_path or has_indexed_files))
    
    def _show_search_loading(self):
        """Show loading state on search bar (like ChatGPT/Claude)."""
        # Change button to loading spinner
        if hasattr(self, 'search_button'):
            self._original_btn_text = self.search_button.text()
            self.search_button.setText("⏳")
            self.search_button.setEnabled(False)
        
        # Change AI label to show searching
        if hasattr(self, 'ai_label'):
            self._original_ai_text = self.ai_label.text()
            self.ai_label.setText("Searching...")
            self.ai_label.setStyleSheet("color: #7C4DFF; font-size: 13px; font-weight: 500; background: transparent;")
        
        # Add pulsing effect to search container
        if hasattr(self, 'search_container'):
            self.search_container.setStyleSheet("""
                QWidget#searchContainerLarge {
                    border: 2px solid #7C4DFF;
                }
            """)
        
        # Process events to show the loading state immediately
        QApplication.processEvents()
    
    def _hide_search_loading(self):
        """Hide loading state and restore normal search bar."""
        # Restore button
        if hasattr(self, 'search_button') and hasattr(self, '_original_btn_text'):
            self.search_button.setText(self._original_btn_text)
            self.search_button.setEnabled(True)
        
        # Restore AI label
        if hasattr(self, 'ai_label') and hasattr(self, '_original_ai_text'):
            self.ai_label.setText(self._original_ai_text)
            self.ai_label.setStyleSheet("")  # Reset to QSS default
        
        # Remove pulsing effect
        if hasattr(self, 'search_container'):
            self.search_container.setStyleSheet("")  # Reset to QSS default
    
    def search_files(self):
        """Search for files with NLP parsing and filters."""
        query = self.search_input.text().strip()
        
        # Check if we have UI filters even without a query
        ui_type = self.type_filter.currentText()
        ui_date = self.date_filter.currentText()
        has_ui_filters = ui_type != "All Types" or ui_date != "Any Time"
        
        if not query and not has_ui_filters:
            return
        
        # Show loading state
        self._show_search_loading()
        
        self.status_bar.showMessage(f"Searching for: {query}" if query else "Browsing files...")
        
        # Parse query for natural language filters
        parsed = parse_query(query) if query else {'clean_query': '', 'date_filter': None, 'type_filter': None, 'date_range': (None, None), 'extensions': None}
        clean_query = parsed['clean_query']
        
        # UI filters already retrieved above
        
        # Determine type filter (NLP detected takes priority over UI)
        type_filter = parsed['type_filter']
        extensions = parsed['extensions']
        
        if type_filter:
            # NLP detected a type filter - it takes priority
            # Update UI dropdown to reflect the detected filter
            ui_name = FILTER_TO_UI_TYPE.get(type_filter)
            if ui_name:
                idx = self.type_filter.findText(ui_name)
                if idx >= 0:
                    self.type_filter.blockSignals(True)
                    self.type_filter.setCurrentIndex(idx)
                    self.type_filter.blockSignals(False)
            extensions = TYPE_EXTENSIONS.get(type_filter, [])
        elif ui_type != "All Types":
            # No NLP type filter - use UI dropdown selection
            type_filter = UI_TYPE_MAPPING.get(ui_type)
            extensions = TYPE_EXTENSIONS.get(type_filter, [])
        
        # Determine date filter (NLP detected takes priority over UI)
        date_filter = parsed['date_filter']
        date_start, date_end = parsed['date_range']
        specific_date = parsed.get('specific_date')  # For display purposes
        logger.info(f"[SEARCH] ui_date='{ui_date}', NLP date_filter='{date_filter}', specific_date='{specific_date}'")
        
        if date_filter:
            # NLP detected a date filter - it takes priority
            # Check if it's a specific date (parsed by dateparser)
            if date_filter.startswith('specific_date:'):
                # Specific date - date_start and date_end already set by parser
                # Reset UI dropdown since this is a custom date
                self.date_filter.blockSignals(True)
                self.date_filter.setCurrentIndex(0)  # "Any Time"
                self.date_filter.blockSignals(False)
                logger.info(f"[SEARCH] Specific date: {specific_date}, range={date_start} to {date_end}")
            elif date_filter.startswith('month:') or date_filter.startswith('year:'):
                # Month or year filter - date_start and date_end already set by parser
                # Reset UI dropdown since this is a custom date range
                self.date_filter.blockSignals(True)
                self.date_filter.setCurrentIndex(0)  # "Any Time"
                self.date_filter.blockSignals(False)
                logger.info(f"[SEARCH] Month/Year filter: {date_filter}, range={date_start} to {date_end}")
            elif date_filter.startswith('range:'):
                # Relative range filter (past N days, etc.) - date_start and date_end already set by parser
                # Reset UI dropdown since this is a custom range
                self.date_filter.blockSignals(True)
                self.date_filter.setCurrentIndex(0)  # "Any Time"
                self.date_filter.blockSignals(False)
                logger.info(f"[SEARCH] Range filter: {date_filter}, range={date_start} to {date_end}")
            else:
                # Standard date filter (today, yesterday, etc.)
                # Update UI dropdown to reflect the detected filter
                ui_name = FILTER_TO_UI_DATE.get(date_filter)
                if ui_name:
                    idx = self.date_filter.findText(ui_name)
                    if idx >= 0:
                        self.date_filter.blockSignals(True)
                        self.date_filter.setCurrentIndex(idx)
                        self.date_filter.blockSignals(False)
                # Use the NLP-detected date range (recalculate for standard filters)
                date_start, date_end = get_date_range(date_filter)
                logger.info(f"[SEARCH] NLP standard: date_filter='{date_filter}', date_start={date_start}, date_end={date_end}")
        elif ui_date != "Any Time":
            # No NLP date filter - use UI dropdown selection
            date_filter = UI_DATE_MAPPING.get(ui_date)
            date_start, date_end = get_date_range(date_filter) if date_filter else (None, None)
            logger.info(f"[SEARCH] UI filter: date_filter='{date_filter}', date_start={date_start}, date_end={date_end}")
        
        # Update filter status label
        filter_parts = []
        if type_filter:
            filter_parts.append(f"Type: {type_filter}")
        if date_filter:
            # Show user-friendly date label
            if specific_date:
                filter_parts.append(f"Date: {specific_date}")
            else:
                filter_parts.append(f"Date: {date_filter}")
        if filter_parts:
            self.filter_status_label.setText(f"Active: {', '.join(filter_parts)}")
        else:
            self.filter_status_label.setText("")
        
        # Determine search query - use empty string for date-only searches
        # Note: clean_query can be empty string "" for date-only searches, which is valid
        search_query = clean_query  # Use clean_query directly (can be empty string)
        is_date_only_search = (search_query == "") and (date_start or date_end)
        
        logger.info(f"[SEARCH] search_query='{search_query}', is_date_only={is_date_only_search}")
        
        # Perform search with filters
        results = search_service.search_files(
            search_query,  # Pass empty string for date-only searches
            limit=100,
            type_filter=type_filter,
            date_start=date_start,
            date_end=date_end,
            extensions=extensions
        )
        self._last_search_results = results  # cache for editing
        
        # Show parsed query debug info if available
        dbg = getattr(search_service, 'last_debug_info', '')
        if dbg:
            self.search_debug_label.setText(dbg)
        else:
            self.search_debug_label.setText("")
        
        # Switch to results view (hide hero, show compact search)
        self._switch_to_results_view()
        
        # Display results
        self.display_search_results(results)
        
        # Hide loading state
        self._hide_search_loading()
        
        # Update status message
        if is_date_only_search:
            date_label = specific_date if specific_date else date_filter
            self.status_bar.showMessage(f"Found {len(results)} files from {date_label}")
        else:
            self.status_bar.showMessage(f"Found {len(results)} results for '{query}'")
    
    def _on_filter_changed(self):
        """Re-run search when filter dropdowns change."""
        if self.search_input.text().strip():
            self.search_files()
    
    def _clear_filters(self):
        """Clear all search filters and reset dropdowns."""
        self.type_filter.blockSignals(True)
        self.date_filter.blockSignals(True)
        self.type_filter.setCurrentIndex(0)  # "All Types"
        self.date_filter.setCurrentIndex(0)  # "Any Time"
        self.type_filter.blockSignals(False)
        self.date_filter.blockSignals(False)
        self.filter_status_label.setText("")
        
        # Re-run search if there's a query
        if self.search_input.text().strip():
            self.search_files()
    
    def _create_separator(self) -> QWidget:
        """Create a vertical separator for the Quick Actions bar."""
        from PySide6.QtWidgets import QFrame
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #252535;")
        return sep
    
    def _get_selected_file_ids(self, source: str = 'search') -> List[int]:
        """Get file IDs for all CHECKED rows (checkbox) in the specified table."""
        if source == 'debug':
            table = self.debug_table
            data_cache = getattr(self, '_last_debug_files', [])
        else:
            table = self.search_results_table
            data_cache = getattr(self, '_last_search_results', [])
        
        file_ids = []
        for row in range(table.rowCount()):
            # Check if checkbox in column 0 is checked
            checkbox_item = table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                if row < len(data_cache):
                    file_id = data_cache[row].get('id')
                    if file_id:
                        file_ids.append(file_id)
        return file_ids
    
    def _get_selected_files(self, source: str = 'search') -> List[Dict[str, Any]]:
        """Get full file data for all CHECKED rows (checkbox) in the specified table."""
        if source == 'debug':
            table = self.debug_table
            data_cache = getattr(self, '_last_debug_files', [])
        else:
            table = self.search_results_table
            data_cache = getattr(self, '_last_search_results', [])
        
        files = []
        for row in range(table.rowCount()):
            # Check if checkbox in column 0 is checked
            checkbox_item = table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                if row < len(data_cache):
                    files.append(data_cache[row])
        return files
    
    def _on_selection_changed(self, item=None):
        """Update Quick Actions bar when checkbox changes in Search tab."""
        # Only react to checkbox column changes (column 0)
        if item is not None and item.column() != 0:
            return
        
        checked_count = 0
        for row in range(self.search_results_table.rowCount()):
            checkbox_item = self.search_results_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                checked_count += 1
        
        if checked_count > 0:
            self.selection_count_label.setText(f"{checked_count} file{'s' if checked_count != 1 else ''} selected")
            self.quick_actions_widget.setVisible(True)
        else:
            self.quick_actions_widget.setVisible(False)
    
    def _on_debug_selection_changed(self, item=None):
        """Update Quick Actions bar when checkbox changes in Indexed Files tab."""
        # Only react to checkbox column changes (column 0)
        if item is not None and item.column() != 0:
            return
        
        checked_count = 0
        for row in range(self.debug_table.rowCount()):
            checkbox_item = self.debug_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                checked_count += 1
        
        if checked_count > 0:
            self.debug_selection_count_label.setText(f"{checked_count} file{'s' if checked_count != 1 else ''} selected")
            self.debug_quick_actions_widget.setVisible(True)
        else:
            self.debug_quick_actions_widget.setVisible(False)
    
    def _action_select_all(self, source: str = 'search'):
        """Check all checkboxes in the specified table."""
        if source == 'debug':
            table = self.debug_table
        else:
            table = self.search_results_table
        
        table.blockSignals(True)
        for row in range(table.rowCount()):
            checkbox_item = table.item(row, 0)
            if checkbox_item:
                checkbox_item.setCheckState(Qt.Checked)
        table.blockSignals(False)
        
        # Manually trigger update
        if source == 'debug':
            self._on_debug_selection_changed()
        else:
            self._on_selection_changed()
    
    def _action_clear_selection(self, source: str = 'search'):
        """Uncheck all checkboxes in the specified table."""
        if source == 'debug':
            table = self.debug_table
        else:
            table = self.search_results_table
        
        table.blockSignals(True)
        for row in range(table.rowCount()):
            checkbox_item = table.item(row, 0)
            if checkbox_item:
                checkbox_item.setCheckState(Qt.Unchecked)
        table.blockSignals(False)
        
        # Manually trigger update
        if source == 'debug':
            self._on_debug_selection_changed()
        else:
            self._on_selection_changed()
    
    def _action_remove_from_index(self, source: str = 'search'):
        """Remove selected files from the index using background thread."""
        file_ids = self._get_selected_file_ids(source)
        if not file_ids:
            return
        
        # Confirm removal with modern dialog
        from app.ui.organize_page import ModernConfirmDialog
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Remove from Index",
            message=f"Remove {len(file_ids)} file(s) from the index?",
            info_text="The actual files will NOT be deleted from your PC.",
            yes_text="Remove",
            no_text="Cancel"
        )
        
        if not confirmed:
            return
        
        # Show modern progress dialog
        from app.ui.organize_page import ModernProgressDialog
        
        progress = ModernProgressDialog.create(
            self,
            title="Removing from Index",
            message=f"Removing {len(file_ids)} file(s) from the search index...",
            icon="🗑️",
            can_cancel=True,
            indeterminate=True
        )
        
        # Run in background thread
        self._batch_worker = BatchOperationWorker('remove', file_ids=file_ids)
        self._batch_worker.operation_completed.connect(
            lambda result: self._on_batch_operation_complete('remove', result, progress, source)
        )
        self._batch_worker.operation_error.connect(
            lambda err: self._on_batch_operation_error(err, progress)
        )
        self._batch_worker.start()
    
    def _action_reindex_selected(self, source: str = 'search'):
        """Re-index selected files using background thread."""
        files = self._get_selected_files(source)
        if not files:
            return
        
        file_paths = [f.get('file_path') for f in files if f.get('file_path')]
        if not file_paths:
            return
        
        from app.ui.organize_page import ModernConfirmDialog
        
        confirmed = ModernConfirmDialog.ask(
            self,
            title="Re-index Files",
            message=f"Re-index {len(file_paths)} file(s)?",
            info_text="This will refresh their metadata and AI analysis.",
            yes_text="Re-index",
            no_text="Cancel"
        )
        
        if not confirmed:
            return
        
        # Show modern progress dialog
        from app.ui.organize_page import ModernProgressDialog
        
        progress = ModernProgressDialog.create(
            self,
            title="Re-indexing Files",
            message=f"Refreshing metadata and AI analysis for {len(file_paths)} file(s)...",
            icon="🔄",
            can_cancel=True,
            indeterminate=False
        )
        progress.set_progress(0, len(file_paths), "Starting...")
        
        # Run in background thread
        self._batch_worker = BatchOperationWorker('reindex', file_paths=file_paths)
        self._batch_worker.progress_updated.connect(
            lambda curr, total, msg: self._on_batch_progress(progress, curr, total, msg)
        )
        self._batch_worker.operation_completed.connect(
            lambda result: self._on_batch_operation_complete('reindex', result, progress, source)
        )
        self._batch_worker.operation_error.connect(
            lambda err: self._on_batch_operation_error(err, progress)
        )
        self._batch_worker.start()
    
    def _action_add_tags(self, source: str = 'search'):
        """Add tags to selected files using background thread."""
        file_ids = self._get_selected_file_ids(source)
        if not file_ids:
            return
        
        # Show modern input dialog for tags
        from app.ui.organize_page import ModernInputDialog
        
        tags_text, accepted = ModernInputDialog.get_input(
            self,
            title="Add Tags",
            message=f"Enter tags to add to {len(file_ids)} file(s):",
            placeholder="Separate multiple tags with commas",
            icon="🏷️",
            ok_text="Add Tags",
            cancel_text="Cancel"
        )
        
        if not accepted or not tags_text.strip():
            return
        
        # Parse tags
        new_tags = [t.strip() for t in tags_text.split(',') if t.strip()]
        if not new_tags:
            return
        
        # Show modern progress dialog
        from app.ui.organize_page import ModernProgressDialog
        
        progress = ModernProgressDialog.create(
            self,
            title="Adding Tags",
            message=f"Adding {len(new_tags)} tag(s) to {len(file_ids)} file(s)...",
            icon="🏷️",
            can_cancel=False,
            indeterminate=True
        )
        
        # Run in background thread
        self._batch_worker = BatchOperationWorker('add_tags', file_ids=file_ids, extra_data={'tags': new_tags})
        self._batch_worker.operation_completed.connect(
            lambda result: self._on_batch_operation_complete('add_tags', result, progress, source, extra={'tags': new_tags})
        )
        self._batch_worker.operation_error.connect(
            lambda err: self._on_batch_operation_error(err, progress)
        )
        self._batch_worker.start()
    
    def _on_batch_progress(self, progress, current: int, total: int, message: str):
        """Update progress dialog during batch operation."""
        # Works with both ModernProgressDialog and QProgressDialog
        if hasattr(progress, 'set_progress'):
            progress.set_progress(current, total, message)
        else:
            progress.setMaximum(total)
            progress.setValue(current)
            progress.setLabelText(message)
            QApplication.processEvents()
    
    def _on_batch_operation_complete(self, operation: str, result: dict, progress: QProgressDialog, source: str, extra: dict = None):
        """Handle batch operation completion."""
        progress.close()
        
        from app.ui.organize_page import ModernInfoDialog
        
        if operation == 'remove':
            ModernInfoDialog.show_info(
                self,
                title="Remove Complete",
                message=f"Removed {result.get('removed', 0)} file(s) from index.",
                details=[f"Errors: {result.get('errors', 0)}"] if result.get('errors', 0) > 0 else None
            )
        elif operation == 'reindex':
            ModernInfoDialog.show_info(
                self,
                title="Re-index Complete",
                message=f"Updated {result.get('updated', 0)} file(s).",
                details=[
                    f"Not found: {result.get('not_found', 0)}",
                    f"Errors: {result.get('errors', 0)}"
                ] if (result.get('not_found', 0) > 0 or result.get('errors', 0) > 0) else None
            )
        elif operation == 'add_tags':
            tags = extra.get('tags', []) if extra else []
            ModernInfoDialog.show_info(
                self,
                title="Tags Added",
                message=f"Added tags to {result.get('updated', 0)} file(s).",
                details=[f"Tags: {', '.join(tags)}"],
                info_text=f"Errors: {result.get('errors', 0)}" if result.get('errors', 0) > 0 else ""
            )
        
        # Refresh views
        if self.search_input.text().strip():
            self.search_files()
        self.refresh_debug_view()
    
    def _on_batch_operation_error(self, error: str, progress: QProgressDialog):
        """Handle batch operation error."""
        progress.close()
        QMessageBox.critical(self, "Operation Failed", f"An error occurred:\n{error}")
    
    def _action_copy_paths(self, source: str = 'search'):
        """Copy file paths of selected files to clipboard."""
        files = self._get_selected_files(source)
        if not files:
            return
        
        paths = [f.get('file_path', '') for f in files if f.get('file_path')]
        
        if paths:
            clipboard = QApplication.clipboard()
            clipboard.setText('\n'.join(paths))
            self.status_bar.showMessage(f"Copied {len(paths)} file path(s) to clipboard")
    
    def _action_open_folders(self, source: str = 'search'):
        """Open containing folders of selected files."""
        files = self._get_selected_files(source)
        if not files:
            return
        
        # Get unique folder paths
        folders = set()
        for f in files:
            file_path = f.get('file_path')
            if file_path:
                folder = str(Path(file_path).parent)
                folders.add(folder)
        
        if len(folders) > 5:
            reply = QMessageBox.question(
                self,
                "Open Folders",
                f"This will open {len(folders)} different folders.\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        for folder in folders:
            try:
                if os.path.exists(folder):
                    os.startfile(folder)
            except Exception as e:
                logger.error(f"Error opening folder {folder}: {e}")
        
        self.status_bar.showMessage(f"Opened {len(folders)} folder(s)")
    
    def _action_export_list(self, source: str = 'search'):
        """Export selected files to CSV or TXT."""
        files = self._get_selected_files(source)
        if not files:
            return
        
        # Show save dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export File List",
            "",
            "CSV Files (*.csv);;Text Files (*.txt)"
        )
        
        if not file_path:
            return
        
        # Determine format
        file_format = 'csv' if file_path.endswith('.csv') or 'CSV' in selected_filter else 'txt'
        
        # Ensure correct extension
        if file_format == 'csv' and not file_path.endswith('.csv'):
            file_path += '.csv'
        elif file_format == 'txt' and not file_path.endswith('.txt'):
            file_path += '.txt'
        
        from app.core.file_operations import get_file_operations
        file_ops = get_file_operations()
        
        if file_ops.export_file_list(files, file_path, file_format):
            self.status_bar.showMessage(f"Exported {len(files)} files to {file_path}")
            
            # Ask to open file
            reply = QMessageBox.question(
                self,
                "Export Complete",
                f"Exported {len(files)} file(s) to:\n{file_path}\n\nOpen the file?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                try:
                    os.startfile(file_path)
                except Exception as e:
                    logger.error(f"Error opening export file: {e}")
        else:
            QMessageBox.warning(self, "Export Failed", "Failed to export file list.")
    
    def _switch_to_results_view(self):
        """Switch search page from landing state to results state with smooth animation."""
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
        
        # Skip if already in results view
        if hasattr(self, '_in_results_view') and self._in_results_view:
            return
        self._in_results_view = True
        
        # Show the results container
        if hasattr(self, 'results_container'):
            self.results_container.setVisible(True)
        
        # Create animation group for smooth transition
        self._transition_group = QParallelAnimationGroup(self)
        
        # Animate hero section height collapse
        if hasattr(self, 'hero_section') and self.hero_section.isVisible():
            hero_anim = QPropertyAnimation(self.hero_section, b"maximumHeight")
            hero_anim.setDuration(350)
            hero_anim.setStartValue(self.hero_section.height())
            hero_anim.setEndValue(0)
            hero_anim.setEasingCurve(QEasingCurve.OutCubic)
            hero_anim.finished.connect(lambda: self.hero_section.setVisible(False))
            self._transition_group.addAnimation(hero_anim)
        
        # Animate top spacer collapse
        if hasattr(self, 'hero_top_spacer') and self.hero_top_spacer.isVisible():
            spacer_anim = QPropertyAnimation(self.hero_top_spacer, b"maximumHeight")
            spacer_anim.setDuration(350)
            spacer_anim.setStartValue(self.hero_top_spacer.height())
            spacer_anim.setEndValue(0)
            spacer_anim.setEasingCurve(QEasingCurve.OutCubic)
            spacer_anim.finished.connect(lambda: self.hero_top_spacer.setVisible(False))
            self._transition_group.addAnimation(spacer_anim)
        
        # Animate bottom spacer collapse
        if hasattr(self, 'hero_bottom_spacer') and self.hero_bottom_spacer.isVisible():
            bottom_anim = QPropertyAnimation(self.hero_bottom_spacer, b"maximumHeight")
            bottom_anim.setDuration(350)
            bottom_anim.setStartValue(self.hero_bottom_spacer.height())
            bottom_anim.setEndValue(0)
            bottom_anim.setEasingCurve(QEasingCurve.OutCubic)
            bottom_anim.finished.connect(lambda: self.hero_bottom_spacer.setVisible(False))
            self._transition_group.addAnimation(bottom_anim)
        
        # Animate search bar shrink
        if hasattr(self, 'search_container'):
            search_anim = QPropertyAnimation(self.search_container, b"maximumHeight")
            search_anim.setDuration(300)
            search_anim.setStartValue(self.search_container.height())
            search_anim.setEndValue(56)
            search_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._transition_group.addAnimation(search_anim)
        
        # Show the compact top spacer
        if hasattr(self, 'search_top_spacer'):
            self.search_top_spacer.setVisible(True)
        
        # Start animation
        self._transition_group.start()
    
    def _switch_to_landing_view(self):
        """Switch search page back to landing state with hero content."""
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
        
        # Skip if already in landing view
        if not hasattr(self, '_in_results_view') or not self._in_results_view:
            return
        self._in_results_view = False
        
        # Hide the results container
        if hasattr(self, 'results_container'):
            self.results_container.setVisible(False)
        
        # Show hero elements first (set to visible with 0 height)
        if hasattr(self, 'hero_section'):
            self.hero_section.setVisible(True)
            self.hero_section.setMaximumHeight(0)
        if hasattr(self, 'hero_top_spacer'):
            self.hero_top_spacer.setVisible(True)
            self.hero_top_spacer.setMaximumHeight(0)
        if hasattr(self, 'hero_bottom_spacer'):
            self.hero_bottom_spacer.setVisible(True)
            self.hero_bottom_spacer.setMaximumHeight(0)
        
        # Hide the compact top spacer
        if hasattr(self, 'search_top_spacer'):
            self.search_top_spacer.setVisible(False)
        
        # Create animation group
        self._landing_group = QParallelAnimationGroup(self)
        
        # Animate hero section expand
        if hasattr(self, 'hero_section'):
            hero_anim = QPropertyAnimation(self.hero_section, b"maximumHeight")
            hero_anim.setDuration(350)
            hero_anim.setStartValue(0)
            hero_anim.setEndValue(150)
            hero_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._landing_group.addAnimation(hero_anim)
        
        # Animate spacers expand
        if hasattr(self, 'hero_top_spacer'):
            spacer_anim = QPropertyAnimation(self.hero_top_spacer, b"maximumHeight")
            spacer_anim.setDuration(350)
            spacer_anim.setStartValue(0)
            spacer_anim.setEndValue(16777215)  # QWIDGETSIZE_MAX
            spacer_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._landing_group.addAnimation(spacer_anim)
        
        if hasattr(self, 'hero_bottom_spacer'):
            bottom_anim = QPropertyAnimation(self.hero_bottom_spacer, b"maximumHeight")
            bottom_anim.setDuration(350)
            bottom_anim.setStartValue(0)
            bottom_anim.setEndValue(16777215)
            bottom_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._landing_group.addAnimation(bottom_anim)
        
        # Animate search bar expand
        if hasattr(self, 'search_container'):
            search_anim = QPropertyAnimation(self.search_container, b"maximumHeight")
            search_anim.setDuration(300)
            search_anim.setStartValue(56)
            search_anim.setEndValue(74)
            search_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._landing_group.addAnimation(search_anim)
        
        # Start animation
        self._landing_group.start()
    
    def display_search_results(self, results: List[Dict[str, Any]]):
        """Display search results in the simplified table (File Name, Folder, Size, Actions)."""
        # DIAGNOSTIC: Log signal blocking
        logger.warning(f"[DISPLAY] display_search_results START: {len(results)} results, blocking signals...")
        
        # Block signals to prevent itemChanged from firing during population
        self._populating_search_table = True
        self.search_results_table.blockSignals(True)
        
        self.search_results_table.setRowCount(len(results))
        actions_col = 4
        max_actions_width = self.search_results_table.columnWidth(actions_col)
        
        for row, result in enumerate(results):
            file_path = result.get('file_path', '') or ''
            
            # Ensure the row is tall enough for the action buttons (some styles ignore defaultSectionSize)
            try:
                self.search_results_table.setRowHeight(row, 46)
            except Exception:
                pass
            
            # Checkbox column (col 0)
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            checkbox_item.setCheckState(Qt.Unchecked)
            # Store file path in checkbox item's data for double-click access
            checkbox_item.setData(Qt.UserRole, file_path)
            self.search_results_table.setItem(row, 0, checkbox_item)
            
            # File name (col 1)
            name_item = QTableWidgetItem(result['file_name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, file_path)  # Store path for double-click
            self.search_results_table.setItem(row, 1, name_item)
            
            # Folder (col 2) - extract parent folder from path
            import os
            folder_path = os.path.dirname(file_path) if file_path else ''
            # Show just the last folder name for cleaner display
            folder_name = os.path.basename(folder_path) if folder_path else ''
            folder_item = QTableWidgetItem(folder_name)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemIsEditable)
            folder_item.setToolTip(folder_path)  # Full path on hover
            folder_item.setData(Qt.UserRole, file_path)
            self.search_results_table.setItem(row, 2, folder_item)
            
            # Size (col 3)
            size_item = QTableWidgetItem(result.get('size_formatted', 'Unknown'))
            size_item.setFlags(size_item.flags() & ~Qt.ItemIsEditable)
            size_item.setData(Qt.UserRole, file_path)
            self.search_results_table.setItem(row, 3, size_item)
            
            # Actions column (col 4) - Open and Copy Path buttons
            actions_widget = QWidget()
            actions_widget.setMinimumHeight(40)
            actions_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            actions_layout = QHBoxLayout(actions_widget)
            # Small vertical margins so button text doesn't get clipped by the cell viewport/borders
            actions_layout.setContentsMargins(4, 2, 4, 2)
            actions_layout.setSpacing(6)
            actions_layout.setAlignment(Qt.AlignVCenter)
            
            # Purple brand button style
            action_btn_style = """
                QPushButton {
                    background-color: #7C4DFF;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    background-color: #9575FF;
                }
            """
            
            btn_open = QPushButton("Open")
            btn_open.setToolTip("Open file with default app")
            btn_open.setFixedHeight(30)
            btn_open.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            btn_open.setStyleSheet(action_btn_style)
            
            btn_copy = QPushButton("Copy Path")
            btn_copy.setToolTip("Copy file path to clipboard")
            btn_copy.setFixedHeight(30)
            btn_copy.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            btn_copy.setStyleSheet(action_btn_style)
            
            actions_layout.addWidget(btn_open)
            actions_layout.addWidget(btn_copy)
            # No addStretch() - let Qt size naturally
            
            # Connect buttons to actions
            btn_open.clicked.connect(lambda _, p=file_path: self.open_file_in_os(p))
            btn_copy.clicked.connect(lambda _, p=file_path: self.copy_path_to_clipboard(p))
            
            self.search_results_table.setCellWidget(row, 4, actions_widget)

            # Grow Actions column to fully fit the two buttons (QTableWidget won't auto-measure cellWidget()).
            try:
                actions_layout.activate()
                actions_widget.adjustSize()
                # Extra safety padding: some Windows styles add frame/border pixels that sizeHint under-reports
                needed = actions_widget.sizeHint().width() + 28
                if needed > max_actions_width:
                    max_actions_width = needed
            except Exception:
                pass

        # Apply final Actions width once (prevents jitter while populating)
        try:
            self.search_results_table.setColumnWidth(actions_col, max_actions_width)
        except Exception:
            pass

        # Re-enable signals after population is complete
        self.search_results_table.blockSignals(False)
        self._populating_search_table = False
        logger.warning("[DISPLAY] display_search_results END: signals unblocked")

    def _on_search_result_double_click(self, row: int, col: int):
        """Open file when user double-clicks a search result row."""
        item = self.search_results_table.item(row, 1)  # Get file name item
        if item:
            file_path = item.data(Qt.UserRole)
            if file_path:
                self.open_file_in_os(file_path)

    def on_search_cell_changed(self, item: QTableWidgetItem) -> None:
        """Handle cell changes in search results table (simplified - no editable columns)."""
        # Avoid handling during table population
        if getattr(self, '_populating_search_table', False):
            return
        # Simplified table has no editable columns - ignore all changes
        pass

    def copy_path_to_clipboard(self, file_path: str) -> None:
        try:
            cb = QApplication.clipboard()
            cb.setText(file_path or "")
            self.status_bar.showMessage("Copied path to clipboard")
        except Exception as e:
            QMessageBox.critical(self, "Copy Error", f"Failed to copy path:\n{e}")

    def open_file_in_os(self, file_path: str) -> None:
        try:
            if not file_path:
                return
            # Prefer Qt for cross-platform support
            url = QUrl.fromLocalFile(file_path)
            if QDesktopServices.openUrl(url):
                return
            # Fallbacks
            if os.name == 'nt':
                os.startfile(file_path)  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', file_path])
            else:
                subprocess.Popen(['xdg-open', file_path])
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open file:\n{e}")
    
    # ==================== DRAG AND DROP ====================
    
    def dragEnterEvent(self, event):
        """Handle drag enter - show visual feedback."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # Update drop zone styling to show it's active (works with both old and new object names)
            self.drop_zone.setStyleSheet("""
                QWidget#dropZoneLarge, QWidget#dropZone {
                    border: 3px solid #7C4DFF;
                    border-radius: 24px;
                    background-color: rgba(124, 77, 255, 0.15);
                }
            """)
            # Count files/folders being dragged
            urls = event.mimeData().urls()
            count = len(urls)
            self._update_drop_zone(f"📥 Drop to index {count} item{'s' if count > 1 else ''}", "Release to start indexing")
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave - restore normal styling."""
        self._reset_drop_zone_style()
        event.accept()
    
    def dropEvent(self, event):
        """Handle file/folder drop - start indexing."""
        self._reset_drop_zone_style()
        
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            urls = event.mimeData().urls()
            paths = []
            
            for url in urls:
                if url.isLocalFile():
                    path = url.toLocalFile()
                    paths.append(path)
            
            if paths:
                self._handle_dropped_paths(paths)
        else:
            event.ignore()
    
    def _update_drop_zone(self, title: str, subtitle: str):
        """Update drop zone text labels."""
        if hasattr(self, 'drop_title') and self.drop_title:
            self.drop_title.setText(title)
        if hasattr(self, 'drop_subtitle') and self.drop_subtitle:
            self.drop_subtitle.setText(subtitle)
    
    def _on_drop_zone_clicked(self, event):
        """Handle click on drop zone to open folder dialog."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index",
            str(Path.home())
        )
        if folder:
            self._handle_dropped_paths([folder])
    
    def _clear_all_indexed_paths(self):
        """Clear all indexed files from the database."""
        from app.core.database import file_index
        
        reply = QMessageBox.question(
            self,
            "Clear All Indexed Files",
            "Are you sure you want to remove all files from the index?\n\n"
            "This will not delete your actual files, only the search index.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                file_index.clear_index()
                self.refresh_debug_view()
                self._update_view_files_button_count()
                self.status_bar.showMessage("Index cleared successfully", 3000)
                logger.info("Index cleared by user")
                
            except Exception as e:
                logger.error(f"Error clearing index: {e}")
                QMessageBox.warning(self, "Error", f"Failed to clear index: {e}")
    
    def _update_indexed_paths_list(self):
        """Update indexed paths - simplified, just refreshes the debug table."""
        # Section removed for simpler UI - just refresh the files table
        pass
    
    def _reset_drop_zone_style(self):
        """Reset drop zone to default styling."""
        # Use object name selector for consistency with QSS
        self.drop_zone.setStyleSheet("")  # Let QSS handle styling
        self._update_drop_zone("Add folder to index", "Drag and drop or click to browse")
    
    def _handle_dropped_paths(self, paths: list):
        """Handle dropped file/folder paths and start indexing."""
        # Collect all files to index
        files_to_index = []
        folders_to_index = []
        
        for path_str in paths:
            path = Path(path_str)
            if path.is_dir():
                folders_to_index.append(path)
            elif path.is_file():
                files_to_index.append(path)
        
        # Show confirmation
        msg_parts = []
        if folders_to_index:
            msg_parts.append(f"{len(folders_to_index)} folder{'s' if len(folders_to_index) > 1 else ''}")
        if files_to_index:
            msg_parts.append(f"{len(files_to_index)} file{'s' if len(files_to_index) > 1 else ''}")
        
        msg = f"Index {' and '.join(msg_parts)}?"
        
        # Modern styled confirmation dialog
        from PySide6.QtWidgets import QDialog, QFrame, QGraphicsDropShadowEffect
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        
        confirm_dialog = QDialog(self)
        confirm_dialog.setWindowTitle("Index Files")
        confirm_dialog.setModal(True)
        confirm_dialog.setFixedSize(400, 200)
        
        # Center on parent
        confirm_dialog.move(
            self.x() + (self.width() - 400) // 2,
            self.y() + (self.height() - 200) // 2
        )
        
        # Style the entire dialog
        confirm_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {c['surface']};
                border-radius: 16px;
            }}
        """)
        
        dialog_layout = QVBoxLayout(confirm_dialog)
        dialog_layout.setContentsMargins(28, 24, 28, 24)
        dialog_layout.setSpacing(16)
        
        # Icon + text header
        header_row = QHBoxLayout()
        header_row.setSpacing(14)
        icon_lbl = QLabel("📂")
        icon_lbl.setStyleSheet("font-size: 28px; background: transparent;")
        header_row.addWidget(icon_lbl)
        
        msg_label = QLabel(msg)
        msg_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            color: {c['text']};
            background: transparent;
        """)
        header_row.addWidget(msg_label)
        header_row.addStretch()
        dialog_layout.addLayout(header_row)
        
        # Subtitle
        sub_label = QLabel("Files will be scanned and made searchable.")
        sub_label.setStyleSheet(f"color: {c['text_muted']}; font-size: 13px; background: transparent;")
        dialog_layout.addWidget(sub_label)
        
        dialog_layout.addStretch()
        
        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {c['border_strong']};
                border-radius: 10px;
                color: {c['text_muted']};
                font-size: 14px;
                font-weight: 600;
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                border-color: #7C4DFF;
                color: #7C4DFF;
            }}
        """)
        cancel_btn.clicked.connect(confirm_dialog.reject)
        btn_row.addWidget(cancel_btn)
        
        confirm_btn = QPushButton("Index Now")
        confirm_btn.setMinimumHeight(40)
        confirm_btn.setMinimumWidth(120)
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C4DFF, stop:1 #9575FF);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9575FF, stop:1 #B39DFF);
            }
        """)
        confirm_btn.clicked.connect(confirm_dialog.accept)
        btn_row.addWidget(confirm_btn)
        
        dialog_layout.addLayout(btn_row)
        
        # Apply dark/light title bar
        from app.ui.theme_manager import apply_titlebar_theme
        confirm_dialog.show()
        apply_titlebar_theme(confirm_dialog)
        
        if confirm_dialog.exec() != QDialog.Accepted:
            return
        
        # Index folders
        if folders_to_index:
            if self.is_indexing:
                # Add all to queue
                for folder in folders_to_index:
                    self._add_to_index_queue(folder)
            else:
                # Start first, queue the rest
                first = folders_to_index[0]
                for folder in folders_to_index[1:]:
                    self._add_to_index_queue(folder)
                self.index_path = first
                self.index_label.setText(f"Index folder: {first}")
                self.index_button_action.setEnabled(True)
                self._start_indexing_path(first)
        
        # Index individual files
        if files_to_index and not folders_to_index:
            self._index_individual_files(files_to_index)
    
    def _index_individual_files(self, files: list):
        """Index individual dropped files."""
        from app.core.search import search_service
        
        total = len(files)
        
        # Show progress container
        if hasattr(self, 'index_progress_container'):
            self.index_progress_container.setVisible(True)
        
        # Update drop zone to show status
        self._update_drop_zone(f"Indexing {total} files...", "Drop more files to add to queue")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.index_percent_label.setVisible(True)
        self.index_progress_label.setVisible(True)
        
        indexed = 0
        for file_path in files:
            try:
                result = search_service.index_single_file(file_path)
                if not result.get('error'):
                    indexed += 1
                percent = int((indexed / total) * 100)
                self.progress_bar.setValue(indexed)
                self.index_percent_label.setText(f"{percent}%")
                self.index_progress_label.setText(f"Indexing file {indexed} of {total}")
                QApplication.processEvents()  # Keep UI responsive
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
        
        # Done - hide progress
        if hasattr(self, 'index_progress_container'):
            self.index_progress_container.setVisible(False)
        
        self.progress_bar.setVisible(False)
        self.index_percent_label.setVisible(False)
        self.index_progress_label.setVisible(False)
        
        # Reset drop zone
        self._update_drop_zone("Add folder to index", "Drag and drop or click to browse")
        
        QMessageBox.information(
            self,
            "Indexing Complete",
            f"Successfully indexed {indexed} of {total} files."
        )
        
        # Refresh views and update paths list
        self.refresh_debug_view()
        self._update_indexed_paths_list()
        stats = search_service.get_index_statistics()
        self.update_search_statistics(stats)
        self.search_button.setEnabled(True)
    
    # ==================== END DRAG AND DROP ====================
    
    def update_search_statistics(self, stats: Dict[str, Any]):
        """Update search statistics display."""
        if not stats:
            self.search_stats_label.setText("No files added yet")
            return
        
        total_files = stats.get('total_files', 0)
        files_with_ocr = stats.get('files_with_ocr', 0)
        total_size_mb = stats.get('total_size_mb', 0)
        
        stats_text = f"Indexed: {total_files} files ({files_with_ocr} with OCR) - {total_size_mb} MB"
        self.search_stats_label.setText(stats_text)

    def _update_usage_labels(self):
        """Update the usage indicator labels (main tab and overlay)."""
        try:
            from app.core.supabase_client import supabase_auth, INDEX_LIMIT_STARTER, INDEX_LIMIT_ULTRA
            
            # Get current usage
            usage = supabase_auth.get_index_usage()
            plan = supabase_auth.get_plan_tier()
            
            if plan == 'free':
                usage_text = "Sign in to track media indexing"
            else:
                limit = INDEX_LIMIT_ULTRA if plan == 'ultra' else INDEX_LIMIT_STARTER
                count = usage.get('count', 0)
                remaining = max(0, limit - count)
                usage_text = f"{count:,} / {limit:,} media files indexed • {remaining:,} remaining"
            
            # Update main tab label
            if hasattr(self, 'usage_label') and self.usage_label:
                from app.ui.theme_manager import get_theme_colors
                c = get_theme_colors()
                self.usage_label.setText(usage_text)
                self.usage_label.setStyleSheet(f"""
                    QLabel {{
                        font-size: 11px;
                        color: {c['text_muted']};
                        padding: 4px 0;
                    }}
                """)
            
            # Update overlay label if it exists
            if hasattr(self, '_overlay_usage_label') and self._overlay_usage_label:
                self._overlay_usage_label.setText(usage_text)
                
        except Exception as e:
            logger.debug(f"Could not update usage labels: {e}")
            if hasattr(self, 'usage_label') and self.usage_label:
                self.usage_label.setText("")

    # Debug functionality methods
    def _filter_indexed_files(self, search_text: str):
        """Filter the indexed files table based on search text."""
        if not hasattr(self, 'debug_table'):
            return
        
        search = search_text.lower().strip()
        
        # Columns to search:
        # 1: File Name, 2: Category, 5: Label, 6: Tags, 7: Caption, 11: Purpose, 14: File Path
        searchable_cols = [1, 2, 5, 6, 7, 11, 14]
        
        for row in range(self.debug_table.rowCount()):
            if not search:
                # Show all rows when search is empty
                self.debug_table.setRowHidden(row, False)
                continue
            
            # Check all searchable columns
            match = False
            for col in searchable_cols:
                item = self.debug_table.item(row, col)
                if item and search in item.text().lower():
                    match = True
                    break
            
            self.debug_table.setRowHidden(row, not match)
        
        # Update selection count label
        visible_count = sum(1 for row in range(self.debug_table.rowCount()) if not self.debug_table.isRowHidden(row))
        if search:
            self.debug_selection_count_label.setText(f"{visible_count} files found")
        else:
            self._on_debug_selection_changed()
    
    def refresh_debug_view(self):
        """Refresh the debug view with current database contents."""
        # Skip if debug table doesn't exist (hidden in MVP mode)
        if not hasattr(self, 'debug_table'):
            logger.warning("refresh_debug_view: debug_table not found")
            return
        
        if self.debug_table is None:
            logger.warning("refresh_debug_view: debug_table is None")
            return
            
        try:
            # Get all files from database using a direct query
            import sqlite3
            from app.core.database import file_index
            
            logger.info(f"refresh_debug_view: Querying database at {file_index.db_path}")
            
            with sqlite3.connect(file_index.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM files ORDER BY file_name")
                rows = cursor.fetchall()
            
            logger.info(f"refresh_debug_view: Found {len(rows)} files in database")
            
            # Store file data for Quick Actions
            self._last_debug_files = []
            
            # Update debug table
            self._populating_debug_table = True
            self.debug_table.blockSignals(True)
            self.debug_table.setRowCount(len(rows))
            
            for row_idx, row in enumerate(rows):
                # Build file dict for Quick Actions
                try:
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                except Exception:
                    meta = {}
                try:
                    tags_list = json.loads(row["tags"]) if row["tags"] else []
                except Exception:
                    tags_list = []
                
                file_dict = {
                    'id': row["id"],
                    'file_path': row["file_path"],
                    'file_name': row["file_name"],
                    'category': row["category"],
                    'file_size': row["file_size"],
                    'label': row["label"] if "label" in row.keys() else None,
                    'tags': tags_list,
                    'caption': row["caption"] if "caption" in row.keys() else None,
                    'metadata': meta,
                }
                self._last_debug_files.append(file_dict)
                
                # Checkbox column (col 0)
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                checkbox_item.setCheckState(Qt.Unchecked)
                self.debug_table.setItem(row_idx, 0, checkbox_item)
                
                # File name (col 1)
                name_item = QTableWidgetItem(row["file_name"])
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                try:
                    name_item.setData(Qt.UserRole, row["id"])
                except Exception:
                    pass
                self.debug_table.setItem(row_idx, 1, name_item)
                
                # Category (col 2)
                category_item = QTableWidgetItem(row["category"])
                category_item.setFlags(category_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 2, category_item)
                
                # Size (col 3)
                size_bytes = row["file_size"]
                size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0
                size_item = QTableWidgetItem(f"{size_mb} MB")
                size_item.setFlags(size_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 3, size_item)
                
                # Has OCR (col 4)
                has_ocr = bool(row["has_ocr"])
                ocr_item = QTableWidgetItem("Yes" if has_ocr else "No")
                ocr_item.setFlags(ocr_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 4, ocr_item)

                # Label (col 5)
                label = row["label"] if "label" in row.keys() else None
                label_item = QTableWidgetItem(label or '')
                label_item.setFlags((label_item.flags() | Qt.ItemIsEditable))
                self.debug_table.setItem(row_idx, 5, label_item)

                # Tags (col 6)
                tags_raw = row["tags"] if "tags" in row.keys() else None
                try:
                    tags_list = json.loads(tags_raw) if tags_raw else []
                    tags_text = ", ".join(tags_list)
                except Exception:
                    tags_text = tags_raw or ''
                tags_item = QTableWidgetItem(tags_text)
                tags_item.setFlags((tags_item.flags() | Qt.ItemIsEditable))
                self.debug_table.setItem(row_idx, 6, tags_item)

                # Caption (col 7)
                caption = row["caption"] if "caption" in row.keys() else None
                caption_item = QTableWidgetItem(caption or '')
                caption_item.setFlags((caption_item.flags() | Qt.ItemIsEditable))
                self.debug_table.setItem(row_idx, 7, caption_item)
                
                # OCR text preview (col 8)
                ocr_text = row["ocr_text"] or ""
                if ocr_text:
                    preview = ocr_text[:100] + "..." if len(ocr_text) > 100 else ocr_text
                else:
                    preview = "No OCR text"
                ocr_preview_item = QTableWidgetItem(preview)
                ocr_preview_item.setFlags(ocr_preview_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 8, ocr_preview_item)

                # AI source (col 9)
                ai_source = row["ai_source"] if "ai_source" in row.keys() else None
                ai_source_item = QTableWidgetItem(ai_source or '')
                ai_source_item.setFlags(ai_source_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 9, ai_source_item)

                # Vision score (col 10)
                try:
                    vscore = float(row["vision_confidence"]) if row["vision_confidence"] is not None else None
                except Exception:
                    vscore = None
                vscore_item = QTableWidgetItem(f"{vscore:.2f}" if vscore is not None else '')
                vscore_item.setFlags(vscore_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 10, vscore_item)

                # Purpose (col 11)
                purpose_item = QTableWidgetItem((meta.get('purpose') or ''))
                purpose_item.setFlags((purpose_item.flags() | Qt.ItemIsEditable))
                self.debug_table.setItem(row_idx, 11, purpose_item)

                # Suggested filename (col 12)
                sfile_item = QTableWidgetItem((meta.get('suggested_filename') or ''))
                sfile_item.setFlags((sfile_item.flags() | Qt.ItemIsEditable))
                self.debug_table.setItem(row_idx, 12, sfile_item)

                # Detected text (col 13)
                dtxt = meta.get('detected_text') or ''
                if dtxt:
                    dtxt_preview = dtxt[:100] + "..." if len(dtxt) > 100 else dtxt
                else:
                    dtxt_preview = ''
                dtxt_item = QTableWidgetItem(dtxt_preview)
                dtxt_item.setFlags(dtxt_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 13, dtxt_item)
                
                # File path (col 14)
                file_path_val = row["file_path"] or ""
                path_item = QTableWidgetItem(file_path_val)
                path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
                self.debug_table.setItem(row_idx, 14, path_item)
                
                # Actions (col 15)
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(6, 2, 6, 2)  # Tight vertical margins
                actions_layout.setSpacing(12)  # More space between buttons
                actions_layout.setAlignment(Qt.AlignVCenter)
                
                btn_style = """
                    QPushButton {
                        background-color: #7C4DFF;
                        color: white;
                        font-size: 11px;
                        font-weight: bold;
                        border: none;
                        border-radius: 5px;
                        padding: 2px 10px;
                        min-height: 22px;
                        max-height: 22px;
                    }
                    QPushButton:hover {
                        background-color: #9575FF;
                    }
                """
                
                btn_copy = QPushButton("Copy")
                btn_open = QPushButton("Open")
                btn_copy.setFixedHeight(22)
                btn_open.setFixedHeight(22)
                btn_copy.setMinimumWidth(52)
                btn_open.setMinimumWidth(52)
                btn_copy.setStyleSheet(btn_style)
                btn_open.setStyleSheet(btn_style)
                btn_copy.setToolTip("Copy file path to clipboard")
                btn_open.setToolTip("Open file with default app")
                actions_layout.addWidget(btn_copy)
                actions_layout.addWidget(btn_open)
                self.debug_table.setCellWidget(row_idx, 15, actions_widget)
                
                # Set row height to fit buttons properly
                self.debug_table.setRowHeight(row_idx, 48)
                
                # Connect actions
                btn_copy.clicked.connect(lambda _, p=file_path_val: self.copy_path_to_clipboard(p))
                btn_open.clicked.connect(lambda _, p=file_path_val: self.open_file_in_os(p))
            
            self.debug_table.blockSignals(False)
            self._populating_debug_table = False
            self.debug_info_label.setText(f"Showing {len(rows)} indexed files")
            self.status_bar.showMessage(f"Files view refreshed - {len(rows)} files shown")
            
            # Force table to update and scroll to top
            self.debug_table.scrollToTop()
            self.debug_table.viewport().update()
            
            logger.info(f"refresh_debug_view: Table updated with {len(rows)} rows")
            
            # Update indexed paths list in sidebar
            self._update_indexed_paths_list()
            
            # Update the View Files button count
            if hasattr(self, 'view_files_btn'):
                self.view_files_btn.setText(f"View Indexed Files ({len(rows)})")
            
        except Exception as e:
            QMessageBox.critical(self, "Debug Error", f"Error refreshing debug view:\n{e}")
            self.debug_info_label.setText("Error loading debug data")
    
    def clear_index(self):
        """Clear the search index."""
        reply = QMessageBox.question(
            self, "Clear Index",
            "Are you sure you want to clear the entire search index?\n\n"
            "This will remove all indexed files and OCR data.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                search_service.index.clear_index()
                self.refresh_debug_view()
                self.update_search_statistics({})
                self.search_button.setEnabled(False)
                self.status_bar.showMessage("Search index cleared")
            except Exception as e:
                QMessageBox.critical(self, "Clear Error", f"Error clearing index:\n{e}")

    def dump_active_dialog_tree(self) -> None:
        """Debug helper: dump the active window's controls (UIA and win32) to logs."""
        try:
            from pywinauto import Desktop
            logger.info("[QS] --- Dumping active dialog tree (UIA) ---")
            try:
                win = Desktop(backend='uia').get_active()
                if win:
                    logger.info("[QS] UIA Active: '%s' class='%s'", win.window_text(), getattr(win.element_info, 'class_name', '?'))
                    # Dump buttons and edits
                    for btn in win.descendants(control_type='Button')[:50]:
                        try:
                            r = btn.rectangle();
                            logger.info("[QS] UIA Button name='%s' id='%s' rect=%s", btn.window_text(), getattr(btn.element_info, 'automation_id', ''), (r.left, r.top, r.right, r.bottom))
                        except Exception:
                            pass
                    for ed in win.descendants(control_type='Edit')[:50]:
                        try:
                            r = ed.rectangle();
                            logger.info("[QS] UIA Edit name='%s' id='%s' rect=%s", ed.window_text(), getattr(ed.element_info, 'automation_id', ''), (r.left, r.top, r.right, r.bottom))
                        except Exception:
                            pass
                else:
                    logger.info("[QS] UIA: no active window")
            except Exception:
                logger.info("[QS] UIA dump failed", exc_info=True)
            logger.info("[QS] --- Dumping active dialog tree (win32) ---")
            try:
                winw = Desktop(backend='win32').get_active()
                if winw:
                    logger.info("[QS] win32 Active: '%s' class='%s'", winw.window_text(), getattr(winw.element_info, 'class_name', '?'))
                    for btn in winw.descendants(class_name='Button')[:50]:
                        try:
                            r = btn.rectangle(); logger.info("[QS] win32 Button name='%s' rect=%s", btn.window_text(), (r.left, r.top, r.right, r.bottom))
                        except Exception:
                            pass
                    for ed in winw.descendants(class_name='Edit')[:50]:
                        try:
                            r = ed.rectangle(); logger.info("[QS] win32 Edit name='%s' rect=%s", ed.window_text(), (r.left, r.top, r.right, r.bottom))
                        except Exception:
                            pass
                else:
                    logger.info("[QS] win32: no active window")
            except Exception:
                logger.info("[QS] win32 dump failed", exc_info=True)
        except Exception:
            logger.info("[QS] dump_active_dialog_tree outer failed", exc_info=True)
    
    def debug_comprehensive_state(self) -> None:
        """Phase 4: Debug helper for comprehensive system state logging."""
        try:
            from app.ui.win_hotkey import log_system_state
            
            logger.info("[QS] === MANUAL DEBUG TRIGGER (Ctrl+Alt+S) ===")
            
            # Log comprehensive system state
            log_system_state(logger, "[QS]")
            
            # If quick search overlay has saved state, log that too
            overlay = getattr(self, 'quick_overlay', None)
            if overlay and overlay.has_valid_saved_state():
                logger.info("[QS] Quick Search Overlay has saved state:")
                overlay.log_debug_target_window()
            else:
                logger.info("[QS] No saved state in Quick Search Overlay")
            
            # Log current autofill settings
            from app.core.settings import settings
            logger.info(f"[QS] Auto-paste: {settings.quick_search_autopaste}")
            logger.info(f"[QS] Auto-confirm: {settings.quick_search_auto_confirm}")
            logger.info(f"[QS] Shortcut: {settings.quick_search_shortcut}")
            
            logger.info("[QS] === END MANUAL DEBUG ===")
            self.status_bar.showMessage("Debug state logged - check console/logs")
            
        except Exception as e:
            logger.error(f"[QS] Error in debug_comprehensive_state: {e}")
            self.status_bar.showMessage("Debug logging failed")

    def _run_background_db_cleanup(self):
        """Run database cleanup in background to remove stale entries."""
        try:
            from app.core.database import file_index
            
            logger.info("Starting background database cleanup...")
            
            # Run cleanup in a thread to avoid blocking UI
            def do_cleanup():
                try:
                    stats = file_index.cleanup_stale_entries()
                    if stats['removed'] > 0:
                        logger.info(f"Database cleanup complete: removed {stats['removed']} stale entries")
                    else:
                        logger.debug("Database cleanup complete: no stale entries found")
                except Exception as e:
                    logger.error(f"Database cleanup error: {e}")
            
            # Run in thread
            import threading
            cleanup_thread = threading.Thread(target=do_cleanup, daemon=True)
            cleanup_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start database cleanup: {e}")
