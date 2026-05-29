"""
Interactive Onboarding Overlay for Filect
A floating panel that guides users through the app's features
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QWidget, QGraphicsDropShadowEffect, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QPoint, QTimer, QPropertyAnimation, QRect, QRectF, QEasingCurve, Property, QPointF
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen, QBrush, QPainterPath, QRegion, QLinearGradient
from datetime import datetime, timedelta
import random
import math


class OnboardingAnimation(QWidget):
    """Animated illustration widget for onboarding steps"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(440, 130)
        self._step = 0
        self._frame = 0
        self._max_frames = 300  # 5 seconds at 60fps then loops
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(16)  # ~60fps
        
        # Accent colors (same in both themes)
        self._purple = QColor("#7C4DFF")
        self._purple_light = QColor("#B39DDB")
        self._gold = QColor("#FFD700")
        self._green = QColor("#4CAF50")
        self._blue = QColor("#42A5F5")
        self._pink = QColor("#FF69B4")
        self._cyan = QColor("#00CED1")
        # Theme-dependent colors
        self._update_theme_colors()
    
    def _update_theme_colors(self):
        """Set theme-dependent colors for animations."""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self._purple_bg = QColor(c['surface'])
        self._white = QColor(c['text'])
        self._dark = QColor(c['text'])
        self._gray = QColor(c['text_disabled'])
        self._card_bg = QColor(c['border_strong'])
        self._surface = QColor(c['card'])

    def set_step(self, step_index):
        """Switch to a different step animation"""
        self._step = step_index
        self._frame = 0
        if not self._timer.isActive():
            self._timer.start()
        self.update()
    
    def stop(self):
        self._timer.stop()
    
    def _tick(self):
        self._frame += 1
        if self._frame >= self._max_frames:
            self._frame = 0  # Loop
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw rounded background
        bg = QPainterPath()
        bg.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 12, 12)
        painter.fillPath(bg, QColor("#16161F"))
        
        # Border
        painter.setPen(QPen(QColor("#252535"), 1.5))
        painter.drawRoundedRect(QRectF(0.5, 0.5, self.width()-1, self.height()-1), 12, 12)
        
        if self._step == 0:
            self._draw_welcome(painter)
        elif self._step == 1:
            self._draw_search(painter)
        elif self._step == 2:
            self._draw_organize(painter)
        elif self._step == 3:
            self._draw_auto_organize(painter)
        elif self._step == 4:
            self._draw_index(painter)
        elif self._step == 5:
            self._draw_settings(painter)
        elif self._step == 6:
            self._draw_ready(painter)
        
        painter.end()

    # ── Step 0: Welcome ──────────────────────────────
    def _draw_welcome(self, p: QPainter):
        cx, cy = self.width() / 2, self.height() / 2
        t = self._frame / 60.0  # time in seconds
        
        # Pulsing app icon silhouette
        pulse = 1.0 + 0.08 * math.sin(t * 3)
        size = 40 * pulse
        
        # Folder icon
        p.save()
        p.translate(cx, cy - 8)
        p.scale(pulse, pulse)
        p.setPen(Qt.NoPen)
        p.setBrush(self._purple)
        # Folder tab
        p.drawRoundedRect(QRectF(-20, -18, 16, 8), 3, 3)
        # Folder body
        p.drawRoundedRect(QRectF(-22, -12, 44, 30), 4, 4)
        # AI sparkle inside
        p.setBrush(self._white)
        sparkle_alpha = int(180 + 75 * math.sin(t * 5))
        sparkle_color = QColor(255, 255, 255, sparkle_alpha)
        p.setBrush(sparkle_color)
        # Draw a 4-point star
        star_size = 8 + 2 * math.sin(t * 4)
        self._draw_star(p, 0, 3, star_size)
        p.restore()
        
        # Floating sparkles around
        for i in range(5):
            angle = t * 1.2 + i * (2 * math.pi / 5)
            radius = 50 + 8 * math.sin(t * 2 + i)
            sx = cx + radius * math.cos(angle)
            sy = cy + radius * math.sin(angle) - 5
            alpha = int(100 + 100 * math.sin(t * 3 + i * 1.5))
            color = QColor(self._purple)
            color.setAlpha(alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            s = 3 + 2 * math.sin(t * 4 + i)
            p.drawEllipse(QPointF(sx, sy), s, s)
        
        # "AI File Organizer" text below
        p.setPen(self._purple)
        font = QFont("Segoe UI", 11, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, cy + 30, self.width(), 25), Qt.AlignCenter, "Filect")
    
    def _draw_star(self, p: QPainter, cx, cy, size):
        """Draw a 4-point sparkle star"""
        path = QPainterPath()
        path.moveTo(cx, cy - size)
        path.lineTo(cx + size * 0.25, cy - size * 0.25)
        path.lineTo(cx + size, cy)
        path.lineTo(cx + size * 0.25, cy + size * 0.25)
        path.lineTo(cx, cy + size)
        path.lineTo(cx - size * 0.25, cy + size * 0.25)
        path.lineTo(cx - size, cy)
        path.lineTo(cx - size * 0.25, cy - size * 0.25)
        path.closeSubpath()
        p.drawPath(path)

    # ── Step 1: Smart Search ─────────────────────────
    def _draw_search(self, p: QPainter):
        t = self._frame / 60.0
        w, h = self.width(), self.height()
        
        # Draw file cards on the left
        file_colors = [self._blue, self._green, self._pink, self._purple_light, self._cyan]
        file_labels = ["📄", "🖼️", "📊", "📝", "🎵"]
        
        for i in range(5):
            fx = 30 + i * 56
            fy = 25
            
            # File card
            p.setPen(Qt.NoPen)
            card_color = QColor("#252535")
            p.setBrush(card_color)
            p.drawRoundedRect(QRectF(fx, fy, 46, 56), 6, 6)
            
            # File color bar on top
            p.setBrush(file_colors[i])
            p.drawRoundedRect(QRectF(fx, fy, 46, 10), 6, 6)
            p.drawRect(QRectF(fx, fy + 5, 46, 5))
            
            # File content lines
            p.setBrush(QColor("#3A3A4A"))
            p.drawRoundedRect(QRectF(fx + 6, fy + 16, 34, 3), 1, 1)
            p.drawRoundedRect(QRectF(fx + 6, fy + 23, 26, 3), 1, 1)
            p.drawRoundedRect(QRectF(fx + 6, fy + 30, 30, 3), 1, 1)
            
            # Icon
            font = QFont("Segoe UI Emoji", 10)
            p.setFont(font)
            p.setPen(self._dark)
            p.drawText(QRectF(fx, fy + 36, 46, 18), Qt.AlignCenter, file_labels[i])
        
        # Magnifying glass sweeping across
        scan_x = 30 + (t * 40) % (5 * 56 + 20)
        
        # Scan beam (vertical line)
        beam_alpha = int(120 + 60 * math.sin(t * 8))
        beam_color = QColor(self._purple)
        beam_color.setAlpha(beam_alpha)
        p.setPen(QPen(beam_color, 2))
        p.drawLine(QPointF(scan_x, 20), QPointF(scan_x, 85))
        
        # Glow on scanned area
        grad = QLinearGradient(scan_x - 30, 0, scan_x, 0)
        grad.setColorAt(0, QColor(124, 77, 255, 0))
        grad.setColorAt(1, QColor(124, 77, 255, 40))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRect(QRectF(scan_x - 30, 20, 30, 65))
        
        # Magnifying glass icon
        mg_x = scan_x + 5
        mg_y = 40
        p.setPen(QPen(self._purple, 3))
        p.setBrush(QColor(124, 77, 255, 30))
        p.drawEllipse(QPointF(mg_x, mg_y), 14, 14)
        # Handle
        p.setPen(QPen(self._purple, 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(mg_x + 10, mg_y + 10), QPointF(mg_x + 18, mg_y + 18))
        
        # "Found!" label appears when glass is over a file
        file_index = int((scan_x - 30) / 56)
        if 0 <= file_index < 5:
            # Highlight the found file
            hx = 30 + file_index * 56
            highlight_alpha = int(80 + 40 * math.sin(t * 6))
            p.setPen(QPen(self._purple, 2))
            p.setBrush(QColor(124, 77, 255, highlight_alpha))
            p.drawRoundedRect(QRectF(hx - 2, 23, 50, 60), 6, 6)
        
        # Search bar at bottom
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#252535"))
        bar_rect = QRectF(60, 95, w - 120, 26)
        p.drawRoundedRect(bar_rect, 13, 13)
        p.setPen(QPen(self._purple, 1.5))
        p.drawRoundedRect(bar_rect, 13, 13)
        
        # Typing animation in search bar
        query = "vacation photos"
        chars_shown = min(len(query), int(t * 3) % (len(query) + 8))
        if chars_shown > len(query):
            chars_shown = len(query)
        display_text = query[:chars_shown]
        
        # Blinking cursor
        cursor = "│" if int(t * 3) % 2 == 0 else ""
        
        p.setPen(self._dark)
        font = QFont("Segoe UI", 9)
        p.setFont(font)
        p.drawText(bar_rect.adjusted(12, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, f"🔍 {display_text}{cursor}")

    # ── Step 2: Organize Files ───────────────────────
    def _draw_organize(self, p: QPainter):
        t = self._frame / 60.0
        w, h = self.width(), self.height()
        cx = w / 2
        
        # Phase: 0-2s scattered, 2-4s sliding into folders
        phase = (t % 5)
        
        # Target folders on the right
        folders = [
            {"label": "Documents", "color": self._blue, "y": 20},
            {"label": "Photos", "color": self._green, "y": 55},
            {"label": "Music", "color": self._pink, "y": 90},
        ]
        
        folder_x = w - 120
        for f in folders:
            # Folder shape
            p.setPen(Qt.NoPen)
            p.setBrush(f["color"])
            p.drawRoundedRect(QRectF(folder_x, f["y"], 10, 5), 2, 2)
            p.drawRoundedRect(QRectF(folder_x - 2, f["y"] + 4, 100, 28), 4, 4)
            # Label
            p.setPen(self._white)
            font = QFont("Segoe UI", 8, QFont.Bold)
            p.setFont(font)
            p.drawText(QRectF(folder_x - 2, f["y"] + 4, 100, 28), Qt.AlignCenter, f["label"])
        
        # Scattered files on the left
        files = [
            {"icon": "📄", "target_folder": 0, "start_x": 30, "start_y": 15},
            {"icon": "🖼️", "target_folder": 1, "start_x": 80, "start_y": 50},
            {"icon": "📄", "target_folder": 0, "start_x": 50, "start_y": 80},
            {"icon": "🎵", "target_folder": 2, "start_x": 120, "start_y": 30},
            {"icon": "🖼️", "target_folder": 1, "start_x": 140, "start_y": 75},
            {"icon": "📄", "target_folder": 0, "start_x": 100, "start_y": 95},
        ]
        
        for i, f in enumerate(files):
            target_f = folders[f["target_folder"]]
            target_x = folder_x + 40
            target_y = target_f["y"] + 10
            
            # Animation progress for this file (staggered)
            file_phase = max(0, min(1, (phase - 1.5 - i * 0.3) / 0.8))
            # Ease out
            file_phase = 1 - (1 - file_phase) ** 3
            
            curr_x = f["start_x"] + (target_x - f["start_x"]) * file_phase
            curr_y = f["start_y"] + (target_y - f["start_y"]) * file_phase
            
            # File card
            p.setPen(Qt.NoPen)
            card_alpha = 255 if file_phase < 0.95 else int(255 * (1 - (file_phase - 0.95) / 0.05) * 0.5 + 128)
            card_color = QColor("#252535")
            card_color.setAlpha(card_alpha)
            p.setBrush(card_color)
            size = 28 - 8 * file_phase  # Shrink as it enters folder
            p.drawRoundedRect(QRectF(curr_x - size/2, curr_y - size/2, size, size), 4, 4)
            
            # Icon
            if file_phase < 0.9:
                font = QFont("Segoe UI Emoji", max(6, int(10 - 4 * file_phase)))
                p.setFont(font)
                p.setPen(self._dark)
                p.drawText(QRectF(curr_x - size/2, curr_y - size/2, size, size), Qt.AlignCenter, f["icon"])
        
        # Arrow in the middle
        arrow_alpha = int(150 + 80 * math.sin(t * 3))
        arrow_color = QColor(self._purple)
        arrow_color.setAlpha(arrow_alpha)
        p.setPen(QPen(arrow_color, 3, Qt.SolidLine, Qt.RoundCap))
        mid_x = w / 2 + 10
        mid_y = h / 2
        p.drawLine(QPointF(mid_x - 20, mid_y), QPointF(mid_x + 20, mid_y))
        # Arrow head
        p.drawLine(QPointF(mid_x + 20, mid_y), QPointF(mid_x + 12, mid_y - 8))
        p.drawLine(QPointF(mid_x + 20, mid_y), QPointF(mid_x + 12, mid_y + 8))

    # ── Step 3: Auto-Organize ────────────────────────
    def _draw_auto_organize(self, p: QPainter):
        t = self._frame / 60.0
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        
        # Clock icon
        p.save()
        p.translate(cx - 80, cy - 5)
        clock_color = self._purple
        p.setPen(QPen(clock_color, 2.5))
        p.setBrush(QColor(124, 77, 255, 30))
        p.drawEllipse(QPointF(0, 0), 22, 22)
        # Clock hands
        sec_angle = t * 60  # Fast second hand
        min_angle = t * 5
        # Minute hand
        p.setPen(QPen(self._purple, 2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(0, 0), QPointF(12 * math.sin(math.radians(min_angle)), -12 * math.cos(math.radians(min_angle))))
        # Second hand
        p.setPen(QPen(self._gold, 1.5, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(0, 0), QPointF(16 * math.sin(math.radians(sec_angle)), -16 * math.cos(math.radians(sec_angle))))
        # Center dot
        p.setPen(Qt.NoPen)
        p.setBrush(self._purple)
        p.drawEllipse(QPointF(0, 0), 3, 3)
        p.restore()
        
        # Files appearing and auto-sorting
        file_cycle = t % 3  # Every 3 seconds a new file
        
        # Conveyor belt effect - file slides in from left and goes into folder
        file_x = cx - 30 + min(file_cycle / 2.0, 1.0) * 160
        file_alpha = 255
        if file_cycle > 2.5:
            file_alpha = int(255 * (3 - file_cycle) / 0.5)
        
        p.setPen(Qt.NoPen)
        fc = QColor("#252535")
        fc.setAlpha(file_alpha)
        p.setBrush(fc)
        p.drawRoundedRect(QRectF(file_x - 14, cy - 14, 28, 28), 5, 5)
        font = QFont("Segoe UI Emoji", 10)
        p.setFont(font)
        p.setPen(QColor(232, 232, 240, file_alpha))
        p.drawText(QRectF(file_x - 14, cy - 14, 28, 28), Qt.AlignCenter, "📄")
        
        # Dotted path
        p.setPen(QPen(QColor(124, 77, 255, 80), 1.5, Qt.DashLine))
        p.drawLine(QPointF(cx - 30, cy), QPointF(cx + 130, cy))
        
        # Target folder
        fx = cx + 130
        p.setPen(Qt.NoPen)
        p.setBrush(self._green)
        p.drawRoundedRect(QRectF(fx, cy - 20, 10, 5), 2, 2)
        p.drawRoundedRect(QRectF(fx - 2, cy - 16, 55, 32), 4, 4)
        p.setPen(self._white)
        font = QFont("Segoe UI", 7, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(fx - 2, cy - 16, 55, 32), Qt.AlignCenter, "Auto\nSorted")
        
        # "⚡ Automatic" label
        p.setPen(self._purple)
        font = QFont("Segoe UI", 9, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, h - 28, w, 20), Qt.AlignCenter, "⚡ Set it and forget it")

    # ── Step 4: Index Files ──────────────────────────
    def _draw_index(self, p: QPainter):
        t = self._frame / 60.0
        w, h = self.width(), self.height()
        
        # Grid of files
        cols, rows = 8, 3
        cell_w, cell_h = 40, 30
        grid_x = (w - cols * cell_w) / 2
        grid_y = 15
        
        # Scanning progress
        scan_progress = (t * 0.4) % 1.2  # 0 to 1.2 (extra for completion)
        scanned_cells = int(scan_progress * cols * rows)
        
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                x = grid_x + col * cell_w + 4
                y = grid_y + row * cell_h + 2
                
                is_scanned = idx < scanned_cells
                
                # File card
                p.setPen(Qt.NoPen)
                if is_scanned:
                    card_bg = QColor(124, 77, 255, 40)
                    border_color = self._purple
                else:
                    card_bg = QColor("#252535")
                    border_color = QColor("#3A3A4A")
                
                p.setBrush(card_bg)
                p.drawRoundedRect(QRectF(x, y, cell_w - 8, cell_h - 4), 4, 4)
                p.setPen(QPen(border_color, 1))
                p.drawRoundedRect(QRectF(x, y, cell_w - 8, cell_h - 4), 4, 4)
                
                # Checkmark for scanned
                if is_scanned:
                    p.setPen(QPen(self._green, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    cx_c = x + (cell_w - 8) / 2
                    cy_c = y + (cell_h - 4) / 2
                    p.drawLine(QPointF(cx_c - 5, cy_c), QPointF(cx_c - 1, cy_c + 4))
                    p.drawLine(QPointF(cx_c - 1, cy_c + 4), QPointF(cx_c + 5, cy_c - 3))
                else:
                    # File icon placeholder
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor("#3A3A4A"))
                    p.drawRoundedRect(QRectF(x + 8, y + 6, 16, 3), 1, 1)
                    p.drawRoundedRect(QRectF(x + 8, y + 12, 12, 3), 1, 1)
        
        # Scanning bar
        scan_y = grid_y + (scan_progress / 1.0) * (rows * cell_h)
        if scan_progress < 1.0:
            scan_bar_color = QColor(124, 77, 255, 150)
            p.setPen(QPen(scan_bar_color, 2))
            p.drawLine(QPointF(grid_x, scan_y), QPointF(grid_x + cols * cell_w, scan_y))
            # Glow
            grad = QLinearGradient(0, scan_y - 8, 0, scan_y)
            grad.setColorAt(0, QColor(124, 77, 255, 0))
            grad.setColorAt(1, QColor(124, 77, 255, 60))
            p.setPen(Qt.NoPen)
            p.setBrush(grad)
            p.drawRect(QRectF(grid_x, scan_y - 8, cols * cell_w, 8))
        
        # Progress text at bottom
        pct = min(100, int(scan_progress / 1.0 * 100))
        p.setPen(self._purple)
        font = QFont("Segoe UI", 9, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, h - 28, w, 20), Qt.AlignCenter, f"Indexing... {pct}%")

    # ── Step 5: Settings ─────────────────────────────
    def _draw_settings(self, p: QPainter):
        t = self._frame / 60.0
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        
        # Rotating gear
        p.save()
        p.translate(cx, cy - 10)
        p.rotate(t * 30)
        
        gear_color = self._purple
        p.setPen(Qt.NoPen)
        p.setBrush(gear_color)
        
        # Gear teeth
        teeth = 8
        outer_r = 28
        inner_r = 20
        tooth_w = 0.25
        
        path = QPainterPath()
        for i in range(teeth):
            angle = i * (2 * math.pi / teeth)
            # Outer point
            a1 = angle - tooth_w
            a2 = angle + tooth_w
            if i == 0:
                path.moveTo(outer_r * math.cos(a1), outer_r * math.sin(a1))
            else:
                path.lineTo(outer_r * math.cos(a1), outer_r * math.sin(a1))
            path.lineTo(outer_r * math.cos(a2), outer_r * math.sin(a2))
            # Inner point
            a3 = angle + tooth_w + 0.15
            a4 = angle + (2 * math.pi / teeth) - tooth_w - 0.15
            path.lineTo(inner_r * math.cos(a3), inner_r * math.sin(a3))
            path.lineTo(inner_r * math.cos(a4), inner_r * math.sin(a4))
        path.closeSubpath()
        p.drawPath(path)
        
        # Center hole
        p.setBrush(QColor("#16161F"))
        p.drawEllipse(QPointF(0, 0), 10, 10)
        p.restore()
        
        # Shield icon appearing
        shield_alpha = int(128 + 127 * math.sin(t * 2))
        p.save()
        p.translate(cx + 60, cy - 10)
        shield_color = QColor(self._green)
        shield_color.setAlpha(shield_alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(shield_color)
        
        shield = QPainterPath()
        shield.moveTo(0, -18)
        shield.lineTo(14, -10)
        shield.lineTo(14, 4)
        shield.quadTo(14, 18, 0, 22)
        shield.quadTo(-14, 18, -14, 4)
        shield.lineTo(-14, -10)
        shield.closeSubpath()
        p.drawPath(shield)
        
        # Checkmark on shield
        p.setPen(QPen(self._white, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(-5, 2), QPointF(-1, 7))
        p.drawLine(QPointF(-1, 7), QPointF(6, -4))
        p.restore()
        
        # Exclusion patterns text
        patterns = [".py", ".json", ".env", ".git"]
        for i, pat in enumerate(patterns):
            px = cx - 100 + i * 52
            py = cy + 30
            # Tag style
            p.setPen(Qt.NoPen)
            tag_color = QColor(self._purple_bg)
            p.setBrush(tag_color)
            p.drawRoundedRect(QRectF(px, py, 44, 20), 10, 10)
            p.setPen(self._purple)
            font = QFont("Segoe UI", 8)
            p.setFont(font)
            p.drawText(QRectF(px, py, 44, 20), Qt.AlignCenter, pat)

    # ── Step 6: Ready ────────────────────────────────
    def _draw_ready(self, p: QPainter):
        t = self._frame / 60.0
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        
        # Big "Ctrl+Alt+H" keyboard shortcut display
        # Pulsing glow effect behind the keys
        glow_alpha = int(30 + 20 * math.sin(t * 2))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(124, 77, 255, glow_alpha))
        p.drawRoundedRect(QRectF(30, 8, w - 60, h - 16), 16, 16)
        
        # "Quick Search Anywhere" title
        p.setPen(self._purple)
        font = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 10, w, 22), Qt.AlignCenter, "⚡ Quick Search — Anytime, Anywhere")
        
        # Draw keyboard keys: Ctrl + Alt + H
        key_y = cy - 8
        key_h = 38
        keys = ["Ctrl", "Alt", "H"]
        key_widths = [60, 50, 42]
        total_w = sum(key_widths) + 60  # keys + plus signs + spacing
        start_x = (w - total_w) / 2
        
        curr_x = start_x
        for i, (key, kw) in enumerate(zip(keys, key_widths)):
            # Key press animation - each key presses in sequence
            press_time = t * 1.5 - i * 0.3
            press_offset = 0
            if press_time > 0:
                cycle = press_time % 3
                if cycle < 0.15:
                    press_offset = 3 * (cycle / 0.15)  # Press down
                elif cycle < 0.4:
                    press_offset = 3 * (1 - (cycle - 0.15) / 0.25)  # Release
            
            ky = key_y + press_offset
            
            # Key shadow
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 25))
            p.drawRoundedRect(QRectF(curr_x + 1, ky + 3, kw, key_h), 8, 8)
            
            # Key background
            key_bg = QColor("#252535") if press_offset < 1 else QColor("#2A2040")
            p.setBrush(key_bg)
            p.setPen(QPen(QColor("#B39DDB"), 2))
            p.drawRoundedRect(QRectF(curr_x, ky, kw, key_h), 8, 8)
            
            # Key label
            p.setPen(self._purple)
            font = QFont("Segoe UI", 14 if key == "H" else 11, QFont.Bold)
            p.setFont(font)
            p.drawText(QRectF(curr_x, ky, kw, key_h), Qt.AlignCenter, key)
            
            curr_x += kw + 6
            
            # Plus sign between keys
            if i < len(keys) - 1:
                p.setPen(QColor("#7A7A90"))
                font = QFont("Segoe UI", 13, QFont.Bold)
                p.setFont(font)
                p.drawText(QRectF(curr_x - 2, key_y, 16, key_h), Qt.AlignCenter, "+")
                curr_x += 16
        
        # "Try it now!" prompt at bottom with bounce
        bounce = 2 * math.sin(t * 4)
        p.setPen(self._purple)
        font = QFont("Segoe UI", 9)
        p.setFont(font)
        p.drawText(QRectF(0, h - 26 + bounce, w, 20), Qt.AlignCenter, "👆 Try it now — works from any screen!")


class OnboardingOverlay(QDialog):
    """
    Interactive onboarding panel that floats over the app
    and guides users through key features
    """
    
    finished_onboarding = Signal()
    remind_later = Signal()  # Signal for "remind me later"
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.current_step = 0
        self.drag_position = None
        self.is_minimized = False  # For "Try It" mode
        
        # Steps definition with shorter, bullet-point text
        # nav_index: 0=Search, 1=Organize, 2=Index Files, 3=Settings
        # highlight: attribute name on main_window to spotlight
        self.steps = [
            {
                "title": "Welcome to Filect! 🎉",
                "description": "• Quick tour of key features\n• Takes about 30 seconds\n• Use ← → keys to navigate",
                "nav_index": None,
                "button_text": "Let's Go!",
                "show_try_it": False,
                "highlight": None
            },
            {
                "title": "🔍 Smart Search",
                "description": "• Find files by content, not just names\n• Try: \"vacation photo\" or \"tax document\"\n• AI understands what's inside files",
                "nav_index": 0,
                "button_text": "Next",
                "show_try_it": True,
                "highlight": "search_input"
            },
            {
                "title": "🗂️ Organize Files",
                "description": "• Select a destination folder\n• Click \"Generate Plan\"\n• Review suggestions & apply",
                "nav_index": 1,
                "sub_tab": 0,
                "button_text": "Next",
                "show_try_it": True,
                "highlight": "organize_page.content_stack"
            },
            {
                "title": "⚡ Auto-Organize",
                "description": "• Set it and forget it\n• Files sorted automatically on arrival\n• Configure watched folders here",
                "nav_index": 1,
                "sub_tab": 1,
                "button_text": "Next",
                "show_try_it": True,
                "highlight": "organize_page.watch_card"
            },
            {
                "title": "📁 Index Files",
                "description": "• AI learns about your files first\n• Click \"Add Folder\" to start\n• Required before search/organize",
                "nav_index": 2,
                "button_text": "Next",
                "show_try_it": True,
                "highlight": None
            },
            {
                "title": "⚙️ Settings",
                "description": "• Protect files from being moved\n• Add exclusion patterns (.json, .py)\n• Configure app behavior",
                "nav_index": 3,
                "button_text": "Next",
                "show_try_it": False,
                "highlight": None
            },
            {
                "title": "✅ You're Ready!",
                "description": "• Press Ctrl+Alt+H for quick search\n• Check History for past actions\n• Pin files to lock them in place",
                "nav_index": 1,
                "sub_tab": 0,
                "button_text": "Start Using the App",
                "show_try_it": False,
                "highlight": None
            }
        ]
        
        # Spotlight overlay for Phase 3
        self.spotlight = None
        
        self._setup_ui()
        self._apply_styling()
        self._update_step()
    
    def _setup_ui(self):
        """Set up the UI components"""
        # Window settings
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(520, 620)
        
        # Main container with shadow
        self.container = QFrame(self)
        self.container.setObjectName("onboardingContainer")
        self.container.setGeometry(10, 10, 500, 600)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.container.setGraphicsEffect(shadow)
        
        # Layout
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        
        # Header with step indicator
        header = QHBoxLayout()
        header.setSpacing(8)
        
        self.step_label = QLabel("Step 1 of 7")
        self.step_label.setObjectName("stepLabel")
        header.addWidget(self.step_label)
        
        header.addStretch()
        
        # Remind Me Later button (replaces Skip)
        self.remind_btn = QPushButton("⏰ Remind Me Later")
        self.remind_btn.setObjectName("remindButton")
        self.remind_btn.setCursor(Qt.PointingHandCursor)
        self.remind_btn.clicked.connect(self._remind_later)
        header.addWidget(self.remind_btn)
        
        layout.addLayout(header)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Title
        self.title_label = QLabel("Welcome!")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        
        # Animation widget
        self.anim_widget = OnboardingAnimation(self.container)
        self.anim_widget.setFixedSize(440, 130)
        layout.addWidget(self.anim_widget, 0, Qt.AlignCenter)
        
        # Description
        self.desc_label = QLabel("Description goes here")
        self.desc_label.setObjectName("descLabel")
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.desc_label, 1)
        
        # Keyboard hint
        self.keyboard_hint = QLabel("💡 Use ← → arrow keys  •  Esc to skip")
        self.keyboard_hint.setObjectName("keyboardHint")
        self.keyboard_hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.keyboard_hint)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(12)
        
        self.back_btn = QPushButton("← Back")
        self.back_btn.setObjectName("backButton")
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)
        
        nav_layout.addStretch()
        
        # Try It button
        self.try_btn = QPushButton("🎯 Try It")
        self.try_btn.setObjectName("tryButton")
        self.try_btn.setCursor(Qt.PointingHandCursor)
        self.try_btn.clicked.connect(self._try_it)
        self.try_btn.setVisible(False)
        nav_layout.addWidget(self.try_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setObjectName("nextButton")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        # Minimized "Continue" button (shown when trying features)
        self.continue_btn = QPushButton("▶ Continue Tour")
        self.continue_btn.setObjectName("continueButton")
        self.continue_btn.setCursor(Qt.PointingHandCursor)
        self.continue_btn.clicked.connect(self._restore_from_try)
        self.continue_btn.setFixedSize(140, 40)
        self.continue_btn.setParent(self.main_window)
        self.continue_btn.hide()
    
    def _apply_styling(self):
        """Apply theme-aware styling"""
        from app.ui.theme_manager import get_theme_colors
        c = get_theme_colors()
        self.setStyleSheet(f"""
            QFrame#onboardingContainer {{
                background-color: {c['surface']};
                border-radius: 16px;
                border: 2px solid rgba(124, 77, 255, 0.3);
            }}
            
            QLabel#stepLabel {{
                color: #7C4DFF;
                font-size: 12px;
                font-weight: 600;
            }}
            
            QLabel#titleLabel {{
                color: {c['text']};
                font-size: 22px;
                font-weight: 700;
            }}
            
            QLabel#descLabel {{
                color: {c['text_secondary']};
                font-size: 15px;
                line-height: 1.6;
            }}
            
            QLabel#keyboardHint {{
                color: {c['text_muted']};
                font-size: 11px;
                padding: 4px;
            }}
            
            QProgressBar#progressBar {{
                background-color: {c['border']};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar#progressBar::chunk {{
                background-color: #7C4DFF;
                border-radius: 3px;
            }}
            
            QPushButton#remindButton {{
                background-color: transparent;
                border: none;
                color: {c['text_muted']};
                font-size: 12px;
                padding: 4px 8px;
            }}
            QPushButton#remindButton:hover {{
                color: #7C4DFF;
            }}
            
            QPushButton#backButton {{
                background-color: transparent;
                border: 2px solid {c['border_strong']};
                border-radius: 10px;
                color: {c['text_muted']};
                font-size: 14px;
                font-weight: 600;
                padding: 10px 20px;
                min-width: 90px;
            }}
            QPushButton#backButton:hover {{
                border-color: #7C4DFF;
                color: #7C4DFF;
            }}
            QPushButton#backButton:disabled {{
                border-color: {c['border']};
                color: {c['text_disabled']};
            }}
            
            QPushButton#tryButton {{
                background-color: transparent;
                border: 2px solid #7C4DFF;
                border-radius: 10px;
                color: #7C4DFF;
                font-size: 14px;
                font-weight: 600;
                padding: 10px 16px;
                min-width: 90px;
            }}
            QPushButton#tryButton:hover {{
                background-color: rgba(124, 77, 255, 0.1);
            }}
            
            QPushButton#nextButton {{
                background-color: #7C4DFF;
                border: none;
                border-radius: 10px;
                color: white;
                font-size: 14px;
                font-weight: 600;
                padding: 10px 24px;
                min-width: 120px;
            }}
            QPushButton#nextButton:hover {{
                background-color: #9575FF;
            }}
            
            QPushButton#continueButton {{
                background-color: #7C4DFF;
                border: none;
                border-radius: 20px;
                color: white;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton#continueButton:hover {{
                background-color: #9575FF;
            }}
        """)
    
    # Telemetry slugs — MUST match the order of ``self.steps``. Used by
    # _update_step / _finish_tour / _remind_later to label the step in the
    # ``onboarding_step_viewed`` / ``onboarding_dismissed`` events.
    _STEP_SLUGS = [
        "welcome", "smart_search", "organize_files",
        "auto_organize", "index_files", "settings", "ready",
    ]

    def _update_step(self):
        """Update the UI for the current step"""
        # Telemetry: emit before any UI updates so we record the view even
        # if a downstream render bug throws.
        try:
            from app.core.supabase_client import track
            slug = (
                self._STEP_SLUGS[self.current_step]
                if self.current_step < len(self._STEP_SLUGS)
                else "unknown"
            )
            track(
                "onboarding_step_viewed",
                step=slug,
                step_index=self.current_step,
                total_steps=len(self.steps),
            )
        except Exception:
            pass

        step = self.steps[self.current_step]

        # Update labels
        self.step_label.setText(f"Step {self.current_step + 1} of {len(self.steps)}")
        self.title_label.setText(step["title"])
        self.desc_label.setText(step["description"])
        self.next_btn.setText(step["button_text"])
        
        # Update progress bar
        progress = int((self.current_step + 1) / len(self.steps) * 100)
        self.progress_bar.setValue(progress)
        
        # Update back button visibility
        self.back_btn.setVisible(self.current_step > 0)
        
        # Update remind button visibility (hide on last step)
        self.remind_btn.setVisible(self.current_step < len(self.steps) - 1)
        
        # Update Try It button visibility
        self.try_btn.setVisible(step.get("show_try_it", False))
        
        # Update animation
        self.anim_widget.set_step(self.current_step)
        
        # Navigate to the appropriate page in the app
        nav_index = step["nav_index"]
        if nav_index is not None:
            if hasattr(self.main_window, 'page_stack'):
                self.main_window.page_stack.setCurrentIndex(nav_index)
            if hasattr(self.main_window, 'nav_buttons') and nav_index < len(self.main_window.nav_buttons):
                self.main_window.nav_buttons[nav_index].setChecked(True)
        
        # Handle sub-tab switching for Organize page
        sub_tab = step.get("sub_tab")
        if sub_tab is not None and hasattr(self.main_window, 'organize_page'):
            self.main_window.organize_page._switch_tab(sub_tab)
        
        # Phase 3: Update spotlight overlay
        self._update_spotlight(step.get("highlight"))
    
    def _update_spotlight(self, highlight_attr):
        """Update the spotlight overlay to highlight a widget"""
        # Hide existing spotlight when not needed or on certain steps
        if not highlight_attr:
            if self.spotlight:
                self.spotlight.hide()
            return
        
        # Get the widget to highlight - supports dot-path like "organize_page.content_stack"
        target_widget = None
        parts = highlight_attr.split(".")
        obj = self.main_window
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None
                break
        target_widget = obj
        
        if not target_widget:
            if self.spotlight:
                self.spotlight.hide()
            return
        
        # Create spotlight if needed
        if not self.spotlight:
            self.spotlight = SpotlightOverlay(self.main_window)
        
        # Position and show spotlight
        self.spotlight.setGeometry(self.main_window.rect())
        
        # Get widget rect relative to main window
        widget_pos = target_widget.mapTo(self.main_window, QPoint(0, 0))
        widget_rect = QRect(widget_pos.x(), widget_pos.y(), target_widget.width(), target_widget.height())
        
        self.spotlight.set_spotlight(widget_rect)
        self.spotlight.show()
        self.spotlight.raise_()
        
        # Make sure our panel is above the spotlight
        self.raise_()
    
    def _go_next(self):
        """Go to the next step or finish"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._update_step()
        else:
            self._finish_tour()
    
    def _go_back(self):
        """Go to the previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_step()
    
    def _try_it(self):
        """Minimize panel so user can try the feature"""
        self.is_minimized = True
        self.hide()
        
        # Hide spotlight while trying
        if self.spotlight:
            self.spotlight.hide()
        
        # Show the "Continue Tour" button at bottom-right
        if self.main_window:
            main_geo = self.main_window.geometry()
            btn_x = main_geo.width() - self.continue_btn.width() - 20
            btn_y = main_geo.height() - self.continue_btn.height() - 20
            self.continue_btn.move(btn_x, btn_y)
            self.continue_btn.show()
            self.continue_btn.raise_()
    
    def _restore_from_try(self):
        """Restore the onboarding panel after trying"""
        self.is_minimized = False
        self.continue_btn.hide()
        self.show()
        self.raise_()
        # Re-show spotlight
        step = self.steps[self.current_step]
        self._update_spotlight(step.get("highlight"))
    
    def _remind_later(self):
        """Remind the user later instead of skipping entirely"""
        try:
            from app.core.supabase_client import track
            slug = (
                self._STEP_SLUGS[self.current_step]
                if self.current_step < len(self._STEP_SLUGS)
                else "unknown"
            )
            track(
                "onboarding_dismissed",
                step=slug,
                step_index=self.current_step,
                total_steps=len(self.steps),
            )
        except Exception:
            pass

        if self.spotlight:
            self.spotlight.hide()
        self.remind_later.emit()
        self.reject()

    def _finish_tour(self):
        """Complete the onboarding with celebration"""
        try:
            from app.core.supabase_client import track
            track("onboarding_completed", total_steps=len(self.steps))
        except Exception:
            pass

        self.continue_btn.hide()
        if self.spotlight:
            self.spotlight.hide()
        self._show_confetti()  # Phase 4: Confetti celebration!
        self.finished_onboarding.emit()
        self.accept()
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation"""
        if event.key() == Qt.Key_Right or event.key() == Qt.Key_Return:
            self._go_next()
        elif event.key() == Qt.Key_Left:
            if self.current_step > 0:
                self._go_back()
        elif event.key() == Qt.Key_Escape:
            self._remind_later()
        else:
            super().keyPressEvent(event)
    
    def mousePressEvent(self, event):
        """Enable dragging the panel"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle dragging"""
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        """Reset drag position"""
        self.drag_position = None
    
    def showEvent(self, event):
        """Position the panel when shown"""
        super().showEvent(event)
        # Position at bottom-right of main window
        if self.main_window:
            main_geo = self.main_window.geometry()
            x = main_geo.right() - self.width() - 30
            y = main_geo.bottom() - self.height() - 30
            self.move(x, y)
    
    def closeEvent(self, event):
        """Clean up when closing"""
        self.continue_btn.hide()
        if hasattr(self, 'anim_widget'):
            self.anim_widget.stop()
        if hasattr(self, 'spotlight') and self.spotlight:
            self.spotlight.hide()
        super().closeEvent(event)
    
    def _show_confetti(self):
        """Show confetti celebration animation"""
        if not self.main_window:
            return
        self.confetti = ConfettiWidget(self.main_window)
        self.confetti.show()
        self.confetti.start_animation()


class ConfettiWidget(QWidget):
    """Confetti celebration animation widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        # Cover the parent window
        if parent:
            self.setGeometry(parent.rect())
        
        # Confetti particles: (x, y, size, color, speed, angle)
        self.particles = []
        self.colors = [
            QColor("#7C4DFF"),  # Purple
            QColor("#B39DDB"),  # Light purple
            QColor("#E8E0FF"),  # Very light purple
            QColor("#FFD700"),  # Gold
            QColor("#FF69B4"),  # Pink
            QColor("#00CED1"),  # Cyan
        ]
        
        # Create particles
        import random
        for _ in range(60):
            self.particles.append({
                'x': random.randint(0, self.width() if self.width() > 0 else 800),
                'y': random.randint(-100, -10),
                'size': random.randint(6, 12),
                'color': random.choice(self.colors),
                'speed': random.uniform(3, 8),
                'wobble': random.uniform(-2, 2),
                'rotation': random.randint(0, 360),
            })
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_particles)
        self.frame_count = 0
        self.max_frames = 120  # 2 seconds at 60fps
    
    def start_animation(self):
        """Start the confetti animation"""
        # Reinitialize particles when starting
        import random
        parent_width = self.parent().width() if self.parent() else 800
        self.particles = []
        for _ in range(60):
            self.particles.append({
                'x': random.randint(0, parent_width),
                'y': random.randint(-100, -10),
                'size': random.randint(6, 12),
                'color': random.choice(self.colors),
                'speed': random.uniform(3, 8),
                'wobble': random.uniform(-2, 2),
                'rotation': random.randint(0, 360),
            })
        self.frame_count = 0
        self.timer.start(16)  # ~60fps
    
    def _update_particles(self):
        """Update particle positions"""
        import random
        self.frame_count += 1
        
        for p in self.particles:
            p['y'] += p['speed']
            p['x'] += p['wobble']
            p['rotation'] += 5
        
        self.update()
        
        # Stop after max frames
        if self.frame_count >= self.max_frames:
            self.timer.stop()
            self.hide()
            self.deleteLater()
    
    def paintEvent(self, event):
        """Draw confetti particles"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fade out in last 30 frames
        opacity = 1.0
        if self.frame_count > self.max_frames - 30:
            opacity = (self.max_frames - self.frame_count) / 30.0
        
        for p in self.particles:
            color = QColor(p['color'])
            color.setAlphaF(opacity)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            
            painter.save()
            painter.translate(p['x'], p['y'])
            painter.rotate(p['rotation'])
            
            # Draw rectangle confetti
            painter.drawRect(-p['size']//2, -p['size']//4, p['size'], p['size']//2)
            
            painter.restore()


class SpotlightOverlay(QWidget):
    """Semi-transparent overlay with a spotlight hole"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        self.spotlight_rect = None
        self.opacity = 0.5
        
        if parent:
            self.setGeometry(parent.rect())
            self.raise_()
    
    def set_spotlight(self, rect):
        """Set the spotlight area (QRect in parent coordinates)"""
        if rect:
            self.spotlight_rect = QRect(
                rect.x() - 10, rect.y() - 10,
                rect.width() + 20, rect.height() + 20
            )
        else:
            self.spotlight_rect = None
        self.update()
    
    def paintEvent(self, event):
        """Draw the overlay with spotlight hole"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw semi-transparent overlay
        overlay_color = QColor(0, 0, 0, int(255 * self.opacity))
        
        if self.spotlight_rect:
            # Create a path for the entire widget minus the spotlight
            path = QPainterPath()
            path.addRect(0, 0, self.width(), self.height())
            
            # Cut out the spotlight area (rounded rect)
            spotlight_path = QPainterPath()
            spotlight_path.addRoundedRect(
                self.spotlight_rect.x(), self.spotlight_rect.y(),
                self.spotlight_rect.width(), self.spotlight_rect.height(),
                12, 12
            )
            path = path.subtracted(spotlight_path)
            
            painter.fillPath(path, overlay_color)
            
            # Draw glowing border around spotlight
            glow_pen = QPen(QColor("#7C4DFF"))
            glow_pen.setWidth(3)
            painter.setPen(glow_pen)
            painter.drawRoundedRect(self.spotlight_rect, 12, 12)
        else:
            painter.fillRect(self.rect(), overlay_color)
