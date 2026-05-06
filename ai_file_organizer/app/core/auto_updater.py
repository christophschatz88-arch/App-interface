"""
Auto-updater for the Filect app.
Downloads installer from releases and runs it to apply updates.
"""

import logging
import os
import sys
import shutil
import tempfile
import subprocess
import ssl
import certifi
from pathlib import Path
from typing import Optional, Callable

# Use requests library - much better SSL handling for PyInstaller apps
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


def get_app_dir() -> Path:
    """Get the directory where the app is installed."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return Path(sys.executable).parent
    else:
        # Running as script - use the ai_file_organizer folder
        return Path(__file__).parent.parent.parent


def get_update_dir() -> Path:
    """Get temporary directory for update downloads."""
    update_dir = Path(tempfile.gettempdir()) / "filect_update"
    
    # Clean up any existing directory to avoid permission issues
    if update_dir.exists():
        try:
            shutil.rmtree(update_dir)
        except Exception as e:
            logger.warning(f"Could not clean update dir: {e}")
            # Try alternative directory with timestamp
            import time
            update_dir = Path(tempfile.gettempdir()) / f"filect_update_{int(time.time())}"
    
    update_dir.mkdir(exist_ok=True)
    return update_dir


def _is_valid_windows_executable(file_path: Path) -> bool:
    """
    Check if a file is a valid Windows PE executable.
    
    Returns:
        True if the file has valid PE headers (MZ header + PE signature)
    """
    try:
        with open(file_path, 'rb') as f:
            # Check DOS header magic number (MZ)
            dos_header = f.read(2)
            if dos_header != b'MZ':
                logger.error(f"Invalid DOS header: {dos_header!r} (expected b'MZ')")
                return False
            
            # Read PE header offset from DOS header (at offset 0x3C)
            f.seek(0x3C)
            pe_offset_bytes = f.read(4)
            if len(pe_offset_bytes) < 4:
                logger.error("File too small to contain PE offset")
                return False
            
            pe_offset = int.from_bytes(pe_offset_bytes, 'little')
            
            # Check PE signature at the offset
            f.seek(pe_offset)
            pe_sig = f.read(4)
            if pe_sig != b'PE\x00\x00':
                logger.error(f"Invalid PE signature: {pe_sig!r} (expected b'PE\\x00\\x00')")
                return False
            
            return True
    except Exception as e:
        logger.error(f"Error validating executable: {e}")
        return False


def _log_file_contents_preview(file_path: Path, max_bytes: int = 500):
    """Log the first bytes of a file for debugging invalid downloads."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read(max_bytes)
            # Check if it looks like HTML (common when download fails)
            if content.startswith(b'<!') or content.startswith(b'<html') or b'<head>' in content.lower():
                logger.error(f"Downloaded file appears to be HTML (likely an error page)")
                # Try to decode and log
                try:
                    text = content.decode('utf-8', errors='replace')
                    logger.error(f"HTML content preview: {text[:300]}")
                except:
                    pass
            else:
                logger.error(f"File header bytes: {content[:64]!r}")
    except Exception as e:
        logger.error(f"Could not read file for preview: {e}")


def download_update(
    download_url: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None
) -> Optional[Path]:
    """
    Download update installer from URL.
    
    Args:
        download_url: URL to download from (GitHub Release asset)
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
        status_callback: Optional callback(status_message) for UI updates
        
    Returns:
        Path to downloaded installer file, or None on failure
    """
    def update_status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)
    
    try:
        update_dir = get_update_dir()
        
        # Determine filename from URL
        filename = download_url.split('/')[-1]
        if not filename.endswith('.exe'):
            filename = "Filect-Setup.exe"
        
        installer_path = update_dir / filename
        
        # Clean up any previous download
        if installer_path.exists():
            installer_path.unlink()
        
        update_status("Connecting to server...")
        logger.info(f"Downloading update from: {download_url}")
        
        if HAS_REQUESTS:
            result = _download_with_requests(download_url, installer_path, progress_callback, update_status)
        else:
            result = _download_with_urllib(download_url, installer_path, progress_callback, update_status)
        
        # Validate the downloaded file is actually a Windows executable
        if result and result.exists():
            update_status("Verifying download...")
            
            # Check minimum file size (Inno Setup installers are typically > 1MB)
            file_size = result.stat().st_size
            min_size = 500 * 1024  # 500 KB minimum
            if file_size < min_size:
                logger.error(f"Downloaded file too small: {file_size} bytes (expected at least {min_size})")
                _log_file_contents_preview(result)
                update_status("Download failed: File too small (incomplete download)")
                try:
                    result.unlink()
                except:
                    pass
                return None
            
            if not _is_valid_windows_executable(result):
                logger.error("Downloaded file is not a valid Windows executable!")
                _log_file_contents_preview(result)
                update_status("Download failed: Invalid installer file")
                # Clean up the invalid file
                try:
                    result.unlink()
                except:
                    pass
                return None
            logger.info(f"Download verified as valid Windows executable ({file_size / (1024*1024):.2f} MB)")
        
        return result
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        if status_callback:
            status_callback(f"Download failed: {str(e)[:50]}")
        return None


def _download_with_requests(
    download_url: str,
    installer_path: Path,
    progress_callback: Optional[Callable[[int, int], None]],
    update_status: Callable[[str], None]
) -> Optional[Path]:
    """Download using requests library - better SSL handling."""
    try:
        update_status("Establishing secure connection...")
        
        # Use requests with streaming for large files
        # verify=True uses certifi's certificates which work in PyInstaller
        # GitHub release assets require proper Accept header and redirect handling
        response = requests.get(
            download_url,
            stream=True,
            timeout=(30, 300),  # (connect timeout, read timeout)
            headers={
                'User-Agent': 'Filect-Updater/2.0',
                'Accept': 'application/octet-stream, application/x-msdownload, */*'
            },
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Check if we got HTML instead of binary (common error response)
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            logger.error(f"Server returned HTML instead of binary. Content-Type: {content_type}")
            logger.error(f"Final URL after redirects: {response.url}")
            update_status("Download failed: Server returned an error page")
            return None
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 131072  # 128KB chunks
        
        if total_size > 0:
            update_status(f"Downloading... 0 / {total_size / (1024*1024):.1f} MB")
            logger.info(f"Download size: {total_size / (1024*1024):.2f} MB")
        else:
            update_status("Downloading...")
        
        # Initial progress callback
        if progress_callback:
            progress_callback(0, total_size if total_size > 0 else 1)
        
        with open(installer_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback:
                        progress_callback(downloaded, total_size if total_size > 0 else downloaded)
                    
                    # Update status every ~5MB
                    if total_size > 0 and downloaded % (5 * 1024 * 1024) < chunk_size:
                        percent = int((downloaded / total_size) * 100)
                        update_status(f"Downloading... {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({percent}%)")
        
        # Verify the file was downloaded
        if installer_path.exists() and installer_path.stat().st_size > 0:
            actual_size = installer_path.stat().st_size
            logger.info(f"Download complete: {installer_path} ({actual_size / (1024*1024):.2f} MB)")
            update_status("Download complete!")
            return installer_path
        else:
            logger.error("Downloaded file is empty or missing")
            update_status("Download failed - file is empty")
            return None
            
    except requests.exceptions.SSLError as e:
        logger.error(f"SSL Error: {e}")
        update_status("SSL certificate error - trying fallback...")
        # Try with SSL verification disabled as fallback (not ideal but works)
        return _download_with_requests_no_verify(download_url, installer_path, progress_callback, update_status)
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection Error: {e}")
        update_status("Connection failed - check internet")
        return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout: {e}")
        update_status("Connection timed out")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} {e.response.reason}")
        update_status(f"Server error: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        update_status(f"Download error: {str(e)[:40]}")
        return None


def _download_with_requests_no_verify(
    download_url: str,
    installer_path: Path,
    progress_callback: Optional[Callable[[int, int], None]],
    update_status: Callable[[str], None]
) -> Optional[Path]:
    """Fallback download without SSL verification."""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        update_status("Retrying download...")
        
        response = requests.get(
            download_url,
            stream=True,
            timeout=(30, 300),
            headers={
                'User-Agent': 'Filect-Updater/2.0',
                'Accept': 'application/octet-stream, application/x-msdownload, */*'
            },
            allow_redirects=True,
            verify=False  # Disable SSL verification as fallback
        )
        response.raise_for_status()
        
        # Check if we got HTML instead of binary
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            logger.error(f"Server returned HTML instead of binary. Content-Type: {content_type}")
            update_status("Download failed: Server returned an error page")
            return None
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 131072
        
        if progress_callback:
            progress_callback(0, total_size if total_size > 0 else 1)
        
        with open(installer_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size if total_size > 0 else downloaded)
        
        if installer_path.exists() and installer_path.stat().st_size > 0:
            logger.info(f"Fallback download complete: {installer_path}")
            update_status("Download complete!")
            return installer_path
        return None
        
    except Exception as e:
        logger.error(f"Fallback download also failed: {e}")
        update_status("Download failed")
        return None


def _download_with_urllib(
    download_url: str,
    installer_path: Path,
    progress_callback: Optional[Callable[[int, int], None]],
    update_status: Callable[[str], None]
) -> Optional[Path]:
    """Fallback download using urllib (when requests not available)."""
    import urllib.request
    import urllib.error
    
    try:
        update_status("Establishing connection...")
        
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        request = urllib.request.Request(
            download_url,
            headers={
                'User-Agent': 'Filect-Updater/2.0',
                'Accept': 'application/octet-stream'
            }
        )
        
        with urllib.request.urlopen(request, timeout=120, context=ssl_context) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 131072
            
            if total_size > 0:
                update_status(f"Downloading... 0 / {total_size / (1024*1024):.1f} MB")
                logger.info(f"Download size: {total_size / (1024*1024):.2f} MB")
            
            if progress_callback:
                progress_callback(0, total_size if total_size > 0 else 1)
            
            with open(installer_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback:
                        progress_callback(downloaded, total_size if total_size > 0 else downloaded)
        
        if installer_path.exists() and installer_path.stat().st_size > 0:
            logger.info(f"Download complete: {installer_path}")
            update_status("Download complete!")
            return installer_path
        else:
            logger.error("Downloaded file is empty or missing")
            return None
            
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error: {e.code} {e.reason}")
        update_status(f"Server error: {e.code}")
        return None
    except urllib.error.URLError as e:
        logger.error(f"URL Error: {e.reason}")
        update_status("Connection failed")
        return None
    except Exception as e:
        logger.error(f"urllib download failed: {e}", exc_info=True)
        return None


def run_installer_and_exit(installer_path: Path) -> bool:
    """
    Run the installer and exit the current app.
    
    The installer will handle updating the app files.
    
    Args:
        installer_path: Path to the downloaded installer
        
    Returns:
        True if installer was launched successfully
    """
    try:
        if not installer_path.exists():
            logger.error(f"Installer not found: {installer_path}")
            return False
        
        logger.info(f"Launching installer: {installer_path}")
        
        if sys.platform == 'win32':
            # Create a VBS script that:
            # 1. Waits for the app to close
            # 2. Runs installer and WAITS for it to complete
            # 3. Inno Setup will auto-launch the app (skipifnotsilent flag)
            # 4. Fallback: launch app if not already running
            vbs_script = installer_path.parent / "run_update.vbs"
            
            # Escape backslashes for VBS string
            installer_str = str(installer_path).replace("\\", "\\\\")
            
            vbs_content = f'''
On Error Resume Next
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Wait for the old app to fully close
WScript.Sleep 3000

' Run the installer with /SILENT flag
' The installer will auto-launch the app via [Run] section with skipifnotsilent flag
installerPath = "{installer_str}"
returnCode = WshShell.Run(Chr(34) & installerPath & Chr(34) & " /SILENT", 1, True)

' Wait for Windows to finish any cleanup
WScript.Sleep 2000

' Clean up this script
fso.DeleteFile WScript.ScriptFullName
'''
            with open(vbs_script, 'w') as f:
                f.write(vbs_content)
            
            # Launch the VBS script
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            subprocess.Popen(
                ['wscript', str(vbs_script)],
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            logger.info("Update script launched - app will close now")
            
            # Exit the application
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                logger.info("Closing application for update...")
                app.quit()
        else:
            # Non-Windows: just open the installer
            subprocess.Popen(
                [str(installer_path)],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        logger.info("Installer launched successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to launch installer: {e}", exc_info=True)
        return False


def apply_update_and_restart(installer_path: Path) -> bool:
    """
    Apply update by running the installer and closing the app.
    
    Args:
        installer_path: Path to downloaded installer
        
    Returns:
        True if update process started successfully
    """
    return run_installer_and_exit(installer_path)


def cleanup_update_files():
    """Clean up any leftover update files."""
    try:
        update_dir = get_update_dir()
        if update_dir.exists():
            shutil.rmtree(update_dir)
            logger.info("Cleaned up update files")
    except Exception as e:
        logger.debug(f"Could not clean up update files: {e}")
