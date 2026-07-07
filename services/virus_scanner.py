"""
services/virus_scanner.py — ClamAV Virus Scanning Integration

Scans uploaded files using ClamAV daemon (clamd) before ingestion.
ClamAV is a required dependency — if the daemon is unavailable, the
upload is BLOCKED with a 503 error.

Configuration:
  CLAMAV_HOST:    IP/hostname of clamd daemon  (default: 127.0.0.1)
  CLAMAV_PORT:    Port clamd listens on         (default: 3310)
  CLAMAV_TIMEOUT: Connection timeout in seconds  (default: 30.0)

Error policy:
  - Daemon unreachable  → raise VirusScanUnavailableError (→ HTTP 503)
  - Virus detected      → raise VirusDetectedError        (→ HTTP 422)
  - Clean              → return ScanResult(clean=True)
"""
from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class VirusScanUnavailableError(Exception):
    """Raised when the ClamAV daemon cannot be reached."""
    def __init__(self, detail: str = ""):
        super().__init__(
            f"ClamAV daemon is unavailable. Upload blocked until service is restored. {detail}".strip()
        )


class VirusDetectedError(Exception):
    """Raised when ClamAV detects a threat in the uploaded file."""
    def __init__(self, threat_name: str, filename: str):
        self.threat_name = threat_name
        self.filename = filename
        super().__init__(
            f"Virus detected in '{filename}': {threat_name}. File rejected and deleted."
        )


# ── Scan result ───────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    clean: bool
    threat_name: Optional[str] = None
    engine_version: Optional[str] = None


# ── ClamAV client ─────────────────────────────────────────────────────────────

class ClamdClient:
    """
    Minimal ClamAV network client using the clamd stream protocol.
    Uses the `clamd` Python package if available, otherwise falls back
    to a raw socket implementation (TCP stream scan).
    """

    def __init__(self):
        self._clamd = None
        self._clamd_attempted = False

    def _get_clamd(self):
        if self._clamd_attempted:
            return self._clamd
        self._clamd_attempted = True
        try:
            import clamd
            self._clamd = clamd.ClamdNetworkSocket(
                host=settings.CLAMAV_HOST,
                port=settings.CLAMAV_PORT,
                timeout=settings.CLAMAV_TIMEOUT,
            )
            self._clamd.ping()
            logger.info(
                "ClamAV daemon connected",
                extra={"host": settings.CLAMAV_HOST, "port": settings.CLAMAV_PORT},
            )
        except ImportError:
            logger.warning("clamd package not installed — using raw socket fallback")
            self._clamd = None
        except Exception as exc:
            logger.error(
                "ClamAV daemon unreachable",
                extra={"host": settings.CLAMAV_HOST, "port": settings.CLAMAV_PORT, "error": str(exc)},
            )
            self._clamd = None
        return self._clamd

    def scan_file(self, file_path: str) -> ScanResult:
        """
        Scan a file on disk.

        Returns ScanResult(clean=True) if no threats found.
        Raises VirusScanUnavailableError if daemon cannot be reached.
        Raises VirusDetectedError if a threat is detected.
        """
        clamd = self._get_clamd()

        if clamd is not None:
            return self._scan_with_clamd(clamd, file_path)
        else:
            return self._scan_with_raw_socket(file_path)

    def _scan_with_clamd(self, clamd, file_path: str) -> ScanResult:
        """Scan using the clamd Python package."""
        try:
            result = clamd.scan(file_path)
            # result = {file_path: ('OK', None)} or {file_path: ('FOUND', 'Eicar-Test-Signature')}
            if result is None:
                raise VirusScanUnavailableError("clamd returned no result")

            file_result = result.get(file_path, ("ERROR", "No result"))
            status, threat = file_result

            if status == "OK":
                logger.info("Virus scan passed", extra={"file": file_path})
                return ScanResult(clean=True)
            elif status == "FOUND":
                logger.error(
                    "Virus detected",
                    extra={"file": file_path, "threat": threat},
                )
                raise VirusDetectedError(
                    threat_name=threat or "Unknown threat",
                    filename=file_path,
                )
            else:
                raise VirusScanUnavailableError(f"clamd scan error: {status} {threat}")

        except (VirusDetectedError, VirusScanUnavailableError):
            raise
        except Exception as exc:
            # Daemon dropped connection, timeout, etc.
            raise VirusScanUnavailableError(str(exc)) from exc

    def _scan_with_raw_socket(self, file_path: str) -> ScanResult:
        """
        Fallback raw TCP socket scan using the INSTREAM protocol.
        Reads the file in 4096-byte chunks and streams to clamd.
        """
        CHUNK_SIZE = 4096
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(settings.CLAMAV_TIMEOUT)
            sock.connect((settings.CLAMAV_HOST, settings.CLAMAV_PORT))
            sock.sendall(b"zINSTREAM\0")

            with open(file_path, "rb") as f:
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    size = len(data)
                    sock.sendall(size.to_bytes(4, "big") + data)

            # Terminate stream
            sock.sendall((0).to_bytes(4, "big"))

            response = sock.recv(1024).decode("utf-8", errors="replace").strip("\0").strip()
            sock.close()

            logger.debug("ClamAV raw response", extra={"response": response})

            if "OK" in response and "FOUND" not in response:
                return ScanResult(clean=True)
            elif "FOUND" in response:
                # Response format: "stream: ThreatName FOUND"
                parts = response.split(":")
                threat = parts[1].strip().replace(" FOUND", "").strip() if len(parts) > 1 else "Unknown"
                raise VirusDetectedError(threat_name=threat, filename=file_path)
            else:
                raise VirusScanUnavailableError(f"Unexpected clamd response: {response}")

        except (VirusDetectedError, VirusScanUnavailableError):
            raise
        except Exception as exc:
            raise VirusScanUnavailableError(
                f"Cannot connect to ClamAV at {settings.CLAMAV_HOST}:{settings.CLAMAV_PORT} — {exc}"
            ) from exc


# ── Singleton ─────────────────────────────────────────────────────────────────
_scanner = ClamdClient()


def scan_file(file_path: str) -> ScanResult:
    """
    Scan a file for viruses.

    This is the public interface used by the ingestion pipeline.
    Raises VirusScanUnavailableError or VirusDetectedError on failure.
    Returns ScanResult(clean=True) on success.
    """
    logger.info("Starting virus scan", extra={"file": file_path})
    return _scanner.scan_file(file_path)
