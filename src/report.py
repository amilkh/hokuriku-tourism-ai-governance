"""Report and metrics writer.

Centralises all text output so that every analysis section appends to
the same report buffer.  The ``Reporter`` instance is passed to every
module; at the end of the pipeline, call ``save()`` to flush to disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.figure


class Reporter:
    """Accumulates report lines and metrics for the analysis pipeline.

    Attributes:
        report_lines: Human-readable log lines (printed to stdout).
        metrics_lines: Machine-readable metrics for ``analysis_metrics.txt``.
        out_dir: Directory where output files are written.
        fig_dir: Directory where figures are written.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        repo_dir: Path = cfg["_resolved"]["repo_dir"]
        self.out_dir = repo_dir / cfg["paths"]["output"]
        self.fig_dir = repo_dir / cfg["paths"]["figures"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.report_lines: list[str] = []
        self.metrics_lines: list[str] = []

        self._dpi: int = cfg.get("visualization", {}).get("dpi", 150)
        self._ja_copy: bool = cfg.get("visualization", {}).get("ja_copy", True)

    # ── Text output ───────────────────────────────────────────────────────

    def log(self, msg: str = "") -> None:
        """Print a message and append it to the report buffer."""
        print(msg)
        self.report_lines.append(msg)

    def metrics(self, msg: str = "") -> None:
        """Append a line to the machine-readable metrics buffer."""
        self.metrics_lines.append(msg)

    def section(self, number: int | str, title: str) -> None:
        """Print a section header."""
        self.log("")
        self.log("=" * 80)
        self.log(f"SECTION {number} – {title}")
        self.log("=" * 80)

    # ── Figure saving ─────────────────────────────────────────────────────

    def save_fig(
        self,
        fig: matplotlib.figure.Figure,
        fname: str | Path,
        *,
        dpi: int | None = None,
        ja_copy: bool = False,
    ) -> Path:
        """Save a matplotlib figure and optionally create a ``_ja`` copy.

        Args:
            fig: Matplotlib ``Figure`` to save.
            fname: Output file path (absolute or relative to ``fig_dir``).
            dpi: Resolution; defaults to value in ``settings.yaml``.
            ja_copy: When ``True``, duplicate the file with a ``_ja`` suffix.

        Returns:
            The absolute path of the saved figure.
        """
        dpi = dpi or self._dpi
        path = Path(fname) if Path(fname).is_absolute() else self.fig_dir / fname
        fig.savefig(str(path), dpi=dpi)
        self._optimize_png(path)
        self.log(f"  Saved {path}")

        if ja_copy or self._ja_copy:
            ja_path = path.with_name(path.stem + "_ja" + path.suffix)
            shutil.copyfile(str(path), str(ja_path))
            self._optimize_png(ja_path)
            self.log(f"  Saved {ja_path}")

        return path

    def optimize_png(self, path: str | Path) -> None:
        """Optimize a PNG file if pngquant is available."""
        self._optimize_png(Path(path))

    def _optimize_png(self, path: Path) -> None:
        if path.suffix.lower() != ".png":
            return
        if shutil.which("pngquant") is None:
            return
        try:
            subprocess.run(
                [
                    "pngquant",
                    "--force",
                    "--ext",
                    ".png",
                    "--quality=80-100",
                    "--speed",
                    "1",
                    "--strip",
                    "--skip-if-larger",
                    str(path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass

    # ── Flush to disk ─────────────────────────────────────────────────────

    def save(self) -> Path:
        """Write ``analysis_metrics.txt`` to the output directory.

        Returns:
            Path of the written file.
        """
        out_path = self.out_dir / "analysis_metrics.txt"
        lines = self.metrics_lines if self.metrics_lines else self.report_lines
        out_path.write_text("\n".join(lines))
        self.log(f">>> Metrics saved to {out_path}")
        return out_path
