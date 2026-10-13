#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mbedit_qt.py — a PySide6 bathymetry editor that integrates with MB-System

A feature-rich reimplementation of MB-System's X11/Motif "mbedit" tool,
based on mbedit.c / mbedit_callbacks.c / mbedit_creation.c / mbedit.uil.

Implemented:
  * Reads fast-bathymetry (.fbt, MBF_MBLDEOIH format id 71) V4/V5 records.
  * Generates .fbt via `mbdatalist -O` if missing; expands simple datalists.
  * Reads/writes MB-System edit-save (.esf) files with real MB-System edit
    action codes (FLAG/UNFLAG/ZERO/FILTER) and beam-flag byte semantics
    (none/manual/filter/sonar/null), written continuously as edits happen.
  * Backs up any pre-existing .esf to .esf.tmp before a fresh session.
  * Updates the .par file via `mbset` (EDITSAVEMODE / EDITSAVEFILE).
  * View modes: Waterfall / Alongtrack (overlay) / Acrosstrack (along-track
    waterfall), matching mbedit's three view modes exactly.
  * Show modes: flag-state coloring, bottom-detect coloring, pulse-type
    coloring (detect/pulse degrade gracefully: .fbt carries no such arrays).
  * Edit modes: Toggle / Pick / Erase / Restore / Grab / Info, with
    reverse-mouse-buttons and reverse-key-macros options.
  * Ping-level actions: Bad/Good/Left/Right/Zero ping, Flag/Unflag View,
    Unflag Forward, Flag Out-of-Bounds ("zap").
  * Filters: median-spike, wrong-side, cut-by-beam, cut-by-distance,
    cut-by-angle.
  * Buffer/paging navigation, Go-To-Time, Annotation (scale/exaggeration),
    multi-file session with a File List dialog.
  * Keyboard shortcuts mirroring mbedit's key macros.

Author: Christian dos Santos Ferreira using Claude Sonnet 5.5 Thinking
"""
from __future__ import annotations

import os
import sys
import struct
import subprocess
import datetime
import shutil
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Dict, Iterable, Set, Callable

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QToolBar, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QCheckBox, QButtonGroup, QGroupBox, QDialog,
    QDialogButtonBox, QListWidget, QListWidgetItem, QStatusBar,
    QPlainTextEdit,
)


# ============================================================================
# MB-System beam-flag / edit-action constants
# ============================================================================
MBP_EDIT_FLAG = 1
MBP_EDIT_UNFLAG = 2
MBP_EDIT_ZERO = 3
MBP_EDIT_FILTER = 4

MB_FLAG_NONE = 0x00
MB_FLAG_NULL = 0x01
MB_FLAG_MANUAL = 0x02
MB_FLAG_FILTER = 0x04
MB_FLAG_FILTER2 = 0x08
MB_FLAG_SONAR = 0x10

MB_ESF_MULTIPLICITY_FACTOR = 1_000_000


def beam_ok(f: int) -> bool:
    return f == MB_FLAG_NONE


def beam_is_null(f: int) -> bool:
    return bool(f & MB_FLAG_NULL) and f != MB_FLAG_NONE and not (f & 0x1E)


def beam_is_manual(f: int) -> bool:
    return bool(f & MB_FLAG_MANUAL)


def beam_is_filter(f: int) -> bool:
    return bool(f & (MB_FLAG_FILTER | MB_FLAG_FILTER2))


def beam_is_sonar(f: int) -> bool:
    return bool(f & MB_FLAG_SONAR)


def set_flag_manual(f: int) -> int:
    return (MB_FLAG_MANUAL) & 0xFF


def set_flag_filter(f: int) -> int:
    return (MB_FLAG_FILTER2) & 0xFF


def set_flag_sonar(f: int) -> int:
    return (MB_FLAG_SONAR) & 0xFF


def set_flag_none(_f: int) -> int:
    return MB_FLAG_NONE


def set_flag_null(_f: int) -> int:
    return MB_FLAG_NULL


def beam_category(f: int) -> str:
    """Classify a beamflag byte for display purposes."""
    if beam_ok(f):
        return "good"
    if f == MB_FLAG_NULL:
        return "null"
    if beam_is_manual(f):
        return "manual"
    if beam_is_filter(f):
        return "filter"
    if beam_is_sonar(f):
        return "sonar"
    return "manual"


FLAG_COLORS = {
    "good": "k",
    "manual": "r",
    "filter": "b",
    "sonar": "g",
    "null": None,  # not plotted
}

DETECT_NAMES = ["Unknown", "Amplitude", "Phase", "Lidar", "Photogrammetry"]
PULSE_NAMES = ["Unknown", "CW", "Up-Chirp", "Down-Chirp", "Lidar"]


# ============================================================================
# FBT record constants
# ============================================================================
FBT_RECORD_OLD = 28270       # b"nn"
FBT_RECORD_V4 = 22068        # b"V4"
FBT_RECORD_V5 = 22069        # b"V5"
FBT_RECORD_COMMENT = 25443   # b"cc"


# ============================================================================
# Enumerations mirroring mbedit.c's mode defines
# ============================================================================
class ViewMode(IntEnum):
    WATERFALL = 0
    ALONGTRACK = 1
    ACROSSTRACK = 2


class ShowMode(IntEnum):
    FLAG = 0
    DETECT = 1
    PULSE = 2


class EditMode(IntEnum):
    TOGGLE = 0
    PICK = 1
    ERASE = 2
    RESTORE = 3
    GRAB = 4
    INFO = 5


# ============================================================================
# Data model
# ============================================================================
@dataclass
class Ping:
    time_i: Tuple[int, int, int, int, int, int, int]
    time_d: float
    lon: float
    lat: float
    heading: float
    speed: float
    altitude: float
    sonardepth: float
    roll: float
    pitch: float
    heave: float
    beams_bath: int
    depth_scale: float
    distance_scale: float
    xtrack: np.ndarray       # meters
    along: np.ndarray        # meters
    depth: np.ndarray        # meters, positive down
    flags_orig: np.ndarray   # uint8 at buffer-load time (incl. pre-applied esf)
    flags_active: np.ndarray  # uint8, live editable state
    multiplicity: int = 0
    time_interval: float = 0.0
    distance: float = 0.0

    # cached plot positions (pixel + data), set by canvas each draw
    plot_x: Optional[np.ndarray] = None
    plot_y: Optional[np.ndarray] = None
    label_xy: Optional[Tuple[float, float]] = None
    outbounds: bool = False


@dataclass
class Swath:
    src_path: str
    fbt_path: str
    pings: List[Ping] = field(default_factory=list)
    multiplicity_map: Dict[float, Dict[int, int]] = field(default_factory=dict)
    esf_path: str = ""
    esf_fh = None  # open binary file handle, append mode
    changed_this_session: Set[Tuple[int, int]] = field(default_factory=set)

    def rebuild_multiplicity(self):
        buckets: Dict[float, List[int]] = {}
        for i, pg in enumerate(self.pings):
            buckets.setdefault(pg.time_d, []).append(i)
        self.multiplicity_map.clear()
        for t, idxs in buckets.items():
            table = {i: mult for mult, i in enumerate(sorted(idxs))}
            self.multiplicity_map[t] = table
            for i in idxs:
                self.pings[i].multiplicity = table[i]

    def compute_derived(self):
        prev_distance = 0.0
        for i, pg in enumerate(self.pings):
            pg.time_interval = 0.0 if i == 0 else (pg.time_d - self.pings[i - 1].time_d)
            pg.distance = prev_distance + pg.speed * pg.time_interval / 3.6
            prev_distance = pg.distance

    def multiplicity_for(self, ping_index: int) -> int:
        return self.pings[ping_index].multiplicity


def _time_i_from_time_d(time_d: float) -> Tuple[int, int, int, int, int, int, int]:
    dt = datetime.datetime.fromtimestamp(time_d, datetime.timezone.utc)
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)


# ============================================================================
# FBT file reader
# ============================================================================
class FBTReader:
    # candidate sample sizes (amp bytes, ss bytes, number of ss arrays); (2, 2, 3) is expected
    _LAYOUTS = [(2, 2, 3), (1, 1, 3), (2, 1, 3), (1, 2, 3), (2, 2, 1), (1, 1, 1), (2, 2, 2)]

    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def _label_ok(data, pos, size):
        if pos >= size - 1:
            return True
        (t,) = struct.unpack_from(">H", data, pos)
        return t in (FBT_RECORD_COMMENT, FBT_RECORD_OLD, FBT_RECORD_V4, FBT_RECORD_V5)

    def read(self) -> List[Ping]:
        pings: List[Ping] = []
        with open(self.path, "rb") as f:
            data = f.read()
        size = len(data)
        pos = 0
        while pos < size:
            if size - pos < 2:
                break
            (rtype,) = struct.unpack_from(">H", data, pos)
            rec_start = pos
            pos += 2
            if rtype == FBT_RECORD_COMMENT:
                pos += 128
                continue
            elif rtype in (FBT_RECORD_V4, FBT_RECORD_V5):
                fmt = ">dddddfffffffhhhhffBBBB" if rtype == FBT_RECORD_V4 else ">dddddfffffffiiiiffBBBB"
                need = struct.calcsize(fmt)
                vals = struct.unpack_from(fmt, data, pos)
                pos += need
                (
                    time_d, lon, lat, sonardepth, altitude,
                    heading, speed, roll, pitch, heave,
                    _beam_xw, _beam_lw,
                    beams_bath, beams_amp, pixels_ss, _spare1,
                    depth_scale, distance_scale,
                    _ss_scalepower, _ss_type, _imagery_type, _topo_type,
                ) = vals
                nb, na, ns = int(beams_bath), int(beams_amp), int(pixels_ss)
                if not (0 <= nb <= 1_000_000 and 0 <= na <= 1_000_000 and 0 <= ns <= 1_000_000):
                    raise ValueError(f"Unreasonable counts bath={nb} amp={na} ss={ns} in record at offset {rec_start}")

                end = None
                for ab, sb, narr in self._LAYOUTS:
                    e = pos + nb + nb * 2 + nb * 4 + na * ab + ns * sb * narr
                    if e <= size and self._label_ok(data, e, size):
                        end = e
                        break
                if end is None:
                    raise ValueError(
                        f"Cannot determine ping layout at offset {rec_start}: bath={nb} amp={na} ss={ns}")

                # record body order: flags, bath, acrosstrack, alongtrack, amp, ss, ss_x, ss_a
                beamflag = np.frombuffer(data, dtype=np.uint8, count=nb, offset=pos).copy()
                p0 = pos + nb
                bath = np.frombuffer(data, dtype=">i2", count=nb, offset=p0).astype(np.float32)
                xtrack = np.frombuffer(data, dtype=">i2", count=nb, offset=p0 + nb * 2).astype(np.float32)
                along = np.frombuffer(data, dtype=">i2", count=nb, offset=p0 + nb * 4).astype(np.float32)
                pos = end

                # stored depths are relative to the sonar; mbio adds sensor depth
                depth_m = bath * depth_scale + float(sonardepth)
                xtrack_m = xtrack * distance_scale
                along_m = along * distance_scale

                ping = Ping(
                    time_i=_time_i_from_time_d(time_d),
                    time_d=float(time_d), lon=float(lon), lat=float(lat),
                    heading=float(heading), speed=float(speed),
                    altitude=float(altitude), sonardepth=float(sonardepth),
                    roll=float(roll), pitch=float(pitch), heave=float(heave),
                    beams_bath=nb,
                    depth_scale=float(depth_scale), distance_scale=float(distance_scale),
                    xtrack=xtrack_m, along=along_m, depth=depth_m,
                    flags_orig=beamflag.copy(), flags_active=beamflag.copy(),
                )
                pings.append(ping)
            elif rtype == FBT_RECORD_OLD:
                break
            else:
                raise ValueError(f"Unknown FBT record type {rtype} at file offset {rec_start}")
        return pings



# ============================================================================
# ESF I/O
# ============================================================================
@dataclass
class ESFEvent:
    time_d: float
    beam_with_mult: int
    action: int


def read_esf(path: str) -> List[ESFEvent]:
    events: List[ESFEvent] = []
    if not os.path.exists(path):
        return events
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16)
            if not chunk or len(chunk) < 16:
                break
            time_d, beam, action = struct.unpack(">dii", chunk)
            events.append(ESFEvent(time_d=time_d, beam_with_mult=beam, action=action))
    return events


def write_esf(path: str, events: Iterable[ESFEvent]) -> None:
    with open(path, "wb") as f:
        for ev in events:
            f.write(struct.pack(">dii", float(ev.time_d), int(ev.beam_with_mult), int(ev.action)))


def apply_esf_events(swath: Swath, events: List[ESFEvent]):
    """Apply a list of pre-existing ESF events to a freshly-loaded swath.

    Both flags_orig and flags_active are updated since these are treated as
    the "as found" state for this editing session.
    """
    time_map: Dict[float, List[int]] = {}
    for i, pg in enumerate(swath.pings):
        time_map.setdefault(pg.time_d, []).append(i)

    for ev in events:
        beam = ev.beam_with_mult % MB_ESF_MULTIPLICITY_FACTOR
        mult = ev.beam_with_mult // MB_ESF_MULTIPLICITY_FACTOR
        idxs = time_map.get(ev.time_d)
        if not idxs:
            continue
        pi = idxs[mult] if mult < len(idxs) else idxs[0]
        pg = swath.pings[pi]
        if beam < 0 or beam >= pg.beams_bath:
            continue
        if ev.action == MBP_EDIT_UNFLAG:
            newf = set_flag_none(pg.flags_active[beam])
        elif ev.action == MBP_EDIT_FLAG:
            newf = set_flag_manual(pg.flags_active[beam])
        elif ev.action == MBP_EDIT_FILTER:
            newf = set_flag_filter(pg.flags_active[beam])
        elif ev.action == MBP_EDIT_ZERO:
            newf = set_flag_null(pg.flags_active[beam])
        else:
            continue
        pg.flags_active[beam] = newf
        pg.flags_orig[beam] = newf


# ============================================================================
# MB-System helper calls
# ============================================================================
def ensure_fbt_exists(swath_path: str) -> str:
    if swath_path.lower().endswith(".fbt"):
        return swath_path
    fbt = swath_path + ".fbt"
    if os.path.exists(fbt):
        return fbt
    try:
        subprocess.run(["mbdatalist", "-I", swath_path, "-O"], check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError(f"Failed to generate .fbt via mbdatalist for {swath_path}: {e}")
    if not os.path.exists(fbt):
        raise FileNotFoundError(f"mbdatalist did not produce expected {fbt}")
    return fbt


def set_par_edit_flags(swath_path: str, esf_path: str) -> None:
    try:
        subprocess.run(
            ["mbset", "-I", swath_path, "-P", "EDITSAVEMODE:1",
             "-P", f"EDITSAVEFILE:{os.path.basename(esf_path)}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to update parameter file via mbset: {e}")


def run_mbprocess(swath_path: str) -> None:
    try:
        subprocess.run(["mbprocess", "-I", swath_path], check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError(f"mbprocess failed for {swath_path}: {e}")


def parse_datalist(path: str) -> List[str]:
    """Return a flat list of swath file paths referenced by `path`.

    If `path` looks like a single swath/fbt file it is returned as-is.
    Otherwise it's treated as an MB-System style datalist: lines of
    "relative/path format [weight]", '#' comments and blanks skipped,
    nested datalists expanded recursively.
    """
    if not os.path.isfile(path):
        return [path]

    lower = path.lower()
    if lower.endswith((".fbt", ".all", ".xtf", ".gsf", ".kmall")):
        return [path]

    base_dir = os.path.dirname(os.path.abspath(path))
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return [path]

    looks_like_datalist = False
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if len(s.split()) >= 2:
            looks_like_datalist = True
        break
    if not looks_like_datalist:
        return [path]

    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        rel = parts[0]
        candidate = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if os.path.isfile(candidate):
            out.extend(parse_datalist(candidate))
    return out if out else [path]


# ============================================================================
# Filters — operate in-place on pings; `record(i, j, action)` is called
# AFTER the caller updates pg.flags_active[j], so it can log/persist.
# ============================================================================
RecordFn = Callable[[int, int, int], None]


def filter_median_spike(pings: List[Ping], threshold_pct: float,
                         xtrack_win: int, ltrack_win: int, record: RecordFn) -> int:
    n = 0
    nping = len(pings)
    for i, pg in enumerate(pings):
        istart = max(0, i - ltrack_win // 2)
        iend = min(nping - 1, i + ltrack_win // 2)
        for j in range(pg.beams_bath):
            f = pg.flags_active[j]
            if not beam_ok(f):
                continue
            jstart = max(0, j - xtrack_win // 2)
            jend = min(pg.beams_bath - 1, j + xtrack_win // 2)
            vals = []
            for ii in range(istart, iend + 1):
                pg2 = pings[ii]
                jj_end = min(jend, pg2.beams_bath - 1)
                for jj in range(jstart, jj_end + 1):
                    if beam_ok(pg2.flags_active[jj]):
                        vals.append(pg2.depth[jj])
            if not vals:
                continue
            med = float(np.median(vals))
            alt = pg.altitude if pg.altitude else 1.0
            if alt == 0:
                continue
            if 100.0 * abs(pg.depth[j] - med) / abs(alt) > threshold_pct:
                pg.flags_active[j] = set_flag_filter(f)
                record(i, j, MBP_EDIT_FILTER)
                n += 1
    return n


def filter_wrongside(pings: List[Ping], threshold_beams: int, record: RecordFn) -> int:
    n = 0
    for i, pg in enumerate(pings):
        nb = pg.beams_bath
        end1 = nb // 2 - threshold_beams
        for j in range(0, max(0, end1)):
            f = pg.flags_active[j]
            if beam_ok(f) and pg.xtrack[j] > 0.0:
                pg.flags_active[j] = set_flag_filter(f)
                record(i, j, MBP_EDIT_FILTER)
                n += 1
        start2 = nb // 2 + threshold_beams
        for j in range(min(nb, max(0, start2)), nb):
            f = pg.flags_active[j]
            if beam_ok(f) and pg.xtrack[j] < 0.0:
                pg.flags_active[j] = set_flag_filter(f)
                record(i, j, MBP_EDIT_FILTER)
                n += 1
    return n


def filter_cutbeam(pings: List[Ping], begin: int, end: int, record: RecordFn) -> int:
    n = 0
    for i, pg in enumerate(pings):
        nb = pg.beams_bath
        if begin <= end:
            for j in range(max(0, begin), min(nb, end)):
                f = pg.flags_active[j]
                if beam_ok(f):
                    pg.flags_active[j] = set_flag_filter(f)
                    record(i, j, MBP_EDIT_FILTER)
                    n += 1
        else:
            for j in range(nb):
                if (j <= end or j >= begin) and beam_ok(pg.flags_active[j]):
                    pg.flags_active[j] = set_flag_filter(pg.flags_active[j])
                    record(i, j, MBP_EDIT_FILTER)
                    n += 1
    return n


def filter_cutdistance(pings: List[Ping], begin: float, end: float, record: RecordFn) -> int:
    n = 0
    for i, pg in enumerate(pings):
        for j in range(pg.beams_bath):
            f = pg.flags_active[j]
            if not beam_ok(f):
                continue
            x = pg.xtrack[j]
            inside = (begin <= x <= end) if begin <= end else (x >= begin or x <= end)
            if inside:
                pg.flags_active[j] = set_flag_filter(f)
                record(i, j, MBP_EDIT_FILTER)
                n += 1
    return n


def filter_cutangle(pings: List[Ping], begin: float, end: float, record: RecordFn) -> int:
    n = 0
    for i, pg in enumerate(pings):
        if not pg.altitude:
            continue
        for j in range(pg.beams_bath):
            f = pg.flags_active[j]
            if not beam_ok(f):
                continue
            ang = float(np.degrees(np.arctan2(pg.xtrack[j], pg.altitude)))
            inside = (begin <= ang <= end) if begin <= end else (ang >= begin or ang <= end)
            if inside:
                pg.flags_active[j] = set_flag_filter(f)
                record(i, j, MBP_EDIT_FILTER)
                n += 1
    return n


# ============================================================================
# Dialogs
# ============================================================================
class GoToTimeDialog(QDialog):
    def __init__(self, parent, default_dt: datetime.datetime):
        super().__init__(parent)
        self.setWindowTitle("Go To Specified Time")
        grid = QGridLayout(self)
        self.sp_year = QSpinBox(); self.sp_year.setRange(1970, 2100)
        self.sp_month = QSpinBox(); self.sp_month.setRange(1, 12)
        self.sp_day = QSpinBox(); self.sp_day.setRange(1, 31)
        self.sp_hour = QSpinBox(); self.sp_hour.setRange(0, 23)
        self.sp_minute = QSpinBox(); self.sp_minute.setRange(0, 59)
        self.sp_second = QSpinBox(); self.sp_second.setRange(0, 59)
        for sp, val in ((self.sp_year, default_dt.year), (self.sp_month, default_dt.month),
                        (self.sp_day, default_dt.day), (self.sp_hour, default_dt.hour),
                        (self.sp_minute, default_dt.minute), (self.sp_second, default_dt.second)):
            sp.setValue(val)
        labels = ["Year:", "Month:", "Day:", "Hour:", "Minute:", "Second:"]
        widgets = [self.sp_year, self.sp_month, self.sp_day, self.sp_hour, self.sp_minute, self.sp_second]
        for r, (lab, w) in enumerate(zip(labels, widgets)):
            grid.addWidget(QLabel(lab), r, 0)
            grid.addWidget(w, r, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        grid.addWidget(bb, len(labels), 0, 1, 2)

    def get_datetime(self) -> datetime.datetime:
        return datetime.datetime(self.sp_year.value(), self.sp_month.value(), self.sp_day.value(),
                                  self.sp_hour.value(), self.sp_minute.value(), self.sp_second.value())


class AnnotationDialog(QDialog):
    def __init__(self, parent, plot_width, exagger, x_interval, y_interval):
        super().__init__(parent)
        self.setWindowTitle("Annotation")
        grid = QGridLayout(self)
        self.sp_width = QDoubleSpinBox(); self.sp_width.setRange(1, 1_000_000); self.sp_width.setValue(plot_width)
        self.sp_exag = QDoubleSpinBox(); self.sp_exag.setRange(0.01, 50); self.sp_exag.setDecimals(2); self.sp_exag.setValue(exagger)
        self.sp_xint = QDoubleSpinBox(); self.sp_xint.setRange(1, 100_000); self.sp_xint.setValue(x_interval)
        self.sp_yint = QDoubleSpinBox(); self.sp_yint.setRange(1, 100_000); self.sp_yint.setValue(y_interval)
        rows = [("Acrosstrack width (m):", self.sp_width),
                ("Vertical exaggeration:", self.sp_exag),
                ("X axis tick interval (m):", self.sp_xint),
                ("Y axis tick interval (m):", self.sp_yint)]
        for r, (lab, w) in enumerate(rows):
            grid.addWidget(QLabel(lab), r, 0)
            grid.addWidget(w, r, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        grid.addWidget(bb, len(rows), 0, 1, 2)

    def values(self):
        return (self.sp_width.value(), self.sp_exag.value(), self.sp_xint.value(), self.sp_yint.value())


class FiltersDialog(QDialog):
    def __init__(self, parent, defaults: dict):
        super().__init__(parent)
        self.setWindowTitle("Filters")
        v = QVBoxLayout(self)

        g1 = QGroupBox("Median Spike Filter"); g1.setCheckable(True)
        g1.setChecked(defaults.get("medianspike", False))
        l1 = QGridLayout(g1)
        self.sp_med_thresh = QSpinBox(); self.sp_med_thresh.setRange(1, 100)
        self.sp_med_thresh.setValue(defaults.get("medianspike_threshold", 10))
        self.sp_med_x = QSpinBox(); self.sp_med_x.setRange(1, 999)
        self.sp_med_x.setValue(defaults.get("medianspike_xtrack", 5))
        self.sp_med_l = QSpinBox(); self.sp_med_l.setRange(1, 999)
        self.sp_med_l.setValue(defaults.get("medianspike_ltrack", 1))
        l1.addWidget(QLabel("% median depth threshold:"), 0, 0); l1.addWidget(self.sp_med_thresh, 0, 1)
        l1.addWidget(QLabel("Acrosstrack window (beams):"), 1, 0); l1.addWidget(self.sp_med_x, 1, 1)
        l1.addWidget(QLabel("Alongtrack window (pings):"), 2, 0); l1.addWidget(self.sp_med_l, 2, 1)
        self.g_median = g1
        v.addWidget(g1)

        g2 = QGroupBox("Wrong Side Filter"); g2.setCheckable(True)
        g2.setChecked(defaults.get("wrongside", False))
        l2 = QGridLayout(g2)
        self.sp_wrong = QSpinBox(); self.sp_wrong.setRange(0, 999)
        self.sp_wrong.setValue(defaults.get("wrongside_threshold", 15))
        l2.addWidget(QLabel("Beams from center threshold:"), 0, 0); l2.addWidget(self.sp_wrong, 0, 1)
        self.g_wrongside = g2
        v.addWidget(g2)

        g3 = QGroupBox("Flag by Beam Number"); g3.setCheckable(True)
        g3.setChecked(defaults.get("cutbeam", False))
        l3 = QGridLayout(g3)
        self.sp_cb_begin = QSpinBox(); self.sp_cb_begin.setRange(0, 9999)
        self.sp_cb_begin.setValue(defaults.get("cutbeam_begin", 0))
        self.sp_cb_end = QSpinBox(); self.sp_cb_end.setRange(0, 9999)
        self.sp_cb_end.setValue(defaults.get("cutbeam_end", 0))
        l3.addWidget(QLabel("Start beam:"), 0, 0); l3.addWidget(self.sp_cb_begin, 0, 1)
        l3.addWidget(QLabel("End beam:"), 1, 0); l3.addWidget(self.sp_cb_end, 1, 1)
        self.g_cutbeam = g3
        v.addWidget(g3)

        g4 = QGroupBox("Flag by Distance (m)"); g4.setCheckable(True)
        g4.setChecked(defaults.get("cutdistance", False))
        l4 = QGridLayout(g4)
        self.sp_cd_begin = QDoubleSpinBox(); self.sp_cd_begin.setRange(-100000, 100000)
        self.sp_cd_begin.setValue(defaults.get("cutdistance_begin", 0.0))
        self.sp_cd_end = QDoubleSpinBox(); self.sp_cd_end.setRange(-100000, 100000)
        self.sp_cd_end.setValue(defaults.get("cutdistance_end", 0.0))
        l4.addWidget(QLabel("Start distance:"), 0, 0); l4.addWidget(self.sp_cd_begin, 0, 1)
        l4.addWidget(QLabel("End distance:"), 1, 0); l4.addWidget(self.sp_cd_end, 1, 1)
        self.g_cutdistance = g4
        v.addWidget(g4)

        g5 = QGroupBox("Flag by Beam Angle (deg)"); g5.setCheckable(True)
        g5.setChecked(defaults.get("cutangle", False))
        l5 = QGridLayout(g5)
        self.sp_ca_begin = QDoubleSpinBox(); self.sp_ca_begin.setRange(-90, 90)
        self.sp_ca_begin.setValue(defaults.get("cutangle_begin", 0.0))
        self.sp_ca_end = QDoubleSpinBox(); self.sp_ca_end.setRange(-90, 90)
        self.sp_ca_end.setValue(defaults.get("cutangle_end", 0.0))
        l5.addWidget(QLabel("Start angle:"), 0, 0); l5.addWidget(self.sp_ca_begin, 0, 1)
        l5.addWidget(QLabel("End angle:"), 1, 0); l5.addWidget(self.sp_ca_end, 1, 1)
        self.g_cutangle = g5
        v.addWidget(g5)

        bb = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        self.btn_apply = bb.button(QDialogButtonBox.Apply)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        v.addWidget(bb)
        self.bb = bb

    def values(self) -> dict:
        return dict(
            medianspike=self.g_median.isChecked(),
            medianspike_threshold=self.sp_med_thresh.value(),
            medianspike_xtrack=self.sp_med_x.value(),
            medianspike_ltrack=self.sp_med_l.value(),
            wrongside=self.g_wrongside.isChecked(),
            wrongside_threshold=self.sp_wrong.value(),
            cutbeam=self.g_cutbeam.isChecked(),
            cutbeam_begin=self.sp_cb_begin.value(),
            cutbeam_end=self.sp_cb_end.value(),
            cutdistance=self.g_cutdistance.isChecked(),
            cutdistance_begin=self.sp_cd_begin.value(),
            cutdistance_end=self.sp_cd_end.value(),
            cutangle=self.g_cutangle.isChecked(),
            cutangle_begin=self.sp_ca_begin.value(),
            cutangle_end=self.sp_ca_end.value(),
        )


class FileListDialog(QDialog):
    editRequested = Signal(int)
    removeRequested = Signal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Files Available for Editing")
        v = QVBoxLayout(self)
        self.list = QListWidget()
        v.addWidget(self.list)
        h = QHBoxLayout()
        self.btn_edit = QPushButton("Edit Selected File")
        self.btn_remove = QPushButton("Remove Selected File")
        self.btn_close = QPushButton("Dismiss")
        h.addWidget(self.btn_edit); h.addWidget(self.btn_remove); h.addWidget(self.btn_close)
        v.addLayout(h)
        self.btn_edit.clicked.connect(self._on_edit)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_close.clicked.connect(self.accept)

    def set_files(self, paths: List[str], current_index: int):
        self.list.clear()
        for i, p in enumerate(paths):
            marker = "<loaded> " if i == current_index else "          "
            QListWidgetItem(f"{marker}{p}", self.list)
        if 0 <= current_index < len(paths):
            self.list.setCurrentRow(current_index)

    def _on_edit(self):
        row = self.list.currentRow()
        if row >= 0:
            self.editRequested.emit(row)

    def _on_remove(self):
        row = self.list.currentRow()
        if row >= 0:
            self.removeRequested.emit(row)


class AboutDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("About mbedit_qt")
        v = QVBoxLayout(self)
        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(
            "mbedit Qt6 version\n\n"
            "A PySide6 reimplementation of MB-System's interactive swath\n"
            "bathymetry editor (mbedit): view/edit modes, filters, buffer\n"
            "navigation, multi-file sessions, and continuous ESF edit-save\n"
            "output compatible with mbprocess.\n\n"
            "Qt6 version created by: Christian d. S. Ferreira, David W. Caress,\n"
            "and Dale N. Chayes, and the MB-System project.\n"
            "Original X11 mbedit: David W. Caress, Dale N. Chayes, and the\n"
            "MB-System project.\n"
        )
        v.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        v.addWidget(bb)
        


# ---- base editor logic carried over from v2 (its UI is replaced by MainWindowMotif below) ----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mbedit_qt — MB-System compatible bathymetry editor")
        self.resize(1300, 900)

        # ---- state -------------------------------------------------
        self.swath: Optional[Swath] = None
        self.file_list: List[str] = []
        self.file_index: int = -1

        self.current_id: int = 0
        self.nplot: int = 10
        self.step_size: int = 5

        self.view_mode: ViewMode = ViewMode.WATERFALL
        self.show_mode: ShowMode = ShowMode.FLAG
        self.show_flagged_soundings: bool = True
        self.show_flagged_profiles: bool = False

        self.plot_width: float = 2000.0
        self.exagger: float = 1.0
        self.x_interval: float = 250.0
        self.y_interval: float = 250.0

        self.reverse_mouse: bool = False
        self.reverse_keys: bool = False

        self.edit_mode: EditMode = EditMode.TOGGLE
        self.filters: dict = dict(
            medianspike=False, medianspike_threshold=10, medianspike_xtrack=5, medianspike_ltrack=1,
            wrongside=False, wrongside_threshold=15,
            cutbeam=False, cutbeam_begin=0, cutbeam_end=0,
            cutdistance=False, cutdistance_begin=0.0, cutdistance_end=0.0,
            cutangle=False, cutangle_begin=0.0, cutangle_end=0.0,
        )

        self.iping_save: Optional[int] = None
        self.jbeam_save: Optional[int] = None
        self.info_pingbeam: Optional[Tuple[int, int]] = None
        self.key_g_down: bool = False
        self._grab_rect_data: Optional[Tuple[float, float, float, float]] = None

        self.run_mbprocess_on_close = False

        # ---- central widget -----------------------------------------
        self.canvas = BathyCanvas(self)
        central = QWidget(self)
        v = QVBoxLayout(central)
        v.addWidget(self.canvas)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Pings shown:"))
        self.spin_nplot = QSpinBox(); self.spin_nplot.setRange(1, 200); self.spin_nplot.setValue(self.nplot)
        self.spin_nplot.valueChanged.connect(self._on_nplot_changed)
        ctrl.addWidget(self.spin_nplot)

        ctrl.addWidget(QLabel("Step:"))
        self.spin_step = QSpinBox(); self.spin_step.setRange(1, 200); self.spin_step.setValue(self.step_size)
        self.spin_step.valueChanged.connect(lambda val: setattr(self, "step_size", val))
        ctrl.addWidget(self.spin_step)

        ctrl.addWidget(QLabel("Width (m):"))
        self.spin_width = QDoubleSpinBox(); self.spin_width.setRange(1, 1_000_000); self.spin_width.setValue(self.plot_width)
        self.spin_width.valueChanged.connect(self._on_width_changed)
        ctrl.addWidget(self.spin_width)

        ctrl.addWidget(QLabel("V.E.:"))
        self.spin_exag = QDoubleSpinBox(); self.spin_exag.setRange(0.01, 50); self.spin_exag.setDecimals(2); self.spin_exag.setValue(self.exagger)
        self.spin_exag.valueChanged.connect(self._on_exag_changed)
        ctrl.addWidget(self.spin_exag)

        ctrl.addStretch(1)
        self.btn_start = QPushButton("Start"); self.btn_start.clicked.connect(self.action_start)
        self.btn_reverse = QPushButton("Reverse"); self.btn_reverse.clicked.connect(self.action_reverse)
        self.btn_forward = QPushButton("Forward"); self.btn_forward.clicked.connect(self.action_forward)
        self.btn_end = QPushButton("End"); self.btn_end.clicked.connect(self.action_end)
        self.btn_next_file = QPushButton("Done / Next File"); self.btn_next_file.clicked.connect(self.action_done)
        for b in (self.btn_start, self.btn_reverse, self.btn_forward, self.btn_end, self.btn_next_file):
            ctrl.addWidget(b)
        v.addLayout(ctrl)

        ping_actions = QHBoxLayout()
        for text, slot in [
            ("Bad Ping", self.action_bad_ping),
            ("Good Ping", self.action_good_ping),
            ("Left Ping", self.action_left_ping),
            ("Right Ping", self.action_right_ping),
            ("Zero Ping", self.action_zero_ping),
            ("Flag View", self.action_flag_view),
            ("Unflag View", self.action_unflag_view),
            ("Unflag Forward", self.action_unflag_all),
            ("Flag Out-of-Bounds", self.action_flag_outbounds),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            ping_actions.addWidget(btn)
        v.addLayout(ping_actions)

        self.setCentralWidget(central)

        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        self._build_menus_and_toolbar()
        self.canvas.setFocus()

    # ---------------------------------------------------------- menus
    def _build_menus_and_toolbar(self):
        menubar = self.menuBar()

        # ---------------- File menu ----------------
        m_file = menubar.addMenu("&File")
        act_open = QAction("Open…", self); act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.action_open)
        m_file.addAction(act_open)

        act_filelist = QAction("File Selection List…", self)
        act_filelist.triggered.connect(self.action_show_filelist)
        m_file.addAction(act_filelist)

        m_file.addSeparator()
        act_save = QAction("Save ESF (consolidated)…", self)
        act_save.triggered.connect(self.action_save_consolidated_esf)
        m_file.addAction(act_save)

        m_file.addSeparator()
        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.action_quit)
        m_file.addAction(act_quit)

        # ---------------- View menu ----------------
        m_view = menubar.addMenu("&View")

        grp_view = QActionGroup(self)
        for vm, label in [(ViewMode.WATERFALL, "Waterfall View"),
                           (ViewMode.ALONGTRACK, "Alongtrack View (overlay)"),
                           (ViewMode.ACROSSTRACK, "Acrosstrack View (along-track waterfall)")]:
            act = QAction(label, self, checkable=True)
            act.setChecked(vm == self.view_mode)
            act.triggered.connect(lambda checked, vm=vm: self._set_view_mode(vm))
            grp_view.addAction(act)
            m_view.addAction(act)
        m_view.addSeparator()

        self.act_show_flagged_soundings = QAction("Show Flagged Soundings", self, checkable=True)
        self.act_show_flagged_soundings.setChecked(self.show_flagged_soundings)
        self.act_show_flagged_soundings.triggered.connect(self._toggle_show_flagged_soundings)
        m_view.addAction(self.act_show_flagged_soundings)

        self.act_show_flagged_profiles = QAction("Show Flagged Profile", self, checkable=True)
        self.act_show_flagged_profiles.setChecked(self.show_flagged_profiles)
        self.act_show_flagged_profiles.triggered.connect(self._toggle_show_flagged_profiles)
        m_view.addAction(self.act_show_flagged_profiles)

        m_view.addSeparator()
        grp_show = QActionGroup(self)
        for sm, label in [(ShowMode.FLAG, "Show Flag States"),
                           (ShowMode.DETECT, "Show Bottom Detect Algorithms"),
                           (ShowMode.PULSE, "Show Source Pulse Types")]:
            act = QAction(label, self, checkable=True)
            act.setChecked(sm == self.show_mode)
            act.triggered.connect(lambda checked, sm=sm: self._set_show_mode(sm))
            grp_show.addAction(act)
            m_view.addAction(act)

        # ---------------- Edit Mode menu ----------------
        m_mode = menubar.addMenu("&Edit Mode")
        grp_mode = QActionGroup(self)
        for em, label, shortcut in [
            (EditMode.TOGGLE, "Toggle", "U"),
            (EditMode.PICK, "Pick", "I"),
            (EditMode.ERASE, "Erase", "O"),
            (EditMode.RESTORE, "Restore", "P"),
            (EditMode.GRAB, "Grab", "T"),
            (EditMode.INFO, "Info", "Y"),
        ]:
            act = QAction(label, self, checkable=True)
            act.setChecked(em == self.edit_mode)
            act.triggered.connect(lambda checked, em=em: self._set_edit_mode(em))
            grp_mode.addAction(act)
            m_mode.addAction(act)

        # ---------------- Controls menu ----------------
        m_ctrl = menubar.addMenu("&Controls")
        act_goto = QAction("Go To Specified Time…", self)
        act_goto.triggered.connect(self.action_goto_time)
        m_ctrl.addAction(act_goto)

        act_annot = QAction("Annotation…", self)
        act_annot.triggered.connect(self.action_annotation)
        m_ctrl.addAction(act_annot)

        act_filters = QAction("Filters…", self)
        act_filters.triggered.connect(self.action_filters)
        m_ctrl.addAction(act_filters)

        m_ctrl.addSeparator()
        self.act_reverse_keys = QAction("Reverse Right/Left Key Macros", self, checkable=True)
        self.act_reverse_keys.triggered.connect(lambda c: setattr(self, "reverse_keys", c))
        m_ctrl.addAction(self.act_reverse_keys)

        self.act_reverse_mouse = QAction("Reverse Mouse Buttons", self, checkable=True)
        self.act_reverse_mouse.triggered.connect(lambda c: setattr(self, "reverse_mouse", c))
        m_ctrl.addAction(self.act_reverse_mouse)

        self.act_run_mbprocess = QAction("Run mbprocess on Done/Next File", self, checkable=True)
        self.act_run_mbprocess.triggered.connect(lambda c: setattr(self, "run_mbprocess_on_close", c))
        m_ctrl.addAction(self.act_run_mbprocess)

        m_help = menubar.addMenu("&Help")
        act_about = QAction("About…", self)
        act_about.triggered.connect(lambda: AboutDialog(self).exec())
        m_help.addAction(act_about)

        # ---------------- Toolbar (navigation + quick edit) ----------
        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        tb.addAction(act_open)
        tb.addSeparator()
        self._add_toolbar_step(tb)

    def _add_toolbar_step(self, tb: QToolBar):
        act_prev = QAction("◀ Step", self); act_prev.triggered.connect(lambda: self.step(-self.step_size))
        act_next = QAction("Step ▶", self); act_next.triggered.connect(lambda: self.step(self.step_size))
        tb.addAction(act_prev); tb.addAction(act_next)
        
        
    # ======================================================== File mgmt
    def action_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open swath file, .fbt, or datalist", os.getcwd(),
            "All files (*)")
        if not path:
            return
        try:
            self.file_list = parse_datalist(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        if not self.file_list:
            QMessageBox.warning(self, "No files", "No files found to edit.")
            return
        self.file_index = 0
        self.load_current_file()

    def action_show_filelist(self):
        dlg = FileListDialog(self)
        dlg.set_files(self.file_list, self.file_index)

        def on_edit(row):
            dlg.accept()
            self.close_current_file()
            self.file_index = row
            self.load_current_file()

        def on_remove(row):
            if row == self.file_index:
                QMessageBox.information(self, "Cannot remove", "Cannot remove the currently loaded file.")
                return
            del self.file_list[row]
            if row < self.file_index:
                self.file_index -= 1
            dlg.set_files(self.file_list, self.file_index)

        dlg.editRequested.connect(on_edit)
        dlg.removeRequested.connect(on_remove)
        dlg.exec()

    def load_current_file(self):
        if not (0 <= self.file_index < len(self.file_list)):
            return
        path = self.file_list[self.file_index]
        try:
            if path.lower().endswith(".fbt"):
                fbt = path
                src = path[:-4] if os.path.exists(path[:-4]) else path
            else:
                src = path
                fbt = ensure_fbt_exists(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return

        try:
            pings = FBTReader(fbt).read()
        except Exception as e:
            QMessageBox.critical(self, "Read failed", f"Failed reading {fbt}:\n{e}")
            return
        if not pings:
            QMessageBox.warning(self, "No data", f"No pings were read from {fbt}")
            return

        swath = Swath(src_path=src, fbt_path=fbt, pings=pings)
        swath.rebuild_multiplicity()
        swath.compute_derived()
        swath.esf_path = src + ".esf"

        apply_previous = False
        if os.path.exists(swath.esf_path):
            ret = QMessageBox.question(
                self, "Existing edits found",
                f"An edit save file exists for this input:\n{swath.esf_path}\n\n"
                "Do you want to apply the saved edits before editing?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            apply_previous = (ret == QMessageBox.Yes)

        if os.path.exists(swath.esf_path):
            try:
                shutil.copyfile(swath.esf_path, swath.esf_path + ".tmp")
            except Exception:
                pass

        if apply_previous:
            try:
                events = read_esf(swath.esf_path)
                apply_esf_events(swath, events)
            except Exception as e:
                QMessageBox.warning(self, "ESF apply failed", f"Continuing without ESF: {e}")
            mode = "ab"
        else:
            mode = "wb"  # fresh start; .tmp backup already saved above

        try:
            swath.esf_fh = open(swath.esf_path, mode)
        except Exception as e:
            QMessageBox.warning(self, "ESF open failed", f"Could not open edit-save file: {e}")
            swath.esf_fh = None

        self.swath = swath
        self.current_id = 0
        self.iping_save = None
        self.jbeam_save = None
        self.info_pingbeam = None
        self.refresh_plot()
        self.setWindowTitle(f"mbedit_qt — {src}  [{self.file_index+1}/{len(self.file_list)}]")

    def close_current_file(self, write_par: bool = True):
        if not self.swath:
            return
        try:
            if self.swath.esf_fh:
                self.swath.esf_fh.flush()
                self.swath.esf_fh.close()
        except Exception:
            pass
        if write_par:
            try:
                set_par_edit_flags(self.swath.src_path, self.swath.esf_path)
            except Exception as e:
                QMessageBox.warning(self, "mbset warning", str(e))
            if self.run_mbprocess_on_close:
                try:
                    run_mbprocess(self.swath.src_path)
                except Exception as e:
                    QMessageBox.warning(self, "mbprocess warning", str(e))
        self.swath = None

    def action_done(self):
        self.close_current_file()
        if 0 <= self.file_index + 1 < len(self.file_list):
            self.file_index += 1
            self.load_current_file()
        else:
            self.file_index = -1
            self.canvas.clear()
            self.status.showMessage("No more files in session.")

    def action_quit(self):
        self.close_current_file()
        QApplication.instance().quit()

    def closeEvent(self, event):
        self.close_current_file()
        event.accept()

    # ======================================================== Navigation
    def _on_nplot_changed(self, val):
        self.nplot = val
        self.refresh_plot()

    def _on_width_changed(self, val):
        self.plot_width = val
        self.refresh_plot()

    def _on_exag_changed(self, val):
        self.exagger = val
        self.refresh_plot()

    def refresh_plot(self):
        if not self.swath:
            self.canvas.clear()
            self.status.showMessage("")
            return
        n = len(self.swath.pings)
        self.current_id = max(0, min(self.current_id, max(0, n - 1)))
        self.canvas.draw_swath()
        self._update_status()

    def _update_status(self):
        if not self.swath:
            return
        start = self.current_id
        end = min(len(self.swath.pings), start + self.nplot)
        total = flagged = 0
        for i in range(start, end):
            pg = self.swath.pings[i]
            total += pg.beams_bath
            flagged += int(np.sum(pg.flags_active != MB_FLAG_NONE))
        msg = f"Pings {start+1}-{end} / {len(self.swath.pings)} | beams {total} | flagged {flagged}"
        if self.info_pingbeam:
            i, j = self.info_pingbeam
            pg = self.swath.pings[i]
            msg += (f" | Ping {i} Beam {j}: depth={pg.depth[j]:.2f} xtrack={pg.xtrack[j]:.2f} "
                    f"along={pg.along[j]:.2f} lon={pg.lon:.5f} lat={pg.lat:.5f} "
                    f"hdg={pg.heading:.1f} spd={pg.speed:.1f}")
        self.status.showMessage(msg)

    def step(self, n: int):
        if not self.swath:
            return
        self.current_id = max(0, min(len(self.swath.pings) - 1, self.current_id + n))
        self.refresh_plot()

    def action_start(self):
        if self.swath:
            self.current_id = 0
            self.refresh_plot()

    def action_end(self):
        if self.swath:
            self.current_id = max(0, len(self.swath.pings) - self.nplot)
            self.refresh_plot()

    def action_reverse(self):
        if self.key_g_down:
            self.action_start()
        else:
            self.step(-self.step_size)

    def action_forward(self):
        if self.key_g_down:
            self.action_end()
        else:
            self.step(self.step_size)

    def action_goto_time(self):
        if not self.swath or not self.swath.pings:
            QMessageBox.information(self, "No data", "No file loaded.")
            return
        default_dt = datetime.datetime(*self.swath.pings[self.current_id].time_i[:6])
        dlg = GoToTimeDialog(self, default_dt)
        if dlg.exec() != QDialog.Accepted:
            return
        target = dlg.get_datetime()
        target_d = (target - datetime.datetime(1970, 1, 1)).total_seconds()
        idx = None
        for i, pg in enumerate(self.swath.pings):
            if pg.time_d >= target_d:
                idx = i
                break
        if idx is None:
            QMessageBox.warning(self, "Not found", "Target time is beyond the end of the data.")
            return
        self.current_id = idx
        self.refresh_plot()

    def action_annotation(self):
        dlg = AnnotationDialog(self, self.plot_width, self.exagger, self.x_interval, self.y_interval)
        if dlg.exec() == QDialog.Accepted:
            w, e, xi, yi = dlg.values()
            self.plot_width, self.exagger, self.x_interval, self.y_interval = w, e, xi, yi
            self.spin_width.blockSignals(True); self.spin_width.setValue(w); self.spin_width.blockSignals(False)
            self.spin_exag.blockSignals(True); self.spin_exag.setValue(e); self.spin_exag.blockSignals(False)
            self.refresh_plot()

    # ======================================================== Key macros
    def handle_key(self, key: str, pressed: bool):
        if key is None:
            return
        k = key.lower()
        if k == 'g':
            self.key_g_down = pressed
            return
        if not pressed:
            return

        if k in ('f',):
            self.action_forward()
        elif k in ('v',):
            self.action_reverse()
        elif k in ('m', 'z'):
            self.action_bad_ping()
        elif k in ('k', 's'):
            self.action_good_ping()
        elif k in ('j', 'a'):
            if not self.reverse_keys:
                self.action_left_ping()
            else:
                self.action_right_ping()
        elif k in ('l', 'd'):
            if not self.reverse_keys:
                self.action_right_ping()
            else:
                self.action_left_ping()
        elif k in ('x', ','):
            self.action_flag_view()
        elif k in ('c', '.'):
            self.action_unflag_view()
        elif k == '!':
            self.action_zero_ping()
        elif k in ('u', 'q'):
            self._set_edit_mode(EditMode.TOGGLE)
        elif k in ('i', 'w'):
            self._set_edit_mode(EditMode.PICK)
        elif k in ('o', 'e'):
            self._set_edit_mode(EditMode.ERASE)
        elif k in ('p', 'r'):
            self._set_edit_mode(EditMode.RESTORE)
        elif k in ('t', '['):
            self._set_edit_mode(EditMode.GRAB)
        elif k in ('y', ']'):
            self._set_edit_mode(EditMode.INFO)
        elif k == '2':
            self._set_view_mode(ViewMode.WATERFALL)
        elif k == '3':
            self._set_view_mode(ViewMode.ALONGTRACK)
        elif k == '4':
            self._set_view_mode(ViewMode.ACROSSTRACK)

    # ======================================================== Simple setters
    def _set_view_mode(self, vm: ViewMode):
        self.view_mode = vm
        self.refresh_plot()

    def _set_show_mode(self, sm: ShowMode):
        self.show_mode = sm
        self.refresh_plot()

    def _set_edit_mode(self, em: EditMode):
        self.edit_mode = em
        self.status.showMessage(f"Edit mode: {em.name}", 2000)

    def _toggle_show_flagged_soundings(self, checked: bool):
        self.show_flagged_soundings = checked
        self.refresh_plot()

    def _toggle_show_flagged_profiles(self, checked: bool):
        self.show_flagged_profiles = checked
        self.refresh_plot()
        
        # ======================================================== ESF logging
    def record_edit(self, ping_idx: int, beam_idx: int, action: int):
        if not self.swath:
            return
        pg = self.swath.pings[ping_idx]
        mult = self.swath.multiplicity_for(ping_idx)
        beam_with_mult = int(beam_idx + mult * MB_ESF_MULTIPLICITY_FACTOR)
        self.swath.changed_this_session.add((ping_idx, beam_idx))
        if self.swath.esf_fh:
            try:
                self.swath.esf_fh.write(struct.pack(">dii", pg.time_d, beam_with_mult, int(action)))
                self.swath.esf_fh.flush()
            except Exception:
                pass

    # ======================================================== Canvas callbacks
    def on_toggle(self, ping_idx: int, beam_idx: int):
        pg = self.swath.pings[ping_idx]
        f = pg.flags_active[beam_idx]
        if beam_ok(f):
            pg.flags_active[beam_idx] = set_flag_manual(f)
            self.record_edit(ping_idx, beam_idx, MBP_EDIT_FLAG)
        else:
            pg.flags_active[beam_idx] = set_flag_none(f)
            self.record_edit(ping_idx, beam_idx, MBP_EDIT_UNFLAG)
        self.iping_save, self.jbeam_save = ping_idx, beam_idx
        self.refresh_plot()

    def on_pick_flag(self, ping_idx: int, beam_idx: int):
        pg = self.swath.pings[ping_idx]
        pg.flags_active[beam_idx] = set_flag_manual(pg.flags_active[beam_idx])
        self.record_edit(ping_idx, beam_idx, MBP_EDIT_FLAG)
        self.iping_save, self.jbeam_save = ping_idx, beam_idx
        self.refresh_plot()

    def on_erase(self, ping_idx: int, beam_idx: int):
        pg = self.swath.pings[ping_idx]
        pg.flags_active[beam_idx] = set_flag_manual(pg.flags_active[beam_idx])
        self.record_edit(ping_idx, beam_idx, MBP_EDIT_FLAG)
        self.iping_save, self.jbeam_save = ping_idx, beam_idx
        self.refresh_plot()

    def on_restore(self, ping_idx: int, beam_idx: int):
        pg = self.swath.pings[ping_idx]
        pg.flags_active[beam_idx] = set_flag_none(pg.flags_active[beam_idx])
        self.record_edit(ping_idx, beam_idx, MBP_EDIT_UNFLAG)
        self.iping_save, self.jbeam_save = ping_idx, beam_idx
        self.refresh_plot()

    def on_info(self, ping_idx: int, beam_idx: int):
        self.info_pingbeam = (ping_idx, beam_idx)
        self._update_status()

    def finish_grab(self):
        if not self.swath or not self._grab_rect_data:
            return
        xmin, xmax, ymin, ymax = self._grab_rect_data
        start = self.current_id
        end = min(len(self.swath.pings), start + self.nplot)
        changed = 0
        for i in range(start, end):
            pg = self.swath.pings[i]
            if pg.plot_x is None:
                continue
            for j in range(pg.beams_bath):
                if not beam_ok(pg.flags_active[j]):
                    continue
                x, y = pg.plot_x[j], pg.plot_y[j]
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    pg.flags_active[j] = set_flag_manual(pg.flags_active[j])
                    self.record_edit(i, j, MBP_EDIT_FLAG)
                    changed += 1
                    self.iping_save, self.jbeam_save = i, j
        self._grab_rect_data = None
        if changed:
            self.refresh_plot()

    # ======================================================== Ping-level actions
    def _require_last_beam(self) -> bool:
        if self.iping_save is None or self.jbeam_save is None or not self.swath:
            self.status.showMessage("Pick/toggle a beam first.", 3000)
            return False
        return True

    def action_bad_ping(self):
        if not self._require_last_beam():
            return
        pg = self.swath.pings[self.iping_save]
        for j in range(pg.beams_bath):
            if beam_ok(pg.flags_active[j]):
                pg.flags_active[j] = set_flag_manual(pg.flags_active[j])
                self.record_edit(self.iping_save, j, MBP_EDIT_FLAG)
        self.refresh_plot()

    def action_good_ping(self):
        if not self._require_last_beam():
            return
        pg = self.swath.pings[self.iping_save]
        for j in range(pg.beams_bath):
            f = pg.flags_active[j]
            if f != MB_FLAG_NONE and f != MB_FLAG_NULL:
                pg.flags_active[j] = set_flag_none(f)
                self.record_edit(self.iping_save, j, MBP_EDIT_UNFLAG)
        self.refresh_plot()

    def action_left_ping(self):
        if not self._require_last_beam():
            return
        pg = self.swath.pings[self.iping_save]
        for j in range(0, self.jbeam_save + 1):
            if beam_ok(pg.flags_active[j]):
                pg.flags_active[j] = set_flag_manual(pg.flags_active[j])
                self.record_edit(self.iping_save, j, MBP_EDIT_FLAG)
        self.refresh_plot()

    def action_right_ping(self):
        if not self._require_last_beam():
            return
        pg = self.swath.pings[self.iping_save]
        for j in range(self.jbeam_save, pg.beams_bath):
            if beam_ok(pg.flags_active[j]):
                pg.flags_active[j] = set_flag_manual(pg.flags_active[j])
                self.record_edit(self.iping_save, j, MBP_EDIT_FLAG)
        self.refresh_plot()

    def action_zero_ping(self):
        if not self._require_last_beam():
            return
        pg = self.swath.pings[self.iping_save]
        for j in range(pg.beams_bath):
            pg.flags_active[j] = set_flag_null(pg.flags_active[j])
            self.record_edit(self.iping_save, j, MBP_EDIT_ZERO)
        self.refresh_plot()

    def action_flag_view(self):
        if not self.swath:
            return
        start = self.current_id
        end = min(len(self.swath.pings), start + self.nplot)
        for i in range(start, end):
            pg = self.swath.pings[i]
            for j in range(pg.beams_bath):
                if beam_ok(pg.flags_active[j]):
                    pg.flags_active[j] = set_flag_manual(pg.flags_active[j])
                    self.record_edit(i, j, MBP_EDIT_FLAG)
        self.refresh_plot()

    def action_unflag_view(self):
        if not self.swath:
            return
        start = self.current_id
        end = min(len(self.swath.pings), start + self.nplot)
        for i in range(start, end):
            pg = self.swath.pings[i]
            for j in range(pg.beams_bath):
                f = pg.flags_active[j]
                if f != MB_FLAG_NONE and f != MB_FLAG_NULL:
                    pg.flags_active[j] = set_flag_none(f)
                    self.record_edit(i, j, MBP_EDIT_UNFLAG)
        self.refresh_plot()

    def action_unflag_all(self):
        if not self.swath:
            return
        for i in range(self.current_id, len(self.swath.pings)):
            pg = self.swath.pings[i]
            for j in range(pg.beams_bath):
                f = pg.flags_active[j]
                if f != MB_FLAG_NONE and f != MB_FLAG_NULL:
                    pg.flags_active[j] = set_flag_none(f)
                    self.record_edit(i, j, MBP_EDIT_UNFLAG)
        self.refresh_plot()

    def action_flag_outbounds(self):
        if not self.swath:
            return
        start = self.current_id
        end = min(len(self.swath.pings), start + self.nplot)
        changed = 0
        half = self.plot_width / 2.0
        for i in range(start, end):
            pg = self.swath.pings[i]
            if not pg.outbounds:
                continue
            for j in range(pg.beams_bath):
                f = pg.flags_active[j]
                if beam_ok(f) and abs(pg.xtrack[j]) > half:
                    pg.flags_active[j] = set_flag_manual(f)
                    self.record_edit(i, j, MBP_EDIT_FLAG)
                    changed += 1
        if changed:
            self.refresh_plot()

    # ======================================================== Filters
    def action_filters(self):
        dlg = FiltersDialog(self, self.filters)

        def do_apply():
            self.filters = dlg.values()
            self._run_filters()

        dlg.btn_apply.clicked.connect(do_apply)
        dlg.exec()

    def _run_filters(self):
        if not self.swath:
            QMessageBox.information(self, "No data", "No file loaded.")
            return
        f = self.filters
        total = 0

        def rec(i, j, action):
            self.record_edit(i, j, action)

        if f["medianspike"]:
            total += filter_median_spike(self.swath.pings, f["medianspike_threshold"],
                                          f["medianspike_xtrack"], f["medianspike_ltrack"], rec)
        if f["wrongside"]:
            total += filter_wrongside(self.swath.pings, f["wrongside_threshold"], rec)
        if f["cutbeam"]:
            total += filter_cutbeam(self.swath.pings, f["cutbeam_begin"], f["cutbeam_end"], rec)
        if f["cutdistance"]:
            total += filter_cutdistance(self.swath.pings, f["cutdistance_begin"], f["cutdistance_end"], rec)
        if f["cutangle"]:
            total += filter_cutangle(self.swath.pings, f["cutangle_begin"], f["cutangle_end"], rec)

        self.refresh_plot()
        self.status.showMessage(f"Filters applied: {total} beams flagged.", 4000)

    # ======================================================== Save (consolidated)
    def action_save_consolidated_esf(self):
        """Rewrite the .esf file as a minimal, consolidated set of edit
        events reflecting the net difference between the original
        (as-loaded, including any previously-applied edits) and the
        current in-memory flag state.

        This is in addition to the continuous per-edit logging that
        already happens via record_edit()/self.swath.esf_fh during the
        session (which mirrors mbedit's live mb_esf_save() behavior and
        provides crash recovery). Running this consolidation is optional
        and mainly useful to compact a long session's edit history before
        final hand-off to mbprocess.
        """
        if not self.swath:
            QMessageBox.information(self, "No data", "No file loaded.")
            return

        events: List[ESFEvent] = []
        for i, pg in enumerate(self.swath.pings):
            mult = self.swath.multiplicity_for(i)
            for j in range(pg.beams_bath):
                orig = pg.flags_orig[j]
                cur = pg.flags_active[j]
                if orig == cur:
                    continue
                beam_with_mult = int(j + mult * MB_ESF_MULTIPLICITY_FACTOR)
                if cur == MB_FLAG_NONE:
                    action = MBP_EDIT_UNFLAG
                elif cur == MB_FLAG_NULL:
                    action = MBP_EDIT_ZERO
                elif beam_is_filter(cur):
                    action = MBP_EDIT_FILTER
                else:
                    action = MBP_EDIT_FLAG
                events.append(ESFEvent(time_d=pg.time_d, beam_with_mult=beam_with_mult, action=action))

        if not events:
            QMessageBox.information(self, "No changes", "No edits to save relative to the original data.")
            return

        esf_path = self.swath.esf_path

        # Close the live (continuously-written) handle before rewriting.
        try:
            if self.swath.esf_fh:
                self.swath.esf_fh.flush()
                self.swath.esf_fh.close()
                self.swath.esf_fh = None
        except Exception:
            pass

        # Preserve the raw session stream (every individual edit event,
        # including ones later undone) as a .stream backup, mirroring
        # mbedit's crash-recovery behavior, before writing the clean
        # consolidated version to the primary .esf path.
        try:
            if os.path.exists(esf_path):
                shutil.copyfile(esf_path, esf_path + ".stream")
        except Exception:
            pass

        try:
            write_esf(esf_path, events)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Failed writing {esf_path}:\n{e}")
            return

        # Reopen in append mode so further interactive edits keep being
        # logged on top of the consolidated baseline we just wrote.
        try:
            self.swath.esf_fh = open(esf_path, "ab")
        except Exception as e:
            QMessageBox.warning(self, "Reopen failed", f"Could not reopen edit-save file: {e}")

        try:
            set_par_edit_flags(self.swath.src_path, esf_path)
        except Exception as e:
            QMessageBox.warning(self, "mbset warning",
                                 f"Saved .esf, but failed to update .par via mbset:\n{e}")

        QMessageBox.information(self, "Saved",
                                 f"Wrote {esf_path} with {len(events)} consolidated edit events.")


# ============================================================================
# MBedit-Qt v3 : Motif-faithful GUI layer (QPainter canvas, no matplotlib)
# ============================================================================
import math
import sys
from enum import IntEnum
from PySide6.QtCore import Qt, QRect, QRectF, QPointF, QPoint, QSize, QTimer
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
                           QPolygonF, QAction, QActionGroup)
from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QVBoxLayout,
                               QHBoxLayout, QFrame, QLabel, QPushButton,
                               QRadioButton, QButtonGroup, QSlider, QMenu,
                               QStatusBar, QDialog, QDialogButtonBox,
                               QMessageBox, QSizePolicy, QFileDialog)


class PlotMode(IntEnum):
    WIDE = 0
    TIME = 1
    INTERVAL = 2
    LON = 3
    LAT = 4
    HEADING = 5
    SPEED = 6
    DEPTH = 7
    ALTITUDE = 8
    SENSORDEPTH = 9
    ROLL = 10
    PITCH = 11
    HEAVE = 12


TS_LABELS = {
    PlotMode.INTERVAL: "Time Interval (s)", PlotMode.LON: "Longitude (deg)",
    PlotMode.LAT: "Latitude (deg)", PlotMode.HEADING: "Heading (deg)",
    PlotMode.SPEED: "Speed (km/hr)", PlotMode.DEPTH: "Depth (m)",
    PlotMode.ALTITUDE: "Altitude (m)", PlotMode.SENSORDEPTH: "Sonar Depth (m)",
    PlotMode.ROLL: "Roll (deg)", PlotMode.PITCH: "Pitch (deg)",
    PlotMode.HEAVE: "Heave (m)",
}

COL_BLACK = QColor(0, 0, 0)
COL_RED = QColor(255, 0, 0)
COL_BLUE = QColor(0, 0, 255)
COL_GREEN = QColor(0, 190, 0)
CAT_COLORS = {"good": COL_BLACK, "manual": COL_RED, "filter": COL_BLUE, "sonar": COL_GREEN}


def center_depth(pg) -> float:
    n = pg.beams_bath
    if n > 0 and beam_ok(pg.flags_active[n // 2]):
        return float(pg.depth[n // 2])
    if n > 0:
        return float(pg.altitude + pg.sonardepth)
    return 0.0


def xtrack_slope_deg(pg) -> float:
    m = pg.flags_active == MB_FLAG_NONE
    if int(m.sum()) < 3:
        return 0.0
    x = pg.xtrack[m].astype(float)
    d = pg.depth[m].astype(float)
    xv = x - x.mean()
    den = float((xv * xv).sum())
    if den <= 0.0:
        return 0.0
    return math.degrees(math.atan(float((xv * (d - d.mean())).sum()) / den))


def ping_ts_value(pg, mode: PlotMode) -> float:
    if mode == PlotMode.INTERVAL:
        return float(pg.time_interval)
    if mode == PlotMode.LON:
        return float(pg.lon)
    if mode == PlotMode.LAT:
        return float(pg.lat)
    if mode == PlotMode.HEADING:
        return float(pg.heading)
    if mode == PlotMode.SPEED:
        return float(pg.speed)
    if mode == PlotMode.DEPTH:
        return center_depth(pg)
    if mode == PlotMode.ALTITUDE:
        return float(pg.altitude)
    if mode == PlotMode.SENSORDEPTH:
        return float(pg.sonardepth)
    if mode == PlotMode.ROLL:
        return float(pg.roll)
    if mode == PlotMode.PITCH:
        return float(pg.pitch)
    if mode == PlotMode.HEAVE:
        return float(pg.heave)
    return 0.0


def initial_plot_width_from_swath(swath) -> float:
    # legacy mbedit: full width = 2 * 1.2 * max |acrosstrack| of the data
    best = 0.0
    for pg in swath.pings:
        m = pg.flags_active == MB_FLAG_NONE
        if m.any():
            best = max(best, float(np.nanmax(np.abs(pg.xtrack[m]))))
    if best <= 0.0:
        for pg in swath.pings:
            m = pg.flags_active != MB_FLAG_NULL
            if m.any():
                best = max(best, float(np.nanmax(np.abs(pg.xtrack[m]))))
    if not np.isfinite(best) or best <= 0.0:
        return 5000.0
    return float(max(1, math.ceil(2.4 * best)))


def nice_step(rng: float, target: int = 8) -> float:
    raw = max(rng, 1e-9) / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for k in (1.0, 2.0, 2.5, 5.0, 10.0):
        if k * mag >= raw:
            return k * mag
    return 10.0 * mag


# ---------------------------------------------------------------- widgets
class ScaleSlider(QSlider):
    # integer QSlider exposing a float value, like an XmScale with decimalPoints
    def __init__(self, div=1, lo=1, hi=100, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.div = div
        self.hook = None
        QSlider.setRange(self, int(round(lo * div)), int(round(hi * div)))

    def value(self):
        return QSlider.value(self) / self.div

    def setValue(self, v):
        iv = int(round(float(v) * self.div))
        if iv > self.maximum():
            QSlider.setMaximum(self, iv)
        if iv < self.minimum():
            QSlider.setMinimum(self, iv)
        QSlider.setValue(self, iv)
        if self.hook:
            self.hook()

    def set_max(self, hi):
        QSlider.setMaximum(self, int(round(hi * self.div)))
        if self.hook:
            self.hook()


class _SliderBox(QWidget):
    def __init__(self, slider, valabel):
        super().__init__()
        self.s = slider
        self.v = valabel
        slider.setParent(self)
        valabel.setParent(self)
        self.setFixedHeight(38)
        self.setMinimumWidth(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def place(self):
        w = self.width()
        self.s.setGeometry(0, 18, w, 20)
        rng = self.s.maximum() - self.s.minimum()
        hw = 34
        frac = (QSlider.value(self.s) - self.s.minimum()) / rng if rng else 0.0
        cx = int(hw / 2 + frac * max(w - hw, 1))
        vw = self.v.sizeHint().width() + 8
        self.v.setGeometry(cx - vw // 2, 0, vw, 16)

    def resizeEvent(self, e):
        self.place()


class MotifScale(QWidget):
    def __init__(self, label, lo, hi, div, fmt, label_w):
        super().__init__()
        self.fmt = fmt
        self.div = div
        self.lbl = QLabel(label)
        self.lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl.setFixedWidth(label_w)
        self.lo_lbl = QLabel(fmt.format(lo))
        self.hi_lbl = QLabel(fmt.format(hi))
        self.hi_lbl.setMinimumWidth(52)
        self.slider = ScaleSlider(div, lo, hi)
        self.val = QLabel("")
        self.val.setAlignment(Qt.AlignCenter)
        self.box = _SliderBox(self.slider, self.val)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(self.lbl)
        h.addWidget(self.lo_lbl)
        h.addWidget(self.box, 1)
        h.addWidget(self.hi_lbl)
        self.slider.hook = self.refresh
        self.slider.valueChanged.connect(lambda _v: self.refresh())
        self.refresh()

    def refresh(self):
        self.val.setText(self.fmt.format(self.slider.value()))
        self.hi_lbl.setText(self.fmt.format(self.slider.maximum() / self.div))
        self.box.place()


class DiamondRadio(QRadioButton):
    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 34, 26)

    def paintEvent(self, ev):
        p = QPainter(self)
        h = self.height()
        cx, cy, r = 11.0, h / 2.0, 7.0
        poly = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                          QPointF(cx, cy + r), QPointF(cx - r, cy)])
        p.setPen(QPen(QColor("#404040"), 1))
        p.setBrush(QColor("#3c3c3c") if self.isChecked() else QColor("#c4c4c4"))
        p.drawPolygon(poly)
        p.setPen(Qt.black)
        p.setFont(self.font())
        p.drawText(QRect(26, 0, self.width() - 26, h), Qt.AlignVCenter | Qt.AlignLeft, self.text())
        p.end()


class BufferDialog(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.setWindowTitle("Buffer Controls")
        v = QVBoxLayout(self)
        self.size_sc = MotifScale("Data Buffer Size", 1, 5000, 1, "{:.0f}", 150)
        self.hold_sc = MotifScale("Buffer Retain Size", 1, 5000, 1, "{:.0f}", 150)
        self.size_sc.slider.setValue(mw.buffer_size)
        self.hold_sc.slider.setValue(mw.buffer_hold)
        v.addWidget(self.size_sc)
        v.addWidget(self.hold_sc)
        note = QLabel("Whole files are held in memory in this version; the values are stored "
                      "for parity with the X11 dialog.")
        note.setWordWrap(True)
        v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        v.addWidget(bb)
        self.accepted.connect(lambda: (setattr(mw, "buffer_size", int(self.size_sc.slider.value())),
                                       setattr(mw, "buffer_hold", int(self.hold_sc.slider.value()))))


# ---------------------------------------------------------------- canvas
class MBCanvas(QWidget):
    PICK_PIXELS = 18.0
    ERASE_PIXELS = 12.0

    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(700, 380)
        self.setAutoFillBackground(False)
        self.setCursor(Qt.CrossCursor)
        self._font = QFont("Courier New")
        self._font.setStyleHint(QFont.Monospace)
        self._font.setPixelSize(13)
        self._font.setBold(True)
        self._px = {}
        self._oob = {}
        self._zaps = []
        self._dragging = False
        self._grab_start = None
        self._grab_cur = None
        self._repeat_dir = 0
        self._repeat_timer = QTimer(self)
        self._repeat_timer.setInterval(220)
        self._repeat_timer.timeout.connect(self._on_repeat)

    # compatibility with the old matplotlib canvas API
    def clear(self):
        self._px.clear()
        self._oob.clear()
        self.update()

    def draw_swath(self):
        self.update()

    def draw_idle(self):
        self.update()

    def geometry_info(self):
        W, H = self.width(), self.height()
        m = W / 16.0
        xmin = (2 * m + 20) if self.mw.show_time == PlotMode.WIDE else 5 * m
        xmax = W - m
        ymin = 66.0
        ymax = H - 82.0
        if xmax - xmin < 50 or ymax - ymin < 50:
            return None
        return dict(W=W, H=H, m=m, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    xcen=(xmin + xmax) / 2.0)

    # ------------------------------------------------------------ painting
    def paintEvent(self, ev):
        mw = self.mw
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.white)
        self._zaps = []
        self._px = {}
        self._oob = {}
        g = self.geometry_info()
        if g is None:
            p.end()
            return
        p.setFont(self._font)
        fm = QFontMetrics(self._font)
        xscale = (g["xmax"] - g["xmin"]) / max(mw.plot_width, 1e-6)
        yscale = xscale * mw.exagger
        self._draw_file_header(p, g, fm)
        self._draw_titles(p, g, fm)
        self._draw_grid(p, g, fm, xscale, yscale)
        sw = mw.swath
        if sw and sw.pings:
            self._draw_pings(p, g, fm, xscale, yscale)
        self._draw_info(p, g, fm)
        if self._dragging and mw.edit_mode == EditMode.GRAB and self._grab_start and self._grab_cur:
            r = QRectF(QPointF(*self._grab_start), QPointF(*self._grab_cur)).normalized()
            p.setPen(QPen(COL_RED, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
        p.end()

    def _text_center(self, p, fm, xc, y, txt, color=COL_BLACK):
        p.setPen(color)
        p.drawText(QPointF(xc - fm.horizontalAdvance(txt) / 2.0, y), txt)

    def _draw_file_header(self, p, g, fm):
        mw = self.mw
        m = g["m"]
        x0 = m / 2.0
        p.setPen(COL_BLACK)
        if mw.swath and 0 <= mw.file_index < len(mw.file_list):
            p.drawText(QPointF(x0, 14),
                       f"File {mw.file_index + 1} of {len(mw.file_list)}:"
                       f"{os.path.basename(mw.file_list[mw.file_index])}")
        n = len(mw.swath.pings) if mw.swath else 0
        bar = QRectF(x0, 22, 4 * m, m / 4.0 + 2)
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar)
        p.drawRect(bar.adjusted(-1, -1, 1, 1))
        if n > 0:
            fpdx = max(4 * m * mw.nplot / n, 5.0)
            fpx = x0 + 4 * m * mw.current_id / n
            fpx = min(fpx, x0 + 4 * m - fpdx)
            p.fillRect(QRectF(fpx, bar.y(), fpdx, bar.height()), QColor(200, 200, 200))
            p.drawRect(QRectF(fpx, bar.y(), fpdx, bar.height()))
        base = bar.bottom() - 2
        p.drawText(QPointF(x0 - fm.horizontalAdvance("0") - 6, base), "0")
        p.drawText(QPointF(x0 + 4 * m + 6, base), str(n))

    def _draw_titles(self, p, g, fm):
        mw = self.mw
        xc = g["xcen"]
        title = {ViewMode.WATERFALL: "Single Swath per Ping",
                 ViewMode.ALONGTRACK: "Overlaid Swaths: Acrosstrack Distance vs Depth",
                 ViewMode.ACROSSTRACK: "Overlaid Swaths: Alongtrack Distance vs Depth"}[mw.view_mode]
        self._text_center(p, fm, xc, 28, title)
        if mw.show_mode == ShowMode.FLAG:
            parts = [("Sounding Colors by Flagging:  ", COL_BLACK), ("Unflagged ", COL_BLACK),
                     ("Manual ", COL_RED), ("Filter ", COL_BLUE), ("Sonar", COL_GREEN)]
        elif mw.show_mode == ShowMode.DETECT:
            parts = [("Sounding Colors by Bottom Detect:  ", COL_BLACK), ("Amplitude ", COL_BLACK),
                     ("Phase ", COL_RED), ("Other", COL_GREEN)]
        else:
            parts = [("Sounding Colors by Pulse Type:  ", COL_BLACK), ("CW ", COL_BLACK),
                     ("Up-Chirp ", COL_RED), ("Down-Chirp ", COL_BLUE), ("Other", COL_GREEN)]
        total = sum(fm.horizontalAdvance(t) for t, _ in parts)
        x = xc - total / 2.0
        for t, c in parts:
            p.setPen(c)
            p.drawText(QPointF(x, 42), t)
            x += fm.horizontalAdvance(t)
        self._text_center(p, fm, xc, 56,
                          f"Vertical exaggeration: {mw.exagger:.2f}    All Distances and Depths in Meters")

    def _draw_grid(self, p, g, fm, xscale, yscale):
        mw = self.mw
        xmin, xmax, ymin, ymax, xcen = g["xmin"], g["xmax"], g["ymin"], g["ymax"], g["xcen"]
        p.setBrush(Qt.NoBrush)
        solid = QPen(COL_BLACK, 1)
        dash = QPen(COL_BLACK, 1, Qt.DashLine)
        p.setPen(solid)
        p.drawRect(QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
        xi = max(float(mw.x_interval), 1e-6)
        nx = int(0.5 * mw.plot_width / xi) + 1
        if nx <= 600:
            for i in range(nx):
                xx = i * xi * xscale
                for sgn in ((1, -1) if i else (1,)):
                    xp = xcen + sgn * xx
                    if xmin <= xp <= xmax:
                        p.setPen(dash)
                        p.drawLine(QPointF(xp, ymin), QPointF(xp, ymax))
                        lab = f"{i * xi:g}"
                        self._text_center(p, fm, xp, ymax + 16, lab)
        yi = max(float(mw.y_interval), 1e-6)
        ny = int((ymax - ymin) / max(yscale, 1e-9) / yi) + 1
        if ny <= 600:
            for i in range(ny):
                yy = ymax - i * yi * yscale
                if yy < ymin:
                    break
                p.setPen(dash)
                p.drawLine(QPointF(xmin, yy), QPointF(xmax, yy))
                p.setPen(COL_BLACK)
                p.drawText(QPointF(xmax + 5, yy + 4), f"{i * yi:g}")

    def _draw_pings(self, p, g, fm, xscale, yscale):
        mw = self.mw
        xmin, xmax, ymin, ymax, xcen, m = g["xmin"], g["xmax"], g["ymin"], g["ymax"], g["xcen"], g["m"]
        pings = mw.swath.pings
        n = len(pings)
        start = min(max(mw.current_id, 0), n - 1)
        end = min(n, start + max(mw.nplot, 1))
        nplot = end - start
        vals = [pings[i].depth[pings[i].flags_active == MB_FLAG_NONE] for i in range(start, end)]
        vals = [v for v in vals if len(v)]
        if not vals:
            vals = [pings[i].depth[pings[i].flags_active != MB_FLAG_NULL] for i in range(start, end)]
            vals = [v for v in vals if len(v)]
        median = float(np.median(np.concatenate(vals))) if vals else 0.0
        dy = (ymax - ymin) / max(nplot, 1)
        dist_c = pings[min(n - 1, start + nplot // 2)].distance
        ycen = (ymin + ymax) / 2.0
        plot_rect = QRect(int(xmin), int(ymin), int(xmax - xmin) + 1, int(ymax - ymin) + 1)
        info = mw.info_pingbeam
        label_x = xmin - 5
        for k, i in enumerate(range(start, end)):
            pg = pings[i]
            f = pg.flags_active
            usable = f != MB_FLAG_NULL
            yslot = ymax - dy / 2.0 - k * dy
            if mw.view_mode == ViewMode.WATERFALL:
                xs = xcen + xscale * pg.xtrack.astype(float)
                ys = yslot + yscale * (pg.depth.astype(float) - median)
            elif mw.view_mode == ViewMode.ALONGTRACK:
                xs = xcen + xscale * pg.xtrack.astype(float)
                ys = ycen + yscale * (pg.depth.astype(float) - median)
            else:
                xs = xcen + xscale * (pg.along.astype(float) + pg.distance - dist_c)
                ys = ycen + yscale * (pg.depth.astype(float) - median)
            xs = np.clip(xs, -1e6, 1e6)
            ys = np.clip(ys, -1e6, 1e6)
            self._px[i] = (xs, ys)
            pg.plot_x = (xs - xcen) / max(xscale, 1e-12)
            pg.plot_y = ys
            oob = usable & ((xs < xmin) | (xs > xmax) | (ys < ymin) | (ys > ymax))
            self._oob[i] = oob
            pg.outbounds = 0
            if oob.any():
                pg.outbounds = 2 if (oob & (f == MB_FLAG_NONE)).any() else 1

            p.save()
            p.setClipRect(plot_rect)
            xi_ = np.rint(xs).astype(int)
            yi_ = np.rint(ys).astype(int)
            good = f == MB_FLAG_NONE
            gidx = np.nonzero(good)[0]
            if len(gidx) > 1:
                p.setPen(QPen(COL_BLACK, 1))
                p.drawPolyline(QPolygonF([QPointF(xs[j], ys[j]) for j in gidx]))
            if mw.show_flagged_profiles:
                nidx = np.nonzero(usable)[0]
                p.setPen(QPen(COL_RED, 1))
                for a, b in zip(nidx[:-1], nidx[1:]):
                    if not (good[a] and good[b]):
                        p.drawLine(QPointF(xs[a], ys[a]), QPointF(xs[b], ys[b]))
            if len(gidx):
                p.setPen(Qt.NoPen)
                p.setBrush(COL_BLACK)
                p.drawRects([QRect(int(xi_[j]) - 2, int(yi_[j]) - 2, 4, 4) for j in gidx])
            if mw.show_flagged_soundings:
                p.setBrush(Qt.NoBrush)
                groups = {}
                for j in np.nonzero(usable & ~good)[0]:
                    groups.setdefault(beam_category(f[j]), []).append(int(j))
                for cat, js in groups.items():
                    p.setPen(QPen(CAT_COLORS.get(cat, COL_RED), 1))
                    p.drawRects([QRect(int(xi_[j]) - 2, int(yi_[j]) - 2, 4, 4) for j in js])
            if info and info[0] == i and 0 <= info[1] < pg.beams_bath and usable[info[1]]:
                p.setPen(Qt.NoPen)
                p.setBrush(COL_BLUE)
                p.drawRect(QRect(int(xi_[info[1]]) - 4, int(yi_[info[1]]) - 4, 8, 8))
            p.restore()

            rec = i + 1
            ti = pg.time_i
            cd = center_depth(pg)
            if mw.show_time == PlotMode.TIME:
                txt = (f"{rec:5d} {int(ti[1]):02d}/{int(ti[2]):02d}/{int(ti[0]):04d} "
                       f"{int(ti[3]):02d}:{int(ti[4]):02d}:{int(ti[5]):02d}."
                       f"{int(0.001 * ti[6]):03d} {cd:10.3f}")
            else:
                txt = f"{rec:5d} {cd:10.3f}"
            tw = fm.horizontalAdvance(txt)
            asc, desc = fm.ascent(), fm.descent()
            lx = label_x - tw
            rect = QRectF(lx, yslot - asc / 2.0 - 1, tw, asc + desc + 2)
            if pg.outbounds == 2:
                p.fillRect(rect, COL_RED)
            elif pg.outbounds == 1:
                p.fillRect(rect, COL_GREEN)
            if pg.outbounds:
                zr = QRect(int(lx - 20), int(rect.y()), 10, int(rect.height()))
                p.setPen(QPen(COL_BLACK, 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(zr)
                self._zaps.append((zr.adjusted(-2, -2, 2, 2), i))
            p.setPen(COL_BLACK)
            p.drawText(QPointF(lx, yslot + asc / 2.0 - 1), txt)

        if mw.show_time > PlotMode.TIME:
            self._draw_timeseries(p, g, fm, start, end, dy)

    def _draw_timeseries(self, p, g, fm, start, end, dy):
        mw = self.mw
        m, ymin, ymax = g["m"], g["ymin"], g["ymax"]
        pings = mw.swath.pings
        mode = PlotMode(mw.show_time)
        rows = list(range(start, end))
        v = [ping_ts_value(pings[i], mode) for i in rows]
        extra_s, extra_b = [], []
        if mode == PlotMode.ROLL:
            extra_s = [xtrack_slope_deg(pings[i]) for i in rows]
            extra_b = [a - b for a, b in zip(v, extra_s)]
        allv = v + extra_s + extra_b
        tsmin, tsmax = min(allv), max(allv)
        if tsmax - tsmin < 1e-9:
            tsmin -= 1.0
            tsmax += 1.0
        pad = 0.05 * (tsmax - tsmin)
        tsmin -= pad
        tsmax += pad
        tsscale = 2.0 * m / (tsmax - tsmin)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(COL_BLACK, 1))
        p.drawRect(QRectF(m / 2.0, ymin, 2 * m, ymax - ymin))
        p.setPen(QPen(COL_BLACK, 1, Qt.DashLine))
        for fx in (m, 1.5 * m, 2 * m):
            p.drawLine(QPointF(fx, ymin), QPointF(fx, ymax))
        p.setPen(COL_BLACK)
        self._text_center(p, fm, 1.5 * m, ymin - 4, TS_LABELS.get(mode, ""))
        self._text_center(p, fm, m / 2.0, ymax + 16, f"{tsmin:.4g}")
        self._text_center(p, fm, 2.5 * m, ymax + 16, f"{tsmax:.4g}")

        def draw_series(series, color):
            pts = [QPointF(m / 2.0 + (val - tsmin) * tsscale, ymax - dy / 2.0 - k * dy)
                   for k, val in enumerate(series)]
            p.setPen(QPen(color, 1))
            if len(pts) > 1:
                p.drawPolyline(QPolygonF(pts))
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            for q in pts:
                p.drawRect(QRectF(q.x() - 2, q.y() - 2, 4, 4))

        draw_series(v, COL_BLACK)
        if mode == PlotMode.ROLL:
            draw_series(extra_s, COL_RED)
            draw_series(extra_b, COL_BLUE)

    def _draw_info(self, p, g, fm):
        mw = self.mw
        info = mw.info_pingbeam
        if not (info and mw.swath):
            return
        i, j = info
        if not (0 <= i < len(mw.swath.pings)):
            return
        pg = mw.swath.pings[i]
        if not (0 <= j < pg.beams_bath):
            return
        ti = pg.time_i
        xc = g["xcen"]
        y0 = g["ymax"] + 36
        l1 = (f"Ping {i + 1} Beam {j} Time {int(ti[1]):02d}/{int(ti[2]):02d}/{int(ti[0]):04d} "
              f"{int(ti[3]):02d}:{int(ti[4]):02d}:{int(ti[5]):02d}.{int(0.001 * ti[6]):03d}")
        l2 = f"Lon {pg.lon:.5f} deg Lat {pg.lat:.5f} deg Hdg {pg.heading:.1f} deg Spd {pg.speed:.1f} km/hr"
        l3 = (f"Depth {pg.depth[j]:.2f} X {pg.xtrack[j]:.2f} L {pg.along[j]:.2f} "
              f"Alt {pg.altitude:.2f}")
        for k, t in enumerate((l1, l2, l3)):
            self._text_center(p, fm, xc, y0 + 14 * k, t)

    # ------------------------------------------------------------- picking
    def _nearest_beam(self, px, py, max_px, predicate):
        mw = self.mw
        if not mw.swath:
            return None
        best, best_d = None, max_px
        start = mw.current_id
        end = min(len(mw.swath.pings), start + mw.nplot)
        for i in range(start, end):
            if i not in self._px:
                continue
            xs, ys = self._px[i]
            pg = mw.swath.pings[i]
            d = np.hypot(xs - px, ys - py)
            for j in np.nonzero(d < best_d)[0]:
                if predicate(pg.flags_active[j]) and d[j] < best_d:
                    best_d, best = float(d[j]), (i, int(j))
        return best

    def mousePressEvent(self, ev):
        self.setFocus()
        mw = self.mw
        btn = {Qt.LeftButton: 1, Qt.MiddleButton: 2, Qt.RightButton: 3}.get(ev.button())
        if btn is None:
            return
        pos = ev.position()
        x, y = pos.x(), pos.y()
        left_is_edit = (btn == 1 and not mw.reverse_mouse) or (btn == 3 and mw.reverse_mouse)
        fwd = (btn == 3 and not mw.reverse_mouse) or (btn == 1 and mw.reverse_mouse)
        if left_is_edit:
            for r, i in self._zaps:
                if r.contains(int(x), int(y)):
                    self._zap_ping(i)
                    return
            self._dragging = True
            self._edit(x, y, True)
        elif btn == 2:
            mw.step(-mw.step_size)
            self._repeat_dir = -1
            self._repeat_timer.start()
        elif fwd:
            mw.step(mw.step_size)
            self._repeat_dir = 1
            self._repeat_timer.start()

    def mouseMoveEvent(self, ev):
        if self._dragging:
            pos = ev.position()
            self._edit(pos.x(), pos.y(), False)

    def mouseReleaseEvent(self, ev):
        self._repeat_timer.stop()
        self._repeat_dir = 0
        if self._dragging:
            self._dragging = False
            if self.mw.edit_mode == EditMode.GRAB:
                self.mw.finish_grab()
            self._grab_start = self._grab_cur = None
            self.update()

    def wheelEvent(self, ev):
        d = ev.angleDelta().y()
        if d:
            self.mw.step(-self.mw.step_size if d > 0 else self.mw.step_size)

    def _on_repeat(self):
        if self._repeat_dir:
            self.mw.step(self._repeat_dir * self.mw.step_size)

    def _zap_ping(self, i):
        mw = self.mw
        pg = mw.swath.pings[i]
        oob = self._oob.get(i)
        if oob is None:
            return
        for j in np.nonzero(oob)[0]:
            if beam_ok(pg.flags_active[j]):
                pg.flags_active[j] = set_flag_manual(pg.flags_active[j])
                mw.record_edit(i, int(j), MBP_EDIT_FLAG)
        mw.refresh_plot()

    def _edit(self, px, py, start):
        mw = self.mw
        if not mw.swath:
            return
        mode = mw.edit_mode
        if mode == EditMode.GRAB:
            if start:
                self._grab_start = (px, py)
            self._grab_cur = (px, py)
            g = self.geometry_info()
            if g and self._grab_start:
                xscale = (g["xmax"] - g["xmin"]) / max(mw.plot_width, 1e-6)
                x0 = (self._grab_start[0] - g["xcen"]) / xscale
                x1 = (px - g["xcen"]) / xscale
                y0, y1 = self._grab_start[1], py
                mw._grab_rect_data = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            self.update()
            return
        if mode == EditMode.TOGGLE:
            if start:
                hit = self._nearest_beam(px, py, self.PICK_PIXELS, lambda f: f != MB_FLAG_NULL)
                if hit:
                    mw.on_toggle(*hit)
        elif mode == EditMode.PICK:
            if start:
                hit = self._nearest_beam(px, py, self.PICK_PIXELS, beam_ok)
                if hit:
                    mw.on_pick_flag(*hit)
        elif mode == EditMode.ERASE:
            hit = self._nearest_beam(px, py, self.ERASE_PIXELS, beam_ok)
            if hit:
                mw.on_erase(*hit)
        elif mode == EditMode.RESTORE:
            hit = self._nearest_beam(px, py, self.ERASE_PIXELS,
                                     lambda f: (not beam_ok(f)) and f != MB_FLAG_NULL)
            if hit:
                mw.on_restore(*hit)
        elif mode == EditMode.INFO:
            if start:
                hit = self._nearest_beam(px, py, self.PICK_PIXELS, lambda f: f != MB_FLAG_NULL)
                if hit:
                    mw.on_info(*hit)

    @staticmethod
    def _key_text(ev):
        t = ev.text()
        if t:
            return t
        k = ev.key()
        return chr(k).lower() if 32 <= k < 127 else ""

    def keyPressEvent(self, ev):
        t = self._key_text(ev)
        if not t:
            return super().keyPressEvent(ev)
        if ev.isAutoRepeat() and t.lower() == "g":
            return
        self.mw.handle_key(t, pressed=True)

    def keyReleaseEvent(self, ev):
        if ev.isAutoRepeat():
            return
        t = self._key_text(ev)
        if t:
            self.mw.handle_key(t, pressed=False)


# ---------------------------------------------------------------- main window
MOTIF_QSS = """
QMainWindow, QDialog { background: #bdbdbd; }
QFrame#hdr { background: #bdbdbd; }
QLabel { color: black; font: bold 13px "Helvetica", "Arial"; }
QPushButton { background: #c2c2c2; color: black; border: 2px outset #f2f2f2;
              padding: 2px 12px; min-height: 22px; font: bold 13px "Helvetica", "Arial"; }
QPushButton:pressed { border-style: inset; }
QPushButton:disabled { color: #8a8a8a; }
QPushButton::menu-indicator { image: none; width: 0px; }
QSlider::groove:horizontal { height: 10px; background: #9c9c9c; border: 1px solid #6c6c6c; }
QSlider::handle:horizontal { width: 34px; margin: -4px 0; background: #c8c8c8;
                             border: 2px outset #f2f2f2; }
QStatusBar { background: #bdbdbd; }
QMenu { background: #c2c2c2; color: black; border: 2px outset #f2f2f2; font: bold 13px "Helvetica", "Arial"; }
QMenu::item:selected { background: #8f8f8f; }
"""


class MainWindowMotif(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("MBedit")
        self.resize(1300, 900)
        self.swath = None
        self.file_list = []
        self.file_index = -1
        self.current_id = 0
        self.nplot = 10
        self.step_size = 5
        self.view_mode = ViewMode.WATERFALL
        self.show_mode = ShowMode.FLAG
        self.show_flagged_soundings = True
        self.show_flagged_profiles = False
        self.show_time = PlotMode.TIME
        self.plot_width = 5000.0
        self.exagger = 1.0
        self.x_interval = 1000.0
        self.y_interval = 250.0
        self.reverse_mouse = False
        self.reverse_keys = False
        self.edit_mode = EditMode.TOGGLE
        self.filters = dict(
            medianspike=False, medianspike_threshold=10, medianspike_xtrack=5, medianspike_ltrack=1,
            wrongside=False, wrongside_threshold=15,
            cutbeam=False, cutbeam_begin=0, cutbeam_end=0,
            cutdistance=False, cutdistance_begin=0.0, cutdistance_end=0.0,
            cutangle=False, cutangle_begin=0.0, cutangle_end=0.0)
        self.iping_save = None
        self.jbeam_save = None
        self.info_pingbeam = None
        self.key_g_down = False
        self._grab_rect_data = None
        self.run_mbprocess_on_close = False
        self.browse_only = False
        self.buffer_size = 5000
        self.buffer_hold = 100
        self._view_actions = {}
        self._show_actions = {}
        self._time_actions = {}

        self.canvas = MBCanvas(self)
        central = QWidget(self)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_header(), 0)
        v.addWidget(self.canvas, 1)
        self.setCentralWidget(central)
        self.status = QStatusBar(self)
        self.status.setSizeGripEnabled(False)
        self.setStatusBar(self.status)
        self.setStyleSheet(MOTIF_QSS)
        self.canvas.setFocus()

    def _menu_button(self, text, menu):
        b = QPushButton(text)
        b.setMenu(menu)
        return b

    def _act(self, menu, text, slot=None, checkable=False, checked=False, group=None):
        a = QAction(text, self)
        a.setCheckable(checkable)
        if checkable:
            a.setChecked(checked)
        if slot is not None:
            if checkable:
                a.triggered.connect(lambda c, s=slot: s(c))
            else:
                a.triggered.connect(lambda c=False, s=slot: s())
        if group is not None:
            group.addAction(a)
        menu.addAction(a)
        return a

    def _build_menus(self):
        mf = QMenu(self)
        self._act(mf, "Open Swath File / Datalist...", self.action_open)
        self._act(mf, "File Selection List...", self.action_show_filelist)
        mf.addSeparator()
        g = QActionGroup(self)
        self.act_out_edit = self._act(mf, "Output Edits", lambda c: self._set_browse(False),
                                      True, True, g)
        self.act_out_browse = self._act(mf, "Browse Only", lambda c: self._set_browse(True),
                                        True, False, g)
        mf.addSeparator()
        self._act(mf, "Save ESF (consolidated)...", self.action_save_consolidated_esf)
        self._act(mf, "Done / Next File", self.action_done)
        self._act(mf, "Quit", self.action_quit)

        mv = QMenu(self)
        self.act_show_flagged_soundings = self._act(
            mv, "Show Flagged Soundings", self._toggle_show_flagged_soundings, True, True)
        self.act_show_flagged_profiles = self._act(
            mv, "Show Flagged Profile", self._toggle_show_flagged_profiles, True, False)
        mv.addSeparator()
        g = QActionGroup(self)
        for sm, label in [(ShowMode.FLAG, "Show Flag States"),
                          (ShowMode.DETECT, "Show Bottom Detect Algorithms"),
                          (ShowMode.PULSE, "Show Source Pulse Types")]:
            self._show_actions[sm] = self._act(mv, label, lambda c, sm=sm: self._set_show_mode(sm),
                                               True, sm == self.show_mode, g)
        mv.addSeparator()
        g = QActionGroup(self)
        for pm, label in [(PlotMode.WIDE, "Wide Plot"), (PlotMode.TIME, "Time Stamps"),
                          (PlotMode.INTERVAL, "Time Interval"), (PlotMode.LON, "Longitude"),
                          (PlotMode.LAT, "Latitude"), (PlotMode.HEADING, "Heading"),
                          (PlotMode.SPEED, "Speed"), (PlotMode.DEPTH, "Depth"),
                          (PlotMode.ALTITUDE, "Altitude"), (PlotMode.SENSORDEPTH, "Sonar Depth"),
                          (PlotMode.ROLL, "Roll"), (PlotMode.PITCH, "Pitch"),
                          (PlotMode.HEAVE, "Heave")]:
            self._time_actions[pm] = self._act(mv, label, lambda c, pm=pm: self._set_show_time(pm),
                                               True, pm == self.show_time, g)
        mv.addSeparator()
        g = QActionGroup(self)
        for vm, label in [(ViewMode.WATERFALL, "Waterfall View"),
                          (ViewMode.ALONGTRACK, "Alongtrack View (overlay)"),
                          (ViewMode.ACROSSTRACK, "Acrosstrack View (along-track waterfall)")]:
            self._view_actions[vm] = self._act(mv, label, lambda c, vm=vm: self._set_view_mode(vm),
                                               True, vm == self.view_mode, g)

        mc = QMenu(self)
        self._act(mc, "Buffer Controls...", self.action_buffer)
        self._act(mc, "Annotation...", self.action_annotation)
        self._act(mc, "Filters...", self.action_filters)
        self._act(mc, "Go To Specified Time...", self.action_goto_time)
        self._act(mc, "Autoscale Width From Data", self.auto_scale_width)
        mc.addSeparator()
        for text, slot in [("Bad Ping", self.action_bad_ping), ("Good Ping", self.action_good_ping),
                           ("Left Ping", self.action_left_ping), ("Right Ping", self.action_right_ping),
                           ("Zero Ping", self.action_zero_ping),
                           ("Flag Out-of-Bounds", self.action_flag_outbounds)]:
            self._act(mc, text, slot)
        mc.addSeparator()
        self.act_reverse_keys = self._act(mc, "Reverse Right/Left Key Macros",
                                          lambda c: setattr(self, "reverse_keys", c), True, False)
        self.act_reverse_mouse = self._act(mc, "Reverse Mouse Buttons",
                                           lambda c: setattr(self, "reverse_mouse", c), True, False)
        self.act_run_mbprocess = self._act(mc, "Run mbprocess on Done/Next File",
                                           lambda c: setattr(self, "run_mbprocess_on_close", c),
                                           True, False)
        return mf, mv, mc

    def _build_header(self):
        hdr = QFrame()
        hdr.setObjectName("hdr")
        hdr.setFrameShape(QFrame.NoFrame)
        vl = QVBoxLayout(hdr)
        vl.setContentsMargins(6, 4, 6, 4)
        vl.setSpacing(2)
        mf, mv, mc = self._build_menus()
        self._menus = (mf, mv, mc)

        r0 = QHBoxLayout()
        r0.setSpacing(6)
        for t, m in (("File", mf), ("View", mv), ("Controls", mc)):
            r0.addWidget(self._menu_button(t, m))
        r0.addStretch(1)
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.action_start)
        self.btn_reverse = QPushButton("Reverse")
        self.btn_reverse.clicked.connect(self.action_reverse)
        self.btn_forward = QPushButton("Forward")
        self.btn_forward.clicked.connect(self.action_forward)
        self.btn_end = QPushButton("End")
        self.btn_end.clicked.connect(self.action_end)
        for b in (self.btn_start, self.btn_reverse, self.btn_forward, self.btn_end):
            r0.addWidget(b)
        r0.addStretch(1)
        self.btn_next_buffer = QPushButton("Next Buffer")
        self.btn_next_buffer.setEnabled(False)
        self.btn_next_file = QPushButton("Next File")
        self.btn_next_file.clicked.connect(self.action_done)
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.action_quit)
        self.btn_about = QPushButton("About")
        self.btn_about.clicked.connect(lambda: AboutDialog(self).exec())
        for b in (self.btn_next_buffer, self.btn_next_file, self.btn_quit):
            r0.addWidget(b)
        r0.addStretch(1)
        r0.addWidget(self.btn_about)
        vl.addLayout(r0)

        r1 = QHBoxLayout()
        r1.setSpacing(30)
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.sc_width = MotifScale("Acrosstrack Width (m):", 1, 30000, 1, "{:.0f}", 185)
        self.sc_exag = MotifScale("Vertical Exaggeration:", 0.01, 20.0, 100, "{:.2f}", 185)
        self.sc_nplot = MotifScale("Pings shown:", 1, 20, 1, "{:.0f}", 110)
        self.sc_step = MotifScale("Pings to step:", 1, 20, 1, "{:.0f}", 110)
        self.spin_width = self.sc_width.slider
        self.spin_exag = self.sc_exag.slider
        self.spin_nplot = self.sc_nplot.slider
        self.spin_step = self.sc_step.slider
        self.spin_width.setValue(self.plot_width)
        self.spin_exag.setValue(self.exagger)
        self.spin_nplot.setValue(self.nplot)
        self.spin_step.setValue(self.step_size)
        self.spin_width.valueChanged.connect(lambda _v: self._on_width_changed(self.spin_width.value()))
        self.spin_exag.valueChanged.connect(lambda _v: self._on_exag_changed(self.spin_exag.value()))
        self.spin_nplot.valueChanged.connect(lambda _v: self._on_nplot_changed(int(round(self.spin_nplot.value()))))
        self.spin_step.valueChanged.connect(lambda _v: setattr(self, "step_size", int(round(self.spin_step.value()))))
        left.addWidget(self.sc_width)
        left.addWidget(self.sc_exag)
        right.addWidget(self.sc_nplot)
        right.addWidget(self.sc_step)
        r1.addLayout(left, 1)
        r1.addLayout(right, 1)
        vl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.setSpacing(2)
        r2.addSpacing(70)
        r2.addWidget(QLabel("Mode:"))
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for em, label in [(EditMode.TOGGLE, "Toggle"), (EditMode.PICK, "Pick"),
                          (EditMode.ERASE, "Erase"), (EditMode.RESTORE, "Restore"),
                          (EditMode.GRAB, "Grab"), (EditMode.INFO, "Info")]:
            rb = DiamondRadio(label)
            rb.setChecked(em == self.edit_mode)
            rb.toggled.connect(lambda c, em=em: c and self._set_edit_mode(em))
            self.mode_group.addButton(rb)
            self.mode_buttons[em] = rb
            r2.addWidget(rb)
        r2.addStretch(1)
        for text, slot in [("Flag View", self.action_flag_view), ("Unflag View", self.action_unflag_view),
                           ("Unflag Forward", self.action_unflag_all)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            r2.addWidget(b)
        vl.addLayout(r2)
        return hdr

    def _set_edit_mode(self, em):
        super()._set_edit_mode(em)
        rb = self.mode_buttons.get(em)
        if rb and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)
        for b in self.mode_buttons.values():
            b.update()

    def _set_view_mode(self, vm):
        super()._set_view_mode(vm)
        a = self._view_actions.get(vm)
        if a:
            a.setChecked(True)

    def _set_show_mode(self, sm):
        super()._set_show_mode(sm)
        a = self._show_actions.get(sm)
        if a:
            a.setChecked(True)

    def _set_show_time(self, pm):
        self.show_time = PlotMode(pm)
        self.refresh_plot()

    def _set_browse(self, flag):
        self.browse_only = bool(flag)
        self.status.showMessage("Browse only: edits are not saved" if flag else "Output edits to ESF", 3000)

    def record_edit(self, ping_idx, beam_idx, action):
        if self.browse_only:
            if self.swath:
                self.swath.changed_this_session.add((ping_idx, beam_idx))
            return
        super().record_edit(ping_idx, beam_idx, action)

    def on_info(self, ping_idx, beam_idx):
        super().on_info(ping_idx, beam_idx)
        self.canvas.update()

    def _update_status(self):
        super()._update_status()
        self.canvas.update()

    def action_buffer(self):
        BufferDialog(self).exec()

    def auto_scale_width(self):
        if not self.swath or not self.swath.pings:
            return
        self.plot_width = initial_plot_width_from_swath(self.swath)
        self.spin_width.blockSignals(True)
        self.spin_width.set_max(max(30000.0, self.plot_width * 1.25))
        self.spin_width.setValue(self.plot_width)
        self.spin_width.blockSignals(False)
        g = self.canvas.geometry_info()
        self.x_interval = nice_step(self.plot_width, 12)
        if g:
            xscale = (g["xmax"] - g["xmin"]) / self.plot_width
            self.y_interval = nice_step((g["ymax"] - g["ymin"]) / (xscale * self.exagger), 8)
        self.refresh_plot()

    def load_current_file(self):
        super().load_current_file()
        if self.swath is not None:
            self.auto_scale_width()
        self.canvas.update()

    def open_path(self, path):
        try:
            self.file_list = parse_datalist(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        if not self.file_list:
            QMessageBox.warning(self, "No files", "No files found to edit.")
            return
        self.file_index = 0
        self.load_current_file()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="MBedit-Qt (Qt6 port of the MB-System X11 mbedit)")
    ap.add_argument("input", nargs="?", help="swath file, .fbt file or datalist")
    ap.add_argument("-I", "--input-file", dest="input2")
    ap.add_argument("-D", "--browse", action="store_true", help="browse only, do not write edits")
    a = ap.parse_args(argv[1:] if argv is not None else None)
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    win = MainWindowMotif()
    if a.browse:
        win.act_out_browse.setChecked(True)
        win._set_browse(True)
    win.show()
    path = a.input2 or a.input
    if path:
        QTimer.singleShot(0, lambda: win.open_path(path))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
