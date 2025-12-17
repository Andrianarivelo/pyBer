# gui_postprocessing.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from analysis_core import ProcessedTrial


def _opt_plot(w: pg.PlotWidget) -> None:
    w.setMenuEnabled(True)
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setMouseEnabled(x=True, y=True)
    pi = w.getPlotItem()
    pi.setClipToView(True)
    pi.setDownsampling(auto=True, mode="peak")
    pi.setAutoVisible(y=True)


def _extract_rising_edges(time: np.ndarray, dio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    t = np.asarray(time, float)
    x = np.asarray(dio, float)
    if t.size < 2 or x.size != t.size:
        return np.array([], float)
    b = x > threshold
    on = np.where((~b[:-1]) & (b[1:]))[0] + 1
    return t[on]


def _load_behavior_csv(path: str) -> Dict[str, Any]:
    """
    Supports:
      - binary trace: columns 'time' and 'behavior'
      - event list table: first column optional, other columns are behaviors, cell values are timepoints
    Returns dict with keys:
      kind: 'binary' or 'event_table'
      for binary: time, behavior
      for event_table: behaviors(list), table(dict behavior->times array)
    """
    import pandas as pd
    df = pd.read_csv(path)

    cols = [c.strip() for c in df.columns]
    if "time" in [c.lower() for c in cols] and "behavior" in [c.lower() for c in cols]:
        # normalize column names
        tcol = next(c for c in df.columns if c.lower() == "time")
        bcol = next(c for c in df.columns if c.lower() == "behavior")
        return {"kind": "binary", "time": df[tcol].to_numpy(float), "behavior": df[bcol].to_numpy(int)}

    # event table: each column = behavior; cell values = times
    table: Dict[str, np.ndarray] = {}
    for c in df.columns:
        vals = df[c].to_numpy()
        vals = vals[np.isfinite(vals.astype(float, copy=False))]
        try:
            times = vals.astype(float)
        except Exception:
            continue
        table[str(c)] = times

    return {"kind": "event_table", "behaviors": sorted(table.keys()), "table": table}


def _compute_psth_matrix(
    t: np.ndarray,
    y: np.ndarray,
    event_times: np.ndarray,
    window: Tuple[float, float],
    baseline_win: Tuple[float, float],
    resample_hz: float,
    smooth_sigma_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tvec (relative time), mat (n_events x n_samples) with NaNs if missing
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ev = np.asarray(event_times, float)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return np.array([], float), np.zeros((0, 0), float)

    dt = 1.0 / float(resample_hz)
    tvec = np.arange(window[0], window[1] + 0.5 * dt, dt)

    mat = np.full((ev.size, tvec.size), np.nan, float)

    for i, et in enumerate(ev):
        # baseline
        bmask = (t >= et + baseline_win[0]) & (t <= et + baseline_win[1])
        base = y[bmask]
        if base.size < 5 or not np.any(np.isfinite(base)):
            continue
        bmean = np.nanmean(base)
        bstd = np.nanstd(base)
        if not np.isfinite(bstd) or bstd <= 1e-12:
            bstd = 1.0

        # extract window and interpolate onto tvec
        wmask = (t >= et + window[0]) & (t <= et + window[1])
        tw = t[wmask] - et
        yw = y[wmask]
        good = np.isfinite(tw) & np.isfinite(yw)
        if np.sum(good) < 5:
            continue
        # sparse interpolation
        mat[i, :] = np.interp(tvec, tw[good], (yw[good] - bmean) / bstd)

    if smooth_sigma_s and smooth_sigma_s > 0:
        # simple gaussian smoothing along time axis
        from scipy.ndimage import gaussian_filter1d
        sigma = smooth_sigma_s * resample_hz
        mat = gaussian_filter1d(mat, sigma=sigma, axis=1, mode="nearest")

    return tvec, mat


class PostProcessingPanel(QtWidgets.QWidget):
    # bridge signals to main
    requestCurrentProcessed = QtCore.Signal()
    requestDioList = QtCore.Signal()
    requestDioData = QtCore.Signal(str, str)  # (path, dio)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._processed: List[ProcessedTrial] = []
        self._dio_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}  # (path,dio)->(t,x)
        self._behavior_files: Dict[str, str] = {}  # stem->csv path
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Left controls (scroll)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setSpacing(10)

        grp_src = QtWidgets.QGroupBox("Signal Source")
        fsrc = QtWidgets.QFormLayout(grp_src)
        self.lbl_current = QtWidgets.QLabel("Current: (none)")
        self.btn_use_current = QtWidgets.QPushButton("Use current preprocessed selection")
        self.btn_use_current.setProperty("class", "compactPrimary")
        self.btn_refresh_dio = QtWidgets.QPushButton("Refresh DIO list")
        self.btn_refresh_dio.setProperty("class", "compact")

        fsrc.addRow(self.lbl_current)
        fsrc.addRow(self.btn_use_current)
        fsrc.addRow(self.btn_refresh_dio)

        grp_align = QtWidgets.QGroupBox("Behavior / Events")
        fal = QtWidgets.QFormLayout(grp_align)

        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(["DIO (from Doric)", "CSV behavior (binary trace)", "CSV event table (columns=behaviors)"])

        self.combo_dio = QtWidgets.QComboBox()

        self.btn_load_beh = QtWidgets.QPushButton("Load behavior CSV(s)…")
        self.btn_load_beh.setProperty("class", "compact")
        self.lbl_beh = QtWidgets.QLabel("(none)")
        self.lbl_beh.setProperty("class", "hint")

        self.combo_behavior_name = QtWidgets.QComboBox()

        fal.addRow("Align source", self.combo_align)
        fal.addRow("DIO channel", self.combo_dio)
        fal.addRow(self.btn_load_beh)
        fal.addRow("Behavior file(s)", self.lbl_beh)
        fal.addRow("Behavior name", self.combo_behavior_name)

        grp_opt = QtWidgets.QGroupBox("PSTH Options")
        fopt = QtWidgets.QFormLayout(grp_opt)

        self.spin_pre = QtWidgets.QDoubleSpinBox(); self.spin_pre.setRange(0.1, 60); self.spin_pre.setValue(2.0); self.spin_pre.setDecimals(2)
        self.spin_post= QtWidgets.QDoubleSpinBox(); self.spin_post.setRange(0.1, 120); self.spin_post.setValue(5.0); self.spin_post.setDecimals(2)
        self.spin_b0  = QtWidgets.QDoubleSpinBox(); self.spin_b0.setRange(-60, 0); self.spin_b0.setValue(-1.0); self.spin_b0.setDecimals(2)
        self.spin_b1  = QtWidgets.QDoubleSpinBox(); self.spin_b1.setRange(-60, 0); self.spin_b1.setValue(0.0); self.spin_b1.setDecimals(2)

        self.spin_resample = QtWidgets.QDoubleSpinBox(); self.spin_resample.setRange(1, 1000); self.spin_resample.setValue(50); self.spin_resample.setDecimals(1)
        self.spin_smooth = QtWidgets.QDoubleSpinBox(); self.spin_smooth.setRange(0, 5); self.spin_smooth.setValue(0.0); self.spin_smooth.setDecimals(2)

        fopt.addRow("Window pre (s)", self.spin_pre)
        fopt.addRow("Window post (s)", self.spin_post)
        fopt.addRow("Baseline start (s)", self.spin_b0)
        fopt.addRow("Baseline end (s)", self.spin_b1)
        fopt.addRow("Resample (Hz)", self.spin_resample)
        fopt.addRow("Gaussian smooth σ (s)", self.spin_smooth)

        self.btn_compute = QtWidgets.QPushButton("Post-process (compute PSTH)")
        self.btn_compute.setProperty("class", "compactPrimary")
        self.btn_update = QtWidgets.QPushButton("Update Preview")
        self.btn_update.setProperty("class", "compact")

        lv.addWidget(grp_src)
        lv.addWidget(grp_align)
        lv.addWidget(grp_opt)
        lv.addWidget(self.btn_compute)
        lv.addWidget(self.btn_update)
        lv.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)

        # Right plots: trace preview + heatmap + avg
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setSpacing(10)

        self.plot_trace = pg.PlotWidget(title="Trace preview (events as vertical lines)")
        self.plot_heat = pg.PlotWidget(title="Heatmap (trials or recordings)")
        self.plot_avg = pg.PlotWidget(title="Average PSTH ± SEM")

        for w in (self.plot_trace, self.plot_heat, self.plot_avg):
            _opt_plot(w)

        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen((90, 190, 255), width=1.1))
        self.event_lines: List[pg.InfiniteLine] = []

        self.img = pg.ImageItem()
        self.plot_heat.addItem(self.img)
        self.plot_heat.setLabel("bottom", "Time (s)")
        self.plot_heat.setLabel("left", "Trials / Recordings")

        self.curve_avg = self.plot_avg.plot(pen=pg.mkPen((90, 190, 255), width=1.3))
        self.curve_sem_hi = self.plot_avg.plot(pen=pg.mkPen((220, 220, 220), width=1.0))
        self.curve_sem_lo = self.plot_avg.plot(pen=pg.mkPen((220, 220, 220), width=1.0))
        self.plot_avg.addLine(x=0, pen=pg.mkPen((200, 200, 200), style=QtCore.Qt.PenStyle.DashLine))

        rv.addWidget(self.plot_trace, stretch=1)
        rv.addWidget(self.plot_heat, stretch=2)
        rv.addWidget(self.plot_avg, stretch=1)

        # Layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        root.addWidget(splitter)

        # Wiring
        self.btn_use_current.clicked.connect(self.requestCurrentProcessed.emit)
        self.btn_refresh_dio.clicked.connect(self.requestDioList.emit)
        self.btn_load_beh.clicked.connect(self._load_behavior_files)
        self.btn_compute.clicked.connect(self._compute_psth)
        self.btn_update.clicked.connect(self._compute_psth)

        self.combo_align.currentIndexChanged.connect(self._refresh_behavior_list)

    # ---- bridge reception ----

    def set_current_source_label(self, filename: str, channel: str) -> None:
        self.lbl_current.setText(f"Current: {filename} [{channel}]")

    def notify_preprocessing_updated(self, _processed: ProcessedTrial) -> None:
        # no-op; user presses compute or update
        pass

    @QtCore.Slot(list)
    def receive_current_processed(self, processed_list: List[ProcessedTrial]) -> None:
        self._processed = processed_list or []
        # update trace preview with first entry
        self._refresh_behavior_list()
        self._update_trace_preview()

    @QtCore.Slot(list)
    def receive_dio_list(self, dio_list: List[str]) -> None:
        self.combo_dio.clear()
        for d in dio_list or []:
            self.combo_dio.addItem(d)

    @QtCore.Slot(str, str, object, object)
    def receive_dio_data(self, path: str, dio_name: str, t: Optional[np.ndarray], x: Optional[np.ndarray]) -> None:
        if t is None or x is None:
            return
        self._dio_cache[(path, dio_name)] = (np.asarray(t, float), np.asarray(x, float))

    # ---- behavior files ----

    def _load_behavior_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Load behavior CSV(s)", os.getcwd(), "CSV files (*.csv)")
        if not paths:
            return
        self._behavior_files.clear()
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            self._behavior_files[stem] = p
        self.lbl_beh.setText(f"{len(paths)} file(s) loaded")
        self._refresh_behavior_list()

    def _refresh_behavior_list(self) -> None:
        # Behavior name list is only relevant for event-table CSV; for binary, just one "behavior"
        self.combo_behavior_name.clear()
        if self.combo_align.currentText().startswith("CSV event"):
            # if we have any file, inspect first
            if not self._behavior_files:
                return
            any_path = next(iter(self._behavior_files.values()))
            try:
                from analysis_core import ProcessedTrial  # noqa
                info = _load_behavior_csv(any_path)
                if info.get("kind") == "event_table":
                    for name in info.get("behaviors", []):
                        self.combo_behavior_name.addItem(name)
            except Exception:
                pass
        else:
            self.combo_behavior_name.addItem("behavior")

    # ---- PSTH compute ----

    def _match_behavior_file(self, proc: ProcessedTrial) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(proc.path))[0]
        return self._behavior_files.get(stem, None)

    def _get_event_times_for_proc(self, proc: ProcessedTrial) -> np.ndarray:
        align = self.combo_align.currentText()

        if align.startswith("DIO"):
            dio_name = self.combo_dio.currentText().strip()
            if not dio_name:
                return np.array([], float)

            # request if not cached
            key = (proc.path, dio_name)
            if key not in self._dio_cache:
                self.requestDioData.emit(proc.path, dio_name)
                return np.array([], float)

            t, x = self._dio_cache[key]
            return _extract_rising_edges(t, x, threshold=0.5)

        # CSV behavior
        csv_path = self._match_behavior_file(proc)
        if not csv_path:
            return np.array([], float)

        info = _load_behavior_csv(csv_path)

        if align.startswith("CSV behavior"):
            if info.get("kind") != "binary":
                return np.array([], float)
            t = np.asarray(info["time"], float)
            b = np.asarray(info["behavior"], int)
            # extract rising edges
            return _extract_rising_edges(t, b.astype(float), threshold=0.5)

        # event table
        if info.get("kind") != "event_table":
            return np.array([], float)
        beh = self.combo_behavior_name.currentText().strip()
        if not beh:
            return np.array([], float)
        return np.asarray(info["table"].get(beh, np.array([], float)), float)

    def _update_trace_preview(self) -> None:
        # show first processed trace
        if not self._processed:
            self.curve_trace.setData([], [])
            for ln in self.event_lines:
                self.plot_trace.removeItem(ln)
            self.event_lines = []
            return

        proc = self._processed[0]
        t = proc.time
        y = proc.output if proc.output is not None else np.full_like(t, np.nan)

        self.curve_trace.setData(t, y, connect="finite", skipFiniteCheck=True)

        # draw event lines if possible
        for ln in self.event_lines:
            self.plot_trace.removeItem(ln)
        self.event_lines = []

        ev = self._get_event_times_for_proc(proc)
        if ev.size:
            # limit lines to a reasonable amount for UI
            ev = ev[:200]
            for et in ev:
                ln = pg.InfiniteLine(pos=float(et), angle=90, pen=pg.mkPen((220, 220, 220), width=1.0, style=QtCore.Qt.PenStyle.DashLine))
                self.plot_trace.addItem(ln)
                self.event_lines.append(ln)

    def _compute_psth(self) -> None:
        if not self._processed:
            return

        # Update trace preview each time (also updates event lines)
        self._update_trace_preview()

        pre = float(self.spin_pre.value())
        post = float(self.spin_post.value())
        b0 = float(self.spin_b0.value())
        b1 = float(self.spin_b1.value())
        res_hz = float(self.spin_resample.value())
        smooth = float(self.spin_smooth.value())

        window = (-pre, post)
        baseline = (b0, b1)

        # Single vs multi logic
        if len(self._processed) == 1:
            proc = self._processed[0]
            ev = self._get_event_times_for_proc(proc)
            if ev.size == 0:
                return
            tvec, mat = _compute_psth_matrix(proc.time, proc.output, ev, window, baseline, res_hz, smooth_sigma_s=smooth)
            self._render_heatmap(mat, tvec)
            self._render_avg(mat, tvec)
        else:
            # multiple recordings: one row per recording = average PSTH over events within that recording
            rows = []
            for proc in self._processed:
                ev = self._get_event_times_for_proc(proc)
                if ev.size == 0:
                    continue
                tvec, mat = _compute_psth_matrix(proc.time, proc.output, ev, window, baseline, res_hz, smooth_sigma_s=smooth)
                if mat.size == 0:
                    continue
                avg = np.nanmean(mat, axis=0)
                rows.append(avg)
            if not rows:
                return
            mat_rec = np.vstack(rows)
            self._render_heatmap(mat_rec, tvec)
            self._render_avg(mat_rec, tvec)

    def _render_heatmap(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0:
            self.img.setImage(np.zeros((1, 1)))
            return

        # pyqtgraph ImageItem expects (col,row) or (row,col). We'll provide (rows, cols)
        img = np.asarray(mat, float)
        # set image (auto-level)
        self.img.setImage(img, autoLevels=True)

        # set transform so x-axis is time
        # scale x by dt, translate to window start
        dt = float(tvec[1] - tvec[0]) if tvec.size > 1 else 1.0
        self.img.resetTransform()
        self.img.scale(dt, 1.0)
        self.img.translate(float(tvec[0]), 0.0)

        self.plot_heat.setXRange(float(tvec[0]), float(tvec[-1]), padding=0)

    def _render_avg(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0:
            self.curve_avg.setData([], [])
            self.curve_sem_hi.setData([], [])
            self.curve_sem_lo.setData([], [])
            return

        avg = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0) / np.sqrt(max(1, np.sum(np.any(np.isfinite(mat), axis=1))))

        self.curve_avg.setData(tvec, avg, connect="finite", skipFiniteCheck=True)
        self.curve_sem_hi.setData(tvec, avg + sem, connect="finite", skipFiniteCheck=True)
        self.curve_sem_lo.setData(tvec, avg - sem, connect="finite", skipFiniteCheck=True)
        self.plot_avg.setXRange(float(tvec[0]), float(tvec[-1]), padding=0)
