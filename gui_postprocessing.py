# gui_postprocessing.py
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from analysis_core import ProcessedTrial
from ethovision_process_gui import clean_sheet


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

def _extract_events_with_durations(
    time: np.ndarray,
    dio: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns event onset times and durations (seconds) based on threshold crossings.
    """
    t = np.asarray(time, float)
    x = np.asarray(dio, float)
    if t.size < 2 or x.size != t.size:
        return np.array([], float), np.array([], float)

    b = x > threshold
    rising = np.where((~b[:-1]) & (b[1:]))[0] + 1
    falling = np.where((b[:-1]) & (~b[1:]))[0] + 1
    if rising.size == 0 or falling.size == 0:
        return np.array([], float), np.array([], float)

    times = []
    durations = []
    fi = 0
    for ri in rising:
        while fi < falling.size and falling[fi] <= ri:
            fi += 1
        if fi >= falling.size:
            break
        t0 = float(t[ri])
        t1 = float(t[falling[fi]])
        if t1 > t0:
            times.append(t0)
            durations.append(t1 - t0)
        fi += 1

    return np.asarray(times, float), np.asarray(durations, float)


def _extract_onsets_offsets(
    time: np.ndarray,
    dio: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time, float)
    x = np.asarray(dio, float)
    if t.size < 2 or x.size != t.size:
        return np.array([], float), np.array([], float), np.array([], float)
    b = x > threshold
    rising = np.where((~b[:-1]) & (b[1:]))[0] + 1
    falling = np.where((b[:-1]) & (~b[1:]))[0] + 1
    if rising.size == 0 or falling.size == 0:
        return np.array([], float), np.array([], float), np.array([], float)

    on = []
    off = []
    dur = []
    fi = 0
    for ri in rising:
        while fi < falling.size and falling[fi] <= ri:
            fi += 1
        if fi >= falling.size:
            break
        t0 = float(t[ri])
        t1 = float(t[falling[fi]])
        if t1 > t0:
            on.append(t0)
            off.append(t1)
            dur.append(t1 - t0)
        fi += 1
    return np.asarray(on, float), np.asarray(off, float), np.asarray(dur, float)

def _binary_columns_from_df(df) -> Tuple[str, Dict[str, np.ndarray]]:
    cols = [c.strip() for c in df.columns]
    time_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"time", "trial time", "recording time"}:
            time_col = c
            break
    if time_col is None and cols:
        time_col = df.columns[0]

    t = np.asarray(df[time_col], float)
    behaviors: Dict[str, np.ndarray] = {}

    for c in df.columns:
        if c == time_col:
            continue
        arr = np.asarray(df[c], float)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        uniq = np.unique(finite)
        if np.all(np.isin(uniq, [0.0, 1.0])):
            behaviors[str(c)] = arr.astype(float)

    return time_col, behaviors


def _load_behavior_csv(path: str) -> Dict[str, Any]:
    import pandas as pd
    df = pd.read_csv(path)
    time_col, behaviors = _binary_columns_from_df(df)
    return {"kind": "binary_columns", "time": np.asarray(df[time_col], float), "behaviors": behaviors}


def _load_behavior_ethovision(path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    if sheet_name is None:
        import pandas as pd
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_name = xls.sheet_names[0] if xls.sheet_names else None
    if not sheet_name:
        return {"kind": "binary_columns", "time": np.array([], float), "behaviors": {}}
    df = clean_sheet(Path(path), sheet_name, interpolate=True)
    time_col, behaviors = _binary_columns_from_df(df)
    return {"kind": "binary_columns", "time": np.asarray(df[time_col], float), "behaviors": behaviors, "sheet": sheet_name}


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
        self._behavior_sources: Dict[str, Dict[str, Any]] = {}  # stem->behavior data
        self._last_mat: Optional[np.ndarray] = None
        self._last_tvec: Optional[np.ndarray] = None
        self._last_events: Optional[np.ndarray] = None
        self._last_durations: Optional[np.ndarray] = None
        self._last_metrics: Optional[Dict[str, float]] = None
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
        vsrc = QtWidgets.QVBoxLayout(grp_src)

        self.tab_sources = QtWidgets.QTabWidget()
        tab_single = QtWidgets.QWidget()
        tab_group = QtWidgets.QWidget()
        self.tab_sources.addTab(tab_single, "Single")
        self.tab_sources.addTab(tab_group, "Group")

        single_layout = QtWidgets.QVBoxLayout(tab_single)
        self.lbl_current = QtWidgets.QLabel("Current: (none)")
        self.btn_use_current = QtWidgets.QPushButton("Use current preprocessed selection")
        self.btn_use_current.setProperty("class", "compactPrimary")
        single_layout.addWidget(self.lbl_current)
        single_layout.addWidget(self.btn_use_current)
        single_layout.addStretch(1)

        group_layout = QtWidgets.QVBoxLayout(tab_group)
        self.btn_load_processed = QtWidgets.QPushButton("Load processed files (CSV/H5)")
        self.btn_load_processed.setProperty("class", "compact")
        self.lbl_group = QtWidgets.QLabel("(none)")
        self.lbl_group.setProperty("class", "hint")
        group_layout.addWidget(self.btn_load_processed)
        group_layout.addWidget(self.lbl_group)
        group_layout.addStretch(1)

        self.btn_refresh_dio = QtWidgets.QPushButton("Refresh DIO list")
        self.btn_refresh_dio.setProperty("class", "compact")

        vsrc.addWidget(self.tab_sources)
        vsrc.addWidget(self.btn_refresh_dio)

        grp_align = QtWidgets.QGroupBox("Behavior / Events")
        fal = QtWidgets.QFormLayout(grp_align)

        self.combo_align = QtWidgets.QComboBox()
        self.combo_align.addItems(["DIO (from Doric)", "Behavior (CSV/XLSX)"])

        self.combo_dio = QtWidgets.QComboBox()
        self.combo_dio_polarity = QtWidgets.QComboBox()
        self.combo_dio_polarity.addItems(["Event high (0→1)", "Event low (1→0)"])
        self.combo_dio_align = QtWidgets.QComboBox()
        self.combo_dio_align.addItems(["Align to onset", "Align to offset"])

        self.btn_load_beh = QtWidgets.QPushButton("Load behavior CSV/XLSX…")
        self.btn_load_beh.setProperty("class", "compact")
        self.lbl_beh = QtWidgets.QLabel("(none)")
        self.lbl_beh.setProperty("class", "hint")

        self.combo_behavior_name = QtWidgets.QComboBox()
        self.combo_behavior_align = QtWidgets.QComboBox()
        self.combo_behavior_align.addItems(["Align to onset", "Align to offset", "Transition A→B"])
        self.combo_behavior_from = QtWidgets.QComboBox()
        self.combo_behavior_to = QtWidgets.QComboBox()
        self.spin_transition_gap = QtWidgets.QDoubleSpinBox()
        self.spin_transition_gap.setRange(0, 60)
        self.spin_transition_gap.setValue(1.0)
        self.spin_transition_gap.setDecimals(2)

        fal.addRow("Align source", self.combo_align)
        fal.addRow("DIO channel", self.combo_dio)
        fal.addRow("DIO polarity", self.combo_dio_polarity)
        fal.addRow("DIO align", self.combo_dio_align)
        fal.addRow(self.btn_load_beh)
        fal.addRow("Behavior file(s)", self.lbl_beh)
        fal.addRow("Behavior name", self.combo_behavior_name)
        fal.addRow("Behavior align", self.combo_behavior_align)
        fal.addRow("Transition from", self.combo_behavior_from)
        fal.addRow("Transition to", self.combo_behavior_to)
        fal.addRow("Transition gap (s)", self.spin_transition_gap)

        grp_opt = QtWidgets.QGroupBox("PSTH Options")
        fopt = QtWidgets.QFormLayout(grp_opt)

        self.spin_pre = QtWidgets.QDoubleSpinBox(); self.spin_pre.setRange(0.1, 60); self.spin_pre.setValue(2.0); self.spin_pre.setDecimals(2)
        self.spin_post= QtWidgets.QDoubleSpinBox(); self.spin_post.setRange(0.1, 120); self.spin_post.setValue(5.0); self.spin_post.setDecimals(2)
        self.spin_b0  = QtWidgets.QDoubleSpinBox(); self.spin_b0.setRange(-60, 0); self.spin_b0.setValue(-1.0); self.spin_b0.setDecimals(2)
        self.spin_b1  = QtWidgets.QDoubleSpinBox(); self.spin_b1.setRange(-60, 0); self.spin_b1.setValue(0.0); self.spin_b1.setDecimals(2)

        self.spin_resample = QtWidgets.QDoubleSpinBox(); self.spin_resample.setRange(1, 1000); self.spin_resample.setValue(50); self.spin_resample.setDecimals(1)
        self.spin_smooth = QtWidgets.QDoubleSpinBox(); self.spin_smooth.setRange(0, 5); self.spin_smooth.setValue(0.0); self.spin_smooth.setDecimals(2)

        self.spin_event_start = QtWidgets.QSpinBox(); self.spin_event_start.setRange(1, 1000000); self.spin_event_start.setValue(1)
        self.spin_event_end = QtWidgets.QSpinBox(); self.spin_event_end.setRange(0, 1000000); self.spin_event_end.setValue(0)
        self.spin_dur_min = QtWidgets.QDoubleSpinBox(); self.spin_dur_min.setRange(0, 1e6); self.spin_dur_min.setValue(0.0); self.spin_dur_min.setDecimals(2)
        self.spin_dur_max = QtWidgets.QDoubleSpinBox(); self.spin_dur_max.setRange(0, 1e6); self.spin_dur_max.setValue(0.0); self.spin_dur_max.setDecimals(2)
        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems(["AUC", "Mean z"])
        self.spin_metric_pre0 = QtWidgets.QDoubleSpinBox(); self.spin_metric_pre0.setRange(-120, 0); self.spin_metric_pre0.setValue(-1.0); self.spin_metric_pre0.setDecimals(2)
        self.spin_metric_pre1 = QtWidgets.QDoubleSpinBox(); self.spin_metric_pre1.setRange(-120, 0); self.spin_metric_pre1.setValue(0.0); self.spin_metric_pre1.setDecimals(2)
        self.spin_metric_post0 = QtWidgets.QDoubleSpinBox(); self.spin_metric_post0.setRange(0, 120); self.spin_metric_post0.setValue(0.0); self.spin_metric_post0.setDecimals(2)
        self.spin_metric_post1 = QtWidgets.QDoubleSpinBox(); self.spin_metric_post1.setRange(0, 120); self.spin_metric_post1.setValue(1.0); self.spin_metric_post1.setDecimals(2)

        fopt.addRow("Window pre (s)", self.spin_pre)
        fopt.addRow("Window post (s)", self.spin_post)
        fopt.addRow("Baseline start (s)", self.spin_b0)
        fopt.addRow("Baseline end (s)", self.spin_b1)
        fopt.addRow("Resample (Hz)", self.spin_resample)
        fopt.addRow("Event index start (1-based)", self.spin_event_start)
        fopt.addRow("Event index end (0=all)", self.spin_event_end)
        fopt.addRow("Event duration min (s)", self.spin_dur_min)
        fopt.addRow("Event duration max (s)", self.spin_dur_max)
        fopt.addRow("Gaussian smooth σ (s)", self.spin_smooth)
        fopt.addRow("Metric", self.combo_metric)
        fopt.addRow("Metric pre start (s)", self.spin_metric_pre0)
        fopt.addRow("Metric pre end (s)", self.spin_metric_pre1)
        fopt.addRow("Metric post start (s)", self.spin_metric_post0)
        fopt.addRow("Metric post end (s)", self.spin_metric_post1)

        self.btn_compute = QtWidgets.QPushButton("Post-process (compute PSTH)")
        self.btn_compute.setProperty("class", "compactPrimary")
        self.btn_update = QtWidgets.QPushButton("Update Preview")
        self.btn_update.setProperty("class", "compact")
        self.btn_export = QtWidgets.QPushButton("Export results")
        self.btn_export.setProperty("class", "compact")

        lv.addWidget(grp_src)
        lv.addWidget(grp_align)
        lv.addWidget(grp_opt)
        lv.addWidget(self.btn_compute)
        lv.addWidget(self.btn_update)
        lv.addWidget(self.btn_export)
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
        self.plot_dur = pg.PlotWidget(title="Event duration")
        self.plot_avg = pg.PlotWidget(title="Average PSTH ± SEM")
        self.plot_metrics = pg.PlotWidget(title="Metrics (pre vs post)")

        for w in (self.plot_trace, self.plot_heat, self.plot_dur, self.plot_avg, self.plot_metrics):
            _opt_plot(w)

        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen((90, 190, 255), width=1.1))
        self.event_lines: List[pg.InfiniteLine] = []

        self.img = pg.ImageItem()
        self.plot_heat.addItem(self.img)
        self.plot_heat.setLabel("bottom", "Time (s)")
        self.plot_heat.setLabel("left", "Trials / Recordings")
        self.plot_dur.setLabel("bottom", "Count")
        self.plot_dur.setLabel("left", "Duration (s)")

        self.curve_avg = self.plot_avg.plot(pen=pg.mkPen((90, 190, 255), width=1.3))
        self.curve_sem_hi = self.plot_avg.plot(pen=pg.mkPen((220, 220, 220), width=1.0))
        self.curve_sem_lo = self.plot_avg.plot(pen=pg.mkPen((220, 220, 220), width=1.0))
        self.plot_avg.addLine(x=0, pen=pg.mkPen((200, 200, 200), style=QtCore.Qt.PenStyle.DashLine))
        self.metrics_bar = pg.BarGraphItem(x=[0, 1], height=[0, 0], width=0.6, brushes=["#5a8fd6", "#d67a5a"])
        self.plot_metrics.addItem(self.metrics_bar)
        self.plot_metrics.setXRange(-0.5, 1.5, padding=0)
        self.plot_metrics.getAxis("bottom").setTicks([[(0, "pre"), (1, "post")]])

        heat_row = QtWidgets.QHBoxLayout()
        heat_row.addWidget(self.plot_heat, stretch=4)
        heat_row.addWidget(self.plot_dur, stretch=1)

        avg_row = QtWidgets.QHBoxLayout()
        avg_row.addWidget(self.plot_avg, stretch=4)
        avg_row.addWidget(self.plot_metrics, stretch=1)

        rv.addWidget(self.plot_trace, stretch=1)
        rv.addLayout(heat_row, stretch=2)
        rv.addLayout(avg_row, stretch=1)

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
        self.btn_export.clicked.connect(self._export_results)

        self.combo_align.currentIndexChanged.connect(self._update_align_ui)
        self.combo_behavior_align.currentIndexChanged.connect(self._update_align_ui)
        self.combo_align.currentIndexChanged.connect(self._refresh_behavior_list)
        for w in (
            self.combo_dio,
            self.combo_dio_polarity,
            self.combo_dio_align,
            self.combo_behavior_name,
            self.combo_behavior_align,
            self.combo_behavior_from,
            self.combo_behavior_to,
        ):
            w.currentIndexChanged.connect(self._compute_psth)
        self.spin_transition_gap.valueChanged.connect(self._compute_psth)
        for w in (
            self.spin_event_start,
            self.spin_event_end,
            self.spin_dur_min,
            self.spin_dur_max,
            self.spin_metric_pre0,
            self.spin_metric_pre1,
            self.spin_metric_post0,
            self.spin_metric_post1,
        ):
            w.valueChanged.connect(self._compute_psth)
        self.combo_metric.currentIndexChanged.connect(self._compute_psth)

        self._update_align_ui()

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

    def append_processed(self, processed_list: List[ProcessedTrial]) -> None:
        if not processed_list:
            return
        self._processed.extend(processed_list)
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

    def _update_align_ui(self) -> None:
        use_dio = self.combo_align.currentText().startswith("DIO")
        for w in (self.combo_dio, self.combo_dio_polarity, self.combo_dio_align):
            w.setEnabled(use_dio)

        use_beh = not use_dio
        self.btn_load_beh.setEnabled(use_beh)
        self.combo_behavior_name.setEnabled(use_beh)
        self.combo_behavior_align.setEnabled(use_beh)

        is_transition = self.combo_behavior_align.currentText().startswith("Transition") and use_beh
        self.combo_behavior_from.setVisible(is_transition)
        self.combo_behavior_to.setVisible(is_transition)
        self.spin_transition_gap.setVisible(is_transition)

        if use_beh:
            self.combo_behavior_from.setEnabled(is_transition)
            self.combo_behavior_to.setEnabled(is_transition)
            self.spin_transition_gap.setEnabled(is_transition)

    def _load_behavior_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load behavior CSV/XLSX",
            os.getcwd(),
            "Behavior files (*.csv *.xlsx)",
        )
        if not paths:
            return
        self._behavior_sources.clear()
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext == ".csv":
                    info = _load_behavior_csv(p)
                elif ext == ".xlsx":
                    import pandas as pd
                    xls = pd.ExcelFile(p, engine="openpyxl")
                    sheet = None
                    if len(xls.sheet_names) > 1:
                        sheet, ok = QtWidgets.QInputDialog.getItem(
                            self,
                            "Select sheet",
                            f"{os.path.basename(p)}: choose sheet",
                            xls.sheet_names,
                            0,
                            False,
                        )
                        if not ok:
                            continue
                    info = _load_behavior_ethovision(p, sheet_name=sheet)
                else:
                    continue
                self._behavior_sources[stem] = info
            except Exception:
                continue
        self.lbl_beh.setText(f"{len(paths)} file(s) loaded")
        self._refresh_behavior_list()

    def _refresh_behavior_list(self) -> None:
        self.combo_behavior_name.clear()
        if not self._behavior_sources:
            return
        any_info = next(iter(self._behavior_sources.values()))
        behaviors = sorted(list((any_info.get("behaviors") or {}).keys()))
        for name in behaviors:
            self.combo_behavior_name.addItem(name)
        self.combo_behavior_from.clear()
        self.combo_behavior_to.clear()
        for name in behaviors:
            self.combo_behavior_from.addItem(name)
            self.combo_behavior_to.addItem(name)

    # ---- PSTH compute ----

    def _match_behavior_source(self, proc: ProcessedTrial) -> Optional[Dict[str, Any]]:
        stem = os.path.splitext(os.path.basename(proc.path))[0]
        return self._behavior_sources.get(stem, None)

    def _get_events_for_proc(self, proc: ProcessedTrial) -> Tuple[np.ndarray, np.ndarray]:
        align = self.combo_align.currentText()

        if align.startswith("DIO"):
            dio_name = self.combo_dio.currentText().strip()
            if not dio_name:
                return np.array([], float), np.array([], float)

            # request if not cached
            key = (proc.path, dio_name)
            if key not in self._dio_cache:
                self.requestDioData.emit(proc.path, dio_name)
                return np.array([], float), np.array([], float)

            t, x = self._dio_cache[key]
            polarity = self.combo_dio_polarity.currentText()
            align_edge = self.combo_dio_align.currentText()
            sig = np.asarray(x, float)
            if polarity.startswith("Event low"):
                sig = 1.0 - sig
            on, off, dur = _extract_onsets_offsets(t, sig, threshold=0.5)
            if align_edge.endswith("offset"):
                return off, dur
            return on, dur

        # Behavior binary columns
        info = self._match_behavior_source(proc)
        if not info:
            return np.array([], float), np.array([], float)
        t = np.asarray(info.get("time", np.array([], float)), float)
        behaviors = info.get("behaviors") or {}
        if t.size == 0 or not behaviors:
            return np.array([], float), np.array([], float)

        align_mode = self.combo_behavior_align.currentText()
        if align_mode.startswith("Transition"):
            beh_a = self.combo_behavior_from.currentText().strip()
            beh_b = self.combo_behavior_to.currentText().strip()
            if beh_a not in behaviors or beh_b not in behaviors:
                return np.array([], float), np.array([], float)
            on_a, off_a, _ = _extract_onsets_offsets(t, behaviors[beh_a], threshold=0.5)
            on_b, _, dur_b = _extract_onsets_offsets(t, behaviors[beh_b], threshold=0.5)
            if on_a.size == 0 or on_b.size == 0:
                return np.array([], float), np.array([], float)
            gap = float(self.spin_transition_gap.value())
            times = []
            durs = []
            bi = 0
            for a_off in off_a:
                while bi < on_b.size and on_b[bi] < a_off:
                    bi += 1
                if bi >= on_b.size:
                    break
                if 0 <= on_b[bi] - a_off <= gap:
                    times.append(on_b[bi])
                    durs.append(dur_b[bi] if bi < dur_b.size else np.nan)
            return np.asarray(times, float), np.asarray(durs, float)

        beh = self.combo_behavior_name.currentText().strip()
        if beh not in behaviors:
            return np.array([], float), np.array([], float)
        on, off, dur = _extract_onsets_offsets(t, behaviors[beh], threshold=0.5)
        if align_mode.endswith("offset"):
            return off, dur
        return on, dur

    def _filter_events(self, times: np.ndarray, durations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, float)
        durations = np.asarray(durations, float)
        if times.size == 0:
            return times, durations

        start_idx = int(self.spin_event_start.value())
        end_idx = int(self.spin_event_end.value())
        if start_idx < 1:
            start_idx = 1
        if end_idx <= 0 or end_idx > times.size:
            end_idx = times.size
        if start_idx > end_idx:
            return np.array([], float), np.array([], float)

        times = times[start_idx - 1:end_idx]
        durations = durations[start_idx - 1:end_idx]

        min_dur = float(self.spin_dur_min.value())
        max_dur = float(self.spin_dur_max.value())
        if np.any(np.isfinite(durations)) and (min_dur > 0 or max_dur > 0):
            mask = np.ones_like(durations, dtype=bool)
            if min_dur > 0:
                mask &= durations >= min_dur
            if max_dur > 0:
                mask &= durations <= max_dur
            times = times[mask]
            durations = durations[mask]

        return times, durations

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

        ev, dur = self._get_events_for_proc(proc)
        ev, dur = self._filter_events(ev, dur)
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
            ev, dur = self._get_events_for_proc(proc)
            ev, dur = self._filter_events(ev, dur)
            if ev.size == 0:
                return
            tvec, mat = _compute_psth_matrix(proc.time, proc.output, ev, window, baseline, res_hz, smooth_sigma_s=smooth)
            self._render_heatmap(mat, tvec)
            self._render_avg(mat, tvec)
            self._render_duration_hist(dur if dur is not None else np.array([], float))
            self._render_metrics(mat, tvec)
            self._last_mat = mat
            self._last_tvec = tvec
            self._last_events = ev
            self._last_durations = dur
        else:
            # multiple recordings: one row per recording = average PSTH over events within that recording
            rows = []
            all_dur = []
            for proc in self._processed:
        ev, dur = self._get_events_for_proc(proc)
        ev, dur = self._filter_events(ev, dur)
                if ev.size == 0:
                    continue
                tvec, mat = _compute_psth_matrix(proc.time, proc.output, ev, window, baseline, res_hz, smooth_sigma_s=smooth)
                if mat.size == 0:
                    continue
                avg = np.nanmean(mat, axis=0)
                rows.append(avg)
                if dur is not None and len(dur):
                    all_dur.append(np.asarray(dur, float))
            if not rows:
                return
            mat_rec = np.vstack(rows)
            self._render_heatmap(mat_rec, tvec)
            self._render_avg(mat_rec, tvec)
            dur_all = np.concatenate(all_dur) if all_dur else np.array([], float)
            self._render_duration_hist(dur_all)
            self._render_metrics(mat_rec, tvec)
            self._last_mat = mat_rec
            self._last_tvec = tvec
            self._last_events = None
            self._last_durations = dur_all

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

    def _render_duration_hist(self, durations: np.ndarray) -> None:
        self.plot_dur.clear()
        if durations is None or durations.size == 0:
            return
        d = np.asarray(durations, float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return
        bins = min(20, max(5, int(np.sqrt(d.size))))
        hist, edges = np.histogram(d, bins=bins)
        y = edges[:-1]
        h = hist
        bg = pg.BarGraphItem(x=h / 2.0, y=y, width=h, height=np.diff(edges), brushes="#5a8fd6")
        self.plot_dur.addItem(bg)
        self.plot_dur.setYRange(float(edges[0]), float(edges[-1]), padding=0.1)
        self.plot_dur.setXRange(0, float(np.max(h)) if h.size else 1.0, padding=0.1)

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

    def _render_metrics(self, mat: np.ndarray, tvec: np.ndarray) -> None:
        if mat.size == 0:
            self.metrics_bar.setOpts(height=[0, 0])
            self._last_metrics = None
            return
        metric = self.combo_metric.currentText()
        pre0 = float(self.spin_metric_pre0.value())
        pre1 = float(self.spin_metric_pre1.value())
        post0 = float(self.spin_metric_post0.value())
        post1 = float(self.spin_metric_post1.value())

        def _window_vals(a: float, b: float) -> np.ndarray:
            mask = (tvec >= a) & (tvec <= b)
            if not np.any(mask):
                return np.array([], float)
            return mat[:, mask]

        pre = _window_vals(pre0, pre1)
        post = _window_vals(post0, post1)

        def _metric_vals(win: np.ndarray, duration: float) -> np.ndarray:
            if win.size == 0:
                return np.array([], float)
            if metric.startswith("AUC"):
                return np.nanmean(win, axis=1) * float(abs(duration))
            return np.nanmean(win, axis=1)

        pre_vals = _metric_vals(pre, pre1 - pre0)
        post_vals = _metric_vals(post, post1 - post0)
        pre_mean = float(np.nanmean(pre_vals)) if pre_vals.size else 0.0
        post_mean = float(np.nanmean(post_vals)) if post_vals.size else 0.0
        self.metrics_bar.setOpts(height=[pre_mean, post_mean])
        self._last_metrics = {"pre": pre_mean, "post": post_mean, "metric": metric}

    def _export_results(self) -> None:
        if self._last_mat is None or self._last_tvec is None:
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        choices = dlg.choices()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", os.getcwd())
        if not out_dir:
            return
        prefix = "postprocess"
        if self._processed:
            prefix = os.path.splitext(os.path.basename(self._processed[0].path))[0]

        if choices.get("heatmap"):
            np.savetxt(os.path.join(out_dir, f"{prefix}_heatmap.csv"), self._last_mat, delimiter=",")
            np.savetxt(os.path.join(out_dir, f"{prefix}_heatmap_tvec.csv"), self._last_tvec, delimiter=",")
        if choices.get("avg"):
            avg = np.nanmean(self._last_mat, axis=0)
            sem = np.nanstd(self._last_mat, axis=0) / np.sqrt(max(1, np.sum(np.any(np.isfinite(self._last_mat), axis=1))))
            arr = np.vstack([self._last_tvec, avg, sem]).T
            np.savetxt(os.path.join(out_dir, f"{prefix}_avg_psth.csv"), arr, delimiter=",", header="time,avg,sem", comments="")
        if choices.get("events") and self._last_events is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_events.csv"), self._last_events, delimiter=",")
        if choices.get("durations") and self._last_durations is not None:
            np.savetxt(os.path.join(out_dir, f"{prefix}_durations.csv"), self._last_durations, delimiter=",")
        if choices.get("metrics") and self._last_metrics:
            import csv
            with open(os.path.join(out_dir, f"{prefix}_metrics.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "pre", "post"])
                w.writerow([self._last_metrics.get("metric", ""), self._last_metrics.get("pre", ""), self._last_metrics.get("post", "")])


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        self.cb_heatmap = QtWidgets.QCheckBox("Heatmap matrix")
        self.cb_avg = QtWidgets.QCheckBox("Average PSTH")
        self.cb_events = QtWidgets.QCheckBox("Event times")
        self.cb_durations = QtWidgets.QCheckBox("Event durations")
        self.cb_metrics = QtWidgets.QCheckBox("Metrics table")
        for cb in (self.cb_heatmap, self.cb_avg, self.cb_events, self.cb_durations, self.cb_metrics):
            cb.setChecked(True)
            layout.addWidget(cb)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.setDefault(True)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        layout.addLayout(row)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def choices(self) -> Dict[str, bool]:
        return {
            "heatmap": self.cb_heatmap.isChecked(),
            "avg": self.cb_avg.isChecked(),
            "events": self.cb_events.isChecked(),
            "durations": self.cb_durations.isChecked(),
            "metrics": self.cb_metrics.isChecked(),
        }
