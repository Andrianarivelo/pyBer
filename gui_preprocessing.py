# gui_preprocessing.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from analysis_core import ProcessingParams, ProcessedTrial, OUTPUT_MODES, BASELINE_METHODS


def _optimize_plot(w: pg.PlotWidget) -> None:
    w.setMenuEnabled(True)
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setMouseEnabled(x=True, y=True)
    pi = w.getPlotItem()
    pi.setClipToView(True)
    pi.setDownsampling(auto=True, mode="peak")
    pi.setAutoVisible(y=True)
    try:
        pi.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
    except TypeError:
        try:
            pi.enableAutoRange(pg.ViewBox.YAxis)
        except Exception:
            pass


class FileQueuePanel(QtWidgets.QGroupBox):
    openFileRequested = QtCore.Signal()
    openFolderRequested = QtCore.Signal()
    selectionChanged = QtCore.Signal()

    channelChanged = QtCore.Signal(str)
    triggerChanged = QtCore.Signal(str)

    updatePreviewRequested = QtCore.Signal()
    metadataRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()
    toggleArtifactsRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Data", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Open .doric File…")
        self.btn_folder = QtWidgets.QPushButton("Add Folder…")
        self.btn_open.setProperty("class", "compact")
        self.btn_folder.setProperty("class", "compact")
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_folder)

        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_files.setMinimumHeight(110)

        self.grp_sel = QtWidgets.QGroupBox("Selection")
        form = QtWidgets.QFormLayout(self.grp_sel)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.combo_channel = QtWidgets.QComboBox()
        self.combo_channel.setMinimumWidth(220)

        self.combo_trigger = QtWidgets.QComboBox()
        self.combo_trigger.setMinimumWidth(220)
        self.combo_trigger.addItem("")

        form.addRow("Channel (preview)", self.combo_channel)
        form.addRow("Digital trigger (overlay)", self.combo_trigger)

        btnrow = QtWidgets.QGridLayout()
        btnrow.setHorizontalSpacing(8)
        btnrow.setVerticalSpacing(8)

        self.btn_metadata = QtWidgets.QPushButton("Metadata…")
        self.btn_update = QtWidgets.QPushButton("Update")
        self.btn_artifacts = QtWidgets.QPushButton("Artifacts…")
        self.btn_export = QtWidgets.QPushButton("Export CSV/H5…")

        for b in (self.btn_metadata, self.btn_update, self.btn_artifacts, self.btn_export):
            b.setProperty("class", "compact")
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                            QtWidgets.QSizePolicy.Policy.Fixed)

        self.btn_update.setProperty("class", "compactPrimary")
        self.btn_export.setProperty("class", "compactPrimary")

        btnrow.addWidget(self.btn_metadata, 0, 0)
        btnrow.addWidget(self.btn_update,   0, 1)
        btnrow.addWidget(self.btn_artifacts, 1, 0)
        btnrow.addWidget(self.btn_export,    1, 1)

        self.lbl_hint = QtWidgets.QLabel("")
        self.lbl_hint.setProperty("class", "hint")

        v.addLayout(row)
        v.addWidget(self.list_files)
        v.addWidget(self.grp_sel)
        v.addLayout(btnrow)
        v.addWidget(self.lbl_hint)

        self.btn_open.clicked.connect(self.openFileRequested.emit)
        self.btn_folder.clicked.connect(self.openFolderRequested.emit)
        self.list_files.itemSelectionChanged.connect(self.selectionChanged.emit)

        self.combo_channel.currentTextChanged.connect(self.channelChanged.emit)
        self.combo_trigger.currentTextChanged.connect(self.triggerChanged.emit)

        self.btn_update.clicked.connect(self.updatePreviewRequested.emit)
        self.btn_metadata.clicked.connect(self.metadataRequested.emit)
        self.btn_export.clicked.connect(self.exportRequested.emit)
        self.btn_artifacts.clicked.connect(self.toggleArtifactsRequested.emit)

    def add_file(self, path: str) -> None:
        self.list_files.addItem(path)

    def all_paths(self) -> List[str]:
        return [self.list_files.item(i).text() for i in range(self.list_files.count())]

    def selected_paths(self) -> List[str]:
        return [it.text() for it in self.list_files.selectedItems()]

    def set_available_channels(self, chans: List[str]) -> None:
        self.combo_channel.blockSignals(True)
        try:
            self.combo_channel.clear()
            for c in chans:
                self.combo_channel.addItem(c)
        finally:
            self.combo_channel.blockSignals(False)
        if chans:
            self.combo_channel.setCurrentIndex(0)

    def set_available_triggers(self, triggers: List[str]) -> None:
        self.combo_trigger.blockSignals(True)
        try:
            self.combo_trigger.clear()
            self.combo_trigger.addItem("")
            for t in triggers:
                self.combo_trigger.addItem(t)
        finally:
            self.combo_trigger.blockSignals(False)

    def set_channel(self, ch: str) -> None:
        idx = self.combo_channel.findText(ch)
        if idx >= 0:
            self.combo_channel.setCurrentIndex(idx)

    def set_trigger(self, trig: str) -> None:
        idx = self.combo_trigger.findText(trig)
        if idx >= 0:
            self.combo_trigger.setCurrentIndex(idx)

    def set_path_hint(self, path: str) -> None:
        self.lbl_hint.setText(path)

    def current_dir_hint(self) -> str:
        sel = self.selected_paths()
        if sel:
            return str(os.path.dirname(sel[0]))
        allp = self.all_paths()
        if allp:
            return str(os.path.dirname(allp[0]))
        txt = self.lbl_hint.text().strip()
        if txt and os.path.isdir(txt):
            return txt
        return ""


class ParameterPanel(QtWidgets.QGroupBox):
    paramsChanged = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Processing Parameters", parent)
        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        form = QtWidgets.QFormLayout(self)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        def mk_dspin(minw=210, decimals=3) -> QtWidgets.QDoubleSpinBox:
            s = QtWidgets.QDoubleSpinBox()
            s.setMinimumWidth(minw)
            s.setDecimals(decimals)
            s.setKeyboardTracking(False)
            return s

        def mk_spin(minw=210) -> QtWidgets.QSpinBox:
            s = QtWidgets.QSpinBox()
            s.setMinimumWidth(minw)
            s.setKeyboardTracking(False)
            return s

        self.combo_artifact = QtWidgets.QComboBox()
        self.combo_artifact.addItems(["Global MAD (dx)", "Adaptive MAD (windowed)"])

        self.spin_mad = mk_dspin()
        self.spin_mad.setRange(1.0, 50.0)
        self.spin_mad.setValue(8.0)

        self.spin_adapt_win = mk_dspin()
        self.spin_adapt_win.setRange(0.2, 60.0)
        self.spin_adapt_win.setValue(5.0)

        self.spin_pad = mk_dspin()
        self.spin_pad.setRange(0.0, 10.0)
        self.spin_pad.setValue(0.25)

        self.spin_lowpass = mk_dspin()
        self.spin_lowpass.setRange(0.1, 200.0)
        self.spin_lowpass.setValue(12.0)

        self.spin_filt_order = mk_spin()
        self.spin_filt_order.setRange(1, 8)
        self.spin_filt_order.setValue(3)

        self.spin_target_fs = mk_dspin(decimals=1)
        self.spin_target_fs.setRange(1.0, 1000.0)
        self.spin_target_fs.setValue(100.0)

        self.combo_baseline = QtWidgets.QComboBox()
        self.combo_baseline.addItems([m for m in BASELINE_METHODS])

        self.spin_lambda = mk_dspin(decimals=0)
        self.spin_lambda.setRange(1e3, 1e12)
        self.spin_lambda.setValue(1e8)

        self.spin_diff = mk_spin()
        self.spin_diff.setRange(1, 3)
        self.spin_diff.setValue(2)

        self.spin_iter = mk_spin()
        self.spin_iter.setRange(1, 200)
        self.spin_iter.setValue(50)

        self.spin_tol = mk_dspin(decimals=6)
        self.spin_tol.setRange(1e-8, 1e-1)
        self.spin_tol.setValue(1e-3)

        self.spin_asls_p = mk_dspin(decimals=4)
        self.spin_asls_p.setRange(0.001, 0.5)
        self.spin_asls_p.setValue(0.01)

        self.combo_output = QtWidgets.QComboBox()
        self.combo_output.addItems(OUTPUT_MODES)

        self.combo_ref_fit = QtWidgets.QComboBox()
        self.combo_ref_fit.addItems(["OLS (recommended)", "Lasso"])

        self.spin_lasso = mk_dspin(decimals=6)
        self.spin_lasso.setRange(1e-6, 1.0)
        self.spin_lasso.setValue(1e-3)

        self.lbl_fs = QtWidgets.QLabel("FS: —")
        self.lbl_fs.setProperty("class", "hint")

        form.addRow("Artifact detection", self.combo_artifact)
        form.addRow("MAD threshold (k)", self.spin_mad)
        form.addRow("Adaptive window (s)", self.spin_adapt_win)
        form.addRow("Artifact pad (s)", self.spin_pad)
        form.addRow("Low-pass cutoff (Hz)", self.spin_lowpass)
        form.addRow("Filter order", self.spin_filt_order)

        form.addRow("Target FS (Hz)", self.spin_target_fs)

        form.addRow("Baseline method", self.combo_baseline)
        form.addRow("Baseline λ", self.spin_lambda)
        form.addRow("diff_order", self.spin_diff)
        form.addRow("max_iter", self.spin_iter)
        form.addRow("tol", self.spin_tol)
        form.addRow("AsLS p", self.spin_asls_p)

        form.addRow("Output mode", self.combo_output)
        form.addRow("Ref fit (z-reg only)", self.combo_ref_fit)
        form.addRow("Lasso α", self.spin_lasso)

        form.addRow("", self.lbl_fs)

    def _wire(self) -> None:
        def emit_noargs(*_args) -> None:
            self.paramsChanged.emit()

        widgets = (
            self.combo_artifact,
            self.spin_mad,
            self.spin_adapt_win,
            self.spin_pad,
            self.spin_lowpass,
            self.spin_filt_order,
            self.spin_target_fs,
            self.combo_baseline,
            self.spin_lambda,
            self.spin_diff,
            self.spin_iter,
            self.spin_tol,
            self.spin_asls_p,
            self.combo_output,
            self.combo_ref_fit,
            self.spin_lasso,
        )
        for w in widgets:
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(emit_noargs)
            else:
                w.valueChanged.connect(emit_noargs)

    def get_params(self) -> ProcessingParams:
        return ProcessingParams(
            artifact_mode=self.combo_artifact.currentText(),
            mad_k=float(self.spin_mad.value()),
            adaptive_window_s=float(self.spin_adapt_win.value()),
            artifact_pad_s=float(self.spin_pad.value()),
            lowpass_hz=float(self.spin_lowpass.value()),
            filter_order=int(self.spin_filt_order.value()),
            target_fs_hz=float(self.spin_target_fs.value()),
            baseline_method=self.combo_baseline.currentText(),
            baseline_lambda=float(self.spin_lambda.value()),
            baseline_diff_order=int(self.spin_diff.value()),
            baseline_max_iter=int(self.spin_iter.value()),
            baseline_tol=float(self.spin_tol.value()),
            asls_p=float(self.spin_asls_p.value()),
            output_mode=self.combo_output.currentText(),
            reference_fit=self.combo_ref_fit.currentText(),
            lasso_alpha=float(self.spin_lasso.value()),
        )

    def set_params(self, p: ProcessingParams) -> None:
        self.blockSignals(True)
        try:
            def set_combo(combo: QtWidgets.QComboBox, text: str) -> None:
                idx = combo.findText(text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

            set_combo(self.combo_artifact, p.artifact_mode)
            self.spin_mad.setValue(float(p.mad_k))
            self.spin_adapt_win.setValue(float(p.adaptive_window_s))
            self.spin_pad.setValue(float(p.artifact_pad_s))
            self.spin_lowpass.setValue(float(p.lowpass_hz))
            self.spin_filt_order.setValue(int(p.filter_order))

            self.spin_target_fs.setValue(float(p.target_fs_hz))

            set_combo(self.combo_baseline, p.baseline_method)
            self.spin_lambda.setValue(float(p.baseline_lambda))
            self.spin_diff.setValue(int(p.baseline_diff_order))
            self.spin_iter.setValue(int(p.baseline_max_iter))
            self.spin_tol.setValue(float(p.baseline_tol))
            self.spin_asls_p.setValue(float(p.asls_p))

            set_combo(self.combo_output, p.output_mode)
            set_combo(self.combo_ref_fit, p.reference_fit)
            self.spin_lasso.setValue(float(p.lasso_alpha))
        finally:
            self.blockSignals(False)
        self.paramsChanged.emit()

    def set_fs_info(self, fs_actual: float, fs_target: float, fs_used: float) -> None:
        self.lbl_fs.setText(f"FS: actual={fs_actual:.2f} Hz → used={fs_used:.2f} Hz (target={fs_target:.2f})")


class MetadataDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], existing: Optional[Dict[str, Dict[str, str]]] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.resize(520, 360)

        self.channels = channels
        self.existing = existing or {}
        self._edits: Dict[str, Dict[str, QtWidgets.QLineEdit]] = {}

        v = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()

        for ch in channels:
            w = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(w)
            f.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

            e_animal = QtWidgets.QLineEdit()
            e_session = QtWidgets.QLineEdit()
            e_trial = QtWidgets.QLineEdit()
            e_treat = QtWidgets.QLineEdit()

            prev = self.existing.get(ch, {})
            e_animal.setText(prev.get("animal_id", ""))
            e_session.setText(prev.get("session", ""))
            e_trial.setText(prev.get("trial", ""))
            e_treat.setText(prev.get("treatment", ""))

            f.addRow("Animal ID", e_animal)
            f.addRow("Session", e_session)
            f.addRow("Trial", e_trial)
            f.addRow("Treatment", e_treat)

            self._edits[ch] = {
                "animal_id": e_animal,
                "session": e_session,
                "trial": e_trial,
                "treatment": e_treat,
            }

            tabs.addTab(w, ch)

        v.addWidget(tabs)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        ok = QtWidgets.QPushButton("OK")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.setProperty("class", "compactPrimary")
        cancel.setProperty("class", "compact")
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)

        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

    def get_metadata(self) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for ch, eds in self._edits.items():
            out[ch] = {k: e.text().strip() for k, e in eds.items()}
        return out


class ArtifactPanel(QtWidgets.QWidget):
    regionsChanged = QtCore.Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(8)

        self.list = QtWidgets.QListWidget()

        row = QtWidgets.QHBoxLayout()
        self.spin_t0 = QtWidgets.QDoubleSpinBox()
        self.spin_t1 = QtWidgets.QDoubleSpinBox()
        for s in (self.spin_t0, self.spin_t1):
            s.setDecimals(3)
            s.setRange(-1e9, 1e9)
            s.setMinimumWidth(120)
            s.setKeyboardTracking(False)

        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_del = QtWidgets.QPushButton("Delete")
        self.btn_add.setProperty("class", "compactPrimary")
        self.btn_del.setProperty("class", "compact")

        row.addWidget(QtWidgets.QLabel("t0"))
        row.addWidget(self.spin_t0)
        row.addWidget(QtWidgets.QLabel("t1"))
        row.addWidget(self.spin_t1)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_del)

        v.addWidget(self.list)
        v.addLayout(row)

        self.btn_add.clicked.connect(self._add)
        self.btn_del.clicked.connect(self._delete)

    def set_regions(self, regions: List[Tuple[float, float]]) -> None:
        self.list.clear()
        for t0, t1 in regions:
            self.list.addItem(f"{t0:.3f}  →  {t1:.3f}")

    def get_regions(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for i in range(self.list.count()):
            txt = self.list.item(i).text().replace("→", " ")
            parts = txt.split()
            if len(parts) >= 2:
                try:
                    t0 = float(parts[0])
                    t1 = float(parts[-1])
                    out.append((min(t0, t1), max(t0, t1)))
                except Exception:
                    pass
        return out

    def _add(self) -> None:
        t0 = float(self.spin_t0.value())
        t1 = float(self.spin_t1.value())
        self.list.addItem(f"{min(t0,t1):.3f}  →  {max(t0,t1):.3f}")
        self.regionsChanged.emit(self.get_regions())

    def _delete(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            self.list.takeItem(row)
            self.regionsChanged.emit(self.get_regions())


class PlotDashboard(QtWidgets.QWidget):
    manualRegionFromSelectorRequested = QtCore.Signal()
    clearManualRegionsRequested = QtCore.Signal()
    showArtifactsRequested = QtCore.Signal()

    # emits (x0, x1) for sync
    xRangeChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sync_guard = False
        self._build_ui()

    def _build_ui(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.lbl_title = QtWidgets.QLabel("No file loaded")
        self.lbl_title.setStyleSheet("font-weight: 900; font-size: 12pt;")
        top.addWidget(self.lbl_title)
        top.addStretch(1)

        self.btn_add_region = QtWidgets.QPushButton("Add from selector")
        self.btn_clear_regions = QtWidgets.QPushButton("Clear manual")
        self.btn_artifacts = QtWidgets.QPushButton("Artifacts…")

        self.btn_add_region.setProperty("class", "compactPrimary")
        self.btn_clear_regions.setProperty("class", "compact")
        self.btn_artifacts.setProperty("class", "compactPrimary")

        top.addWidget(self.btn_add_region)
        top.addWidget(self.btn_clear_regions)
        top.addWidget(self.btn_artifacts)
        v.addLayout(top)

        self.plot_raw = pg.PlotWidget(title="Raw signals (465 / 405) — decimated")
        self.plot_proc = pg.PlotWidget(title="Filtered + baselines (decimated)")
        self.plot_out = pg.PlotWidget(title="Output (decimated)")

        for w in (self.plot_raw, self.plot_proc, self.plot_out):
            _optimize_plot(w)

        self.curve_465 = self.plot_raw.plot(pen=pg.mkPen((80, 250, 160), width=1.3))
        self.curve_405 = self.plot_raw.plot(pen=pg.mkPen((160, 120, 255), width=1.2))
        pen_env = pg.mkPen((240, 200, 90), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        self.curve_thr_hi = self.plot_raw.plot(pen=pen_env)
        self.curve_thr_lo = self.plot_raw.plot(pen=pen_env)

        self.curve_f465 = self.plot_proc.plot(pen=pg.mkPen((80, 250, 160), width=1.1))
        self.curve_f405 = self.plot_proc.plot(pen=pg.mkPen((160, 120, 255), width=1.0))
        self.curve_b465 = self.plot_proc.plot(pen=pg.mkPen((220, 220, 220), width=1.0, style=QtCore.Qt.PenStyle.DashLine))
        self.curve_b405 = self.plot_proc.plot(pen=pg.mkPen((160, 160, 160), width=1.0, style=QtCore.Qt.PenStyle.DashLine))

        self.curve_out = self.plot_out.plot(pen=pg.mkPen((90, 190, 255), width=1.2))

        self.selector = pg.LinearRegionItem(values=(0, 1), brush=(80, 120, 200, 60))
        self.plot_raw.addItem(self.selector)

        self.lbl_log = QtWidgets.QLabel("")
        self.lbl_log.setProperty("class", "hint")

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.plot_splitter.addWidget(self.plot_raw)
        self.plot_splitter.addWidget(self.plot_proc)
        self.plot_splitter.addWidget(self.plot_out)
        self.plot_splitter.setSizes([320, 260, 260])

        v.addWidget(self.plot_splitter, stretch=1)
        v.addWidget(self.lbl_log)

        self.btn_add_region.clicked.connect(self.manualRegionFromSelectorRequested.emit)
        self.btn_clear_regions.clicked.connect(self.clearManualRegionsRequested.emit)
        self.btn_artifacts.clicked.connect(self.showArtifactsRequested.emit)

        # IMPORTANT FIX:
        # sigXRangeChanged emits (viewbox, (x0, x1)), not ((x0,x1),(y0,y1))
        self.plot_raw.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)
        self.plot_proc.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)
        self.plot_out.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)

    def _emit_xrange_from_any(self, _vb, x_range) -> None:
        if self._sync_guard:
            return
        try:
            x0, x1 = x_range
            self.xRangeChanged.emit(float(x0), float(x1))
        except Exception:
            pass

    def set_xrange_all(self, x0: float, x1: float) -> None:
        self._sync_guard = True
        try:
            self.plot_raw.setXRange(x0, x1, padding=0)
            self.plot_proc.setXRange(x0, x1, padding=0)
            self.plot_out.setXRange(x0, x1, padding=0)
        finally:
            self._sync_guard = False

    def set_title(self, text: str) -> None:
        self.lbl_title.setText(text)

    def set_log(self, msg: str) -> None:
        self.lbl_log.setText(msg)

    def selector_region(self) -> Tuple[float, float]:
        r = self.selector.getRegion()
        return float(min(r)), float(max(r))

    def show_raw(
        self,
        time: np.ndarray,
        raw465: np.ndarray,
        raw405: np.ndarray,
        trig_time: Optional[np.ndarray],
        trig: Optional[np.ndarray],
        trig_label: str,
        manual_regions: List[Tuple[float, float]],
    ) -> None:
        t = np.asarray(time, float)
        self.curve_465.setData(t, np.asarray(raw465, float), connect="finite", skipFiniteCheck=True)
        self.curve_405.setData(t, np.asarray(raw405, float), connect="finite", skipFiniteCheck=True)
        self.curve_thr_hi.setData([], [])
        self.curve_thr_lo.setData([], [])
        if t.size:
            self.selector.setRegion((float(t[0]), float(t[0] + min(1.0, (t[-1] - t[0]) * 0.05))))

    def update_plots(self, processed: ProcessedTrial) -> None:
        t = np.asarray(processed.time, float)

        self.curve_465.setData(t, np.asarray(processed.raw_signal, float), connect="finite", skipFiniteCheck=True)
        self.curve_405.setData(t, np.asarray(processed.raw_reference, float), connect="finite", skipFiniteCheck=True)

        if processed.raw_thr_hi is not None and processed.raw_thr_lo is not None:
            self.curve_thr_hi.setData(t, np.asarray(processed.raw_thr_hi, float), connect="finite", skipFiniteCheck=True)
            self.curve_thr_lo.setData(t, np.asarray(processed.raw_thr_lo, float), connect="finite", skipFiniteCheck=True)

        if processed.sig_f is not None:
            self.curve_f465.setData(t, np.asarray(processed.sig_f, float), connect="finite", skipFiniteCheck=True)
        if processed.ref_f is not None:
            self.curve_f405.setData(t, np.asarray(processed.ref_f, float), connect="finite", skipFiniteCheck=True)
        if processed.baseline_sig is not None:
            self.curve_b465.setData(t, np.asarray(processed.baseline_sig, float), connect="finite", skipFiniteCheck=True)
        if processed.baseline_ref is not None:
            self.curve_b405.setData(t, np.asarray(processed.baseline_ref, float), connect="finite", skipFiniteCheck=True)

        if processed.output is not None:
            self.curve_out.setData(t, np.asarray(processed.output, float), connect="finite", skipFiniteCheck=True)

        self.plot_out.setTitle(f"Output: {processed.output_label}")
