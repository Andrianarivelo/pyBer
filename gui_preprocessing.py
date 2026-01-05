# gui_preprocessing.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os

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


def _first_not_none(d: dict, *keys, default=None):
    """
    Return the first present key in d whose value is not None.
    Important: does NOT use boolean evaluation, so numpy arrays are safe.
    """
    for k in keys:
        if k in d:
            v = d.get(k, None)
            if v is not None:
                return v
    return default


# ----------------------------- Metadata dialog -----------------------------

class MetadataDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial: Optional[Dict[str, str]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.setModal(True)
        self._initial = dict(initial or {})
        self._build_ui()
        self._load_initial()

    def _build_ui(self) -> None:
        self.resize(520, 380)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.ed_animal = QtWidgets.QLineEdit()
        self.ed_session = QtWidgets.QLineEdit()
        self.ed_trial = QtWidgets.QLineEdit()
        self.ed_treat = QtWidgets.QLineEdit()

        form.addRow("Animal ID", self.ed_animal)
        form.addRow("Session", self.ed_session)
        form.addRow("Trial", self.ed_trial)
        form.addRow("Treatment", self.ed_treat)

        layout.addLayout(form)

        grp = QtWidgets.QGroupBox("Additional metadata (key/value)")
        v = QtWidgets.QVBoxLayout(grp)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        btnrow = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add row")
        self.btn_del = QtWidgets.QPushButton("Delete row")
        btnrow.addWidget(self.btn_add)
        btnrow.addWidget(self.btn_del)
        btnrow.addStretch(1)

        v.addWidget(self.table)
        v.addLayout(btnrow)
        layout.addWidget(grp, stretch=1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok.setDefault(True)
        row.addWidget(self.btn_ok)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        self.btn_add.clicked.connect(self._add_row)
        self.btn_del.clicked.connect(self._del_row)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

    def _load_initial(self) -> None:
        self.ed_animal.setText(self._initial.get("animal_id", ""))
        self.ed_session.setText(self._initial.get("session", ""))
        self.ed_trial.setText(self._initial.get("trial", ""))
        self.ed_treat.setText(self._initial.get("treatment", ""))

        reserved = {"animal_id", "session", "trial", "treatment"}
        extras = [(k, v) for k, v in self._initial.items() if k not in reserved]
        for k, v in extras:
            self._add_row(k, v)

    def _add_row(self, key: str = "", value: str = "") -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(key)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(value)))

    def _del_row(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        self.table.removeRow(sel[0].row())

    def metadata(self) -> Dict[str, str]:
        out: Dict[str, str] = {
            "animal_id": self.ed_animal.text().strip(),
            "session": self.ed_session.text().strip(),
            "trial": self.ed_trial.text().strip(),
            "treatment": self.ed_treat.text().strip(),
        }

        for r in range(self.table.rowCount()):
            k_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            k = (k_item.text().strip() if k_item else "")
            v = (v_item.text().strip() if v_item else "")
            if k:
                out[k] = v
        return out


# ----------------------------- Artifact panel -----------------------------

class ArtifactPanel(QtWidgets.QDialog):
    regionsChanged = QtCore.Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Artifact Regions")
        self.setModal(False)
        self.resize(520, 360)

        self._regions: List[Tuple[float, float]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Start (s)", "End (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        layout.addWidget(self.table)

        addrow = QtWidgets.QHBoxLayout()
        self.ed_start = QtWidgets.QDoubleSpinBox()
        self.ed_end = QtWidgets.QDoubleSpinBox()
        for ed in (self.ed_start, self.ed_end):
            ed.setDecimals(3)
            ed.setRange(-1e9, 1e9)
            ed.setKeyboardTracking(False)
            ed.setMinimumWidth(140)

        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_update = QtWidgets.QPushButton("Update selected")
        self.btn_del = QtWidgets.QPushButton("Delete selected")
        self.btn_clear = QtWidgets.QPushButton("Clear all")

        addrow.addWidget(QtWidgets.QLabel("Start:"))
        addrow.addWidget(self.ed_start)
        addrow.addWidget(QtWidgets.QLabel("End:"))
        addrow.addWidget(self.ed_end)
        addrow.addStretch(1)
        addrow.addWidget(self.btn_add)
        layout.addLayout(addrow)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.btn_update)
        btnrow.addWidget(self.btn_del)
        btnrow.addWidget(self.btn_clear)
        btnrow.addStretch(1)

        self.btn_close = QtWidgets.QPushButton("Close")
        btnrow.addWidget(self.btn_close)
        layout.addLayout(btnrow)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_update.clicked.connect(self._on_update_selected)
        self.btn_del.clicked.connect(self._on_delete_selected)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_close.clicked.connect(self.close)

        self.table.itemSelectionChanged.connect(self._sync_edits_from_selected)

    def set_regions(self, regions: List[Tuple[float, float]]) -> None:
        self._regions = [(float(a), float(b)) for a, b in (regions or [])]
        self._regions = [(min(a, b), max(a, b)) for a, b in self._regions]
        self._rebuild_table()

    def regions(self) -> List[Tuple[float, float]]:
        return list(self._regions)

    def _rebuild_table(self) -> None:
        self.table.setRowCount(0)
        for (a, b) in self._regions:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{a:.3f}"))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{b:.3f}"))

    def _emit(self) -> None:
        self.regionsChanged.emit(self.regions())

    def _sync_edits_from_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._regions):
            a, b = self._regions[r]
            self.ed_start.setValue(float(a))
            self.ed_end.setValue(float(b))

    def _on_add(self) -> None:
        a = float(self.ed_start.value())
        b = float(self.ed_end.value())
        if not np.isfinite(a) or not np.isfinite(b):
            return
        a, b = (min(a, b), max(a, b))
        self._regions.append((a, b))
        self._regions.sort(key=lambda x: x[0])
        self._rebuild_table()
        self._emit()

    def _on_update_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        a = float(self.ed_start.value())
        b = float(self.ed_end.value())
        a, b = (min(a, b), max(a, b))
        if 0 <= r < len(self._regions):
            self._regions[r] = (a, b)
            self._regions.sort(key=lambda x: x[0])
            self._rebuild_table()
            self._emit()

    def _on_delete_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._regions):
            del self._regions[r]
            self._rebuild_table()
            self._emit()

    def _on_clear(self) -> None:
        self._regions = []
        self._rebuild_table()
        self._emit()


# ----------------------------- File queue panel -----------------------------

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
        self._current_dir_hint: str = ""
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
        self.lbl_hint.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

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

    def set_path_hint(self, text: str) -> None:
        self.lbl_hint.setText(text)
        if text and os.path.isdir(text):
            self._current_dir_hint = text

    def path_hint(self) -> str:
        return self.lbl_hint.text()

    def set_current_dir_hint(self, dir_path: str) -> None:
        self._current_dir_hint = dir_path or ""
        if dir_path:
            self.lbl_hint.setText(dir_path)

    def current_dir_hint(self) -> str:
        return self._current_dir_hint

    def add_file(self, path: str) -> None:
        self.list_files.addItem(path)
        try:
            d = os.path.dirname(path)
            if d and os.path.isdir(d):
                self._current_dir_hint = d
        except Exception:
            pass

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


# ----------------------------- Parameter panel -----------------------------

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

        lam_row = QtWidgets.QHBoxLayout()
        lam_row.setSpacing(6)

        self.spin_lam_x = mk_dspin(minw=90, decimals=3)
        self.spin_lam_x.setRange(0.1, 9.999)
        self.spin_lam_x.setValue(1.0)

        self.lbl_e = QtWidgets.QLabel("e")
        self.lbl_e.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.spin_lam_y = mk_spin(minw=80)
        self.spin_lam_y.setRange(-3, 12)
        self.spin_lam_y.setValue(9)

        self.lbl_lam_preview = QtWidgets.QLabel("= 1e9")
        self.lbl_lam_preview.setProperty("class", "hint")

        lam_row.addWidget(self.spin_lam_x)
        lam_row.addWidget(self.lbl_e)
        lam_row.addWidget(self.spin_lam_y)
        lam_row.addWidget(self.lbl_lam_preview, stretch=1)

        lam_widget = QtWidgets.QWidget()
        lam_widget.setLayout(lam_row)

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
        form.addRow("Baseline γ / λ (x e y)", lam_widget)
        form.addRow("diff_order", self.spin_diff)
        form.addRow("max_iter", self.spin_iter)
        form.addRow("tol", self.spin_tol)
        form.addRow("AsLS p", self.spin_asls_p)

        form.addRow("Output mode", self.combo_output)
        form.addRow("Ref fit (z-reg only)", self.combo_ref_fit)
        form.addRow("Lasso α", self.spin_lasso)
        form.addRow("", self.lbl_fs)

        self._update_lambda_preview()

    def _update_lambda_preview(self) -> None:
        x = float(self.spin_lam_x.value())
        y = int(self.spin_lam_y.value())
        if abs(x - 1.0) < 1e-9:
            self.lbl_lam_preview.setText(f"= 1e{y}")
        else:
            self.lbl_lam_preview.setText(f"= {x:.3g}e{y}")

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
            self.spin_lam_x,
            self.spin_lam_y,
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

        self.spin_lam_x.valueChanged.connect(lambda *_: self._update_lambda_preview())
        self.spin_lam_y.valueChanged.connect(lambda *_: self._update_lambda_preview())

    def _lambda_value(self) -> float:
        x = float(self.spin_lam_x.value())
        y = int(self.spin_lam_y.value())
        return float(x * (10.0 ** y))

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
            baseline_lambda=float(self._lambda_value()),
            baseline_diff_order=int(self.spin_diff.value()),
            baseline_max_iter=int(self.spin_iter.value()),
            baseline_tol=float(self.spin_tol.value()),
            asls_p=float(self.spin_asls_p.value()),
            output_mode=self.combo_output.currentText(),
            reference_fit=self.combo_ref_fit.currentText(),
            lasso_alpha=float(self.spin_lasso.value()),
        )

    def set_fs_info(self, fs_actual: float, fs_target: float, fs_used: float) -> None:
        self.lbl_fs.setText(f"FS: actual={fs_actual:.2f} Hz → used={fs_used:.2f} Hz (target={fs_target:.2f})")


# ----------------------------- Plot dashboard -----------------------------

class PlotDashboard(QtWidgets.QWidget):
    manualRegionFromSelectorRequested = QtCore.Signal()
    clearManualRegionsRequested = QtCore.Signal()
    showArtifactsRequested = QtCore.Signal()

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
        self.curve_b465 = self.plot_proc.plot(
            pen=pg.mkPen((220, 220, 220), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.curve_b405 = self.plot_proc.plot(
            pen=pg.mkPen((160, 160, 160), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        )

        self.curve_out = self.plot_out.plot(pen=pg.mkPen((90, 190, 255), width=1.2))

        self.selector = pg.LinearRegionItem(values=(0, 1), brush=(80, 120, 200, 60))
        self.plot_raw.addItem(self.selector)

        self._dio_pen = pg.mkPen((230, 180, 80), width=1.2)
        self.vb_dio_raw, self.curve_dio_raw = self._add_dio_axis(self.plot_raw, "DIO")
        self.vb_dio_proc, self.curve_dio_proc = self._add_dio_axis(self.plot_proc, "DIO")
        self.vb_dio_out, self.curve_dio_out = self._add_dio_axis(self.plot_out, "DIO")

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

        self.plot_raw.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)
        self.plot_proc.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)
        self.plot_out.getViewBox().sigXRangeChanged.connect(self._emit_xrange_from_any)

    def _add_dio_axis(self, plot: pg.PlotWidget, label: str):
        pi = plot.getPlotItem()
        vb = pg.ViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setYRange(-0.1, 1.1)
        pi.showAxis("right")
        pi.getAxis("right").setLabel(label)
        pi.scene().addItem(vb)
        pi.getAxis("right").linkToView(vb)
        vb.setXLink(pi.vb)

        curve = pg.PlotCurveItem(pen=self._dio_pen)
        vb.addItem(curve)

        def _update():
            vb.setGeometry(pi.vb.sceneBoundingRect())
            vb.linkedViewChanged(pi.vb, vb.XAxis)

        pi.vb.sigResized.connect(_update)
        _update()
        return vb, curve

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

    def _set_dio(self, t: np.ndarray, dio: Optional[np.ndarray], name: str = "") -> None:
        if dio is None or np.asarray(dio).size == 0:
            self.curve_dio_raw.setData([], [])
            self.curve_dio_proc.setData([], [])
            self.curve_dio_out.setData([], [])
            return

        tt = np.asarray(t, float)
        yy = np.asarray(dio, float)
        n = min(tt.size, yy.size)
        tt, yy = tt[:n], yy[:n]

        self.curve_dio_raw.setData(tt, yy, connect="finite", skipFiniteCheck=True)
        self.curve_dio_proc.setData(tt, yy, connect="finite", skipFiniteCheck=True)
        self.curve_dio_out.setData(tt, yy, connect="finite", skipFiniteCheck=True)

        if name:
            self.plot_raw.getPlotItem().getAxis("right").setLabel(f"DIO ({name})")
            self.plot_proc.getPlotItem().getAxis("right").setLabel(f"DIO ({name})")
            self.plot_out.getPlotItem().getAxis("right").setLabel(f"DIO ({name})")
        else:
            self.plot_raw.getPlotItem().getAxis("right").setLabel("DIO")
            self.plot_proc.getPlotItem().getAxis("right").setLabel("DIO")
            self.plot_out.getPlotItem().getAxis("right").setLabel("DIO")

    # -------------------- Compatibility API expected by main.py --------------------

    def show_raw(self, *args, **kwargs) -> None:
        """
        Backward/forward compatible raw display.

        Supports any of:
          - show_raw(time, raw465, raw405, ...)
          - show_raw(time=..., raw465=..., raw405=...)
          - show_raw(time=..., signal_465=..., reference_405=...)
          - show_raw(time=..., raw_signal=..., raw_reference=...)
        """
        # Positional support: (time, sig, ref)
        t = s = r = None
        if len(args) >= 3:
            t, s, r = args[0], args[1], args[2]
        else:
            # keyword aliases (main.py uses raw465/raw405)
            t = _first_not_none(kwargs, "time", "t", "Time")
            s = _first_not_none(kwargs, "raw465", "signal_465", "raw_signal", "sig", "signal")
            r = _first_not_none(kwargs, "raw405", "reference_405", "raw_reference", "ref", "reference")

        if t is None or s is None or r is None:
            # fail silently but clear plot (prevents hard crashes)
            self.curve_465.setData([], [])
            self.curve_405.setData([], [])
            self.curve_thr_hi.setData([], [])
            self.curve_thr_lo.setData([], [])
            self._set_dio(np.asarray([]), None, "")
            return

        t = np.asarray(t, float)
        s = np.asarray(s, float)
        r = np.asarray(r, float)
        n = min(t.size, s.size, r.size)
        t, s, r = t[:n], s[:n], r[:n]

        self.curve_465.setData(t, s, connect="finite", skipFiniteCheck=True)
        self.curve_405.setData(t, r, connect="finite", skipFiniteCheck=True)

        # Thresholds (either scalars or arrays) — MUST NOT use "or" on arrays
        thr_hi = _first_not_none(kwargs, "thr_hi", "raw_thr_hi", "mad_hi", "hi_thr")
        thr_lo = _first_not_none(kwargs, "thr_lo", "raw_thr_lo", "mad_lo", "lo_thr")

        if thr_hi is None or thr_lo is None:
            self.curve_thr_hi.setData([], [])
            self.curve_thr_lo.setData([], [])
        else:
            th = np.asarray(thr_hi, float)
            tl = np.asarray(thr_lo, float)
            if th.size == 1:
                th = np.full_like(t, float(th))
            if tl.size == 1:
                tl = np.full_like(t, float(tl))
            nn = min(t.size, th.size, tl.size)
            self.curve_thr_hi.setData(t[:nn], th[:nn], connect="finite", skipFiniteCheck=True)
            self.curve_thr_lo.setData(t[:nn], tl[:nn], connect="finite", skipFiniteCheck=True)

        dio = _first_not_none(kwargs, "dio", "digital", "dio_y")
        dio_name = _first_not_none(kwargs, "dio_name", "digital_name", "trigger_name", default="") or ""
        self._set_dio(t, dio, str(dio_name))

        title = _first_not_none(kwargs, "title", "file_label")
        if title is not None:
            self.set_title(str(title))

    def show_processing(self, *args, **kwargs) -> None:
        """
        Compatible processing display.

        Supports:
          - show_processing(time, sig_f=..., ref_f=..., baseline_sig=..., baseline_ref=...)
          - show_processing(time=..., sig_f=..., ref_f=..., b_sig=..., b_ref=...)
        """
        if len(args) >= 1:
            t = args[0]
        else:
            t = _first_not_none(kwargs, "time", "t", "Time")

        if t is None:
            self.curve_f465.setData([], [])
            self.curve_f405.setData([], [])
            self.curve_b465.setData([], [])
            self.curve_b405.setData([], [])
            self._set_dio(np.asarray([]), None, "")
            return

        t = np.asarray(t, float)

        sig_f = _first_not_none(kwargs, "sig_f", "signal_f", "f465")
        ref_f = _first_not_none(kwargs, "ref_f", "reference_f", "f405")

        baseline_sig = _first_not_none(kwargs, "baseline_sig", "b_sig", "sig_base")
        baseline_ref = _first_not_none(kwargs, "baseline_ref", "b_ref", "ref_base")

        def _set_curve(curve, y):
            if y is None:
                curve.setData([], [])
                return
            y = np.asarray(y, float)
            n = min(t.size, y.size)
            curve.setData(t[:n], y[:n], connect="finite", skipFiniteCheck=True)

        _set_curve(self.curve_f465, sig_f)
        _set_curve(self.curve_f405, ref_f)
        _set_curve(self.curve_b465, baseline_sig)
        _set_curve(self.curve_b405, baseline_ref)

        dio = _first_not_none(kwargs, "dio", "digital", "dio_y")
        dio_name = _first_not_none(kwargs, "dio_name", "digital_name", "trigger_name", default="") or ""
        self._set_dio(t, dio, str(dio_name))

    def show_output(self, *args, **kwargs) -> None:
        """
        Compatible output display.

        Supports:
          - show_output(time, output, label="...", dio=..., dio_name=...)
          - show_output(time=..., y=..., label=...)
        """
        if len(args) >= 2:
            t, y = args[0], args[1]
        else:
            t = _first_not_none(kwargs, "time", "t", "Time")
            y = _first_not_none(kwargs, "output", "y", "dff", "zscore")

        if t is None or y is None:
            self.curve_out.setData([], [])
            self._set_dio(np.asarray([]), None, "")
            return

        t = np.asarray(t, float)
        y = np.asarray(y, float)
        n = min(t.size, y.size)
        t, y = t[:n], y[:n]

        self.curve_out.setData(t, y, connect="finite", skipFiniteCheck=True)

        label = _first_not_none(kwargs, "label", "output_label", default="Output")
        self.plot_out.setTitle(f"Output: {label}")

        dio = _first_not_none(kwargs, "dio", "digital", "dio_y")
        dio_name = _first_not_none(kwargs, "dio_name", "digital_name", "trigger_name", default="") or ""
        self._set_dio(t, dio, str(dio_name))

    # -------------------- Modern API (kept) --------------------

    def update_plots(self, processed: ProcessedTrial) -> None:
        t = np.asarray(processed.time, float)
        self.show_raw(
            t, processed.raw_signal, processed.raw_reference,
            dio=processed.dio, dio_name=processed.dio_name,
            thr_hi=processed.raw_thr_hi, thr_lo=processed.raw_thr_lo
        )
        self.show_processing(
            t,
            sig_f=processed.sig_f, ref_f=processed.ref_f,
            baseline_sig=processed.baseline_sig, baseline_ref=processed.baseline_ref,
            dio=processed.dio, dio_name=processed.dio_name
        )
        self.show_output(
            t, processed.output,
            label=processed.output_label,
            dio=processed.dio, dio_name=processed.dio_name
        )
