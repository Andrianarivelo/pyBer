# main.py
"""
Fiber Photometry Processor (Doric .doric) — PySide6 + pyqtgraph

Run:
    python main.py

Dependencies:
    pip install PySide6 pyqtgraph h5py numpy scipy scikit-learn pybaselines
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from analysis_core import (
    PhotometryProcessor,
    ProcessingParams,
    LoadedDoricFile,
    LoadedTrial,
    ProcessedTrial,
    export_processed_csv,
    export_processed_h5,
    safe_stem_from_metadata,
)
from gui_preprocessing import FileQueuePanel, ParameterPanel, PlotDashboard, MetadataDialog, ArtifactPanel
from gui_postprocessing import PostProcessingPanel
from styles import APP_QSS
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fiber Photometry Processor (Doric .doric) — PySide6")
        self.resize(1500, 900)

        # Core
        self.processor = PhotometryProcessor()

        # State
        self._loaded_files: Dict[str, LoadedDoricFile] = {}
        self._current_path: Optional[str] = None
        self._current_channel: Optional[str] = None
        self._current_trigger: Optional[str] = None

        self._manual_regions_by_key: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._metadata_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}

        self._last_processed: Dict[Tuple[str, str], ProcessedTrial] = {}

        # Worker infra (stable)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._job_counter = 0
        self._latest_job_id = 0

        # Debounce
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._start_preview_processing)

        # Settings (persist folder + params)
        self.settings = QtCore.QSettings("FiberPhotometryApp", "DoricProcessor")

        self._build_ui()
        self._restore_settings()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.setStyleSheet(APP_QSS)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Preprocessing tab
        pre = QtWidgets.QWidget()
        self.tabs.addTab(pre, "Preprocessing")

        self.file_panel = FileQueuePanel()
        self.param_panel = ParameterPanel()
        self.plots = PlotDashboard()
        self.artifact_panel = ArtifactPanel()

        # Right artifact panel dock (hidden by default)
        self.art_dock = QtWidgets.QDockWidget("Artifacts", self)
        self.art_dock.setWidget(self.artifact_panel)
        self.art_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.art_dock.setVisible(False)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.art_dock)

        # Left column: file + selection + buttons + parameters, all scrollable
        left_container = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_container)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(10)
        left_v.addWidget(self.file_panel)
        left_v.addWidget(self.param_panel, stretch=1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)

        # Make panels resizable: left | plots (right dock already resizable by Qt)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.plots)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        pre_layout = QtWidgets.QVBoxLayout(pre)
        pre_layout.setContentsMargins(10, 10, 10, 10)
        pre_layout.addWidget(splitter)

        # Postprocessing tab
        self.post_tab = PostProcessingPanel()
        self.tabs.addTab(self.post_tab, "Post Processing")

        # Wiring - file panel
        self.file_panel.openFileRequested.connect(self._open_files_dialog)
        self.file_panel.openFolderRequested.connect(self._open_folder_dialog)
        self.file_panel.selectionChanged.connect(self._on_file_selection_changed)
        self.file_panel.channelChanged.connect(self._on_channel_changed)
        self.file_panel.triggerChanged.connect(self._on_trigger_changed)

        self.file_panel.updatePreviewRequested.connect(self._trigger_preview)
        self.file_panel.metadataRequested.connect(self._edit_metadata_for_current)
        self.file_panel.exportRequested.connect(self._export_selected_or_all)
        self.file_panel.toggleArtifactsRequested.connect(self._toggle_artifacts_panel)

        # Parameters -> debounce preview
        self.param_panel.paramsChanged.connect(self._trigger_preview)

        # Plot sync
        self.plots.xRangeChanged.connect(self.plots.set_xrange_all)

        # Manual artifacts
        self.plots.manualRegionFromSelectorRequested.connect(self._add_manual_region_from_selector)
        self.plots.clearManualRegionsRequested.connect(self._clear_manual_regions_current)
        self.plots.showArtifactsRequested.connect(self._toggle_artifacts_panel)

        self.artifact_panel.regionsChanged.connect(self._artifact_regions_changed)

        # Postprocessing needs access to "current processed"
        self.post_tab.requestCurrentProcessed.connect(self._post_get_current_processed)
        self.post_tab.requestDioList.connect(self._post_get_current_dio_list)
        self.post_tab.requestDioData.connect(self._post_get_dio_data_for_path)

    # ---------------- Settings persistence ----------------

    def _restore_settings(self) -> None:
        last_dir = self.settings.value("last_open_dir", "", type=str)
        if last_dir and os.path.isdir(last_dir):
            self.file_panel.set_path_hint(last_dir)

        # restore params
        try:
            raw = self.settings.value("params_json", "", type=str)
            if raw:
                d = json.loads(raw)
                p = ProcessingParams.from_dict(d)
                self.param_panel.set_params(p)
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            last_dir = self.file_panel.current_dir_hint()
            if last_dir:
                self.settings.setValue("last_open_dir", last_dir)
        except Exception:
            pass

        try:
            p = self.param_panel.get_params()
            self.settings.setValue("params_json", json.dumps(p.to_dict()))
        except Exception:
            pass

    # ---------------- File loading ----------------

    def _open_files_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Doric .doric files",
            start_dir,
            "Doric files (*.doric *.h5 *.hdf5);;All files (*.*)",
        )
        if not paths:
            return

        self.settings.setValue("last_open_dir", os.path.dirname(paths[0]))
        self._add_files(paths)

    def _open_folder_dialog(self) -> None:
        start_dir = self.file_panel.current_dir_hint() or self.settings.value("last_open_dir", "", type=str) or os.getcwd()
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add folder with .doric", start_dir)
        if not folder:
            return
        self.settings.setValue("last_open_dir", folder)

        paths: List[str] = []
        for fn in os.listdir(folder):
            if fn.lower().endswith((".doric", ".h5", ".hdf5")):
                paths.append(os.path.join(folder, fn))
        paths.sort()
        self._add_files(paths)

    def _add_files(self, paths: List[str]) -> None:
        for p in paths:
            if p in self._loaded_files:
                continue
            try:
                doric = self.processor.load_file(p)
                self._loaded_files[p] = doric
                self.file_panel.add_file(p)
                self.plots.set_log(f"Loaded: {p}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load:\n{p}\n\n{e}")

        # set current selection -> triggers preview
        self._on_file_selection_changed()

    # ---------------- Current selection ----------------

    def _selected_paths(self) -> List[str]:
        return self.file_panel.selected_paths()

    def _current_key(self) -> Optional[Tuple[str, str]]:
        if not self._current_path or not self._current_channel:
            return None
        return (self._current_path, self._current_channel)

    def _on_file_selection_changed(self) -> None:
        sel = self._selected_paths()
        if not sel:
            return

        # preview shows first selected
        path = sel[0]
        self._current_path = path

        doric = self._loaded_files.get(path)
        if not doric:
            return

        self.file_panel.set_available_channels(doric.channels)
        self.file_panel.set_available_triggers(sorted(doric.digital_by_name.keys()))

        # keep channel if still valid
        if self._current_channel in doric.channels:
            self.file_panel.set_channel(self._current_channel)
        else:
            self._current_channel = doric.channels[0] if doric.channels else None
            if self._current_channel:
                self.file_panel.set_channel(self._current_channel)

        # keep trigger if still valid
        if self._current_trigger and self._current_trigger not in doric.digital_by_name:
            self._current_trigger = None
            self.file_panel.set_trigger("")

        self._update_raw_plot()
        self._trigger_preview()

        # update post tab selection context
        self.post_tab.set_current_source_label(os.path.basename(path), self._current_channel or "")

    def _on_channel_changed(self, ch: str) -> None:
        self._current_channel = ch
        self._update_raw_plot()
        self._trigger_preview()
        self.post_tab.set_current_source_label(os.path.basename(self._current_path or ""), self._current_channel or "")

    def _on_trigger_changed(self, trig: str) -> None:
        self._current_trigger = trig if trig else None
        self._update_raw_plot()

    # ---------------- Raw plot update ----------------

    def _update_raw_plot(self) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)
        key = (self._current_path, self._current_channel)
        manual = self._manual_regions_by_key.get(key, [])

        self.plots.set_title(os.path.basename(self._current_path))
        self.plots.show_raw(
            time=trial.time,
            raw465=trial.signal_465,
            raw405=trial.reference_405,
            trig_time=trial.trigger_time,
            trig=trial.trigger,
            trig_label=self._current_trigger or "",
            manual_regions=manual,
        )

    # ---------------- Preview processing (worker) ----------------

    def _trigger_preview(self) -> None:
        # persist params quickly
        self._save_settings()
        self._preview_timer.start()

    def _start_preview_processing(self) -> None:
        if not self._current_path or not self._current_channel:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        params = self.param_panel.get_params()
        trial = doric.make_trial(self._current_channel, trigger_name=self._current_trigger)

        key = (self._current_path, self._current_channel)
        manual = self._manual_regions_by_key.get(key, [])
        self._job_counter += 1
        job_id = self._job_counter
        self._latest_job_id = job_id

        self.plots.set_log(
            f"Processing preview… (fs={trial.sampling_rate:.2f} Hz → target {params.target_fs_hz:.1f} Hz, "
            f"baseline={params.baseline_method})"
        )

        task = self.processor.make_preview_task(
            trial=trial,
            params=params,
            manual_regions_sec=manual,
            job_id=job_id,
        )
        task.signals.finished.connect(self._on_preview_finished)
        task.signals.failed.connect(self._on_preview_failed)
        self._pool.start(task)

    @QtCore.Slot(object, int, float)
    def _on_preview_finished(self, processed: ProcessedTrial, job_id: int, elapsed_s: float) -> None:
        if job_id != self._latest_job_id:
            return  # ignore stale jobs

        key = (processed.path, processed.channel_id)
        self._last_processed[key] = processed

        # Update artifact panel regions list
        regs = processed.artifact_regions_sec or []
        self.artifact_panel.set_regions(regs)

        # Update plots (decimated signals)
        self.plots.update_plots(processed)

        self.plots.set_log(
            f"Preview updated: {processed.output_label} | fs={processed.fs_actual:.2f}→{processed.fs_used:.2f} Hz "
            f"(target {processed.fs_target:.2f}) | n={processed.time.size} | {elapsed_s*1000:.0f} ms"
        )

        # Inform post tab that current processed changed
        self.post_tab.notify_preprocessing_updated(processed)

    @QtCore.Slot(str, int)
    def _on_preview_failed(self, err: str, job_id: int) -> None:
        if job_id != self._latest_job_id:
            return
        self.plots.set_log(f"Preview error: {err}")

    # ---------------- Manual artifacts ----------------

    def _add_manual_region_from_selector(self) -> None:
        key = self._current_key()
        if not key:
            return
        t0, t1 = self.plots.selector_region()
        regs = self._manual_regions_by_key.get(key, [])
        regs.append((min(t0, t1), max(t0, t1)))
        self._manual_regions_by_key[key] = regs
        self.artifact_panel.set_regions(regs)
        self._trigger_preview()

    def _clear_manual_regions_current(self) -> None:
        key = self._current_key()
        if not key:
            return
        self._manual_regions_by_key[key] = []
        self.artifact_panel.set_regions([])
        self._trigger_preview()

    def _artifact_regions_changed(self, regions: List[Tuple[float, float]]) -> None:
        key = self._current_key()
        if not key:
            return
        self._manual_regions_by_key[key] = regions
        self._trigger_preview()

    def _toggle_artifacts_panel(self) -> None:
        self.art_dock.setVisible(not self.art_dock.isVisible())

    # ---------------- Metadata ----------------

    def _edit_metadata_for_current(self) -> None:
        if not self._current_path:
            return
        doric = self._loaded_files.get(self._current_path)
        if not doric:
            return

        # existing per channel
        existing: Dict[str, Dict[str, str]] = {}
        for ch in doric.channels:
            existing[ch] = self._metadata_by_key.get((self._current_path, ch), {})

        dlg = MetadataDialog(channels=doric.channels, existing=existing, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        meta = dlg.get_metadata()
        for ch, md in meta.items():
            self._metadata_by_key[(self._current_path, ch)] = md

    # ---------------- Export (multi-file) ----------------

    def _export_selected_or_all(self) -> None:
        selected = self._selected_paths()
        if not selected:
            selected = self.file_panel.all_paths()
        if not selected:
            return

        start_dir = self.settings.value("last_save_dir", "", type=str) or self.file_panel.current_dir_hint() or os.getcwd()
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", start_dir)
        if not out_dir:
            return
        self.settings.setValue("last_save_dir", out_dir)

        params = self.param_panel.get_params()

        # Process/export each selected file, for each channel detected
        # For speed, use pipeline (decimated) and reuse loaded files.
        n_total = 0
        for path in selected:
            doric = self._loaded_files.get(path)
            if not doric:
                continue
            for ch in doric.channels:
                key = (path, ch)
                trial = doric.make_trial(ch, trigger_name=self._current_trigger)  # export uses current trigger selection
                manual = self._manual_regions_by_key.get(key, [])
                meta = self._metadata_by_key.get(key, {})

                try:
                    processed = self.processor.process_trial(
                        trial=trial,
                        params=params,
                        manual_regions_sec=manual,
                        preview_mode=False,
                    )
                    stem = safe_stem_from_metadata(path, ch, meta)
                    csv_path = os.path.join(out_dir, f"{stem}.csv")
                    h5_path = os.path.join(out_dir, f"{stem}.h5")
                    export_processed_csv(csv_path, processed)
                    export_processed_h5(h5_path, processed, metadata=meta)
                    n_total += 1
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Export error", f"Failed export:\n{path} [{ch}]\n\n{e}")

        self.plots.set_log(f"Export complete: {n_total} recording(s) written to {out_dir}")

        # optional: update post tab list by loading exported results? (user can load later)

    # ---------------- Postprocessing bridge ----------------

    @QtCore.Slot()
    def _post_get_current_processed(self):
        # Determine selection context: if multiple selected, provide multiple processed outputs if available
        paths = self._selected_paths()
        if not paths:
            paths = [self._current_path] if self._current_path else []

        out: List[ProcessedTrial] = []
        for p in paths:
            doric = self._loaded_files.get(p)
            if not doric:
                continue
            # Use current channel for that file if previewed; otherwise use AIN01
            ch = self._current_channel if (p == self._current_path and self._current_channel) else (doric.channels[0] if doric.channels else "AIN01")
            key = (p, ch)
            if key in self._last_processed:
                out.append(self._last_processed[key])
            else:
                # compute on-demand (fast due to decimation), using current params
                try:
                    params = self.param_panel.get_params()
                    trial = doric.make_trial(ch, trigger_name=self._current_trigger)
                    manual = self._manual_regions_by_key.get(key, [])
                    proc = self.processor.process_trial(trial, params, manual_regions_sec=manual, preview_mode=False)
                    self._last_processed[key] = proc
                    out.append(proc)
                except Exception:
                    pass

        self.post_tab.receive_current_processed(out)

    @QtCore.Slot()
    def _post_get_current_dio_list(self):
        # DIO list for current/selected files: intersection or union; easiest = union
        paths = self._selected_paths()
        if not paths:
            paths = [self._current_path] if self._current_path else []

        dio = set()
        for p in paths:
            f = self._loaded_files.get(p)
            if f:
                dio |= set(f.digital_by_name.keys())
        self.post_tab.receive_dio_list(sorted(dio))

    @QtCore.Slot(str, str)
    def _post_get_dio_data_for_path(self, path: str, dio_name: str):
        """
        Returns (t_dio, y_dio) for the requested dio_name for a given *raw* path
        currently loaded/parsed in the cache.

        Fixes numpy array truth-value ambiguity by checking None/len explicitly.
        """
        f = self._raw_cache.get(path, None)

        if f is None:
            return None, None

        # digital_time may be a numpy array
        if getattr(f, "digital_time", None) is None:
            return None, None

        t_dio = np.asarray(f.digital_time)
        if t_dio.size == 0:
            return None, None

        digital_map = getattr(f, "digital_by_name", None)
        if not isinstance(digital_map, dict) or dio_name not in digital_map:
            return None, None

        y_dio = np.asarray(digital_map[dio_name])
        if y_dio.size == 0:
            return None, None

        # Ensure same length
        n = min(t_dio.size, y_dio.size)
        t_dio = t_dio[:n]
        y_dio = y_dio[:n]

        return t_dio, y_dio

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
