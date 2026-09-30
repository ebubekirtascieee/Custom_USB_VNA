import sys
import numpy as np
import pyqtgraph as pg
from scipy.signal import medfilt, get_window
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QDoubleSpinBox, QGroupBox,
                             QComboBox, QMessageBox, QFileDialog, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTabWidget)
from PyQt6.QtCore import Qt


class VNAViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VNA Offline Analyzer: S11 & Time-Domain Gating")
        self.resize(1600, 950)

        self.sweep_points = 0
        self.freq_array = np.array([])
        
        self.raw_mag_data = np.array([])
        self.raw_phase_data = np.array([])
        self.plot_mag_data = np.array([])
        self.plot_phase_data = np.array([])

        self.markers = []

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # --- ROW 1: File Operations ---
        top_row = QHBoxLayout()
        file_group = QGroupBox("File Management")
        file_layout = QHBoxLayout()
        
        self.btn_load_s1p = QPushButton("Load .s1p File")
        self.btn_load_s1p.setStyleSheet("background-color: #005577; color: white; font-weight: bold;")
        self.btn_load_s1p.clicked.connect(self.load_s1p)
        
        self.file_label = QLabel("No file loaded.")
        self.file_label.setStyleSheet("color: #AAA;")
        
        file_layout.addWidget(self.btn_load_s1p)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        file_group.setLayout(file_layout)
        top_row.addWidget(file_group)

        export_group = QGroupBox("Export Modified Data & Graphics")
        export_layout = QHBoxLayout()
        self.input_img_name = QLineEdit()
        self.input_img_name.setPlaceholderText("Modified_Export")

        self.btn_export_png = QPushButton("Save PNG")
        self.btn_export_png.clicked.connect(self.export_png)

        self.btn_export_s1p = QPushButton("Save Modified (.s1p)")
        self.btn_export_s1p.clicked.connect(self.export_s1p)

        export_layout.addWidget(QLabel("Filename:"))
        export_layout.addWidget(self.input_img_name)
        export_layout.addWidget(self.btn_export_png)
        export_layout.addWidget(self.btn_export_s1p)
        export_group.setLayout(export_layout)
        top_row.addWidget(export_group)
        layout.addLayout(top_row)

        # --- ROW 2: Filter Settings (Display Ranges Removed) ---
        mid_row = QHBoxLayout()
        display_group = QGroupBox("Filter Settings")
        display_layout = QHBoxLayout()
        
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Off", "3 (Light)", "5 (Medium)", "11 (Aggressive)"])
        self.combo_filter.currentIndexChanged.connect(self.restore_s11) # Auto apply filter

        display_layout.addWidget(QLabel("Median Filter:"))
        display_layout.addWidget(self.combo_filter)
        display_layout.addStretch()
        
        display_group.setLayout(display_layout)
        mid_row.addWidget(display_group)
        layout.addLayout(mid_row)

        self.info_label = QLabel("Load a .s1p Touchstone file to begin analysis.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #FF5555; background-color: #111; padding: 10px; font-size: 12pt;")
        layout.addWidget(self.info_label)

        self.tabs = QTabWidget()

        # === TAB 1: METROLOGY (S11) ===
        self.metrology_panel = QWidget()
        metrology_layout = QHBoxLayout(self.metrology_panel)
        metrology_layout.setContentsMargins(0, 0, 0, 0)

        self.metrology_graph = pg.GraphicsLayoutWidget()

        self.p_mag = self.metrology_graph.addPlot(row=0, col=0, title="S11 Magnitude (dB)")
        self.p_mag.showGrid(x=True, y=True, alpha=0.4)
        self.c_mag = self.p_mag.plot(pen=pg.mkPen('y', width=2))

        self.p_phase = self.metrology_graph.addPlot(row=1, col=0, title="S11 Phase (Degrees)")
        self.p_phase.showGrid(x=True, y=True, alpha=0.4)
        self.c_phase = self.p_phase.plot(pen=pg.mkPen('c', width=2))
        self.p_phase.setXLink(self.p_mag)

        self.p_smith = self.metrology_graph.addPlot(row=0, col=1, rowspan=2, title="Smith Chart")
        self.draw_smith_grid(self.p_smith)
        self.c_smith = self.p_smith.plot(pen=pg.mkPen('m', width=2))

        self.metrology_graph.scene().sigMouseClicked.connect(self.on_mouse_click)
        metrology_layout.addWidget(self.metrology_graph, stretch=3)

        marker_panel = QWidget()
        marker_layout = QVBoxLayout(marker_panel)
        marker_layout.setContentsMargins(0, 0, 0, 0)
        marker_control_layout = QHBoxLayout()
        self.spin_marker = QDoubleSpinBox()
        self.spin_marker.setRange(0.0, 100000.0)
        self.spin_marker.setValue(1000.0)
        self.btn_add_marker = QPushButton("Add")
        self.btn_add_marker.clicked.connect(lambda: self.add_marker())
        self.btn_clear_markers = QPushButton("Clear")
        self.btn_clear_markers.clicked.connect(self.clear_markers)
        marker_control_layout.addWidget(self.spin_marker)
        marker_control_layout.addWidget(self.btn_add_marker)
        marker_control_layout.addWidget(self.btn_clear_markers)
        marker_layout.addLayout(marker_control_layout)

        self.marker_table = QTableWidget(0, 5)
        self.marker_table.setHorizontalHeaderLabels(["ID", "Freq (MHz)", "Mag (dB)", "Phase (°)", "Z (Norm)"])
        self.marker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        marker_layout.addWidget(self.marker_table)
        metrology_layout.addWidget(marker_panel, stretch=1)

        self.tabs.addTab(self.metrology_panel, "Frequency Domain (S11)")

        # === TAB 2: TIME-DOMAIN GATING ===
        self.gating_panel = QWidget()
        gating_layout = QHBoxLayout(self.gating_panel)
        gating_layout.setContentsMargins(0, 0, 0, 0)

        self.gating_graph = pg.PlotWidget(title="Time-Domain Gating (Absolute Time Limits)")
        self.gating_graph.showGrid(x=True, y=True, alpha=0.4)
        self.gating_graph.setLabel('bottom', "One-Way Time (ns)")
        self.c_gating_tdr = self.gating_graph.plot(pen=pg.mkPen('m', width=2))

        self.gate_line_start = pg.InfiniteLine(pos=1.0, angle=90, movable=True,
                                               pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))
        self.gate_line_stop = pg.InfiniteLine(pos=5.0, angle=90, movable=True,
                                              pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine))
        self.gating_graph.addItem(self.gate_line_start)
        self.gating_graph.addItem(self.gate_line_stop)

        gating_right_panel = QWidget()
        gating_right_layout = QVBoxLayout(gating_right_panel)

        self.btn_load_gating = QPushButton("1. Initialize Gating Arrays")
        self.btn_load_gating.clicked.connect(self.load_gating_data)

        gating_right_layout.addWidget(QLabel("Gate Start (ns):"))
        self.spin_gate_start = QDoubleSpinBox()
        self.spin_gate_start.setRange(0.0, 5000.0)
        self.spin_gate_start.setValue(1.0)
        gating_right_layout.addWidget(self.spin_gate_start)

        gating_right_layout.addWidget(QLabel("Gate Stop (ns):"))
        self.spin_gate_stop = QDoubleSpinBox()
        self.spin_gate_stop.setRange(0.0, 5000.0)
        self.spin_gate_stop.setValue(5.0)
        gating_right_layout.addWidget(self.spin_gate_stop)

        self.btn_apply_gate = QPushButton("2. Apply Gate & View S11")
        self.btn_apply_gate.setStyleSheet("background-color: #008800; color: white;")
        self.btn_apply_gate.clicked.connect(self.apply_gating)

        self.btn_restore_s11 = QPushButton("Restore Original S11")
        self.btn_restore_s11.setStyleSheet("background-color: #880000; color: white;")
        self.btn_restore_s11.clicked.connect(self.restore_s11)

        gating_right_layout.addWidget(self.btn_load_gating)
        gating_right_layout.addSpacing(20)
        gating_right_layout.addWidget(self.btn_apply_gate)
        gating_right_layout.addWidget(self.btn_restore_s11)
        gating_right_layout.addStretch()

        gating_layout.addWidget(self.gating_graph, stretch=3)
        gating_layout.addWidget(gating_right_panel, stretch=1)

        self.tabs.addTab(self.gating_panel, "Time-Domain Gating")

        def sync_gate_start(): self.spin_gate_start.setValue(self.gate_line_start.value())
        def sync_gate_stop(): self.spin_gate_stop.setValue(self.gate_line_stop.value())
        def sync_spin_start(): self.gate_line_start.setValue(self.spin_gate_start.value())
        def sync_spin_stop(): self.gate_line_stop.setValue(self.spin_gate_stop.value())

        self.gate_line_start.sigPositionChanged.connect(sync_gate_start)
        self.gate_line_stop.sigPositionChanged.connect(sync_gate_stop)
        self.spin_gate_start.valueChanged.connect(sync_spin_start)
        self.spin_gate_stop.valueChanged.connect(sync_spin_stop)

        layout.addWidget(self.tabs, stretch=2)

    def draw_smith_grid(self, plot_widget):
        plot_widget.setAspectLocked(True)
        plot_widget.setRange(xRange=[-1.15, 1.15], yRange=[-1.15, 1.15], padding=0)
        plot_widget.hideAxis('bottom')
        plot_widget.hideAxis('left')

        pen_main = pg.mkPen(color=(100, 100, 100), width=1.5)
        pen_sub = pg.mkPen(color=(60, 60, 60), width=1, style=Qt.PenStyle.DashLine)

        theta = np.linspace(0, 2 * np.pi, 200)
        for r in [0, 0.2, 0.5, 1, 2, 5]:
            center_x = r / (r + 1)
            radius = 1 / (r + 1)
            x = center_x + radius * np.cos(theta)
            y = radius * np.sin(theta)
            plot_widget.plot(x, y, pen=pen_main if r in [0, 1] else pen_sub)

            if r in [0, 0.5, 1.0, 2.0]:
                text = pg.TextItem(f"{r}", color=(180, 180, 180), anchor=(0.5, 1))
                text.setPos(center_x - radius, 0)
                plot_widget.addItem(text)

        for x in [0.2, 0.5, 1, 2, 5]:
            for sign in [1, -1]:
                z_line = np.linspace(0, 50, 200) + 1j * (x * sign)
                gamma = (z_line - 1) / (z_line + 1)
                plot_widget.plot(np.real(gamma), np.imag(gamma), pen=pen_sub)

        plot_widget.plot([-1, 1], [0, 0], pen=pen_main)

    def load_s1p(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Touchstone File", "", "Touchstone Files (*.s1p);;All Files (*)")
        if not filename: return

        freqs, mags, phases = [], [], []
        unit_mult = 1.0  # internal unit is MHz
        format_type = 'DB'

        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!'):
                        continue
                    if line.startswith('#'):
                        parts = line.upper().split()
                        if 'HZ' in parts: unit_mult = 1e-6
                        elif 'KHZ' in parts: unit_mult = 1e-3
                        elif 'MHZ' in parts: unit_mult = 1.0
                        elif 'GHZ' in parts: unit_mult = 1000.0
                        
                        if 'DB' in parts: format_type = 'DB'
                        elif 'MA' in parts: format_type = 'MA'
                        elif 'RI' in parts: format_type = 'RI'
                        continue
                        
                    parts = line.split()
                    if len(parts) >= 3:
                        f_val = float(parts[0]) * unit_mult
                        v1, v2 = float(parts[1]), float(parts[2])
                        freqs.append(f_val)
                        if format_type == 'DB':
                            mags.append(v1)
                            phases.append(v2)
                        elif format_type == 'MA':
                            mags.append(20 * np.log10(v1 + 1e-12))
                            phases.append(v2)
                        elif format_type == 'RI':
                            c = complex(v1, v2)
                            mags.append(20 * np.log10(abs(c) + 1e-12))
                            phases.append(np.degrees(np.angle(c)))

            if len(freqs) == 0:
                raise ValueError("No valid data points found in file.")

            self.freq_array = np.array(freqs)
            self.raw_mag_data = np.array(mags)
            self.raw_phase_data = np.array(phases)
            self.sweep_points = len(self.freq_array)

            self.file_label.setText(f"Loaded: {filename.split('/')[-1]} ({self.sweep_points} pts)")
            self.info_label.setStyleSheet("color: #00FF00; background-color: #111; padding: 10px;")
            self.info_label.setText("Data Loaded Successfully.")
            
            # Initial auto-range framing
            self.p_mag.setXRange(self.freq_array[0], self.freq_array[-1], padding=0)
            self.p_mag.setYRange(-40, 0, padding=0)
            
            # Reset and apply base plots/filters
            self.restore_s11()

        except Exception as e:
            QMessageBox.critical(self, "File Load Error", str(e))

    def on_mouse_click(self, event):
        if event.double():
            pos = event.scenePos()
            if self.p_mag.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.p_mag.vb.mapSceneToView(pos)
                self.add_marker(mousePoint.x())
            elif self.p_phase.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.p_phase.vb.mapSceneToView(pos)
                self.add_marker(mousePoint.x())

    def add_marker(self, freq=None):
        if len(self.freq_array) == 0: return
        if freq is None: freq = self.spin_marker.value()
        freq = max(self.freq_array[0], min(self.freq_array[-1], freq))

        m_id = f"M{len(self.markers) + 1}"
        line_m = pg.InfiniteLine(pos=freq, angle=90, movable=True, pen=pg.mkPen('r', width=2))
        line_p = pg.InfiniteLine(pos=freq, angle=90, movable=True, pen=pg.mkPen('r', width=2))
        pg.InfLineLabel(line_m, text=m_id, position=0.95, color='r')

        smith_pt = pg.ScatterPlotItem(size=12, pen=pg.mkPen('k'), brush=pg.mkBrush('r'))
        smith_label = pg.TextItem(text=m_id, color='r', anchor=(0, 1))

        self.p_mag.addItem(line_m)
        self.p_phase.addItem(line_p)
        self.p_smith.addItem(smith_pt)
        self.p_smith.addItem(smith_label)

        def on_move(line):
            val = line.value()
            line_m.setValue(val)
            line_p.setValue(val)
            self.update_marker_table()

        line_m.sigPositionChanged.connect(on_move)
        line_p.sigPositionChanged.connect(on_move)

        self.markers.append({'id': m_id, 'line_m': line_m, 'line_p': line_p, 'smith_pt': smith_pt, 'smith_label': smith_label})
        self.marker_table.setRowCount(len(self.markers))
        self.update_marker_table()

    def clear_markers(self):
        for m in self.markers:
            self.p_mag.removeItem(m['line_m'])
            self.p_phase.removeItem(m['line_p'])
            self.p_smith.removeItem(m['smith_pt'])
            self.p_smith.removeItem(m['smith_label'])
        self.markers.clear()
        self.marker_table.setRowCount(0)

    def update_marker_table(self):
        if len(self.freq_array) == 0: return
        f_arr = self.freq_array
        mag_arr = self.plot_mag_data
        ph_arr = self.plot_phase_data

        for i, m in enumerate(self.markers):
            freq = m['line_m'].value()
            if f_arr[0] <= freq <= f_arr[-1]:
                mag = np.interp(freq, f_arr, mag_arr)
                ph = np.interp(freq, f_arr, ph_arr)
                mag_str, ph_str = f"{mag:.2f}", f"{ph:.2f}"

                mag_linear = 10 ** (mag / 20)
                phase_rad = np.radians(ph)
                gamma = mag_linear * np.exp(1j * phase_rad)

                if abs(gamma - 1.0) < 1e-6:
                    z_str = "High Z"
                else:
                    z_norm = (1 + gamma) / (1 - gamma)
                    sign = "+" if np.imag(z_norm) >= 0 else "-"
                    z_str = f"{np.real(z_norm):.2f} {sign} j{abs(np.imag(z_norm)):.2f}"

                m['smith_pt'].setData([np.real(gamma)], [np.imag(gamma)])
                m['smith_label'].setPos(np.real(gamma), np.imag(gamma))
            else:
                mag_str, ph_str, z_str = "---", "---", "---"
                m['smith_pt'].setData([], [])
                m['smith_label'].setPos(-2, -2)

            self.marker_table.setItem(i, 0, QTableWidgetItem(m['id']))
            self.marker_table.setItem(i, 1, QTableWidgetItem(f"{freq:.1f}"))
            self.marker_table.setItem(i, 2, QTableWidgetItem(mag_str))
            self.marker_table.setItem(i, 3, QTableWidgetItem(ph_str))
            self.marker_table.setItem(i, 4, QTableWidgetItem(z_str))

    def export_s1p(self):
        if len(self.freq_array) == 0:
            QMessageBox.warning(self, "Export Error", "No valid data to export. Load a file first.")
            return

        name = self.input_img_name.text().strip()
        if not name: name = "VNA_Modified_Export"
        if not name.lower().endswith(".s1p"): name += ".s1p"

        try:
            with open(name, 'w') as f:
                f.write("! Touchstone File generated by Pico VNA Offline Analyzer\n")
                f.write("! 1-Port S-Parameters (S11) - Modified/Filtered/Gated\n")
                f.write("# MHz S DB R 50.0\n")
                f.write("! Freq (MHz)    Mag (dB)    Phase (deg)\n")

                for freq, mag, ph in zip(self.freq_array, self.plot_mag_data, self.plot_phase_data):
                    f.write(f"{freq:.6f} {mag:.4f} {ph:.4f}\n")

            QMessageBox.information(self, "Success", f"Modified parameters saved to {name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_png(self):
        name = self.input_img_name.text().strip()
        if not name: name = "VNA_Export"
        if not name.lower().endswith(".png"): name += ".png"
        current_widget = self.tabs.currentWidget()
        pixmap = current_widget.grab()
        try:
            pixmap.save(name)
            QMessageBox.information(self, "Success", f"Dashboard saved as {name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def restore_s11(self):
        if len(self.freq_array) == 0: return

        filter_txt = self.combo_filter.currentText()
        if filter_txt != "Off":
            k = int(filter_txt.split(' ')[0])
            self.plot_mag_data = medfilt(self.raw_mag_data, kernel_size=k)
            self.plot_phase_data = medfilt(self.raw_phase_data, kernel_size=k)
        else:
            self.plot_mag_data = self.raw_mag_data.copy()
            self.plot_phase_data = self.raw_phase_data.copy()

        self.c_mag.setData(self.freq_array, self.plot_mag_data)
        self.c_phase.setData(self.freq_array, self.plot_phase_data)

        mag_linear = 10 ** (self.plot_mag_data / 20)
        phase_rad = np.radians(self.plot_phase_data)
        s11_complex = mag_linear * np.exp(1j * phase_rad)
        self.c_smith.setData(np.real(s11_complex), np.imag(s11_complex))

        self.update_marker_table()
        
        # Only show "Restored" text if gating was previously applied
        if "Gating" in self.info_label.text():
            self.info_label.setText("Original S11 Data Restored (Filters Applied).")
            self.info_label.setStyleSheet("color: #00FF00; background-color: #111; padding: 10px;")

    def load_gating_data(self):
        if len(self.freq_array) == 0:
            QMessageBox.warning(self, "No Data", "Load a .s1p file first (or run a sweep).")
            return

        mag_linear = 10 ** (self.raw_mag_data / 20)
        phase_rad = np.radians(self.raw_phase_data)
        s11_complex = mag_linear * np.exp(1j * phase_rad)

        f_hz = self.freq_array * 1e6
        df_hz = f_hz[1] - f_hz[0]
        
        self.start_idx = int(round(f_hz[0] / df_hz))
        
        min_n_fft = self.start_idx + self.sweep_points
        self.n_fft = 2 ** int(np.ceil(np.log2(min_n_fft * 1.2))) # Next power of 2
        if self.n_fft < 4096: self.n_fft = 4096

        padded_s11 = np.zeros(self.n_fft, dtype=complex)
        padded_s11[self.start_idx : self.start_idx + self.sweep_points] = s11_complex

        tdr_response = np.fft.ifft(padded_s11)
        tdr_mag = np.abs(tdr_response)

        dt = 1.0 / (self.n_fft * df_hz)
        self.gating_t_array = (np.arange(self.n_fft) * dt * 1e9) / 2.0
        self.gating_tdr_complex = tdr_response

        # Plot limits (Auto scale view to 100ns or max 500 points)
        view_limit = np.searchsorted(self.gating_t_array, 100.0)
        view_limit = max(500, min(view_limit, len(self.gating_t_array)-1))
        
        self.c_gating_tdr.setData(self.gating_t_array[:view_limit], tdr_mag[:view_limit])
        self.gating_graph.setXRange(0, self.gating_t_array[view_limit], padding=0)
        
        self.info_label.setText("Gating Arrays Initialized. (True RF Bandpass Mapping Applied)")
        self.info_label.setStyleSheet("color: #00FF00; background-color: #111; padding: 10px;")

    def apply_gating(self):
        if not hasattr(self, 'gating_tdr_complex'):
            QMessageBox.warning(self, "No Data", "Click 'Initialize Gating Arrays' first.")
            return

        start_ns = min(self.spin_gate_start.value(), self.spin_gate_stop.value())
        stop_ns = max(self.spin_gate_start.value(), self.spin_gate_stop.value())

        idx_start = np.searchsorted(self.gating_t_array, start_ns)
        idx_stop = np.searchsorted(self.gating_t_array, stop_ns)

        gate_len = idx_stop - idx_start
        mask = np.zeros(len(self.gating_tdr_complex))

        if gate_len > 0:
            window = get_window(('tukey', 0.2), gate_len)
            
            if idx_start == 0:
                mid_point = gate_len // 2
                window[:mid_point] = 1.0
                
            mask[idx_start:idx_stop] = window

        # Gate the time domain
        gated_tdr = self.gating_tdr_complex * mask
        
        # FFT back to frequency domain
        gated_s11_full = np.fft.fft(gated_tdr)
        
        gated_s11 = gated_s11_full[self.start_idx : self.start_idx + self.sweep_points]

        self.plot_mag_data = 20 * np.log10(np.abs(gated_s11) + 1e-12)
        self.plot_phase_data = np.degrees(np.angle(gated_s11))

        self.c_mag.setData(self.freq_array, self.plot_mag_data)
        self.c_phase.setData(self.freq_array, self.plot_phase_data)

        mag_linear = 10 ** (self.plot_mag_data / 20)
        phase_rad = np.radians(self.plot_phase_data)
        s11_complex = mag_linear * np.exp(1j * phase_rad)
        self.c_smith.setData(np.real(s11_complex), np.imag(s11_complex))
        
        self.update_marker_table()

        self.tabs.setCurrentIndex(0)
        self.info_label.setText(f"Time-Domain Gating Applied: [{start_ns:.2f}ns - {stop_ns:.2f}ns]")
        self.info_label.setStyleSheet("color: #FFFF00; background-color: #111; padding: 10px;")

app = QApplication(sys.argv)
window = VNAViewer()
window.show()
sys.exit(app.exec())