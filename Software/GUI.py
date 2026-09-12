import sys
import struct
import time
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import serial
import serial.tools.list_ports
from scipy.signal import butter, filtfilt, resample, medfilt, find_peaks, get_window
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QDoubleSpinBox, QGroupBox,
                             QComboBox, QMessageBox, QFileDialog, QLineEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTabWidget)
from PyQt6.QtCore import QTimer, Qt


def interp_complex(x_new, x_old, y_complex):
    real_interp = np.interp(x_new, x_old, np.real(y_complex))
    imag_interp = np.interp(x_new, x_old, np.imag(y_complex))
    return real_interp + 1j * imag_interp


class VNAMaster(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VNA Metrology Engine: S11, TDR & Time-Domain Gating")
        self.resize(1600, 950)
        self.port = None
        self.rx_buffer = bytearray()

        self.is_sweeping = False
        self.sweep_mode = 'IDLE'
        self.sweep_idx = 0
        self.sweep_points = 1001
        self.sweep_start_time = 0.0

        self.freq_array = np.zeros(self.sweep_points)
        self.active_E_D = np.zeros(self.sweep_points, dtype=complex)
        self.active_E_S = np.zeros(self.sweep_points, dtype=complex)
        self.active_E_T = np.zeros(self.sweep_points, dtype=complex)

        self.raw_mag_data = np.full(self.sweep_points, np.nan)
        self.raw_phase_data = np.full(self.sweep_points, np.nan)
        self.plot_mag_data = np.full(self.sweep_points, np.nan)
        self.plot_phase_data = np.full(self.sweep_points, np.nan)

        self.cal_status = {'OPEN': False, 'SHORT': False, 'LOAD': False}
        self.is_calibrated = False
        self.cal_freq_array = None
        self.base_E_D = None
        self.base_E_S = None
        self.base_E_T = None

        self.markers = []
        self.tdr_markers = []
        self.tdr_d_array = np.array([])
        self.tdr_mag_array = np.array([])

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # --- ROW 1: Connection & Export ---
        top_row = QHBoxLayout()
        conn_group = QGroupBox("Hardware Connection")
        conn_layout = QHBoxLayout()
        self.combo_com = QComboBox()
        self.refresh_ports()
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setMaximumWidth(40)
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setCheckable(True)
        self.btn_connect.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(QLabel("COM Port:"))
        conn_layout.addWidget(self.combo_com)
        conn_layout.addWidget(self.btn_refresh)
        conn_layout.addWidget(self.btn_connect)
        conn_group.setLayout(conn_layout)
        top_row.addWidget(conn_group)

        export_group = QGroupBox("Export Data & Graphics")
        export_layout = QHBoxLayout()
        self.input_img_name = QLineEdit()
        self.input_img_name.setPlaceholderText("VNA_Export")

        self.btn_export_png = QPushButton("Save PNG")
        self.btn_export_png.clicked.connect(self.export_png)

        self.btn_export_s1p = QPushButton("Save CST (.s1p)")
        self.btn_export_s1p.clicked.connect(self.export_s1p)

        export_layout.addWidget(QLabel("Filename:"))
        export_layout.addWidget(self.input_img_name)
        export_layout.addWidget(self.btn_export_png)
        export_layout.addWidget(self.btn_export_s1p)
        export_group.setLayout(export_layout)
        top_row.addWidget(export_group)
        layout.addLayout(top_row)

        # --- ROW 2: Sweep & Calibration ---
        mid_row = QHBoxLayout()
        control_group = QGroupBox("Sweep Configuration")
        control_layout = QHBoxLayout()
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(50.0, 4400.0)
        self.spin_start.setValue(400.0)
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setRange(50.0, 4400.0)
        self.spin_stop.setValue(3000.0)
        self.combo_pts = QComboBox()
        self.combo_pts.addItems(["101", "201", "501", "1001"])
        self.combo_pts.setCurrentText("1001")
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Off", "3 (Light)", "5 (Medium)", "11 (Aggressive)"])

        self.spin_ed = QDoubleSpinBox()
        self.spin_ed.setRange(-5000.0, 5000.0)
        self.spin_ed.setValue(0.0)
        self.spin_ed.setSuffix(" ps")
        self.spin_ed.setToolTip("Port Extension (One-way electrical delay)")

        self.btn_normal = QPushButton("Start Sweep")
        self.btn_normal.setStyleSheet("background-color: #005577; color: white;")
        self.btn_normal.clicked.connect(lambda: self.start_sweep('NORMAL'))
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: #880000; color: white;")
        self.btn_stop.clicked.connect(self.stop_sweep)
        self.btn_toggle_adc = QPushButton("Hide ADC")
        self.btn_toggle_adc.setCheckable(True)
        self.btn_toggle_adc.clicked.connect(self.toggle_adc)

        control_layout.addWidget(QLabel("Start:"))
        control_layout.addWidget(self.spin_start)
        control_layout.addWidget(QLabel("Stop:"))
        control_layout.addWidget(self.spin_stop)
        control_layout.addWidget(QLabel("Pts:"))
        control_layout.addWidget(self.combo_pts)
        control_layout.addWidget(QLabel("Filter:"))
        control_layout.addWidget(self.combo_filter)
        control_layout.addWidget(QLabel("Delay:"))
        control_layout.addWidget(self.spin_ed)
        control_layout.addWidget(self.btn_normal)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_toggle_adc)
        control_group.setLayout(control_layout)
        mid_row.addWidget(control_group)

        cal_group = QGroupBox("OSL Memory")
        cal_layout = QHBoxLayout()
        self.btn_open = QPushButton("OPEN")
        self.btn_short = QPushButton("SHORT")
        self.btn_load = QPushButton("LOAD")
        self.btn_save_cal = QPushButton("Save Cal")
        self.btn_load_cal = QPushButton("Load Cal")
        self.btn_open.clicked.connect(lambda: self.start_sweep('OPEN'))
        self.btn_short.clicked.connect(lambda: self.start_sweep('SHORT'))
        self.btn_load.clicked.connect(lambda: self.start_sweep('LOAD'))
        self.btn_save_cal.clicked.connect(self.save_cal)
        self.btn_load_cal.clicked.connect(self.load_cal)
        cal_layout.addWidget(self.btn_open)
        cal_layout.addWidget(self.btn_short)
        cal_layout.addWidget(self.btn_load)
        cal_layout.addWidget(self.btn_save_cal)
        cal_layout.addWidget(self.btn_load_cal)
        cal_group.setLayout(cal_layout)
        mid_row.addWidget(cal_group)
        layout.addLayout(mid_row)

        self.info_label = QLabel("Connect hardware to begin.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #FF5555; background-color: #111; padding: 10px; font-size: 14pt;")
        layout.addWidget(self.info_label)

        self.tabs = QTabWidget()

        # === TAB 1: METROLOGY (S11) ===
        self.metrology_panel = QWidget()
        metrology_layout = QHBoxLayout(self.metrology_panel)
        metrology_layout.setContentsMargins(0, 0, 0, 0)

        self.metrology_graph = pg.GraphicsLayoutWidget()

        self.p_mag = self.metrology_graph.addPlot(row=0, col=0, title="S11 Magnitude (dB)")
        self.p_mag.showGrid(x=True, y=True, alpha=0.4)
        self.p_mag.setYRange(-40, 0, padding=0)
        self.p_mag.setXRange(self.spin_start.value(), self.spin_stop.value(), padding=0)
        self.c_mag = self.p_mag.plot(pen=pg.mkPen('y', width=2))

        self.p_phase = self.metrology_graph.addPlot(row=1, col=0, title="S11 Phase (Degrees)")
        self.p_phase.showGrid(x=True, y=True, alpha=0.4)
        self.p_phase.setYRange(-180, 180, padding=0)
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
        self.spin_marker.setRange(50.0, 4400.0)
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

        # === TAB 2: TIME DOMAIN REFLECTOMETRY (TDR) & CABLE LOSS ===
        self.tdr_panel = QWidget()
        tdr_layout = QVBoxLayout(self.tdr_panel)
        tdr_layout.setContentsMargins(0, 0, 0, 0)

        tdr_controls = QHBoxLayout()
        self.spin_vf = QDoubleSpinBox()
        self.spin_vf.setRange(0.1, 1.0)
        self.spin_vf.setValue(0.66)
        self.spin_vf.setSingleStep(0.01)
        self.combo_window = QComboBox()
        self.combo_window.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.btn_calc_tdr = QPushButton("Compute TDR & Extract Cable Loss")
        self.btn_calc_tdr.setStyleSheet("background-color: #550077; color: white;")
        self.btn_calc_tdr.clicked.connect(self.update_tdr)

        tdr_controls.addWidget(QLabel("Velocity Factor (VF):"))
        tdr_controls.addWidget(self.spin_vf)
        tdr_controls.addWidget(QLabel("Window:"))
        tdr_controls.addWidget(self.combo_window)
        tdr_controls.addWidget(self.btn_calc_tdr)
        tdr_controls.addStretch()
        tdr_layout.addLayout(tdr_controls)

        tdr_body = QHBoxLayout()

        tdr_left_panel = QWidget()
        tdr_left_layout = QVBoxLayout(tdr_left_panel)
        tdr_left_layout.setContentsMargins(0, 0, 0, 0)

        self.tdr_graph = pg.PlotWidget(title="Distance to Fault (Magnitude vs Meters)")
        self.tdr_graph.showGrid(x=True, y=True, alpha=0.4)
        self.tdr_graph.setLabel('bottom', "Distance", units='m')
        self.c_tdr = self.tdr_graph.plot(pen=pg.mkPen('g', width=2))
        self.tdr_line = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
        self.tdr_graph.addItem(self.tdr_line)
        self.tdr_graph.scene().sigMouseClicked.connect(self.on_tdr_mouse_click)

        self.loss_graph = pg.PlotWidget(title="Extracted Cable Loss")
        self.loss_graph.showGrid(x=True, y=True, alpha=0.4)
        self.loss_graph.setLabel('bottom', "Frequency (MHz)")
        self.loss_graph.setLabel('left', "Loss (dB/m)")
        self.c_loss_raw = self.loss_graph.plot(pen=pg.mkPen(color=(150, 150, 150), width=1, style=Qt.PenStyle.DashLine))
        self.c_loss_fit = self.loss_graph.plot(pen=pg.mkPen('y', width=2))

        tdr_left_layout.addWidget(self.tdr_graph)
        tdr_left_layout.addWidget(self.loss_graph)

        tdr_right_panel = QWidget()
        tdr_right_layout = QVBoxLayout(tdr_right_panel)
        tdr_right_layout.setContentsMargins(0, 0, 0, 0)

        tdr_right_layout.addWidget(QLabel("Auto Detected Faults:"))
        self.tdr_fault_table = QTableWidget(0, 3)
        self.tdr_fault_table.setHorizontalHeaderLabels(["ID", "Distance (m)", "Reflection Amp"])
        self.tdr_fault_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tdr_fault_table.itemDoubleClicked.connect(self.on_tdr_double_click)
        tdr_right_layout.addWidget(self.tdr_fault_table)

        tdr_right_layout.addWidget(QLabel("Manual Markers:"))
        tdr_m_ctrl = QHBoxLayout()
        self.spin_tdr_marker = QDoubleSpinBox()
        self.spin_tdr_marker.setRange(0.0, 5000.0)
        self.spin_tdr_marker.setValue(1.0)
        self.btn_add_tdr_marker = QPushButton("Add")
        self.btn_add_tdr_marker.clicked.connect(lambda: self.add_tdr_marker())
        self.btn_clear_tdr_markers = QPushButton("Clear")
        self.btn_clear_tdr_markers.clicked.connect(self.clear_tdr_markers)
        tdr_m_ctrl.addWidget(self.spin_tdr_marker)
        tdr_m_ctrl.addWidget(self.btn_add_tdr_marker)
        tdr_m_ctrl.addWidget(self.btn_clear_tdr_markers)
        tdr_right_layout.addLayout(tdr_m_ctrl)

        self.tdr_marker_table = QTableWidget(0, 3)
        self.tdr_marker_table.setHorizontalHeaderLabels(["ID", "Distance (m)", "Amplitude"])
        self.tdr_marker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tdr_right_layout.addWidget(self.tdr_marker_table)

        tdr_body.addWidget(tdr_left_panel, stretch=3)
        tdr_body.addWidget(tdr_right_panel, stretch=1)
        tdr_layout.addLayout(tdr_body)

        self.tabs.addTab(self.tdr_panel, "Time Domain (TDR) & Diagnostics")

        # === TAB 3: TIME-DOMAIN GATING (Anechoic Simulator) ===
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

        self.btn_load_gating = QPushButton("1. Load Current Sweep")
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

        # === ADC PANEL ===
        self.adc_panel = QWidget()
        adc_layout = QVBoxLayout(self.adc_panel)
        adc_layout.setContentsMargins(0, 0, 0, 0)
        self.adc_graph = pg.GraphicsLayoutWidget()
        self.p_ref = self.adc_graph.addPlot(row=0, col=0, title="ADC Ref")
        self.c_ref = self.p_ref.plot(pen='y')
        self.p_refl = self.adc_graph.addPlot(row=1, col=0, title="ADC Refl")
        self.c_refl = self.p_refl.plot(pen='c')
        adc_layout.addWidget(self.adc_graph)
        layout.addWidget(self.adc_panel, stretch=1)

        self.fs = 93750.0
        self.fc = 24000.0
        self.time_ms = np.arange(256) * (1000.0 / self.fs)
        self.target_len = 256 * 10
        self.time_interp = np.linspace(0, self.time_ms[-1], self.target_len)
        nyquist = self.fs / 2
        self.b, self.a = butter(4, 35000 / nyquist, btype='low')
        t_arr = np.arange(256) / self.fs
        self.lo_i = np.cos(2 * np.pi * self.fc * t_arr)
        self.lo_q = -np.sin(2 * np.pi * self.fc * t_arr)

        self.timer = QTimer()
        self.timer.timeout.connect(self.poll_usb)
        self.timer.start(5)

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

        text_j1 = pg.TextItem("+j1.0", color=(180, 180, 180), anchor=(0.5, 1))
        text_j1.setPos(0, 1)
        plot_widget.addItem(text_j1)
        text_mj1 = pg.TextItem("-j1.0", color=(180, 180, 180), anchor=(0.5, 0))
        text_mj1.setPos(0, -1)
        plot_widget.addItem(text_mj1)

    def on_mouse_click(self, event):
        if event.double():
            pos = event.scenePos()
            if self.p_mag.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.p_mag.vb.mapSceneToView(pos)
                self.add_marker(mousePoint.x())
            elif self.p_phase.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.p_phase.vb.mapSceneToView(pos)
                self.add_marker(mousePoint.x())

    def on_tdr_mouse_click(self, event):
        if event.double():
            pos = event.scenePos()
            if self.tdr_graph.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.tdr_graph.vb.mapSceneToView(pos)
                self.add_tdr_marker(mousePoint.x())

    def add_marker(self, freq=None):
        if freq is None: freq = self.spin_marker.value()
        freq = max(self.spin_start.value(), min(self.spin_stop.value(), freq))

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

        self.markers.append(
            {'id': m_id, 'line_m': line_m, 'line_p': line_p, 'smith_pt': smith_pt, 'smith_label': smith_label})
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
        valid_len = self.sweep_idx if self.is_sweeping else self.sweep_points
        valid_len = max(1, valid_len)
        f_arr = self.freq_array[:valid_len]
        mag_arr = self.plot_mag_data[:valid_len]
        ph_arr = self.plot_phase_data[:valid_len]

        for i, m in enumerate(self.markers):
            freq = m['line_m'].value()
            if len(f_arr) > 1 and f_arr[0] <= freq <= f_arr[-1]:
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

    def add_tdr_marker(self, dist=None):
        if dist is None: dist = self.spin_tdr_marker.value()
        dist = max(0.0, dist)

        m_id = f"T{len(self.tdr_markers) + 1}"
        line = pg.InfiniteLine(pos=dist, angle=90, movable=True, pen=pg.mkPen('c', width=2))
        pg.InfLineLabel(line, text=m_id, position=0.95, color='c')

        self.tdr_graph.addItem(line)

        def on_move(l):
            self.spin_tdr_marker.setValue(l.value())
            self.update_tdr_marker_table()

        line.sigPositionChanged.connect(on_move)

        self.tdr_markers.append({'id': m_id, 'line': line})
        self.tdr_marker_table.setRowCount(len(self.tdr_markers))
        self.update_tdr_marker_table()

    def clear_tdr_markers(self):
        for m in self.tdr_markers:
            self.tdr_graph.removeItem(m['line'])
        self.tdr_markers.clear()
        self.tdr_marker_table.setRowCount(0)

    def update_tdr_marker_table(self):
        for i, m in enumerate(self.tdr_markers):
            dist = m['line'].value()
            amp_str = "---"
            if len(self.tdr_d_array) > 1 and self.tdr_d_array[0] <= dist <= self.tdr_d_array[-1]:
                amp = np.interp(dist, self.tdr_d_array, self.tdr_mag_array)
                amp_str = f"{amp:.4f}"

            self.tdr_marker_table.setItem(i, 0, QTableWidgetItem(m['id']))
            self.tdr_marker_table.setItem(i, 1, QTableWidgetItem(f"{dist:.3f}"))
            self.tdr_marker_table.setItem(i, 2, QTableWidgetItem(amp_str))

    def export_s1p(self):
        name = self.input_img_name.text().strip()
        if not name: name = "VNA_S11_Export"
        if not name.lower().endswith(".s1p"): name += ".s1p"

        valid_len = self.sweep_idx if self.is_sweeping else self.sweep_points
        valid_len = max(1, valid_len)

        if np.isnan(self.plot_mag_data[0]):
            QMessageBox.warning(self, "Export Error", "No valid sweep data to export.")
            return

        try:
            with open(name, 'w') as f:
                f.write("! Touchstone File generated by Pico VNA Metrology Engine\n")
                f.write("! 1-Port S-Parameters (S11)\n")
                f.write("# MHz S DB R 50.0\n")
                f.write("! Freq (MHz)    Mag (dB)    Phase (deg)\n")

                f_arr = self.freq_array[:valid_len]
                mag_arr = self.plot_mag_data[:valid_len]
                ph_arr = self.plot_phase_data[:valid_len]

                for freq, mag, ph in zip(f_arr, mag_arr, ph_arr):
                    f.write(f"{freq:.6f} {mag:.4f} {ph:.4f}\n")

            QMessageBox.information(self, "Success",
                                    f"Touchstone parameters saved to {name}\nReady for CST Studio import.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def refresh_ports(self):
        self.combo_com.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.combo_com.addItems(ports)

    def toggle_connection(self):
        if self.btn_connect.isChecked():
            port_name = self.combo_com.currentText()
            try:
                self.port = serial.Serial(port_name, 115200, timeout=1)
                self.btn_connect.setText("Disconnect")
                self.info_label.setStyleSheet("color: #00FF00; background-color: #111; padding: 10px;")
                self.info_label.setText("Hardware Connected.")
            except Exception as e:
                self.btn_connect.setChecked(False)
                QMessageBox.critical(self, "Connection Error", str(e))
        else:
            if self.port and self.port.is_open:
                self.port.close()
            self.port = None
            self.btn_connect.setText("Connect")
            self.info_label.setStyleSheet("color: #FF5555; background-color: #111; padding: 10px;")
            self.info_label.setText("Hardware Disconnected.")

    def stop_sweep(self):
        self.is_sweeping = False
        self.info_label.setStyleSheet("color: #FF5555; background-color: #111; padding: 10px;")
        self.info_label.setText("Sweep Aborted by User.")

    def save_cal(self):
        if not self.is_calibrated:
            QMessageBox.warning(self, "Save Failed", "Perform a full OSL calibration before saving.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "", "NumPy Archives (*.npz)")
        if filename:
            np.savez(filename, freqs=self.cal_freq_array, ed=self.base_E_D, es=self.base_E_S, et=self.base_E_T)
            self.info_label.setText(f"Calibration saved to {filename}")

    def load_cal(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "NumPy Archives (*.npz)")
        if filename:
            data = np.load(filename)
            self.cal_freq_array = data['freqs']
            self.base_E_D = data['ed']
            self.base_E_S = data['es']
            self.base_E_T = data['et']
            self.cal_status = {'OPEN': True, 'SHORT': True, 'LOAD': True}
            self.is_calibrated = True
            self.btn_open.setStyleSheet("background-color: #228B22; color: white;")
            self.btn_short.setStyleSheet("background-color: #228B22; color: white;")
            self.btn_load.setStyleSheet("background-color: #228B22; color: white;")
            self.info_label.setStyleSheet("color: #00FF00; background-color: #111; font-weight: bold;")
            self.info_label.setText(
                f"Calibration Loaded: {self.cal_freq_array[0]:.1f} - {self.cal_freq_array[-1]:.1f} MHz")

    def export_png(self):
        name = self.input_img_name.text().strip()
        if not name: name = "VNA_S11_Export"
        if not name.lower().endswith(".png"): name += ".png"

        current_widget = self.tabs.currentWidget()
        pixmap = current_widget.grab()
        try:
            pixmap.save(name)
            QMessageBox.information(self, "Success", f"Dashboard saved as {name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def toggle_adc(self):
        hidden = self.btn_toggle_adc.isChecked()
        self.adc_panel.setVisible(not hidden)
        self.btn_toggle_adc.setText("Show ADC" if hidden else "Hide ADC")

    def start_sweep(self, mode):
        if not self.port or not self.port.is_open:
            QMessageBox.warning(self, "Hardware Error", "Connect to a COM port first.")
            return

        start_f = self.spin_start.value()
        stop_f = self.spin_stop.value()
        pts = int(self.combo_pts.currentText())

        if mode == 'NORMAL' and self.is_calibrated:
            if start_f < self.cal_freq_array[0] or stop_f > self.cal_freq_array[-1]:
                QMessageBox.critical(self, "Calibration Error",
                                     f"Sweep range ({start_f} - {stop_f} MHz) is OUTSIDE the calibrated bounds.\nPlease adjust sweep bounds or re-calibrate.")
                return
            if pts > len(self.cal_freq_array):
                QMessageBox.warning(self, "Upscaling Warning",
                                    "Sweep points exceed calibration points. Resolution will be interpolated.")

        self.sweep_mode = mode
        self.sweep_points = pts
        self.freq_array = np.linspace(start_f, stop_f, pts)
        self.sweep_idx = 0
        self.sweep_start_time = time.time()

        self.p_mag.setXRange(start_f, stop_f, padding=0)

        self.raw_mag_data = np.full(pts, np.nan)
        self.raw_phase_data = np.full(pts, np.nan)
        self.plot_mag_data = np.full(pts, np.nan)
        self.plot_phase_data = np.full(pts, np.nan)

        if mode in ['OPEN', 'SHORT', 'LOAD']:
            if not hasattr(self, f'mem_{mode}'):
                self.mem_OPEN = np.zeros(pts, dtype=complex)
                self.mem_SHORT = np.zeros(pts, dtype=complex)
                self.mem_LOAD = np.zeros(pts, dtype=complex)

        if mode == 'NORMAL' and self.is_calibrated:
            self.active_E_D = interp_complex(self.freq_array, self.cal_freq_array, self.base_E_D)
            self.active_E_S = interp_complex(self.freq_array, self.cal_freq_array, self.base_E_S)
            self.active_E_T = interp_complex(self.freq_array, self.cal_freq_array, self.base_E_T)

        self.is_sweeping = True
        self.info_label.setStyleSheet("color: #FFFF00; background-color: #111;")
        self.request_next_point()

    def load_gating_data(self):
        if self.is_sweeping:
            QMessageBox.warning(self, "Sweep In Progress", "Please wait for the frequency sweep to finish.")
            return
        if np.isnan(self.raw_mag_data[0]):
            QMessageBox.warning(self, "No Data", "No valid sweep data found. Run a sweep first.")
            return

        mag_linear = 10 ** (self.raw_mag_data / 20)
        phase_rad = np.radians(self.raw_phase_data)
        s11_complex = mag_linear * np.exp(1j * phase_rad)

        n_fft = 4096
        tdr_response = np.fft.ifft(s11_complex, n=n_fft)
        tdr_mag = np.abs(tdr_response)

        bw_hz = (self.freq_array[-1] - self.freq_array[0]) * 1e6
        df = bw_hz / (len(self.freq_array) - 1)
        dt = 1.0 / (n_fft * df)

        # Convert absolute time to nanoseconds (One-Way Delay to match physical intuition)
        self.gating_t_array = (np.arange(n_fft) * dt * 1e9) / 2.0
        self.gating_tdr_complex = tdr_response

        view_limit = 500
        self.c_gating_tdr.setData(self.gating_t_array[:view_limit], tdr_mag[:view_limit])
        self.gating_graph.setXRange(0, self.gating_t_array[view_limit], padding=0)

    def apply_gating(self):
        if not hasattr(self, 'gating_tdr_complex'):
            QMessageBox.warning(self, "No Data", "Click 'Load Current Sweep' to initialize gating data first.")
            return

        start_ns = min(self.spin_gate_start.value(), self.spin_gate_stop.value())
        stop_ns = max(self.spin_gate_start.value(), self.spin_gate_stop.value())

        idx_start = np.searchsorted(self.gating_t_array, start_ns)
        idx_stop = np.searchsorted(self.gating_t_array, stop_ns)

        gate_len = idx_stop - idx_start
        mask = np.zeros(len(self.gating_tdr_complex))

        if gate_len > 0:
            mask[idx_start:idx_stop] = get_window(('tukey', 0.2), gate_len)

        gated_tdr = self.gating_tdr_complex * mask

        gated_s11_full = np.fft.fft(gated_tdr)
        gated_s11 = gated_s11_full[:self.sweep_points]

        self.plot_mag_data = 20 * np.log10(np.abs(gated_s11) + 1e-12)
        self.plot_phase_data = np.degrees(np.angle(gated_s11))

        self.c_mag.setData(self.freq_array, self.plot_mag_data)
        self.c_phase.setData(self.freq_array, self.plot_phase_data)

        self.c_smith.setData(np.real(gated_s11), np.imag(gated_s11))
        self.update_marker_table()

        self.tabs.setCurrentIndex(0)
        self.info_label.setText(f"Time-Domain Gating Applied: [{start_ns:.2f}ns - {stop_ns:.2f}ns]")
        self.info_label.setStyleSheet("color: #00FF00; background-color: #111;")

    def restore_s11(self):
        if np.isnan(self.raw_mag_data[0]):
            return

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
        self.tabs.setCurrentIndex(0)
        self.info_label.setText("Original S11 Data Restored.")

    def update_tdr(self):
        if self.is_sweeping:
            QMessageBox.warning(self, "Sweep In Progress",
                                "Please wait for the frequency sweep to finish before computing TDR.")
            return

        if not self.is_calibrated:
            QMessageBox.critical(self, "Calibration Error",
                                 "You MUST perform an OSL calibration before calculating Distance to Fault (TDR).")
            return

        if np.isnan(self.plot_mag_data[0]):
            QMessageBox.warning(self, "No Data", "No valid sweep data found. Please run a sweep first.")
            return

        if self.freq_array[0] < self.cal_freq_array[0] or self.freq_array[-1] > self.cal_freq_array[-1]:
            QMessageBox.critical(self, "Calibration Limits Error",
                                 f"Current sweep data ({self.freq_array[0]} - {self.freq_array[-1]} MHz) is OUTSIDE the calibrated interval.\nCannot compute accurate TDR.")
            return

        mag_linear = 10 ** (self.plot_mag_data / 20)
        phase_rad = np.radians(self.plot_phase_data)
        s11_complex = mag_linear * np.exp(1j * phase_rad)

        window_name = self.combo_window.currentText()
        if window_name != "rectangular":
            win = get_window(window_name, len(s11_complex))
            s11_complex = s11_complex * win

        n_fft = 4096
        tdr_response = np.fft.ifft(s11_complex, n=n_fft)
        tdr_mag = np.abs(tdr_response)

        bw_hz = (self.freq_array[-1] - self.freq_array[0]) * 1e6
        df = bw_hz / (len(self.freq_array) - 1)
        dt = 1.0 / (n_fft * df)

        t_array = np.arange(n_fft) * dt
        c = 299792458.0
        vf = self.spin_vf.value()
        d_array = (t_array * c * vf) / 2.0

        view_limit = 500
        self.tdr_d_array = d_array[:view_limit]
        self.tdr_mag_array = tdr_mag[:view_limit]

        self.c_tdr.setData(self.tdr_d_array, self.tdr_mag_array)
        self.tdr_graph.setXRange(0, self.tdr_d_array[-1], padding=0)

        peaks, _ = find_peaks(self.tdr_mag_array, height=0.01, distance=10)
        self.tdr_fault_table.setRowCount(len(peaks))
        for i, p_idx in enumerate(peaks):
            dist = self.tdr_d_array[p_idx]
            amp = self.tdr_mag_array[p_idx]
            self.tdr_fault_table.setItem(i, 0, QTableWidgetItem(f"Fault {i + 1}"))
            self.tdr_fault_table.setItem(i, 1, QTableWidgetItem(f"{dist:.3f}"))
            self.tdr_fault_table.setItem(i, 2, QTableWidgetItem(f"{amp:.4f}"))

        self.update_tdr_marker_table()

        valid_indices = np.where(self.tdr_d_array > 0.1)[0]
        if len(valid_indices) > 0:
            peak_idx = valid_indices[np.argmax(self.tdr_mag_array[valid_indices])]
            cable_length = self.tdr_d_array[peak_idx]

            if cable_length > 0.1:
                valid_len = self.sweep_idx if self.is_sweeping else self.sweep_points
                valid_len = max(1, valid_len)

                f_arr = self.freq_array[:valid_len]
                mag_arr = self.plot_mag_data[:valid_len]

                raw_loss_per_m = np.abs(mag_arr) / (2.0 * cable_length)
                coeffs = np.polyfit(f_arr, raw_loss_per_m, 2)
                smooth_loss = np.polyval(coeffs, f_arr)

                self.c_loss_raw.setData(f_arr, raw_loss_per_m)
                self.c_loss_fit.setData(f_arr, smooth_loss)

                self.loss_graph.setTitle(f"Extracted Cable Loss (Detected Length: {cable_length:.2f} m)")
                self.loss_graph.setXRange(f_arr[0], f_arr[-1], padding=0)

    def on_tdr_double_click(self, item):
        row = item.row()
        dist_str = self.tdr_fault_table.item(row, 1).text()
        self.tdr_line.setValue(float(dist_str))

    def request_next_point(self):
        if self.sweep_idx < self.sweep_points:
            target_freq = self.freq_array[self.sweep_idx]
            freq_bytes = struct.pack('<f', target_freq)
            while b'\x03' in freq_bytes or b'\x04' in freq_bytes:
                target_freq += 0.0001
                freq_bytes = struct.pack('<f', target_freq)
            self.port.write(b'F' + freq_bytes)
        else:
            self.is_sweeping = False
            elapsed_time = time.time() - self.sweep_start_time
            if self.sweep_mode in self.cal_status:
                self.cal_status[self.sweep_mode] = True
                self.update_cal_gui()
            if not self.is_calibrated:
                self.info_label.setText(f"Sweep Complete: {self.sweep_mode} data saved in {elapsed_time:.1f}s.")

    def update_cal_gui(self):
        if self.cal_status['OPEN']: self.btn_open.setStyleSheet("background-color: #228B22; color: white;")
        if self.cal_status['SHORT']: self.btn_short.setStyleSheet("background-color: #228B22; color: white;")
        if self.cal_status['LOAD']: self.btn_load.setStyleSheet("background-color: #228B22; color: white;")

        if all(self.cal_status.values()) and not self.is_calibrated:
            self.cal_freq_array = self.freq_array.copy()
            self.base_E_D = self.mem_LOAD
            self.base_E_S = (self.mem_OPEN + self.mem_SHORT - 2 * self.mem_LOAD) / (
                    self.mem_OPEN - self.mem_SHORT + 1e-12j)
            self.base_E_T = (self.mem_OPEN - self.mem_LOAD) * (1 - self.base_E_S)
            self.is_calibrated = True
            self.info_label.setText("OSL Calibration Complete. Matrix Generated.")
            self.info_label.setStyleSheet("color: #00FF00; background-color: #111; font-weight: bold;")

    def poll_usb(self):
        if not self.port or not self.port.is_open:
            return

        if self.port.in_waiting > 0:
            self.rx_buffer.extend(self.port.read(self.port.in_waiting))

        while len(self.rx_buffer) >= 2056:
            idx = self.rx_buffer.find(b'VNA1')
            if idx == -1:
                self.rx_buffer = self.rx_buffer[-3:]
                break

            if len(self.rx_buffer) >= idx + 2056:
                packet = self.rx_buffer[idx: idx + 2056]
                self.rx_buffer = self.rx_buffer[idx + 2056:]

                raw_audio = packet[8:2056]
                raw_samples = np.frombuffer(raw_audio, dtype=np.int32)
                samples = np.left_shift(raw_samples, 1)

                ref_raw = samples[0::2] / 2147483648.0
                refl_raw = samples[1::2] / 2147483648.0

                I_ref, Q_ref = np.mean(ref_raw * self.lo_i), np.mean(ref_raw * self.lo_q)
                I_refl, Q_refl = np.mean(refl_raw * self.lo_i), np.mean(refl_raw * self.lo_q)

                V_ref = complex(I_ref, Q_ref)
                V_refl = complex(I_refl, Q_refl)
                S11_raw = V_refl / V_ref if abs(V_ref) > 0 else 0j

                if self.sweep_idx < self.sweep_points:
                    if self.sweep_mode == 'NORMAL' and self.is_calibrated:
                        ed = self.active_E_D[self.sweep_idx]
                        es = self.active_E_S[self.sweep_idx]
                        et = self.active_E_T[self.sweep_idx]
                        denom = et + es * (S11_raw - ed)
                        S11 = (S11_raw - ed) / denom if abs(denom) > 1e-12 else 0j
                    else:
                        S11 = S11_raw
                        if self.sweep_mode == 'OPEN':
                            self.mem_OPEN[self.sweep_idx] = S11
                        elif self.sweep_mode == 'SHORT':
                            self.mem_SHORT[self.sweep_idx] = S11
                        elif self.sweep_mode == 'LOAD':
                            self.mem_LOAD[self.sweep_idx] = S11

                    delay_ps = self.spin_ed.value()
                    if delay_ps != 0.0:
                        f_hz = self.freq_array[self.sweep_idx] * 1e6
                        t_sec = delay_ps * 1e-12
                        phase_shift = 4.0 * np.pi * f_hz * t_sec
                        S11 = S11 * np.exp(1j * phase_shift)

                    mag_db = 20 * np.log10(abs(S11) + 1e-12)
                    phase_deg = np.degrees(np.angle(S11))

                    self.raw_mag_data[self.sweep_idx] = mag_db
                    self.raw_phase_data[self.sweep_idx] = phase_deg

                    filter_txt = self.combo_filter.currentText()
                    valid_len = self.sweep_idx + 1

                    if filter_txt != "Off":
                        k = int(filter_txt.split(' ')[0])
                        if valid_len >= k:
                            self.plot_mag_data[:valid_len] = medfilt(self.raw_mag_data[:valid_len], kernel_size=k)
                            self.plot_phase_data[:valid_len] = medfilt(self.raw_phase_data[:valid_len], kernel_size=k)
                        else:
                            self.plot_mag_data[self.sweep_idx] = mag_db
                            self.plot_phase_data[self.sweep_idx] = phase_deg
                    else:
                        self.plot_mag_data[self.sweep_idx] = mag_db
                        self.plot_phase_data[self.sweep_idx] = phase_deg

                    self.c_mag.setData(self.freq_array, self.plot_mag_data)
                    self.c_phase.setData(self.freq_array, self.plot_phase_data)

                    mag_linear = 10 ** (self.plot_mag_data[:valid_len] / 20)
                    phase_rad = np.radians(self.plot_phase_data[:valid_len])
                    s11_complex = mag_linear * np.exp(1j * phase_rad)
                    self.c_smith.setData(np.real(s11_complex), np.imag(s11_complex))

                    self.update_marker_table()

                    pct = (self.sweep_idx / (self.sweep_points - 1)) * 100
                    elapsed = time.time() - self.sweep_start_time
                    self.info_label.setText(
                        f"[{self.sweep_mode}] {self.freq_array[self.sweep_idx]:.1f} MHz ({pct:.1f}% | {elapsed:.1f}s) | Mag: {mag_db:.2f} dB")

                    if not self.btn_toggle_adc.isChecked():
                        self.c_ref.setData(self.time_interp,
                                           resample(filtfilt(self.b, self.a, ref_raw), self.target_len))
                        self.c_refl.setData(self.time_interp,
                                            resample(filtfilt(self.b, self.a, refl_raw), self.target_len))

                    if self.is_sweeping:
                        self.sweep_idx += 1
                        self.request_next_point()
            else:
                break


app = QApplication(sys.argv)
window = VNAMaster()
window.show()
sys.exit(app.exec())