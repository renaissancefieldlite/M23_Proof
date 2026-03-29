#!/usr/bin/env python3
"""
m23_monitor_qt.py - PyQt6 Monitor for M23 Search with Start/Stop Controls
Watches testjson/ in real-time, shows live progress from multiple instances.
"""

import glob
import json
import os
import signal
import subprocess
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QBrush, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

JSON_DIR = "testjson"
REFRESH_RATE = 2000
AUTO_SCRIPT = "auto_m23_forever.py"
PID_FILES = {
    "1": "m23_search_1.pid",
    "2": "m23_search_2.pid",
}
LOG_FILES = {
    "1": "m23_search_1.log",
    "2": "m23_search_2.log",
}


def read_pid(instance_id: str):
    pid_file = PID_FILES[instance_id]
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_pid(instance_id: str, pid: int):
    with open(PID_FILES[instance_id], "w", encoding="utf-8") as f:
        f.write(str(pid))


def remove_pid(instance_id: str):
    pid_file = PID_FILES[instance_id]
    if os.path.exists(pid_file):
        os.remove(pid_file)


def process_running(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_log_tail(log_file: str, lines: int = 8) -> str:
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return "Log not available"


class M23Monitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M23 Hunt Monitor - Dual Instance")
        self.setGeometry(100, 100, 1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Watching testjson/ and both instances...")

        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Both Instances")
        self.start_btn.setStyleSheet(
            "background-color: #2e2e2e; color: #00ff00; font-weight: bold; padding: 8px;"
        )
        self.start_btn.clicked.connect(self.start_both)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Both Instances")
        self.stop_btn.setStyleSheet(
            "background-color: #2e2e2e; color: #ff0000; font-weight: bold; padding: 8px;"
        )
        self.stop_btn.clicked.connect(self.stop_both)
        control_layout.addWidget(self.stop_btn)

        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.setStyleSheet(
            "background-color: #2e2e2e; color: #ffffff; padding: 8px;"
        )
        self.refresh_btn.clicked.connect(self.refresh)
        control_layout.addWidget(self.refresh_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        inst_layout = QHBoxLayout()
        self.inst1_label = QLabel("Instance 1: STOPPED")
        self.inst1_label.setStyleSheet("font-size: 10pt; color: #ff0000;")
        inst_layout.addWidget(self.inst1_label)

        self.inst2_label = QLabel("Instance 2: STOPPED")
        self.inst2_label.setStyleSheet("font-size: 10pt; color: #ff0000;")
        inst_layout.addWidget(self.inst2_label)
        layout.addLayout(inst_layout)

        stats_layout = QHBoxLayout()

        self.iter_label = QLabel("Total Iterations: --")
        self.iter_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: #00ff00;"
        )
        stats_layout.addWidget(self.iter_label)

        self.best_label = QLabel("Best: --")
        self.best_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: #ffff00;"
        )
        stats_layout.addWidget(self.best_label)

        self.files_label = QLabel("Files: --")
        self.files_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: #00ffff;"
        )
        stats_layout.addWidget(self.files_label)

        layout.addLayout(stats_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p% toward 6/9 (%v/100)")
        layout.addWidget(self.progress)

        log_split = QSplitter()
        log_split.setOrientation(Qt.Orientation.Vertical)
        layout.addWidget(log_split)

        inst1_widget = QWidget()
        inst1_layout = QVBoxLayout(inst1_widget)
        inst1_layout.addWidget(QLabel("Instance 1 Log:"))
        self.inst1_log = QTextEdit()
        self.inst1_log.setReadOnly(True)
        self.inst1_log.setFont(QFont("Courier", 9))
        self.inst1_log.setMaximumHeight(150)
        inst1_layout.addWidget(self.inst1_log)
        log_split.addWidget(inst1_widget)

        inst2_widget = QWidget()
        inst2_layout = QVBoxLayout(inst2_widget)
        inst2_layout.addWidget(QLabel("Instance 2 Log:"))
        self.inst2_log = QTextEdit()
        self.inst2_log.setReadOnly(True)
        self.inst2_log.setFont(QFont("Courier", 9))
        self.inst2_log.setMaximumHeight(150)
        inst2_layout.addWidget(self.inst2_log)
        log_split.addWidget(inst2_widget)

        split = QSplitter()
        layout.addWidget(split)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_file)
        left_layout.addWidget(QLabel("Result Files:"))
        left_layout.addWidget(self.file_list)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.candidate_table = QTableWidget()
        self.candidate_table.setColumnCount(5)
        self.candidate_table.setHorizontalHeaderLabels(["Idx", "lambda", "mu", "Tested", "Score"])
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(QLabel("Candidates in selected file:"))
        right_layout.addWidget(self.candidate_table)

        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setMaximumHeight(200)
        self.raw_output.setFont(QFont("Courier", 9))
        right_layout.addWidget(QLabel("Raw Output:"))
        right_layout.addWidget(self.raw_output)

        split.addWidget(right)
        split.setSizes([300, 900])

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_RATE)

        self.refresh()

    def start_instance(self, instance_id: str):
        pid = read_pid(instance_id)
        if process_running(pid):
            return False, f"Instance {instance_id} already running (PID {pid})"

        env = os.environ.copy()
        env["INSTANCE_ID"] = instance_id

        log_handle = open(LOG_FILES[instance_id], "ab")
        process = subprocess.Popen(
            [sys.executable, AUTO_SCRIPT],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd(),
            start_new_session=True,
        )
        log_handle.close()
        write_pid(instance_id, process.pid)
        return True, f"Instance {instance_id} started (PID {process.pid})"

    def start_both(self):
        try:
            messages = []
            for instance_id in ("1", "2"):
                _, message = self.start_instance(instance_id)
                messages.append(message)
            self.status.showMessage(" | ".join(messages), 5000)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start: {e}")

    def stop_instance(self, instance_id: str):
        pid = read_pid(instance_id)
        if not pid:
            return False, f"Instance {instance_id} not running"

        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        remove_pid(instance_id)
        return True, f"Instance {instance_id} stopped"

    def stop_both(self):
        try:
            messages = []
            for instance_id in ("1", "2"):
                _, message = self.stop_instance(instance_id)
                messages.append(message)

            subprocess.run(["pkill", "-f", AUTO_SCRIPT], check=False)
            self.status.showMessage(" | ".join(messages), 5000)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop: {e}")

    def refresh(self):
        self.update_instance_status()
        self.update_file_list()
        self.update_stats()

        files = sorted(
            glob.glob(os.path.join(JSON_DIR, "exact_test_results_*.json")),
            key=os.path.getmtime,
        )
        if files:
            latest = os.path.basename(files[-1])
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.text() == latest:
                    self.file_list.setCurrentItem(item)
                    self.load_file(item)
                    break

    def update_instance_status(self):
        for instance_id, label, log_widget in (
            ("1", self.inst1_label, self.inst1_log),
            ("2", self.inst2_label, self.inst2_log),
        ):
            pid = read_pid(instance_id)
            if process_running(pid):
                label.setText(f"Instance {instance_id}: RUNNING (PID {pid})")
                label.setStyleSheet("font-size: 10pt; color: #00ff00;")
                log_widget.setText(read_log_tail(LOG_FILES[instance_id]))
            else:
                label.setText(f"Instance {instance_id}: STOPPED")
                label.setStyleSheet("font-size: 10pt; color: #ff0000;")
                log_widget.setText(read_log_tail(LOG_FILES[instance_id], lines=5))
                remove_pid(instance_id)

    def update_file_list(self):
        current = self.file_list.currentItem()
        current_text = current.text() if current else None

        self.file_list.clear()
        files = sorted(
            glob.glob(os.path.join(JSON_DIR, "exact_test_results_*.json")),
            key=os.path.getmtime,
        )

        for file_path in files[-50:]:
            name = os.path.basename(file_path)
            item = QListWidgetItem(name)
            if name == current_text:
                item.setSelected(True)
            self.file_list.addItem(item)

    def update_stats(self):
        total_iters = 0
        log_files = ["m23_search_1.log", "m23_search_2.log", "m23_search.log"]
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    log = f.read()
                    total_iters += log.count("ITERATION")
            except Exception:
                pass
        self.iter_label.setText(f"Total Iterations: {total_iters}")

        best_score = 0.0
        files = glob.glob(os.path.join(JSON_DIR, "exact_test_results_*.json"))

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for cand in data:
                    if cand["result"].get("success", False):
                        score = cand["result"].get("consistency_score", 0.0)
                        if score > best_score:
                            best_score = score
            except Exception:
                pass

        if best_score > 0:
            self.best_label.setText(
                f"Best: {best_score * 100:.1f}% ({int(round(best_score * 9))}/9)"
            )
            self.progress.setValue(int(best_score * 100))
        else:
            self.best_label.setText("Best: --")
            self.progress.setValue(0)

        file_count = len(glob.glob(os.path.join(JSON_DIR, "exact_test_results_*.json")))
        self.files_label.setText(f"Files: {file_count}")

    def load_file(self, item):
        file_path = os.path.join(JSON_DIR, item.text())
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.candidate_table.setRowCount(len(data))

            for i, cand in enumerate(data):
                self.candidate_table.setItem(i, 0, QTableWidgetItem(str(i)))
                lam = cand["candidate"].get("lambda_expr", cand["candidate"].get("λ_expr", ""))
                self.candidate_table.setItem(i, 1, QTableWidgetItem(lam))
                mu = cand["candidate"].get("mu_expr", cand["candidate"].get("μ_expr", ""))
                self.candidate_table.setItem(i, 2, QTableWidgetItem(mu))
                tested = cand["result"].get("tested_count", 0)
                self.candidate_table.setItem(i, 3, QTableWidgetItem(str(tested)))
                score = cand["result"].get("consistency_score", 0.0)
                score_item = QTableWidgetItem(f"{score * 100:.1f}%")
                if score > 0:
                    score_item.setForeground(QBrush(QColor("#00ff00")))
                self.candidate_table.setItem(i, 4, score_item)

            if data and "output" in data[0]["result"]:
                self.raw_output.setText(data[0]["result"]["output"][-3000:])
            else:
                self.raw_output.clear()

        except Exception as e:
            self.raw_output.setText(f"Error loading file: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.black)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

    window = M23Monitor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
