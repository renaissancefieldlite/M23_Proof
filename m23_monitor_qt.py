#!/usr/bin/env python3
"""
m23_monitor_qt.py - PyQt6 Monitor for M23 Search with Start/Stop Controls
Watches testjson/ in real-time and supports a configurable worker count.
"""

from dataclasses import dataclass
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
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
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
DEFAULT_WORKER_COUNT = max(
    1,
    int(os.environ.get("M23_WORKER_COUNT", os.environ.get("WORKER_COUNT", "2"))),
)
MAX_WORKER_COUNT = max(
    DEFAULT_WORKER_COUNT,
    int(os.environ.get("M23_MAX_WORKERS", "32")),
)


def pid_path(instance_id: str) -> str:
    return f"m23_search_{instance_id}.pid"


def log_path(instance_id: str) -> str:
    return f"m23_search_{instance_id}.log"


def read_pid(instance_id: str):
    try:
        with open(pid_path(instance_id), "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_pid(instance_id: str, pid: int):
    with open(pid_path(instance_id), "w", encoding="utf-8") as f:
        f.write(str(pid))


def remove_pid(instance_id: str):
    target = pid_path(instance_id)
    if os.path.exists(target):
        os.remove(target)


def process_running(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_log_tail(target: str, lines: int = 8) -> str:
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return "Log not available"


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


@dataclass
class WorkerWidgets:
    panel: QGroupBox
    status: QLabel
    log: QTextEdit


class M23Monitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M23 Hunt Monitor - N Worker")
        self.setGeometry(100, 100, 1440, 980)

        self.worker_widgets = {}
        self._last_worker_count = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Watching testjson/ and configured workers...")

        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Workers")
        self.start_btn.setStyleSheet(
            "background-color: #2e2e2e; color: #00ff00; font-weight: bold; padding: 8px;"
        )
        self.start_btn.clicked.connect(self.start_all)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Workers")
        self.stop_btn.setStyleSheet(
            "background-color: #2e2e2e; color: #ff0000; font-weight: bold; padding: 8px;"
        )
        self.stop_btn.clicked.connect(self.stop_all)
        control_layout.addWidget(self.stop_btn)

        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.setStyleSheet(
            "background-color: #2e2e2e; color: #ffffff; padding: 8px;"
        )
        self.refresh_btn.clicked.connect(self.refresh)
        control_layout.addWidget(self.refresh_btn)

        control_layout.addWidget(QLabel("Workers:"))
        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(1, MAX_WORKER_COUNT)
        self.worker_spin.setValue(DEFAULT_WORKER_COUNT)
        self.worker_spin.valueChanged.connect(self.rebuild_worker_panels)
        control_layout.addWidget(self.worker_spin)

        self.worker_count_label = QLabel("")
        self.worker_count_label.setStyleSheet("color: #aaaaaa;")
        control_layout.addWidget(self.worker_count_label)

        control_layout.addStretch()
        layout.addLayout(control_layout)

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

        worker_scroll = QScrollArea()
        worker_scroll.setWidgetResizable(True)
        worker_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.worker_container = QWidget()
        self.worker_layout = QVBoxLayout(self.worker_container)
        self.worker_layout.setContentsMargins(0, 0, 0, 0)
        self.worker_layout.setSpacing(8)
        worker_scroll.setWidget(self.worker_container)
        layout.addWidget(worker_scroll)

        split = QSplitter()
        layout.addWidget(split, 2)

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
        self.candidate_table.setHorizontalHeaderLabels(
            ["Idx", "lambda", "mu", "Tested", "Score"]
        )
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
        split.setSizes([360, 1040])

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_RATE)

        self.rebuild_worker_panels()
        self.refresh()

    def worker_ids(self):
        return [str(i) for i in range(1, self.worker_spin.value() + 1)]

    def discovered_worker_ids(self):
        ids = set()
        for target in glob.glob("m23_search_*.pid"):
            name = os.path.basename(target)
            suffix = name[len("m23_search_") : -4]
            if suffix.isdigit():
                ids.add(suffix)
        return sorted(ids, key=int)

    def rebuild_worker_panels(self):
        clear_layout(self.worker_layout)
        self.worker_widgets.clear()

        for instance_id in self.worker_ids():
            panel = QGroupBox(f"Worker {instance_id}")
            panel_layout = QVBoxLayout(panel)

            status = QLabel("STOPPED")
            status.setStyleSheet("font-size: 10pt; color: #ff0000;")
            panel_layout.addWidget(status)

            log_widget = QTextEdit()
            log_widget.setReadOnly(True)
            log_widget.setFont(QFont("Courier", 9))
            log_widget.setMaximumHeight(160)
            panel_layout.addWidget(log_widget)

            self.worker_layout.addWidget(panel)
            self.worker_widgets[instance_id] = WorkerWidgets(
                panel=panel,
                status=status,
                log=log_widget,
            )

        self.worker_layout.addStretch(1)
        self.worker_count_label.setText(
            f"Configured workers: {self.worker_spin.value()} (max {MAX_WORKER_COUNT})"
        )
        self.start_btn.setText(f"Start {self.worker_spin.value()} Workers")
        self.stop_btn.setText("Stop Workers")
        self.status.showMessage(
            f"Watching testjson/ and {self.worker_spin.value()} configured worker(s)..."
        )
        self._last_worker_count = self.worker_spin.value()

    def start_instance(self, instance_id: str):
        pid = read_pid(instance_id)
        if process_running(pid):
            return False, f"Worker {instance_id} already running (PID {pid})"

        env = os.environ.copy()
        env["INSTANCE_ID"] = instance_id
        env["WORKER_COUNT"] = str(self.worker_spin.value())

        log_handle = open(log_path(instance_id), "ab")
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
        return True, f"Worker {instance_id} started (PID {process.pid})"

    def start_all(self):
        try:
            messages = []
            for instance_id in self.worker_ids():
                _, message = self.start_instance(instance_id)
                messages.append(message)
            self.status.showMessage(" | ".join(messages), 5000)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to start: {e}")

    def stop_instance(self, instance_id: str):
        pid = read_pid(instance_id)
        if not pid:
            return False, f"Worker {instance_id} not running"

        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        remove_pid(instance_id)
        return True, f"Worker {instance_id} stopped"

    def stop_all(self):
        try:
            messages = []
            for instance_id in sorted(set(self.worker_ids()) | set(self.discovered_worker_ids()), key=int):
                _, message = self.stop_instance(instance_id)
                messages.append(message)

            subprocess.run(["pkill", "-f", AUTO_SCRIPT], check=False)
            self.status.showMessage(" | ".join(messages), 5000)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop: {e}")

    def refresh(self):
        if self._last_worker_count != self.worker_spin.value() or not self.worker_widgets:
            self.rebuild_worker_panels()
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
        for instance_id in self.worker_ids():
            widgets = self.worker_widgets.get(instance_id)
            if widgets is None:
                continue

            pid = read_pid(instance_id)
            if process_running(pid):
                widgets.status.setText(f"Worker {instance_id}: RUNNING (PID {pid})")
                widgets.status.setStyleSheet("font-size: 10pt; color: #00ff00;")
                widgets.log.setText(read_log_tail(log_path(instance_id)))
            else:
                widgets.status.setText(f"Worker {instance_id}: STOPPED")
                widgets.status.setStyleSheet("font-size: 10pt; color: #ff0000;")
                widgets.log.setText(read_log_tail(log_path(instance_id), lines=5))
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
        for log_file in glob.glob("m23_search_*.log"):
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    total_iters += f.read().count("ITERATION")
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
