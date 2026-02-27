#!/usr/bin/env python3
"""
m23_monitor.py - GUI monitor for M23 autonomous search
Shows live progress, best candidates, and controls.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import glob
import os
import subprocess
import threading
import time
from datetime import datetime

JSON_DIR = "testjson"
PROCESS = None

class M23Monitor:
    def __init__(self, root):
        self.root = root
        self.root.title("M23 Search Monitor")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        
        # Status variables
        self.running = False
        self.process = None
        self.update_job = None
        
        self.setup_ui()
        self.refresh_data()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="M₂₃ Autonomous Search", 
                        font=("Helvetica", 16, "bold"),
                        bg='#1e1e1e', fg='#00ff00')
        title.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#1e1e1e')
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(control_frame, text="▶ Start Search", 
                                   command=self.start_search,
                                   bg='#2e2e2e', fg='#00ff00',
                                   font=("Helvetica", 10, "bold"),
                                   padx=20, pady=5)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="⏹ Stop Search", 
                                  command=self.stop_search,
                                  bg='#2e2e2e', fg='#ff0000',
                                  font=("Helvetica", 10, "bold"),
                                  padx=20, pady=5, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        self.refresh_btn = tk.Button(control_frame, text="↻ Refresh", 
                                     command=self.refresh_data,
                                     bg='#2e2e2e', fg='#ffffff',
                                     font=("Helvetica", 10),
                                     padx=20, pady=5)
        self.refresh_btn.grid(row=0, column=2, padx=5)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#2e2e2e', relief='sunken', bd=2)
        status_frame.pack(pady=10, padx=20, fill='x')
        
        self.status_label = tk.Label(status_frame, text="Status: Idle", 
                                     font=("Helvetica", 11),
                                     bg='#2e2e2e', fg='#ffff00')
        self.status_label.pack(pady=5, padx=10)
        
        self.best_label = tk.Label(status_frame, text="Best: No candidates yet", 
                                   font=("Helvetica", 11),
                                   bg='#2e2e2e', fg='#00ffff')
        self.best_label.pack(pady=5, padx=10)
        
        self.iter_label = tk.Label(status_frame, text="Iterations: 0", 
                                   font=("Helvetica", 11),
                                   bg='#2e2e2e', fg='#ffffff')
        self.iter_label.pack(pady=5, padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate', maximum=100)
        self.progress.pack(pady=10)
        self.progress['value'] = 0
        
        # Results display
        result_frame = tk.Frame(self.root, bg='#1e1e1e')
        result_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        result_label = tk.Label(result_frame, text="Recent Results:", 
                               font=("Helvetica", 12, "bold"),
                               bg='#1e1e1e', fg='#ffffff')
        result_label.pack(anchor='w')
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15,
                                                     bg='#2e2e2e', fg='#00ff00',
                                                     font=("Courier", 10))
        self.result_text.pack(fill='both', expand=True, pady=5)
        
        # Bottom status
        self.footer = tk.Label(self.root, text=f"Monitor running - JSON dir: {JSON_DIR}",
                              font=("Helvetica", 8),
                              bg='#1e1e1e', fg='#888888')
        self.footer.pack(pady=5)
        
    def start_search(self):
        """Start the auto search in background"""
        global PROCESS
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Status: RUNNING", fg='#00ff00')
        
        # Start the process
        self.process = subprocess.Popen(
            ["python3", "auto_m23_forever.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_output)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start periodic refresh
        self.schedule_refresh()
        
    def stop_search(self):
        """Stop the search"""
        global PROCESS
        if self.process:
            self.process.terminate()
            self.process = None
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: STOPPED", fg='#ff0000')
        
        if self.update_job:
            self.root.after_cancel(self.update_job)
            
    def monitor_output(self):
        """Monitor process output in real-time"""
        while self.running and self.process:
            output = self.process.stdout.readline()
            if output:
                self.root.after(0, self.append_output, output.strip())
                
    def append_output(self, text):
        """Add output to text widget"""
        self.result_text.insert(tk.END, text + '\n')
        self.result_text.see(tk.END)
        
    def schedule_refresh(self):
        """Schedule periodic refresh"""
        self.refresh_data()
        if self.running:
            self.update_job = self.root.after(5000, self.schedule_refresh)
            
    def refresh_data(self):
        """Refresh display with latest data"""
        # Get latest results file
        files = glob.glob(os.path.join(JSON_DIR, "exact_test_results_*.json"))
        if files:
            latest = max(files, key=os.path.getmtime)
            try:
                with open(latest, 'r') as f:
                    data = json.load(f)
                    
                # Find best candidate
                best_score = 0
                best_candidate = None
                for r in data:
                    if r['result'].get('success', False):
                        score = r['result'].get('consistency_score', 0)
                        if score > best_score:
                            best_score = score
                            best_candidate = r
                            
                if best_candidate:
                    count = int(round(best_score * 9))
                    self.best_label.config(
                        text=f"Best: {best_score*100:.1f}% ({count}/9 primes)"
                    )
                    self.progress['value'] = best_score * 100
                    
                    # Show candidate details
                    cand = best_candidate['candidate']
                    self.result_text.insert(tk.END, 
                        f"\n--- Best Candidate ---\n"
                        f"λ = {cand['λ_expr']}\n"
                        f"μ = {cand['μ_expr']}\n"
                        f"Score: {best_score*100:.1f}%\n"
                        f"File: {os.path.basename(latest)}\n"
                        f"{'-'*40}\n")
                    
            except Exception as e:
                pass
                
        # Count iterations from log
        try:
            with open('m23_search.log', 'r') as f:
                log = f.read()
                iter_count = log.count('ITERATION')
                self.iter_label.config(text=f"Iterations: {iter_count}")
        except:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = M23Monitor(root)
    root.mainloop()