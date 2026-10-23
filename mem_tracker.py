import os
import time
import threading
import psutil


class MemoryTracker:
    def __init__(self, interval=0.1):
        self.parent = psutil.Process(os.getpid())
        self.interval = interval
        self.peak_kb = 0.0
        self.running = False
        self.thread = None

    def measure(self):
        total_bytes = 0

        try:
            processes = [self.parent]
            processes.extend(self.parent.children(recursive=True))

            for proc in processes:
                try:
                    total_bytes += proc.memory_info().rss
                except psutil.Error:
                    continue

        except psutil.Error:
            pass

        return total_bytes / (1024)

    def monitor(self):
        while self.running:
            self.peak_kb = max(
                self.peak_kb,
                self.measure()
            )
            time.sleep(self.interval)

    def start(self):
        self.peak_kb = self.measure()
        self.running = True

        self.thread = threading.Thread(
            target=self.monitor,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

        self.peak_kb = max(
            self.peak_kb,
            self.measure()
        )

        return self.peak_kb