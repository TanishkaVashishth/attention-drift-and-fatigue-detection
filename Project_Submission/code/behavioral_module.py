import pygetwindow as gw
import time
import psutil

class BehavioralModule:
    def __init__(self):
        self.tab_switch_count = 0
        self.last_active_window = None
        self.out_of_focus_time = 0
        self.start_time = time.time()
        self.last_check_time = time.time()
        
        # We can define a "target" application name if needed
        self.app_window_keywords = ["AI PROJECT 2", "Python", "Code", "Chrome"] # Example keywords

    def update(self):
        try:
            current_window = gw.getActiveWindow()
            if current_window is None:
                return self.tab_switch_count, self.out_of_focus_time

            current_title = current_window.title
            now = time.time()
            dt = now - self.last_check_time
            self.last_check_time = now

            if self.last_active_window is None:
                self.last_active_window = current_title
            
            # Detect switch
            if current_title != self.last_active_window:
                self.tab_switch_count += 1
                self.last_active_window = current_title
            
            # Check if out of focus (not the main app or related work apps)
            # For simplicity, if it's NOT our dashboard or a code editor, count it as out of focus
            is_focus = any(kw.lower() in current_title.lower() for kw in self.app_window_keywords)
            if not is_focus:
                self.out_of_focus_time += dt

            return self.tab_switch_count, self.out_of_focus_time
        except Exception as e:
            # Some windows might not allow title access
            return self.tab_switch_count, self.out_of_focus_time

    def reset_stats(self):
        self.tab_switch_count = 0
        self.out_of_focus_time = 0
        self.start_time = time.time()
