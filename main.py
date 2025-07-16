import tkinter as tk
from tkinter import ttk
import subprocess
import os
import time
import threading

class GestureControllerDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Controller Dashboard")
        self.root.geometry("600x400")
        self.process = None
        self.running = False

        # GUI Elements
        self.start_button = ttk.Button(root, text="Start Gesture Controller", command=self.start_gesture_controller)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Stop Gesture Controller", command=self.stop_gesture_controller, state='disabled')
        self.stop_button.pack(pady=10)

        self.gesture_text = tk.Text(root, height=15, width=60)
        self.gesture_text.pack(pady=10)
        self.gesture_text.insert(tk.END, "Gesture Output:\n")
        self.gesture_text.config(state='disabled')

        self.update_thread = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_gesture_controller(self):
        if not self.running:
            try:
                self.process = subprocess.Popen(['python', 'f:\\exp_handgesture\\Gesture_Controller.py'], 
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.running = True
                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                self.gesture_text.config(state='normal')
                self.gesture_text.insert(tk.END, "Gesture Controller started\n")
                self.gesture_text.config(state='disabled')
                self.update_thread = threading.Thread(target=self.update_gesture_text, daemon=True)
                self.update_thread.start()
            except Exception as e:
                self.gesture_text.config(state='normal')
                self.gesture_text.insert(tk.END, f"Error starting controller: {e}\n")
                self.gesture_text.config(state='disabled')

    def stop_gesture_controller(self):
        if self.running and self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.running = False
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.gesture_text.config(state='normal')
            self.gesture_text.insert(tk.END, "Gesture Controller stopped\n")
            self.gesture_text.config(state='disabled')

    def update_gesture_text(self):
        last_gesture = None
        while self.running:
            try:
                if os.path.exists("detected_gestures.txt"):
                    with open("detected_gestures.txt", "r") as file:
                        gesture = file.read().strip()
                        if gesture and gesture != last_gesture:
                            self.gesture_text.config(state='normal')
                            self.gesture_text.insert(tk.END, f"{gesture}\n")
                            self.gesture_text.see(tk.END)
                            self.gesture_text.config(state='disabled')
                            last_gesture = gesture
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("KeyboardInterrupt in update_gesture_text, stopping gracefully")
                self.stop_gesture_controller()
                break
            except Exception as e:
                print(f"Error in update_gesture_text: {e}")
                time.sleep(0.1)

    def on_closing(self):
        self.stop_gesture_controller()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureControllerDashboard(root)
    root.mainloop()