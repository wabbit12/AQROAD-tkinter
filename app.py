import tkinter as tk
from tkinter import ttk, font, messagebox
import cv2
from PIL import Image, ImageTk
from detector import RoadSignDetector
import threading
from queue import Queue
from tkinter.font import nametofont

class ModernButton(ttk.Button):
    def __init__(self, master=None, **kwargs):
        self.default_style = kwargs.get('style', 'Modern.TButton')
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        if 'Capture' in self.default_style:
            self['style'] = 'CaptureHover.TButton'
        elif 'Resume' in self.default_style:
            self['style'] = 'ResumeHover.TButton'
        elif 'TTS' in self.default_style:
            self['style'] = 'TTSHover.TButton'

    def on_leave(self, e):
        self['style'] = self.default_style

class RoadSignDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AQROAD: AI Road Sign Detector")
        self.window.geometry("1200x800")
        self.window.configure(bg='#000000')
        
        self.DISPLAY_WIDTH = 640

        self.setup_fonts()

        self.configure_styles()

        self.detector = RoadSignDetector()

        try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise ValueError("No camera detected")
                
                self.video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.DISPLAY_HEIGHT = int(self.DISPLAY_WIDTH * (self.video_height / self.video_width))
                
        except Exception as e:
            messagebox.showerror("Camera Error", "No camera detected or unable to access camera.")
            self.window.quit()
            return
        
        self.cap = cv2.VideoCapture(0)
        self.video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.DISPLAY_HEIGHT = int(self.DISPLAY_WIDTH * (self.video_height / self.video_width))
        
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.frozen = False
        self.frozen_frame = None
        self.frozen_detection = None

        self.current_detection = None

        self.setup_ui()

        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        self.update_frame()

        self.window.resizable(False, False)

    def setup_fonts(self):
        available_fonts = font.families()
        self.font_family = "Poppins" if "Poppins" in available_fonts else \
                          "Segoe UI" if "Segoe UI" in available_fonts else \
                          "Helvetica" if "Helvetica" in available_fonts else \
                          "TkDefaultFont"
        
        default_font = nametofont("TkDefaultFont")
        default_font.configure(family=self.font_family)

    def configure_styles(self):
        style = ttk.Style()

        style.configure("Capture.TButton",
                       padding=(20, 12),
                       background='#FF4444',
                       foreground='white',
                       font=(self.font_family, 11, 'bold'),
                       borderwidth=0,
                       relief="flat")
        
        style.configure("CaptureHover.TButton",
                       padding=(20, 12),
                       background='#FF6666',
                       foreground='white',
                       font=(self.font_family, 11, 'bold'),
                       borderwidth=0,
                       relief="flat")

        style.configure("Resume.TButton",
                       padding=(20, 12),
                       background='#44AA44',
                       foreground='white',
                       font=(self.font_family, 11, 'bold'),
                       borderwidth=0,
                       relief="flat")
        
        style.configure("ResumeHover.TButton",
                       padding=(20, 12),
                       background='#55BB55',
                       foreground='white',
                       font=(self.font_family, 11, 'bold'),
                       borderwidth=0,
                       relief="flat")

        style.configure("TTS.TButton",
                       padding=(20, 12),
                       background='#4444FF',
                       foreground='white',
                       font=(self.font_family, 11, 'bold'),
                       borderwidth=0,
                       relief="flat")
        
        style.configure("TTSHover.TButton",
                       padding=(20, 12),
                       background='#6666FF',
                       foreground='white',
                       font=(self.font_family, 11, 'bold'),
                       borderwidth=0,
                       relief="flat")
        
        style.configure("Modern.TFrame",
                       background='#000000')
        
        style.configure("Modern.TLabel",
                       background='#000000',
                       foreground='white',
                       font=(self.font_family, 12),
                       padding=(0, 5))
        
        style.configure("ModernHeader.TLabel",
                       background='#000000',
                       foreground='white',
                       font=(self.font_family, 14, 'bold'),
                       padding=(0, 10))

        style.layout('Capture.TButton', [
            ('Button.frame', {'sticky': 'nswe', 'border': '1',
                            'children': [('Button.padding', {'sticky': 'nswe',
                                                           'children': [('Button.label', {'sticky': 'nswe'})]})]})])
        
        style.layout('Resume.TButton', [
            ('Button.frame', {'sticky': 'nswe', 'border': '1',
                            'children': [('Button.padding', {'sticky': 'nswe',
                                                           'children': [('Button.label', {'sticky': 'nswe'})]})]})])
        
        style.layout('TTS.TButton', [
            ('Button.frame', {'sticky': 'nswe', 'border': '1',
                            'children': [('Button.padding', {'sticky': 'nswe',
                                                           'children': [('Button.label', {'sticky': 'nswe'})]})]})])

    def setup_ui(self):

        main_container = ttk.Frame(self.window, style="Modern.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        left_frame = ttk.Frame(main_container, style="Modern.TFrame")
        left_frame.pack(side=tk.LEFT, padx=(0, 30), anchor='n')
        
        right_frame = ttk.Frame(main_container, style="Modern.TFrame")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(
            left_frame, 
            bg='#000000', 
            highlightthickness=1,
            highlightbackground='#333333',
            width=self.DISPLAY_WIDTH,
            height=self.DISPLAY_HEIGHT
        )
        self.canvas.pack()

        header_label = ttk.Label(
            left_frame,
            text="AQROAD: AI Road Sign Detector",
            style="ModernHeader.TLabel",
            font=(self.font_family, 24, 'bold')  
        )
        header_label.pack(pady=(20, 0))  
        
        ttk.Label(right_frame, text="DETECTED SIGN", style="ModernHeader.TLabel").pack(anchor='w')
        self.sign_label = ttk.Label(right_frame, text="No signs detected", style="Modern.TLabel", wraplength=400)
        self.sign_label.pack(anchor='w')
        
        ttk.Label(right_frame, text="DESCRIPTION", style="ModernHeader.TLabel").pack(anchor='w')
        self.description_label = ttk.Label(right_frame, text="No sign detected", style="Modern.TLabel", wraplength=400)
        self.description_label.pack(anchor='w')
        
        ttk.Label(right_frame, text="CONFIDENCE", style="ModernHeader.TLabel").pack(anchor='w')
        self.confidence_label = ttk.Label(right_frame, text="N/A", style="Modern.TLabel")
        self.confidence_label.pack(anchor='w')

        buttons_frame = ttk.Frame(right_frame, style="Modern.TFrame")
        buttons_frame.pack(pady=30, anchor='w')
        
        self.capture_button = ModernButton(
            buttons_frame,
            text="⏸  Capture Frame",
            style="Capture.TButton",
            command=self.toggle_capture
        )
        self.capture_button.pack(pady=(0, 15))
        
        self.tts_button = ModernButton(
            buttons_frame,
            text="🔊  Read Sign Info",
            style="TTS.TButton",
            command=self.speak_current_detection
        )
        self.tts_button.pack()

        if not self.current_detection and not self.frozen_detection:
            self.tts_button.state(['disabled'])

    def toggle_capture(self):
        if not self.frozen:
            self.frozen = True
            self.capture_button.configure(text="▶  Resume Live", style="Resume.TButton")
            self.capture_button.default_style = "Resume.TButton"
            if not self.frame_queue.empty():
                frame, detection = self.frame_queue.get()
                self.frozen_frame = frame
                self.frozen_detection = detection
                if detection:
                    self.tts_button.state(['!disabled'])
                else:
                    self.tts_button.state(['disabled'])
        else:
            self.frozen = False
            self.capture_button.configure(text="⏸  Capture Frame", style="Capture.TButton")
            self.capture_button.default_style = "Capture.TButton"
            self.frozen_frame = None
            self.frozen_detection = None
            if self.current_detection:
                self.tts_button.state(['!disabled'])
            else:
                self.tts_button.state(['disabled'])

    def speak_current_detection(self):
        detection = self.frozen_detection if self.frozen else self.current_detection
        if detection:
            text = f"Detected {detection['name']}. {detection['description']}"
            self.detector.speak_description(text)
    
    def video_loop(self):
        while self.running:
            if not self.frozen:
                ret, frame = self.cap.read()
                if ret:
                    detections = self.detector.detect_signs(frame)
                    processed_frame = self.detector.draw_detections(frame, detections)
                    
                    if detections:
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        self.current_detection = best_detection
                        self.frame_queue.put((processed_frame, best_detection))
                    else:
                        self.current_detection = None
                        self.frame_queue.put((processed_frame, None))
    
    def update_frame(self):
        try:
            if self.frozen:
                frame = self.frozen_frame
                detection = self.frozen_detection
            else:
                frame, detection = self.frame_queue.get_nowait()
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
                frame = Image.fromarray(frame)
                self.photo = ImageTk.PhotoImage(image=frame)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            if detection:
                self.sign_label.config(text=detection['name'])
                self.description_label.config(text=detection['description'])
                self.confidence_label.config(text=f"{detection['confidence']:.2f}")
                self.tts_button.state(['!disabled'])
            else:
                self.sign_label.config(text="No signs detected")
                self.description_label.config(text="No sign detected")
                self.confidence_label.config(text="N/A")
                if not self.frozen:
                    self.tts_button.state(['disabled'])
                
        except:
            pass
        
        self.window.after(33, self.update_frame)
    
    def __del__(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

def main():
    root = tk.Tk()
    app = RoadSignDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()