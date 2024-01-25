import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Canvas, ttk, Label, Toplevel
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torchvision
import threading
import time  

# Mediapipe for hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandTrackingApp:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("American Sign Language Detection")
        self.root.geometry("1200x650")

        # Define and configure styles for UI elements
        style = ttk.Style()
        style.configure("Futuristic.TButton", foreground="blue", background="#00aaff", font=("Helvetica", 12))
        style.configure("Futuristic.TLabel",
                        foreground="#00aaff",
                        font=("Segoe UI", 12, "italic"),
                        padding=(5, 5),
                        anchor="w",
                        shadowcolor="black",
                        shadowoffset=(2, 2))

        # Top frame containing buttons for various actions
        self.frame_top = tk.Frame(root)
        self.frame_top.pack(side=tk.TOP, pady=10)

        # Buttons for starting detection, displaying ASL guide, showing credits, and quitting
        self.start_detection_button = ttk.Button(self.frame_top, text="Start Detection", style="Futuristic.TButton", command=self.start_detection)
        self.start_detection_button.pack(side=tk.LEFT, padx=0.02*self.root.winfo_screenwidth())

        self.asl_reference_button = ttk.Button(self.frame_top, text="ASL Guide", style="Futuristic.TButton", command=self.show_asl_reference)
        self.asl_reference_button.pack(side=tk.LEFT, padx=0.02*self.root.winfo_screenwidth())

        self.credits_button = ttk.Button(self.frame_top, text="Credits", style="Futuristic.TButton", command=self.show_credits)
        self.credits_button.pack(side=tk.LEFT, padx=0.02*self.root.winfo_screenwidth())

        self.quit_button = ttk.Button(self.frame_top, text="Quit", style="Futuristic.TButton", command=self.root.destroy)
        self.quit_button.pack(side=tk.LEFT, padx=0.02*self.root.winfo_screenwidth())

        # Main frame for video display and detected text
        self.frame_main = tk.Frame(root)
        self.frame_main.pack(pady=10)

        # Left canvas for displaying video feed
        self.video_canvas_left = Canvas(self.frame_main, width=0.53*self.root.winfo_screenwidth(), height=0.73*self.root.winfo_screenheight(), bd=2, relief=tk.GROOVE)
        self.video_canvas_left.pack(side=tk.LEFT, padx=(0.02*self.root.winfo_screenwidth(), 10))
        self.video_canvas_width = int(0.53 * self.root.winfo_screenwidth())
        self.video_canvas_height = int(0.73 * self.root.winfo_screenheight())
        self.video_canvas_left.config(width=self.video_canvas_width, height=self.video_canvas_height)

        # Label for displaying detected text
        self.detected_label = ttk.Label(self.root, text="Text Detected: ", style="Futuristic.TLabel")
        self.detected_label.config(font=("Segoe UI", 24))
        self.detected_label.pack(side=tk.BOTTOM, pady=(0, 20))

        # Video capture and hand tracking initialization
        self.cap = None
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1,
        )

        # Delay for prediction and flags for detection state
        self.predict_delay = 1
        self.is_detection_started = False

        # Threading for video update and prediction
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

        self.predict_thread = threading.Thread(target=self.predict_thread_function)
        self.predict_thread.daemon = True
        self.predict_thread.start()

        # ASL guide image
        self.asl_poster_image = Image.open("./asl_alphabets.png")
        self.asl_poster_image = self.asl_poster_image.resize((int(0.53 * self.root.winfo_screenwidth()), int(0.73 * self.root.winfo_screenheight())))

    def update(self):
        # Continuously update video frames and hand tracking
        while True:
            if self.is_detection_started and self.cap is not None:
                ret, frame = self.cap.read()

                if frame is not None:
                    frame = cv2.flip(frame, 1)

                    # Process hand landmarks using Mediapipe
                    results = self.hands.process(frame)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]
                            x, y, w, h = cv2.boundingRect(np.array(landmarks))

                            if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
                                padding = int(0.02 * self.root.winfo_screenwidth())
                                bbox = (max(0, x - padding), max(0, y - padding), min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding))

                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                    # Display the video frame
                    self.display_video_left(frame)

            # Delay to avoid excessive resource usage
            time.sleep(0.03)

    def display_video_left(self, frame):
        # Display the video frame on the left canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (self.video_canvas_width, self.video_canvas_height))
        img = Image.fromarray(resized_frame)
        img = ImageTk.PhotoImage(image=img)

        self.video_canvas_left.create_image(0, 0, anchor=tk.NW, image=img)
        self.video_canvas_left.image = img

    def start_detection(self):
        # Start video capture and hand detection
        if not self.is_detection_started:
            self.cap = cv2.VideoCapture(0)
            self.is_detection_started = True

    def predict_thread_function(self):
        # Continuously predict letters from hand gestures
        while True:
            if self.is_detection_started and self.cap is not None:
                ret, frame = self.cap.read()

                if frame is not None:
                    frame = cv2.flip(frame, 1)

                    # Process hand landmarks using Mediapipe
                    results = self.hands.process(frame)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in hand_landmarks.landmark]
                            x, y, w, h = cv2.boundingRect(np.array(landmarks))

                            if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w < frame.shape[1] and y + h < frame.shape[0]:
                                padding = int(0.02 * self.root.winfo_screenwidth())
                                bbox = (max(0, x - padding), max(0, y - padding), min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding))

                                hand_crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                                # Predict letter from the cropped hand image
                                letter, probab = self.predict_letter(hand_crop)
                                if probab > 0.3:
                                    existing_text = self.detected_label.cget("text")
                                    if letter != "nothing":
                                        existing_text += letter
                                        self.detected_label.config(text=existing_text)

                    # Display the video frame
                    self.display_video_left(frame)

            # Delay for prediction
            time.sleep(self.predict_delay)

    def predict_letter(self, image):
        # Predict the letter from the hand gesture using a pre-trained model
        device = None
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        pretrained_vit_transforms = pretrained_vit_weights.transforms()

        model = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

        for parameter in model.parameters():
            parameter.requires_grad = False

        model.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(
            device
        )

        model.load_state_dict(
            torch.load("./pretrained_vit_8020.pth", map_location=device)
        )

        img = Image.fromarray(image)

        model.to(device)

        model.eval()
        with torch.inference_mode():
            transformed_image = pretrained_vit_transforms(img).unsqueeze(dim=0)

            target_image_pred = model(transformed_image.to(device))

        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        return class_names[target_image_pred_label], target_image_pred_probs.max()

    def show_asl_reference(self):
        # Display the ASL reference guide in a new window
        asl_reference_window = Toplevel(self.root)
        asl_reference_window.title("ASL Reference")

        asl_reference_window.geometry(f"{800}x{650}")

        asl_poster_image_tk = ImageTk.PhotoImage(self.asl_poster_image)

        asl_poster_label = Label(asl_reference_window, image=asl_poster_image_tk)
        asl_poster_label.image = asl_poster_image_tk
        asl_poster_label.pack()

        def on_close():
            asl_reference_window.destroy()

        asl_reference_window.protocol("WM_DELETE_WINDOW", on_close)

    def show_credits(self):
        # Display project credits in a new window
        credits_window = Toplevel(self.root)
        credits_window.title("Credits")

        padding = 10
        credits_window.geometry(f"{600 + 2 * padding}x{400}")

        credits_window.geometry("+%d+%d" % ((self.root.winfo_screenwidth() - credits_window.winfo_reqwidth()) // 2,
                                            (self.root.winfo_screenheight() - credits_window.winfo_reqheight()) // 2))

        credits_window_frame = ttk.Frame(credits_window, padding=(padding, padding))
        credits_window_frame.pack(expand=True, fill=tk.BOTH)

        logo_path = "./Sapienza_Logo.png"
        original_logo_image = Image.open(logo_path)
        original_logo_image = original_logo_image.resize((600, 150)).convert("RGBA")

        university_logo_image = Image.new("RGBA", (original_logo_image.width + 2 * padding, original_logo_image.height), (255, 255, 255, 0))
        university_logo_image.paste(original_logo_image, (padding, 0))

        university_logo_image_tk = ImageTk.PhotoImage(university_logo_image)

        university_logo_label = Label(credits_window_frame, image=university_logo_image_tk)
        university_logo_label.image = university_logo_image_tk
        university_logo_label.pack(pady=10)

        credits_label = ttk.Label(credits_window_frame, text="Project by:", style="Futuristic.TLabel")
        credits_label.config(font=("Segoe UI", 16), foreground="#aa0000")
        credits_label.pack(pady=5)

        group_members = [
            "Waddah Alhajar (Mat. 2049298)", 
            "Daniel Fu (Mat. 2121690)", 
            "Srinjan Ghosh (Mat. 2053796)",
            "Mursal Furqan Kumbhar (Mat. 2047419)", 
            ]

        for member in group_members:
            member_label = ttk.Label(credits_window_frame, text=f"\u2022 {member}", style="Futuristic.TLabel")
            member_label.config(font=("Segoe UI", 14), foreground="#aa0011")
            member_label.pack(anchor=tk.W)

        def on_close():
            credits_window.destroy()

        credits_window.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":
    # Create and run the main application
    root = tk.Tk()
    app = HandTrackingApp(root)
    root.mainloop()