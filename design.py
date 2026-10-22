import os
import pickle
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext

import easyocr
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import translators as ts
from PIL import Image, ImageGrab
from model import CRNN
# --- Dragable GUI Windows ---


class CaptureWindow(tk.Toplevel):

    def __init__(self, master, bg_color="#262729", *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.x = None
        self.y = None

        self.label = tk.Label(self, bg=bg_color)
        self.label.pack(fill="both", expand=True)
        self.label.bind("<ButtonPress-1>", self.start_move)
        self.label.bind("<ButtonRelease-1>", self.stop_move)
        self.label.bind("<B1-Motion>", self.do_move)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        if self.x is not None and self.y is not None:
            deltax = event.x - self.x
            deltay = event.y - self.y
            x_now = self.winfo_x() + deltax
            y_now = self.winfo_y() + deltay
            self.geometry(f"+{x_now}+{y_now}")


class QueryWindow(tk.Toplevel):

    def __init__(self, master, bg_color="#262729", *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.x = None
        self.y = None

        self.scrolledText = scrolledtext.ScrolledText(
            self,
            wrap="word",
            bg=bg_color,
            font=("Arial", 20),
            fg="white",
            state="disabled",
        )
        self.scrolledText.pack(fill="both", expand=True)
        self.scrolledText.bind("<ButtonPress-1>", self.start_move)
        self.scrolledText.bind("<ButtonRelease-1>", self.stop_move)
        self.scrolledText.bind("<B1-Motion>", self.do_move)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        if self.x is not None and self.y is not None:
            deltax = event.x - self.x
            deltay = event.y - self.y
            x_now = self.winfo_x() + deltax
            y_now = self.winfo_y() + deltay
            self.geometry(f"+{x_now}+{y_now}")


# --- On-Screen Translator Main Class ---


class OnScreenTranslator:

    def __init__(self):
        self.dark = "#262729"

        self.language_dict = {
            "Afrikaans": "af",
            "Arabic": "ar",
            "Chinese (Simplified)": "zh-cn",
            "Chinese (Traditional)": "zh-tw",
            "English": "en",
            "French": "fr",
            "German": "de",
            "Indonesian": "id",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru",
            "Spanish": "es",
            "Vietnamese": "vi",
        }
        self.language_list = list(self.language_dict.keys())

        self.engine_dict = {
            "Google Translate": "google",
            "Bing": "bing",
            "Yandex": "yandex",
            "DeepL": "deepl",
            "Baidu": "baidu",
        }
        self.engine_list = list(self.engine_dict.keys())

        self.captureW = None
        self.queryW = None
        self.resultW = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Loading OCR Engine on Device: {self.device}")

        # Vocabulary Loading
        vocab_path = "data/generated/train_text_lines/vocab.pkl"
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                self.vocab = list(pickle.load(f))
        else:
            self.vocab = []

        # Model Checkpoint Loading
        model_path = "crnn_epoch.pth"
        if not os.path.exists(model_path):
            model_path = "checkpoints/best_crnn.pth"

        if os.path.exists(model_path):
            print(f"Loading checkpoint: {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            if "vocab" in checkpoint:
                self.vocab = checkpoint["vocab"]

            self.model = CRNN(input_height=32, num_classes=len(self.vocab)).to(
                self.device
            )
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
            print(
                f"Model successfully loaded with {len(self.vocab)} classes."
            )
        else:
            print(f"Error: Model checkpoint not found at {model_path}")
            self.model = None

        self.reader = easyocr.Reader(["ja"], gpu=torch.cuda.is_available())

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Tkinter UI Initialization
        self.main_window = tk.Tk()
        self.main_window.title("On Screen Translator")
        self.main_window.configure(bg=self.dark)

        self.whitevar = tk.IntVar(value=1)

        self._build_ui()

        # UI Event Loops
        self.main_window.after(1000, self.ocr_process_step)
        self.main_window.after(500, self.translation_process_step)
        self.main_window.after(200, self.match_windows)

        self.main_window.mainloop()

    def _build_ui(self):
        main_frame = tk.Frame(self.main_window, bg=self.dark)
        main_frame.pack(fill="x", padx=5, pady=5)

        ttk.Button(
            main_frame,
            text="Capture Window",
            command=self.toggle_capture_window,
        ).pack(side="left", padx=5)
        ttk.Button(
            main_frame, text="Query Window", command=self.toggle_query_window
        ).pack(side="left", padx=5)
        ttk.Button(
            main_frame, text="Result Window", command=self.toggle_result_window
        ).pack(side="left", padx=5)

        tk.Checkbutton(
            main_frame,
            text="White Text",
            variable=self.whitevar,
            onvalue=1,
            offvalue=0,
            bg=self.dark,
            fg="white",
            selectcolor="black",
            activebackground=self.dark,
            activeforeground="white",
        ).pack(side="left", padx=5)

        self.input_text = scrolledtext.ScrolledText(
            self.main_window,
            wrap="word",
            height=4,
            bg=self.dark,
            fg="white",
            font=("Arial", 18),
        )
        self.input_text.pack(fill="both", padx=10, pady=5, expand=True)

        options_frame = tk.Frame(self.main_window, bg=self.dark)
        options_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(
            options_frame, text="Engine:", fg="white", bg=self.dark
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.engine_combo = ttk.Combobox(
            options_frame, values=self.engine_list, state="readonly", width=15
        )
        self.engine_combo.grid(row=0, column=1, padx=5, pady=5)
        self.engine_combo.current(0)

        tk.Label(options_frame, text="From:", fg="white", bg=self.dark).grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        self.from_combo = ttk.Combobox(
            options_frame,
            values=self.language_list,
            state="readonly",
            width=15,
        )
        self.from_combo.grid(row=0, column=3, padx=5, pady=5)
        self.from_combo.set("Japanese")

        tk.Label(options_frame, text="To:", fg="white", bg=self.dark).grid(
            row=0, column=4, padx=5, pady=5, sticky="w"
        )
        self.to_combo = ttk.Combobox(
            options_frame,
            values=self.language_list,
            state="readonly",
            width=15,
        )
        self.to_combo.grid(row=0, column=5, padx=5, pady=5)
        self.to_combo.set("English")

        ttk.Button(options_frame, text="⇆ Swap", command=self.swap_languages).grid(
            row=0, column=6, padx=5, pady=5
        )

        self.output_text = scrolledtext.ScrolledText(
            self.main_window,
            wrap="word",
            height=4,
            bg=self.dark,
            fg="white",
            font=("Arial", 18),
        )
        self.output_text.pack(fill="both", padx=10, pady=5, expand=True)

    def toggle_capture_window(self):
        if self.captureW is None or not self.captureW.winfo_exists():
            self.captureW = CaptureWindow(self.main_window, bg_color=self.dark)
            self.captureW.title("Capture Window")
            self.captureW.geometry("1280x360+0+0")
            self.captureW.wm_attributes("-topmost", True)
            self.captureW.attributes("-alpha", 0.8)

            slider_frame = tk.Frame(self.captureW, bg=self.dark)
            slider_frame.pack(side="right", padx=5, anchor="e")

            tk.Checkbutton(
                slider_frame,
                text="White Text",
                variable=self.whitevar,
                onvalue=1,
                offvalue=0,
                bg=self.dark,
                fg="white",
                selectcolor="black",
            ).pack(side="right", padx=5)

            style = ttk.Style()
            style.configure("TScale", background=self.dark)
            opacity_slider = ttk.Scale(
                slider_frame,
                from_=0.1,
                to=1.0,
                value=0.8,
                orient="horizontal",
                style="TScale",
                command=lambda val: self.captureW.attributes(
                    "-alpha", float(val)
                ),
            )
            opacity_slider.pack(side="right", padx=5)
        else:
            self.captureW.destroy()
            self.captureW = None

    def toggle_query_window(self):
        if self.queryW is None or not self.queryW.winfo_exists():
            self.queryW = QueryWindow(self.main_window, bg_color=self.dark)
            self.queryW.title("Query Window")
            self.queryW.geometry("480x240+0+0")
            self.queryW.wm_attributes("-topmost", True)
        else:
            self.queryW.destroy()
            self.queryW = None

    def toggle_result_window(self):
        if self.resultW is None or not self.resultW.winfo_exists():
            self.resultW = QueryWindow(self.main_window, bg_color=self.dark)
            self.resultW.title("Result Window")
            self.resultW.geometry("480x240+500+0")
            self.resultW.wm_attributes("-topmost", True)
        else:
            self.resultW.destroy()
            self.resultW = None

    def swap_languages(self):
        src_curr = self.from_combo.get()
        dest_curr = self.to_combo.get()
        self.from_combo.set(dest_curr)
        self.to_combo.set(src_curr)

    def match_windows(self):
        if self.queryW and self.queryW.winfo_exists():
            inp = self.get_text(self.input_text)
            if inp != self.get_text(self.queryW.scrolledText):
                self.set_text(self.queryW.scrolledText, inp, readonly=True)

        if self.resultW and self.resultW.winfo_exists():
            out = self.get_text(self.output_text)
            if out != self.get_text(self.resultW.scrolledText):
                self.set_text(self.resultW.scrolledText, out, readonly=True)

        self.main_window.after(200, self.match_windows)

    def decode_ctc(self, logits):
        """Internal CTC path decoder matching model output format."""
        predicted_indices = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
        decoded_text = []
        previous_index = None

        for idx in predicted_indices:
            idx = int(idx)
            if idx != 0 and idx != previous_index:
                char_idx = idx - 1
                if char_idx < len(self.vocab):
                    decoded_text.append(self.vocab[char_idx])
            previous_index = idx

        return "".join(decoded_text)

    def ocr_process_step(self):
        if self.captureW and self.captureW.winfo_exists() and self.model:
            try:
                x = self.captureW.winfo_rootx()
                y = self.captureW.winfo_rooty()
                w = self.captureW.winfo_width()
                h = self.captureW.winfo_height()

                if w > 10 and h > 10:
                    screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                    tensors = self.extract_line_tensors(screenshot)

                    if tensors:
                        predicted_lines = self.predict_ocr(tensors)
                        full_text = "".join(predicted_lines)
                        if full_text and full_text != self.get_text(
                            self.input_text
                        ):
                            self.set_text(
                                self.input_text, full_text, readonly=False
                            )
            except Exception as e:
                print(f"OCR Error: {e}")

        self.main_window.after(1000, self.ocr_process_step)

    def extract_line_tensors(self, screenshot_img):
        grayscale = screenshot_img.convert("L")
        results = self.reader.readtext(np.array(screenshot_img))

        if not results:
            return [self.transform(grayscale)]

        tensors = []
        for bbox, _, _ in results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            left, right = int(min(x_coords)), int(max(x_coords))
            top, bottom = int(min(y_coords)), int(max(y_coords))

            if right - left > 5 and bottom - top > 5:
                cropped = grayscale.crop((left, top, right, bottom))
                tensors.append(self.transform(cropped))

        return tensors if tensors else [self.transform(grayscale)]

    def predict_ocr(self, tensor_list):
        predictions = []
        with torch.inference_mode():
            for t in tensor_list:
                img_batch = t.unsqueeze(0).to(self.device)
                outputs = self.model(img_batch)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                text = self.decode_ctc(log_probs)
                predictions.append(text)
        return predictions

    def translation_process_step(self):
        txt = self.get_text(self.input_text).strip()
        if txt:
            src_lang = self.language_dict.get(self.from_combo.get(), "ja")
            dest_lang = self.language_dict.get(self.to_combo.get(), "en")
            engine_name = self.engine_dict.get(
                self.engine_combo.get(), "google"
            )

            def run_translate():
                try:
                    translated = ts.translate_text(
                        txt,
                        translator=engine_name,
                        from_language=src_lang,
                        to_language=dest_lang,
                    )
                    self.main_window.after(
                        0, self.set_text, self.output_text, translated, True
                    )
                except Exception as e:
                    print(f"Translation Error: {e}")

            threading.Thread(target=run_translate, daemon=True).start()

        self.main_window.after(1500, self.translation_process_step)

    @staticmethod
    def get_text(widget: scrolledtext.ScrolledText) -> str:
        return widget.get("1.0", "end-1c")

    @staticmethod
    def set_text(
        widget: scrolledtext.ScrolledText, value: str, readonly: bool = False
    ):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", value)
        if readonly:
            widget.configure(state="disabled")


if __name__ == "__main__":
    OnScreenTranslator()