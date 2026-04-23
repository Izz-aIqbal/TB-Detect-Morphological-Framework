import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from filters import apply_butterworth_lowpass, apply_clahe
from grading import calculate_advanced_metrics

class Clinical_TB_App_Final:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated TB Severity Grading - Multistage DIP")
        self.root.geometry("1450x950")
        self.root.configure(bg="#0f172a")

        self.current_page = 1
        self.current_report_data = None
        self.setup_ui()

    def setup_ui(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        if self.current_page == 1:
            self.render_landing_page()
        else:
            self.render_analysis_page()

    def render_landing_page(self):
        header_card = tk.Frame(self.root, bg="#1e293b", padx=40, pady=60,
                               highlightbackground="#38bdf8", highlightthickness=3)
        header_card.pack(pady=150, padx=100, fill="both")

        tk.Label(header_card, text="AUTOMATED TB SEVERITY GRADING",
                 font=("Verdana", 44, "bold"), bg="#1e293b", fg="#38bdf8").pack()

        tk.Label(header_card, text="Multistage DIP Pipeline for Rural Diagnostics",
                 font=("Verdana", 18, "italic"), bg="#1e293b", fg="#94a3b8").pack(pady=15)

        tk.Button(header_card, text="LAUNCH ANALYSIS ENGINE", command=self.go_to_analysis,
                  bg="#38bdf8", fg="#0f172a", font=("Verdana", 16, "bold"),
                  padx=60, pady=20, relief="flat", activebackground="#7dd3fc").pack(pady=50)

    def render_analysis_page(self):
        nav_bar = tk.Frame(self.root, bg="#1e293b", height=70)
        nav_bar.pack(fill="x")
        nav_bar.pack_propagate(False)

        tk.Button(nav_bar, text="<- BACK TO DASHBOARD", command=self.go_to_home,
                  bg="#334155", fg="white", font=("Verdana", 10, "bold"),
                  relief="flat", padx=25, pady=10).pack(side="left", padx=20, pady=15)

        tk.Label(nav_bar, text="TB SEVERITY DIAGNOSTIC SUITE",
                 font=("Verdana", 22, "bold"), bg="#1e293b", fg="#38bdf8").pack(pady=15)

        control_card = tk.Frame(self.root, bg="#1e293b", height=130)
        control_card.pack(fill="x", side="bottom")
        control_card.pack_propagate(False)

        left_frame = tk.Frame(control_card, bg="#1e293b")
        left_frame.pack(side="left", fill="both", padx=40, pady=10)

        self.res_grade = tk.Label(left_frame, text="STATUS: READY",
                                  font=("Verdana", 20, "bold"), fg="#38bdf8", bg="#1e293b",
                                  anchor="w")
        self.res_grade.pack(anchor="w")

        self.res_stats = tk.Label(left_frame, text="Metrics: Pending Upload...",
                                  font=("Verdana", 12), fg="white", bg="#1e293b",
                                  anchor="w")
        self.res_stats.pack(anchor="w", pady=(4, 0))

        right_frame = tk.Frame(control_card, bg="#1e293b")
        right_frame.pack(side="right", fill="y", padx=40)

        btn_cfg = dict(font=("Verdana", 10, "bold"), width=16, height=2,
                       relief="flat", cursor="hand2")

        tk.Button(right_frame, text="BATCH DATA",
                  command=self.run_batch_processing,
                  bg="#8b5cf6", fg="white", **btn_cfg).pack(
                      side="right", padx=8, pady=0, anchor="center",
                      expand=True, fill="y")

        tk.Button(right_frame, text="GENERATE REPORT",
                  command=self.save_report,
                  bg="#10b981", fg="white", **btn_cfg).pack(
                      side="right", padx=8, pady=0, anchor="center",
                      expand=True, fill="y")

        tk.Button(right_frame, text="UPLOAD SCAN",
                  command=self.run_pipeline,
                  bg="#38bdf8", fg="#0f172a", **btn_cfg).pack(
                      side="right", padx=8, pady=0, anchor="center",
                      expand=True, fill="y")

        self.mid_frame = tk.Frame(self.root, bg="#0f172a")
        self.mid_frame.pack(fill="both", expand=True, padx=30, pady=10)

        self.canvases = []
        titles = ["INPUT RADIOGRAPH", "INFILTRATION DENSITY HEATMAP", "SKELETONIZED LESIONS"]
        for i in range(3):
            img_card = tk.Frame(self.mid_frame, bg="#1e293b", padx=10, pady=10,
                                highlightbackground="#334155", highlightthickness=1)
            img_card.grid(row=0, column=i, padx=15)
            tk.Label(img_card, text=titles[i], fg="#38bdf8", bg="#1e293b",
                     font=("Verdana", 11, "bold")).pack(pady=8)
            c = tk.Label(img_card, bg="black", width=400, height=400)
            c.pack()
            placeholder = ImageTk.PhotoImage(
                Image.fromarray(np.zeros((400, 400))).convert("L"))
            c.config(image=placeholder)
            c.image = placeholder
            self.canvases.append(c)

    def go_to_analysis(self):
        self.current_page = 2
        self.setup_ui()

    def go_to_home(self):
        self.current_page = 1
        self.setup_ui()

    # --- UPDATED VALIDATION ---
    def is_xray_image(self, img_gray):
        mu = np.mean(img_gray)
        std = np.std(img_gray)
        
        # Calculate Edge density to distinguish from flat images or random photos
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img_gray.shape[0] * img_gray.shape[1])

        # Chest X-rays usually have specific mean range, contrast (std), and edge patterns
        if not (60 < mu < 185): # Tightened range
            return False, f"Intensity check failed ({mu:.1f})."
        if std < 35: # Natural X-rays have higher variance than flat graphics
            return False, f"Contrast check failed (std={std:.1f})."
        if edge_density < 0.01 or edge_density > 0.15: # X-rays have moderate complexity
            return False, "Structural profile does not match a radiograph."
        
        return True, "OK"

    def validate_and_measure(self, raw, restored):
        valid, reason = self.is_xray_image(raw)
        if not valid:
            messagebox.showerror("Invalid Scan",
                f"Validation Error: {reason}\n\nPlease upload a valid chest X-ray image.")
            return False
        p = psnr(raw, restored)
        s, _ = ssim(raw, restored, full=True)
        return f"PSNR: {p:.1f} | SSIM: {s:.3f}", p, s

    def run_batch_processing(self):
        folder = filedialog.askdirectory()
        if not folder: return
        rows, skipped = [], 0
        for file in os.listdir(folder):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img = cv2.imread(os.path.join(folder, file), 0)
            if img is None: continue
            img = cv2.resize(img, (400, 400))
            valid, _ = self.is_xray_image(img)
            if not valid:
                skipped += 1
                continue
            restored = apply_butterworth_lowpass(img)
            enhanced = apply_clahe(restored)
            _, _, iar, count, grade, _ = calculate_advanced_metrics(enhanced)
            p = psnr(img, restored)
            rows.append({"File": file, "IAR": iar, "Grade": grade, "PSNR": p})

        if rows:
            pd.DataFrame(rows).to_csv("Research_Results.csv", index=False)
            msg = f"Done!\n{len(rows)} X-rays processed."
            if skipped: msg += f"\n{skipped} images were ignored (not X-rays)."
            messagebox.showinfo("Success", msg)
        else:
            messagebox.showwarning("Warning", "No valid chest X-rays found in folder.")

    def run_pipeline(self):
        path = filedialog.askopenfilename()
        if not path: return
        raw = cv2.imread(path, 0)
        if raw is None: return
        raw = cv2.resize(raw, (400, 400))
        restored = apply_butterworth_lowpass(raw)

        result = self.validate_and_measure(raw, restored)
        if result is False: return

        q_text, p_val, s_val = result
        enhanced = apply_clahe(restored)
        skel, heatmap, iar, count, grade, symmetry_gui = calculate_advanced_metrics(enhanced)

        self.display(raw, self.canvases[0])
        self.display(heatmap, self.canvases[1], is_color=True)
        self.display(skel, self.canvases[2])

        self.res_grade.config(text=grade.upper())
        self.res_stats.config(text=f"{q_text} | IAR: {iar:.2f}% | {symmetry_gui}")

        self.current_report_data = {
            "grade": grade, "iar": iar, "lesions": count, "file": os.path.basename(path),
            "symmetry_raw": (symmetry_gui.split(' ', 1)[1] if ' ' in symmetry_gui else symmetry_gui),
            "psnr": p_val, "ssim": s_val,
        }

    def save_report(self):
        if self.current_report_data is None:
            messagebox.showerror("No Data", "Process an image first.")
            return
        data = self.current_report_data
        self.write_text_file(data)

        report_win = tk.Toplevel(self.root)
        report_win.title(f"Report - {data['file']}")
        report_win.geometry("850x980")
        report_win.configure(bg="#0f172a")

        tk.Label(report_win, text="V-SCAN MEDICAL AI: DIGITAL REPORT",
                 font=("Verdana", 22, "bold"), fg="#38bdf8", bg="#0f172a", pady=25).pack()

        card = tk.Frame(report_win, bg="#1e293b", padx=40, pady=40,
                        highlightbackground="#38bdf8", highlightthickness=1)
        card.pack(fill="both", expand=True, padx=50, pady=10)

        table_frame = tk.Frame(card, bg="#0f172a", pady=2)
        table_frame.pack(fill="x", pady=10)

        report_rows = [
            ("Severity Classification", data["grade"]),
            ("Infection Area Ratio (IAR)", f"{data['iar']:.2f}%"),
            ("Detected Lesion Clusters", f"{data['lesions']} Clusters"),
            ("Bilateral Symmetry", data["symmetry_raw"]),
            ("Image Quality (PSNR)", f"{data['psnr']:.1f} dB"),
            ("Structural Similarity (SSIM)", f"{data['ssim']:.3f}"),
        ]

        for i, (label, val) in enumerate(report_rows):
            bg_color = "#1e293b" if i % 2 == 0 else "#16213e"
            r_frame = tk.Frame(table_frame, bg=bg_color, pady=12)
            r_frame.pack(fill="x")
            tk.Label(r_frame, text=label, font=("Verdana", 11), fg="white", bg=bg_color, width=30, anchor="w").pack(side="left", padx=20)
            tk.Label(r_frame, text=val, font=("Verdana", 11, "bold"), fg="#10b981", bg=bg_color, width=25, anchor="w").pack(side="left")

        tk.Button(report_win, text="CLOSE REPORT PREVIEW", command=report_win.destroy,
                  bg="#ef4444", fg="white", font=("Verdana", 10, "bold"), padx=40, pady=12).pack(pady=15)

    def write_text_file(self, data):
        try:
            filename = f"Report_{data['file']}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("       V-SCAN MEDICAL AI: RADIOLOGY DIAGNOSTIC REPORT\n")
                f.write("=" * 70 + "\n")
                f.write(f" FILE ID : {data['file']:<30} DATE: 02-04-2026\n")
                f.write("-" * 70 + "\n\n")
                f.write(f" Severity Grade: {data['grade']}\n")
                f.write(f" IAR: {data['iar']:.2f}%\n")
                f.write(f" PSNR: {data['psnr']:.1f} dB\n")
                f.write("=" * 70 + "\n")
        except: pass

    def display(self, arr, label_obj, is_color=False):
        if is_color: img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        else: img = Image.fromarray(arr)
        img_tk = ImageTk.PhotoImage(image=img.resize((400, 400)))
        label_obj.config(image=img_tk)
        label_obj.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = Clinical_TB_App_Final(root)
    root.mainloop()