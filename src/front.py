import customtkinter as ctk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import shutil
import os

from download_contaminants_info import download_contaminants_data

# Configuration initiale
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ContaminantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Contaminant Visualization Tool")
        self.geometry("800x800")

        # Variables globales
        self.csv_path = ctk.StringVar()
        self.start_date = ctk.StringVar()
        self.end_date = ctk.StringVar()
        self.selected_disease = ctk.StringVar()
        self.selected_contaminant = ctk.StringVar()
        self.selected_map_type = ctk.StringVar()
        self.disease_options = []

        # Frame Étape 1
        self.frame_step1 = ctk.CTkFrame(self)
        self.frame_step1.pack(pady=30, padx=20, fill="x")

        self.label_csv = ctk.CTkLabel(self.frame_step1, text="Step 1: Load contaminant data (CSV)")
        self.label_csv.pack(pady=(10, 5))

        self.button_browse = ctk.CTkButton(self.frame_step1, text="Browse CSV", command=self.browse_csv)
        self.button_browse.pack()

        self.label_dates = ctk.CTkLabel(self.frame_step1, text="Select date range (YYYY-MM-DD)")
        self.label_dates.pack(pady=(10, 5))

        self.entry_start = ctk.CTkEntry(self.frame_step1, placeholder_text="Start date", textvariable=self.start_date)
        self.entry_start.pack(pady=5)

        self.entry_end = ctk.CTkEntry(self.frame_step1, placeholder_text="End date", textvariable=self.end_date)
        self.entry_end.pack(pady=5)

        self.button_download = ctk.CTkButton(self.frame_step1, text="Validate step 1", command=self.download_and_load)
        self.button_download.pack()

        # Frame Étape 2
        self.frame_step2 = ctk.CTkFrame(self)
        self.frame_step2.pack(pady=15, padx=20, fill="x")

        self.label_disease = ctk.CTkLabel(self.frame_step2, text="Step 2: Select disease")
        self.label_disease.pack(pady=(10, 5))

        self.disease_menu = ctk.CTkOptionMenu(self.frame_step2, values=["-- Select disease --"], variable=self.selected_disease)
        self.disease_menu.pack()

        self.button_validate_step2 = ctk.CTkButton(self.frame_step2, text="Validate Step 2", command=self.validate_step2)
        self.button_validate_step2.pack(pady=10)

        # Frame Étape 3
        self.frame_step3 = ctk.CTkFrame(self)
        self.frame_step3.pack(pady=15, padx=20, fill="x")

        self.label_contaminant = ctk.CTkLabel(self.frame_step3, text="Step 3: Select contaminant")
        self.label_contaminant.pack(pady=(10, 5))

        self.contaminant_menu = ctk.CTkOptionMenu(self.frame_step3, values=["NO2", "PM10", "O3"], variable=self.selected_contaminant)
        self.contaminant_menu.pack()

        self.label_map_type = ctk.CTkLabel(self.frame_step3, text="Select map type")
        self.label_map_type.pack(pady=(10, 5))

        self.map_menu = ctk.CTkOptionMenu(self.frame_step3, values=["Heatmap", "Cluster Map"], variable=self.selected_map_type)
        self.map_menu.pack()

        self.button_show_map = ctk.CTkButton(self.frame_step3, text="Display map", command=self.open_map_window)
        self.button_show_map.pack(pady=10)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                os.makedirs("data-csv", exist_ok=True)
                # On copie directement le fichier sélectionné dans metadata.csv
                shutil.copy(file_path, "data-csv/metadata.csv")
                self.csv_path.set("data-csv/metadata.csv")  # on met à jour la variable pour qu’elle pointe vers metadata.csv
                messagebox.showinfo("File Selected", f"Selected file copied as metadata.csv")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy file: {e}")

    def download_and_load(self):
        try:
            download_contaminants_data(self.start_date.get(), self.end_date.get())
            self.csv_path.set("data-csv/contaminants_data.csv")
            self.validate_step1()
        except Exception as e:
            messagebox.showerror("Download error", str(e))

    def validate_step1(self):
        metadata_path = "data-csv/metadata.csv"

        if not os.path.exists(metadata_path):
            messagebox.showerror("Missing file", "metadata.csv not found. Please select a file first.")
            return

        if not self.start_date.get() or not self.end_date.get():
            messagebox.showerror("Missing dates", "Please enter both start and end dates.")
            return

        try:
            try:
                df = pd.read_csv(metadata_path)
            except UnicodeDecodeError:
                df = pd.read_csv(metadata_path, sep=";", encoding="latin1")

            df.columns = df.columns.str.strip().str.lower()

            if 'cod' in df.columns:
                self.disease_options = sorted(df['cod'].dropna().unique().tolist())
                self.disease_menu.configure(values=self.disease_options)
                if self.disease_options:
                    self.selected_disease.set(self.disease_options[0])
                messagebox.showinfo("Success", "Diseases updated from metadata.csv.")
            else:
                messagebox.showwarning("Missing column", f"'cod' column not found.\nColumns found: {df.columns.tolist()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))



    def validate_step2(self):
        if not self.selected_disease.get():
            messagebox.showerror("Missing selection", "Please select a disease.")

    def open_map_window(self):
        new_window = ctk.CTkToplevel(self)
        new_window.title("Map Viewer")
        new_window.geometry("800x600")
        label = ctk.CTkLabel(new_window, text="Map will be displayed here.")
        label.pack(pady=30)

if __name__ == "__main__":
    app = ContaminantApp()
    app.mainloop()