from tkinter import filedialog
import tkinter as tk

from data.index import FILTERS_INFO

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select a Folder")
    return folder_path

def get_filter_info(filter_name):
    filter_info = FILTERS_INFO.get(filter_name)
    if filter_info:
        return filter_info
    else:
        return {"error": f"Filter '{filter_name}' not found."}




