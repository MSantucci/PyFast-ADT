# handpanel for digital micrograph in tkinter for FEI tecnai
# the script generate a tkinter window with the basic commands used in the handpanel of a tecnai microscope, necessary to handle 3ded experiments
try:
    import temscript
except Exception as err:
    print('import failed:', err)
try:
    import DigitalMicrograph as DM
    DM.ClearResults()
except Exception as err:
    print('import failed:', err)
import tkinter as tk
from tkinter import ttk
import os
import sys
import webbrowser
import numpy as np
print(sys.path)
dir_path = os.getcwd()
# dir_path = r'C:\Users\remote\Desktop\test_temscript_from_digit_micrograph'
#dir_path = r"/python_FEI_FastADT/07042023_build_up/main"
if dir_path in sys.path:
    print('dir already there')
else:
    sys.path.append(dir_path)
    #sys.path.append(r"/python_FEI_FastADT/07042023_build_up/main/adaptor/microscope")
    #sys.path.append(r"/python_FEI_FastADT/07042023_build_up/main/adaptor/camera")
    #sys.path.append(r"/python_FEI_FastADT/07042023_build_up/main/FASTADT")

from FAST_ADT_gui import FastADT
from handpanels_simulator import HandPanel

class MainGUI():
    def __init__(self):
        super().__init__()
        self.root = tk.Tk(baseName = "GUI")
        #label0 = tk.Label(self.root).pack()
        label1 = tk.Label(self.root, text = "\n3DED Tools for data acquisition \nplease select the brand interface from the box below").pack()

        label1_1 = tk.Label(self.root, text="microscope adaptor").pack()

        self.brand_var = tk.StringVar(self.root, value = "fei")
        self.brand = ttk.Combobox(self.root, textvariable=self.brand_var, state = "readonly", width=15)
        self.brand["values"] = list(("fei", "jeol", "gatan_fei", "gatan_jeol", "fei_temspy", "power_user"))
        self.brand.pack()

        label1_2 = tk.Label(self.root, text="detector adaptor").pack()

        self.brand_cam_var = tk.StringVar(self.root, value="timepix1")
        self.brand_cam = ttk.Combobox(self.root, textvariable=self.brand_cam_var, state="readonly", width=15)
        self.brand_cam["values"] = list(("timepix1", "xf416r", "us4000", "us2000", "ceta", "medipix3", "merlin", "power_user"))
        self.brand_cam.pack()

        label2 = tk.Label(self.root, text = "\nHandpanels is a simulator for the microscope \nhandpanels to help working in remote\n").pack()

        self.button1 = tk.Button(self.root, text="start handpanels", command=lambda: self.start_handpanels())
        self.button1.pack()

        label3 = tk.Label(self.root, text = "\nPyFast-ADT is a tool to automate the \ndata collection of 3ded experiments\n").pack()

        self.button2 = tk.Button(self.root, text="start pyFast-ADT", command=lambda: self.start_fastadt())
        self.button2.pack()

        link = "https://naned.eu/training-by-research/phd-projects#rt9"
        label4 = tk.Label(self.root, text = "\nPlease don't close this window, \nall the child windows depends by this one\n").pack()
        label5 = tk.Label(self.root, text = "Developed by Marco Santucci, \nMainz April 2023, \nNanED EU Project: No 956099").pack()
        label6 = tk.Label(self.root, text="NanED link click here to know more", fg = "blue", cursor = "hand2")
        label6.pack()
        label6.bind("<Button-1>", lambda e: self.link_browser(link))

        self.root.title("3DED Data Acquisition Tools")
        self.root.geometry("330x460")
        self.root.mainloop()

    def start_handpanels(self):
        interface, camera = self.get_brand()
        self.handpanel = HandPanel(self.root, brand = interface, camera = camera)

    def start_fastadt(self):
        interface, camera = self.get_brand()
        self.fastadt = FastADT(self.root, brand = interface, camera = camera)

    def get_brand(self):
        interface = self.brand_var.get()
        camera = self.brand_cam_var.get()
        print("choosen interface and camera:", interface, camera)
        return interface, camera

    def link_browser(self, url):
        webbrowser.open_new(url)

if __name__ == '__main__':
    app = MainGUI()

