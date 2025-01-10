import time
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pyautogui
import win32gui
import win32con
import ctypes

class Outputs_bot:
    """"this bot is designed to conenct to the outputs window of temspy and enter a new value for the DL (diffraction lens)"""
    def __init__(self):
        self.DL = None
        self.app = Application().connect(title=u'Outputs', timeout = 0.5)
        print('connected to Outputs')
        self.window = self.app.Dialog
        self.edit = self.window.Edit5
        self.handle = win32gui.FindWindow(None, 'Outputs')
        self.user32 = ctypes.windll.user32

    def bot_start(self, configuration, DL):
        try:
            self.user32.BlockInput(True)
            self.DL = DL
            print('choosen DL value %s' %str(self.DL))
            win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
            win32gui.SetForegroundWindow(self.handle)
            #self.window.set_focus()
            time.sleep(0.1)
            self.edit.double_click()
            time.sleep(0.1)
            #self.edit.set_text(self.DL)
            pyautogui.typewrite(str(self.DL), interval = 0.01)
            time.sleep(0.33)
            send_keys('{ENTER}')
            #print('DL changed done')
            time.sleep(0.1)
            self.window.minimize()
            time.sleep(0.1)
            self.user32.BlockInput(False)
        except:
            self.user32.BlockInput(False)

#connect to outputs and type a DL value by overwriting the previous one

# need to be tested

# if covered will work in every case thank to set_focus()
# if minimized it work still
# automatically ask to minimize the app when finished
