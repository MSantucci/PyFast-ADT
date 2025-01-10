import time
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pyautogui
import win32gui
import win32con
import ctypes
class Compustage_bot:
    def __init__(self):
        self.alpha = None
        self.velocity = None
        self.app = Application().connect(title=u'CompuStage', timeout = 0.5)
        #self.app = Application().connect(title=u'CompuStage Mark I', class_name='#32770')

        print('connected to Compustage')
        self.window = self.app.Dialog
        # position to edit in degree
        self.edit = self.window.Edit11
        # speed entry to edit
        self.edit2 = self.window.Edit12
        # button goto to start rotation
        self.button = self.window[u'&Goto']
        self.handle = win32gui.FindWindow(None, 'CompuStage')
        self.user32 = ctypes.windll.user32
    def bot_setup(self, configuration, alpha, velocity):
        """alpha in deg and velocity in a.u. from fei"""
        ######## to check the buttons and edit b oxes
        try:
            self.user32.BlockInput(True)
            self.alpha = alpha
            self.velocity = velocity
            print('choosen alpha value %s with speed %s' %(str(self.alpha), str(self.velocity)))

            #self.window.set_focus()
            win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
            win32gui.SetForegroundWindow(self.handle)
            time.sleep(0.1)
            self.edit.double_click()
            time.sleep(0.1)
            pyautogui.typewrite(str(self.alpha), interval=0.01)
            time.sleep(0.33)
            send_keys('{ENTER}')

            time.sleep(0.1)
            self.edit2.double_click()
            time.sleep(0.1)
            pyautogui.typewrite(str(self.velocity), interval=0.01)
            time.sleep(0.33)
            send_keys('{ENTER}')

            time.sleep(0.1)
            win32gui.SetForegroundWindow(self.handle)
            time.sleep(0.1)
            self.user32.BlockInput(False)
        except:
            self.user32.BlockInput(False)

    def bot_start(self, configuration):
        # start the rotation here
        win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
        win32gui.SetForegroundWindow(self.handle)
        self.button.double_click()

        #time.sleep(0.33)
        #self.window.minimize()
        #time.sleep(0.1)

#connect to outputs and type a DL value by overwriting the previous one

#import ctypes
#ctypes.windll.shell32.ShellExecuteW(None, "runas", "python", "temspy_compustage.py", None, 1)

# from pywinauto.application import Application
# import time
# from pywinauto.application import Application
# from pywinauto.keyboard import send_keys
# import pyautogui
#
# app = Application().connect(title=u'CompuStage Mark I', class_name='#32770')
# window = app.Dialog
#
# # goto button to start
# #button = window[u'&Goto']
# #button.click()
#
# # position to edit in degree
# edit = window.Edit11
# #edit.click_input()
#
# # speed entry to edit
# #edit2 = window.Edit12
# #edit2.click()
#
# # select the axis for the rotation (which has to be A)
# #window = app.Dialog
# #combobox = window.ComboBox
# #combobox.click()
# #combobox.select(u'A')
# position = 50
# velocity = 0.071
#
# window.set_focus()
# time.sleep(0.1)
# edit.double_click()
# time.sleep(0.1)
# #edit.set_text(self.DL)
# pyautogui.typewrite(str(position), interval = 0.01)
# time.sleep(0.33)
# send_keys('{ENTER}')
#
# edit2 = window.Edit12
# time.sleep(0.1)
# edit2.double_click()
# time.sleep(0.1)
# #edit.set_text(self.DL)
# pyautogui.typewrite(str(velocity), interval = 0.01)
# time.sleep(0.33)
# send_keys('{ENTER}')
#
# button = window[u'&Goto']
# window.set_focus()
# button.double_click()


