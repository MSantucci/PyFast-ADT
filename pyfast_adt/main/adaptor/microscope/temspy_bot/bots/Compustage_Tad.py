# this script must be run as SYSTEM user! even administrator is not enough in the f30 because feirootbrick.exe (parent process of CompuStage) is runned by SYSTEM!
# check admin rights
import ctypes
if ctypes.windll.shell32.IsUserAnAdmin():
    print("Python has admin rights.")
else:
    print("Python is NOT running as admin!")
import time
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pyautogui
import win32gui
import win32con
import ctypes
class Compustage_bot:
    def __init__(self):
        self.value = None
        self.velocity = None
        self.app = Application().connect(title=u'CompuStage', timeout = 0.5)
        #self.app = Application().connect(title=u'CompuStage Mark I', class_name='#32770')

        print('connected to Compustage')
        self.window = self.app.Dialog
        # position to edit in degree
        self.edit = self.window.Edit11
        # speed entry to edit
        self.edit2 = self.window.Edit12
        # position to edit for set the axis used
        self.combobox = self.window.combobox
        # button goto to start rotation
        self.button = self.window[u'&Goto']
        self.handle = win32gui.FindWindow(None, 'CompuStage')
        self.user32 = ctypes.windll.user32
        self.combobox_dict = {"X": 0, "Y": 1, "Z": 2, "A": 3, "B": 4}

    def bot_setup(self, configuration, value, velocity, axis = "A" ):
        """compustage window bot. parameters, axis to move, value and velocity in a.u. from fei. if alpha is selected the value is in deg"""

        ######## to check the buttons and edit boxes
        try:
            self.user32.BlockInput(True)
            self.axis = axis.upper()
            self.value = value
            self.velocity = velocity

            print('chosen axis %s value %s with speed %s' % (str(self.axis), str(self.value), str(self.velocity)))

            #self.window.set_focus()
            win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
            win32gui.SetForegroundWindow(self.handle)
            time.sleep(0.1)

            self.edit.double_click()
            time.sleep(0.1)
            pyautogui.typewrite(str(self.value), interval=0.01)
            time.sleep(0.33)
            send_keys('{ENTER}')

            time.sleep(0.1)
            self.edit2.double_click()
            time.sleep(0.1)
            pyautogui.typewrite(str(self.velocity), interval=0.01)
            time.sleep(0.33)
            send_keys('{ENTER}')

            handle = self.combobox.handle
            print("combobox handle:", handle)
            print("self.axis", self.axis)

            time.sleep(0.1)
            self.combobox.select(u"%s" % str(self.axis))
            time.sleep(0.1)
            try:
                win32gui.SetForegroundWindow(self.handle)
            except:
                self.window.minimize()
            time.sleep(0.1)
            self.user32.BlockInput(False)
            print("finished compustage setup")
        except Exception as err:
            print("error bot_setup compustage", err)
            self.user32.BlockInput(False)
    # backup
    # def bot_setup(self, configuration, alpha, velocity):
    #     """compustage window bot. parameters, axis to move, value and velocity in a.u. from fei. if alpha is selected the value is in deg"""
    #
    #     ######## to check the buttons and edit b oxes
    #     try:
    #         self.user32.BlockInput(True)
    #         self.alpha = alpha
    #         self.velocity = velocity
    #         print('choosen alpha value %s with speed %s' % (str(self.alpha), str(self.velocity)))
    #
    #         # self.window.set_focus()
    #         win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
    #         win32gui.SetForegroundWindow(self.handle)
    #         time.sleep(0.1)
    #         self.edit.double_click()
    #         time.sleep(0.1)
    #         pyautogui.typewrite(str(self.alpha), interval=0.01)
    #         time.sleep(0.33)
    #         send_keys('{ENTER}')
    #
    #         time.sleep(0.1)
    #         self.edit2.double_click()
    #         time.sleep(0.1)
    #         pyautogui.typewrite(str(self.velocity), interval=0.01)
    #         time.sleep(0.33)
    #         send_keys('{ENTER}')
    #
    #         time.sleep(0.1)
    #         win32gui.SetForegroundWindow(self.handle)
    #         time.sleep(0.1)
    #         self.user32.BlockInput(False)
    #     except:
    #         self.user32.BlockInput(False)

    def bot_start(self, configuration, wait = False):
        # start the rotation here
        win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
        win32gui.SetForegroundWindow(self.handle)
        self.button.double_click()
        if wait == True:
            self.wait_for_button()


    def wait_for_button(self):
        while True:
            if self.button.is_enabled():
                time.sleep(0.3)
                print("stage free")
                return
            time.sleep(0.1)



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

if __name__ == "__main__":
    compustage = Compustage_bot()
    #compustage.bot_setup("f30", 100, 0.07, axis="X")
    #compustage.bot_start("f30")





