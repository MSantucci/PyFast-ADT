import time
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pyautogui
import os

class StageTUI_bot:
    """this bot is designed to connect to the stage control of TUI and to input movements in the xyz and press goto button"""
    def __init__(self):
        self.path = os.getcwd()
        self.app = Application().connect(path='peoui.exe', timeout = 0.5)
        print('connected to peoui/TUI stage')
        self.window = self.app.Dialog
        # X entry
        self.x = self.window.Edit2
        # Y entry
        self.y = self.window.Edit3
        # Z entry
        self.z = self.window.Edit4
        # a entry
        self.a = self.window.Edit5
        # # b entry
        # self.b = self.window.Edit4
        # button goto to start movement
        self.goto_button = self.window[u'Go to']
        # button flap
        #self.flap_button = self.window[u'TFlapButton']
        self.flap_button = self.window[u'6']
        self.locate1 = pyautogui.locateCenterOnScreen(self.path + '/temspy_bot/test_images\stage2.png')
        pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.1)
        pyautogui.click()

    def bot_setup(self, axis):
        """fill the entry of an axis in stage-tui. axis is a dictionary containing the axis to move and how much in um.
        the set is possible even if the flap is not visible. example: axis = {'x': 555.0, 'y': 55.0, 'z': 5.0} set xyz to their respective values"""

        self.window.set_focus()
        time.sleep(0.1)
        print("received the following dictionary: ", axis)
        pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.1)
        pyautogui.click()

        self.x.set_text("")
        self.y.set_text("")
        self.z.set_text("")
        self.a.set_text("")
        time.sleep(0.1)

        for key, value in axis.items():
            if key == 'x':
                self.x.set_text(str(value))
            elif key == 'y':
                self.y.set_text(str(value))
            elif key == 'z':
                self.z.set_text(str(value))
            time.sleep(1)

    def bot_start(self, wait = False):
        pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.1)
        pyautogui.click()
        time.sleep(0.1)
        # start the rotation here
        try:
            self.goto_button.double_click()
        except:
            print("crashed retry")
            time.sleep(1)
            self.goto_button.double_click()
        if wait == True:
            self.wait_for_button()
        time.sleep(1)

    def wait_for_button(self):
        while True:
            if self.goto_button.is_enabled():
                time.sleep(1)
                print("stage free")
                return
            time.sleep(0.1)

# tui = StageTUI_bot()
# tui.bot_setup(1, goto = False)

        
