import time
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pyautogui
import os

class Directalignment_bot:
    """this bot is designed to connect to the Direct Allignments and to press diffraction alignment button to reset the diffraction shift"""
    def __init__(self):
        self.path = os.getcwd()
        #print(self.path)
        #print(self.path + '/test_images\Directalignment/Directalignment.png')
        try:
            self.locate1 = pyautogui.locateCenterOnScreen(self.path + '/temspy_bot/test_images\Directalignment/Directalignment.png')
            pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.1)
            pyautogui.click()
            self.locate2 = pyautogui.locateCenterOnScreen(self.path + '/temspy_bot/test_images\Directalignment/Diffractionalignment_click.png')
            pyautogui.moveTo(self.locate2[0], self.locate2[1], duration=0.1)

            self.app = Application().connect(title=u'Direct Alignments', timeout=0.5)
            print('connected to Direct Allignments')
        except:
            print('Direct Allignments not found, click Direct Allignments')

        self.window = self.app.Dialog
        self.button_done = self.window.Done

        # self.direct_alignment = self.window.AfxWnd100u

        # self.systreeview = self.window.TreeView
        # self.diffraction_alignment = self.systreeview.get_item([u'Diffraction alignment']) # se e visibile funzioan sempre!!!!



    def bot_start(self, configuration):
        # click direct alignment top band
        pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.01)
        pyautogui.click()
        time.sleep(0.1)

        # click diffraction alignment
        pyautogui.moveTo(self.locate2[0], self.locate2[1], duration=0.01)
        pyautogui.click()

        #self.diffraction_alignment.click_input()
        time.sleep(0.1)
        self.button_done.click_input()
        time.sleep(0.1)
        #print('diffraction alignment reset')

#diffraction_alignment.click_input() # se visibile anche in altre tab lo trovera sempre!!
#button_done.click_input() # done funzionera sempre perche diffraction alignment mette in primo piano la sua finestra!
#direct_alignment.click_input() # settato cosi funziona per cliccare il top_bar della sezione in primo piano ! es clicca STEM Detector(User) ecc ...
