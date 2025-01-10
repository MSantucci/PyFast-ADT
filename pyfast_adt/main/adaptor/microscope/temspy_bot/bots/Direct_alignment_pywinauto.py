import time
from pywinauto.application import Application
from pywinauto.keyboard import send_keys

class Directalignmentpywin_bot:
    def __init__(self):
        try:
            self.app = Application().connect(title=u'Direct Alignments', timeout=3)
            print('connected to Direct Allignments')
        except:
            print('Direct Allignments not found, click Direct Allignments')
            quit()
        self.window = self.app.Dialog
        self.direct_alignment = self.window.AfxWnd100u
        #self.direct_alignment.click_input()
        self.systreeview = self.window.TreeView
        self.diffraction_alignment = self.systreeview.get_item([u'Diffraction alignment']) # se e visibile funzioan sempre!!!!
        self.button_done = self.window.Done

    def bot_start(self):
        time.sleep(0.1)
        self.diffraction_alignment.click_input()
        time.sleep(0.3)
        self.button_done.click_input()
        time.sleep(0.1)
        print('diffraction alignment reset')

#diffraction_alignment.click_input() # se visibile anche in altre tab lo trovera sempre!!
#button_done.click_input() # done funzionera sempre perche diffraction alignment mette in primo piano la sua finestra!
#direct_alignment.click_input() # settato cosi funziona per cliccare il top_bar della sezione in primo piano ! es clicca STEM Detector(User) ecc ...
