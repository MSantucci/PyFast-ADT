import time
from pywinauto.application import Application
from pywinauto import ElementNotFoundError
from pywinauto.keyboard import send_keys

class Temspy_bot:
    def __init__(self):
        try:
            self.app = Application().connect(title=u'Temspy - [Temspy1]',timeout = 0.5)
            print('connected to Temspy')
        except:
            print('temspy not found, opening temspy')
            self.app = Application().start(cmd_line=u'"temspy.exe"')
        self.temspy = self.app[u'Temspy - [Temspy1]']
        self.temspy.wait('ready')
        self.menu_item1 = self.temspy.menu_item(u'&Dialogues->&Outputs')

    def bot_start(self):
        self.menu_item1.click_input()
        time.sleep(0.1)
        print('outputs open')
        self.temspy.minimize()
        self.temspy.close()

#connect to Temspy in order to open outputs

# need to be tested

# i'm not able to control the single exception i will try in future 

