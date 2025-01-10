import time
from pywinauto.application import Application 
import os
import pyautogui
import PIL

class Stem_detector_bot:
    """"this bot is designed to connect to the STEM Detector (User) and to enable the insert detector (haadf) which is not allowed by scripting"""
    def __init__(self):
        self.HAADF_pos = None
        self.path = os.getcwd()
        self.bot_target_tab1 = u'STEM' # where is placed the clicker for the insert detector
        self.bot_target_tab2 = u'Setup' # where is the page that you want to return everytime

        try:
            #enable workset into STEM Detector (User)
            self.app0 = Application().connect(title=u' Workset', timeout = 0.5) ## check if that space is blocking the work of the bot
            print("connected to Tecnai UI -> Workset!")
        except:
            print('Tecnai UI not found, please open Tecnai User Interface, STEMDetector.py line 18')
            
        try:
            self.window0 = self.app0.Workset.child_window(title="Tab1").wrapper_object()
            print("connected to STEM tab!")
        except:
            print('please rename a workset tab into: STEM\nand put inside STEM Detector (User) from workspace layout and direct alignments')
            print('quit code')
            quit()

    def bot_start(self, configuration):
        if configuration == 'spirit':
            self.window0.select(self.bot_target_tab2)
            time.sleep(0.1)
            self.window0.select(self.bot_target_tab1) # maybe change it to STEM to let work also the direct allignments!
            time.sleep(0.1)

            #enable insert Detector ####################################################### can i put connect STEM Detector in the initialization?
            self.app1 = Application().connect(title=u'STEM Detector (User)', timeout=0.5)
            self.window1 = self.app1.Dialog.child_window(title="STEM Detector (User)_Main")
            self.detector = self.app1.Dialog.child_window(title=u'insert detector', class_name="Button").wrapper_obecm34ject()
            #[u'Button', u'Button0', u'Button1', u'ins. HAADF', u'ins. HAADFButton']

            self.detector.click_input()
            time.sleep(0.1)

        elif configuration == 'f30':
            self.window0.select(self.bot_target_tab2)
            time.sleep(0.1)
            self.window0.select(
                self.bot_target_tab1)  # maybe change it to STEM to let work also the direct allignments!
            time.sleep(0.1)

            # enable insert Detector ####################################################### can i put connect STEM Detector in the initialization?
            self.app1 = Application().connect(title=u'STEM Detector (Expert)', timeout=0.5)
            self.window1 = self.app1.Dialog.child_window(title="STEM Detector (Expert)_Main")
            self.detector = self.app1.Dialog.child_window(title=u'ins. HAADF', class_name="Button").wrapper_object()
            # [u'Button', u'Button0', u'Button1', u'ins. HAADF', u'ins. HAADFButton']

            self.detector.click_input()
            time.sleep(0.1)
        #window = app.Dialog.print_control_identifiers()
        #button = app.Dialog.child_window(title="Insert detectors", class_name="Button").wrapper_object()
        #button.click_input()

    def HAADF_position(self, configuration):
        if configuration == 'spirit':
            self.window0.select(self.bot_target_tab2)
            time.sleep(0.1)
            self.window0.select(self.bot_target_tab1)  # maybe change it to STEM to let work also the direct allignments!
            time.sleep(0.1)
            try:
                self.locate1 = pyautogui.locateOnScreen(self.path+'/temspy_bot/test_images\HAADF_spirit/insert_detector_gray_small.png')
                pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.1)
                self.HAADF_pos = 'OUTSIDE'
                return self.HAADF_pos
            except TypeError:
                #print('gray detector object not found')
                try:
                    self.locate2 = pyautogui.locateOnScreen(self.path+'/temspy_bot/test_images\HAADF_spirit/insert_detector_yellow_small.png')

                    pyautogui.moveTo(self.locate2[0], self.locate2[1], duration=0.1)
                    self.HAADF_pos = 'INSIDE'
                    return self.HAADF_pos
                except TypeError:
                    #print('yellow detector object not found')
                    print('impossible determine where the HAADF is, please check that Tecnai UI is visible in the desktop')
                    self.HAADF_pos = 'UNKNOWN'

            return self.HAADF_pos

        if configuration == 'f30':
            self.window0.select(self.bot_target_tab2)
            time.sleep(0.1)
            self.window0.select(
                self.bot_target_tab1)  # maybe change it to STEM to let work also the direct allignments!
            time.sleep(0.1)
            try:
                self.locate1 = pyautogui.locateOnScreen(self.path + '/temspy_bot/test_images\HAADF_f30/insert_detector_gray_small.png')
                pyautogui.moveTo(self.locate1[0], self.locate1[1], duration=0.1)
                self.HAADF_pos = 'OUTSIDE'
                return self.HAADF_pos
            except TypeError:
                # print('gray detector object not found')
                try:
                    self.locate2 = pyautogui.locateOnScreen(
                        self.path + '/temspy_bot/test_images\HAADF_f30/insert_detector_yellow_small.png')

                    pyautogui.moveTo(self.locate2[0], self.locate2[1], duration=0.1)
                    self.HAADF_pos = 'INSIDE'
                    return self.HAADF_pos
                except TypeError:
                    # print('yellow detector object not found')
                    print(
                        'impossible determine where the HAADF is, please check that Tecnai UI is visible in the desktop')
                    self.HAADF_pos = 'UNKNOWN'

            return self.HAADF_pos
# sembra tutto regolare! pronto per essere testato in live,
# worksetnon puo essere coperto altrimenti non funziona!

# per risolvere il problema editare una sezione come bot e mettere come ultimo
# blocco STEM Detector (User)
# e poi far tornare tutto su STEM tab

# issue1: tutti i blocchi chiamati nello stesso modo sono classi identiche.
# questo vuol dire che STEM Imaging block e STEM Detector block vengono chaimati dalla stessa classe
# l'unico modo per differenziarli e in funzione di chi e selezionato come attivo al momento
# easy solut. generare un nuovo tab (o rinominare uno) come bot dove c'e solo STEM detector

# per ora sembra tutto funzionare proviamo a rilasciarlo e vedere che bug ci sono nel metodo!
