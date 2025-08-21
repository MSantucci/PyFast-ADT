import tkinter
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import temscript
import time
from .bots import Outputs_DLchange, STEMDetector, Temspy_into_Outputs, Direct_alignment, Compustage_Tad, Stage_TUI
import os
import pyautogui
import PIL
try:
    from bots.Direct_alignment import Directalignment_bot
    from bots.Direct_alignment_pywinauto import Directalignmentpywin_bot
    from bots.Outputs_DLchange import Outputs_bot
    from bots.STEMDetector import Stem_detector_bot
    from bots.Compustage_Tad import Compustage_bot
    from bots.Temspy_into_Outputs import Temspy_bot

    from bots.Stage_TUI import StageTUI_bot
except: print('bots import error')
class Bot_TEMpc:
    ######## to mod in order to be able to control the majority of the stuff to work
    ######## the user need just to put the temspy_bot_folder somewhere and to run the server, the rest will be done by pyFastADT
    def __init__(self, blogtest2=None):
        self.blogtest2 = blogtest2
        self.DL = None
        self.error = None
        self.configuration = None
        # try:
        #     self.microscope = temscript.Microscope()
        # except:
        #     print('\n')
        #     print("####\nit's not possible to connect to the temspy bot, call marco!\n###")
        #     print('\n')
        self.inizialize()

    def inizialize(self):
        try:
            # connect to Dialogues
            """"this bot is designed to conenct to the outputs window of temspy and enter a new value for the DL (diffraction lens)"""
            self.outputs_bot = Outputs_DLchange.Outputs_bot()
            print('Outputs connected')
        except:
            self.error = "###############################################################\nSomething wrong.\nPlease, check that you have Outputs (temspy) tab open and visible and try again!\n###############################################################"
            print('\n', self.error, '\n')
        try:
            # connect to Tecnai UI, STEM Detector (User)
            """"this bot is designed to connect to the STEM Detector (User) and to enable the insert detector (haadf) which is not allowed by scripting"""
            self.stem_detector_bot = STEMDetector.Stem_detector_bot()
            print('STEM Detector (User) connected')
        except:
            self.error = "###############################################################\nSomething wrong.\nPlease, check that you have stem detector (user) tab open and visible and try again!\n###############################################################"
            print('\n', self.error, '\n')
        try:
            # connect to Tecnai UI, Direct Allignment
            """this bot is designed to connect to the Direct Allignments and to press diffraction alignment button to reset the diffraction shift"""
            self.diffraction_alignment_bot = Direct_alignment.Directalignment_bot()  # maybe to let it work is necessary to do some arrangements!!!
            print('Direct Alignment connected')
        except:
            self.error = "###############################################################\nSomething wrong.\nPlease, check that you have Direct allignment tab open and visible and try again!\n###############################################################"
            print('\n', self.error, '\n')
        try:
            self.compustage_bot = Compustage_Tad.Compustage_bot()
            print('Compustage connected')
            print('Temspy Bot ready to work!')
        except:
            self.error = "###############################################################\nSomething wrong.\nPlease, check that you have Compustage (temspy) tab open and visible and try again!\n###############################################################"
            print('\n', self.error, '\n')

        try:
            self.stagetui_bot = Stage_TUI.StageTUI_bot()
            print('stage tui connected')
            print('Stage TUI Bot ready to work!')
        except:
            self.error = "###############################################################\nSomething wrong.\nPlease, check that you have Stage in TUI flap open and visible and try again!\n###############################################################"
            print('\n', self.error, '\n')
    #def inizialize(self):
    #     #try: # connect to dialogues and Tecnai UI
    #    try: # connect to Dialogues
    #        self.outputs_bot = Outputs_DLchange.Outputs_bot()
    #    except: # if not connect to Temspy
    #        #try:
    #        #    print('opening temspy.exe into outputs')
    #        #    self.temspy_bot = Temspy_into_Outputs.Temspy_bot.bot_start()
    #        #    print('restart initialization')
    #        #    self.inizialize()
    #        #except:
    #        self.error = "Something wrong.\nPlease, check that you have Dialogues open and visible"
    #        print(self.error)
            
        
        #try:
        #    self.stem_detector_bot = STEMDetector.Stem_detector_bot()
        #    print('STEM Detector (User) connected')
        #    self.diffraction_alignment_bot = Direct_alignment.Directalignment_bot() # maybe to let it work is necessary to do some arrangements!!!
        #    print('Direct Alignment connected')
        #except:
        #    print('open Tecnai User Interface please!')
        #    self.error = "Something wrong.\nPlease, check that you have Dialogues and Tecnai User Interface open and visible"
        #    print(self.error)
        #     
         #except:
          #   self.error = "Something wrong.\nPlease, check that you have Dialogues and Tecnai User Interface open and visible"
           #  print(self.error)
            # return self.error

    # def lift_screen(self):
    #     if self.microscope.get_screen_position() == 'DOWN':
    #         print('lifting the screen')
    #         self.microscope.set_screen_position(mode = 'UP')
    #     else: print('the screen is already up')
    #
    # def close_screen(self):
    #     if self.microscope.get_screen_position() == 'UP':
    #         print('closing the screen')
    #         self.microscope.set_screen_position(mode='DOWN')
    #     else:
    #         print('the screen is already closed')

    def check_configuration(self, value):
        self.configuration = value

    def check_HAADF_position(self):
        self.status = self.stem_detector_bot.HAADF_position(self.configuration)
        print('Detector:'+str(self.status))
        return self.status

    def click_HAADF(self):
        self.stem_detector_bot.bot_start(self.configuration)
        print('HAADF clicked')
    # def reset_projector(self):
    #     # up to now i will do it using 2 times the diffraction button
    #     self.modes = self.microscope.get_projection_mode()
    #
    #     if self.modes == "IMAGING":
    #         self.microscope.set_projection_mode(mode="DIFFRACTION")
    #
    #     if self.modes == "DIFFRACTION":
    #         self.microscope.set_projection_mode(mode="IMAGING")
    #         self.microscope.set_projection_mode(mode="DIFFRACTION")

    def diff_into_imag(self): # e1
        print('\n')
        # self.close_screen()
        if self.check_HAADF_position() == 'OUTSIDE':
            self.stem_detector_bot.bot_start(self.configuration)
        #self.stem_detector_bot.bot_start()
        time.sleep(0.3)
        self.diffraction_alignment_bot.bot_start(self.configuration)
        time.sleep(0.3)
        # self.reset_projector()
        print('imaging mode')
        print('\n')

        ## screen down (if check)
        ## HAADF in
        ## Diffraction off
        ## Diffraction on
        # Diffraction allignment click (Tecnai UI)
        # Direct allignment Done button!

    def imag_into_diff(self, DL): # e2
        print('\n')
        # self.lift_screen()
        if self.check_HAADF_position() == 'INSIDE':
            self.stem_detector_bot.bot_start(self.configuration)
        #self.stem_detector_bot.bot_start()
        self.DL = DL
        self.outputs_bot.bot_start(self.configuration, self.DL)
        print('diffraction mode')
        print('\n')
        # lift the screen up
        # HAADF out
        # input DL value (BOT)

    def cred_temspy_setup(self, target_angle, velocity, axis):
        self.compustage_bot.bot_setup(self.configuration, target_angle, velocity, axis)

    def cred_temspy_go(self, wait=False):
        self.compustage_bot.bot_start(self.configuration, wait)

    def stage_tui_setup(self, axis):
        self.stagetui_bot.bot_setup(axis)
        pass

    def stage_tui_go(self, wait = False):
        self.stagetui_bot.bot_start(wait)
        pass
