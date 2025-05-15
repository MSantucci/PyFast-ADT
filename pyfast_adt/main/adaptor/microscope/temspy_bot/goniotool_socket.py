# here the goniotool inspired by instamatic project, thsi is a mod to respect the rules of pyfast-adt,
# based on the skeleton of temspy_socket.
import ctypes
if ctypes.windll.shell32.IsUserAnAdmin():
    print("Python has admin rights.")
else:
    print("Python is NOT running as admin!")

import socket
from datetime import datetime
import atexit
import json
import win32com.client
import pythoncom
import threading

from pywinauto import Application
# from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import time
import pyautogui
import win32gui
import win32con
# import subprocess as sp
# from functools import wraps
# from instamatic import config
# from instamatic.exceptions import TEMCommunicationError, exception_list
# from instamatic.server.serializer import dumper, loader
# from __future__ import annotations

GONIOTOOL_EXE = 'C:\\JEOL\\TOOL\\GonioTool.exe' #the path is correct for darmstadt JEM2100F

# max speed gonio, 1 is around 50 deg/min (1: 0.833 deg/s, if really linear 12: 10 deg/s TO CHECK)
DEFAULT_SPEED = 12

class SocketServerClient_JEOL:
    """this socket specialized for JEOL microscopes handle both client and server communication as a function of the
    mode chosen. based on the structure of temspy_socket.py of pyfast-adt. mainly used to handle speed continuous
    rotation stage movement in JEOL microscopes trough goniotool.exe, but probably in future will be expanded with
    additional bots like it's FEI/TFS counterpart.
    parameters:
        mode: str -> 'server', 'client'. must be one of these 2. if server is used, establish connection with goniotool
        host: str -> '192.168.x.x'
        port: int -> 8083 highly recommended
        tem: 'JEM2100F' just a placeholder for the future
        """

    def __init__(self, mode, host='127.0.0.1', port=8083, tem = None):
        self.tem = tem
        print("guard for tem variable line 14 goniotool socket:", self.tem)
        self.mode = mode
        self.host = host
        self.port = port
        # self.action_list = ["a", "b", "get_time", "get_stem_beam", "set_stem_beam", "check_configuration",
        #                     "check_HAADF_position", "click_HAADF", "diff_into_imag", "image_into_diff",
        #                     "cred_temspy_setup", "cred_temspy_go"]
        self.action_list = ["a", "b", "get_time", "startup", "closedown", "list_f1rate", "list_f1",
                            "click_get_button", "click_set_button", "click_tkb", "click_cmd", "set_rate",
                            "get_rate", "cred_goniotool_setup", "cred_goniotool_go"]
        # self.combobox_dict = {"X": 0, "Y": 1, "Z": 2, "A": 3, "B": 4}
        # self.handle = win32gui.FindWindow(None, 'GonioTool')
        # self.user32 = ctypes.windll.user32
        self.client_socket = None
        self.server_socket = None
        # self.goniotool_lock = threading.Lock()  # Add a lock for thread safety
        if mode is "server":
            self.connect_goniotool()

    def connect_goniotool(self):
        try:
            self.goniotool = Goniotool()
            print('Initialized connection to GonioTool')
        except Exception as err:
            print('unable to connect to Goniotool:\n', err)

    def disconnect(self):
        if self.client_socket is not None:
            self.client_socket.close()
        if self.server_socket is not None:
            self.server_socket.close()
            del self.goniotool

    def start(self):
        if self.mode == 'server':
            self.start_server()
        elif self.mode == 'client':
            self.start_client()
        else:
            raise ValueError("Invalid mode. Use 'server' or 'client'.")

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("Server listening on %s : %s" % (str(self.host), str(self.port)))

        #self.client_socket, addr = self.server_socket.accept()
        #print("Connection from %s" % str(addr))
        try:
            atexit.register(self.goniotool.closedown)
        except:
            print("atexit self.goniotool.closedown failed. remember to reset the speed to the default value 12!")
        while True:
            try:
                self.client_socket, addr = self.server_socket.accept()
                print("Connection from %s" % str(addr))

                # Handle the client in a new thread or process for concurrency
                client_handler_thread = threading.Thread(target=self.handle_client, args=(self.client_socket,))
                client_handler_thread.start()
            except KeyboardInterrupt:
                print("Closing the server")
                break

    def handle_client(self, client_socket):
        """this is the core of the client to server interaction. the server listen only and respond to the client
        whenever a dictionary is sent to it. the server will always answer to the client when the function called is
        finished. in case of success if the function don't have something to return will return the string done,
        otherwise a string containing your answer will be send. the client is allowed to send only a dictionary
        containing 1 item in the following pair of 'cmd_name':value where 'cmd_name' must be part of the methods
        names in self.action_list.     i.e. self.client.client_send_action({"get_time": 0})"""
        pythoncom.CoInitialize()
        # tia = win32com.client.Dispatch("ESVision.Application")
        atexit.register(self.disconnect)
        atexit.register(pythoncom.CoUninitialize)
        try:
            while True:
                data = client_socket.recv(1024).decode()
                data_dict = json.loads(data)
                # decompose the dictionary into key and value to be used
                data = list(data_dict.keys())[0]
                value = data_dict[data]
                print("received cmd", data, "value", value)

                # if data in ["cred_temspy_setup", "cred_temspy_go"]:
                #     # Forward request to the cred_temspy server
                #     response = self.forward_to_cred_temspy(data, value)
                #     client_socket.sendall(json.dumps(response).encode())
                # else:

                if data == "get_time":
                    response = {"get_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "a":
                    response = {"func a": "print a"}
                    # do action a here using try
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "b":
                    response = {"func b": "print b"}
                    # do action b here using try
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "startup":
                    response = {"startup": self.goniotool.startup()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "closedown":
                    response = {"closedown": self.goniotool.closedown()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "list_f1rate":
                    response = {"list_f1rate": self.goniotool.list_f1rate()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "list_f1":
                    response = {"list_f1": self.goniotool.list_f1()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "click_get_button":
                    response = {"click_get_button": self.goniotool.click_get_button()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "click_set_button":
                    response = {"click_set_button": self.goniotool.click_set_button()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "click_tkb":
                    response = {"click_tkb": self.goniotool.click_tkb()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "click_cmd":
                    response = {"click_cmd": self.goniotool.click_cmd()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "set_rate":
                    speed = value
                    response = {"set_rate": self.goniotool.set_rate(speed)}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "get_rate":
                    response = {"get_rate": self.goniotool.get_rate()}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "cred_goniotool_setup":
                    response = {"cred_goniotool_setup": self.goniotool.cred_goniotool_setup(value[0], value[1], value[2])}
                    client_socket.sendall(json.dumps(response).encode())
                elif data == "cred_goniotool_go":
                    response = {"cred_goniotool_go": self.goniotool.cred_goniotool_go(wait = eval(value))}
                    client_socket.sendall(json.dumps(response).encode())
                else:
                    client_socket.send("Invalid command".encode())
                    break
        except KeyboardInterrupt:
            print("closing the server")
        finally:
            client_socket.close()

    def start_client(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))
        atexit.register(self.disconnect)

    def client_send_action(self, action=None):
        action_key = list(action.keys())[0]
        if action_key in self.action_list:
            try:
                action = json.dumps(action)
                self.client_socket.sendall(action.encode())
                print("client send action:", action_key)
                response = self.client_socket.recv(1024).decode()
                response = json.loads(response)
                print("Response from server: %s" % response)
                return response
            except Exception as e:
                print("Response from server: %s" % e)
                return "failed"

########################################################################################################################
#all the functions here are just placeholders of the temspy socket.
# maybe one day i will implement also for jeol is there is the need
########################################################################################################################
    # def get_stem_beam(self, tia):
    #     """return the beam position in stem as a tuple of 2 elements (x,y). the unit are meters in the camera unit"""
    #     pos = tia.ScanningServer().BeamPosition
    #     pos = (pos.X, pos.Y)
    #     return pos
    #
    # def set_stem_beam(self, tia, pos):
    #     """set the beam position in stem as a tuple of 2 elements (x,y). the unit are meters in the camera unit.
    #     it return the current position of the beam or an error"""
    #     try:
    #         tia.ScanningServer().BeamPosition = tia.Position2D(pos[0], pos[1])
    #         return self.get_stem_beam(tia)
    #     except Exception as err:
    #         return err

    ################################# adding the bots here! ############################################################
    #"check_HAADF_position", "diff_into_imag", "image_into_diff", "cred_temspy"
    # def check_configuration(self, value):
    #     if value in ["spirit", "f30"]:
    #         self.tem = value
    #         self.bot.check_configuration(value)
    #         print("bot configuration selected:", self.bot.configuration)
    #     else:
    #         print("not implemented tem configuration:", value)
    #         print("supported tem are 'spirit' and 'f30' for now")

    # def check_HAADF_position(self):
    #     response = self.bot.check_HAADF_position()
    #     return response
    #
    # def click_HAADF(self):
    #     response = self.bot.click_HAADF()
    #     return response
    #
    # def diff_into_imag(self):
    #     self.bot.diff_into_imag()
    #     return "done"
    #
    # def image_into_diff(self, DL):
    #     self.bot.imag_into_diff(DL)
    #     return ("done, set DL: %s" % str(DL))
    #
    # # def cred_temspy_setup(self, target_angle, velocity, axis):
    # #     self.bot.cred_temspy_setup(target_angle, velocity, axis)
    # #     return "done"
    # #
    # # def cred_temspy_go(self, wait = False):
    # #     self.bot.cred_temspy_go(wait)
    # #     return "done"
    #
    # def forward_to_cred_temspy(self, command, value):
    #     """Forward requests to the CredTemspyServer via a socket."""
    #     try:
    #         temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         print("debug1")
    #         #temp_socket.connect(('127.0.0.1', 8084))  # Connect to CredTemspyServer
    #         temp_socket.connect((self.host, 8084))  # Connect to CredTemspyServer
    #         print("debug2 connected!, cmd and value", command, value)
    #         action = json.dumps({command: value})
    #         temp_socket.sendall(action.encode())
    #         print("debug3")
    #         response = temp_socket.recv(1024).decode()
    #         response = json.loads(response)
    #         print("debug4, response", response)
    #         temp_socket.close()
    #         return response
    #
    #     except Exception as e:
    #         print("Error forwarding to CredTemspyServer: {}".format(e))
    #         return {"error": str(e)}

class Goniotool:
    """Interfaces with Goniotool to automate setting the rotation speed on a
    JEOL microscope by adjusting the stepping frequency of the motor. The
    values can be set from 1 to 12, where 12 is maximum speed and 1 is the
    slowest. The speed is linear up the maximum speed, where 1 is approximately
    50 degrees/minute. this is a modded version wrt the one of instamatic,
    adapted to work with pyfast-adt server.
    """

    def __init__(self):
        super().__init__()

        self.app = Application().start(GONIOTOOL_EXE)
        input('Enter password and press <ENTER> to continue...')
        # delay for password (gonio is the password),
        self.startup()
        self.handle = win32gui.FindWindow(None, 'GonioTool')
        self.user32 = ctypes.windll.user32
        self.combobox_dict = {"X": 0, "Y": 1, "Z": 2, "A": 3, "B": 4}


    def startup(self):
        """Initialize and start up the GonioTool interface."""
        self.f1rate = self.app.TMainForm['f1/rate']

        self.f1 = self.app.TMainForm.f1
        self.rb_cmd = self.f1.CMDRadioButton
        self.rb_tkb = self.f1.TKBRadioButton
        self.set_button = self.f1.SetButton
        self.get_button = self.f1.GetButton

        self.click_get_button()
        self.click_cmd()

        self.edit = self.app.TMainForm.f1.Edit7
        return "done"

    def closedown(self):
        """Set default speed and close the program."""
        self.set_rate(DEFAULT_SPEED)
        self.click_tkb()
        time.sleep(1)
        self.app.kill()
        return "done"

    def list_f1rate(self):
        """List GUI control identifiers for `f1/rate` tab."""
        res = self.f1rate.print_control_identifiers()
        return str(res)

    def list_f1(self):
        """List GUI control identifiers for `f1` box."""
        res = self.f1.print_control_identifiers()
        return str(res)

    def click_get_button(self):
        """Click GET button."""
        self.get_button.click()
        return "done"

    def click_set_button(self):
        """Click SET button."""
        self.set_button.click()
        return "done"

    def click_tkb(self):
        """Select TKB radio button."""
        self.rb_tkb.click()
        return "done"

    def click_cmd(self):
        """Select CMD radio button."""
        self.rb_cmd.click()
        return "done"

    def set_rate(self, speed: int):
        """Set rate value for TX."""
        assert isinstance(speed, int), 'Variable `speed` must be of type `int`, is %s' %(str(type(speed)))
        assert 0 < speed <= 12, 'Variable `speed` must have a value of 1 to 12.'

        s = self.edit.select()
        s.set_text(speed)
        self.click_set_button()
        return "done"

    def get_rate(self):
        """Get current rate value for TX."""
        s = self.edit.select()
        val = s.text_block()
        return str(int(val))

    def cred_goniotool_setup(self, value, velocity, axis="A"):
        """goniotool window bot. parameters, axis to move, value and velocity in a.u. from JEOL.
        if alpha is selected the value is in deg??"""

        ######## to check the buttons and edit boxes
        try:
            # self.user32.BlockInput(True)
            self.axis = axis.upper()
            self.value = value # this is the angle to go in compustage
            self.velocity = velocity # this is the velocity to set up

            print('chosen axis %s value %s with speed %s' % (str(self.axis), str(self.value), str(self.velocity)))
            ############################################################################################################
            ## this is the instamatic way
            ############################################################################################################
            # 1 is 0.833 deg/s and they guess is linear so 12 should be 10 deg/s

            velocity_tabulated = {0.83333: 1,  1.66666: 2,
                                  2.49999: 3,  3.33332: 4,
                                  4.16665: 5,  4.99998: 6,
                                  5.83331: 7,  6.66664: 8,
                                  7.49997: 9,  8.3333: 10,
                                  9.16663: 11, 9.99996: 12}

            # rounding to the closest velocity preset usable
            closest_key = min(velocity_tabulated.keys(), key=lambda k: abs(k - velocity))
            closest_vel = velocity_tabulated[closest_key]

            self.set_rate(speed=closest_vel)

            ############################################################################################################
            ## this is the original handle of compustage. we will try before in the instamatic way!
            ############################################################################################################
            # win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
            # win32gui.SetForegroundWindow(self.handle)
            # time.sleep(0.1)
            #
            # self.edit.double_click()
            # time.sleep(0.1)
            # pyautogui.typewrite(str(self.value), interval=0.01)
            # time.sleep(0.33)
            # send_keys('{ENTER}')
            #
            # time.sleep(0.1)
            # self.edit2.double_click()
            # time.sleep(0.1)
            # pyautogui.typewrite(str(self.velocity), interval=0.01)
            # time.sleep(0.33)
            # send_keys('{ENTER}')
            #
            # handle = self.combobox.handle
            # print("combobox handle:", handle)
            # print("self.axis", self.axis)
            #
            # time.sleep(0.1)
            # self.combobox.select(u"%s" % str(self.axis))
            # time.sleep(0.1)
            # try:
            #     win32gui.SetForegroundWindow(self.handle)
            # except:
            #     self.window.minimize()
            time.sleep(0.1)
            # self.user32.BlockInput(False)
            print("finished goniotool setup")
        except Exception as err:
            print("error goniotool setup", err)
            # self.user32.BlockInput(False)

        return closest_key

    def cred_goniotool_go(self, wait=False):
        ############################################################################################################
        ## instamatic way here
        ############################################################################################################
        #code here
        """here look like that goniotool just change the available speed to run the gonio, so set_rate(value = 1-to-12)
        should be enough. moreover the function startup, instantiate the variables of the GUI but also run 2 methods,
        such as self.click_get_button() and self.click_cmd(). tomorrow we need to spot how to use goniotool."""
        print("fake go to angle in JEOL line 444 goniotool_socket.py")
        return "done"

        ############################################################################################################
        ## original method from compustage
        ############################################################################################################
        # # start the rotation here
        # win32gui.ShowWindow(self.handle, win32con.SW_NORMAL)
        # win32gui.SetForegroundWindow(self.handle)
        # self.button.double_click()
        # if wait == True:
        #     self.wait_for_button()



    def wait_for_button(self):
        while True:
            if self.button.is_enabled():
                time.sleep(0.3)
                print("stage free")
                return
            time.sleep(0.1)

if __name__ == "__main__":
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)
    #server = SocketServerClient(mode='server', host=IPAddr, port=8083)
    # this should be enabled for the tem pc to be able to connect to goniotool
    server = SocketServerClient_JEOL(mode='server', host=IPAddr, port=8083)
    server.start()
