import socket
from datetime import datetime
import atexit
import json
import win32com.client
import pythoncom
import threading

class SocketServerClient:
    def __init__(self, mode, host='127.0.0.1', port=8083, temspy=None, tem = None):
        self.tem = tem
        print("guard for tem variable line 12 temspy socket:", self.tem)
        self.mode = mode
        self.host = host
        self.port = port
        self.action_list = ["a", "b", "get_time", "get_stem_beam", "set_stem_beam", "check_configuration",
                            "check_HAADF_position", "click_HAADF", "diff_into_imag", "image_into_diff",
                            "cred_temspy_setup", "cred_temspy_go"]

        self.client_socket = None
        self.server_socket = None
        self.tia_lock = threading.Lock()  # Add a lock for thread safety
        if temspy is True:
            self.connect_temspy()

    def connect_temspy(self):
        try:
            from .temspy_bot.bot_tecnai_temspy_pyFastADT import Bot_TEMpc
        except:
            from temspy_bot.bot_tecnai_temspy_pyFastADT import Bot_TEMpc
        try:
            self.bot = Bot_TEMpc()
            print('Temspy connected correctly!')
        except Exception as err:
            print('unable to connect the temspy:\n', err)

    def disconnect(self):
        if self.client_socket is not None:
            self.client_socket.close()
        if self.server_socket is not None:
            self.server_socket.close()
            del self.tia

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
        # atexit.register(self.client_socket.close)
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
        """Modified client handler to forward cred_temspy requests to CredTemspyServer."""
        pythoncom.CoInitialize()
        tia = win32com.client.Dispatch("ESVision.Application")
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

                if data in ["cred_temspy_setup", "cred_temspy_go"]:
                    # Forward request to the cred_temspy server
                    response = self.forward_to_cred_temspy(data, value)
                    client_socket.sendall(json.dumps(response).encode())
                else:

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
                    elif data == "get_stem_beam":
                        resp = self.get_stem_beam(tia)
                        response = {"get_stem_beam": resp}
                        client_socket.sendall(json.dumps(response).encode())
                    elif data == "set_stem_beam":
                        response = {"set_stem_beam": self.set_stem_beam(tia, value)}
                        client_socket.sendall(json.dumps(response).encode())
                    elif data == "check_configuration":
                        response = {"check_configuration": self.check_configuration(value[0])}
                        client_socket.sendall(json.dumps(response).encode())
                    elif data == "check_HAADF_position":
                        response = {"check_HAADF_position": self.check_HAADF_position()}
                        client_socket.sendall(json.dumps(response).encode())
                    elif data == "click_HAADF":
                        response = {"click_HAADF": self.click_HAADF()}
                        client_socket.sendall(json.dumps(response).encode())
                    elif data == "diff_into_imag":
                        response = {"diff_into_imag": self.diff_into_imag()}
                        client_socket.sendall(json.dumps(response).encode())
                    elif data == "image_into_diff":
                        response = {"image_into_diff": self.image_into_diff(value)}
                        client_socket.sendall(json.dumps(response).encode())
                    # elif data == "cred_temspy_setup":
                    #     response = {"cred_temspy_setup": self.cred_temspy_setup(value[0], value[1], value[2])}
                    #     client_socket.sendall(json.dumps(response).encode())
                    # elif data == "cred_temspy_go":
                    #     response = {"cred_temspy_go": self.cred_temspy_go(eval(value[0]))}
                    #     client_socket.sendall(json.dumps(response).encode())
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

    def get_stem_beam(self, tia):
        """return the beam position in stem as a tuple of 2 elements (x,y). the unit are meters in the camera unit"""
        pos = tia.ScanningServer().BeamPosition
        pos = (pos.X, pos.Y)
        return pos

    def set_stem_beam(self, tia, pos):
        """set the beam position in stem as a tuple of 2 elements (x,y). the unit are meters in the camera unit.
        it return the current position of the beam or an error"""
        try:
            tia.ScanningServer().BeamPosition = tia.Position2D(pos[0], pos[1])
            return self.get_stem_beam(tia)
        except Exception as err:
            return err

    ################################# adding the bots here! ############################################################
    #"check_HAADF_position", "diff_into_imag", "image_into_diff", "cred_temspy"
    def check_configuration(self, value):
        if value in ["spirit", "f30"]:
            self.tem = value
            self.bot.check_configuration(value)
            print("bot configuration selected:", self.bot.configuration)
        else:
            print("not implemented tem configuration:", value)
            print("supported tem are 'spirit' and 'f30' for now")



    def check_HAADF_position(self):
        response = self.bot.check_HAADF_position()
        return response

    def click_HAADF(self):
        response = self.bot.click_HAADF()
        return response

    def diff_into_imag(self):
        self.bot.diff_into_imag()
        return "done"

    def image_into_diff(self, DL):
        self.bot.imag_into_diff(DL)
        return ("done, set DL: %s" % str(DL))

    # def cred_temspy_setup(self, target_angle, velocity, axis):
    #     self.bot.cred_temspy_setup(target_angle, velocity, axis)
    #     return "done"
    #
    # def cred_temspy_go(self, wait = False):
    #     self.bot.cred_temspy_go(wait)
    #     return "done"

    def forward_to_cred_temspy(self, command, value):
        """Forward requests to the CredTemspyServer via a socket."""
        try:
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("debug1")
            #temp_socket.connect(('127.0.0.1', 8084))  # Connect to CredTemspyServer
            temp_socket.connect((self.host, 8084))  # Connect to CredTemspyServer
            print("debug2 connected!, cmd and value", command, value)
            action = json.dumps({command: value})
            temp_socket.sendall(action.encode())
            print("debug3")
            response = temp_socket.recv(1024).decode()
            response = json.loads(response)
            print("debug4, response", response)
            temp_socket.close()
            return response

        except Exception as e:
            print("Error forwarding to CredTemspyServer: {}".format(e))
            return {"error": str(e)}

if __name__ == "__main__":
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)
    #server = SocketServerClient(mode='server', host=IPAddr, port=8083)
    # this should be enabled for the tem pc to be able to connect to the bots
    server = SocketServerClient(mode='server', host=IPAddr, port=8083, temspy=True)
    server.start()
