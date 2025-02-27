import socket
import threading
import json

class CredTemspyServer:
    """Dedicated server for handling cred_temspy commands in a separate thread."""

    def __init__(self, host='127.0.0.1', port=8084):
        self.host = host
        self.port = port
        self.server_socket = None
        self.connect_temspy()
        self.start()

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

    def start(self):
        """Start the server in a separate thread."""
        threading.Thread(target=self.run_server).start()

    def run_server(self):
        """Server loop to handle cred_temspy commands."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("CredTemspyServer listening on {}:{}".format(self.host, self.port))

        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                print("CredTemspyServer: Connection from {}".format(addr))
                threading.Thread(target=self.handle_client, args=(client_socket,)).start()
            except KeyboardInterrupt:
                print("Closing the server")
                break

    def handle_client(self, client_socket):
        """Handle requests from the main server."""
        try:
            while True:
                data = client_socket.recv(1024).decode()
                data_dict = json.loads(data)
                command = list(data_dict.keys())[0]
                value = data_dict[command]

                print("CredTemspyServer received: {}, value: {}".format(command, value))

                # Process the command using the parent server's methods
                if command == "cred_temspy_setup":
                    response = {"cred_temspy_setup": self.bot.cred_temspy_setup(value[0], value[1], value[2])}
                elif command == "cred_temspy_go":
                    response = {"cred_temspy_go": self.bot.cred_temspy_go(eval(value[0]))}
                else:
                    response = {"error": "Invalid command"}

                client_socket.sendall(json.dumps(response).encode())

        except Exception as e:
            print("CredTemspyServer Error: {}".format(e))
        finally:
            client_socket.close()

if __name__ == "__main__":
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)
    # server = SocketServerClient(mode='server', host=IPAddr, port=8083)
    # this should be enabled for the tem pc to be able to connect to the bots
    server = CredTemspyServer(host=IPAddr, port=8084)
