# tcp ip socket for sending images to client from server this work combined with scratch_57.py
import socket
import pickle
import struct
import time
import numpy as np
import atexit

class server_img:
    def __init__(self, host='127.0.0.1', port=8089):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(100)
        print("Waiting for a connection...")
        self.image = None
        self.send_img = False
        self.quit = False
        self.client_socket = None

        # self.black_img = np.zeros((512, 512), dtype = np.uint8)
        # self.orig_img = None
        # self.i = 0

    def send_image(self):

        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr}")
            atexit.register(self.client_socket.close)
            atexit.register(self.server_socket.close)
        except Exception as e:
            pass
        # Keep the connection open, waiting for a command to send an image
        i = 0
        while True:
            if self.send_img == True:
                try:

                    # Receive the command from any connected program
                    if self.send_img == True:
                        # Read the image and encode it
                        image_data = pickle.dumps(self.image)  # Serialize the image
                        image_size = len(image_data)  # Get the size of the image data
                        # print(np.max(self.image))
                        # Send the size of the image first
                        self.client_socket.sendall(struct.pack("!I", image_size))

                        # Send the image data in chunks
                        self.client_socket.sendall(image_data)
                        print("Image sent successfully.", i)
                        i += 1
                    elif self.quit == True:
                        print("Quit command received. Closing connection.")
                        break
                    else:
                        print("passing")

                except Exception as e:
                    print(f"Error: {e}")
                    break

                # wait for the feedback from the client
                while True:
                    try:
                        data = self.client_socket.recv(1024)
                        if data == b"done":
                            # print("Client received the image.")
                            # if self.i % 2 == 0:
                            #     self.image = self.black_img.copy()
                            # else:
                            #     self.image = self.orig_img.copy()
                            break
                        elif data == b"disconnected":
                            print("received disconnection")
                            self.client_socket.close()
                    except Exception as e:
                        print(f"Error: {e}")
                        break
                # time.sleep(0.5)
                # self.i += 1
                # msg = pickle.dumps()
                # self.client_socket.sendall(b"image end")
            else:
                if i != 0:
                    i = 0
                else: pass




if __name__ == "__main__":
    import cv2
    import threading
    server = server_img()
    server_thread = threading.Thread(target = server.send_image, daemon = False)
    path_img = r"L:\Marco\datasets\pyfastadt_tracking_test\tracking_precision_02032024\first_test\tracking_precision_std_bad\tracking_images\1_scan\001.tif"
    img = cv2.imread(path_img, flags = -1)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    server.send_img = False
    server.image = img.copy()
    server_thread.start()


    # # debugging funtions to run
    # server.image = np.zeros((512, 512), dtype=np.uint8)
    # for a in range(50):
    #     for i in range(254):
    #         server.image[:, :] += 1
    #     for i in range(254):
    #         server.image[:, :] -= 1

