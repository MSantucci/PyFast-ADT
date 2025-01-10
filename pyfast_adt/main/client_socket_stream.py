# tcp ip socket for sending images to client from server this work combined with scratch_56.py
import socket
import atexit
import sys
import numpy as np
import pickle
import struct
import matplotlib.pyplot as plt
import time
import requests
import tkinter as tk
from tkinter.messagebox import showinfo
import threading
import queue
from PIL import Image, ImageTk
from io import BytesIO
import cv2


def receive_img(queue, stop_event, root, host='127.0.0.1', port=8089):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        atexit.register(release_server, client_socket, root)
        root.protocol("WM_DELETE_WINDOW", lambda arg=(client_socket, root): release_server(arg[0], arg[1]))
        print("Connected to server.")

        while True:
            # Receive the size of the incoming image
            size_header = client_socket.recv(4)
            if not size_header:
                print("Connection closed by the server.")
                break

            image_size = struct.unpack("!I", size_header)[0]
            image_data = b""
            while len(image_data) < image_size:
                packet = client_socket.recv(4096)
                if not packet:
                    print("Connection closed while receiving image.")
                    return
                image_data += packet

            # Deserialize the image and add it to the queue
            image = pickle.loads(image_data)
            if queue.qsize() > 2:
                queue.queue.clear()
            queue.put(image)

            if stop_event.is_set():
                print("received_stop_event")
                break

            # Acknowledge the server
            client_socket.sendall(b"done")
    except Exception as e:
        print("Error:", e)


def receive_img_request(queue, stop_event, destination = "image", host='127.0.0.1', port=8089):
    try:
        from serval_toolkit.camera import Camera as ServalCamera
    except:
        from .serval_toolkit.camera import Camera as ServalCamera

    serval_url = "tcp://127.0.0.1:8089"
    conn = ServalCamera()
    conn.connect(serval_url)

    while True:
        try:
            db = conn.dashboard
            if db['Measurement']['Status'] == 'DA_RECORDING':
                response = conn.get_request('/measurement/'+str(destination))
                image = Image.open(BytesIO(response.content))
                img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img = shuffle_chip_order(img, order=[1, 2, 3, 4])
                if queue.qsize() > 2:
                    queue.queue.clear()
                queue.put(img)
        except Exception as e:
            print("Error:", e)

        if stop_event.is_set():
            print("received_stop_event")
            break

def update_image(label, queue):
    # atexit._run_exitfuncs() # debugger
    try:
        if not queue.empty():
            # print("queue size", queue.qsize())
            image = queue.get_nowait()
            image = Image.fromarray(image)
            tk_image = ImageTk.PhotoImage(image)
            label.config(image=tk_image)
            label.image = tk_image
            # print("image replaced", np.max(image), queue.qsize())
    except Exception as e:
        print("Error in UI update:", e)
        tk.Tk.quit()


    after_id = label.after(10, update_image, label, queue)

def main(source = "preview"):
    # Create a queue for communication between threads
    image_queue = queue.Queue()
    global thread
    global root
    stop_event = threading.Event()

    # Create the Tkinter UI
    root = tk.Tk()
    root.title("Live Image Display")
    image_label = tk.Label(root)
    image_label.pack()

    if source == "socket":
        # Start the image receiving thread
        thread = threading.Thread(target=receive_img, args=(image_queue, stop_event, root), daemon=False)
        thread.start()
    else:
        thread = threading.Thread(target=receive_img_request, args=(image_queue, stop_event, source), daemon=False)
        thread.start()

    # atexit.register(print_stop_event, stop_event, thread)
    atexit.register(stop_event.set)
    # atexit.register(root.destroy)
    atexit.register(root.quit)

    # Start updating the UI
    update_image(image_label, image_queue)

    root.mainloop()

def release_server(client_socket, root):
    # client_socket.sendall(b"disconnected")
    client_socket.close()
    root.quit()

def print_stop_event(stop_event, thread):
    print("i'm here")
    time.sleep(1)
    showinfo("stop live view", "stop_triggered ? %s, thread_alive ? %s" % (str(stop_event.is_set()), str(thread.is_alive())))

def shuffle_chip_order(raw, order = [1, 2, 3, 4]):
    """Shuffle the chips in the image based on the given order.

        Parameters:
        - raw: np.ndarray, Input 512x512 image.
        - order: list of int, Desired order of chips."""

    img_data = np.empty((512, 512), dtype=raw.dtype)

    dict_order = {1: raw[0:256, 0:256],
                  2: raw[0:256, 256:512],
                  3: raw[256:512, 0:256],
                  4: raw[256:512, 256:512]}

    img_data[0:256, 0:256] = dict_order[order[0]]
    img_data[0:256, 256:512] = dict_order[order[1]]
    img_data[256:512, 0:256] = dict_order[order[2]]
    img_data[256:512, 256:512] = dict_order[order[3]]


    corrected_image = img_data.copy()

    return corrected_image

if __name__ == "__main__":
    main(source = "socket")
    # main(source = "preview")


    # atexit._run_exitfuncs()




