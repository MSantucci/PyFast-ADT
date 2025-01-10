# using shared memory to pass a live feed to other processes

import cv2
import numpy as np
from multiprocessing import shared_memory, Process
import time
import atexit

class camera_server:
    def __init__(self, shared_mem_name = "camera_stream", frame_shape = (512, 512)):
        self.shared_mem_name = shared_mem_name
        self.frame_shape = frame_shape
        self.frame = np.zeros(frame_shape, dtype=np.uint8) + 254
        self.black_frame = np.zeros(frame_shape, dtype=np.uint8)
        self.initialize_server()

    def initialize_server(self):
        # Create shared memory block
        self.shm = shared_memory.SharedMemory(name=self.shared_mem_name, create=True, size=int(np.prod(self.frame_shape)))
        print("Shared Memory Created:", self.shm.name)

        # Create a numpy array backed by shared memory
        self.frame_buffer = np.ndarray(self.frame_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Register cleanup for when the server exits
        atexit.register(self.cleanup)

    def cleanup(self):
        """ Clean up the shared memory """
        print("Cleaning up shared memory...")
        self.shm.close()
        self.shm.unlink()

    def update_buffer(self, frame):
        self.frame_buffer[:, :] = frame

    def start_simulation(self):
        """ Simulate streaming by continuously reading shared memory """
        print("Streaming frames to shared memory...")
        i = 0
        try:
            while True:
                if i % 2 == 0:
                    # Write the frame data to shared memory
                    self.frame_buffer[:, :] = self.frame
                else:
                    self.frame_buffer[:, :] = self.black_frame
                i += 1
                time.sleep(0.03)  # ~30 FPS delay
                print(np.max(self.frame_buffer))
        except KeyboardInterrupt:
            self.frame_buffer[:, :] = self.black_frame
            print("exception")


if __name__ == "__main__":
    # Define shared memory name and frame shape
    SHARED_MEM_NAME = "camera_stream"
    FRAME_SHAPE = (512, 512)  # Height x Width

    server = camera_server(SHARED_MEM_NAME, FRAME_SHAPE)
    server.initialize_server()

    # p1 = Process(target=server.start_streaming)
    # atexit.register(p1.terminate)
    # p1.start()

    path_img = r"L:\Marco\datasets\pyfastadt_tracking_test\tracking_precision_02032024\first_test\tracking_precision_std_bad\tracking_images\1_scan\001.tif"
    img = cv2.imread(path_img, flags=-1)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.resize(img, (512, 512))
    # server.update_buffer(img)

    # for a in range(300):
    #     if a % 2 == 0:
    #         server.update_buffer(img)
    #     else:
    #         server.update_buffer(server.black_frame)
    #     time.sleep(0.03)


