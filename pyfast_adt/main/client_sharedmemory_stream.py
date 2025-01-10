# using shared memory to receive a live feed from other processes
import cv2
import numpy as np
from multiprocessing import shared_memory, Process
import atexit

def camera_client(shared_mem_name, frame_shape):
    # Connect to the existing shared memory
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    print("Connected to shared memory:", shm.name)

    # Create a numpy array backed by shared memory
    frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

    try:
        print("Displaying frames from shared memory...")
        while True:
            # Read the latest frame from shared memory
            frame = frame_buffer.copy()  # Copy to avoid race condition

            # Display the frame
            cv2.imshow("Live-Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Client stopped")
    finally:
        shm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Shared memory name and frame shape (must match server)
    SHARED_MEM_NAME = "camera_stream"
    FRAME_SHAPE = (512, 512)  # Height x Width x Channels
    p1 = Process(target=camera_client, args=(SHARED_MEM_NAME, FRAME_SHAPE))
    atexit.register(p1.terminate)
    p1.start()
    # camera_client(SHARED_MEM_NAME, FRAME_SHAPE)
