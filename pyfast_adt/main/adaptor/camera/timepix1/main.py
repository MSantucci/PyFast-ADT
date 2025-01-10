from CServer import CServer
from CCameraInterface import CCameraInterface



def main():

    tpxDetectorID = 0
    myCamera = CCameraInterface(tpxDetectorID)
    myServer = CServer(myCamera)

    myServer.core()







if __name__ == "__main__":
    main()

