import DigitalMicrograph as DM
DM.ClearResults()
from tkinter import *
import os
import sys
print(sys.path)
sys.path.append(r'C:\Users\remote\Desktop\test_temscript_from_digit_micrograph')
#os.chdir(r'C:\Users\remote\Desktop\test_temscript_from_digit_micrograph')
print(os.getcwd())
#from temscript_remote_test import beamshift_test
try:
    from temscript_remote_test import beamshift_test
    #import beamshift_test
except Exception as err:
    print('exception: ',err)
# add widgets here
window=Tk(baseName = 'test')
window.title('test_tkinter')
window.geometry("350x100")
label = Label(window, text = 'this is a simple script to test tkinter in \n \
in digital micrograph and see if can work in background').pack()
button = Button(window, text = 'start', command = beamshift_test).pack()


window.mainloop()
print('done')