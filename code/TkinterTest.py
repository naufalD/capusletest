import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from scipy.misc import imresize
import os
import threading
import time
import numpy as np

class WindowClass():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Join")
        self.root.geometry("300x300")
        array = np.random.randint(0,255,[32,32,3], np.int32)
        array = imresize(array, [500,500,3])
        self.img = ImageTk.PhotoImage(Image.frombytes("RGB", [500,500], array))
        self.canvas = tk.Canvas(self.root)
        self.canvas.create_image(0,0,image=self.img)
        self.canvas.pack(side = "bottom",fill = "both", expand = "yes")
        self.root.after(2000, self.update)

    def start(self):
        self.root.mainloop()

    def update(self):
        self.canvas.delete('all')
        self.canvas.create_image(0,0,image=self.img)
        self.root.update_idletasks()
        self.root.after(1000, self.update)
    
    def newimg(self, array):
        array = imresize(array, [500,500,3])
        self.img = ImageTk.PhotoImage(Image.frombytes("RGB", [500,500], array))
'''
def changeit():
    time.sleep(5)
    array = np.random.randint(0,255,[32,32,3], np.int32)
    mainwindow.newimg(array)

mainwindow = WindowClass()

t = threading.Thread(target=changeit)
t.start()

mainwindow.start()
'''