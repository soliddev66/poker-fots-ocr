import tkinter as tk
from tkinter import *
import time
from tkinter import messagebox
from tracker import Tracker
import os
import tkinter.font as font

flag = True

def clicked():
    global flag
    if btn['text'] == 'Start Tracker':
        if flag == True:
            x.start()
            flag = False
        else:
            x.resume()
        btn['text'] = 'Stop Tracker'
    else:
        x.pause()
        btn['text'] = 'Start Tracker'

def session_clicked():
    os.startfile(r'log.txt')

def onClose():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        x.stop()
        time.sleep(0.1)
        window.destroy()

if __name__ == "__main__":

    window = Tk()
    window.title("Tracking App")
    window['bg'] = 'black'
    #window.geometry('200x100')
    window.protocol("WM_DELETE_WINDOW", onClose)

    x = Tracker()
    #x.start() #x.start, pause, resume, stop
    buttonFont = font.Font(family='Helvetica', size=16, weight='bold')
    btn = Button(window, text='Start Tracker', font=buttonFont, command=clicked)
    btn.grid(column=0, row=1, padx=15, pady=30)
    btn['bg'] = '#222'
    btn['fg'] = '#FFF'
    btn_session = Button(window, text='Sessions', font=buttonFont, command=session_clicked)
    btn_session.grid(column=1, row=1, padx=15, pady=30)
    btn_session['bg'] = '#222'
    btn_session['fg'] = '#FFF'

    window.mainloop()
