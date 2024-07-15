import tkinter as tk
from tkinter import ttk
from tkinter import *

OPTIONS = [
    "egg",
    "bunny",
    "chicken"
]

master = tk.Tk()

variable = StringVar(master)
variable.set(OPTIONS[0]) # default value
master.geometry("200x200")

ttk.Combobox(master, textvariable=variable, values=OPTIONS, state='readonly').pack(fill='x', side='top', padx='5', pady='5')

mainloop()

 