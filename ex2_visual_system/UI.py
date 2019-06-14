'''
Simulation of a retina implant. 

Run file Ex_2_Visual.py

This file defines the UI of Ex2_Visual_utils.py
Detailed proceedings are in the Ex2_Visual_utils.py file

Authors: Jan Wiegner, Diego Gallegos Salinas, Philippe Blatter
Version: 6
Date: 27.04.2019
'''


import os
from tkinter import *
from tkinter import filedialog, messagebox


class UI:
    def __init__(self, main_func):
        '''
        Constructor function that starts the GUI.

        Arguments:
            - main_func: function that should be started when running Ex_2_Visual.py
        '''
        self.root = Tk()
        self.main_func = main_func
        self.entries = []
        self.in_path = ''
        self.run_UI()

    def start_simulation(self, verbose):
        '''
        Starts the actual main function with the correct parameters.

        Arguments: 
            - verbose: Boolean that indicates if computation details should be displayed or not.
        '''
        filename = self.entries[0].get()
        self.main_func(filename, verbose)


    def choose_image_file(self):
        '''
        Function that lets the user choose the image file. 
        '''
        filename =  filedialog.askopenfilename(initialdir = "./Images/",title = "Select file",
                                            filetypes = (("image files","*.jpg *.png *.tif *.bmp"),("all files","*.*")))

        self.entries[0].delete(0, END)
        self.entries[0].insert(END, filename)
    

    def close_window(self):
        '''
        Function that lets the user close the GUI and terminates the computations. 
        '''
        exit()

    def run_UI(self):
        '''
        Builds the user interface and defines which actions to take when pressing the different buttons. 
        '''
        
        inputs = ["Input File", "Verbose"]
        for row, input in enumerate(inputs):
            Label(self.root, text=input).grid(row=row)

        default_values = [""]
        for row, value in enumerate(default_values):
            e = Entry(self.root)
            e.grid(row=row, column=1, columnspan=2)
            e.insert(END, value)
            self.entries.append(e)

        
        Button(self.root, text="Choose Image", width=20, command=self.choose_image_file).grid(row=0, column=3)

        verbose_Var = BooleanVar()
        Radiobutton(self.root, variable=verbose_Var, value=True, text="Verbose").grid(row=1, column=1)
        Radiobutton(self.root, variable=verbose_Var, value=False, text="Non-verbose").grid(row=1, column=2)

        Button(self.root, text="Run Simulation", command=lambda: self.start_simulation(verbose_Var.get())).grid(row=3, column=1)
        
        Button(self.root, text="Quit", width=5, command=self.close_window).grid(row=4, column=0)
        

        self.root.mainloop()



