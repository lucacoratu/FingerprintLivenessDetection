from future.moves import tkinter
from future.moves.tkinter import messagebox
from LivenessDet import LivenessDetectionModel
from concurrent.futures import ThreadPoolExecutor
from future.moves.tkinter import ttk

class Application:
    window = None
    model = LivenessDetectionModel()
    def __init__(self):
        if self.window != None:
            raise Exception('Application gui is already initialized!')
        self.window = tkinter.Tk()
        self.window.geometry('800x600')
        self.window.title('Liveness Fingerprint Detection')
        self.window.configure(bg='#333')

        #Greeting label
        lbl = tkinter.Label(self.window, fg='white', text='Welcome to the fingerprint liveness detection', bg='#333', font=('Arial Bold', 16))
        lbl.grid(column=0, row=0, columnspan=3)

        #Label for features entry
        lbl_entry = tkinter.Label(self.window, fg='white', text='Insert filename for features', bg='#333', font=('Arial Bold', 12))
        lbl_entry.grid(row=1, column=0)
        #Input for the filename for where the training data should be stored
        entryFeatures = tkinter.Entry(self.window)
        entryFeatures.grid(row=1, column=1)

        #Label for classifications entry
        lbl_classifications = tkinter.Label(self.window, fg='white', text='Insert filename for classifications', bg='#333', font=('Arial Bold', 12))
        lbl_classifications.grid(row=2, column=0)
        #Input for the filename for where the training data should be stored
        entryClassifications = tkinter.Entry(self.window)
        entryClassifications.grid(row=2, column=1)

        def StartTestingButtonClicked():
            #Verify that the inputs are not empty
            if entryFeatures.get() == '' or entryClassifications.get() == '':
                messagebox.showerror('Empty input', 'Cannot leave the filename empty')
                return

            #Start the training operation
            with ThreadPoolExecutor() as executor:
                executor.submit(self.model.TrainModel)

            pb1 = ttk.ProgressBar(self.window, orient='horizontal', mode='indeterminate')
            pb1.grid(row=3, column=0, columnspan=2)

            #The model finished training
            print('Training finished')


        #Button for training
        btn_start_test = tkinter.Button(self.window, text='Start training', command=StartTestingButtonClicked)
        btn_start_test.grid(row=3, column=0, columnspan=2)



    def Run(self):
        self.window.mainloop()


        #initialize the window for the gui