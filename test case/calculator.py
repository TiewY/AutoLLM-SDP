import tkinter as tk
from tkinter import messagebox

mainWindow = tk.Tk()
mainWindow.title("Calculator")

First = tk.Label(mainWindow,text="Enter Your First Number",pady=(10))
First.pack()

First_Value = tk.Entry(mainWindow)
First_Value.pack()

Second = tk.Label(mainWindow,text="Enter Your Second Number",pady=(10),padx=10)
Second.pack()

Second_Value = tk.Entry(mainWindow)
Second_Value.pack()


    
button = tk.Button(mainWindow,text="+",
                   command = lambda:Add())
button.pack()

button = tk.Button(mainWindow,text="-",
                   command = lambda:Sub())
button.pack()

button = tk.Button(mainWindow,text="*",
                   command = lambda:Mul())
button.pack()

button = tk.Button(mainWindow,text="/",
                   command = lambda:Div())
button.pack()

result_label=tk.Label(mainWindow,text="Operation Result is : ")
result_label.pack()

def Add():
    a=int(First_Value.get())
    b=int(Second_Value.get())
    add=a+b
    result_label.config(text = "Operation Result is : " +str(add))
    
    
    
    
def Sub():
    a=int(First_Value.get())
    b=int(Second_Value.get())
    Sub=a-b
    result_label.config(text = "Operation Result is : " +str(Sub))
    
    
def Mul():
    a=int(First_Value.get())
    b=int(Second_Value.get())
    Mul=a*b
    result_label.config(text = "Operation Result is : " +str(Mul))
    
def Div():
    a=int(First_Value.get())
    b=int(Second_Value.get())
    try:
        Div=a/b
        
    except:
        messagebox.showerror("Error", "Zero Division Error. Please Enter Non Zero value in denominator.")
    
    result_label.config(text = "Operation Result is : " +str(Div))

mainWindow.mainloop()
    