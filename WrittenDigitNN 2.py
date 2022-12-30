import numpy as np
from os import *
import csv as csv
import pandas as pd
from tkinter import *
import numpy as np
from math import *
from random import *

data = pd.read_csv("MNIST_CSV/mnist_train.csv", header=None)
test_data = pd.read_csv("MNIST_CSV/mnist_test.csv", header=None)
data


root = Tk()
r = Canvas(root, width=327, height=400, bg='black')
r.pack()

curr_grid = np.zeros(784)
curr_id = 60000

engaged = False
pix_size = 10

disp_pred = ""

x_margin, y_margin = 15, 25

def click(event):
    global engaged
    #print(engaged)
    if engaged:
        engaged = False
    else:
        engaged = True

#data.append(pd.DataFrame([0 for i in range(784)]), ignore_index=True)
#data.append(pd.Series([0 for i in range(785)], index=data.columns, name=str(curr_id)))
data.loc[curr_id, :] = [0 for i in range(785)]
data = data.astype(int)
#new_row = pd.DataFrame([[int(0) for i in range(785)]], index='60000')
        
def motion(event):
    global engaged
    global pix_size
    x, y = event.x - x_margin, event.y - y_margin
    if engaged:
        #print('{}, {}'.format(y, x))
        display(curr_id)
        row = get_sample(data, curr_id)
        #print(row[300])
        cursor_row = y//pix_size
        cursor_col = x//pix_size
        tgt = cursor_row*28 + cursor_col
        row[tgt] = 255
        borders = [tgt+1, tgt-1, tgt+28, tgt-28]
        for i in borders:
            if tgt-1 < 0 or tgt+1 > 783 or tgt+28 > 783 or tgt-28 < 0:
                continue
            if row[i] >= 255:
                row[i] = 255
            else:
                row[i] += 51
        '''
        r.create_rectangle(x, y, x+pix_size, y+pix_size,
                           fill="white", width=0)
        r.create_rectangle(x+pix_size, y+pix_size, x+2*pix_size, y+2*pix_size,
                           fill="white", width=0)
        r.create_rectangle(x, y, x-pix_size, y-pix_size,
                           fill="white", width=0)
        '''
        #grid[y][x] = ['', 0]

root.bind("<Motion>", motion)
root.bind("<Button-1>", click)

def rgb_to_hex(r, g, b):
    conv = {'10': 'a', '11': 'b', '12': 'c', '13': 'd', '14': 'e', '15': 'f'}
    res = '#'
    for count in range(3):
        i = [r, g, b][count]
        first = str(i//16)
        second = str(i%16)
        if first in conv:
            first = conv[first]
        if second in conv:
            second = conv[second]
        res += first + second
    return res

def get_sample(df, sample_no):
    return df.iloc[sample_no]

def display(sample_no):
    global pix_size
    global x_margin
    global y_margin
    global disp_pred
    vect = get_sample(data, sample_no)
    r.delete('all')
    label = vect[0]
    img = vect[1:]
    x, y = x_margin, y_margin
    for i in img:
        if x >= x_margin + 28*pix_size:
            y += pix_size
            x = x_margin
        x += pix_size
        r.create_rectangle(x, y, x+pix_size, y+pix_size,
                           fill=rgb_to_hex(i, i, i), width=0)
    disp_int = r.create_text(250, 350, text=f"{disp_pred}" , anchor=NW, font=('Calibri', 20), fill='white')

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_len, output_len, hidden_layers, nodes_per_layer):
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_layers = hidden_layers
        self.nodes_per_layer = nodes_per_layer
        #self.input_weights = (np.random.uniform(-0.5, 0.5, (nodes_per_layer, input_len)))
        self.input_weights = (np.random.uniform(-0.5, 0.5, (nodes_per_layer, 784)))
        #self.hidden_weights = (np.random.uniform(-0.5, 0.5, (hidden_layers, nodes_per_layer, nodes_per_layer)))
        self.hidden_weights = (np.random.uniform(-0.5, 0.5, (output_len, nodes_per_layer)))
        #self.output_weights = (np.random.uniform(-0.5, 0.5, (output_len, nodes_per_layer)))
        self.input_bias = np.zeros((nodes_per_layer, 1))
        self.hidden_bias = np.zeros((output_len, 1))
        self.record = []

    def forward_pass(self, input_vect):
        input_vect = (input_vect[1:].T)/255
        #input_vect.shape += (1,)
        temp = self.input_bias + (self.input_weights @ input_vect).reshape(self.nodes_per_layer, 1)
        temp_vect = sigmoid(temp)
        #print(temp_vect)
        self.record.append(temp_vect)
        #for layer in range(1):#self.hidden_weights:
        temp_vect = sigmoid(self.hidden_bias + (self.hidden_weights @ temp_vect).reshape(self.output_len, 1))
        self.record.append(temp_vect)
            #print(temp_vect)
        #output_vect = sigmoid(np.dot(self.output_weights, temp_vect))
        #print(output_vect)
        return temp_vect, np.argmax(temp_vect)

    def back_prop(self, input_vect, lr):
        self.record = []
        label = np.zeros(self.output_len)
        label[input_vect[0]] = 1
        label = label.reshape(self.output_len, 1)
        pred, pred_int = self.forward_pass(input_vect)
        
        error = (1/len(pred) * np.sum((label - pred)**2, axis=0))
        #print(f"Error shape is {error.shape}")
        #nr_correct += int(np.argmax(pred) == np.argmax(label))

        delta_h = pred - label
        #print(f"delta_h shape is {delta_h.shape}")
        
        self.hidden_weights -= lr*delta_h.reshape(self.output_len, 1) @ self.record[-2].reshape(1, self.nodes_per_layer)
        #print(f"hidden_weights shape is {self.hidden_weights.shape}")
        self.hidden_bias -= lr*delta_h.reshape(self.output_len, 1)
        #print(f"hidden_bias shape is {self.hidden_bias.shape}")

        delta_i = np.transpose(self.hidden_weights) @ delta_h * (self.record[-2]*(1-self.record[-2]))
        #print(delta_i.shape)
        #self.input_weights -= lr*delta_i.reshape(784, 1)
        #part = -np.dot(np.transpose(self.record.pop(-1)), loss)
        #prev_a = self.record[-2]
        #a = self.record[-1]
        #dLdo = loss*prev_a

    def train(self, n, epochs):
        for i in range(epochs):
            self.test(100)
            for batch in range(n):
                sample = get_sample(data, randint(0, 59999))
                self.back_prop(sample, 0.01)
                if batch % 10000 == 0:
                    label = np.zeros(self.output_len)
                    label[sample[0]] = 1
                    label = label.reshape(self.output_len, 1)
                    #print(f"Predicted: {self.forward_pass(sample)}\nActual: {label}")
        self.test(100)
            
                
    def test(self, n):
        nr_correct = 0
        for batch in range(n):
            sample = get_sample(test_data, randint(0, 9999))
            pred, pred_int = nn.forward_pass(sample)
            if sample[0] == pred_int:
                nr_correct += 1
        print(f"Accuracy: {round(nr_correct/n*100, 3)}%")

def run_last():
    global disp_pred
    global engaged
    pred, disp_pred = nn.forward_pass(get_sample(data, curr_id))
    display(curr_id)
    engaged = False

def erase():
    global data
    global curr_id
    global engaged
    global disp_pred
    r.delete('all')
    data.loc[curr_id, :] = [0 for i in range(785)]
    data = data.astype(int)
    engaged = False
    disp_pred = ""
    
        
run = Button(root, text='Run', command=run_last, height=2, width = 1)
erase = Button(root, text='Erase', command=erase, height=2, width = 1)
run.place(x=30, y=350)
erase.place(x=80, y=350)


nn = NeuralNetwork(784, 10, 1, 256) 
#display(data.iloc[0])
