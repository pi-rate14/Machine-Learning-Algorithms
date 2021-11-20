import matplotlib.pyplot as plt
import numpy as np
import random

w = [0,0]
threshold = 0
bias = 1
learning_rate = 1
max_iterations = 100

x = [
    [0.72,0.82,-1],
    [0.91,-0.69,-1],
    [0.03,0.93,-1],
    [0.12,0.25,-1],
    [0.96,0.47,-1],
    [0.8,-0.75,-1],
    [0.46,0.98,-1],
    [0.66,0.24,-1],
    [0.72,-0.15,-1],
    [0.35,0.01,-1],
    [-0.11,0.1,1],
    [0.31,-0.96,1],
    [0.0,-0.26,1],
    [-0.43,-0.65,1],
    [0.57,-0.97,1],
    [-0.72,-0.64,1],
    [-0.25,-0.43,1],
    [-0.12,-0.9,1],
    [-0.58,0.62,1],
    [-0.77,-0.76,1]
]

y = 0

color = ""

answer = ""

data_dictionary = {
    # 'Keys' : 'Values',
    '0.72,0.82' : '-1',
    '0.91,-0.69' : '-1',
    '0.03,0.93' : '-1',
    '0.12,0.25' : '-1',
    '0.96,0.47' : '-1',
    '0.8,-0.75' : '-1',
    '0.46,0.98' : '-1',
    '0.66,0.24' : '-1',
    '0.72,-0.15' : '-1',
    '0.35,0.01' : '-1',
    '-0.11,0.1' : '1',
    '0.31,-0.96' : '1',
    '0.0,-0.26' : '1',
    '-0.43,-0.65' : '1',
    '0.57,-0.97' : '1',
    '-0.72,-0.64' : '1',
    '-0.25,-0.43' : '1',
    '-0.12,-0.9' : '1',
    '-0.58,0.62' : '1',
    '-0.77,-0.76' : '1'
}

#store x and y coordinates of each point in the respective vectors
def get_points_of_color(data, label):
    x_coords = [float(point.split(",")[0]) for point in data.keys() if data[point] == label]
    y_coords = [float(point.split(",")[1]) for point in data.keys() if data[point] == label]
    return x_coords, y_coords

#turn on interactive graphics mode
plt.ion()

for k in range(1, max_iterations):
    hits = 0
    print("\n------------------------- ITERATION "+str(k)+" ------------------------- ")

    for i in range(0,len(x)):
        sum = 0

        # Weighted sum
        for j in range(0,len(x[i])-1):
            sum += x[i][j] * w[j]
        print("weighted sum: ", sum)
        # Output = Bias + Weighted sum
        output = bias + sum
        print("output: ", output)
        # Output is determined by the Threshold
        if output > threshold:
            y = 1
        else:
            y = -1     

        # Update the Weights if the output does not match with the Desired output
        if y == x[i][2]:
            hits += 1
            answer = "Correct!"
        else:
            for j in range (0,len(w)):             
                w[j] = w[j] + (learning_rate * x[i][2] * x[i][j])
            bias = bias + learning_rate * x[i][2]
            answer = "Error - Updating weight to: "+str(w)

        # Prints the answer
        if y == 1:
            print("\n"+answer)
        elif y == -1:
            print("\n"+answer)

        xA = 1
        xB = -1

        if w[1] != 0:
            yA = (- w[0] * xA - bias) / w[1]
            yB = (- w[0] * xB - bias) / w[1]
        else:
            xA = - bias / w[0]
            xB = - bias / w[0]

            yA = 1
            yB = -1

        x_coords, y_coords = get_points_of_color(data_dictionary, '-1')
       
        x_coords, y_coords = get_points_of_color(data_dictionary, '1')
        

    if hits == len(x):
        print("\n---------------------------------------------------------------")
        print("\nFunctionality learned with "+str(k)+" iterations!")
        break;
    '''
    else:
        print("\n---------------------------------------------------------------")
        print("\nFunctionality not learned!")
        break;
    '''
    
print("\nDone!\n")