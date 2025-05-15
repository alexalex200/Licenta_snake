import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numpy.linalg import norm
import copy
import tkinter as tk
from PIL import Image, ImageDraw
import cv2

matplotlib.use('TkAgg')

def save_data(weights, biases, file_name):
    array_weights = np.array(weights, dtype=object)
    np.save(file_name + '_weights', array_weights)
    array_biases = np.array(biases, dtype=object)
    np.save(file_name + '_biases', array_biases)
def load_data(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        label = []
        images = []
        for row in reader:
            label.append(int(row[0]))
            images.append(list(map(int, row[1:])))
    return label, np.array(images, dtype=np.uint8)


def reproduction(weights1, weights2, biases1, biases2, learning_rate):

    new_weights = copy.deepcopy(weights1)
    new_biases = copy.deepcopy(biases1)

    for i in range(len(weights2)):
        for j in range(len(weights2[i])):
            for k in range(len(weights2[i][j])):
                if np.random.rand() < 0.5:
                    new_weights[i][j][k] = weights2[i][j][k]
                new_weights[i][j][k] = new_weights[i][j][k] + np.random.normal(0, 0.1)

    for i in range(len(biases2)):
        for j in range(len(biases2[i])):
            if np.random.rand() < 0.5:
                new_biases[i][j] = biases2[i][j]
            new_biases[i][j] = new_biases[i][j] + np.random.normal(0, 0.02)

    return new_weights, new_biases


def create_weights_and_biases(len_input, len_output, num_hidden_layers, len_hidden_layers, epsilon=1):
    weights = []
    biases = []

    weights.append(np.random.uniform(-epsilon, epsilon, (len_input, len_hidden_layers[0])))
    biases.append(np.zeros(len_hidden_layers[0]))

    for i in range(1, num_hidden_layers):
        weights.append(np.random.uniform(-epsilon, epsilon, (len_hidden_layers[i - 1], len_hidden_layers[i])))
        biases.append(np.zeros(len_hidden_layers[i]))

    weights.append(np.random.uniform(-epsilon, epsilon, (len_hidden_layers[-1], len_output)))

    return weights, biases


def predict(input, weights, biases):
    prev_layer = np.dot(input, weights[0]) + biases[0]
    prev_layer = np.maximum(0, prev_layer)
    mean = np.mean(prev_layer, axis=0)
    std = np.std(prev_layer, axis=0) + 1e-8
    prev_layer = (prev_layer - mean) / std

    for i in range(1, len(weights) - 1):
        layer = np.dot(prev_layer, weights[i]) + biases[i]
        layer = np.maximum(0, layer)
        mean = np.mean(layer, axis=0)
        std = np.std(layer, axis=0) + 1e-8
        layer = (layer - mean) / std
        prev_layer = layer

    output = np.dot(prev_layer, weights[-1])
    output = np.array(output, dtype=np.float64)
    output = np.exp(output) / np.sum(np.exp(output))
    return output

class Individual:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.fitness = 0

    def evaluate(self, images, labels):
        self.fitness = 0

        for i in range(1000):
            random_index = np.random.randint(0, len(images))
            input = images[random_index] / 255
            output = predict(input, self.weights, self.biases)

            if np.argmax(output) == labels[random_index]:
                self.fitness += 1
        self.fitness /= 1000

        # for i in range(len(images)):
        #     input = images[i] / 255
        #     output = predict(input, self.weights, self.biases)
        #
        #     if np.argmax(output) == labels[i]:
        #         self.fitness += 1
        # self.fitness /= len(images)

    def reproduce(self, other, learning_rate):
        new_weights, new_biases = reproduction(self.weights, other.weights, self.biases, other.biases, learning_rate)
        return Individual(new_weights, new_biases)


def evaluate_individual(individual, images, labels):
    individual.evaluate(images, labels)
    return individual

def train():
    train_labels, train_images = load_data('mnist/mnist_train.csv')
    test_labels, test_images = load_data('mnist/mnist_test.csv')

    len_input = 784
    len_output = 10
    num_hidden_layers = 2
    len_hidden_layers = [100, 100]
    num_individuals = 100
    num_generations = 100
    learning_rate = 1

    individuals = []
    for i in range(num_individuals):
        weights, biases = create_weights_and_biases(len_input, len_output, num_hidden_layers, len_hidden_layers)
        individuals.append(Individual(weights, biases))

    for generation in range(num_generations):
        individuals = pool.starmap(evaluate_individual, [(individual, train_images, train_labels) for individual in individuals])

        individuals.sort(reverse=True,key=lambda x: x.fitness)

        # correct = 0
        # for i in range(len(test_images)):
        #     input = test_images[i] / 255
        #     output = predict(input, individuals[0].weights, individuals[0].biases)
        #     if np.argmax(output) == test_labels[i]:
        #         correct += 1
        #
        # acurracy = correct / len(test_images)
        learning_rate = 1 / individuals[0].fitness

        print('Generation:', generation, 'Fitness:', individuals[0].fitness)
        for i in range(10, num_individuals):
            individuals[i] = individuals[np.random.randint(0,9)].reproduce(individuals[np.random.randint(0,9)], learning_rate)

    individuals[0].evaluate(test_images, test_labels)
    print('Test Fitness:', individuals[0].fitness)
    save_data(individuals[0].weights, individuals[0].biases, 'best_individual'+str(individuals[0].fitness))


loaded_weights = np.load("best_weights/best_individual0.7786_weights.npy", allow_pickle=True)
loaded_biases = np.load("best_weights/best_individual0.7786_biases.npy", allow_pickle=True)

# Convert back to a list of NumPy arrays
best_weights = [np.array(w) for w in loaded_weights]
best_biases = [np.array(b) for b in loaded_biases]

class DrawApp:
    def __init__(self, root, predict_fn):
        self.root = root
        self.root.title("Draw a Number")

        self.canvas_size = 280  # Larger canvas for drawing
        self.image_size = 28  # Size for MNIST model

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)

        # Bind mouse events to drawing functions
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create an image to store drawing
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)  # Black background
        self.draw = ImageDraw.Draw(self.image)

        self.predict_fn = predict_fn  # Neural network prediction function

    def paint(self, event):
        radius = 10  # Brush size
        x, y = event.x, event.y
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
        self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        # Resize image to 28x28
        img = self.image.resize((self.image_size, self.image_size))
        img_array = np.array(img) / 255  # Normalize pixel values

        # Invert colors (since MNIST digits are white on black)

        # Flatten and pass to predict function
        img_array = img_array.flatten()

        prediction = self.predict_fn(img_array,best_weights,best_biases)  # Call the user's predict function
        predicted_digit = np.argmax(prediction)  # Get the most probable class

        # Show result
        self.root.title(f"Predicted Digit: {predicted_digit}")


if __name__ == '__main__':
    pool = Pool(processes=12)
    train()

    # root = tk.Tk()
    # app = DrawApp(root, predict)  # Replace dummy_predict with your actual `predict` function
    # root.mainloop()

