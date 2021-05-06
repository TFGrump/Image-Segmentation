from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


class Vertex:
    def __init__(self, label, value):
        self.label = label  # A value of 1 is corresponds to the background
        self.value = value  # So I don't have to look back at the graph when I need to get the value
        self.neighbors = {}

    def add_adjacent(self, neighbor, weight):
        self.neighbors[neighbor] = weight

    def set_label(self, label):
        """For when the label needs to change from
        background to foreground, or vice versa"""
        self.label = label

    def average_weight(self):
        sum = 0
        for neighbor in self.neighbors:
            sum = sum + self.neighbors[neighbor]
        return sum / len(self.neighbors)


def load_image(filepath):
    """Loads an image into a numpy array.
    Note: image will have 3 color channels [r, g, b]."""
    img = Image.open(filepath)
    return (np.asarray(img).astype(np.float)/255)[:, :, :3]


def onclick(event):
    """Found code here to be able to click on the graph:
    https://matplotlib.org/stable/users/event_handling.html

    This will set the selected pixel to be in the foreground"""
    global source_node, sink_node
    if source_node is None:
        source_node = graph[int(event.ydata), int(event.xdata)]
        source_node.set_label(0)
        print("Foreground has been set")
    if sink_node is None and event.dblclick:
        """The sink label is already set to 1.
           There is no need to change it to 1"""
        sink_node = graph[int(event.ydata), int(event.xdata)]
        print("Background has been set")


def generate_markov_random_feild(image):
    graph = np.array([[Vertex(label=1, value=image[j, i]) for i in range(image.shape[1])] for j in range(image.shape[0])])
    # The weights are simply going to be the difference in intensities, because I am only working with one color channel
    for j in range(graph.shape[1]):
        for i in range(graph.shape[0]):
            if i - 1 >= 0:
                graph[i, j].add_adjacent(graph[i - 1, j], abs(image[i, j]-image[i-1, j]))
            if i + 1 < graph.shape[0]:
                graph[i, j].add_adjacent(graph[i + 1, j], abs(image[i, j]-image[i+1, j]))
            if j - 1 >= 0:
                graph[i, j].add_adjacent(graph[i, j - 1], abs(image[i, j]-image[i, j-1]))
            if j + 1 < graph.shape[1]:
                graph[i, j].add_adjacent(graph[i, j + 1], abs(image[i, j]-image[i, j+1]))
    return graph


"""
The Energy Function:
E(x, y) = phi(xi, yi) + sum(i,j) psi(xi, xj)
"""


def calculate_label(node, label, sigma):
    # Calculate the difference between the value at that pixel and the label pixel
    """The Unary term of the Energy Equation.
       This returns a 0 (zero) if the labels are below a threshold,
       otherwise it will return a 1 (one)."""
    return 1 - math.exp((-1/sigma)*math.pow(node.value-label.value, 2))


def calculate_smoothness(node, label, sigma):
    # Calculate the average pixel intensity (because I am only using one color channel) of neighbors, compare w/label
    """The Pairwise term of the Energy Equation
       The result will be closer to 1 if the pixels are not close to the label value pixel"""
    return 1 - math.exp((-1/sigma) * math.pow(node.average_weight() - label.value, 2))


def update_markov_random_field(MRF, source, sink, thresh):
    if source is not None and sink is not None:
        for j in range(MRF.shape[0]):
            for i in range(MRF.shape[1]):
                val = calculate_label(MRF[j, i], source, 0.5) + calculate_smoothness(MRF[j, i], source, 5)
                if val < thresh:
                    MRF[j, i].set_label(0)
                val = calculate_label(MRF[j, i], sink, 0.5) + calculate_smoothness(MRF[j, i], sink, 5)
                if val < 0.2:
                    MRF[j, i].set_label(1)


def cut_graph(graph, image):
    for j in range(graph.shape[1]):
        for i in range(graph.shape[0]):
            if graph[i, j].label != 0:
                image[i, j] = 0
    return image


def show_image(image, other_image=None):
    figure = plt.figure()
    cid = figure.canvas.mpl_connect('button_press_event', onclick)
    plt.subplot(1,2,1)
    plt.imshow(image)
    if other_image is not None:
        plt.subplot(1,2,2)
        plt.imshow(other_image)
    plt.show()
    figure.canvas.mpl_disconnect(cid)


torchick = load_image("torchick.png")[:, :, 0]  # I don't want to deal with other color channels atm

# Initializes the MRF
graph = generate_markov_random_feild(torchick)

# Have the user pick a pixels to be in the foreground and background
source_node = None
sink_node = None
print("Single click to set the foreground Pixel\nDouble click to set the background pixel")
print("I have not implemented the ability to change Foreground or Background pixels, so choose wisely")
while True:
    if source_node is None:
        print("Select a Foreground pixel")
    if sink_node is None:
        print("Select a Background pixel")
    # I had this as the while loop condition and it just would not loop when only picking a foreground pixel
    if source_node is not None and sink_node is not None:
        break
    show_image(torchick)

update_markov_random_field(graph, source_node, sink_node, .5)
new_image = torchick.copy()
cut_graph(graph, new_image)

show_image(torchick, new_image)
