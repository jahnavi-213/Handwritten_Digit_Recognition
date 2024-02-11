import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import keras

# Load model from file mnist.h5
model = keras.models.load_model('mnist.h5')

class PaintApp:
    def __init__(self, root, width, height):
        self.root = root
        self.root.title("Handwritten Digit Recognition")  # app name

        # create a canvas to write
        self.canvas = tk.Canvas(root, width=width, height=height, bg='black')
        self.canvas.pack()

        # pen configuration
        self.pen_color = 'white'
        self.pen_size = 10

        # setup button controls
        self.setup_buttons()

        # link mouse drag to write on canvas
        self.canvas.bind("<B1-Motion>", self.paint)

    def setup_buttons(self):
        # digit prediction
        self.predict_button = ttk.Button(root, text="Recognize", command=self.predict_digit)
        self.predict_button.pack(pady=10)

        # clear button
        self.clear_button = ttk.Button(root, text="clear", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

    def paint(self, event):
        # add a round paint at the position where the mouse moves
        x1, y1 = (event.x - self.pen_size), (event.y - self.pen_size)
        x2, y2 = (event.x + self.pen_size), (event.y + self.pen_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.pen_color, outline=self.pen_color)

    def predict_digit(self):
        # get the image from the canvas and preprocess it to fit the model
        image = self.get_canvas_image()
        image = image.convert('L')
        image = image.resize((28, 28))
        image = np.array(image)
        image = image.reshape(1, 28, 28, 1)
        image = image.astype('float32')
        image /= 255

        # predict numbers from pictures
        result = model.predict(image)
        digit = np.argmax(result[0])

        # show predicted results
        result_label.config(text="Predicted digit: " + str(digit), font=("Arial", 16, "bold"), fg="blue")

    def get_canvas_image(self):
        # create images from canvas content
        image = Image.new("RGB", (self.canvas.winfo_width(), self.canvas.winfo_height()), "black")
        draw = ImageDraw.Draw(image)

        # create a rectangle around the canvas with black color
        draw.rectangle([(0, 0), (self.canvas.winfo_width(), self.canvas.winfo_height())], fill="black")

        # get all the items on the canvas and return them on image
        items = self.canvas.find_all()
        for item in items:
            x1, y1, x2, y2 = self.canvas.coords(item)
            color = self.canvas.itemcget(item, "fill")
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=color)

        return image

    def clear_canvas(self):
        # delete all the items on canvas
        self.canvas.delete("all")

if __name__ == "__main__":
    width, height = 400, 400
    root = tk.Tk()

    # create paint application
    app = PaintApp(root, width, height)

    # show prediction results and highlight results
    result_label = tk.Label(root, text="Predicted digit: ", font=("Arial", 16, "bold"), fg="blue")
    result_label.pack(pady=10)

    # start the main loop of the application
    root.mainloop()