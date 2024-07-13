import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk

class PointSelector:
    def __init__(self, image_path):
        self.root = tk.Tk()
        self.canvas = Canvas(self.root, cursor="cross")
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.points = []

        self.setup()

    def setup(self):
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if len(self.points) < 2:
            # Capture and store the point
            self.points.append((event.x, event.y))
            # Draw a circle for visual feedback
            self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill='red', outline='red')

            if len(self.points) == 2:
                print(f"Selected points: {self.points}")
                # Here you can add code to use the selected points for further processing

    def run(self):
        self.root.mainloop()

# Usage
image_path = 'inverted_depth_map.png'
selector = PointSelector(image_path)
selector.run()