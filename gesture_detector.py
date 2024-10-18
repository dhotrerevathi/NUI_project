import tkinter as tk
from tkinter import filedialog, messagebox
import os

class GestureDrawer:
    def __init__(self, master):
        self.master = master
        self.master.title("Gesture Drawer")

        self.canvas = tk.Canvas(master, width=400, height=400, bg="white")
        self.canvas.pack()

        self.points = []
        self.is_drawing = False

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.save_button = tk.Button(master, text="Save Gesture", command=self.save_gesture)
        self.save_button.pack()

        self.clear_button = tk.Button(master, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

    def start_drawing(self, event):
        self.is_drawing = True
        self.points.append((event.x, event.y))
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black")

    def draw(self, event):
        if self.is_drawing:
            self.points.append((event.x, event.y))
            self.canvas.create_line(self.points[-2][0], self.points[-2][1], event.x, event.y, fill="black")

    def stop_drawing(self, event):
        self.is_drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []

    def save_gesture(self):
        if not self.points:
            messagebox.showwarning("Warning", "No gesture drawn")
            return

        initial_file = filedialog.asksaveasfilename(defaultextension=".txt")
        if not initial_file:
            return

        base_name = os.path.splitext(os.path.basename(initial_file))[0]
        gesture_file = os.path.join(os.path.dirname(initial_file), f"{base_name}.txt")

        gesture_file_exists = True if os.path.exists(gesture_file) else False
        
        # Always append to gesture file
        with open(gesture_file, 'a') as f:
            if os.path.getsize(gesture_file) == 0:
                f.write(f"{base_name}\n")
            f.write("BEGIN\n")
            for x, y in self.points:
                f.write(f"{x},{y}\n")
            f.write("END\n")

        # Check if eventstream file already exists
        event_file = os.path.join(os.path.dirname(gesture_file), f"{base_name}_eventfile.txt")
        if not os.path.exists(event_file):
            # Save eventstream file only if it doesn't exist
            with open(event_file, 'w') as f:
                f.write("MOUSEDOWN\n")
                for x, y in self.points:
                    f.write(f"{x},{y}\n")
                f.write("MOUSEUP\n")
                f.write("RECOGNIZE\n")
            eventfile_message = f"Saved event file: {event_file}"
        else:
            eventfile_message = f"Event file already exists: {event_file}"
        
        if gesture_file_exists is True:
            messagebox.showinfo("Success", f"Appended to gesture file: {gesture_file}\n{eventfile_message}")
        else:
            messagebox.showinfo("Success", f"Saved gesture file: {gesture_file}\n{eventfile_message}")
        self.clear_canvas()

root = tk.Tk()
app = GestureDrawer(root)
root.mainloop()