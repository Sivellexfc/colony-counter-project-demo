import tkinter as tk
from tkinter import filedialog, messagebox
from counter import Counter
import os
from PIL import Image, ImageOps, ImageTk



def browse_image():
    """Opens file dialog to select an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        entry_image_path.delete(0, tk.END)  
        entry_image_path.insert(0, file_path)  
        
        
        image = Image.open(file_path)
        
        
        width, height = image.size
        
        image.thumbnail((400, 400))  
        img_display = ImageTk.PhotoImage(image)
        
        # Display image
        label_image.config(image=img_display)
        label_image.image = img_display 
        label_size.config(text=f"Width: {width}px, Height: {height}px") 


def run_analysis():
    """Runs the colony detection analysis using the input parameters."""
    try:
        image_path = entry_image_path.get()
        if not os.path.isfile(image_path):
            raise ValueError("Please select a valid image file.")
        
        # Get values from input fields
        radius = int(entry_radius.get())
        min_size = int(entry_min_size.get())
        max_size = int(entry_max_size.get())
        threshold = float(entry_threshold.get())
        shrinkage_ratio = float(entry_shrinkage.get())

        image = Image.open(image_path)
        
        if invert_var.get(): 
            inverted_image = ImageOps.invert(image.convert('RGB'))
            image = inverted_image
            inverted_image.save("inverted_image.jpg")
            counter = Counter(image_path="inverted_image.jpg")  
        else:
            counter = Counter(image_path=image_path)
        
        counter.detect_area_by_canny(radius=radius, verbose=False)
        counter.crop_samples(shrinkage_ratio=shrinkage_ratio)
        counter.plot_cropped_samples(inverse=True)
        counter.subtract_background()
        counter.detect_colonies(min_size=min_size, max_size=max_size, threshold=threshold, verbose=True)

        messagebox.showinfo("Success", "Analysis completed successfully!")

    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Colony Detection")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

label_image_path = tk.Label(frame, text="Select Image:")
label_image_path.grid(row=0, column=0, sticky="w", pady=5)

entry_image_path = tk.Entry(frame, width=50)
entry_image_path.grid(row=0, column=1, pady=5)

button_browse = tk.Button(frame, text="Browse", command=browse_image)
button_browse.grid(row=0, column=2, padx=10)

label_image = tk.Label(frame)
label_image.grid(row=1, column=0, columnspan=3, pady=10)

label_size = tk.Label(frame, text="Width: 0px, Height: 0px")
label_size.grid(row=2, column=0, columnspan=3, pady=5)

label_radius = tk.Label(frame, text="Radius:")
label_radius.grid(row=3, column=0, sticky="w", pady=5)

entry_radius = tk.Entry(frame)
entry_radius.grid(row=3, column=1, pady=5)
entry_radius.insert(0, "300")  # Default value

label_min_size = tk.Label(frame, text="Min Size:")
label_min_size.grid(row=4, column=0, sticky="w", pady=5)

entry_min_size = tk.Entry(frame)
entry_min_size.grid(row=4, column=1, pady=5)
entry_min_size.insert(0, "3")  # Default value

label_max_size = tk.Label(frame, text="Max Size:")
label_max_size.grid(row=5, column=0, sticky="w", pady=5)

entry_max_size = tk.Entry(frame)
entry_max_size.grid(row=5, column=1, pady=5)
entry_max_size.insert(0, "25")  # Default value

label_threshold = tk.Label(frame, text="Threshold:")
label_threshold.grid(row=6, column=0, sticky="w", pady=5)

entry_threshold = tk.Entry(frame)
entry_threshold.grid(row=6, column=1, pady=5)
entry_threshold.insert(0, "0.18")  # Default value

label_shrinkage = tk.Label(frame, text="Shrinkage Ratio:")
label_shrinkage.grid(row=7, column=0, sticky="w", pady=5)

entry_shrinkage = tk.Entry(frame)
entry_shrinkage.grid(row=7, column=1, pady=5)
entry_shrinkage.insert(0, "0.9")  # Default value


invert_var = tk.BooleanVar() 
invert_checkbox = tk.Checkbutton(frame, text="Invert Image", variable=invert_var)
invert_checkbox.grid(row=8, column=1, pady=5)

button_run = tk.Button(root, text="Hesapla", command=run_analysis, bg="green", fg="white")
button_run.pack(pady=20)

root.mainloop()
