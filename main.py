from counter import Counter
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams["savefig.dpi"] = 400

def main():
    path = "inverted.jpg"
    image = Image.open(path)

    
    counter = Counter(image_path=path)
    counter.detect_area_by_canny(radius=300,verbose=False)

    counter.crop_samples(shrinkage_ratio=0.8)
    

    counter.plot_cropped_samples(inverse=True)
    counter.subtract_background()

    counter.detect_colonies(min_size=3, max_size=25, threshold=0.18, verbose=True)


if __name__ == "__main__":
    main()




