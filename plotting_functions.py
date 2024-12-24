# -*- coding: utf-8 -*-


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['image.cmap'] = 'inferno'


def plot_bboxs(bbox_list, ax, args={"edgecolor":'white', "linewidth":2, "alpha": 0.5}):
    for bb in bbox_list:
        minr, minc, maxr, maxc = bb
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, **args)
        ax.add_patch(rect)

def plot_texts(text_list, cordinate_list, ax, shift=[0, 0],
               fontdict={'color':  'white',
                            'weight': 'normal','size': 10}):
    for text, cordinate in zip(text_list, cordinate_list):
        plt.text(x=cordinate[1]+shift[0], y=cordinate[0]+shift[1], s=str(text),
                 fontdict=fontdict)

def plot_circles(circle_list, ax, args={"color": "white", "linewidth": 1, "alpha": 0.5}):

    for blob in circle_list:
        y, x, r = blob
        c = plt.Circle((x, y), r, **args, fill=False)
        ax.add_patch(c)

def easy_sub_plot(image_list, col_num=1, title_list=None, args={}):
    # Eğer title_list sağlanmamışsa, varsayılan olarak boş bir liste kullanılır
    if title_list is None:
        title_list = [''] * len(image_list)

    row_num = len(image_list) // col_num + (1 if len(image_list) % col_num != 0 else 0)

    fig, axes = plt.subplots(row_num, col_num, figsize=(15, row_num * 5))
    axes = axes.flatten()  # axes array'ini tek boyutlu hale getir

    for i, image in enumerate(image_list):
        ax = axes[i]
        ax.imshow(image, **args)
        ax.set_title(title_list[i])
        ax.axis('off')  # Eksenleri gizle

    # Kalan boş subplotları gizle
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show() 