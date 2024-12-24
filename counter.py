# -*- coding: utf-8 -*-

# import libraries
from skimage import io, measure, filters, segmentation, morphology, color, exposure
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max
from skimage.segmentation import watershed

from math import sqrt
import numpy as np
import pandas as pd

from scipy import ndimage

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['image.cmap'] = 'inferno'

from plotting_functions import (plot_bboxs, plot_texts, plot_circles,
                                 easy_sub_plot)
from image_processing_functions import (invert_image, crop_circle,
                                         background_subtraction,
                                         search_for_blobs,
                                         make_circle_label,
                                         detect_circle_by_canny)





class Counter():
    
    def __init__(self, image_path=None, image_array=None, verbose=True):

        self.props = {}

        if not image_path is None:
            self.load_from_path(image_path, verbose=verbose)
        if not image_array is None:
            self.load_image(image_array, verbose=verbose)

    def load_from_path(self, image_path, verbose=True):
        
        image = io.imread(image_path)
        self.load_image(image, verbose=verbose)

    def load_image(self, image_array, verbose=True):
        
        self.image_raw = image_array.copy()
        self.image_bw = color.rgb2gray(self.image_raw)
        self.image_inverted_bw = invert_image(self.image_bw)

        #if verbose:
            #plt.imshow(self.image_raw)
            #plt.show()
            


    def detect_area_by_canny(self, n_samples=None, radius=395, n_peaks=20, verbose=True):
        
        if verbose:
            print("detecting sample area...")


        # 1. Segmentation
        bw = self.image_bw.copy()

        # detect circles by canny method
        labeled = detect_circle_by_canny(bw, radius=radius, n_peaks=n_peaks)

        self.labeled = labeled

        if verbose:
            plt.title("segmentation")
            #plt.imshow(labeled)
            #plt.show()

        # 2. region props
        props = np.array(measure.regionprops(label_image=labeled, intensity_image=self.image_bw))
        bboxs = np.array([prop.bbox for prop in props])
        areas = np.array([prop.area for prop in props])
        cordinates = np.array([prop.centroid for prop in props])
        eccentricities = np.array([prop.eccentricity for prop in props])
        intensity = np.array([prop.intensity_image.mean() for prop in props])


        # 3. filter object

        selected = (areas >= np.percentile(areas, 90)) & (eccentricities < 0.3)


        # update labels
        labeled = make_circle_label(bb_list=bboxs[selected], img_shape=self.image_bw.shape)

        # region props again
        props = np.array(measure.regionprops(label_image=labeled, intensity_image=self.image_bw))
        bboxs = np.array([prop.bbox for prop in props])
        areas = np.array([prop.area for prop in props])
        cordinates = np.array([prop.centroid for prop in props])
        eccentricities = np.array([prop.eccentricity for prop in props])
        intensity = np.array([prop.intensity_image.mean() for prop in props])

        if not n_samples is None:
            ind = np.argsort(intensity)[-n_samples:]
            props = props[ind]
            bboxs = bboxs[ind]
            areas = areas[ind]
            cordinates = cordinates[ind]
            eccentricities = eccentricities[ind]



        # sort by cordinate y
        idx = np.argsort(cordinates[:, 0])

        self._props = props[idx]
        self.props["bboxs"] = bboxs[idx]
        self.props["areas"] = areas[idx]
        self.props["cordinates"] = cordinates[idx]
        self.props["eccentricities"] = eccentricities[idx]
        self.props["names"] = [f"sample_{i}" for i in range(len(self.props["areas"]))]

        if verbose:
            self.plot_detected_area()


    def plot_detected_area(self):

        print(str(len(self.props['areas'])) +" samples were detected")
        ax = plt.axes()
        plt.title("detected samples")
        ax.imshow(self.image_raw)
        plot_bboxs(bbox_list=self.props["bboxs"], ax=ax)
        plot_texts(text_list=self.props["names"], cordinate_list=self.props["bboxs"], ax=ax, shift=[0, -60])
        plt.show()

    def detect_area(self, n_samples, white_threshold=0.7, use_binelized_image_for_edge_detection=True, verbose=True):
        
        if verbose:
            print("detecting sample area...")

        if use_binelized_image_for_edge_detection:
            # 1. Segmentation
            bw = self.image_bw.copy()
            # get elevation map
            elevation_map = filters.sobel(bw)

            # detect white pixel
            tt = bw > white_threshold

            # fill hole white area
            tt = ndimage.binary_fill_holes(tt)

            # detect edge of white area
            elevation_map = filters.sobel(tt)


            # marker annotation
            markers = np.zeros_like(bw)
            markers[bw < 0.5] = 1
            markers[bw > white_threshold] = 2


            # watershed segmentation using the edge image
            segmentation = morphology.watershed(elevation_map, markers)

            segmentation = ndimage.binary_fill_holes(segmentation - 1)
            labeled, _ = ndimage.label(segmentation)
            self.labeled = labeled

        else:

            # 1. Segmentation
            bw = self.image_bw.copy()
            # get elevation map
            elevation_map = filters.sobel(bw)


            # annotate marker
            markers = np.zeros_like(bw)
            markers[bw < 0.5] = 1
            markers[bw > white_threshold] = 2

            # watershed
            segmentation = morphology.watershed(elevation_map, markers)

            segmentation = ndimage.binary_fill_holes(segmentation - 1)
            labeled, _ = ndimage.label(segmentation)
            self.labeled = labeled

        if verbose:
            plt.title("segmentation")
            plt.imshow(labeled)
            plt.show()

        # 2. region props
        props = np.array(measure.regionprops(label_image=labeled, intensity_image=self.image_bw))
        bboxs = np.array([prop.bbox for prop in props])
        areas = np.array([prop.area for prop in props])
        cordinates = np.array([prop.centroid for prop in props])
        eccentricities = np.array([prop.eccentricity for prop in props])


        # 3. filter object

        #selected = (areas >= np.percentile(areas, 90)) & (eccentricities < 0.3)

        selected_eccent = (eccentricities < 0.3)
        areas_ = areas[selected_eccent]

        selected_areas_ind = ind = np.argsort(areas_)[-n_samples:]
        selected_areas = np.zeros_like(areas_).astype(np.bool)
        selected_areas[selected_areas_ind] = True

        selected = selected_eccent.copy()
        selected[selected_eccent] = selected_areas

        # update labels
        labeled = make_circle_label(bb_list=bboxs[selected], img_shape=self.image_bw.shape)

        # region props again
        props = np.array(measure.regionprops(label_image=labeled, intensity_image=self.image_bw))
        bboxs = np.array([prop.bbox for prop in props])
        areas = np.array([prop.area for prop in props])
        cordinates = np.array([prop.centroid for prop in props])
        eccentricities = np.array([prop.eccentricity for prop in props])



        self._props = props
        self.props["bboxs"] = bboxs
        self.props["areas"] = areas
        self.props["cordinates"] = cordinates
        self.props["eccentricities"] = eccentricities
        self.props["names"] = [f"sample_{i}" for i in range(len(self.props["areas"]))]

        if verbose:
            #print(str(len(self.props['areas'])) +" samples were detected")
            ax = plt.axes()
            plt.title("detected samples")
            ax.imshow(self.image_raw)
            plot_bboxs(bbox_list=self.props["bboxs"], ax=ax)
            plot_texts(text_list=self.props["names"], cordinate_list=self.props["bboxs"], ax=ax, shift=[0, -60])
            plt.show()




    def crop_samples(self, shrinkage_ratio=0.9):
        
        
        self.sample_image_bw = [crop_circle(i.intensity_image, shrinkage_ratio) for i in self._props]
        self.sample_image_inversed_bw = [crop_circle(invert_image(i.intensity_image), shrinkage_ratio) for i in self._props]
        self.sample_image_for_quantification = self.sample_image_inversed_bw.copy()

    def plot_cropped_samples(self, inverse=False, col_num=3):
        

        if not inverse:
            image_list = self.sample_image_bw
            vmax = _get_vmax(image_list)
            #easy_sub_plot(image_list, col_num, self.props["names"], args={"cmap": "gray", "vmin": 0, "vmax": vmax})

        if inverse:
            image_list = self.sample_image_inversed_bw
            vmax = _get_vmax(image_list)
            #easy_sub_plot(image_list, col_num, self.props["names"], args={"vmin": 0, "vmax": vmax})


    def adjust_contrast(self, verbose=True, reset_image=False):
        
        if reset_image:
            self.sample_image_for_quantification = self.sample_image_inversed_bw.copy()
        if verbose:
            print("before_contrast_adjustment")
            vmax = _get_vmax(self.sample_image_for_quantification)
            easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"], {"vmin":0, "vmax": vmax})
            plt.show()


        for i, image in enumerate(self.sample_image_for_quantification):
            result = exposure.adjust_log(image, 1)
            #result = result - result[0,0]
            self.sample_image_for_quantification[i] = result

        if verbose:
            print("after_contrast_adjustment")
            vmax = _get_vmax(self.sample_image_for_quantification)
            easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"], {"vmin":0, "vmax": vmax})
            plt.show()


    def subtract_background(self, sigma=1, verbose=True, reset_image=True):
        
        if reset_image:
            self.sample_image_for_quantification = self.sample_image_inversed_bw.copy()
        if verbose:
            print("before_background_subtraction")
            vmax = _get_vmax(self.sample_image_for_quantification)
            #easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"], {"vmin":0, "vmax": vmax})
            #plt.show()

        for i, image in enumerate(self.sample_image_for_quantification):
            result = background_subtraction(image=image, sigma=sigma, verbose=False)
            result = result - result[0,0]
            result[result<0] = 0
            self.sample_image_for_quantification[i] = result

        if verbose:
            print("after_background_subtractionnnnnn")
            vmax = _get_vmax(self.sample_image_for_quantification)
            #easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"],  {"vmin":0, "vmax": vmax})
            #plt.show()

    def detect_colonies(self, min_size=5, max_size=15, threshold=0.02, num_sigma=10, overlap=0.5, verbose=True):
        
        self.detected_blobs = []
        for image in self.sample_image_for_quantification:
            blobs = search_for_blobs(image=image, min_size=min_size, max_size=max_size, num_sigma=num_sigma, overlap=overlap,
                                     threshold=threshold, verbose=False)
            self.detected_blobs.append(blobs)

        # save result as a dataFrame
        result = []
        for i, blobs in enumerate(self.detected_blobs):
            df = pd.DataFrame(blobs, columns=["x", "y", "radius"])
            df["sample"] = self.props["names"][i]
            result.append(df)
        result = pd.concat(result, axis=0)
        self.quantification_results = result

        # summarize results
        summary = result.groupby('sample').count()
        summary = summary[["x"]]
        summary.columns = ["colony_count"]
        self.quantification_summary = summary

        if verbose:
            self.plot_detected_colonies()

    def plot_detected_colonies(self, plot="final", col_num=3, vmax=None, save=None, overlay_circle=True):
        
        if plot == "raw":
            image_list = self.sample_image_bw
        elif plot == "final":
            image_list = self.sample_image_for_quantification
        elif plot == "raw_inversed":
            image_list = self.sample_image_inversed_bw
        else:
            raise ValueError("plot argment is wrong.")

        if vmax is None:
            vmax = _get_vmax(image_list)
            print("vmax: ", vmax)
        idx = 1

        for i, image in enumerate(image_list):

            k = (i%col_num + 1)
            ax = plt.subplot(1, col_num, k)
            blobs = self.detected_blobs[i]
            if plot == "raw":
                plt.imshow(image, cmap="gray", vmin=0, vmax=vmax)
                if overlay_circle:
                    plot_circles(circle_list=blobs, ax=ax, args={"color": "black"})

            else:
                plt.imshow(image, vmin=0, vmax=vmax)
                if overlay_circle:
                    plot_circles(circle_list=blobs, ax=ax)

            name = self.props["names"][i]
            plt.title(f"{name}: {len(blobs)} colonies")
            if (k == col_num) | (i == len(image_list)):
                if save is not None:
                    plt.savefig(f"{save}_{idx}.png", transparent=True)
                plt.show()
                idx += 1
            
            plt.show()

def _get_vmax(image_list):
    vmax = []
    for i in image_list:
        vmax.append(i.max())
    vmax = np.max(vmax)
    return vmax