from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Union, List
def color_labels(labels) -> list[tuple[float, float, float, float]]:
    d_colors = cm.rainbow(np.linspace(0, 1, np.unique(labels).__len__()))
    colors = []
    for i, l in enumerate(labels):
        colors.append(d_colors[int(l)])

    return colors



class matplotlibComplexPlot:



    def imshow(ax, img, label, **kwargs):
        ax.imshow(img, cmap="gray", **kwargs)
        ax.grid(False)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)  # Ensure spine is visible

        if label is not None:
            ax.set_title(label)
    
    def scatter(ax, points: dict[str: np.ndarray], **kwargs):
        # For KWARGS, : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes
        ax.scatter(**points, **kwargs)
        ax.set_facecolor("white")

        # if kwargs["extra_info"] is not None:

        #     if isinstance(labels, list):

        #         if kwargs["extra_info"]["color_samples"] is not None:
        #             color = kwargs["extra_info"]["color_samples"][img_index]
        #             if color is not None:
        #                 for spine in ax.spines.values():
        #                     spine.set_visible(True)  # Ensure spine is visible
        #                     spine.set_edgecolor(color)  # Set border color to white
        #                     spine.set_linewidth(5)  # Set border width





    def square_multi_subplot(data:list[np.ndarray], subplot: str, labels :Union[List[int], List[np.ndarray]]=None, titles:List[str]=None,
                              save_path=None,
                        save_dir=None, subplots_kwargs: Union[dict, list[dict]] = None, **kwargs):


        """ # Parameters for plotting
            images_per_row = 6  # Number of images per row
            rows_per_plot = 2  # Number of rows in each plot
        """

        if len(data) > 2:
            square_root = np.sqrt(len(data))
            # Confirm if integer
            if square_root*square_root == len(data):
                images_per_row = int(square_root)
                rows_per_plot = int(square_root)
            else:
                raise NotImplementedError("Currently only square number of images supported.")
        else:
                
            if len(data) == 2:
                images_per_row = 2
                rows_per_plot = 1
            else:
                raise NotImplementedError("Must plot at least 2 images.")

        if rows_per_plot == images_per_row:
            fig_size = (10,10)
        else:
            fig_size = (5*images_per_row, 5*rows_per_plot) #x,y
        # Iterate through groups of images and save each as a separate plot

        fig, axes = plt.subplots(rows_per_plot, images_per_row,
                                 figsize=fig_size)
        #fig.patch.set_facecolor('black')  # Set figure background



        # Plot images in the current group
        for i in range(rows_per_plot):
            for j in range(images_per_row):
                if rows_per_plot == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                img_index = i * images_per_row + j

                ax.set_title(titles[img_index] if titles is not None else None)
                matplotlibComplexPlot.__getattribute__(matplotlibComplexPlot, subplot)(ax, data[img_index],
                                            label = labels[img_index] if labels is not None else None,
                                            **subplots_kwargs[img_index] if subplots_kwargs is not None and isinstance(subplots_kwargs, list) else subplots_kwargs)
                



        # file_name = f"label_{labels[0]}"
        # if kwargs["extra_info"] is not None:
        #     grouping = kwargs["extra_info"]["grouping"]
        #     if kwargs["extra_info"]["color_group"] is not None:
        #         color_group : tuple[float, float, float, float] = kwargs["extra_info"]["color_group"]
        #         # Highlight the figure border
        #         fig.patch.set_edgecolor(color_group)  # Blue figure border
        #         fig.patch.set_linewidth(50)  # Thicker border
        #         file_name = f"grouping_{grouping}_{file_name}"
        # Adjust layout and save the plot

        fig.tight_layout()

        #if save_path is None:

            # assert save_dir is not None, f"Either save path or save directory must be given. Both are currently: {save_path}, {save_dir}"

            # save_path = os.path.join(save_dir, f"NN_E_PCA_plot_{file_name}.png")

            # mkdir(os.path.dirname(save_dir), os.path.basename(save_dir))
        if save_path is not None:
            fig.savefig(save_path)
            print(f"Saved plot: {save_path}")