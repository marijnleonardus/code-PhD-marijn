import matplotlib.pyplot as plt


class Plotting:
    def savefig(export_folder, file_name):
        """save figure in export folder with file name"""

        export_location = export_folder + file_name
        plt.savefig(export_location, 
            dpi = 300, pad_inches = 0, bbox_inches = 'tight') 
        plt.show()