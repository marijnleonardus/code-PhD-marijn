import matplotlib.pyplot as plt


class Plotting:
    @staticmethod
    def savefig(export_folder: str, file_name: str):
        """save figure in export folder with file name

        Args:
            export_folder (string): 
            file_name (string): 
        """
        export_location = export_folder + "\\" + file_name
        plt.savefig(export_location, dpi=300, pad_inches=0, bbox_inches='tight') 
        plt.show()
