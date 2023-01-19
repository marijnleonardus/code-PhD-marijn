# author: Marijn Venderbosch
# january 2023

import matplotlib.pyplot as plt
        

class Plotting:
        
    def saving(file_location, file_name):
        """
        Parameters
        ----------
        file_location : string
            location where to save plot. absolute path to working directory.
        file_name : string
            file name to save plus file type. e.g. 'test.png'

        Returns
        -------
        None.

        """
        plt.savefig(file_location + file_name,
                    dpi=300,
                    bbox_inches='tight')
        