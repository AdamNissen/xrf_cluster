# -*- coding: utf-8 -*-
"""
A class for interpreting XRF data using an unsupervised clustering approach. 

TO DO LIST:
    1. update plot_clusters so that binary plots of cluster vs all can be made
    for individual requested cluster, and for each cluster as a whole
    2. make an attribute that is the name of each channel in the list of features
    (i.e. the collumns of self._data)
    3. Create the methods get_stats() set_outpath() save_maps() and save_stats()
    4. Update everything for actual data.
"""
import PIL
import numpy as np
import glob
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#this is for later, let me figure out how to do things first
class X_cluster():
    
    def __init__(self, folderpath = None):
        self.folderpath = folderpath
        self._data = None
        self._shape = None
        self._clusters = None
    
    def set_folderpath(self, folderpath):
        #This method is only really necessary if the folderpath becomes a private or protected variable
        self.folderpath = folderpath

    def _identify_files(self):
        #Collecting all .bmp files found at folderpath
        
        #A suite of error messages to make sure the folderpath is correct
        if not isinstance(self.folderpath, str):
            raise SyntaxError('Please set the folderpath to a string, for example "C:/Users/data"')
        elif self.folderpath[-1] == '/':
            raise SyntaxError('Please ensure the folderpath ends in a folder, for example "C:/Users/data"')
        elif "\\" in self.folderpath:
            raise SyntaxError('Please ensure folders are sepparated with "/" in the folderpath, for example "C:/Users/data"')
        else:
            pass
        
        files = glob.glob(self.folderpath+'/*.bmp')
        output = []
        for file in files:
            if "\\" in file:
                output += [file.replace('\\', '/')]
            else:
                output += [file]
        
        return output
        
    def _image_to_col(self, filepath):
        #adding data from a single image to the class data 
        ## this done with numpy arrays is WAY faster
        
        #create an array from a newly imported bitmap
        new_map = PIL.Image.open(filepath) #add some error flags here to ensure an image is being imported
        temp_array = np.array(new_map) 
        
        #Saving the width and length of original bitmaps to make sure columns can be reshaped later
        if self._shape == None:
            self._shape = temp_array.shape
        else:
            pass
        
        #flattening the array to a row, then saving it to the working data as a column
        row = temp_array.flatten()
        
        #saving flattened arrays as columns, this segment will be reused
        if not isinstance(self._data, np.ndarray):
            self._data = row
        else:    
            self._data = np.column_stack((self._data, row))

    def fit(self):
        #Combining a few methods to create the data 
        
        files = self._identify_files()
        
        #there could be some error checks here
        for file in files:
            self._image_to_col(filepath = file)


    def _unflatten_sheet(self):
        pass
    
    def cluster(self, clusters):
        #start with scaling
        scaler = StandardScaler()
        x_train = scaler.fit_transform(self._data)
        
        #Now clustering, with Kmeans for now
        kmn = KMeans(n_clusters = clusters, n_init = 10).fit(x_train)
        kmn_pred = kmn.predict(x_train)
        
        #Saving to a class attribute
        self._clusters = kmn_pred

        
    def _display(self, array = None):
        #Displays the array, we could add the option to select colours here as well
        if array == None:
            array = self._clusters
        else:
            pass
        
        #this could be changed as well to produce an object rather than display it, this would be better for saving maybe?
        array = array.reshape(self._shape[0], self._shape[1])
        plt.imshow(array)
        
    
    def plot_clusters(self, cluster = None):
        #A method that shows an image of the classified clusters
        #must be updated to show more than one cluster if requested
        #clusters = []
        
        if cluster == None:
            self._display()
        elif cluster == 'All' or cluster == 'all' or cluster == 'ALL':
            pass
        elif isinstance(cluster, list):
            pass
        elif isinstance(cluster, int):
            pass
        elif isinstance(cluster, str):
            pass
        else: 
            raise KeyError("Please enter a valid argument for cluster")
            return

    def get_stats(self):
        pass
    
    def set_outpath(self):
        pass
    
    def save_maps(self):
        pass
    
    def save_stats(self):
        pass
    
#Learning to load a bitmap into 
# =============================================================================
# path = "D:/Bespoke_python/FiLTER/X_cluster/Example data/rectangle_test/blue.bmp"
# img = PIL.Image.open(path) #used to open an image, really a bitmap
# p = np.array(img) #Translating the bitmap into an array
# LxW = p.shape #reserving the shape of a bitmap
# flat = p.flatten() #flattening the bitmap, np.ravel() might work as well, but ravel is likely less stable with changing variables
# q = flat.reshape(LxW[0], LxW[1]) #returning the flattened array to its original shape. NOTE this is the correct order, 0 then 1

# =============================================================================


#It works. Try it.
def test(shape = 1):
    ### IMPORTANT, must reset this to match the new self.fit() method ###
    
    #defining image locations and number of clusters for each test
    if shape == 0:
        test_path = "D:/Bespoke_python/FiLTER/X_cluster/Example data/rectangle_test"
        cluster_number = 5
    elif shape == 1:
        test_path = "D:/Bespoke_python/FiLTER/X_cluster/Example data/CYM_square_test"
        cluster_number = 3
    
    #calling X_cluster
    test = X_cluster(folderpath = test_path)
    test.fit()
    test.cluster(clusters = cluster_number)
    test.plot_clusters()


# =============================================================================
# test(shape = 0)
# test(shape = 1)
# =============================================================================


#################### TO DO LIST ###############################################
# =============================================================================
#     1. Change X_cluster._clusters to the properly sized array, not a single column
#     2. Build one or two methods to:
#         a) find all of the .bmp files in a folder
#         b) import each of them into the working data with image_to_col
#     3. Make a method that plots the data
#     4. Make some way of defining the colours for the plot
#     5. Produce plots for each cluster (i.e. cluster vs all plot)
#     6. Produce statistics for each cluster
#     7. Test with real data
#     8. Make a gui
# =============================================================================


# =============================================================================
# if __name__ == "__main__":
#     test = X_cluster(folderpath = "D:/Bespoke_python/FiLTER/X_cluster/Example data/rectangle_test")
#     test.fit()
#     test.cluster(clusters = 5)
#     test.plot_clusters()
# =============================================================================
    