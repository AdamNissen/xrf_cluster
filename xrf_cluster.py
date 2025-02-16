# -*- coding: utf-8 -*-
"""
A class for interpreting XRF data using an unsupervised clustering approach. 

TO DO LIST:
    
    5. Make docstrings!
    6. Build a method to score the clustering based on existing results.
    10. Maybe change how we deal with images so it is easy to save and export them after the fact,
    that said, generating the images takes so little time, so perhaps we don't need to carry
    them in memory.
    11. Revise save stats to not rely on numpy
    12. Figure out how to build a file with a neat folderset
    13. Define a method for picking colours for individual clusters as opposed to just setting default colur maps

    
DONE LIST
    1. update plot_clusters so that binary plots of cluster vs all can be made
    for individual requested cluster, and for each cluster as a whole
    2. make an attribute that is the name of each channel in the list of features
    (i.e. the collumns of self._data)
    3. Clean up the image output (e.g. no ticks, no tick labels, no apron, etc)
    4. Create the methods get_stats() and save_stats()
    7. Create save_maps() method or methods #actually just added save argument to the plotting image methods
    8. Figure out how to set a colour pallate for the images
    9. Update everything for actual data.
    14. Create a cluster dictionary containing "name" and "colour" (maybe more) to be set by users later
    15. Define a method for setting different colormaps
"""
import PIL
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl


#this is for later, let me figure out how to do things first
class XRF_cluster():
    
    def __init__(self, folderpath = None, data_suffix = None, outpath = None):
        self.folderpath = folderpath
        self._suffix = data_suffix #Probably could be depreciated away
        self._data = None #data used for clustering
        self._shape = None #The length and width of input images, used to rebuild phasemaps
        self._clusters = None #a list or 1d array of clusters output from unsupervised ML
        self._channels = None #the features used in clustering
        self._n_cluster = None
        self._target_values = None
        self.cluster_info = None #Information about each cluster, 0 = alias (e.g. mineral type) 1 = colour code
        
        self.outpath = outpath
        if self.outpath is None:
            self.outpath = self.folderpath
    
    
    def set_folderpath(self, folderpath):
        """
        Sets or resets the path to the folder containing XRF data

        Parameters
        ----------
        folderpath : str, directory
            The complete or relative path to the directory containing XRF data.
        """
        #This method is only really necessary if the folderpath becomes a private or protected variable
        self.folderpath = folderpath

    def set_suffix(self, suffix):
        """
        Sets the file suffix for files containing XRF data (e.g., .txt, .jpg, .csv,
        etc.). self._suffix may be depreciated, and if so this method will be unnecessary

        Parameters
        ----------
        suffix : str
            The file suffix of files containing XRF data, likely ".txt", perhaps ".bpm".
        """
        self._suffix = suffix

    def _identify_files(self):
        """
        A hidden method used to produce a list of all files with either a .bmp, 
        or a .TXT file suffix. It also produces a list of "channels" or elements
        that are represented in the data files

        Returns
        -------
        output : list
            A list of files that contain XRF data in the input folder
        self._channels : list
            A list of elements represented in the output list of XRF data files
            in the folder
        """
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
        
        suffix = self._suffix
        
        #checking for a dot in the file suffix
        if suffix == None:
            raise KeyError("please enter a file suffix (either .TXT or .bmp) using self.set_suffix()")
            return
        elif '.' not in suffix:
            if suffix in ['TXT', 'bmp']:
                suffix = '.'+suffix
            else:
                raise KeyError('Only .TXT and .bmp files can be imported as data files')
                return
        else:
            if suffix not in ['.TXT', '.bmp']:
                raise KeyError('Only .TXT and .bmp files can be imported as data files')
                return

        files = glob.glob(self.folderpath+'/*'+suffix)
        output = []
        for file in files:
            if "\\" in file:
                output += [file.replace('\\', '/')]
            else:
                output += [file]
        
        channels = []
        for i in output:
            channel = i.split('/')[-1]
            channel = channel.split('.')[0]
            channels+= [channel]
            
        self._channels = channels
        return output
    
    def _import_col(self, filepath):
        #adding data from a single image to the class data 
        ## this done with numpy arrays is WAY faster
        
        #create an array from a newly imported bitmap
        if filepath[-4:] == '.bmp':
            new_map = PIL.Image.open(filepath) #add some error flags here to ensure an image is being imported
            temp_array = np.array(new_map) 
        elif filepath[-4:] == '.TXT' or filepath[-4:] == '.txt':
            temp_array = np.loadtxt(filepath, delimiter = ',')
        else:
            raise KeyError('imported data must be in either .bmp or .TXT format')
            return
            
        #Saving the width and length of original bitmaps to make sure columns can be reshaped later
        if self._shape == None:
            self._shape = temp_array.shape
        else:
            pass
        
        #flattening the array to a row, then saving it to the working data as a column
        row = temp_array.flatten()
        
        return row 
    
    def _image_to_col(self, filepath):
        row = self._import_col(filepath = filepath)
        
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
    
    def _generate_cluster_info(self):
        """
        Hidden method used to reset the cluster information dictionary in the 
        class attributes. It first, resets the attrubute self.cluster_info to a 
        blank dictionary. Then it fills that dictionary with two-value lists, keyed
        0 - n where n is the number of clusters prescribed during self.cluster().
        Note that this method fills the lists with default values "mineral" and 
        "colour".
        
        The intended use for the cluster_info dict is to contain two bits of 
        information about each cluster. 1) the proposed mineral type of the cluster
        and 2) the colour used to identify that cluster in the map (likely as a
        hex code) for example:
                                                                    
        self.cluster_info = {0: ['tourmaline', '#5F9EA0'],
                             1: ['muscovite', '#FF00FF'],
                             2: ['spodumene', '#FFDAB9']}
        
        DEVELOPMENT NOTE
        ----------------
        If more values are needed in each self.cluster_info dictionary entry, e.g.
        if we have to add some geochemical value, or anything like that, please
        add a default entry here. That will be easiest.
        """
        self.cluster_info = {}
        for i in list(range(self._n_cluster)):
            self.cluster_info[i] = ['mineral', 'colour']
    
    def set_colourmap(self, cmap):
        """
        Sets the output colourmap to one of the default matplotlib colourmaps.
        Note, this changes the colour for every cluster sequentially, i.e., the
        "lowest value" or "first" colour in the map will be assigned to cluster 
        "0", then the second value goes to cluster "1", etc. 

        NOTE ON SPELLING
        Whenever an established code name is used, I will use that term, e.g.,
        matplotlib colormap. At all other times I will use the Canadian spelling
        of "colour".        

        Parameters
        ----------
        cmap : str, matplotlib colormap
            The colourmap to be used. In general, this class will default to 
            "inferno". See options at https://matplotlib.org/stable/users/explain/colors/colormaps.html#colormaps

        Raises
        ------
        ValueError
            If the cmap value is not a valid colormap.
        """
        #Filtering out non-matplotlib colormaps
        if cmap not in ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']:
            raise ValueError(str(cmap)+" is not a valid value for cmap; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'")
        
        #Extracting values from the cmap
        colours = mpl.colormaps[cmap].resampled(self._n_cluster)
        for n in list(range(self._n_cluster)):
            colour = colours.colors[n]
            self.cluster_info[n][1] = colour
    
    def set_colour(self, cluster, colour):
        """
        Defines a colour to use for a single phase during plotting.

        Parameters
        ----------
        cluster : int
            Number of the cluster to apply the colour to. Note that clusters are
            numbered from 0-n where n is one less than the total number of 
            clusters. Value must be present in self.cluster_info.keys().
        colour : str, array, any kind of colour notation
            Some colour notation that matplotlib will accept. Can be lots of things,
            try the hex code if you are having trouble.

        Raises
        ------
        ValueError
            Will be raised if the cluster argument is not a part of
            self.cluster_info.keys().
        """
        if cluster in self.cluster_info.keys():
            self.cluster_info[cluster][1] = colour
        else: 
            raise ValueError(str(cluster)+" is not a valid cluster. Please use a key from self.cluster_info")
            
    def set_phase(self, cluster, phase):
        """
        Defines an alias to be used when refering to a cluster. This will often
        be the mineral or phase that the cluster represents, e.g., "muscovite",
        or "glass". This can be more descriptive, if appropriate, e.g., "garnet 
        rims", "garnet cores", or "strange inclusions in the garnet".
        
        Parameters
        ----------
        cluster : int
            Number of the cluster to apply the colour to. Note that clusters are
            numbered from 0-n where n is one less than the total number of 
            clusters. Value must be present in self.cluster_info.keys().
        colour : str, array, any kind of colour notation
            Some colour notation that matplotlib will accept. Can be lots of things,
            try the hex code if you are having trouble.

        Raises
        ------
        ValueError
            Will be raised if the cluster argument is not a part of
            self.cluster_info.keys().
        """
        if cluster in self.cluster_info.keys():
            self.cluster_info[cluster][0] = phase
        else: 
            raise ValueError(str(cluster)+" is not a valid cluster. Please use a key from self.cluster_info")
        
    def cluster(self, clusters):
        #Pre-start saving the cluster number
        if not isinstance(clusters, int):
            raise SyntaxError("Please enter an integer above 0 for the number of clusters")
            return
        else:
            self._n_cluster = clusters
        
        #start with scaling
        scaler = StandardScaler()
        x_train = scaler.fit_transform(self._data)
        
        #Now clustering, with Kmeans for now
        kmn = KMeans(n_clusters = clusters, n_init = 10).fit(x_train)
        kmn_pred = kmn.predict(x_train)
        
        #Saving to a class attribute
        self._clusters = kmn_pred
        
        #generating cluster information, and building a default colormap
        self._generate_cluster_info()
        self.set_colourmap('inferno')
        
    def _display(self, phase_list = None, save = False, filepath = None):
        #Displays the array, we could add the option to select colours here as well
        displayed_element = None
        if phase_list is None:
            displayed_element = 'all_clusters'
        else:
            displayed_element = 'clusters'+str(phase_list)
            
        if isinstance(phase_list, type(None)):
           phase_list = self._clusters
        else:
            pass
        
        #this could be changed as well to produce an object rather than display it, this would be better for saving maybe?
        phase_list = phase_list.reshape(self._shape[0], self._shape[1])
        
        #setting the filepath if save == True
        if filepath is None and save == True:
            filepath = self.outpath+displayed_element+"_map.jpg"
        
        #Extracting the colourmap from self.cluster_info
        c_map = []
        for i in self.cluster_info.keys():
            c_map += [self.cluster_info[i][1]]
        c_map = ListedColormap(c_map)
        
        #plotting the array as a false coloured map
        fig = plt.figure()
        plt.axis('off') #removing axis ticks and tick labels
        fig.patch.set_visible(False) #removing the white apron
        
        if save == True:
            plt.imshow(phase_list, cmap=c_map)
            plt.savefig(fname = filepath, dpi = 600)
        if save == False:
            plt.imshow(phase_list, cmap=c_map)
        
    
    def plot_clusters(self, cluster = None, save = False):
        #A method that shows an image of the classified clusters
        #must be updated to show more than one cluster if requested
        #clusters = []
        
        #Extracting the colourmap from self.cluster_info
        c_map = []
        for i in self.cluster_info.keys():
            c_map += [self.cluster_info[i][1]]
        c_map = ListedColormap(c_map)

        if cluster == None:
            #Plotting the full phasemap together by calling the default of display
            if save == True:
                    self._display(save = True)
            elif save == False:
                    self._display()
            return
        
        elif cluster == 'All' or cluster == 'all' or cluster == 'ALL':
            #Creating a list of digits from 0 to the n_clusters-1 so each cluster will be individually plotted
            cluster = list(range(self._n_cluster))
            
        elif isinstance(cluster, int):
            #If the requested cluster is in the proper range, it will be turned into a one item list. If not, kickback an error
            if cluster not in range(self._n_cluster):
                raise KeyError('Requested cluster is outside of the available range. Please select a number beteen 0 and n_clusters-1')
                return
            else:
                cluster = [cluster]
    
        elif isinstance(cluster, list):
            #Building a list of clusters to plot, but also checking for int vs str answers. If there are any errors the method is ended
            temp = []
            for i in cluster:
                if isinstance(i, int):
                    if i not in range(self._n_cluster):
                        raise KeyError('Cluster '+str(i)+' is outside of the available range. Please select a number between 0 and n_clusters-1')
                        return
                    else:
                        temp += [i]

            #removing repeated numbers from the list
            cluster = list(set(temp))
            
        else: 
            raise KeyError("Please enter a valid argument for cluster")
            return
        

        #Iterating through the cluster list making masks of self._clusters
        for i in cluster:
            temp = self._clusters == i
            temp = temp.astype(int)
            
            if save == True:
                self._display(phase_list = temp, save = True)
            elif save == False:
                self._display(phase_list = temp)
        
    def get_stats(self, cluster = 'all', print_stats = True):
        #Setting up a list so that we can select which cluster to return 
        cluster_list = []
        
        #Checking for some types of answers for cluster
        if cluster in ['all', 'All', 'ALL']:
            cluster_list += list(np.unique(self._clusters))
        elif isinstance(cluster, list):
            for i in cluster:
                if i not in np.unique(self._clusters):
                    raise KeyError('Please select a valid cluster, or enter "all" as the cluster argument')
                    return
                else:
                    cluster_list += [i]
        elif cluster not in np.unique(self._clusters):
            raise KeyError('Please select a valid cluster, or enter "all" as the cluster argument')
            return
        else:
            cluster_list += [cluster]
        
        stats = {}
        
        #Okay we will turn each element in self._clusters into a boolean list, and use that to map out the array
        for i in cluster_list:
            bool_map = self._clusters == i #pulling out each rown of the list that correponds to the cluster
            subset = self._data[bool_map, :] #Making a subset from just one cluster
            
            #building parts of a spreadsheet, maybe the index [name]+ get deleted from these later
            cluster = (len(self._channels)*[i])
            channel = self._channels
            mean = subset.mean(axis = 0)
            median = np.median(subset, axis = 0)
            st_dev = +subset.std(axis = 0)
            
            #Filling a summary array
            summary = np.array([cluster,
                                channel, 
                                mean, 
                                median,
                                st_dev])
            
            #saving the cluster stats for each cluster
            stats[i] = summary
        
        #checking for a proper 
        if not isinstance(print_stats, bool):
            raise KeyError('print_stats argument must be a boolean (e.g. "True" or "False"')
            return
        #Printing the summary statistics if print_stats == True
        elif print_stats == True:
            for key in stats.keys():
                print(stats[key])
            return
        #Returning the stats dictionary if print_stats was changed to false
        else:
            return stats
    
    def save_stats(self, filepath = None):
        
        #Checking for a couple of important requirements for the filepath argument
        if filepath is None:
            filepath = self.outpath+"clustering_stats.xlsx"
        elif not isinstance(filepath, str):
            raise SyntaxError('filepath argument must be a string, e.g. "C:/data/mystats.xslx')
            return
        elif '\\' in filepath:
            raise SyntaxError('Please only use "/" to sepparate folders, e.g. C:/data/mystats.xlsx')
            return
        #adding a file suffix if one isn't there
        elif '.' not in filepath:
            filepath+= '.xlsx'
        else:
            pass
        
        #Generating dictionary full of stats arrays
        output = self.get_stats(cluster = 'all', print_stats = False)
        
        #saving the file if the file suffix is proper
        if filepath[-4:] not in ['xlsx', '.csv', '.txt']:
            #Sending an error if the wrong file suffix exists
            raise KeyError('Exported file must be a .xlsx file, .csv and .txt will be developed eventually')
            return
        elif filepath[-5:] == '.xlsx':
            #saving to an excel file with each cluster as a sheet
            with pd.ExcelWriter(filepath) as writer:
                for i in output.keys():
                    temp = pd.DataFrame(data = output[i])
                    temp = temp.drop(0)
                    index = pd.Series(data = ['none', 'channel', 'mean', 'median', 'standard deviation'])
                    temp.insert(0, 'rows', index)
                    temp.to_excel(writer, sheet_name = str(i), index = False, header = False)
        elif filepath[-4:] == '.csv':
            raise KeyError('.csv export not currently implemented, try setting a .xlsx filepath')
            return
        elif filepath[-4:] == '.txt':
            raise KeyError('.txt export is not currently implemented, try setting a .xlsx filepath')
            return
        else:
            raise KeyError('Exported file must be a .xlsx file, .csv and .txt will be developed eventually')
            return
        
    def save_maps(self, cluster= None, filepath = None):
        #Kind of redundant with save arguments added to earlier methods
        pass
    

#It works. Try it.
def test(shape = 1):
    ### IMPORTANT, must reset this to match the new self.fit() method ###
    
    #defining image locations and number of clusters for each test
    if shape == 0:
        test_path = "Example data/rectangle_test"
        cluster_number = 5
    elif shape == 1:
        test_path = "Example data/CYM_square_test"
        cluster_number = 3
   
    #calling X_cluster
    test = XRF_cluster(folderpath = test_path)
    test.fit()
    test.cluster(clusters = cluster_number)
    test.plot_clusters()



if __name__ == "__main__":
    module_test = XRF_cluster(folderpath = "Example data/Geological_examples/NW13-33",
                     data_suffix = '.TXT',
                     outpath="Example data/test_output/NW13-33/")
    module_test.fit()
    module_test.cluster(clusters = 6)
    module_test.plot_clusters(cluster = None, save = False)
    #module_test.get_stats()
    #module_test.save_stats()