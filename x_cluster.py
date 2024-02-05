# -*- coding: utf-8 -*-
"""
A class for interpreting XRF data using an unsupervised clustering approach. 

TO DO LIST:
    
    5. Create same_maps() method or methods
    5. Figure out how to set a colour pallate for the images
    6. Update everything for actual data.
    7. Maybe change how we deal with images so it is easy to save and export them after the fact,
    that said, generating the images takes so little time, so perhaps we don't need to carry
    them in memory.
    8. Revise save stats to not rely on numpy
    9. Figure out how to build a file with a neat folderset
    
Removed from to do list    
    1. update plot_clusters so that binary plots of cluster vs all can be made
    for individual requested cluster, and for each cluster as a whole
    2. make an attribute that is the name of each channel in the list of features
    (i.e. the collumns of self._data)
    3. Clean up the image output (e.g. no ticks, no tick labels, no apron, etc)
    4. Create the methods get_stats() and save_stats()
"""
import PIL
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#this is for later, let me figure out how to do things first
class X_cluster():
    
    def __init__(self, folderpath = None):
        self.folderpath = folderpath
        self._data = None #data used for clustering
        self._shape = None #The length and width of input images, used to rebuild phasemaps
        self._clusters = None #a list or 1d array of clusters output from unsupervised ML
        self._channels = None #the features used in clustering
        self._n_cluster = None
    
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
        
        channels = []
        for i in output:
            channel = i.split('/')[-1]
            channel = channel.split('.')[0]
            channels+= [channel]
            
        self._channels = channels
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

        
    def _display(self, phase_list = None):
        #Displays the array, we could add the option to select colours here as well
        if isinstance(phase_list, type(None)):
           phase_list = self._clusters
        else:
            pass
        
        #this could be changed as well to produce an object rather than display it, this would be better for saving maybe?
        phase_list = phase_list.reshape(self._shape[0], self._shape[1])
        
        #plotting the array as a false coloured map
        fig = plt.figure()
        plt.axis('off') #removing axis ticks and tick labels
        fig.patch.set_visible(False) #removing the white apron
        plt.imshow(phase_list)
        
    
    def plot_clusters(self, cluster = None):
        #A method that shows an image of the classified clusters
        #must be updated to show more than one cluster if requested
        #clusters = []
        
        if cluster == None:
            #Plotting the full phasemap together by calling the default of display
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
    
    def save_stats(self, filepath):
        
        #Checking for a couple of important requirements for the filepath argument
        if not isinstance(filepath, str):
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
        
    def save_maps(self):
        pass
    
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

if __name__ == "__main__":
    test = X_cluster(folderpath = "D:/Bespoke_python/FiLTER/X_cluster/Example data/rectangle_test")
    test.fit()
    test.cluster(clusters = 5)
    test.plot_clusters(cluster = 4)
    #test.get_stats()
    test.save_stats(filepath = "D:/Bespoke_python/FiLTER/X_cluster/Example data/output/rectangle.xlsx")