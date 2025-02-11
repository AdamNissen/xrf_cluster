# -*- coding: utf-8 -*-
"""
Script applying the class XRF_cluster to all of the geological examples provided by 
Alix Osinchuk

@author: adamj
"""

from xrf_cluster import XRF_cluster

samples = {'016A': 5,
            'NW13-33': 6, #Note, ilmenite not well picked up by clustering algorithm, so only 6 clusters are searched for
            'SY22-02A': 10, #Note, may be bumped to 10 due to remove glass type
            'SY22-07C': 11} #May be increased to 12 to account for glass

for i in samples.keys():
    run = XRF_cluster(folderpath = "Example data/Geological_examples/"+i,
                    data_suffix = '.TXT',
                    outpath="Example data/output/"+i+"/")
    run.fit()
    run.cluster(clusters = samples[i])
    run.plot_clusters(cluster = None, save = True)
    run.get_stats()
    run.save_stats()
