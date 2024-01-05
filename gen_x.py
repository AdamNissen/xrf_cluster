# -*- coding: utf-8 -*-
"""
Generative script to create example data to be used with x_cluster.

I am going to create some spreadsheets filled with numbers in a standard sort of
manner so that I can set up code to collapse the spreadsheet into a single column
of a dataframe
"""
import csv

def generate_sheet(rows = 2, columns = 3, starting_value = 0):
    """
    Generates a list of lists, that can be put together in a spreadsheet. The 
    inner lists are rows of a spreadsheet. Put together the spreadsheet counts
    up from the starting value in the top leftmost cell, increasing one digit 
    per cell in the first column, then jumping to the first cell of the second 
    column when the first column is filled. Neatly arranged the output for a
    spreadsheet with 2 rows, 3 columns, and a starting value of 10 would be:
        [[10, 12, 14]
         [11, 13, 15]]

    Parameters
    ----------
    rows : int, optional
        The number of rows that should be in the generated spreadsheet. The
        default is 2.
    columns : int, optional
        The number of columns that should be in the generated spreadsheet. 
        The default is 3.
    starting_value : int, optional
        The number that should appear in the top leftmost cell of the spreadsheet.
        Cell numbers will increase from this digit. The default is 0.

    Returns
    -------
    sheet : list 
        A list of lists that can be put together to form a spreadsheet. The inner
        lists are meant to be rows of the spreadsheet.
    """
    cell_total = columns*rows
    sheet = []
    for i in range(rows):
        newrow = list(range(starting_value + i, cell_total + starting_value, rows))
        sheet = sheet + [newrow]
    return sheet
        
def sheet_to_csv(sheet, filename):
    """
    A function that writes .csv files from two dimension lists (i.e. a list of 
    lists). 

    Parameters
    ----------
    sheet : list
        The "spreadsheet" a two dimensional list. The list should contain lists
        of spreadsheet rows (making it two dimensional). The number of lists
        inside the main list is the number of rows of the final spreadsheet, the
        number of items in each inner list is the number of columns in the final
        spreadsheet, e.g:
            [[1, 2, 3],
             [4, 5, 6]]
    filename : str
        File with full location to save as the .csv. Note, must include the .csv 
        suffix.
    """
    if filename[-4:] == '.csv':
        pass
    else:
        filename = filename + '.csv'
        
    with open(filename, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(sheet)

if __name__ == '__main__':
    for i in range(5):
        sheet_i = generate_sheet(rows = 10, columns = 10, starting_value = (i+1)*1000)
        filename_i = "Example data/test_"+str(i+1)
        sheet_to_csv(sheet = sheet_i, filename = filename_i)
        
    
