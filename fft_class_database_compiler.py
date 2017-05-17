import pandas as pd
import numpy as np
import random
import collections as cll
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean as euc
from IPython.display import display
from fnmatch import fnmatch, fnmatchcase

"""
The correct approach to this problem, of taking a long txt file which has two separate tables, both of which share the same indicies, but is read in only as one long string, and parsing it to ultimately create one large, curated DF which can be written to a CSV, is to take the larger of the tables and merely pop on the different ("outer join") columns from the smaller table, then do all feature engineering and data cleaning.  I have done this for the class multipliers and growth stats, and saved to a file called "fft_class_stats_jp.csv".  As a result, I should create a new script which then reads in that csv file if I need to do operations on that database.  This script is for reading from the big FFT - BMG text file and creating databases from it.
"""

def show_more(df, lines=None):
    """
    Function to display specified length of a DataFrame on a per-call basis.  Default number of lines is full length of DataFrame.  Can be specified as second argument in function call.
    """
    default_rows = pd.get_option("display.max_rows")
    if lines == None:
        lines = df.shape[0]

    pd.options.display.max_rows = lines
    display(df)
    pd.options.display.max_rows = default_rows

def association_dictionary():
    assoc_dic = {'Heroes': ('Ramza1', 'Ramza2', 'Ramza3', 'Ramza4','Zalbag', 'Agrias', 'Simon', 'Orlandu', 'Reis', 'Olan', 'Mustadio', 'Rafa', 'Malak', 'Beowulf', 'Cloud', 'Zalbag', 'Zalbag-Z', 'Steel_Giant_Worker'),
        'Lion_War': ('Delita1', 'Delita2', 'Delita3', 'Algus', 'Dycedarg', 'Larg', 'Goltana', 'Barinten', 'Rudvich', 'Gafgarion'),
        'Pawns': ('Alma', 'Alma?', 'Ovelia', 'Teta'),
        'Lucavi': ('Velius', 'Zalera', 'Hashmalum', 'Altima1', 'Queklain', 'Adramelk', 'Altima2', 'Elidibs'),
        'Shrine_Knights': ('Elmdor', 'Wiegraf1', 'Wiegraf2', 'Rofel', 'Vormav', 'Izlude', 'Meliadoul', 'Balk', 'Kletian', 'Celia', 'Lede'),
        'Glabados': ('Funeral', 'Draclau', 'Zalmo', 'Balmafula', 'Ajora')}

    return assoc_dic

def read_in_fft_text(text):
    stats_dic = cll.defaultdict(list, [])
    liz = []
    with open(text, 'r') as f:
        for line in f:
            liz.append(line)

    return liz

def create_fft_df(df):
    # Find split in TXT file for table creation
    for ind, row in enumerate(df.iloc[:,0]):
        if "####" in row:
            table_split = ind

    # Create DF from Table 1
    dd = cll.defaultdict(list, [])
    for ind in df.index[0: table_split]:
        line = df.iloc[ind].str.split()[0]
        job_stop = len(line) - 9
        job = "_".join(line[1: job_stop])
        for word in line[1: job_stop]:
            line.remove(word)
        line.insert(1, job)
        dd[ind] = line

    dfm = pd.DataFrame(dd)
    dfm = dfm.T

    # Create DF from Table 2
    dd = cll.defaultdict(list, [])
    for ind in df.index[table_split+2: ]:
        line = df.iloc[ind].str.split()[0]
        job_stop = len(line) - 6
        job = "_".join(line[1: job_stop])
        for word in line[1: job_stop]:
            line.remove(word)
        line.insert(1, job)
        dd[ind] = line

    dfc = pd.DataFrame(dd)
    dfc = dfc.T

    # Assign columns
    for df in [dfc, dfm]:
        cols = [col for col in df.iloc[0]]
        cols[0], cols[1], cols[2] = 'Job_ID', 'Job', 'Identity'
        df.columns = cols

    # Merge unique cols into one DF
    df = dfm.merge(dfc, how='outer')

    return df

def feature_engineer(df):
    # Create a Role column from the role the char had in FFT. Must do before removing sub-headers and dashed lines because these are the sub-headers I search for to fill out the "Role" column.
    categories = ['STORY', 'GENERIC', 'MONSTER', 'HUMANOID']
    role = None
    for idx in df.index:
        for word in categories:
            if word in df.loc[idx, 'Job_ID']:
                role = word.capitalize()
        df.loc[idx, 'Role'] = role

    # Create Association column baed on group of association
    assoc_dic = association_dictionary()
    reverse_assoc = {v: k for k, v in assoc_dic.iteritems()}
    for idx in df.index:
        identity = df.loc[idx, 'Identity']
        for key in reverse_assoc.keys():
            if identity in key:
                df.loc[idx, 'Association'] = reverse_assoc[key]
                break
            elif (df.loc[idx, 'Role'] == 'Monster') or (df.loc[idx, 'Role'] == 'Humanoid'):
                df.loc[idx, 'Association'] = 'Monster'
                break
            elif (df.loc[idx, 'Role'] == 'Generic'):
                df.loc[idx, 'Association'] = 'Generic'
                break
            else:
                df.loc[idx, 'Association'] = 'Unknown'

    return df

def clean_df(df):
    # Remove all rows with sub-headers or dashed dividers; could be more specific.
    df = df[~(df['Job_ID'].str.len() > 3)]

    # Reindex DF with sequential indices (0 - df.shape[0]) since we just removed some rows.
    df.index = df.index.reindex([x for x in xrange(df.shape[0])])[0]

    # # Convert number columns to int/float - had to remove dashed rows 1st
    df.loc[:, 'CEV'] = df['CEV'].str.strip('%')
    for col in df.iloc[:, 3:16]: # Range of numeric cols
        df.loc[:, col] = df.loc[:, col].astype(int)
    df.loc[:, 'CEV'] = df['CEV'] / 100

    return df

def add_ranks_stdev(df):
    # Commands to inspect different aspects of the units
    print df.groupby('Association').mean().sort_values('HPc', ascending=True)
    classes = df[df['Identity'] == 'generic']

    # We will add my standard Excel cols of Rank and StDev to the generic classes as an example:
    # Add new columns from any of the ranks. Note that the 'XXm' cols use method='min' (higher num = better) while the 'XXc' cols use method='max' :
    classes.loc[:, 'HPm_rk'] = classes['HPm'].rank(method='min', numeric_only=True, ascending=True)

    # Add StDev cols similarly (ddof=x for degrees of Freedom, default=1)
    classes.loc[:, 'HPm_sig'] = classes.loc[:, 'HPm'].std().round(2)


    # All ranks
    classes.rank(method='min', numeric_only=True, ascending=True)

if __name__ == '__main__':
    txt = 'data/FFT_job_list_to_parse.txt'
    df = pd.read_table(txt)
    df = create_fft_df(df)
    df = feature_engineer(df)
    df = clean_df(df)

    # Reorder cols to put strings at front
    cols = df.columns.tolist()
    cols = cols[0:3] + cols[-2:] + cols[3:16]
    df = df[cols]

    # Write to CSV that I can then load.
    # df.to_csv('~/Desktop/jp_fft_mult_table.csv')

    ### Make Rank and StDev func if desired?





    # Create an items db...
    stn="STRIKING WEAPONS | Range: 1v3 (from above) / 1v2 (from below) ==============================================================================#Bare Hands**...... damage = [(PA * Br) / 100] * PA    XA = [(PA * Br) / 100] #Knife ............ damage = [(PA + Sp) / 2] * WP      XA = [(PA + Sp) / 2] #Ninja Sword ...... damage = [(PA + Sp) / 2] * WP      XA = [(PA * Sp) / 2] #Sword ............ damage = PA * WP                   XA = PA #Knight Sword ..... damage = [(PA * Br) / 100] * WP    XA = [(PA * Br) / 100] #Katana ........... damage = [(PA * Br) / 100] * WP    XA = [(PA * Br) / 100] #Staff ............ damage = MA * WP                   XA = MA #Rod .............. damage = PA * WP                   XA = PA #Flail ............ damage = (1..PA) * WP              XA = (1..PA) #Axe .............. damage = (1..PA) * WP              XA = (1..PA) #Bag .............. damage = (1..PA) * WP              XA = (1..PA) ------------------------------------------------------------------------------- ** If a unit is barehanded, weapon-elemental attacks will receive the bonus    from the Martial Arts support ability (they don't if the unit is equipped    with any weapon -- see 6.2). For barehanded units, WP = 0 in any equations    that use WP (e.g., BATTLE SKILL success rate)."

    # st = stn.split('#')
