# import matplotlib as mpl
# mpl.use('Qt5Agg')
import pandas as pd
import numpy as np
import random
import collections as cll
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as ln
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean as euc
from IPython.display import display
# from fnmatch import fnmatch, fnmatchcase


"""
This script is for opening and manipulating the "fft_class_stats_jp.csv" database.
"""


def load_csv():
    df = pd.read_csv('/Users/jpw/Dropbox/Data_Science/jp_projects/2017-01-25_fft_class_stats/data/fft_class_stats_jp.csv', header=0)
    classes = df[df['Identity'] == 'Generic']

    return df, classes

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

def euc_dist_skew(df, x, y):
    x, y = x.name, y.name
    df['Sum'] = df[x] + df[y]
    df['Skew'] = df[x] - df[y]
    df['Skew_Dist'] = None

    for idx in df.index:
        point = df.loc[idx, 'Sum'] / 2.
        if df.loc[idx, 'Skew'] < 0:
            df.loc[idx, 'Skew_Dist'] = euc([df.loc[idx, x], df.loc[idx, y]], [point, point]) * -1
        else:
            df.loc[idx, 'Skew_Dist'] = euc([df.loc[idx, x], df.loc[idx, y]], [point, point])

    df['Skew_Dist'] = df['Skew_Dist'].astype(float)
    print(df['Skew_Dist'])

def plot_fft_scatter(df, x, y):
    # x, y = x.name, y.name
    # cev = df['CEV'] * 100
    # x = (df[x] - 100.)
    # y = (df[y] - 100.)


    # x = df['PA']
    # y = df['MA']
    # Squire Red = 170, 45, 25 --> #aa2d19
    # Chemist Blue = 56, 95, 145 --> #335e91

    label_set = ['Heroes', 'Lion_War', 'Pawns', 'Glabados', 'Unknown', 'Shrine_Knights', 'Lucavi', 'Generic', 'Monster']
    branch_ind = []
    for job in df['Association']:
        branch_ind.append(label_set.index(job))

    # label_set = ['Squire', 'Chemist', 'Generic', 'Story', 'Monster', 'Humanoid']
    # branch_ind = []
    # for job in df['Role']:
    #     branch_ind.append(label_set.index(job))

    color_set = ['#aa2d19', '#335e91', 'black', 'navy', 'teal', 'orange', 'green', 'brown', 'grey']
    clr_list = [color_set[int(job)] for job in branch_ind]
    hue_data = clr_list

    # hue_data = df['M'] + df['J']
    # hue_data = df['CEV']

    # clr_list = np.array(list(colors.cnames)).astype(str)
    # hue_data = random.sample(clr_list, len(x))
    # hue_data = x+y
    # hue_data = df['SPD'] - 100
    job = df['Job']

    fig = plt.figure(figsize=(9,8.2))
    ax = fig.add_subplot(1,1,1)
    plt.suptitle("Final Fantasy Tactics Class Multipliers")
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    ax.axhline(0, color='k', alpha=.5, linestyle='--', lw=1.3, zorder=0)
    ax.axvline(0, color='k', alpha=.5, linestyle='--', lw=1.3, zorder=0)
    ax.grid(True, zorder=0)

    ax.set_xlim([np.min(x)-5, np.max(x)+10])
    ax.set_ylim([np.min(y)-5, np.max(y)+5])
    sq = 30
    # ax.set_xlim([-sq, sq])
    # ax.set_ylim([-sq, sq])

    # Plot diagonal line
    diag = np.array([num for num in range(int(min(min(x), min(y)) - 10), int(max(max(x), max(y)) + 10))])
    ax3 = ax.plot(diag, diag, c='grey', alpha=.3, zorder=0)

    # z = [0 for num in range(32)] #for lims function
    # lims = set_axis_lims(x, y, z, ax, chart)
    # mymap, my_cmap = custom_cmap(hue_data)

    # ccg_idx = [0, 1, 2, 4]
    # non_ccg = []
    # for num in range(32):
    #     if num not in ccg_idx:
    #         non_ccg.append(num)

    # Plot Teams
    # ax2 = ax.scatter(x[non_ccg], y[non_ccg], s=((win_percent[non_ccg]*500) + ((win_percent[non_ccg]*10)**2)), c=hue_data[non_ccg], cmap='gray', alpha = .3, zorder=5)
    #
    # ax1 = ax.scatter(x[ccg_idx], y[ccg_idx], s=((win_percent[ccg_idx]*600) + ((win_percent[ccg_idx]*10)**2.4)), c=mymap[ccg_idx], cmap='jet', alpha = .9, zorder=10)

    ax1 = ax.scatter(x, y, c=hue_data, cmap='RdYlBu_r', s=200, alpha = .88, zorder=10, linewidths=.75, edgecolors='k')

    # for i in range(len(df['Job'])):
    #     ax.text(x[i], y[i] + .95, '%s' % (str(df['Job'][i]) + '  ' + str(hue_data[i])), ha='center', va='bottom', size=10, color='k',zorder=11)

    # offset = 2
    # for i in range(len(df['Job'])):
    for i in df.index:
        ax.text(x[i], y[i] + 2.1, '%s' % (str(df['Identity'][i])), ha='center', va='bottom', size=8, color='k',zorder=11)


    # playoff_tms = list(ccg_idx)
    # playoff_tms.extend([3,5,10,15,16,19,25,28])
    #
    # for i in playoff_tms:
    #     ax.text(x[i], y[i] -.75, '%s' % (str(df['Team'][i]) + '  ' + str(df['Tot'][i])), ha='center', va='bottom', size=12, color='k',zorder=11)


    # Set Tick Marks for vertical Colorbar
    # tix = [x for x in range(np.min(hue_data).astype(int),np.max(hue_data).astype(int)) if x % 10 == 0]
    # cb = fig.colorbar(ax1, ticks=[tix])
    # cb = fig.colorbar(ax1)
    # cb.set_label("PA + MA")

    # Flip y-axis to make stronger DEF at top and adjust frame
    f = plt.gcf()
    f.subplots_adjust(bottom=0.1, left=0.1, right=.99, top=0.95)
    # plt.gca().invert_yaxis()
    s
    plt.show()


def scatter_plotter(ax, x, y, param_dic):
    """
    Helper function to make scatter plots.
    """
    graph = ax.scatter(x, y, **param_dic)
    return graph



def param_dic(pos):
    pass

def local_vars_bad_idea(job_dic):
    pass
    # This is foolish, unneccesary, and unsafe, but this is technically how you would put all the Jobs into the global namespace (i.e. outside of the container) from a list or dict.
    # for job, obj in job_dic.iteritems():
    #     locals()["%s" % job[0:3]] = obj
    #
    #
    # A1 = Job('Archer')
    # A2 = Job('Archer')
    # M1 = Job('Monk')

def calc_stats(df, start_lvl, end_lvl, Job_ID):
    """
    Calculate the HP of any character at any level, given the job they will be during the level up.
    INPUT: Stats DataFrame (DF), character (JobClass object), and job (str) during level up.
    OUTPUT: HP at target level (int)
    """

    ##############################################################
    #SEE FFT BMG 6.5 Pg. 160 under Female HP Calcs for example of how to do this
    ##############################################################

    """ Here's how to calculate stats and level ups:
    Level 1 stat = (raw_stat * mult_val) / 1638400
        raw_stat = the raw number a char starts with, such as 81920 for Male PA
        mult_val = HPm, MPm, etc. for the given class; class multiplier
        Note: if Lucavi, divisor is 163840, hence much higher HP

    Level Up Bonus = Current_raw_stat_val / (C_val + lvl)
        Current_raw_stat_val = the current value of the stat, such as HP = 31.34
        C_val = HPc, MPc, etc. for the given class; class growth factor
        lvl = the lower level you are leveling up from (e.g. 31, in 31 -> 32)

    The Level Up Bonus will be constant if you stay the same class every level.

    """

    dfj = df[df['Job_ID'] == Job_ID]
    stats = ['HPc', 'MPc', 'PAc', 'MAc', 'SPDc']
    # The mean of Stats with high/low is listed as their raw stat
    raw_dic = {
        'HP_Monster_low': 573440,
        'HP_Monster_high': 622591,
        'MP_Monster_low': 98304,
        'MP_Monster_high': 147455,
        'PA_Monster_low': 81920,
        'PA_Monster_high': 98303,
        'MA_Monster_low': 81920,
        'MA_Monster_high': 98303,
        'HP_Male_low': 491520,
        'HP_Male_high': 524287,
        'MP_Male_low': 229376,
        'MP_Male_high': 245759,
        'HP_Female_low': 458752,
        'HP_Female_high': 491519,
        'MP_Female_low': 245760,
        'MP_Female_high': 262143,
        'HP_Monster': 598015.5,
        'MP_Monster': 122879.5,
        'PA_Monster': 90111.5,
        'MA_Monster': 90111.5,
        'SP_Monster': 81920,
        'HP_Male': 507903.5,
        'MP_Male': 237567.5,
        'PA_Male': 81920,
        'MA_Male': 65536,
        'SP_Male': 98304,
        'HP_Female': 475135.5,
        'MP_Female': 253951.5,
        'PA_Female': 65536,
        'MA_Female': 81920,
        'SP_Female': 98304}

    level_1_divisor = 1638400
    if dfj.loc[dfj.index[0], 'Association'] == 'Lucavi':
        print("LUCAVI")
        level_1_divisor = 163840
    elif dfj.loc[dfj.index[0], 'Association'] == 'Generic':
        gen = input("Enter (1) for male, (2) for female: ")
        if gen == '1':
            dfj.loc[dfj.index[0], 'Gender'] = 'Male'
        else:
            dfj.loc[dfj.index[0], 'Gender'] = 'Female'

    print("{0} - {1}".format(dfj.loc[dfj.index[0], 'Identity'].upper(), dfj.loc[dfj.index[0], 'Job']))

    for col in dfj.columns:
        if col in stats:
            # Assign class and char specific vars to more readable names
            C_val = dfj.loc[dfj.index[0], col]
            mult_val = dfj.loc[dfj.index[0], (col[:2]+'m')]
            raw_stat = raw_dic["{0}_{1}".format(col[:2], dfj.loc[dfj.index[0], 'Gender'])]

            # Get base stat at Lvl 1 for this char. All level ups from this.
            stat = (raw_stat * mult_val) / level_1_divisor

            if start_lvl != 1:
                bonus = stat / (C_val + 1) # This Bonus only works if char never changes classes, else more complicated
                stat += (bonus * (start_lvl - 1))
            for lvl in range(start_lvl, end_lvl):
                print('{0}: {1:.0f} at lvl {2}'.format(col[:2], stat, (lvl)))
                bonus = stat / (C_val + lvl)
                stat += bonus

            print('{0}: {1:.0f} at lvl {2}\n'.format(col[:2], stat, (lvl + 1)))
    # return stat, bonus



    def velius_hp():
        """ Completely hardcoded, no DF needed. This work precisely correctly. At level 31, Velius has around 950-1000 HP. Use as ref.
        """
        start_lvl = 1
        end_lvl = 31
        lucavi_divisor = 163840.
        velius_HPm = 80.
        velius_HPc = 12.
        hp_monster_low = 573440
        hp_monster_high = 622591
        base_low = (hp_monster_low * velius_HPm) / lucavi_divisor # at lvl 1
        base_high = (hp_monster_high * velius_HPm) / lucavi_divisor # at lvl 1
        hp_low = base_low
        hp_high = base_high

        for lvl in range(start_lvl, end_lvl):
            # print 'HP_low: {0:.1f} at lvl {1}'.format(hp_low, (lvl))
            # print 'HP_high: {0:.1f} at lvl {1}'.format(hp_high, (lvl))
            print("On average, Velius has {0:.1f} HP at lvl {1}".format(np.mean([hp_low, hp_high]), lvl))

            bonus_low = hp_low / (velius_HPc + lvl)
            bonus_high = hp_high / (velius_HPc + lvl)
            hp_low += bonus_low
            hp_high += bonus_high

            if lvl == (end_lvl - 1):
                # print 'HP_low: {0:.1f} at lvl {1}'.format(hp_low, (lvl + 1))
                # print 'HP_high: {0:.1f} at lvl {1}'.format(hp_high, (lvl + 1))
                print("On average, Velius has {0:.1f} HP at lvl {1}\n".format(np.mean([hp_low, hp_high]), lvl + 1))

        return np.mean([hp_low, hp_high])

    # velius_hp()


def add_ranks_stdev(df):
    # Commands to inspect different aspects of the units
    print(df.groupby('Association').mean().sort_values('HPc', ascending=True))
    classes = df[df['Identity'] == 'generic']

    # We will add my standard Excel cols of Rank and StDev to the generic classes as an example:
    # Add new columns from any of the ranks. Note that the 'XXm' cols use method='min' (higher num = better) while the 'XXc' cols use method='max' :
    classes.loc[:, 'HPm_rk'] = classes['HPm'].rank(method='min', numeric_only=True, ascending=True)

    # Add StDev cols similarly (ddof=x for degrees of Freedom, default=1)
    classes.loc[:, 'HPm_sig'] = classes.loc[:, 'HPm'].std().round(2)


    # All ranks
    classes.rank(method='min', numeric_only=True, ascending=True)


class JobClass(object):
    def __init__(self, job, level, name='dude'):
        self.job = job
        self.name = name
        self.lvl = level
        for att in df.columns:
            setattr(self, att, df.loc[self.job, att])

    def __repr__(self):
        return "{0}, ({1} {2})".format(self.name, self.lvl, self.job)

    def __str__(self):
        return "{0}, ({1} {2})".format(self.name, self.lvl, self.job)

    def __eq__(self, other):
        return self.job == other.job and self.name == other.name




    def attack(self, other):
        """
        Physically attack target with basic 'Attack' command.
        """
        dmg = (self.PA * .75) * self.PA
        other.HP -= dmg
        print(self.name + " hits " + str(other.name) + " for " + str(dmg) + " damage.")
        return other.HP


if __name__ == '__main__':
    # df = load_df()
    df, classes = load_csv()
    lucavi = df[df['Association'] == 'Lucavi']
    shrine_knights = df[df['Association'] == 'Shrine_Knights']
    heroes = df[df['Association'] == 'Heroes']
    lion_war = df[df['Association'] == 'Lion_War']
    dfplot = df.copy()
    # dfplot = dfplot.iloc[:-2, :] #Drop Onion Knight
    dfplot = df[df['PAm'] != 0]
    x, y = dfplot['PAm'], dfplot['MAm']
    # euc_dist_skew(df, x, y)
    # plot_fft_scatter(df, x, y)
    plot_fft_scatter(dfplot, x, y)
    # m = JobClass('Monk', 33)


    # DATABASE CALCS
    # df, classes = load_csv()
    # start_lvl, end_lvl = 1, 75
    # Job_ID = '97'
    # calc_stats(df, start_lvl, end_lvl, Job_ID)



    # Stat calcs are set up now.
    # Need to either add more DBs, like weapons, or do cool plotting and ML.
    # Also need to make Classes with properties that interact and allow something of a mini-game.
