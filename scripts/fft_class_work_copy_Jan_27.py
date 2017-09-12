import pandas as pd
import numpy as np
import random
import collections as cll
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean as euc
from IPython.display import display


def load_df():
    df = pd.read_csv('/Users/jpw/Dropbox/Data_Science/jp_projects/fft_class_multiplier.csv', header=0, nrows=23)

    # job_list = dfmult['Job']
    # dfa = dfc.append(dfm, ignore_index=True)
    # dfa = dfc.append(dfe)

    df = df.set_index('Job', drop=True)
    return df

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
    print df['Skew_Dist']

def plot_fft_scatter(df, x, y):
    x, y = x.name, y.name
    cev = df['CEV'] * 100
    x = (df[x] - 100.)
    y = (df[y] - 100.)

    # x = df['PA']
    # y = df['MA']
    # Squire Red = 170, 45, 25 --> #aa2d19
    # Chemist Blue = 56, 95, 145 --> #335e91
    label_set = ['S', 'C', 'N']
    branch_ind = []
    for job in df['Tree']:
        branch_ind.append(label_set.index(job))

    color_set = ['#aa2d19', '#335e91', 'gold']
    clr_list = [color_set[int(job)] for job in branch_ind]
    hue_data = clr_list
    # clr_list = np.array(list(colors.cnames)).astype(str)
    # hue_data = random.sample(clr_list, len(x))
    # hue_data = x+y
    # hue_data = df['SPD'] - 100
    job = df['Job']

    fig = plt.figure(figsize=(13,10))
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
    diag = np.array([num for num in xrange(int(min(min(x), min(y)) - 10), int(max(max(x), max(y)) + 10))])
    ax3 = ax.plot(diag, diag, c='grey', alpha=.3, zorder=0)

    # z = [0 for num in xrange(32)] #for lims function
    # lims = set_axis_lims(x, y, z, ax, chart)
    # mymap, my_cmap = custom_cmap(hue_data)

    # ccg_idx = [0, 1, 2, 4]
    # non_ccg = []
    # for num in xrange(32):
    #     if num not in ccg_idx:
    #         non_ccg.append(num)

    # Plot Teams
    # ax2 = ax.scatter(x[non_ccg], y[non_ccg], s=((win_percent[non_ccg]*500) + ((win_percent[non_ccg]*10)**2)), c=hue_data[non_ccg], cmap='gray', alpha = .3, zorder=5)
    #
    # ax1 = ax.scatter(x[ccg_idx], y[ccg_idx], s=((win_percent[ccg_idx]*600) + ((win_percent[ccg_idx]*10)**2.4)), c=mymap[ccg_idx], cmap='jet', alpha = .9, zorder=10)

    ax1 = ax.scatter(x, y, s=500, c=hue_data, cmap='jet', alpha = .78, zorder=10)

    # for i in xrange(len(df['Job'])):
    #     ax.text(x[i], y[i] + .95, '%s' % (str(df['Job'][i]) + '  ' + str(hue_data[i])), ha='center', va='bottom', size=10, color='k',zorder=11)

    offset = 2
    for i in xrange(len(df['Job'])):
        ax.text(x[i], y[i] + 2.5, '%s' % (str(df['Job'][i])), ha='center', va='bottom', size=11, color='k',zorder=11)


    # playoff_tms = list(ccg_idx)
    # playoff_tms.extend([3,5,10,15,16,19,25,28])
    #
    # for i in playoff_tms:
    #     ax.text(x[i], y[i] -.75, '%s' % (str(df['Team'][i]) + '  ' + str(df['Tot'][i])), ha='center', va='bottom', size=12, color='k',zorder=11)


    # Set Tick Marks for vertical Colorbar
    # tix = [x for x in xrange(np.min(hue_data).astype(int),np.max(hue_data).astype(int)) if x % 10 == 0]
    # cb = fig.colorbar(ax1, ticks=[tix]) #DVOA Cbar
    # cb.set_label("PA + MA")

    # Flip y-axis to make stronger DEF at top and adjust frame
    f = plt.gcf()
    f.subplots_adjust(bottom=0.06, left=0.06, right=.99, top=0.95)
    # plt.gca().invert_yaxis()
    plt.show()

def multi_ind_prac(df):



        """
        MULTIINDEX PRACTICE:
        """



        df = df.append(df, ignore_index=True)

        for ind in df.index:
            if ind < 23:
                df.ix[ind, 'Year'] = 2000
            else:
                df.ix[ind, 'Year'] = 2001


        df['Year'] = df['Year'].astype(int)
        df['PA'] = df['PA'] - 100
        df['MA'] = df['MA'] - 100


        index = pd.MultiIndex.from_arrays([df['Year'].values, df['Job'].values])
        df.index = index


        skew_dic = {}
        df['Skew_Dist'] = None
        df['pt'] = None
        skew_ind = None
        ct = 1

        # for yr in df['Year'].unique():
        #     print "===============", yr
        #     for job in df['Job'].unique():
        #         print yr, job, " ...", ct
        #         if ct % 23 == 0:
        #             print "\n"
        #         ct += 1

        # for yr in df['Year'].unique():
        for yr in [2000]:
            print "===============", yr
            for job in df.loc[yr]['Job']:
                # print yr, job, " ...", ct
                # if ct % 23 == 0:
                #     print "\n"
                # ct += 1


                for val in np.arange(min(df['PA']), max(df['PA']), .1):
                    skew_dic.setdefault(job + str(yr), []).append(euc([df.loc[(yr, job), 'PA'], df.loc[(yr, job), 'MA']], [val, val]))
                    skew_dic.setdefault(job + str(yr) + '_pt', []).append(val)


                # if df.loc[(yr, job), 'PA'] > df.loc[(yr, job), 'MA']:
                #     for val in np.arange(0, max(df['PA']), 10):
                #         skew_dic.setdefault(job, []).append(euc([df.loc[(yr, job), 'PA'], df.loc[(yr, job), 'MA']], [val, val]))
                #         skew_dic.setdefault(job + 'pt', []).append(val)
                # else:
                #     for val in np.arange(min(df['PA']), 0, 10):
                #         skew_dic.setdefault(job, []).append(euc([df.loc[(yr, job), 'PA'], df.loc[(yr, job), 'MA']], [val, val]))
                #         skew_dic.setdefault(job + 'pt', []).append(val)



        # for yr in df['Year'].unique():
        for yr in [2000]:
            print "a===============", yr
            for job in df.loc[yr]['Job']:
                if df.loc[(yr, job), 'PA'] == df.loc[(yr, job), 'MA']:
                    df.loc[(yr, job), 'Skew_Dist'] = 0
                elif df.loc[(yr, job), 'PA'] < df.loc[(yr, job), 'MA']:
                    df.loc[(yr, job), 'Skew_Dist'] = min(skew_dic[job + str(yr)]) * -1
                else:
                    df.loc[(yr, job), 'Skew_Dist'] = min(skew_dic[job + str(yr)])


                skew_ind = np.array(skew_dic[job + str(yr)]).argmin()
                df.loc[(yr, job), 'pt'] = skew_dic[job + str(yr) + '_pt'][skew_ind]



        # df.index = [num for num in xrange(32)]   #reset original index ints




        df['sum'] = df['PA'] + df['MA']
        df['ratio'] = df['sum'] / df['pt']
        df['Skew_Dist'] = df['Skew_Dist'].astype(float)
        df['pt'] = df['pt'].astype(float)
        df['ratio'] = df['ratio'].astype(float)

        df2 = df.loc[2000]

def read_fft_text(text):
    stats_dic = cll.defaultdict(list, [])
    liz = []
    with open(text, 'r') as f:
        for line in f:
            liz.append(line)

    return liz

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

def calc_hp(df, char, job_up):
    """
    Calculate the HP of any character at any level, given the job they will be during the level up.
    INPUT: Stats DataFrame (DF), character (JobClass object), and job (str) during level up.
    OUTPUT: HP at target level (int)
    """

    # When leveling from a to b, say 1 to 2, we use the *lower* level in the calculation for the stat bonus.  Because of this, we start the loop on the level we currently are.  This also means we actually end our loop range on the level we are stopping at, not one greater as is normal for Python.  Hence, xrange(1, 31), for example, is for going from level 1 to level 31.

    # VELIUS CODE IS JUST AN EXAMPLE
    # Note: when Char already exists, skip HP assignment for lvl 1 and use current char's raw HP.
    # Velius HP
    start_lvl = 1
    end_lvl = 31
    velius_HPc = 12.
    base_low = 491520 / 163840.
    base_high = 524287 / 163840.
    hp_low = base_low
    hp_high = base_high
    bonus_low = base_low / (velius_HPc + 1)
    bonus_high = base_high / (velius_HPc + 1)

    print 'HP_low = {0:.0f} at lvl 1'.format(hp_low)
    for lvl in xrange(start_lvl, end_lvl):
        bonus_low += float((hp_low / float((velius_HPc + lvl))))
        hp_low += bonus_low
        print 'HP_low = {0:.0f} at lvl {1}'.format(hp_low, (lvl + 1))

    print '\nHP_high = {0:.0f} at lvl 1'.format(hp_high)
    for lvl in xrange(start_lvl, end_lvl):
        bonus_high += float((hp_high / float((velius_HPc + lvl))))
        hp_high += bonus_high
        print 'HP_high = {0:.0f} at lvl {1}'.format(hp_high, (lvl + 1))

    print "On average, Velius has {0:.0f} HP at lvl {1}".format(np.mean([hp_low, hp_high]), end_lvl)



    # Char HP
    start_lvl = 1
    end_lvl = 31
    velius_HPc = 12.
    base_low = 491520 / 163840.
    base_high = 524287 / 163840.
    hp_low = base_low
    hp_high = base_high
    bonus_low = base_low / (velius_HPc + 1)
    bonus_high = base_high / (velius_HPc + 1)

    print 'HP_low = {0:.0f} at lvl 1'.format(hp_low)
    for lvl in xrange(start_lvl, end_lvl):
        bonus_low += float((hp_low / float((velius_HPc + lvl))))
        hp_low += bonus_low
        print 'HP_low = {0:.0f} at lvl {1}'.format(hp_low, (lvl + 1))

    print '\nHP_high = {0:.0f} at lvl 1'.format(hp_high)
    for lvl in xrange(start_lvl, end_lvl):
        bonus_high += float((hp_high / float((velius_HPc + lvl))))
        hp_high += bonus_high
        print 'HP_high = {0:.0f} at lvl {1}'.format(hp_high, (lvl + 1))

    print "On average, Velius has {0:.0f} HP at lvl {1}".format(np.mean([hp_low, hp_high]), end_lvl)




    # Monk HP
    monk_HPc = 9.
    base_low = 491520 / 1638400.
    base_high = 524287 / 1638400.
    hp_low = base_low
    hp_high = base_high
    bonus_low = base_low / (monk_HPc + 1)
    bonus_high = base_high / (monk_HPc + 1)

    print 'HP_low =', hp_low, 'at lvl 1'
    for lvl in xrange(1, 32):
        bonus_low += float((hp_low / float((monk_HPc + lvl))))
        hp_low += bonus_low
        print 'HP =', round(hp_low, 0), 'at lvl', lvl

    print '\nHP_high =', hp_high, 'at lvl 1'
    for lvl in xrange(1, 32):
        bonus_high += float((hp_high / float((monk_HPc + lvl))))
        hp_high += bonus_high
        print 'HP =', round(hp_high, 0), 'at lvl', lvl




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
        print self.name + " hits " + str(other.name) + " for " + str(dmg) + " damage."
        return other.HP


    # def

# Job Attrs
#
# class Squire(object):
#     def __init__(self):
#         self.HP = 100
#         self.MP = 75
#         self.PA = 90
#         self.MA = 80
#         self.SPD = 100
#         self.Move = 4
#         self.Jump = 3
#         self.CEV = .05
#
# class Chemist(object):
#     def __init__(self):
#         self.HP = 80
#         self.MP = 75
#         self.PA = 75
#         self.MA = 80
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Knight(object):
#     def __init__(self):
#         self.HP = 120
#         self.MP = 80
#         self.PA = 120
#         self.MA = 80
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .10
#
# class Archer(object):
#     def __init__(self):
#         self.HP = 100
#         self.MP = 65
#         self.PA = 110
#         self.MA = 80
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .10
#
# class Monk(object):
#     def __init__(self):
#         self.HP = 135
#         self.MP = 80
#         self.PA = 129
#         self.MA = 80
#         self.SPD = 110
#         self.Move = 3
#         self.Jump = 4
#         self.CEV = .2
#
# class Priest(object):
#     def __init__(self):
#         self.HP = 80
#         self.MP = 120
#         self.PA = 90
#         self.MA = 110
#         self.SPD = 110
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Wizard(object):
#     def __init__(self):
#         self.HP = 75
#         self.MP = 120
#         self.PA = 60
#         self.MA = 150
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class TimeMage(object):
#     def __init__(self):
#         self.HP = 75
#         self.MP = 120
#         self.PA = 50
#         self.MA = 130
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Summoner(object):
#     def __init__(self):
#         self.HP = 70
#         self.MP = 125
#         self.PA = 50
#         self.MA = 125
#         self.SPD = 90
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Thief(object):
#     def __init__(self):
#         self.HP = 90
#         self.MP = 50
#         self.PA = 100
#         self.MA = 60
#         self.SPD = 110
#         self.Move = 4
#         self.Jump = 4
#         self.CEV = .25
#
# class Mediator(object):
#     def __init__(self):
#         self.HP = 80
#         self.MP = 70
#         self.PA = 75
#         self.MA = 75
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Oracle(object):
#     def __init__(self):
#         self.HP = 75
#         self.MP = 110
#         self.PA = 50
#         self.MA = 120
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Geomancer(object):
#     def __init__(self):
#         self.HP = 110
#         self.MP = 95
#         self.PA = 110
#         self.MA = 105
#         self.SPD = 100
#         self.Move = 4
#         self.Jump = 3
#         self.CEV = .10
#
# class Lancer(object):
#     def __init__(self):
#         self.HP = 120
#         self.MP = 50
#         self.PA = 120
#         self.MA = 50
#         self.SPD = 100
#         self.Move = 4
#         self.Jump = 3
#         self.CEV = .15
#
# class Samurai(object):
#     def __init__(self):
#         self.HP = 75
#         self.MP = 75
#         self.PA = 128
#         self.MA = 90
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .20
#
# class Ninja(object):
#     def __init__(self):
#         self.HP = 70
#         self.MP = 50
#         self.PA = 120
#         self.MA = 75
#         self.SPD = 120
#         self.Move = 4
#         self.Jump = 4
#         self.CEV = .30
#
# class Calculator(object):
#     def __init__(self):
#         self.HP = 65
#         self.MP = 80
#         self.PA = 50
#         self.MA = 70
#         self.SPD = 50
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Bard(object):
#     def __init__(self):
#         self.HP = 55
#         self.MP = 50
#         self.PA = 30
#         self.MA = 115
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Dancer(object):
#     def __init__(self):
#         self.HP = 60
#         self.MP = 50
#         self.PA = 110
#         self.MA = 95
#         self.SPD = 100
#         self.Move = 3
#         self.Jump = 3
#         self.CEV = .05
#
# class Mime(object):
#     def __init__(self):
#         self.HP = 140
#         self.MP = 50
#         self.PA = 120
#         self.MA = 115
#         self.SPD = 120
#         self.Move = 4
#         self.Jump = 4
#         self.CEV = .05
#



if __name__ == '__main__':
    # df = load_df()
    # dfplot = df.copy()
    # dfplot = dfplot.iloc[:-2, :] #Drop Onion Knight
    # df['HP'] = df['HP'] * 2
    # x, y = df['HP'], df['PA']
    # euc_dist_skew(df, x, y)
    # plot_fft_scatter(df, x, y)
    # m = JobClass('Monk', 33)

    txt = 'FFT_job_list_to_parse.txt'
    df = pd.read_table(txt)

    # Find split in TXT file for table creation
    for ind, row in enumerate(df.iloc[:,0]):
        if "####" in row:
            table_split = ind

    table1_end_idx = table_split
    table2_start_idx = table_split + 2


    # Create DF from Table 1
    dd = cll.defaultdict(list, [])
    for ind in df.index[0: table1_end_idx]:
        line = df.iloc[ind].str.split()[0]
        job_stop = len(line) - 9
        job = "_".join(line[1: job_stop])
        for word in line[1: job_stop]:
            line.remove(word)
        line.insert(1, job)
        dd[ind] = line
        # print len(line), line

        """
    # Create DF from Table 2
    dd = cll.defaultdict(list, [])
    for ind in df.index[table2_start_idx: ]:
        line = df.iloc[ind].str.split()[0]
        job_stop = len(line) - 6
        job = "_".join(line[1: job_stop])
        for word in line[1: job_stop]:
            line.remove(word)
        line.insert(1, job)
        dd[ind] = line

        print len(line), line

        """
    df = pd.DataFrame(dd)
    df = df.T

    # Assign columns
    cols = [col for col in df.iloc[0]]
    cols[0], cols[1], cols[2] = 'Job_ID', 'Job', 'Identity'
    df.columns = cols

    # # Remove all rows with sub-headers or dashed dividers
    # df = df[~(df['Job_ID'].str.len() > 3)]
    #
    # # Reindex DF with sequential indices, 0 - df.shape[0]
    # new_ind = df.index.reindex([x for x in xrange(df.shape[0])])[0]
    # df.index = new_ind

    # Create a Role column from the role the char had in FFT
    categories = ['STORY', 'GENERIC', 'MONSTER', 'HUMANOID']
    role = None
    for idx in df.index:
        for word in categories:
            print idx, df.iloc[idx, 0]
            if word in df.iloc[idx, 0]:
                role = word.capitalize()
        df.loc[idx, 'Role'] = role

    # Assign dictionary to Main
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

    # Remove all rows with sub-headers or dashed dividers
    df = df[~(df['Job_ID'].str.len() > 3)]

    # Reindex DF with sequential indices, 0 - df.shape[0]
    new_ind = df.index.reindex([x for x in xrange(df.shape[0])])[0]
    df.index = new_ind

    # Convert number columns to int/float - had to remove dashed rows 1st
    df['CEV'] = df['CEV'].str.strip('%')
    for col in df.iloc[:, 3:11]: # Range only for Table 1.
    # for col in df.iloc[:, 3:8]: # Range only for Table 2.
        df.loc[:, col] = df.loc[:, col].astype(int)

    df['CEV'] = df['CEV'] / 100

    # Commands to inspect different aspects of the units
    df.groupby('Association').mean().sort_values('PAm', ascending=False)
    df[df['Identity'] == 'generic']


    # Write to CSV that I can then load.  Need to maybe compile more first?
    # df.to_csv('~/Desktop/jp_fft_mult_table.csv')













    # job_dic = {job: Job(job, 1) for job in df.index}

    # A1 = Job('Archer', 'Robin')
    # A2 = job_dic.get('Archer', 'love')
    # M1 = job_dic['Monk']
    # Ramza = Job('Monk', 24, 'Ramza')
    # Delita = Job('Knight', 26, 'Delita')

    # for name in ['Ramza', 'Delita', 'Teta', 'Alma', 'Zalbag']:
    #     locals()["%s" %name] = Job('Squire', name)




    stn="STRIKING WEAPONS | Range: 1v3 (from above) / 1v2 (from below) ==============================================================================#Bare Hands**...... damage = [(PA * Br) / 100] * PA    XA = [(PA * Br) / 100] #Knife ............ damage = [(PA + Sp) / 2] * WP      XA = [(PA + Sp) / 2] #Ninja Sword ...... damage = [(PA + Sp) / 2] * WP      XA = [(PA * Sp) / 2] #Sword ............ damage = PA * WP                   XA = PA #Knight Sword ..... damage = [(PA * Br) / 100] * WP    XA = [(PA * Br) / 100] #Katana ........... damage = [(PA * Br) / 100] * WP    XA = [(PA * Br) / 100] #Staff ............ damage = MA * WP                   XA = MA #Rod .............. damage = PA * WP                   XA = PA #Flail ............ damage = (1..PA) * WP              XA = (1..PA) #Axe .............. damage = (1..PA) * WP              XA = (1..PA) #Bag .............. damage = (1..PA) * WP              XA = (1..PA) ------------------------------------------------------------------------------- ** If a unit is barehanded, weapon-elemental attacks will receive the bonus    from the Martial Arts support ability (they don't if the unit is equipped    with any weapon -- see 6.2). For barehanded units, WP = 0 in any equations    that use WP (e.g., BATTLE SKILL success rate)."

    # st = stn.split('#')
