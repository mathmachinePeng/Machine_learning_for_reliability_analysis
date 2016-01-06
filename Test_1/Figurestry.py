'''
Created on 28 Sep 2015

@author: peng
'''
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import RFclass
import Preprocessdata 
#import Preprocessdata1 as p
import MySVM as mysvc
import TAlogistic as tl
import cPickle, theano
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
import TAmlp as mlp


import TAdbn as dbn
import TAsda as sda
from scipy.interpolate import spline
import seaborn as sns
from IPython.core.pylabtools import figsize


#df_2 = pd.read_csv('/home/peng/git/Machine_learning_for_reliability_analysis//Test_1/score_long_10features_gbt.csv', header=0)

#------------------------ ind = np.arange(N)    # the x locations for the groups
#------- width = 0.35       # the width of the bars: can also be len(x) sequence
#------------------------------------------------------------------------------ 
#------------------- p1 = plt.bar(ind, bag,   width, color='r', label='Bagging')
#----------------- p2 = plt.bar(ind, adb, width, color='y', label='Adaboosting',
             #------------------------------------------------------ bottom=bag)
#------------------------- p3 = plt.bar(ind, rf, width, color='b', label = 'RF',
             #------------------------------------------------------ bottom=adb)
#----------------------- p4 = plt.bar(ind, ext, width, color='w', label = 'ERT',
             #------------------------------------------------------- bottom=rf)
#------------------------- p5 = plt.bar(ind, gbt, width, color='g', label='GTB',
             #------------------------------------------------------ bottom=ext)
#------------------------------------------------------------------------------ 

N = 5


ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars
acc = (0.69,  0.75, 0.75, 0.69,0.75)

fig, ax = plt.subplots(figsize = (8,6.5))
rects1 = ax.bar(ind, acc, width,color='c')

prec = (0.80,  0.83, 0.83, 0.71,0.83)
rects2 = ax.bar(ind + width, prec, width)
#add some text for labels, title and axes ticks
ax.set_xlabel('Classification metrics', fontsize=24)
#ax.set_title('Ensemble methods')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Bagging',  'RF', 'ERT', 'AdaBoosting','GTB'))
plt.ylim(0.5,1)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.ylabel('Scores', fontsize=24)

ax.legend((rects1[0], rects2[0]), ('Accuracy', 'Precision'),fontsize=20)


#===============================================================================
# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')
#===============================================================================




##plot the metrics comparison for 6633 splitting
#------------------------------------------------------------------------- N = 5
#-------------------------- ind = np.arange(N)  # the x locations for the groups
#------------------------------------- width = 0.2       # the width of the bars
#------------------------------------------ acc = (0.81, 0.78, 0.81, 0.80, 0.83)
#------------------------------------------------------------------------------ 
#------------------------------------- fig, ax = plt.subplots(figsize = (8,6.5))
#------------------------------------ rects1 = ax.bar(ind, acc, width,color='c')
#------------------------------------------------------------------------------ 
#----------------------------------------- prec = (0.82, 0.81, 0.82, 0.79, 0.85)
#------------------------------------- rects2 = ax.bar(ind + width, prec, width)
#------------------------------ # add some text for labels, title and axes ticks
#-------------------------- ax.set_ylabel('Classification metrics', fontsize=24)
#--------------------------------------------- #ax.set_title('Ensemble methods')
#---------------------------------------------------- ax.set_xticks(ind + width)
#------------ ax.set_xticklabels(('Bagging', 'AdaBoosting', 'RF', 'ERT', 'GTB'))
#--------------------------------------------------------------- plt.ylim(0.5,1)
#----------------------------------------------------- plt.yticks(fontsize = 14)
#----------------------------------------------------- plt.xticks(fontsize = 14)
#--------------------------------------------- plt.xlabel('Scores', fontsize=24)
#------------------------------------------------------------------------------ 
#------ ax.legend((rects1[0], rects2[0]), ('Accuracy', 'Precision'),fontsize=20)
# #===============================================================================
#----------------------------------------------------------- # autolabel(rects1)
#----------------------------------------------------------- # autolabel(rects2)
# #===============================================================================

plt.show()







cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
#df_2=  df_2[df_2['count']%2 ==0]

#print df_2.describe()






