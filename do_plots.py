#! /usr/bin/python

import sys
import os

if len(sys.argv) != 4:
    print(sys.argv)
    print("Usage error: %s <datafile.csv> <plot title> <output plots path>" % (sys.argv[0]))
    sys.exit(-1)
    
datafile   = open(sys.argv[1], "r")
plot_title = sys.argv[2]
plotspath  = sys.argv[3]

plot_1="gnuplot_im2_conv_shalom"
plot_2="gnuplot_blis"

if not os.path.isdir(plotspath):
    os.mkdir(plotspath)
elif len(os.listdir(plotspath)) != 0:
    for f in os.listdir(plotspath):
        os.remove(os.path.join(plotspath, f))

bestfile = open(plotspath+"/best.csv", "w")
maxGflops=0.0
skip=True
for f in datafile:
    values=f.split(";")
    if skip:        
        bestfile.write("#layer;best(im2-conv);tze-meng;shalom;best(blis)\n")
        skip = False
        continue

    layer     = int(values[0])
    im2col    = float(values[1])
    convgemm  = float(values[2])
    shalom    = float(values[3])
    blis_8x12 = float(values[4])
    blis_4x20 = float(values[5])
    blis_blis = float(values[6])
    tzemeng   = float(values[7])

    maxGflops = max(im2col, convgemm, shalom, blis_8x12, blis_4x20, blis_blis, tzemeng, maxGflops)

    bestfile.write("%d;%.2f;%.2f;%.2f;%.2f\n" % (layer, max(im2col, convgemm), tzemeng, shalom, max(blis_blis, blis_8x12, blis_4x20) ))

bestfile.close()

#print("MAX=%.2f" %(maxGflops))

maxGflops += 2

#ResNet50-v1.5 - Single Carmel core \\@ 2.3 GHz\" font \"Helvetica-Bold,90
print( "Generating gnuplots Scripts...")

plot_blis  = open(plotspath+"/plot_blis.p", "w")
plot_best  = open(plotspath+"/plot_best.p", "w")

gnu_str = " set terminal epscairo enhanced color size 120in,30in font \"Helvetica, 90\" \n\
 set style data histograms \n\
 set style histogram cluster gap 1.5 \n\
 set style fill solid \n\
 set boxwidth 1 \n\
 set grid ytics \n\
 set key top left box Left reverse width 1 \n\
 set yrange [0:"+str(int(maxGflops))+"] \n\
 set title \""+plot_title+"\" \n\
 set datafile separator \";\" \n\
 set ylabel font \",90\" \"GFLOPS\" \n\
 set xlabel font \",90\" \"#CNN Layer\" \n"

plot_blis.write(gnu_str)
plot_best.write(gnu_str)

gnu_str = " set output \""+plotspath+"/plot_best.eps\" \n\
 plot \""+plotspath+"/best.csv"+"\" using 2:xtic(1) t \"Im2col vs ConvGemm\" lc 1, \
 '' u 3 t \"Tze-Meng\" lc 2, \
 '' u 4 t \"Shalom\" lc 3, \
 '' u 5 t \"Blis mk-8x12 vs Blis mk-4x20 vs Blis mk-blis \" lc 4,\n"

plot_best.write(gnu_str)

gnu_str = "set output \""+plotspath+"/plot_blis.eps\" \n\
 plot \""+sys.argv[1]+"\" using 5:xtic(1) t \"Blis-mk-8x12\" lc 4,\
 '' u 6 t \"Blis-mk-4x20\" lc 5,\
 '' u 7 t \"Blis-mk-blis\" lc 6,\n"

plot_blis.write(gnu_str)

plot_blis.close()
plot_best.close()

os.system("gnuplot "+plotspath+"/plot_blis.p")
os.system("gnuplot "+plotspath+"/plot_best.p")
