a=system('a=`tempfile`;cat *.dat > $a;echo "$a"')

set xtic auto
set ytic auto
set xlabel 'time (s)'
set ylabel 'NLL'
plot '01_mnist_pytorch.txt' using 2:3 title 'loss'
#set term postscript landscape enhanced color dashed "Helvetica" 14
set term postscript
set output '01_mnist_pytorch.ps'
replot
set term png
set output '01_mnist_pytorch.png'
replot
set term dumb
set output '01_mnist_pytorch.g'
replot
#set term x11

