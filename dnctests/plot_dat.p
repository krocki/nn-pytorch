FILES = system("ls -1 ./*.dat")
LABEL = system("ls -1 ./*.dat | sed -e 's/\_//' -e 's/.dat//'")
TITLE = 'PTB'

set xtic auto
set ytic auto
set xlabel 'time (s)'
set ylabel 'NLL'
set title TITLE
plot for [i=i=1:words(FILES)] word(FILES,i) u 2:3 w p pointtype (i) lt rgb 'black' title word(LABEL,i) noenhanced
