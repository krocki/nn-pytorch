FILES = system("ls -1 ./ptb*.dat")
LABEL = system("ls -1 ./ptb*.dat | sed -e 's/\_//' -e 's/.dat//'")
TITLE = 'PTB'

set xtic auto
set ytic auto
set xlabel 'time (s)'
set ylabel 'NLL'
set title TITLE
plot for [i=i=1:words(FILES)] word(FILES,i) u 2:3 w p pointtype (i) lt rgb 'black' title word(LABEL,i) noenhanced
fname_ps = sprintf("./plots/%s.ps",TITLE)
fname_png = sprintf("./plots/%s.png",TITLE)
fname_g = sprintf("./plots/%s.g",TITLE)
/* set term postscript */
/* set output fname_ps */
/* replot */
/* set term png */
/* set output fname_png */
/* replot */
/* set term dumb */
/* set output fname_g */
/* replot */
