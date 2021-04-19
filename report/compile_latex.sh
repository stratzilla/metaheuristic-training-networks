#!/bin/bash

printf "\nCompiling $1... ";

pdflatex -halt-on-error -interaction=nonstopmode $1.tex > $1.txt
grep '^!.*' --color=never $1.txt

bibtex $1.aux > $1.txt
grep '^!.*' --color=never $1.txt

pdflatex -halt-on-error -interaction=nonstopmode $1.tex > $1.txt
grep '^!.*' --color=never $1.txt

pdflatex -halt-on-error -interaction=nonstopmode $1.tex > $1.txt
grep '^!.*' --color=never $1.txt

rm -f $1.txt $1.aux $1.bbl $1.blg $1.log $1.out $1.toc

printf "done!\n\n";