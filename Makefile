LATEX=/Library/TeX/texbin/pdflatex
TEX=main.tex

.PHONY: all clean

all: report/main.pdf

report/main.pdf: report/$(TEX)
	cd report && $(LATEX) -interaction=nonstopmode $(TEX) && $(LATEX) -interaction=nonstopmode $(TEX)

clean:
	rm -f report/*.aux report/*.log report/*.out report/*.pdf report/*.toc

