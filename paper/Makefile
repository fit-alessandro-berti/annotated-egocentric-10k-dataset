LATEXMK ?= latexmk
LATEXFLAGS = -pdf -interaction=nonstopmode -halt-on-error -file-line-error
DOCUMENTS = main.pdf
ifneq ($(wildcard cover_letter.tex),)
DOCUMENTS += cover_letter.pdf
endif

.PHONY: all clean

all: $(DOCUMENTS)

main.pdf: main.tex references.bib figures/egocentric10k.png $(wildcard Definitions/*) $(wildcard sections/*.tex) $(wildcard generated/*.tex) $(wildcard generated/*prompt.txt)
	$(LATEXMK) $(LATEXFLAGS) main.tex

cover_letter.pdf: cover_letter.tex
	$(LATEXMK) $(LATEXFLAGS) cover_letter.tex

clean:
	$(LATEXMK) -c main.tex cover_letter.tex
