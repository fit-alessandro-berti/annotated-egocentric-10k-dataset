LATEXMK ?= latexmk
LATEXFLAGS = -pdf -Werror -interaction=nonstopmode -halt-on-error -file-line-error
DOCUMENTS = main.pdf additional_material.pdf
ifneq ($(wildcard cover_letter.tex),)
DOCUMENTS += cover_letter.pdf
endif

.PHONY: all clean manuscript

all: $(DOCUMENTS)

main.pdf additional_material.pdf: manuscript

# Both documents import labels from the other. Build twice in sequence so a
# clean checkout and later edits both resolve the cross-document references.
manuscript: main.tex additional_material.tex references.bib latexmkrc figures/egocentric10k.png $(wildcard Definitions/*) $(wildcard sections/*.tex) $(wildcard generated/*.tex) $(wildcard generated/*prompt.txt)
	$(LATEXMK) $(LATEXFLAGS) main.tex
	$(LATEXMK) $(LATEXFLAGS) additional_material.tex
	$(LATEXMK) $(LATEXFLAGS) main.tex
	$(LATEXMK) $(LATEXFLAGS) additional_material.tex

cover_letter.pdf: cover_letter.tex latexmkrc
	$(LATEXMK) $(LATEXFLAGS) cover_letter.tex

clean:
	$(LATEXMK) -c main.tex additional_material.tex cover_letter.tex
