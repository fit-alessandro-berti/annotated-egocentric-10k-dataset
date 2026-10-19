LATEXMK ?= latexmk
LATEXFLAGS = -pdf -interaction=nonstopmode -halt-on-error -file-line-error
DOCUMENTS = main.pdf
ifneq ($(wildcard cover_letter.tex),)
DOCUMENTS += cover_letter.pdf
endif

.PHONY: all clean submission

all: $(DOCUMENTS)

main.pdf: main.tex references.bib figures/egocentric10k.png $(wildcard Definitions/*)
	$(LATEXMK) $(LATEXFLAGS) main.tex

cover_letter.pdf: cover_letter.tex
	$(LATEXMK) $(LATEXFLAGS) cover_letter.tex

# Keep the cover letter separate from the manuscript source upload.
submission: all
	python3 -c 'from pathlib import Path; from zipfile import ZipFile, ZIP_DEFLATED; paths = [Path("main.tex"), Path("references.bib"), Path("main.bbl"), Path("Makefile"), Path("README.txt"), *sorted(Path("Definitions").glob("*")), *sorted(Path("figures").glob("*"))]; z = ZipFile("systems_submission.zip", "w", ZIP_DEFLATED); [z.write(p, p.as_posix()) for p in paths if p.is_file()]; z.close()'

clean:
	$(LATEXMK) -c main.tex cover_letter.tex
