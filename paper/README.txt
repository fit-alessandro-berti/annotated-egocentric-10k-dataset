Systems submission files

Manuscript: main.tex and main.pdf
Cover letter: cover_letter.tex and cover_letter.pdf

Keep the manuscript and cover letter as separate files. Do not create ZIP
archives for this paper.

Target journal: Systems (MDPI)
Article type: Article
Special issue: Data-Driven Business Processes and Decision Optimization
https://www.mdpi.com/journal/systems/special_issues/EGUYTR6HWI

Build

Run make in this directory to compile both PDFs. The build needs a full
TeX Live installation with pdfLaTeX, BibTeX, and latexmk. To build the
manuscript without make, run:

  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

To build the cover letter, use the same command with cover_letter.tex.
Upload the cover letter separately. Both documents remain editable in LaTeX.

Template

The Definitions directory contains MDPI's official class, bibliography
styles, and logos. Downloaded on 19 October 2026 from:
https://mdpi-res.com/data/MDPI_template.zip
The class identifies its version as 18 June 2025. The manuscript uses
the systems, article, submit, pdftex, and moreauthors options, with the
MDPI numeric bibliography style. Its page-range formatter has one local
change: it writes "to" instead of a dash. All other style functions and
template files are unchanged. The submit option adds review line numbers.
The manuscript hides the unassigned citation and DOI in the submission
copy. Publication identifiers are left to the editorial office.

Author declarations

The author roles follow the supplied description: Alessandro Berti carried
out the experiments; Wil M. P. van der Aalst reviewed the manuscript and
suggested improvements. The funding statement retains the funders and
grant number from the original paper. The conflicts statement discloses
the Celonis affiliation already listed in the original manuscript.

The ethics and consent statements describe reuse of an existing dataset.
No ethics approval, exemption, or participant consent record was found in
the project or in the public dataset card. The declarations do not assert
that approval was granted or that the original participants consented.

The cover letter includes the journal's publication-status and author-
approval declarations as part of the proposed letter for the authors.
No manuscript or letter has been submitted or sent to the editors.

Guidance consulted

https://www.mdpi.com/journal/systems/instructions
https://www.mdpi.com/authors/layout
https://www.mdpi.com/journal/systems/special_issues/EGUYTR6HWI
https://huggingface.co/datasets/builddotai/Egocentric-10K/blob/main/README.md
