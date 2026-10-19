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

Journal revision (19 October 2026)

The paper is split into editable files under sections/. Revision notes,
claim evidence, the 28-item completion map, source metadata, and a dataset
card are under revision/. Longer Markdown drafts in docs/ are historical
working material and are superseded by main.tex and sections/.

Reproduce descriptive calculations, manifests, and appendix tables:

  python3 paper/reproduce.py

Run this command from the repository root, or use an absolute script path.
It has no third-party dependencies and does not call model APIs. Use
--root and --output-dir to change the repository and output locations.
The manifest hashes the five frozen output layers. It also verifies all
65 merged files against the released merge algorithm, entirely in memory.

To repeat the original activity-level discovery configuration, use an
isolated environment with the documented recalculation versions:

  python3 -m venv .venv-assessment
  .venv-assessment/bin/python -m pip install -r paper/requirements-assessment.txt
  .venv-assessment/bin/python paper/docs/assessment_materials/assess.py
  .venv-assessment/bin/python paper/docs/assessment_materials/value_analysis.py
  .venv-assessment/bin/python paper/docs/assessment_materials/pm_utility.py
  python3 paper/reproduce.py

The last command refreshes the typeset discovery table from its saved JSON.
The checked-in discovery result lets the standard-library workflow prepare
that table without rerunning discovery. The miner, noise threshold (0.2),
activity labels, and video cases match the original script. The current
PM4Py version is recorded. The original dependency versions are unknown.
The original scripts take --output-dir; assess.py also takes --root.

The revision retains the historical event-end exclusion rule and category
map. It does not rescale timestamps or regenerate model outputs. Changes to
metric definitions, original discrepancies, and unresolved author queries
are documented in revision/REVISION_NOTES.txt. All 84 audit verdicts remain
unchanged. Process-level precision 0.59 is removed for lack of evidence.

Regeneration through APIs is separate from this offline workflow. Scripts
01 to 04 already accept input/output roots, prompt paths, and model options.
S1 to S3 use requests with Gemini; S4 uses requests with OpenAI; S5 is local.
Use their --help outputs to inspect options. Source access and credentials
are required only for regeneration. Current default model strings and
prompt hashes do not prove historical execution settings.
