# extended_results/ — the extended results

This folder holds a longer version of the supplementary material: 17 notes, Tables S1–S29 and Figs. S1–S4. It reports analyses that the manuscript does not include. Examples are conformal prediction sets, a horseshoe prior on the trend innovations, control charts that are designed under the negative binomial, the tail risk of the expected cost, a second cost model and the dispersion of window counts under a Hawkes process.

`supplementary.tex` is the main file. It inputs the other LaTeX files, and the main README lists the five scripts that write them from the archived result files. `tables/` holds every table as a CSV file, and `figs/` holds the four figures.

## Numbering

The extended results keep their own numbers for notes, tables and figures. They cite the main text by the numbers of the longer version of the manuscript for which they were written. Two tables convert these numbers:

- For supplementary notes and tables, see `../supplementary/README.md`. It gives, for each note and table of the supplementary material of the manuscript, the item of the extended results from which it is taken.
- For sections, tables, figures and equations of the main text, see the section "Working names and numbers" of the main README.
