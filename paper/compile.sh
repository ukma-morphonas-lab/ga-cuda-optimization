#!/bin/bash
# Compile merged.tex into a PDF
# Requirements: pdflatex, biber

set -e

cd "$(dirname "$0")"

echo "Pass 1/4: pdflatex (initial)..."
pdflatex -interaction=nonstopmode merged.tex > /dev/null

echo "Pass 2/4: biber (bibliography)..."
biber merged > /dev/null

echo "Pass 3/4: pdflatex (resolve references)..."
pdflatex -interaction=nonstopmode merged.tex > /dev/null

echo "Pass 4/4: pdflatex (resolve TOC page numbers)..."
pdflatex -interaction=nonstopmode merged.tex > /dev/null

echo "Done: merged.pdf"
