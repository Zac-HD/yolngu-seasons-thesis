@echo off
texify.exe --clean --batch --pdf --quiet --job-name="Zac_Thesis" "tex\thesis.tex" --run-viewer || pause
