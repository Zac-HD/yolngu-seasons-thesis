@echo off
:: Do a clean build, and exit on success - but show log on error
texify.exe --clean --batch --pdf --quiet --job-name="_Zac_Thesis" "tex\thesis.tex" || pause
