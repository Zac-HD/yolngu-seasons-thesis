@echo off
:: Get the latest code
git pull || pause

echo.
echo Words in thesis: (body/headings/other)
texcount -1 -inc -dir=tex\ tex\thesis.tex
echo.

:: Run analysis, if code has changed more recently than outputs
python code\main.py || pause

:: Do a clean build of the document
texify.exe --clean --batch --pdf --quiet --job-name="Zac_Thesis" "tex\thesis.tex" --run-viewer || pause
