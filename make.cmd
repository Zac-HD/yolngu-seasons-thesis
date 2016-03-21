@echo off
:: Get the latest code
git pull || pause

:: Run analysis, if code has changed more recently than outputs
cd code
python main.py || pause
cd ..

:: Do a clean build of the document
texify.exe --clean --batch --pdf --quiet --job-name="Zac_Thesis" "tex\thesis.tex" --run-viewer || pause
