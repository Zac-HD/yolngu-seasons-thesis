@echo off
echo.
echo Words in thesis: (body/headings/other)
echo.
texcount -1 -inc -dir=tex\ tex\thesis.tex
echo.
timeout 5 >nul
