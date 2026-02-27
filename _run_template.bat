@echo off
cd /d D:\Projects\soccer-video
C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe -u _template_filter.py > _template_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> _template_log.txt
