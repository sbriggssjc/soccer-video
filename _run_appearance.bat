@echo off
cd /d D:\Projects\soccer-video
C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe -u _appearance_filter.py > _appearance_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> _appearance_log.txt
