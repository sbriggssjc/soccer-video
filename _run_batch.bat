@echo off
cd /d "D:\Projects\soccer-video"
C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe -u _batch_finalize.py > _batch_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> _batch_log.txt
