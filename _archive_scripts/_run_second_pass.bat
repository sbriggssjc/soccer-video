@echo off
cd /d D:\Projects\soccer-video
C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe -u _second_pass_portrait.py > _second_pass_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> _second_pass_log.txt
