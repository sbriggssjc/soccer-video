@echo off
cd /d D:\Projects\soccer-video
C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe -u _render_v18.py > _v18_render_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> _v18_render_log.txt