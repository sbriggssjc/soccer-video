@echo off
cd /d D:\Projects\soccer-video
"C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe" "_batch_render.py" > "_tmp\batch_stdout.txt" 2> "_tmp\batch_stderr_latest.txt"
echo EXITCODE=%ERRORLEVEL% >> "_tmp\batch_stdout.txt"
