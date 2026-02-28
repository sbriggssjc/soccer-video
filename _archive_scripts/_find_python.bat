@echo off
where python > "D:\Projects\soccer-video\_python_path.txt" 2>&1
where python3 >> "D:\Projects\soccer-video\_python_path.txt" 2>&1
dir "C:\Users\Scott\AppData\Local\Programs\Python\Python*\python.exe" >> "D:\Projects\soccer-video\_python_path.txt" 2>&1
dir "C:\Python*\python.exe" >> "D:\Projects\soccer-video\_python_path.txt" 2>&1
