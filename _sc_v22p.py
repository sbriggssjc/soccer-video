import py_compile
py_compile.compile('tools/ball_telemetry.py', doraise=True)
print('ball_telemetry.py: OK')
py_compile.compile('tools/render_follow_unified.py', doraise=True)
print('render_follow_unified.py: OK')
