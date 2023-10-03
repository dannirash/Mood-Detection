import subprocess
import platform

def open_terminal(command):
    subprocess.Popen(['start', 'cmd', '/k', command], shell=True)

# Command to run Flask in the "backend" directory
flask_command = 'cd server && python -m flask --app server run'

# Command to run npm start
npm_command = 'cd client && npm start'

# Open two terminal windows with the specified commands
open_terminal(flask_command)
open_terminal(npm_command)