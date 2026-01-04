# IntruderDetect

Features
Intruder detection: Detects any face (not just known faces)
Dual recording:
Camera video recording while a person is detected
Screen recording (via ffmpeg)
State management: Starts recording when a person appears, stops when they leave
Email alerts: Saves the recording and emails it when the person leaves
Multiple sessions: Restarts recording if the person returns
How it works
Monitors the camera feed continuously
When a face is detected, starts recording camera and screen
When the person leaves (no detection for ~1 second), stops recording and emails the video
If the person returns, starts a new recording session
Configuration
The program uses your existing email settings. You may need to:
Install ffmpeg for screen recording:
   sudo apt-get install ffmpeg
Adjust screen resolution (line 99) if your Raspberry Pi display is not 1920x1080
The program saves recordings in the intruder_recordings folder
Usage
Run the program:
python "Intruder Detect.py"
Press 'q' to quit. The program will automatically:
Record when intruders are detected
Save and email recordings when they leave
Resume monitoring for the next detection
The program displays a live feed with "INTRUDER" labels and a "RECORDING" status indicator when active.
