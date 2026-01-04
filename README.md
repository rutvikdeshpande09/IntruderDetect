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


Changes Made:
Added video conversion function: Converts the recorded video to H.264 (MPEG-4 Part 10) using ffmpeg, which is compatible with email clients.
Modified stop_recording(): After recording stops, it:
Converts the video to H.264 format
Sends the converted video via email
Removes the original file to save space
Fixed MIME type: Changed from application/octet-stream to video/mp4 so email clients recognize it as a video file.
How it works now:
Records video using OpenCV (temporary format)
When recording stops, converts to H.264 MP4 using ffmpeg
Sends the converted video via email
The converted video should play in email clients


Changes Made:
Created process_and_send_recording() function: Handles video conversion and email sending separately.
Modified stop_recording(): Now starts a background thread to process and send the email, so it doesn't block the main detection loop.
Updated cleanup(): Prevents duplicate processing when the program exits.
How it works now:
When a person leaves, stop_recording() is called immediately.
The recording stops and saves the video file.
A background thread starts to convert the video and send the email.
The main loop continues monitoring for new intruders while the email is being processed.
The email is sent as soon as the conversion completes (usually within seconds).
The system continues monitoring while the email is processed in the background, so you'll receive the email notification right after the person leaves, even if the program keeps running.
