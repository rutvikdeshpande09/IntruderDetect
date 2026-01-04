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

What I Fixed:
Added frame conversion: Before writing to the video file, the code now:
Detects if the frame has 4 channels (XRGB8888 format)
Extracts the RGB channels (skipping the X/alpha channel)
Converts RGB to BGR format that VideoWriter expects
Writes the converted frame
Proper format handling: The code now handles both 3-channel and 4-channel frames correctly.
The video should now record properly. The warnings about "expected 3 channels but got 4" should be gone, and the video files will contain actual video data instead of being empty (0.00 MB).
When you run the program again, it should:
Record video successfully when an intruder is detected
Create video files with actual content
Convert and email the videos properly
Try running it again and the recording should work correctly now.

Changes Made:
Added timestamp to video frames: Each frame now includes a green timestamp (YYYY-MM-DD HH:MM:SS) in the top-left with a semi-transparent black background for readability.
Screen recording integration:
Screen recording is combined with the camera video side-by-side
Both videos are converted to H.264 before combining
The combined video is sent via email
Video processing:
Camera video is converted to H.264
Screen video is converted to H.264 (if it exists)
Both are combined side-by-side using ffmpeg
The final combined video is sent via email
How it works:
Timestamp: Appears on every frame of the camera video in green text with a dark background
Screen recording: Recorded simultaneously with the camera video
Combined output: The email will contain a single video showing:
Left side: Camera feed with timestamp overlay
Right side: Screen recording
The videos are automatically scaled to the same height and combined horizontally. If screen recording fails or isn't available, only the camera video (with timestamp) will be sent.
When you run the program, you'll see:
Timestamps on the live display
Timestamps embedded in the recorded video
A combined video in your email showing both camera and screen

Improvements:
Better error checking:
Verifies the video file exists before sending
Checks if the file is empty (0 bytes)
Validates file paths at each step
Detailed logging:
Logs each step of the email process
Shows file paths and sizes
Prints full error tracebacks for debugging
Return values:
send_recording_email() now returns True/False to indicate success
The processing function checks the return value
Error handling:
Handles specific SMTP errors separately
Shows clear messages for authentication failures
Continues even if some steps fail
What to check:
When you run the program, you'll see detailed console output showing:
Whether video files exist
File sizes
Each step of the email sending process
Any errors with full details
This should help identify why emails aren't being sent. Common issues might be:
Video file doesn't exist or is empty
SMTP authentication failure
Network connectivity issues
File too large for email

Fixes Applied:
Fixed blue camera issue:
Changed color conversion to use cv2.COLOR_RGBA2BGR for XRGB8888 format
Added fallback if the first conversion fails
This should fix the blue tint in recorded videos
Improved screen recording:
Added automatic screen resolution detection using xrandr
Added proper pixel format specification (yuv420p) for better compatibility
Added display offset to capture the correct screen area
Better status messages:
Added [SUCCESS] message when email is sent
Added file path information for debugging
Added clearer progress messages throughout the process
What to expect:
Camera video should have correct colors (no more blue tint)
Screen recording should work better with automatic resolution detection
You'll see clear status messages like:
[INFO] Video processing and email sending started in background...
[SUCCESS] Email sent successfully! Video was sent to [email]
Or error messages if something fails

What I've improved:
Better x11grab options: Tries multiple methods to capture the screen
Wayland detection: Warns if Wayland is detected (x11grab won't work)
Better error handling: Checks if the recording process starts successfully
The blank screen issue:
This happens because the compositor (window manager) draws the desktop in a way that x11grab can't capture directly. The cursor is visible because it's drawn separately.
Possible solutions:
Temporarily disable composition (may affect performance):
   # In terminal, try disabling composition   # This depends on your window manager
Use an alternative screen capture tool like scrot or import (ImageMagick) in a loop, though this is less efficient.
Accept the limitation: The camera video with timestamp should still work, and you'll at least see the cursor movement on screen.
The code will continue to work and record the camera video properly. The screen recording will capture the cursor movement, which can still be useful for seeing user interaction even if the background isn't visible.
If screen recording is critical, consider using a different screen capture method or disabling composition during recording.

Fixes:
Always attempts combination: Even if the screen video is small or blank, it still tries to combine (screen recording is critical).
Better error handling:
Returns None on failure instead of silently falling back
Checks if the combined file exists after creation
Shows detailed error messages from ffmpeg
Longer wait time: Waits 3 seconds for screen recording to finalize before processing.
Detailed logging: Shows file paths, sizes, and each step of the process.
Fallback handling: If combination fails, falls back to camera video only, but logs the error clearly.
Added -shortest flag: Handles cases where camera and screen videos have different durations.
What you'll see:
The console will show:
[INFO] Screen recording file found: ... (X bytes)
[INFO] Converting screen video to H.264...
[INFO] Combining camera and screen videos...
[SUCCESS] Videos combined successfully: ... or [ERROR] Video combination failed!
The code now always attempts to combine the screen recording when it exists, even if it's small or appears blank. If combination fails, you'll see clear error messages explaining why.
Run it again and check the console output to see what's happening with the screen recording combination.
