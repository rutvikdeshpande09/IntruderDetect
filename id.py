# Intruder Detection System
# Detects any person and records camera + screen while they are present
# Saves and emails recording when person leaves

import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import threading
import subprocess
import signal
import sys

# Email configuration - Update these with your email credentials
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "rutvikdeshpande11@gmail.com"
SENDER_PASSWORD = "vpch toji olin pfsc"
RECIPIENT_EMAIL = "rutvikdeshpande11@gmail.com"

# Configuration
cv_scaler = 4  # Frame scaling for performance
recordings_folder = "intruder_recordings"  # Folder to save recordings
min_detection_frames = 5  # Minimum frames to confirm detection
min_absence_frames = 30  # Frames to wait before considering person left (about 1 second at 30fps)
fps_target = 30  # Target FPS for recording

# Create recordings folder if it doesn't exist
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

# Initialize the camera
print("[INFO] Initializing camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()
time.sleep(2)  # Allow camera to warm up

# Global state variables
is_recording = False
person_detected = False
detection_frame_count = 0
absence_frame_count = 0
camera_writer = None
screen_recording_process = None
recording_start_time = None
current_recording_path = None
frame_width = 1920
frame_height = 1080
recording_lock = threading.Lock()

def detect_face(frame):
    """Detect if any face is present in the frame"""
    global detection_frame_count, absence_frame_count
    
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    
    if len(face_locations) > 0:
        detection_frame_count += 1
        absence_frame_count = 0
        
        # Confirm detection after minimum frames
        if detection_frame_count >= min_detection_frames:
            return True, face_locations
    else:
        absence_frame_count += 1
        detection_frame_count = 0
    
    return False, []

def start_recording():
    """Start recording camera and screen"""
    global is_recording, camera_writer, screen_recording_process, recording_start_time, current_recording_path
    
    with recording_lock:
        if is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"intruder_{timestamp}.mp4"
        current_recording_path = os.path.join(recordings_folder, recording_filename)
        
        # Initialize camera video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        camera_writer = cv2.VideoWriter(current_recording_path, fourcc, fps_target, (frame_width, frame_height))
        
        # Start screen recording in a separate thread
        screen_recording_process = start_screen_recording(timestamp)
        
        recording_start_time = datetime.now()
        is_recording = True
        print(f"[INFO] Recording started: {recording_filename}")

def start_screen_recording(timestamp):
    """Start screen recording using ffmpeg"""
    screen_filename = os.path.join(recordings_folder, f"screen_{timestamp}.mp4")
    
    try:
        # Try to record screen using ffmpeg
        # For Raspberry Pi with X11 display
        if os.environ.get('DISPLAY'):
            cmd = [
                'ffmpeg', '-y', '-f', 'x11grab',
                '-framerate', str(fps_target),
                '-s', '1920x1080',  # Adjust to your screen resolution
                '-i', os.environ.get('DISPLAY'),
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', screen_filename
            ]
        else:
            # Try framebuffer (for headless or direct framebuffer access)
            cmd = [
                'ffmpeg', '-y', '-f', 'fbdev',
                '-framerate', str(fps_target),
                '-i', '/dev/fb0',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', screen_filename
            ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        print(f"[INFO] Screen recording started: screen_{timestamp}.mp4")
        return process
    except Exception as e:
        print(f"[WARNING] Could not start screen recording: {str(e)}")
        print("[INFO] Continuing with camera recording only...")
        return None

def convert_video_to_h264(input_path, output_path):
    """Convert video to H.264 format compatible with email clients"""
    try:
        print(f"[INFO] Converting video to H.264 format...")
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '23', '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',  # Enable streaming/quick start
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[INFO] Video converted successfully: {output_path}")
            return True
        else:
            print(f"[WARNING] Video conversion failed: {result.stderr.decode()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Video conversion timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Video conversion error: {str(e)}")
        return False

def combine_videos(camera_video_path, screen_video_path, output_path):
    """Combine camera and screen videos side-by-side"""
    try:
        print(f"[INFO] Combining camera and screen recordings...")
        
        # Check if screen video exists
        if not os.path.exists(screen_video_path):
            print(f"[WARNING] Screen recording not found: {screen_video_path}")
            return camera_video_path  # Return camera video if screen doesn't exist
        
        # Check screen video file size (might be empty if recording failed)
        screen_size = os.path.getsize(screen_video_path)
        if screen_size < 1000:  # Less than 1KB, likely empty
            print(f"[WARNING] Screen recording appears to be empty ({screen_size} bytes)")
            return camera_video_path
        
        # Use ffmpeg to combine videos side-by-side
        # Scale both videos to same height and stack horizontally
        cmd = [
            'ffmpeg', '-y',
            '-i', camera_video_path,
            '-i', screen_video_path,
            '-filter_complex', 
            '[0:v]scale=960:-1[v0];[1:v]scale=960:-1[v1];[v0][v1]hstack=inputs=2[v]',
            '-map', '[v]',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"[INFO] Videos combined successfully: {output_path}")
            return output_path
        else:
            error_msg = result.stderr.decode()
            print(f"[WARNING] Video combination failed: {error_msg}")
            return camera_video_path  # Return camera video if combination fails
    except Exception as e:
        print(f"[ERROR] Error combining videos: {str(e)}")
        return camera_video_path  # Return camera video on error

def process_and_send_recording(video_path, start_time, screen_video_path=None):
    """Process video and send email in background thread"""
    try:
        # Wait a moment for screen recording to finish writing
        if screen_video_path:
            time.sleep(1)  # Give screen recording time to finalize
        
        # Create output path for converted video
        base_name = os.path.splitext(video_path)[0]
        converted_path = f"{base_name}_h264.mp4"
        
        # First convert camera video to H.264
        if convert_video_to_h264(video_path, converted_path):
            final_video_path = converted_path
            
            # If screen recording exists, combine with camera video
            if screen_video_path and os.path.exists(screen_video_path):
                # Convert screen video to H.264 first if needed
                screen_base = os.path.splitext(screen_video_path)[0]
                screen_converted = f"{screen_base}_h264.mp4"
                
                if convert_video_to_h264(screen_video_path, screen_converted):
                    # Combine the converted videos
                    combined_path = f"{base_name}_combined_h264.mp4"
                    final_video_path = combine_videos(converted_path, screen_converted, combined_path)
                    
                    # Clean up intermediate files
                    if final_video_path == combined_path:
                        try:
                            os.remove(converted_path)
                            os.remove(screen_converted)
                        except:
                            pass
                else:
                    # Try combining with original screen video
                    combined_path = f"{base_name}_combined_h264.mp4"
                    final_video_path = combine_videos(converted_path, screen_video_path, combined_path)
                    if final_video_path == combined_path:
                        try:
                            os.remove(converted_path)
                        except:
                            pass
            
            # Send the final video
            send_recording_email(final_video_path, start_time)
            
            # Optionally remove original files to save space
            try:
                os.remove(video_path)
                if screen_video_path and os.path.exists(screen_video_path):
                    os.remove(screen_video_path)
                print(f"[INFO] Removed original files")
            except:
                pass
        else:
            # If conversion fails, try sending original (might not work in email)
            print("[WARNING] Sending original video (may not be compatible with email clients)")
            send_recording_email(video_path, start_time)
    except Exception as e:
        print(f"[ERROR] Error processing recording: {str(e)}")

def stop_recording():
    """Stop recording and save files"""
    global is_recording, camera_writer, screen_recording_process, current_recording_path, recording_start_time
    
    with recording_lock:
        if not is_recording:
            return
        
        is_recording = False
        
        # Get screen recording path before stopping
        screen_video_path = None
        if screen_recording_process is not None and recording_start_time:
            timestamp = recording_start_time.strftime("%Y%m%d_%H%M%S")
            screen_video_path = os.path.join(recordings_folder, f"screen_{timestamp}.mp4")
        
        # Stop camera recording
        if camera_writer is not None:
            camera_writer.release()
            camera_writer = None
            print(f"[INFO] Camera recording saved: {current_recording_path}")
        
        # Stop screen recording
        if screen_recording_process is not None:
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(screen_recording_process.pid), signal.SIGTERM)
                screen_recording_process.wait(timeout=5)
                print("[INFO] Screen recording stopped")
            except Exception as e:
                print(f"[WARNING] Error stopping screen recording: {str(e)}")
            screen_recording_process = None
        
        # Calculate recording duration
        if recording_start_time:
            duration = (datetime.now() - recording_start_time).total_seconds()
            print(f"[INFO] Recording duration: {duration:.2f} seconds")
        
        # Process video and send email in background thread (non-blocking)
        if current_recording_path and os.path.exists(current_recording_path):
            video_path = current_recording_path
            start_time = recording_start_time
            
            # Start background thread to convert and send email
            email_thread = threading.Thread(
                target=process_and_send_recording,
                args=(video_path, start_time, screen_video_path),
                daemon=True
            )
            email_thread.start()
            print("[INFO] Video processing and email sending started in background...")
        
        current_recording_path = None
        recording_start_time = None

def send_recording_email(video_path, start_time):
    """Send recording via email"""
    print(f"[INFO] Preparing to send email with recording...")
    
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        
        timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        msg['Subject'] = f"Intruder Detected - {timestamp_str}"
        
        # Email body
        body = f"""Intruder Detection Alert

An intruder was detected and recorded.

Detection Time: {timestamp_str}
Recording Duration: {(datetime.now() - start_time).total_seconds():.2f} seconds

The video recording is attached to this email.

This is an automated message from the Raspberry Pi Intruder Detection System.
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach video file
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
            print(f"[INFO] Video file size: {file_size:.2f} MB")
            
            # Gmail has a 25MB attachment limit, warn if too large
            if file_size > 20:
                print(f"[WARNING] Video file is large ({file_size:.2f} MB). Email may fail.")
                print("[INFO] Consider compressing the video or using a file sharing service.")
            
            with open(video_path, 'rb') as f:
                part = MIMEBase('video', 'mp4')  # Use proper MIME type for MP4
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(video_path)}'
                )
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        
        print(f"[INFO] Email sent successfully to {RECIPIENT_EMAIL}")
        
    except smtplib.SMTPAuthenticationError:
        print(f"[ERROR] Authentication failed. Please check your email and password.")
        print(f"[INFO] For Gmail, you need to use an App Password, not your regular password.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {str(e)}")

def cleanup():
    """Cleanup resources"""
    global camera_writer, screen_recording_process, is_recording
    
    print("[INFO] Cleaning up...")
    
    # Only stop recording if still recording (don't process again if already stopped)
    if is_recording:
        # Just stop the recording, don't process/send email on cleanup
        is_recording = False
        if camera_writer is not None:
            camera_writer.release()
            camera_writer = None
        if screen_recording_process is not None:
            try:
                os.killpg(os.getpgid(screen_recording_process.pid), signal.SIGTERM)
                screen_recording_process.wait(timeout=2)
            except:
                pass
            screen_recording_process = None
    
    # Clean up any remaining resources
    if camera_writer is not None:
        camera_writer.release()
    
    if screen_recording_process is not None:
        try:
            os.killpg(os.getpgid(screen_recording_process.pid), signal.SIGTERM)
        except:
            pass
    
    cv2.destroyAllWindows()
    picam2.stop()

# Register cleanup on exit
import atexit
atexit.register(cleanup)

# Main loop
print("[INFO] Intruder Detection System started")
print("[INFO] Monitoring for intruders...")
print("[INFO] Press 'q' to quit")

frame_count = 0
start_time = time.time()
fps = 0

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()
        
        # Detect face
        detected, face_locations = detect_face(frame)
        
        # Update state based on detection
        if detected and not person_detected:
            # Person just appeared
            person_detected = True
            print("[ALERT] Intruder detected! Starting recording...")
            start_recording()
        elif not detected and person_detected:
            # Check if person has been absent long enough
            if absence_frame_count >= min_absence_frames:
                # Person has left
                person_detected = False
                print("[INFO] Intruder left. Stopping recording...")
                stop_recording()
                print("[INFO] Monitoring resumed...")
        
        # Record frame if recording
        if is_recording and camera_writer is not None:
            # Convert from XRGB8888 (4 channels) to BGR (3 channels) for video writer
            # XRGB8888 format from Picamera2: channels are [X, R, G, B] where X is unused
            if len(frame.shape) == 3 and frame.shape[2] == 4:  # XRGB8888 format (4 channels)
                # Extract RGB channels (skip the X channel at index 0) and convert to BGR
                rgb_frame = frame[:, :, 1:4]  # Get R, G, B channels
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:  # Already 3 channels
                # Assume it's RGB and convert to BGR
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = frame
            
            # Add timestamp to the video frame
            current_time = datetime.now()
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Draw timestamp on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (0, 255, 0)  # Green color
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
            
            # Draw semi-transparent background for better visibility
            overlay = bgr_frame.copy()
            cv2.rectangle(overlay, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, bgr_frame, 0.4, 0, bgr_frame)
            
            # Draw timestamp text
            cv2.putText(bgr_frame, timestamp_str, (15, 35), font, font_scale, color, thickness)
            
            camera_writer.write(bgr_frame)
        
        # Draw detection box
        display_frame = frame.copy()
        if detected and len(face_locations) > 0:
            for (top, right, bottom, left) in face_locations:
                # Scale back up
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                
                # Draw red box around face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 3)
                
                # Draw label
                cv2.rectangle(display_frame, (left - 3, top - 35), (right + 3, top), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, "INTRUDER", (left + 6, top - 6), font, 1.0, (255, 255, 255), 2)
        
        # Display status
        status_text = "RECORDING" if is_recording else "MONITORING"
        status_color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Intruder Detection', display_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

finally:
    cleanup()
    print("[INFO] Intruder Detection System stopped")

