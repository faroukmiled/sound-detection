from moviepy.editor import VideoFileClip, TextClip
import cv2
# Load the video file
video_file = "AMIGO-ROBOT_COTE.mp4"
video = cv2.VideoCapture(video_file)
ret,frame=video.read()
height, width, _ = frame.shape
video.release()
video = cv2.VideoCapture(video_file)
fps = video.get(cv2.CAP_PROP_FPS)
i=0
target_sound_start_times = [0.0, 1.77, 3.54, 5.31, 7.08, 12.39, 14.16, 15.93, 19.47, 21.24, 23.01, 24.78, 26.55, 28.32, 30.09, 31.86, 33.63, 35.4, 37.17]
j=0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
text_color = (255, 255, 255)  # White color
text_thickness = 2
text_position = (10, 30)  # Left top corner position
output_file = "out.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
while True:
    i+=1
    ret,frame=video.read()
    
    if not ret:
        break
    temps_en_secondes=(i/fps)
    if j<len(target_sound_start_times) and temps_en_secondes>=target_sound_start_times[j]:
        j=j+1
    height, width, _ = frame.shape
    text=str(j)
    cv2.putText(frame, text, text_position, font, font_scale, text_color, text_thickness)
    video_writer.write(frame)


video.release()
video_writer.release()


# Define the starting times of the target sound in seconds


# Define the counter text properties