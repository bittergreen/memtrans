from moviepy import VideoFileClip, concatenate_videoclips
import os

def remove_video_segment(input_path, output_path, start_time, end_time):
    """
    Remove a segment from the input video between specified timestamps and concatenate the remaining parts.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path where the output video will be saved
        start_time (str): Start time of segment to remove in format "HH:MM:SS"
        end_time (str): End time of segment to remove in format "HH:MM:SS"
    """
    # Convert time strings to seconds
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    
    # Load the video with audio
    video = VideoFileClip(input_path, audio=True)
    
    # Create two clips: before and after the segment to remove
    first_part = video.subclipped(0, start_seconds)
    second_part = video.subclipped(end_seconds)
    
    # Concatenate the remaining parts with audio
    final_video = concatenate_videoclips([first_part, second_part])
    
    # Write the output file with audio
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Close the video objects
    video.close()
    first_part.close()
    second_part.close()
    final_video.close()

if __name__ == "__main__":
    # Example usage
    input_video = "meeting_01.mp4"  # Replace with your input video path
    output_video = "output2.mp4"  # Replace with your desired output path
    
    # Remove segment from 00:26:27 to 00:44:00
    remove_video_segment(input_video, output_video, "00:00:27", "02:07:00") 
    # remove_video_segment(input_video, output_video, "00:26:27", "00:44:00") 