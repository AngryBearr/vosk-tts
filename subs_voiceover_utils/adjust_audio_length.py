from pydub import AudioSegment
from pydub.effects import speedup
import subprocess

def adjust_audio_speed_pydub(audio_file_path, save_file_path, file_name, start, end, min_speed, max_speed):
    audio = AudioSegment.from_wav(audio_file_path)
    target_duration = (end - start) / 1000  # Convert to seconds
    current_duration = len(audio) / 1000  # Convert to seconds

    speed_adjustment = current_duration / target_duration

    if speed_adjustment < min_speed:
        speed_adjustment = min_speed
    elif speed_adjustment > max_speed:
        speed_adjustment = max_speed

    adjusted_audio = speedup(audio, playback_speed=speed_adjustment)
    adjusted_audio.export(f'{save_file_path}/{file_name}', format="wav")

def change_speed(audio_file_path, save_file_path, file_name, start, end, min_speed, max_speed, sample_rate):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)
    
    # Calculate the speed adjustment
    current_duration = len(audio) / 1000  # Convert to seconds
    target_duration = (end - start) / 1000  # Convert to seconds
    speed_adjustment = current_duration / target_duration

    if speed_adjustment < min_speed:
        speed_adjustment = min_speed
    elif speed_adjustment > max_speed:
        speed_adjustment = max_speed
    
    print(f"Speed adjustment factor: {speed_adjustment}")

    # Prepare the FFmpeg command
    input_file = audio_file_path
    output_file = f"{save_file_path}/{file_name}"
    
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_file,
        "-filter:a", f"atempo={speed_adjustment}",
        "-ar", f'{sample_rate}',  # Set the output sample rate to match the input
        "-acodec", "pcm_s16le",  # Use 16-bit PCM codec for WAV
        "-y",  # Overwrite output file if it exists
        output_file
    ]

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)
        print(f"Speed adjusted audio saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")
