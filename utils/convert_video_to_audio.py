import librosa,os
def convert_video_to_audio_librosa(video_path_in,video_name,audio_out):
    video=os.path.join(video_path_in,video_name)
    audio,sr=librosa.load(video)
    librosa.output.write_wav(audio_out, audio, sr)
        


        