import whisper
import pandas as pd
import os

def get_speech_to_text(audio_path, language=None):
    """
    Transcribes an audio file using Whisper and returns a pandas DataFrame with segmented text.

    :param audio_path: Path to the audio file.
    :param language: Optional language parameter for the Whisper model.
    :return: DataFrame containing transcribed text segments.
    """

    #loading the model
    model = whisper.load_model("turbo")

    #transcribe the text with desired language if provided
    result = model.transcribe(audio_path, language=language) if language else model.transcribe(audio_path)

    #extract text segments
    text_segmented = []
    for segment in result["segments"]:
        text_segmented.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    #conver to pandas dataframe
    df = pd.DataFrame(text_segmented)

    return df


#list of audio file paths
audio_files = [
    "C:\\Users\\avyan\\OneDrive\\Desktop\\pod.mp3",
    "C:\\Users\\avyan\\OneDrive\\Desktop\\tmc_pod.mp3",
    "C:\\Users\\avyan\\OneDrive\\Desktop\\mkbhd.mp3"
]

#initialize empty master df
master_df = pd.DataFrame(columns=["start", "end", "text", "file"])

#loop through files and append transcriptions
for file_path in audio_files:
    try:
        df = get_speech_to_text(file_path)
        master_df = pd.concat([master_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

#save to csv
master_df.to_csv("all_transcripts.csv", index=False)

#display a preview
print(master_df.head())
print(master_df.shape)