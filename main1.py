import sounddevice as sd
import numpy as np
import openai
import base64

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

def record_audio(duration, sample_rate=16000):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait for recording to complete
    print("Recording finished.")
    return audio

if __name__ == "__main__":
    duration = 20  # Duration in seconds

    # Record audio
    audio_data = record_audio(duration)

    # Convert audio to base64 format
    audio_base64 = base64.b64encode(audio_data.tobytes()).decode("utf-8")

    # Perform speech recognition using OpenAI's Whisper ASR
    try:
        response = openai.Transcription.create(
            audio=audio_base64,
            language="en-US",
            model="whisper-large"
        )
        
        if response.status == "completed":
            transcription = response.transcriptions[0].text
            print("Transcription:", transcription)
            
            # Generate improved text using OpenAI API
            prompt = f"Please help improve the grammar of the following text: '{transcription}'"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50
            )
            
            if response.status == 200:
                improved_text = response.choices[0].text.strip()
                print("Improved Text:", improved_text)
            else:
                print("Error: Unable to generate improved text using OpenAI API")
        else:
            print("Error: Transcription failed.")
    except openai.error.OpenAIError as e:
        print("OpenAI Error:", e)
