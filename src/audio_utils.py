# src/audio_utils.py
import tempfile
import os

def transcribe_audio(uploaded_audio, model_name="tiny"):
    """
    uploaded_audio: stream (BytesIO) from Streamlit
    Returns transcribed text (string) or '' on failure.
    """
    try:
        import whisper
    except Exception:
        return ""

    # write to temp file
    suffix = os.path.splitext(uploaded_audio.name)[1] if hasattr(uploaded_audio, "name") else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_audio.read())
        tmp_path = tmp.name

    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(tmp_path)
        text = result.get("text", "")
    except Exception:
        text = ""
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text
