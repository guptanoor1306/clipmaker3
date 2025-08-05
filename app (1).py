# app.py - Working ClipMaker with Parameter Selection
import os
import json
import tempfile
import traceback
import re
import time
import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import gdown
from openai import OpenAI

# ----------
# Helper Functions
# ----------

def get_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets or environment."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    for key in ("OPENAI_API_KEY", "api_key"):
        if key in st.secrets:
            return st.secrets[key]
    return os.getenv("OPENAI_API_KEY", "")


def get_system_prompt(platform: str, selected_parameters: list, video_duration: float = None) -> str:
    if video_duration:
        duration_minutes = int(video_duration // 60)
        duration_seconds = int(video_duration % 60)
        max_start_time = max(0, video_duration - 60)
        duration_info = f"""
CRITICAL VIDEO CONSTRAINTS:
- Video duration: {video_duration:.1f} seconds ({duration_minutes}:{duration_seconds:02d})
- ALL timestamps MUST be between 00:00:00 and {duration_minutes//60:02d}:{(duration_minutes%60):02d}:{duration_seconds:02d}
- Maximum start time for any clip: {int(max_start_time//60):02d}:{int(max_start_time%60):02d}:{int(max_start_time%60):02d}
- DO NOT generate timestamps beyond the video duration
- Estimate timestamps based on transcript position (beginning = early timestamps, end = later timestamps)
"""
    else:
        duration_info = ""
    
    # Build parameter descriptions based on user selection
    parameter_descriptions = []
    for param in selected_parameters:
        if param == "Educational Value":
            parameter_descriptions.append("🧠 Educational Value: Clear insights, tips, or new perspectives delivered quickly")
        elif param == "Surprise Factor":
            parameter_descriptions.append("😲 Surprise Factor: Plot twists, myth-busting, or unexpected revelations")
        elif param == "Emotional Impact":
            parameter_descriptions.append("😍 Emotional Impact: Inspiration, humor, shock, or relatability that drives engagement")
        elif param == "Replayability":
            parameter_descriptions.append("🔁 Replayability: Content viewers want to watch multiple times or share")
        elif param == "Speaker Energy":
            parameter_descriptions.append("🎤 Speaker Energy: Passionate delivery, voice modulation, natural pauses")
        elif param == "Relatability":
            parameter_descriptions.append("🎯 Relatability: Reflects common struggles, desires, or experiences")
        elif param == "Contrarian Takes":
            parameter_descriptions.append("🔥 Contrarian Takes: Challenges popular beliefs or conventional wisdom")
        elif param == "Storytelling":
            parameter_descriptions.append("📖 Storytelling: Personal anecdotes, case studies, or narrative elements")
    
    parameters_text = "\n".join(parameter_descriptions) if parameter_descriptions else "🎯 General viral potential focusing on engagement and shareability"
    
    return f"""You are a content strategist and social media editor trained to analyze long-form video/podcast transcripts. Your task is to identify 15–60 second segments that are highly likely to perform well as short-form content on {platform}.

{duration_info}

CRITICAL REQUIREMENTS FOR VIRAL SUCCESS:
🎯 ZERO-SECOND HOOK: The first 1-3 seconds MUST grab attention immediately - no slow intros or context setting. Start with the most compelling moment, question, or statement.
🔚 PROPER ENDING: Clips must have a satisfying conclusion - avoid abrupt cuts mid-sentence. End with a complete thought, punchline, or call-to-action.  
⏱️ DURATION: Clips must be between 15-60 seconds. Shorter clips often perform better due to higher completion rates.

SELECTED FOCUS PARAMETERS:
{parameters_text}

PRIORITIZE CONTENT THAT MATCHES THE SELECTED PARAMETERS ABOVE. Focus your analysis on finding segments that excel in these specific areas.

For each recommended cut, provide:
1. Start and end timestamps (HH:MM:SS format) - MUST be within video duration
2. Hook: The exact opening words/question that grabs attention (first 1-3 seconds)
3. Flow: Detailed narrative structure (e.g., "Hook → Context → Pivot → CTA")
4. Focus parameters: Which of the selected parameters this clip excels in
5. Reason why this segment will work (focusing on hook strength, selected parameters, and complete ending)
6. Predicted engagement score (0–100) — your confidence in performance
7. Suggested caption for social media with emojis/hashtags

Output ONLY valid JSON as an array of objects with these exact keys:
- start: "HH:MM:SS"
- end: "HH:MM:SS" 
- hook: "exact opening words/question that starts the clip"
- flow: "detailed narrative structure (Hook → Context → Pivot → etc.)"
- focus_parameters: ["parameter1", "parameter2"] (which selected parameters this clip focuses on)
- reason: "brief rationale focusing on hook strength, selected parameters, virality factors, and proper ending"
- score: integer (0-100)
- caption: "social media caption with emojis and hashtags"

Example format:
[
  {{
    "start": "00:02:15",
    "end": "00:02:45",
    "hook": "Did you know your credit score is basically meaningless?",
    "flow": "Hook (Question) → Shocking claim → Myth debunking → Alternative approach → Action step",
    "focus_parameters": ["Educational Value", "Surprise Factor"],
    "reason": "Opens with shocking question that hooks immediately, myth-busts credit score beliefs (Educational Value + Surprise Factor), ends with complete actionable advice",
    "score": 88,
    "caption": "Did you know your credit score is basically meaningless? 😱 #MoneyMyths #FinanceTips #CreditScore"
  }}
]"""


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video and compress if needed."""
    try:
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_temp.name, codec='mp3', bitrate='64k')
        audio.close()
        video.close()
        return audio_temp.name
    except Exception as e:
        st.error(f"Audio extraction failed: {str(e)}")
        raise


def split_audio_file(audio_path: str, chunk_duration_minutes: int = 10) -> list:
    """Split audio file into smaller chunks if it's too large."""
    try:
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb <= 20:
            return [audio_path]
        
        st.info(f"Audio file is {file_size_mb:.1f}MB. Splitting into chunks...")
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        chunk_duration_seconds = chunk_duration_minutes * 60
        chunks = []
        start_time = 0
        chunk_num = 1
        
        while start_time < duration:
            end_time = min(start_time + chunk_duration_seconds, duration)
            chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{chunk_num}.mp3")
            chunk_audio = audio_clip.subclipped(start_time, end_time)
            chunk_audio.write_audiofile(chunk_temp.name, codec='mp3', bitrate='64k')
            chunks.append(chunk_temp.name)
            chunk_audio.close()
            start_time = end_time
            chunk_num += 1
        
        audio_clip.close()
        st.success(f"Split audio into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        st.error(f"Audio splitting failed: {str(e)}")
        raise


def transcribe_audio(path: str, client: OpenAI) -> str:
    """Transcribe audio via Whisper-1, handling large files by chunking."""
    try:
        st.info("🎵 Extracting audio from video...")
        audio_path = extract_audio_from_video(path)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        st.info(f"Audio file size: {file_size_mb:.1f}MB")
        audio_chunks = split_audio_file(audio_path)
        full_transcript = ""
        
        if len(audio_chunks) > 1:
            st.info(f"Transcribing {len(audio_chunks)} audio chunks...")
            progress_bar = st.progress(0)
            for i, chunk_path in enumerate(audio_chunks):
                st.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
                with open(chunk_path, "rb") as f:
                    resp = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
                chunk_transcript = resp.strip()
                if chunk_transcript:
                    full_transcript += chunk_transcript + " "
                progress_bar.progress((i + 1) / len(audio_chunks))
                try:
                    os.unlink(chunk_path)
                except:
                    pass
        else:
            with open(audio_chunks[0], "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
            full_transcript = resp
        
        try:
            os.unlink(audio_path)
        except:
            pass
        return full_transcript.strip()
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        raise


def analyze_transcript(transcript: str, platform: str, selected_parameters: list, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions via ChatCompletion."""
    transcript_length = len(transcript.split())
    words_per_section = transcript_length // 5
    
    transcript_context = ""
    if video_duration and transcript_length > 0:
        transcript_context = f"""
TRANSCRIPT TIMING CONTEXT:
- Total transcript words: {transcript_length}
- Video duration: {video_duration:.1f} seconds ({int(video_duration//60)}:{int(video_duration%60):02d})
- Speaking rate: ~{transcript_length/(video_duration/60):.0f} words per minute
Use this guide to estimate where content appears in the video timeline.
"""
    
    messages = [
        {"role": "system", "content": get_system_prompt(platform, selected_parameters, video_duration)},
        {"role": "user", "content": f"""Analyze this transcript and identify the best segments for {platform} based on the selected parameters: {', '.join(selected_parameters)}. 

{transcript_context}

Focus on segments with powerful hooks in the first 3 seconds and proper endings. PRIORITIZE content that matches the selected parameters: {', '.join(selected_parameters)}.

CRITICAL: All timestamps must be within 0 to {video_duration:.1f} seconds.

Transcript:
{transcript}"""}
    ]
    
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=2000)
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        raise


def parse_segments(text: str, video_duration: float = None) -> list:
    """Parse JSON text into a list of segments and validate timestamps."""
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        segments = json.loads(text)
        valid_segments = []
        
        for i, seg in enumerate(segments):
            required_keys = ["start", "end", "hook", "flow", "focus_parameters", "reason", "score", "caption"]
            if all(key in seg for key in required_keys):
                try:
                    start_seconds = time_to_seconds(seg["start"])
                    end_seconds = time_to_seconds(seg["end"])
                    
                    if video_duration:
                        if start_seconds >= video_duration:
                            continue
                        if end_seconds > video_duration:
                            end_minutes = int(video_duration // 60)
                            end_secs = int(video_duration % 60)
                            seg["end"] = f"{end_minutes//60:02d}:{end_minutes%60:02d}:{end_secs:02d}"
                            end_seconds = video_duration
                    
                    if start_seconds >= end_seconds:
                        continue
                    if end_seconds - start_seconds < 15:
                        continue
                    if end_seconds - start_seconds > 60:
                        new_end_seconds = start_seconds + 60
                        if video_duration and new_end_seconds > video_duration:
                            new_end_seconds = video_duration
                        end_minutes = int(new_end_seconds // 60)
                        end_secs = int(new_end_seconds % 60)
                        seg["end"] = f"{end_minutes//60:02d}:{end_minutes%60:02d}:{end_secs:02d}"
                    
                    valid_segments.append(seg)
                except:
                    continue
            else:
                # Try to handle old format without new fields
                if all(key in seg for key in ["start", "end", "reason", "score", "caption"]):
                    # Add default values for missing fields
                    seg["hook"] = seg.get("hook", "Hook not specified")
                    seg["flow"] = seg.get("flow", "Flow structure not specified")
                    seg["focus_parameters"] = seg.get("focus_parameters", ["General"])
                    
                    try:
                        start_seconds = time_to_seconds(seg["start"])
                        end_seconds = time_to_seconds(seg["end"])
                        
                        if video_duration:
                            if start_seconds >= video_duration:
                                continue
                            if end_seconds > video_duration:
                                end_minutes = int(video_duration // 60)
                                end_secs = int(video_duration % 60)
                                seg["end"] = f"{end_minutes//60:02d}:{end_minutes%60:02d}:{end_secs:02d}"
                        
                        if start_seconds >= end_seconds:
                            continue
                        if end_seconds - start_seconds < 15:
                            continue
                        if end_seconds - start_seconds > 60:
                            new_end_seconds = start_seconds + 60
                            if video_duration and new_end_seconds > video_duration:
                                new_end_seconds = video_duration
                            end_minutes = int(new_end_seconds // 60)
                            end_secs = int(new_end_seconds % 60)
                            seg["end"] = f"{end_minutes//60:02d}:{end_minutes%60:02d}:{end_secs:02d}"
                        
                        valid_segments.append(seg)
                    except:
                        continue
        
        return valid_segments
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        return []


def time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS to seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])
    except:
        st.error(f"Could not parse time: {time_str}")
        return 0


def generate_clips(video_path: str, segments: list) -> list:
    """Use moviepy to cut video segments."""
    clips = []
    
    try:
        video = VideoFileClip(video_path)
        total_duration = video.duration
        st.info(f"Video duration: {total_duration:.1f} seconds")
        
        for i, seg in enumerate(segments, start=1):
            try:
                start_time = time_to_seconds(seg.get("start", "0"))
                end_time = time_to_seconds(seg.get("end", "0"))
                caption = seg.get("caption", f"clip_{i}")
                score = seg.get("score", 0)
                reason = seg.get("reason", "")
                
                st.info(f"Processing clip {i}: {start_time:.1f}s - {end_time:.1f}s")
                
                if start_time >= end_time or start_time >= total_duration:
                    continue
                if end_time > total_duration:
                    end_time = total_duration
                if end_time - start_time < 1:
                    continue
                
                try:
                    if hasattr(video, 'subclipped'):
                        clip = video.subclipped(start_time, end_time)
                    elif hasattr(video, 'subclip'):
                        clip = video.subclip(start_time, end_time)
                    else:
                        clip = video.cutout(0, start_time).cutout(end_time - start_time, video.duration)
                except AttributeError:
                    from moviepy.video.fx import subclip
                    clip = subclip(video, start_time, end_time)
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"clip_{i}_")
                st.info(f"Writing clip {i} to file...")
                
                try:
                    clip.write_videofile(temp_file.name, codec="libx264", audio_codec="aac", 
                                       temp_audiofile_path=tempfile.gettempdir(), preset='ultrafast', fps=24)
                except Exception:
                    clip.write_videofile(temp_file.name, preset='ultrafast')
                
                if os.path.isfile(temp_file.name) and os.path.getsize(temp_file.name) > 0:
                    clips.append({
                        "path": temp_file.name, "caption": caption, "score": score, "reason": reason,
                        "hook": seg.get("hook", "Hook not specified"),
                        "flow": seg.get("flow", "Flow not specified"), 
                        "focus_parameters": seg.get("focus_parameters", ["General"]),
                        "start": seg.get("start"), "end": seg.get("end"), 
                        "duration": f"{end_time - start_time:.1f}s"
                    })
                    st.success(f"✅ Created clip {i}")
                
                clip.close()
            except Exception as e:
                st.error(f"Error creating clip {i}: {str(e)}")
                continue
        
        video.close()
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        raise
    
    return clips


def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file given its share URL to out_path."""
    import requests
    
    try:
        file_id = None
        if "id=" in drive_url:
            file_id = drive_url.split("id=")[1].split("&")[0]
        else:
            patterns = [r"/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"/file/d/([a-zA-Z0-9_-]+)"]
            for pattern in patterns:
                m = re.search(pattern, drive_url)
                if m:
                    file_id = m.group(1)
                    break
        
        if not file_id:
            raise ValueError("Could not extract file ID from URL")
        
        st.info(f"📥 Attempting to download file ID: {file_id}")
        
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            result = gdown.download(download_url, out_path, quiet=False)
            if result and os.path.isfile(result) and os.path.getsize(result) > 0:
                return result
        except Exception:
            pass
        
        raise Exception("Download failed - please ensure file is publicly accessible")
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        raise


def display_clips(clips: list, platform: str, start_index: int = 0, max_clips: int = 5):
    """Display clips with download buttons."""
    clips_to_show = clips[start_index:start_index + max_clips]
    
    for i, clip in enumerate(clips_to_show, start=start_index + 1):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Clip #{i} (Score: {clip.get('score', 0)}/100)")
            video_path = clip.get("path")
            if video_path and os.path.isfile(video_path):
                try:
                    st.video(video_path)
                except Exception:
                    st.error(f"Error displaying video for clip {i}")
            else:
                st.error(f"Video file not found for clip {i}")
            
            st.markdown("**📝 Suggested Caption:**")
            st.code(clip.get("caption", "No caption available"), language="text")
            
        with col2:
            st.markdown("**📊 Details:**")
            st.write(f"⏱️ **Duration:** {clip.get('duration', 'N/A')}")
            st.write(f"🕐 **Time:** {clip.get('start', 'N/A')} - {clip.get('end', 'N/A')}")
            st.write(f"🎯 **Score:** {clip.get('score', 0)}/100")
            
            # New detailed breakdown
            st.markdown("**🪝 0-Second Hook:**")
            hook_text = clip.get('hook', 'Hook not specified')
            st.write(f"*\"{hook_text}\"*")
            
            st.markdown("**🎬 Content Flow:**")
            flow_text = clip.get('flow', 'Flow structure not specified')
            st.write(flow_text)
            
            st.markdown("**🎯 Focus Parameters:**")
            focus_params = clip.get('focus_parameters', ['General'])
            if isinstance(focus_params, list):
                for param in focus_params:
                    st.write(f"• {param}")
            else:
                st.write(f"• {focus_params}")
            
            st.markdown("**💡 Why this will work:**")
            st.write(clip.get('reason', 'No reason provided'))
            
            # Download button with stable key that doesn't cause rerun
            video_path = clip.get("path")
            if video_path and os.path.isfile(video_path):
                try:
                    with open(video_path, "rb") as file:
                        file_data = file.read()
                        # Create a super stable key that includes session info
                        stable_key = f"dl_{st.session_state.session_id}_{i}_{abs(hash(video_path)) % 1000}"
                        
                        # Add a small container to isolate the download button
                        with st.container():
                            st.download_button(
                                label="⬇️ Download Clip", 
                                data=file_data,
                                file_name=f"clip_{i}_{platform.replace(' ', '_').lower()}.mp4",
                                mime="video/mp4", 
                                use_container_width=True, 
                                key=stable_key,
                                help="Download this clip to your device"
                            )
                except Exception as e:
                    st.error(f"Error preparing download for clip {i}: {str(e)}")
            else:
                st.error("❌ File not available for download")
        
        st.markdown("---")


# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="ClipMaker", layout="wide")
    
    st.title("🎬 Long‑form to Short‑form ClipMaker")
    st.markdown("Transform your long-form content into viral short-form clips using AI-powered analysis!")

    # Initialize session state
    if 'clips_generated' not in st.session_state:
        st.session_state.clips_generated = False
    if 'all_clips' not in st.session_state:
        st.session_state.all_clips = []
    if 'clips_shown' not in st.session_state:
        st.session_state.clips_shown = 0
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hash(str(time.time())) % 100000  # Unique session identifier

    # API Key validation
    API_KEY = get_api_key()
    if not API_KEY:
        st.error("❌ OpenAI API key not found. Add it to Streamlit secrets or env var OPENAI_API_KEY.")
        return
    
    try:
        client = OpenAI(api_key=API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI client: {str(e)}")
        return

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    platform = st.sidebar.selectbox("Target Platform", ["YouTube Shorts", "Instagram Reels", "TikTok"])
    
    # Content focus parameters
    st.sidebar.subheader("🎯 Content Focus")
    st.sidebar.caption("Select the types of content you want to prioritize")
    
    available_parameters = ["Educational Value", "Surprise Factor", "Emotional Impact", "Replayability", 
                           "Speaker Energy", "Relatability", "Contrarian Takes", "Storytelling"]
    
    selected_parameters = []
    for param in available_parameters:
        if st.sidebar.checkbox(param, key=f"param_{param}"):
            selected_parameters.append(param)
    
    if not selected_parameters:
        st.sidebar.warning("⚠️ Select at least one content focus parameter")
    else:
        st.sidebar.success(f"✅ {len(selected_parameters)} parameters selected")

    # Video source
    st.sidebar.subheader("📹 Video Source")
    drive_url = st.sidebar.text_input("Google Drive URL (share link)", 
                                     placeholder="https://drive.google.com/file/d/...")
    
    video_path = None

    if drive_url:
        if st.sidebar.button("📥 Download from Drive"):
            with st.spinner("Downloading from Google Drive…"):
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    result = download_drive_file(drive_url, tmp.name)
                    
                    if result and os.path.isfile(result):
                        size_mb = os.path.getsize(result) / (1024 * 1024)
                        st.success(f"✅ Downloaded {size_mb:.2f} MB from Drive")
                        video_path = result
                        st.session_state['video_path'] = video_path
                        st.session_state['video_size'] = size_mb
                        
                        # Reset processing state
                        st.session_state.clips_generated = False
                        st.session_state.all_clips = []
                        st.session_state.clips_shown = 0
                        st.session_state.processing_complete = False
                        
                        if size_mb <= 500:
                            st.video(video_path)
                        else:
                            st.warning("File is large (>500MB); skipping preview to save memory.")
                except Exception as e:
                    st.error(f"Drive download failed: {str(e)}")
                    return
    else:
        uploaded = st.sidebar.file_uploader("Or upload a video file", type=["mp4", "mov", "mkv", "avi"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tmp.write(uploaded.read())
            video_path = tmp.name
            st.session_state['video_path'] = video_path
            st.session_state['video_size'] = len(uploaded.getvalue()) / (1024 * 1024)
            
            # Reset processing state
            st.session_state.clips_generated = False
            st.session_state.all_clips = []
            st.session_state.clips_shown = 0
            st.session_state.processing_complete = False
            st.video(video_path)
    
    # Use video from session state if available
    if not video_path and 'video_path' in st.session_state:
        video_path = st.session_state['video_path']
        if os.path.isfile(video_path):
            st.info(f"Using previously loaded video ({st.session_state.get('video_size', 0):.2f} MB)")
            if st.session_state.get('video_size', 0) <= 500:
                st.video(video_path)
        else:
            st.warning("Previously loaded video no longer available. Please reload.")
            del st.session_state['video_path']
            video_path = None

    if not video_path:
        st.info("🎯 Provide a Drive link or upload a file to begin.")
        return

    if not selected_parameters:
        st.warning("⚠️ Please select at least one content focus parameter in the sidebar to continue.")
        return

    # Show generated clips if they exist
    if st.session_state.clips_generated and st.session_state.all_clips:
        st.markdown("---")
        st.header("🎬 Generated Clips")
        
        # Check if clips still exist
        valid_clips = [clip for clip in st.session_state.all_clips 
                      if clip.get("path") and os.path.isfile(clip["path"])]
        
        if not valid_clips:
            st.error("❌ All clip files are missing. Please regenerate clips.")
            st.session_state.clips_generated = False
            st.session_state.all_clips = []
            st.session_state.clips_shown = 0
            st.session_state.processing_complete = False
            return
        
        if len(valid_clips) != len(st.session_state.all_clips):
            st.session_state.all_clips = valid_clips
            st.warning(f"Some clip files were missing. Showing {len(valid_clips)} available clips.")
        
        # Display current batch of clips
        clips_to_show = min(5, len(st.session_state.all_clips) - st.session_state.clips_shown)
        
        if clips_to_show > 0:
            st.subheader(f"Top {st.session_state.clips_shown + 1}-{st.session_state.clips_shown + clips_to_show} Clips")
            st.info(f"🎯 **Selected Parameters:** {', '.join(selected_parameters)}")
            display_clips(st.session_state.all_clips, platform, st.session_state.clips_shown, clips_to_show)
            st.session_state.clips_shown += clips_to_show
        
        # Show "Show More" button if there are more clips
        remaining_clips = len(st.session_state.all_clips) - st.session_state.clips_shown
        if remaining_clips > 0:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Use a unique key that doesn't conflict with download buttons
                show_more_key = f"show_more_{st.session_state.clips_shown}_{len(st.session_state.all_clips)}"
                if st.button(f"🎬 Show Next {min(5, remaining_clips)} Clips", 
                           type="primary", use_container_width=True, key=show_more_key):
                    st.rerun()
        
        # Summary stats
        if st.session_state.clips_shown >= len(st.session_state.all_clips):
            st.subheader("📈 Summary")
            avg_score = sum(c.get('score', 0) for c in st.session_state.all_clips) / len(st.session_state.all_clips)
            total_duration = sum(float(c.get('duration', '0').replace('s', '')) for c in st.session_state.all_clips)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Clips", len(st.session_state.all_clips))
            col2.metric("Average Score", f"{avg_score:.1f}/100")
            col3.metric("Total Duration", f"{total_duration:.1f}s")
            col4.metric("Platform", platform)
        
        # Reset button
        reset_key = f"reset_{len(st.session_state.all_clips)}_{st.session_state.clips_shown}"
        if st.button("🔄 Clear All Clips & Start Over", type="secondary", key=reset_key):
            for clip in st.session_state.all_clips:
                try:
                    if clip.get("path") and os.path.isfile(clip["path"]):
                        os.unlink(clip["path"])
                except:
                    pass
            
            st.session_state.clips_generated = False
            st.session_state.all_clips = []
            st.session_state.clips_shown = 0
            st.session_state.processing_complete = False
            st.rerun()
        
        return

    # Main processing
    if not st.session_state.processing_complete:
        if st.button("🚀 Generate Clips", type="primary"):
            if not video_path or not os.path.isfile(video_path):
                st.error("Video file not found. Please reload your video.")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Transcription
            status_text.text("🎤 Transcribing audio...")
            progress_bar.progress(25)
            
            try:
                transcript = transcribe_audio(video_path, client)
                st.success("✅ Transcription complete")
            except Exception as e:
                st.error(f"❌ Transcription failed: {str(e)}")
                return

            progress_bar.progress(50)
            with st.expander("📄 Transcript Preview", expanded=False):
                st.text_area("Full Transcript", transcript, height=200, disabled=True)

            # Step 2: AI analysis
            status_text.text(f"🤖 Analyzing transcript for viral segments based on: {', '.join(selected_parameters)}...")
            progress_bar.progress(75)
            
            # Get video duration for AI context
            try:
                temp_video = VideoFileClip(video_path)
                video_duration = temp_video.duration
                temp_video.close()
            except:
                video_duration = None
                st.warning("Could not determine video duration for AI analysis")
            
            try:
                ai_json = analyze_transcript(transcript, platform, selected_parameters, client, video_duration)
                st.success("✅ Analysis complete")
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return

            with st.expander("🔍 AI Analysis Output", expanded=False):
                st.code(ai_json, language="json")

            # Step 3: Parse segments and sort by score
            status_text.text("📊 Processing segments...")
            progress_bar.progress(90)
            
            segments = parse_segments(ai_json, video_duration)
            if not segments:
                st.warning("⚠️ No valid segments found in AI response.")
                return
                
            segments_sorted = sorted(segments, key=lambda x: x.get('score', 0), reverse=True)
            
            # Step 4: Generate all clips
            status_text.text("✂️ Generating all video clips...")
            progress_bar.progress(95)
            
            try:
                all_clips = generate_clips(video_path, segments_sorted)
                
                if all_clips:
                    st.session_state.all_clips = all_clips
                    st.session_state.clips_generated = True
                    st.session_state.clips_shown = 0
                    st.session_state.processing_complete = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Clips generated successfully!")
                    
                    st.success(f"🎉 Generated {len(all_clips)} clips based on selected parameters: {', '.join(selected_parameters)}! Showing top 5 first.")
                    st.rerun()
                else:
                    st.warning("No clips were generated.")
                    return
                    
            except Exception as e:
                st.error(f"❌ Clip generation failed: {str(e)}")
                return
    else:
        st.info("🎯 Click 'Generate Clips' to start processing your video.")


if __name__ == "__main__":
    main()
