# app.py - Enhanced ClipMaker with Real-time Generation and Vertical Format
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

# MoviePy imports with version compatibility
try:
    # Try newer MoviePy structure
    from moviepy.video.fx.resize import resize
    from moviepy.video.fx.crop import crop
except ImportError:
    try:
        # Try importing as functions
        from moviepy.video.fx import resize, crop
    except ImportError:
        # Define fallback functions using moviepy's apply method
        def resize(clip, newsize):
            return clip.resize(newsize)
        
        def crop(clip, x1=None, y1=None, x2=None, y2=None, width=None, height=None):
            return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height)

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

def seconds_to_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def extract_audio_sample(video_path: str, duration: float = 300) -> tuple:
    """Extract just a small sample of audio for transcription - using very conservative approach."""
    try:
        # Validate video path first
        if not video_path or not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")
            
        # Start with smaller 5-minute sample for stability
        st.info(f"🎵 Extracting {duration/60:.1f} minute audio sample for analysis...")
        
        # Check file size safely
        try:
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            st.info(f"📁 Processing {file_size_mb:.1f}MB video file")
        except Exception as e:
            st.warning(f"Could not determine file size: {e}")
            file_size_mb = 0
        
        video = VideoFileClip(video_path)
        total_duration = video.duration
        
        # Use even smaller sample for large files
        if file_size_mb > 1500:  # For very large files, use tiny sample
            duration = min(duration, 180)  # Max 3 minutes for huge files
            st.info(f"🔧 Using smaller {duration/60:.1f} minute sample for large file")
        
        # If video is shorter than sample duration, use entire video
        if total_duration <= duration:
            sample_duration = total_duration
            start_time = 0
        else:
            # Take sample from 25% into the video (after intro, before conclusion)
            start_time = max(0, total_duration * 0.25)
            sample_duration = min(duration, total_duration - start_time)
        
        st.info(f"📍 Sampling from {start_time/60:.1f} to {(start_time + sample_duration)/60:.1f} minutes")
        
        # Extract just the sample clip - use most conservative method
        try:
            # Try the most memory-efficient method first
            sample_clip = video.subclip(start_time, start_time + sample_duration)
        except AttributeError:
            try:
                sample_clip = video.subclipped(start_time, start_time + sample_duration)
            except AttributeError:
                # Ultra-conservative fallback - create new clip with manual duration
                def get_sample_frame(get_frame, t):
                    actual_t = start_time + (t % sample_duration)
                    return get_frame(actual_t)
                
                sample_clip = video.fl(get_sample_frame, apply_to=['mask', 'audio'])
                sample_clip = sample_clip.set_duration(sample_duration)
        
        if sample_clip.audio is None:
            video.close()
            raise Exception("No audio track found in video")
        
        # Extract audio from sample with aggressive compression
        audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        
        # Use very low quality for memory efficiency
        sample_clip.audio.write_audiofile(
            audio_temp.name, 
            codec='mp3', 
            bitrate='32k',  # Lower bitrate
            temp_audiofile_path=tempfile.gettempdir()
        )
        
        # Immediately close everything to free memory
        sample_clip.audio.close()
        sample_clip.close()
        video.close()
        
        # Verify audio file was created
        if not os.path.exists(audio_temp.name) or os.path.getsize(audio_temp.name) == 0:
            raise Exception("Audio extraction failed - no output file created")
        
        audio_size_mb = os.path.getsize(audio_temp.name) / (1024 * 1024)
        st.success(f"✅ Audio sample extracted successfully ({audio_size_mb:.1f}MB)")
        
        return audio_temp.name, start_time, sample_duration
        
    except Exception as e:
        # Clean up on error
        try:
            if 'video' in locals():
                video.close()
        except:
            pass
        st.error(f"Audio sample extraction failed: {str(e)}")
        raise

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

def analyze_transcript(transcript: str, platform: str, selected_parameters: list, client: OpenAI, video_duration: float = None) -> str:
    """Get segment suggestions via ChatCompletion."""
    transcript_length = len(transcript.split())
    
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
        
        return valid_segments
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}")
        return []

def adjust_timestamps_for_sample(segments: list, sample_start: float, sample_duration: float, video_duration: float) -> list:
    """Adjust AI-generated timestamps to work with the full video."""
    adjusted_segments = []
    
    for segment in segments:
        try:
            # Get timestamps relative to sample
            sample_start_time = time_to_seconds(segment["start"])
            sample_end_time = time_to_seconds(segment["end"])
            
            # Convert to absolute timestamps in full video
            absolute_start = sample_start + sample_start_time
            absolute_end = sample_start + sample_end_time
            
            # Ensure timestamps are within video bounds
            if absolute_start >= video_duration:
                continue
            if absolute_end > video_duration:
                absolute_end = video_duration
            
            # Update the segment with absolute timestamps
            segment["start"] = seconds_to_time(absolute_start)
            segment["end"] = seconds_to_time(absolute_end)
            
            adjusted_segments.append(segment)
            
        except Exception as e:
            st.warning(f"Skipping segment due to timestamp error: {e}")
            continue
    
    return adjusted_segments

def create_vertical_clip(video_path: str, start_time: float, end_time: float, crop_mode: str = "smart") -> str:
    """Create a vertical 9:16 clip optimized for shorts/reels with intelligent cropping."""
    try:
        st.info(f"🎬 Creating clip from {start_time/60:.1f} to {end_time/60:.1f} minutes...")
        
        video = VideoFileClip(video_path)
        
        # Extract the specific segment - try different methods for compatibility
        try:
            # Method 1: Try subclip
            clip = video.subclip(start_time, end_time)
        except AttributeError:
            try:
                # Method 2: Try subclipped
                clip = video.subclipped(start_time, end_time)
            except AttributeError:
                # Method 3: Try cutout method
                try:
                    if start_time > 0:
                        clip = video.cutout(0, start_time)
                        if end_time < video.duration:
                            clip = clip.cutout(end_time - start_time, clip.duration)
                    else:
                        clip = video.cutout(end_time, video.duration)
                except AttributeError:
                    # Method 4: Manual frame extraction fallback
                    def get_clip_frame(get_frame, t):
                        actual_t = start_time + t
                        if actual_t >= end_time:
                            actual_t = end_time - 0.1
                        return get_frame(actual_t)
                    
                    clip = video.fl(get_clip_frame, apply_to=['mask', 'audio'])
                    clip = clip.set_duration(end_time - start_time)
        
        # Get original dimensions
        w, h = clip.size
        aspect_ratio = w / h
        target_aspect = 9 / 16
        
        # Target dimensions for vertical video (9:16)
        target_height = 1920
        target_width = 1080
        
        # Calculate crop parameters
        if crop_mode == "smart":
            if aspect_ratio > target_aspect:
                new_width = int(h * target_aspect)
                x_center = w // 2
                x_offset = max(0, min(w - new_width, x_center - new_width // 3))
                x1, y1, x2, y2 = x_offset, 0, x_offset + new_width, h
            else:
                new_height = int(w / target_aspect)
                y_offset = max(0, h // 4)
                if y_offset + new_height > h:
                    y_offset = h - new_height
                x1, y1, x2, y2 = 0, y_offset, w, y_offset + new_height
        elif crop_mode == "center":
            if aspect_ratio > target_aspect:
                new_width = int(h * target_aspect)
                x_offset = (w - new_width) // 2
                x1, y1, x2, y2 = x_offset, 0, x_offset + new_width, h
            else:
                new_height = int(w / target_aspect)
                y_offset = (h - new_height) // 2
                x1, y1, x2, y2 = 0, y_offset, w, y_offset + new_height
        elif crop_mode == "top":
            if aspect_ratio > target_aspect:
                new_width = int(h * target_aspect)
                x_offset = (w - new_width) // 2
                x1, y1, x2, y2 = x_offset, 0, x_offset + new_width, h
            else:
                new_height = int(w / target_aspect)
                x1, y1, x2, y2 = 0, 0, w, new_height
        
        # Apply cropping - simple method
        def crop_frame(get_frame, t):
            frame = get_frame(t)
            return frame[y1:y2, x1:x2]
        
        cropped = clip.fl(crop_frame)
        
        # Resize to target resolution
        if hasattr(cropped, 'resize'):
            final_clip = cropped.resize((target_width, target_height))
        else:
            final_clip = cropped
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        # Write video with optimized settings
        final_clip.write_videofile(
            temp_file.name,
            codec="libx264",
            audio_codec="aac",
            preset='fast',
            fps=30,
            bitrate="1500k"
        )
        
        # Clean up
        final_clip.close()
        cropped.close()
        clip.close()
        video.close()
        
        st.success(f"✅ Vertical clip created successfully!")
        return temp_file.name
        
    except Exception as e:
        st.error(f"Error creating vertical clip: {str(e)}")
        # Fallback: create simple horizontal clip
        try:
            st.warning("Creating horizontal clip as fallback...")
            video = VideoFileClip(video_path)
            
            # Try different clip extraction methods for fallback too
            try:
                clip = video.subclip(start_time, end_time)
            except AttributeError:
                try:
                    clip = video.subclipped(start_time, end_time)
                except AttributeError:
                    # Simple duration-based clip as last resort
                    def get_fallback_frame(get_frame, t):
                        actual_t = start_time + t
                        if actual_t >= end_time:
                            actual_t = end_time - 0.1
                        return get_frame(actual_t)
                    
                    clip = video.fl(get_fallback_frame, apply_to=['mask', 'audio'])
                    clip = clip.set_duration(end_time - start_time)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            clip.write_videofile(
                temp_file.name,
                codec="libx264",
                audio_codec="aac",
                preset='ultrafast'
            )
            
            clip.close()
            video.close()
            return temp_file.name
        except Exception as fallback_error:
            st.error(f"Fallback clip creation also failed: {str(fallback_error)}")
            raise

def display_single_clip(clip_data: dict, platform: str, index: int):
    """Display a single clip with edit functionality."""
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Clip #{index} (Score: {clip_data.get('score', 0)}/100)")
            
            # Video player
            video_path = clip_data.get("path")
            if video_path and os.path.isfile(video_path):
                try:
                    st.video(video_path)
                except Exception:
                    st.error(f"Error displaying video for clip {index}")
            else:
                st.error(f"Video file not found for clip {index}")
            
            # Caption
            st.markdown("**📝 Suggested Caption:**")
            st.code(clip_data.get("caption", "No caption available"), language="text")
            
        with col2:
            st.markdown("**📊 Details:**")
            st.write(f"⏱️ **Duration:** {clip_data.get('duration', 'N/A')}")
            st.write(f"🕐 **Time:** {clip_data.get('start', 'N/A')} - {clip_data.get('end', 'N/A')}")
            st.write(f"🎯 **Score:** {clip_data.get('score', 0)}/100")
            
            # Hook
            st.markdown("**🪝 0-Second Hook:**")
            hook_text = clip_data.get('hook', 'Hook not specified')
            st.write(f"*\"{hook_text}\"*")
            
            # Flow
            st.markdown("**🎬 Content Flow:**")
            flow_text = clip_data.get('flow', 'Flow structure not specified')
            st.write(flow_text)
            
            # Focus Parameters
            st.markdown("**🎯 Focus Parameters:**")
            focus_params = clip_data.get('focus_parameters', ['General'])
            if isinstance(focus_params, list):
                for param in focus_params:
                    st.write(f"• {param}")
            else:
                st.write(f"• {focus_params}")
            
            # Reason
            st.markdown("**💡 Why this will work:**")
            st.write(clip_data.get('reason', 'No reason provided'))
            
            # Timestamp editing section
            st.markdown("**✏️ Edit Timestamps:**")
            
            # Current timestamps
            current_start = clip_data.get('start', '00:00:00')
            current_end = clip_data.get('end', '00:00:30')
            
            # Input fields for new timestamps
            new_start = st.text_input(
                "Start Time (HH:MM:SS)", 
                value=current_start, 
                key=f"start_{index}_{clip_data.get('score', 0)}"
            )
            new_end = st.text_input(
                "End Time (HH:MM:SS)", 
                value=current_end, 
                key=f"end_{index}_{clip_data.get('score', 0)}"
            )
            
            # Regenerate button
            if st.button(f"🔄 Regenerate Clip", key=f"regen_{index}_{clip_data.get('score', 0)}"):
                try:
                    # Validate timestamps
                    start_seconds = time_to_seconds(new_start)
                    end_seconds = time_to_seconds(new_end)
                    
                    if start_seconds >= end_seconds:
                        st.error("Start time must be before end time")
                    elif end_seconds - start_seconds < 15:
                        st.error("Clip must be at least 15 seconds long")
                    elif end_seconds - start_seconds > 60:
                        st.error("Clip must be 60 seconds or shorter")
                    else:
                        with st.spinner("Regenerating clip with new timestamps..."):
                            # Get video path from session state
                            video_path = st.session_state.get('video_path')
                            if not video_path or not os.path.isfile(video_path):
                                st.error("Original video not found")
                            else:
                                # Get crop mode from session state or default
                                crop_mode = st.session_state.get('crop_mode', 'smart')
                                
                                # Generate new clip
                                new_clip_path = create_vertical_clip(
                                    video_path, 
                                    start_seconds, 
                                    end_seconds, 
                                    crop_mode
                                )
                                
                                # Update clip data
                                clip_data['start'] = new_start
                                clip_data['end'] = new_end
                                clip_data['duration'] = f"{end_seconds - start_seconds:.1f}s"
                                
                                # Remove old file
                                if clip_data.get('path') and os.path.isfile(clip_data['path']):
                                    try:
                                        os.unlink(clip_data['path'])
                                    except:
                                        pass
                                
                                clip_data['path'] = new_clip_path
                                
                                st.success("✅ Clip regenerated successfully!")
                                st.rerun()
                                
                except Exception as e:
                    st.error(f"Error regenerating clip: {str(e)}")
            
            # Download button
            if video_path and os.path.isfile(video_path):
                try:
                    with open(video_path, "rb") as file:
                        file_data = file.read()
                        stable_key = f"dl_{st.session_state.session_id}_{index}_{abs(hash(video_path)) % 1000}"
                        
                        st.download_button(
                            label="⬇️ Download Clip", 
                            data=file_data,
                            file_name=f"clip_{index}_{platform.replace(' ', '_').lower()}_vertical.mp4",
                            mime="video/mp4", 
                            use_container_width=True, 
                            key=stable_key,
                            help="Download this vertical clip to your device"
                        )
                except Exception as e:
                    st.error(f"Error preparing download for clip {index}: {str(e)}")
            else:
                st.error("❌ File not available for download")
        
        st.markdown("---")

def download_drive_file(drive_url: str, out_path: str) -> str:
    """Download a Google Drive file given its share URL to out_path."""
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

# ----------
# Streamlit App
# ----------

def main():
    st.set_page_config(page_title="Enhanced ClipMaker", layout="wide")
    
    st.title("🎬 Enhanced Long‑form to Vertical Short‑form ClipMaker")
    st.markdown("Transform your long-form content into viral vertical short-form clips with real-time generation and smart cropping!")

    # Initialize session state
    if 'clips_generated' not in st.session_state:
        st.session_state.clips_generated = []
    if 'segments_to_process' not in st.session_state:
        st.session_state.segments_to_process = []
    if 'current_processing_index' not in st.session_state:
        st.session_state.current_processing_index = 0
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hash(str(time.time())) % 100000

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
    
    # Vertical video settings
    st.sidebar.subheader("📱 Vertical Video Settings")
    crop_mode = st.sidebar.selectbox(
        "Cropping Mode",
        ["smart", "center", "top"],
        help="Smart: AI-optimized cropping for content, Center: Traditional center crop, Top: Keep upper portion (good for talking heads)"
    )
    st.session_state['crop_mode'] = crop_mode
    
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
                        st.session_state['video_path'] = video_path
                        st.session_state['video_size'] = size_mb
                        
                        # Reset processing state
                        st.session_state.clips_generated = []
                        st.session_state.segments_to_process = []
                        st.session_state.current_processing_index = 0
                        st.session_state.processing_active = False
                        
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
            st.session_state.clips_generated = []
            st.session_state.segments_to_process = []
            st.session_state.current_processing_index = 0
            st.session_state.processing_active = False
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

    # Show processing status if active
    if st.session_state.processing_active:
        st.markdown("---")
        st.header("🔄 Processing Clips")
        
        total_segments = len(st.session_state.segments_to_process)
        current_index = st.session_state.current_processing_index
        
        if current_index < total_segments:
            progress = current_index / total_segments
            st.progress(progress, text=f"Processing clip {current_index + 1} of {total_segments}")
            
            # Process next clip
            segment = st.session_state.segments_to_process[current_index]
            
            with st.spinner(f"Generating clip {current_index + 1}/{total_segments}..."):
                try:
                    start_time = time_to_seconds(segment["start"])
                    end_time = time_to_seconds(segment["end"])
                    
                    # Create vertical clip
                    clip_path = create_vertical_clip(video_path, start_time, end_time, crop_mode)
                    
                    # Create clip data
                    clip_data = {
                        "path": clip_path,
                        "caption": segment.get("caption", f"clip_{current_index + 1}"),
                        "score": segment.get("score", 0),
                        "reason": segment.get("reason", ""),
                        "hook": segment.get("hook", "Hook not specified"),
                        "flow": segment.get("flow", "Flow not specified"),
                        "focus_parameters": segment.get("focus_parameters", ["General"]),
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "duration": f"{end_time - start_time:.1f}s"
                    }
                    
                    # Add to generated clips
                    st.session_state.clips_generated.append(clip_data)
                    st.session_state.current_processing_index += 1
                    
                    st.success(f"✅ Generated clip {current_index + 1}")
                    
                    # Continue processing
                    if st.session_state.current_processing_index < total_segments:
                        st.rerun()
                    else:
                        st.session_state.processing_active = False
                        st.success("🎉 All clips generated!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error generating clip {current_index + 1}: {str(e)}")
                    st.session_state.current_processing_index += 1
                    if st.session_state.current_processing_index < total_segments:
                        st.rerun()
                    else:
                        st.session_state.processing_active = False
        else:
            st.session_state.processing_active = False
            st.rerun()

    # Show generated clips
    if st.session_state.clips_generated:
        st.markdown("---")
        st.header("🎬 Generated Vertical Clips")
        st.info(f"🎯 **Selected Parameters:** {', '.join(selected_parameters)} | **Crop Mode:** {crop_mode.title()}")
        
        # Check if clips still exist
        valid_clips = [clip for clip in st.session_state.clips_generated 
                      if clip.get("path") and os.path.isfile(clip["path"])]
        
        if not valid_clips:
            st.error("❌ All clip files are missing. Please regenerate clips.")
            st.session_state.clips_generated = []
            st.session_state.segments_to_process = []
            st.session_state.current_processing_index = 0
            st.session_state.processing_active = False
            return
        
        if len(valid_clips) != len(st.session_state.clips_generated):
            st.session_state.clips_generated = valid_clips
            st.warning(f"Some clip files were missing. Showing {len(valid_clips)} available clips.")
        
        # Display all generated clips
        for i, clip_data in enumerate(st.session_state.clips_generated, 1):
            display_single_clip(clip_data, platform, i)
        
        # Summary stats
        st.subheader("📈 Summary")
        avg_score = sum(c.get('score', 0) for c in st.session_state.clips_generated) / len(st.session_state.clips_generated)
        total_duration = sum(float(c.get('duration', '0').replace('s', '')) for c in st.session_state.clips_generated)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Clips", len(st.session_state.clips_generated))
        col2.metric("Average Score", f"{avg_score:.1f}/100")
        col3.metric("Total Duration", f"{total_duration:.1f}s")
        col4.metric("Format", "9:16 Vertical")
        
        # Reset button
        if st.button("🔄 Clear All Clips & Start Over", type="secondary"):
            for clip in st.session_state.clips_generated:
                try:
                    if clip.get("path") and os.path.isfile(clip["path"]):
                        os.unlink(clip["path"])
                except:
                    pass
            
            st.session_state.clips_generated = []
            st.session_state.segments_to_process = []
            st.session_state.current_processing_index = 0
            st.session_state.processing_active = False
            st.rerun()
        
        return

    # Initial processing button
    if not st.session_state.processing_active and not st.session_state.clips_generated:
        if st.button("🚀 Generate Vertical Clips", type="primary"):
            if not video_path or not os.path.isfile(video_path):
                st.error("Video file not found. Please reload your video.")
                return
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Smart Transcription (sample-based)
                status_text.text("🎤 Extracting audio sample for analysis...")
                progress_bar.progress(10)
                
                # Extract audio sample and transcribe
                audio_path, sample_start, sample_duration = extract_audio_sample(video_path, duration=300)  # 5 minute sample
                
                st.info("🎤 Transcribing audio sample...")
                with open(audio_path, "rb") as f:
                    resp = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
                
                transcript = resp.strip()
                
                # Clean up audio file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                
                st.success("✅ Transcription complete (from sample)")
                
                progress_bar.progress(40)
                with st.expander("📄 Transcript Preview (Sample)", expanded=False):
                    st.text_area("Sample Transcript", transcript, height=200, disabled=True)
                    st.info(f"📍 Sample taken from {sample_start/60:.1f} to {(sample_start + sample_duration)/60:.1f} minutes")

                # Step 2: AI analysis
                status_text.text(f"🤖 Analyzing transcript sample for viral segments...")
                progress_bar.progress(60)
                
                # Get video duration for AI context
                try:
                    temp_video = VideoFileClip(video_path)
                    video_duration = temp_video.duration
                    temp_video.close()
                    st.info(f"📏 Full video duration: {video_duration/60:.1f} minutes")
                except Exception as e:
                    video_duration = None
                    st.warning(f"Could not determine video duration: {str(e)}")
                
                # Analyze the sample transcript
                ai_json = analyze_transcript(transcript, platform, selected_parameters, client, sample_duration)
                st.success("✅ Analysis complete")

                with st.expander("🔍 AI Analysis Output", expanded=False):
                    st.code(ai_json, language="json")

                # Step 3: Parse segments and adjust timestamps
                status_text.text("📊 Processing segments and adjusting timestamps...")
                progress_bar.progress(80)
                
                segments = parse_segments(ai_json, sample_duration)
                if not segments:
                    st.warning("⚠️ No valid segments found in AI response.")
                    return
                
                # Adjust timestamps to work with full video
                if video_duration:
                    segments_adjusted = adjust_timestamps_for_sample(segments, sample_start, sample_duration, video_duration)
                else:
                    segments_adjusted = segments
                    
                segments_sorted = sorted(segments_adjusted, key=lambda x: x.get('score', 0), reverse=True)
                
                # Store segments for processing
                st.session_state.segments_to_process = segments_sorted
                st.session_state.current_processing_index = 0
                st.session_state.processing_active = True
                
                progress_bar.progress(100)
                status_text.text("✅ Starting clip generation...")
                
                st.success(f"🎉 Found {len(segments_sorted)} segments! Starting real-time generation of vertical clips...")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.info("💡 Try with a smaller video file or check your internet connection")
                return
    else:
        st.info("🎯 Click 'Generate Vertical Clips' to start processing your video.")


if __name__ == "__main__":
    main()
