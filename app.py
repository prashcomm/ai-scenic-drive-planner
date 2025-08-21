import streamlit as st
import os
import requests
import polyline
import google.generativeai as genai
from dotenv import load_dotenv
import time
import imageio.v2 as imageio
from io import BytesIO
from streamlit_searchbox import st_searchbox
from datetime import datetime
import base64 # NEW: Required for embedding the GIF

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please check your API key. Error: {e}")

# --- 2. DATA FETCHING & PROCESSING FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_autocomplete_suggestions(search_text: str) -> list[str]:
    if not search_text: return []
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={search_text}&key={GOOGLE_MAPS_API_KEY}&components=country:IN"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        predictions = response.json().get("predictions", [])
        return [pred['description'] for pred in predictions]
    except Exception: return []

@st.cache_data(show_spinner=False)
def get_route_options(start, end):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start}&destination={end}&key={GOOGLE_MAPS_API_KEY}&alternatives=true&region=IN"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        routes_data = response.json()
        if not routes_data.get("routes"): return None
        return [{"id": f"Route {i+1}", "summary": r.get("summary", ""), "distance_text": r["legs"][0]["distance"]["text"], "duration_text": r["legs"][0]["duration"]["text"], "polyline": r["overview_polyline"]["points"]} for i, r in enumerate(routes_data["routes"])]
    except Exception: return None

@st.cache_data(show_spinner=False)
def get_scenic_spots(route_polyline):
    path = polyline.decode(route_polyline)
    sample_points = [path[int(i * len(path) / 7)] for i in range(7)]
    scenic_places = {}
    keywords = ["scenic lookout", "waterfall", "historic landmark", "temple", "beach", "hiking area"]
    allowed_types = {"tourist_attraction", "park", "museum", "natural_feature", "zoo", "art_gallery", "landmark"}
    for lat, lng in sample_points:
        for keyword in keywords:
            url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=10000&keyword={keyword}&key={GOOGLE_MAPS_API_KEY}"
            try:
                response = requests.get(url, timeout=10)
                results = response.json().get("results", [])
                for place in results:
                    place_types = set(place.get('types', []))
                    if (place.get('photos') and place.get('rating', 0) > 4.0 and not allowed_types.isdisjoint(place_types)):
                        scenic_places[place['place_id']] = {"name": place['name'], "photo_reference": place['photos'][0]['photo_reference']}
            except Exception: continue
    return list(scenic_places.values())

@st.cache_data(show_spinner=False)
def get_pit_stops(route_polyline):
    path = polyline.decode(route_polyline)
    midpoint = path[len(path) // 2]
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={midpoint[0]},{midpoint[1]}&radius=10000&type=restaurant&keyword=cafe OR restaurant&key={GOOGLE_MAPS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        results = response.json().get("results", [])
        return [{"name": p['name'], "rating": p.get('rating', 'N/A'), "total_ratings": p.get('user_ratings_total', 0)} for p in sorted(results, key=lambda x: x.get('rating', 0), reverse=True)[:5]]
    except Exception: return []

@st.cache_data(show_spinner=False)
def create_drive_preview_assets(polyline_str):
    path = polyline.decode(polyline_str)
    images_for_gif, image_urls_for_grid = [], []
    sample_points = [path[int(i * len(path) / 25)] for i in range(1, 25)]
    for lat, lng in sample_points:
        if len(image_urls_for_grid) >= 12: break
        metadata_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
        try:
            meta_response = requests.get(metadata_url, timeout=5)
            if meta_response.json().get("status") == "OK":
                image_url = f"https://maps.googleapis.com/maps/api/streetview?size=400x300&location={lat},{lng}&fov=90&heading=235&pitch=10&key={GOOGLE_MAPS_API_KEY}"
                image_response = requests.get(image_url, timeout=10)
                if image_response.status_code == 200:
                    image_urls_for_grid.append(image_url)
                    if len(images_for_gif) < 10:
                        images_for_gif.append(imageio.imread(image_response.content))
        except Exception: continue
    if not images_for_gif: return None, None
    gif_bytes = BytesIO()
    imageio.mimsave(gif_bytes, images_for_gif, 'GIF', duration=0.5, loop=0)
    gif_bytes.seek(0)
    return gif_bytes, image_urls_for_grid

# --- 3. AI & REPORT GENERATION FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_llm_choice(route_options_with_scenery):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = "Analyze the following driving route options and determine which is the most scenic. Respond with only the number of your chosen route (e.g., '1', '2', '3').\n\n"
    for i, route in enumerate(route_options_with_scenery):
        scenic_summary = f"{len(route['scenic_spots'])} scenic spots found." if route['scenic_spots'] else "Direct route."
        prompt += f"**Option {i+1}**: via {route['summary']}. Details: {scenic_summary}\n"
    try:
        response = model.generate_content(prompt)
        choice_number = int("".join(filter(str.isdigit, response.text)))
        return f"Route {choice_number}"
    except (ValueError, IndexError, Exception): return None

def get_llm_narrative_stream(chosen_route):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    scenic_spot_names = [spot['name'] for spot in chosen_route['scenic_spots']]
    prompt = (
        f"You are an Expert Drive Curator, blending the passion of a driving enthusiast with the precision of a route analyst. Your goal is to create a compelling, informative, and concise summary of the following scenic drive.\n\n"
        f"**Route Data:**\n"
        f"- **Primary Roads:** {chosen_route['summary']}\n"
        f"- **Detected Key Sights:** {', '.join(scenic_spot_names[:3]) if scenic_spot_names else 'None specified'}\n\n"
        f"**Instructions:**\n"
        f"Write an engaging narrative (2-3 short paragraphs). Start with a strong opening that captures the essence of the drive. Then, describe the road itself—mentioning specific highways, the terrain (urban, ghats, coastal), and the driving feel (winding, open, etc.). Weave in a description of one or two of the most significant Key Sights by name. Conclude with a summary of what makes this drive special and who it's perfect for. Your tone should be knowledgeable and exciting, but grounded in the facts provided."
    )
    try:
        return model.generate_content(prompt, stream=True)
    except Exception as e:
        st.error(f"The AI narrative could not be generated: {str(e)[:150]}")
        return None

def stream_text_generator(stream):
    for chunk in stream:
        if chunk.parts:
            yield chunk.text

# UPDATED: This function now includes the animated GIF in the HTML export.
def generate_report_html(data):
    """Builds a standalone HTML report, now with the animated GIF embedded."""
    
    # NEW: Encode the GIF into Base64 to embed it directly in the HTML file.
    gif_base64 = base64.b64encode(data['drive_gif'].getvalue()).decode("utf-8")
    
    drive_preview_html = "".join([f'<img src="{url}" alt="Drive Preview">' for url in data['grid_urls']])
    scenic_spots_html = ""
    for spot in data['scenic_spots'][:8]:
        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={spot['photo_reference']}&key={GOOGLE_MAPS_API_KEY}"
        scenic_spots_html += f'<div><img src="{photo_url}" alt="{spot["name"]}"><b>{spot["name"]}</b></div>'
    pit_stops_html = "".join([f"<li><b>{stop['name']}</b> - ⭐ {stop['rating']} ({stop['total_ratings']} ratings)</li>" for stop in data['pit_stops']])
    
    html_template = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scenic Drive Report</title><style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; background-color: #f0f2f6; }}
    .container {{ max-width: 900px; margin: 20px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    h1, h2 {{ color: #1a1a1a; border-bottom: 2px solid #007bff; padding-bottom: 10px; }} h1 {{ font-size: 2.5em; text-align: center; border: none; }}
    h2 {{ font-size: 1.8em; margin-top: 40px; }} p, li {{ color: #333; line-height: 1.6; }} .header {{ text-align: center; margin-bottom: 30px; }}
    .header p {{ font-size: 1.2em; color: #555; }} .map img, .gif-preview img {{ max-width: 100%; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 15px;}}
    .grid img {{ width: 100%; border-radius: 8px; }} .grid div {{ text-align: center; }} ul {{ padding-left: 20px; }}</style></head><body><div class="container">
    <div class="header"><h1>Scenic Drive Report</h1><p>From <strong>{data['start_destination']}</strong> to <strong>{data['end_destination']}</strong></p>
    <p>{data['chosen_route']['distance_text']} / {data['chosen_route']['duration_text']}</p></div>
    <div class="map"><img src="{data['map_url']}" alt="Route Map"></div>
    <h2>💡 AI Route Analysis</h2><p>{data['ai_narrative'].replace('\\n', '<br>')}</p>
    
    <h2>🛣️ Animated Drive Preview</h2><div class="gif-preview"><img src="data:image/gif;base64,{gif_base64}" alt="Animated Drive Preview"></div>
    
    <h2>Key Vistas</h2><div class="grid">{drive_preview_html}</div>
    <h2>🏞️ Sights to See</h2><div class="grid">{scenic_spots_html}</div>
    <h2>☕ Recommended Pit Stops</h2><ul>{pit_stops_html}</ul></div></body></html>
    """
    return html_template

# --- 4. UI & MAIN LOGIC ---
st.set_page_config(page_title="AI Scenic Route Planner 🚗", layout="wide")
st.title("AI-Powered Scenic Route Planner")
st.subheader("Plan Your Next Great Drive")

if 'report_data' not in st.session_state:
    st.session_state.report_data = None

col1, col2 = st.columns(2)
with col1:
    start_destination = st_searchbox(get_autocomplete_suggestions, placeholder="Type your start destination...", key="start_searchbox")
with col2:
    end_destination = st_searchbox(get_autocomplete_suggestions, placeholder="Type your end destination...", key="end_searchbox")

if st.button("✨ Plan My Scenic Drive", type="primary"):
    if start_destination and end_destination:
        st.session_state.report_data = None
        progress_placeholder = st.empty()
        with st.spinner('Crafting your ultimate scenic drive...'):
            progress_placeholder.info("1/4: Fetching all possible routes...")
            route_options = get_route_options(start_destination, end_destination)
            if not route_options: st.error("No routes found."); st.stop()
            progress_placeholder.info("2/4: Searching for scenic spots...")
            for route in route_options: route['scenic_spots'] = get_scenic_spots(route['polyline'])
            progress_placeholder.info("3/4: Asking AI to choose the best route...")
            chosen_route_id = get_llm_choice(route_options)
            if not chosen_route_id or chosen_route_id not in [r['id'] for r in route_options]:
                st.warning("AI choice inconclusive, defaulting to route with most scenic spots.")
                chosen_route = max(route_options, key=lambda r: len(r['scenic_spots']))
            else:
                chosen_route = next((r for r in route_options if r['id'] == chosen_route_id), None)
            if not chosen_route: st.error("Could not determine a final route."); st.stop()
            
            progress_placeholder.info("4/4: Generating your personalized drive preview...")
            drive_gif, grid_urls = create_drive_preview_assets(chosen_route['polyline'])
            pit_stops = get_pit_stops(chosen_route['polyline'])
            narrative_stream = get_llm_narrative_stream(chosen_route)
            ai_narrative_text = "".join(list(stream_text_generator(narrative_stream))) if narrative_stream else "AI narrative could not be generated."

        progress_placeholder.empty()
        st.success("🎉 Your scenic route is ready!")
        map_url = f"https://maps.googleapis.com/maps/api/staticmap?size=600x400&path=weight:4|color:0x0000ff|enc:{chosen_route['polyline']}&markers=color:green|label:A|{start_destination}&markers=color:red|label:B|{end_destination}&key={GOOGLE_MAPS_API_KEY}"
        
        # UPDATED: The drive_gif object is now passed to the report data state
        st.session_state.report_data = {
            "start_destination": start_destination, "end_destination": end_destination, "chosen_route": chosen_route,
            "ai_narrative": ai_narrative_text, "map_url": map_url, "grid_urls": grid_urls,
            "scenic_spots": chosen_route['scenic_spots'], "pit_stops": pit_stops, "drive_gif": drive_gif
        }

# This block displays the dashboard and download button
if st.session_state.report_data:
    data = st.session_state.report_data
    
    metric_col1, metric_col2, map_col = st.columns([1, 1, 1.5])
    with metric_col1: st.metric(label="Distance", value=data['chosen_route']['distance_text'])
    with metric_col2: st.metric(label="Estimated Duration", value=data['chosen_route']['duration_text'])
    with map_col: st.image(data['map_url'], caption="Your Recommended Route")

    st.header("💡 Your Expert Guide's Analysis")
    st.markdown(data['ai_narrative'])
    
    report_html = generate_report_html(data)
    archive_dir = "archive"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"scenic_route_{timestamp}.html"
    with open(os.path.join(archive_dir, file_name), "w", encoding="utf-8") as f:
        f.write(report_html)
    st.info(f"Report has been automatically saved to the '{archive_dir}' folder.")

    st.download_button(
        label="📥 Download Report as Webpage",
        data=report_html,
        file_name=f"scenic_route_{data['start_destination']}_to_{data['end_destination']}.html",
        mime="text/html"
    )

    st.header("🛣️ Drive Preview")
    gif_col, grid_col = st.columns(2)
    with gif_col:
        if data['drive_gif']:
            st.image(data['drive_gif'], caption="Animated Preview")
        else:
            st.info("Animated preview not available.")
    with grid_col:
        st.subheader("Key Vistas")
        if data['grid_urls']:
            cols = st.columns(4) 
            for i, url in enumerate(data['grid_urls'][:12]):
                with cols[i % 4]:
                    st.image(url, use_container_width=True)
        else:
            st.info("Static previews not available.")

    st.header("🏞️ Sights to See")
    if data['scenic_spots']:
        cols = st.columns(4)
        for i, spot in enumerate(data['scenic_spots'][:8]):
            with cols[i % 4]:
                st.markdown(f"<p style='font-size:14px;'><b>{spot['name']}</b></p>", unsafe_allow_html=True)
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={spot['photo_reference']}&key={GOOGLE_MAPS_API_KEY}"
                try:
                    response = requests.get(photo_url, stream=True)
                    if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                        st.image(photo_url, use_container_width=True)
                except Exception: pass
    else:
        st.info("This route is more direct.")

    st.header("☕ Recommended Pit Stops")
    if data['pit_stops']:
        for stop in data['pit_stops']: st.markdown(f"**{stop['name']}** - ⭐ {stop['rating']} ({stop['total_ratings']} ratings)")
    else:
        st.info("No high-rated pit stops were found near the midpoint.")