import streamlit as st
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# ==========================================
# 0. EPIDEMIOLOGICAL FOOTPRINTS
# ==========================================
# Geographic bounding boxes for major US endemic fungi (Lat min, Lat max), (Lon min, Lon max)
ENDEMIC_REGIONS = {
    "Valley Fever (Coccidioides)": {"lat": (31.0, 38.0), "lon": (-125.0, -108.0)},
    "Histoplasmosis (Histoplasma)": {"lat": (34.0, 43.0), "lon": (-95.0, -79.0)},
    "Blastomycosis (Blastomyces)": {"lat": (30.0, 48.0), "lon": (-95.0, -79.0)},
    "PNW Cryptococcosis (C. gattii)": {"lat": (41.0, 49.0), "lon": (-125.0, -119.0)},
    "Sporotrichosis (Sporothrix)": {"lat": (30.0, 45.0), "lon": (-95.0, -75.0)},
    "Aspergillosis (Aspergillus)": {"lat": (24.0, 50.0), "lon": (-125.0, -66.0)} # Ubiquitous in US soils
}

def check_endemic_overlap(route_lat_bounds, route_lon_bounds):
    """Checks which fungi geographically overlap with the user's route."""
    threats = []
    r_lat_min, r_lat_max = route_lat_bounds
    r_lon_min, r_lon_max = route_lon_bounds
    
    for fungus, bounds in ENDEMIC_REGIONS.items():
        e_lat_min, e_lat_max = bounds["lat"]
        e_lon_min, e_lon_max = bounds["lon"]
        
        # Check for rectangle intersection
        if (r_lat_min <= e_lat_max and r_lat_max >= e_lat_min and 
            r_lon_min <= e_lon_max and r_lon_max >= e_lon_min):
            threats.append(fungus)
            
    return threats

# ==========================================
# 1. CORE FORECASTING ENGINE (UCAR THREDDS)
# ==========================================
class FungalAerosolForecaster:
    def __init__(self, lat_bounds, lon_bounds, resolution=0.25):
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.resolution = resolution

    def fetch_live_ucar_data(self):
        """Connects to the UCAR THREDDS Data Server (Academic Gold Standard)."""
        thredds_url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"
        
        try:
            ds = xr.open_dataset(thredds_url, engine='netcdf4')
            
            lon_min = (self.lon_bounds[0] + 360) % 360
            lon_max = (self.lon_bounds[1] + 360) % 360
            
            lat_min, lat_max = self.lat_bounds
            if ds.lat[0] > ds.lat[-1]:
                lat_slice = slice(lat_max, lat_min) # Descending
            else:
                lat_slice = slice(lat_min, lat_max) # Ascending
            
            u_var = 'u-component_of_wind_height_above_ground'
            v_var = 'v-component_of_wind_height_above_ground'
            soil_var = 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer'
            
            subset = ds[[u_var, v_var, soil_var]].sel(
                lat=lat_slice,
                lon=slice(lon_min, lon_max)
            ).isel(time=0).load()
            
            subset = subset.rename({
                u_var: 'ugrd10m',
                v_var: 'vgrd10m',
                soil_var: 'soilw0_10cm'
            })
            
            subset.coords['lon'] = (subset.coords['lon'] + 180) % 360 - 180
            return subset
            
        except Exception as e:
            st.error(f"Weather Server Connection Error: {e}. Please try again later.")
            return None

    def calculate_risk_index(self, ds):
        wind_speed = np.sqrt(ds.ugrd10m**2 + ds.vgrd10m**2)
        soil_moisture = ds.soilw0_10cm
        
        norm_wind = np.clip(wind_speed / 15.0, 0, 1)
        dryness_factor = np.clip(1.0 - (soil_moisture / 0.3), 0, 1)
        
        risk_index = norm_wind * dryness_factor
        risk_index = risk_index.squeeze()
        
        extra_dims = [dim for dim in risk_index.dims if dim not in ['lat', 'lon']]
        if extra_dims:
            risk_index = risk_index.isel({dim: 0 for dim in extra_dims})
        
        return xr.Dataset(
            {"aerosol_risk": (["lat", "lon"], risk_index.data)},
            coords={"lat": ds.lat.values, "lon": ds.lon.values}
        )

    def apply_fungus_mask(self, risk_ds, fungus_name):
        """Masks out areas of the risk map that fall outside a specific fungus's endemic region."""
        bounds = ENDEMIC_REGIONS[fungus_name]
        lat_min, lat_max = bounds['lat']
        lon_min, lon_max = bounds['lon']

        # Create a boolean mask keeping only valid coordinates for the specific fungus
        mask = (risk_ds.lat >= lat_min) & (risk_ds.lat <= lat_max) & \
               (risk_ds.lon >= lon_min) & (risk_ds.lon <= lon_max)

        # Apply the mask. Out-of-bounds pixels become NaN (invisible on map)
        return risk_ds.where(mask)

    def plot_risk_heatmap(self, risk_ds, threats, route_coords=None, custom_title=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Only plot if there is valid data (not all NaNs)
        if not np.isnan(risk_ds.aerosol_risk.values).all():
            contour = ax.contourf(
                risk_ds.lon, risk_ds.lat, risk_ds.aerosol_risk, 
                levels=np.linspace(0, 1, 11), cmap='YlOrRd', extend='max'
            )
            fig.colorbar(contour, ax=ax, label='Spore Lift Probability (0 = Low, 1 = High)')
        else:
            # Fallback if the map area is completely outside the fungus footprint
            ax.text(0.5, 0.5, "No Endemic Risk in this exact corridor.", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')

        if custom_title:
            ax.set_title(custom_title, fontsize=14)
        else:
            title_threats = ", ".join([t.split(" ")[0] for t in threats]) if threats else "General Dust/Aerosol"
            ax.set_title(f'Live Soil-to-Air Forecast: {title_threats}', fontsize=14)
            
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        ax.set_xlim(self.lon_bounds[0], self.lon_bounds[1])
        ax.set_ylim(self.lat_bounds[0], self.lat_bounds[1])
        
        if route_coords:
            lats, lons = zip(*route_coords)
            ax.plot(lons, lats, color='blue', marker='o', markersize=8, 
                    linestyle='dashed', linewidth=3, label="Your Route")
            ax.legend()
        
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        return fig

# ==========================================
# 2. UI & CLINICAL PROFILER
# ==========================================
st.set_page_config(page_title="Fungal Aerosol Forecaster", page_icon="🌬️", layout="wide")

st.sidebar.title("🩺 Your Health Profile")
st.sidebar.markdown("Tell us about your medical status so we can personalize your safety alerts.")

tac_dose = st.sidebar.slider("Current Tacrolimus (Prograf) Dose (mg/day)", 0.0, 20.0, 2.0, 0.5)
self_id = st.sidebar.select_slider(
    "How immunosuppressed do you consider yourself right now?",
    options=["Standard", "Moderate", "High", "Critical (Early Post-Op / Rejection)"]
)

if tac_dose >= 8.0 or "Critical" in self_id:
    profile_label = "🔴 Highest Caution"
    alert_threshold = 0.45
    advice = "Your immune defenses are currently very low. You will receive alerts even for mild dust events."
elif 4.0 <= tac_dose < 8.0 or self_id == "High":
    profile_label = "🟠 Moderate Caution"
    alert_threshold = 0.60
    advice = "You have lowered immunity. Alerts will trigger during moderate wind and dry conditions."
else:
    profile_label = "🟡 Standard Caution"
    alert_threshold = 0.75
    advice = "You are on a maintenance dose. Alerts will trigger during major dust storms or high winds."

st.sidebar.divider()
st.sidebar.metric("Your Safety Category", profile_label)
st.sidebar.info(advice)

page = st.sidebar.radio("Navigation", ["Dashboard", "Patient Guide & Science", "About the Author"])

# ==========================================
# 3. PAGE CONTENT ROUTING
# ==========================================

if page == "Dashboard":
    st.title("🌬️ Immunocompromised Soil & Dust Forecaster")
    st.markdown("### Protect your lungs from environmental fungi on your daily commute.")
    
    with st.expander("🔬 How does this tool work?"):
        st.write("When soil is extremely dry, fungi turn into microscopic spores. When the wind picks up, those spores are lifted into the air where you can breathe them in. This tool scans live weather satellites to find areas where dry soil and high winds intersect along your route.")

    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Where are you going?")
        st.write("Enter your location to scan for local fungal threats.")
        start_addr = st.text_input("Origin (City, State)", "Bakersfield, CA")
        end_addr = st.text_input("Destination (City, State)", "Fresno, CA")
        analyze_btn = st.button("Scan Route for Threats", type="primary")

    with col2:
        if analyze_btn:
            with st.spinner("Finding your route and checking local soil databases..."):
                geolocator = Nominatim(user_agent="fungal_tool_v9")
                loc1 = geolocator.geocode(start_addr)
                loc2 = geolocator.geocode(end_addr)
                
                if loc1 and loc2:
                    route = [(loc1.latitude, loc1.longitude), (loc2.latitude, loc2.longitude)]
                    
                    lat_pad, lon_pad = 2.0, 2.0
                    lats = [loc1.latitude, loc2.latitude]
                    lons = [loc1.longitude, loc2.longitude]
                    
                    dyn_lat_bounds = (min(lats) - lat_pad, max(lats) + lat_pad)
                    dyn_lon_bounds = (min(lons) - lon_pad, max(lons) + lon_pad)
                    
                    identified_threats = check_endemic_overlap(dyn_lat_bounds, dyn_lon_bounds)
                    
                    if identified_threats:
                        st.warning(f"🦠 **FUNGI DETECTED IN THIS REGION:** \n\n" + "\n".join([f"- {t}" for t in identified_threats]))
                    else:
                        st.info("ℹ️ No major regional fungi detected, but Aspergillus (a common compost/soil fungus) is present everywhere.")

                    with st.spinner("Pulling real-time wind and moisture data..."):
                        forecaster = FungalAerosolForecaster(lat_bounds=dyn_lat_bounds, lon_bounds=dyn_lon_bounds)
                        data = forecaster.fetch_live_ucar_data()
                        
                        if data is not None:
                            risk_matrix = forecaster.calculate_risk_index(data)
                            
                            # --- 1. PLOT COMPOSITE OVERLAY MAP ---
                            st.markdown("### 🗺️ Composite Threat Overlay")
                            st.write("This map shows the combined dust and aerosolization risk for your entire route.")
                            composite_title = "Overall Soil-to-Air Lift Risk"
                            st.pyplot(forecaster.plot_risk_heatmap(risk_matrix, identified_threats, route, custom_title=composite_title))
                            
                            # --- 2. PLOT ISOLATED SPECIFIC MAPS ---
                            if identified_threats:
                                st.divider()
                                st.markdown("### 🔬 Specific Pathogen Footprints")
                                st.write("These maps isolate the risk exclusively to the known geographic footprints of each detected fungus.")
                                
                                for fungus in identified_threats:
                                    # Mask the data to only show risk where THIS fungus lives
                                    masked_risk = forecaster.apply_fungus_mask(risk_matrix, fungus)
                                    specific_title = f"Isolated Threat Area: {fungus}"
                                    st.pyplot(forecaster.plot_risk_heatmap(masked_risk, [fungus], route, custom_title=specific_title))
                            
                            # --- 3. PATIENT ACTION PLAN ---
                            max_r = float(risk_matrix.aerosol_risk.max())
                            
                            st.divider()
                            if max_r >= alert_threshold:
                                st.error(f"🚨 **CLINICAL ALERT: High Spore Lift Risk Detected (Score: {max_r:.2f}).**")
                                st.markdown("### Your Action Plan for Today:")
                                st.markdown("""
                                - **Wear an N95 Mask:** Put on a fitted N95 or P100 respirator *before* you step outside. Standard surgical or cloth masks will not protect you.
                                - **Vehicle Safety:** Roll up all car windows. Turn your air conditioning to the **"Recirculate"** setting so you do not pull outside dust into the cabin.
                                - **Avoid Disturbance:** Cancel any plans for gardening, yard work, or sweeping outdoors. Avoid driving past active construction sites or freshly tilled farms.
                                """)
                            else:
                                st.success(f"✅ **Safe for Transit: The current air risk is low (Score: {max_r:.2f}).**")
                                st.markdown("### Your Action Plan for Today:")
                                st.markdown("""
                                - **Standard Precautions:** No heavy respiratory protection is needed for your commute today. 
                                - **General Hygiene:** Continue to practice standard hand-washing and avoid directly handling potting soil or compost.
                                """)
                else:
                    st.error("We couldn't find coordinates for those addresses. Please check your spelling and try again.")

elif page == "Patient Guide & Science":
    st.title("📚 Patient Education & The Science of Fungi")
    
    st.header("How Fungi Gets Into the Air")
    st.write("""
    Unlike viruses or bacteria, you do not catch endemic fungi from other people. They live naturally in the dirt. 
    When the soil is wet, they grow like a mat. When the soil undergoes severe drought, those mats undergo a process called **rhexolytic dehiscence** [3]. 
    This means their cell walls shatter into microscopic, barrel-shaped pieces called **arthroconidia** (spores). Because they are so tiny and light, 
    even a moderate breeze, a passing truck, or a shovel hitting the dirt can launch them into the air you breathe.
    """)

    st.header("Why Your Medications Put You At Risk")
    st.markdown("""
    When you receive a transplant, you are given medications like **Tacrolimus (Prograf)** or **Cyclosporine**. These are *calcineurin inhibitors* [2]. 
    
    **How they work:**
    * **T-Cells:** Your body has "soldier" cells called T-cells that usually hunt down and destroy fungal spores when you inhale them. 
    * **The Off Switch:** Tacrolimus essentially flips the "off switch" on your T-cells to stop them from attacking your new organ. 
    * **The Danger:** Because your T-cells are turned off, your lungs cannot defend against inhaled spores. While a healthy person might need to inhale 10,000 spores to get sick, a transplant patient can develop a severe, life-threatening infection (called Disseminated Disease) from inhaling just a few spores [2].
    """)

    st.header("Fungi Across the United States")
    st.write("Different regions of the US are home to different soil fungi. If you travel, your risk travels with you [1].")
    st.table({
        "Disease": ["Valley Fever", "Histoplasmosis", "Blastomycosis", "Cryptococcosis", "Sporotrichosis", "Aspergillosis"],
        "Fungus Name": ["Coccidioides", "Histoplasma", "Blastomyces", "Cryptococcus gattii", "Sporothrix", "Aspergillus"],
        "Where It Lives": ["Desert Southwest (AZ, CA)", "Midwest / Ohio River", "Midwest / Great Lakes", "Pacific Northwest", "Central US / Rose Bushes", "Ubiquitous (Everywhere)"],
        "Common Sources": ["Desert dust storms", "Bird/Bat droppings, caves", "Decaying wood near water", "Soil and Douglas Fir trees", "Potting soil, thorns, moss", "Compost, mulch, dust"]
    })

    st.header("What To Do If You Feel Sick")
    st.markdown("""
    Fungal infections often mimic the flu, COVID-19, or bacterial pneumonia. If you develop a fever, a dry cough, night sweats, or extreme fatigue that lasts more than a week, **contact your transplant coordinator immediately.** *Crucial Tip:* Always tell your doctor where you have recently traveled or if you have been exposed to a dust storm. Doctors outside of the Southwest often forget to test for things like Valley Fever! [4]
    """)

    st.divider()
    st.header("Scientific References & Citations")
    st.markdown("""
    1. **Gorris, M. E., et al. (2018).** *Expansion of Coccidioidomycosis Endemic Regions in the United States in Response to Climate Change.* GeoHealth, 2(10), 308-327. (Details the environmental footprint and the relationship between climate factors and endemic fungus).
    2. **Baddley, J. W., et al. (2019).** *Endemic Fungal Infections in Solid Organ Transplant Recipients.* American Journal of Transplantation. (Establishes guidelines on the impact of calcineurin inhibitors, like Tacrolimus, on fungal immunity).
    3. **Comrie, A. C. (2005).** *Climate Factors Influencing Coccidioidomycosis Seasonality and Outbreaks.* Environmental Health Perspectives, 113(6), 688-692. (Explains the mechanics of the "Grow and Blow" hypothesis and environmental triggers for rhexolytic dehiscence).
    4. **Galgiani, J. N., et al. (2016).** *Executive Summary: 2016 Infectious Diseases Society of America (IDSA) Clinical Practice Guideline for the Treatment of Coccidioidomycosis.* Clinical Infectious Diseases, 63(6), 717-722. (Outlines the standard of care for clinical prophylaxis, masking, and avoidance).
    """)

elif page == "About the Author":
    st.title("About the Author")
    st.write("### Neel Agarwal")
    st.markdown("""
    Developed by a third year Med Student at OSU College of Medicine at the intersection of **Atmospheric Physics** and **Precision Public Health**. 
    This tool transforms global weather and satellite datasets into personalized, actionable clinical prophylaxis 
    for the most vulnerable patient populations. Contact me at neel.agarwal@osumc.edu
    """)