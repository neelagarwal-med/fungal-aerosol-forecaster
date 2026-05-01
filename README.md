# 🌬️ Immunocompromised Fungal Aerosolization Forecaster

A clinical-grade, location-agnostic decision support tool built with Streamlit. This application translates real-time, planetary-scale meteorological data into personalized prophylactic advice for post-transplant and highly immunosuppressed patients, protecting them from endemic environmental fungi.

## 🔬 The Science: "Grow and Blow"
Endemic fungi (like *Coccidioides* and *Histoplasma*) live in soil. During droughts, their mycelial mats undergo *rhexolytic dehiscence*, shattering into microscopic arthroconidia (spores). High winds or soil disturbance aerosolize these spores. 

Patients on calcineurin inhibitors (e.g., Tacrolimus) have suppressed T-cell responses, making their lungs highly vulnerable to inhaled spores. This tool calculates the intersection of mechanical lift energy (wind) and soil friability (dryness) to alert patients *before* they step into a high-risk environment.

## ✨ Key Features
* **Clinical Risk Profiler:** Dynamically calibrates safety alert thresholds based on the user's daily Tacrolimus dose (mg) or self-identified vulnerability status.
* **Academic Gold Standard Data:** Bypasses unstable commercial mirrors by utilizing an anonymous `netcdf4` connection to the UCAR THREDDS Data Server for live GFS atmospheric data.
* **Location-Agnostic Scanning:** Users input a start and end destination; the app calculates a dynamic bounding box and cross-references it against the geographic footprints of 6 major US fungal threats.
* **Multi-Layer Heatmaps:** Generates a composite soil-to-air lift risk map, alongside isolated footprints for specific identified pathogens.
* **Actionable Prophylaxis:** Provides clear, bulleted action plans (e.g., N95 masking, HVAC recirculation) when environmental risk exceeds the patient's clinical threshold.

## 🦠 Tracked Pathogens
1. *Coccidioides* (Valley Fever)
2. *Histoplasma* (Histoplasmosis)
3. *Blastomyces* (Blastomycosis)
4. *Cryptococcus gattii* (PNW Cryptococcosis)
5. *Sporothrix* (Sporotrichosis)
6. *Aspergillus* (Aspergillosis)

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR-USERNAME/fungal-aerosol-forecaster.git](https://github.com/YOUR-USERNAME/fungal-aerosol-forecaster.git)
   cd fungal-aerosol-forecaster
