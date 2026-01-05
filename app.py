import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import joblib
import matplotlib.pyplot as plt
import altair as alt
from io import BytesIO
import requests
import datetime
import sklearn.metrics

# =============================================================================
# 1. KONFIGURASI & CSS MODERN
# =============================================================================
st.set_page_config(layout="wide", page_title="Prediksi Curah Hujan Jatim", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;700&display=swap');
    html, body, [class*="css-"] {font-family: 'Montserrat', sans-serif;}
    .main {background-color: #f0f2f6;}
    h1 {font-weight: 700; color: #1e3a8a; text-align: center; margin-bottom: 0.5rem;}
    h3 {font-weight: 600; color: #1e40af;}
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.2rem; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e3a8a); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .map-container {
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: #ffffff; border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;
    }
    .metric-card h4 {color: #1e40af; margin: 0; font-size: 1rem;}
    .metric-card p {font-size: 1.8rem; font-weight: 700; color: #2563eb; margin: 0.5rem 0 0;}
</style>
""", unsafe_allow_html=True)

st.title("Curah Hujan Provinsi Jawa Timur")
st.markdown("<p style='text-align:center; color:#555; font-style:italic;'>Metode: Ordinal Regression (GRU) • Input: Satelit Himawari-9 + Data AWS</p>", unsafe_allow_html=True)

# =============================================================================
# 2. INISIALISASI SESSION STATE
# =============================================================================
for key in ['data_prediksi', 'data_aktual_future', 'waktu_target', 'raw_rain_pred', 'raw_rain_actual', 'model_accuracy']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'selected_kabs' not in st.session_state:
    st.session_state.selected_kabs = []

# =============================================================================
# 3. CUSTOM LAYER ATTENTION
# =============================================================================
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        return K.sum(x * K.expand_dims(alpha, axis=-1), axis=1)

# =============================================================================
# 4. LOAD RESOURCES
# =============================================================================
STATION_TO_KAB = {
    "Juanda": "Sidoarjo", "Banyuwangi": "Banyuwangi", "Batu": "Batu",
    "Bromo": "Lumajang", "Kalianget": "Sumenep", "Karangkates": "Malang",
    "Kediri": "Kediri", "Pasuruan": "Pasuruan", "Probolinggo": "Probolinggo",
    "Tuban": "Tuban"
}

@st.cache_resource
def load_resources():
    model = load_model('model_ordinal_final_GRU.h5', custom_objects={'Attention': Attention}, compile=False)
    X_full = np.load('X_full_seq2seq.npy')
    df_meta = pd.read_csv('metadata_full_seq2seq.csv')
    df_meta['Tanggal'] = pd.to_datetime(df_meta['Tanggal'])
    try: scaler = joblib.load('scaler_jatim_final.pkl')
    except: scaler = None

    try: gdf = gpd.read_file('jawa-timur-simplified-topo.json')
    except: gdf = gpd.read_file('kabupaten_jatim.geojson')
    
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    
    possible_cols = ['kabkot', 'NAME_2', 'KAB_KOTA', 'kabupaten', 'Name', 'KABUPATEN']
    col = next((c for c in possible_cols if c in gdf.columns), gdf.columns[0])
    gdf = gdf.set_index(col)
    
    lookup_dict = dict(zip(zip(df_meta['Station_ID'], df_meta['Tanggal']), df_meta.index))
    return model, X_full, df_meta, scaler, gdf, lookup_dict

try:
    model, X_full, df_meta, scaler, gdf, lookup_dict = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource: {e}")
    st.stop()

# =============================================================================
# 5. PROSES PREDIKSI
# =============================================================================
def proses_prediksi():
    tgl = st.session_state.widget_tanggal
    jam = st.session_state.widget_jam
    target_dt = pd.to_datetime(f"{tgl} {jam}")
    
    batch_meta = df_meta[df_meta['Tanggal'] == target_dt]
    if batch_meta.empty:
        st.session_state.data_prediksi = "KOSONG"
        st.session_state.waktu_target = target_dt
        return
    
    indices = batch_meta.index.tolist()
    X_batch = X_full[indices]
    y_probs = model.predict(X_batch, verbose=0)
    
    def get_raw_mm(idx):
        if idx is None: return 0.0
        seq = X_full[idx]
        if scaler:
            try:
                inv = scaler.inverse_transform(seq.reshape(6, -1))
                return inv[-1, -1]
            except:
                return seq[-1, -1]
        return seq[-1, -1]
    
    if scaler:
        try:
            X_inv = scaler.inverse_transform(X_batch.reshape(-1, X_batch.shape[-1])).reshape(X_batch.shape)
            rain_hist = X_inv[:, :, -1]
        except:
            rain_hist = X_batch[:, :, -1]
    else:
        rain_hist = X_batch[:, :, -1]
    
    def get_lvl(mm): return 0 if mm < 0.5 else (1 if mm < 5.0 else 2)
    def decode(p):
        is_rain = 1 if p[0] > 0.5 else 0
        is_heavy = 1 if p[1] > 0.35 else 0
        return is_rain + is_heavy
    
    times_hist = [-60, -50, -40, -30, -20, -10]
    times_future = [10, 20, 30, 40, 50, 60]
    
    hasil_pred = {}
    hasil_actual = {}
    raw_rain_pred = {}
    raw_rain_actual = {}
    y_true, y_pred = [], []
    
    for i, idx_meta in enumerate(indices):
        st_id = batch_meta.iloc[i]['Station_ID']
        kab = STATION_TO_KAB.get(st_id)
        if kab and kab in gdf.index:
            levels_pred = {t: get_lvl(rain_hist[i][j]) for j, t in enumerate(times_hist)}
            raw_pred = {t: rain_hist[i][j] for j, t in enumerate(times_hist)}
            for j, t in enumerate(times_future):
                lvl = decode(y_probs[i][j])
                levels_pred[t] = lvl
                approx_mm = 0.0 if lvl == 0 else (2.5 if lvl == 1 else 10.0)
                raw_pred[t] = approx_mm
                
                future_dt = target_dt + pd.Timedelta(minutes=t + 10)
                idx_act = lookup_dict.get((st_id, future_dt))
                if idx_act is not None:
                    val = get_raw_mm(idx_act)
                    act_lvl = get_lvl(val)
                    y_true.append(act_lvl)
                    y_pred.append(lvl)
            
            hasil_pred[kab] = levels_pred
            raw_rain_pred[kab] = raw_pred
            
            levels_act = levels_pred.copy()
            raw_act = raw_pred.copy()
            for t_min in times_future:
                future_dt = target_dt + pd.Timedelta(minutes=t_min + 10)
                idx_act = lookup_dict.get((st_id, future_dt))
                val = get_raw_mm(idx_act)
                levels_act[t_min] = get_lvl(val) if idx_act is not None else -1
                raw_act[t_min] = val if idx_act is not None else np.nan
            hasil_actual[kab] = levels_act
            raw_rain_actual[kab] = raw_act
    
    if y_true:
        st.session_state.model_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        st.session_state.model_accuracy = None
    
    st.session_state.data_prediksi = hasil_pred
    st.session_state.data_aktual_future = hasil_actual
    st.session_state.raw_rain_pred = raw_rain_pred
    st.session_state.raw_rain_actual = raw_rain_actual
    st.session_state.waktu_target = target_dt

# =============================================================================
# 6. REAL-TIME API (Open-Meteo)
# =============================================================================
def fetch_real_time_weather(kab):
    coords = {
        "Sidoarjo": (-7.45, 112.72), "Banyuwangi": (-8.22, 114.37), "Batu": (-7.87, 112.52),
        "Probolinggo": (-7.75, 113.22), "Sumenep": (-7.02, 113.87), "Malang": (-7.98, 112.63),
        "Kediri": (-7.82, 112.02), "Pasuruan": (-7.65, 112.90), "Tuban": (-6.90, 112.05), "Bromo": (-7.92, 112.96)
    }
    if kab not in coords:
        return None
    lat, lon = coords[kab]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=precipitation,temperature_2m,relative_humidity_2m"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

# =============================================================================
# 7. SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("Dashboard")
    
    base_map_choice = st.selectbox(
        "Jenis Peta Dasar",
        ["Terang (Positron)", "Gelap (Dark Matter)", "Satelit", "OpenStreetMap", "Terrain (Pegunungan)"],
        index=0,
        key="base_map_select"
    )
    
    kab_list = sorted(STATION_TO_KAB.values())
    st.multiselect("Kabupaten/Kota", options=kab_list, key='selected_kabs')
    
    st.markdown("---")
    st.header("Tanggal dan Waktu")
    dates = sorted(df_meta['Tanggal'].dt.date.unique())
    st.date_input("Tanggal", min_value=min(dates), max_value=max(dates), value=max(dates), key='widget_tanggal')
    
    # INPUT JAM PER 10 MENIT — CARA YANG BENAR
    st.time_input(
        "Jam (WIB)",
        value=datetime.time(12, 0),  # default jam 12:00
        step=datetime.timedelta(minutes=10),  # ← INI YANG UTAMA: langkah 10 menit
        key='widget_jam'
    )
    
    st.button("Cari dan Prediksi", type="primary", on_click=proses_prediksi, use_container_width=True)

# =============================================================================
# 8. MAIN CONTENT
# =============================================================================
pred_data = st.session_state.data_prediksi
target_time = st.session_state.waktu_target
accuracy = st.session_state.model_accuracy
act_data = st.session_state.data_aktual_future

if pred_data == "KOSONG":
    st.info("⚠️ Tidak ada data untuk waktu yang dipilih.")
elif pred_data is not None:
    st.markdown(f"<h3 style='text-align:center; color:#1e40af;'>Tanggal: {target_time.strftime('%d/%m/%Y')}", unsafe_allow_html=True)

    # METRIC CARDS DI ATAS SEMUA TAB
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Jam</h4><p>{target_time.strftime("%H:%M")}</p></div>', unsafe_allow_html=True)
    with col2:
        # TOTAL JUMLAH STASIUN YANG DIPANTAU (ANGKA SAJA)
        total_stasiun = len(pred_data)
        st.markdown(f'''
        <div class="metric-card">
            <h4>Jumlah AWS</h4>
            <p style="font-size: 1.8rem; font-weight: 700; color: #2563eb; margin: 0.5rem 0 0;">{total_stasiun}</p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        acc_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        st.markdown(f'<div class="metric-card"><h4>Akurasi Model</h4><p>{acc_text}</p></div>', unsafe_allow_html=True)

    # TAB BARU: TAB 2 ADALAH REAL-TIME
    tab1, tab_rt, tab2, tab3, tab4 = st.tabs([
        "Peta Prediksi",
        "Real-Time Cuaca",
        "Tabel Data",
        "Grafik Tren",
        "Export"
    ])

     # TAB 1: PETA PREDIKSI
    with tab1:
        waktu_pilihan = st.select_slider(
            "Pilih Waktu Prediksi",
            options=[-60, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60],
            value=10,
            format_func=lambda x: f"T{x:+d} menit" if x != 0 else "T0"
        )

        # Base map mapping
        base_map_mapping = {
            "Terang (Positron)": {"tiles": "CartoDB positron", "attr": "© CartoDB | © OpenStreetMap contributors"},
            "Gelap (Dark Matter)": {"tiles": "CartoDB dark_matter", "attr": "© CartoDB | © OpenStreetMap contributors"},
            "Satelit": {"tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        "attr": "© Esri | Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP"},
            "OpenStreetMap": {"tiles": "openstreetmap", "attr": "© OpenStreetMap contributors"},
            "Terrain (Pegunungan)": {"tiles": "Stamen Terrain", "attr": "Map tiles by Stamen Design, © OpenStreetMap contributors"}
        }

        config = base_map_mapping[base_map_choice]

        # Buat peta
        m = folium.Map(
            location=[-7.8, 112.5],
            zoom_start=8,
            tiles=config["tiles"],
            attr=config["attr"],
            control_scale=True
        )

        # Tambahkan semua base map lain → otomatis muncul layer control kecil
        for name, cfg in base_map_mapping.items():
            if name != base_map_choice:
                folium.TileLayer(
                    tiles=cfg["tiles"],
                    attr=cfg["attr"],
                    name=name,
                    overlay=False  # penting untuk base layer
                ).add_to(m)

        # Fullscreen tetap
        Fullscreen(position='topright', title='Fullscreen', title_cancel='Keluar').add_to(m)

        # GeoJSON prediksi
        gdf_display = gdf.reset_index()
        target_col = gdf.index.name or gdf.columns[0]

        def get_style(feature, t):
            nama = feature['properties'].get(target_col)
            lvl = pred_data.get(nama, {}).get(t, 0) if nama in pred_data else 0
            colors = {0: '#ffffff', 1: '#00FF00', 2: '#FF0000', -1: '#000000'}
            return {'fillColor': colors.get(lvl, '#ffffff'), 'color': '#333', 'weight': 1.5, 'fillOpacity': 0.7}

        folium.GeoJson(
            gdf_display,
            style_function=lambda x: get_style(x, waktu_pilihan),
            tooltip=folium.GeoJsonTooltip([target_col], aliases=["Kabupaten/Kota:"])
        ).add_to(m)

        # LEGENDA KECIL DI KIRI BAWAH
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 15px; left: 15px; width: 200px; height: auto; 
                    background-color: rgba(255, 255, 255, 0.9); 
                    border-radius: 10px; 
                    padding: 10px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    z-index: 1000; 
                    font-size: 12px; 
                    line-height: 1.4;">
          <b style="font-size: 13px; color: #1e3a8a; display: block; margin-bottom: 8px;">Keterangan Curah Hujan</b>
          <i style="background:#ffffff; border:1px solid #aaa; width:16px; height:16px; display:inline-block; border-radius:3px; vertical-align:middle;"></i> <small>Tidak Hujan</small><br>
          <i style="background:#00FF00; width:16px; height:16px; display:inline-block; border-radius:3px; vertical-align:middle;"></i> <small>Hujan Ringan</small><br>
          <i style="background:#FF0000; width:16px; height:16px; display:inline-block; border-radius:3px; vertical-align:middle;"></i> <small>Hujan Lebat</small><br>
          <i style="background:#000000; width:16px; height:16px; display:inline-block; border-radius:3px; vertical-align:middle;"></i> <small>Tidak Tersedia</small>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # CSS CUSTOM: Buat Layer Control (toggle base map) jadi KECIL & CANTIK
        layer_control_css = """
        <style>
            /* Layer control jadi kecil dan transparan */
            .leaflet-control-layers {
                background: rgba(255, 255, 255, 0.9) !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
                font-size: 12px !important;
                padding: 6px 8px !important;
            }
            .leaflet-control-layers-toggle {
                width: 30px !important;
                height: 30px !important;
                background-size: 16px !important;
            }
            .leaflet-control-layers label {
                margin: 4px 0 !important;
            }
        </style>
        """
        m.get_root().html.add_child(folium.Element(layer_control_css))

        # Render peta — PASTI MUNCUL & CANTIK
        st.markdown("<div class='map-container'>", unsafe_allow_html=True)
        st_folium(
            m,
            use_container_width=True,
            height=600,
            key=f"map_prediksi_{waktu_pilihan}_{base_map_choice}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Peringatan Dini
        hujan_lebat = [k for k, v in pred_data.items() if v.get(waktu_pilihan) == 2]
        if hujan_lebat:
            st.warning(f"⚠️ PERINGATAN DINI: Potensi hujan lebat di **{', '.join(hujan_lebat)}** pada T{waktu_pilihan:+d} menit")
        else:
            st.success("✅ Tidak terdeteksi potensi hujan lebat saat ini. Tetap waspada.")

    # TAB 2: REAL-TIME CUACA 
    with tab_rt:
        st.subheader("Data Cuaca Real-Time")
        st.caption("Data diperbarui secara otomatis dari sumber terbuka. Pilih kabupaten di sidebar untuk melihat detail.")

        if st.session_state.selected_kabs:
            for kab in st.session_state.selected_kabs:
                with st.spinner(f"Mengambil data cuaca terkini untuk {kab}..."):
                    data = fetch_real_time_weather(kab)
                    if data and 'current_weather' in data:
                        curr = data['current_weather']
                        hourly_time = [t[-5:] for t in data['hourly']['time'][:12]]
                        hourly_precip = data['hourly']['precipitation'][:12]
                        hourly_temp = data['hourly']['temperature_2m'][:12]
                        hourly_humidity = data['hourly']['relative_humidity_2m'][:12]

                        st.markdown(f"### {kab}")

                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Suhu Saat Ini", f"{curr['temperature']}°C")
                        with col_b:
                            st.metric("Curah Hujan Saat Ini", f"{curr.get('precipitation', 0)} mm")
                        with col_c:
                            st.metric("Kecepatan Angin", f"{curr['windspeed']} km/h")
                        with col_d:
                            st.metric("Kelembaban", f"{hourly_humidity[0]}%")

                        df_rt = pd.DataFrame({
                            'Jam': hourly_time,
                            'Curah Hujan (mm)': hourly_precip,
                            'Suhu (°C)': hourly_temp,
                            'Kelembaban (%)': hourly_humidity
                        })

                        col1_rt, col2_rt = st.columns(2)
                        with col1_rt:
                            st.area_chart(df_rt.set_index('Jam')['Curah Hujan (mm)'], use_container_width=True)
                            st.caption("Prediksi Curah Hujan 12 Jam ke Depan")
                        with col2_rt:
                            st.line_chart(df_rt.set_index('Jam')[['Suhu (°C)', 'Kelembaban (%)']], use_container_width=True)
                            st.caption("Prediksi Suhu & Kelembaban 12 Jam ke Depan")

                        st.markdown("---")
                    else:
                        st.error(f"❌ Gagal mengambil data real-time untuk {kab}. Coba lagi nanti.")
        else:
            st.info("Pilih satu atau lebih kabupaten/kota di sidebar untuk melihat data cuaca real-time terkini.")

    # TAB 3: TABEL DATA
    with tab2:
        st.subheader("Tabel Prediksi")
        times = sorted(list(next(iter(pred_data.values())).keys()))
        df_pred = pd.DataFrame.from_dict(pred_data, orient='index')
        df_pred.columns = [f"T{t} menit" for t in times]
        df_pred = df_pred.map(lambda x: 'Tidak Hujan' if x == 0 else 'Ringan' if x == 1 else 'Lebat' if x == 2 else 'N/A')
        st.dataframe(df_pred.style.background_gradient(cmap='YlOrRd', axis=None))
        
        if any(t >= 0 for t in times):
            df_act = pd.DataFrame.from_dict(act_data, orient='index')
            df_act.columns = [f"T{t} menit" for t in times]
            df_act = df_act.map(lambda x: 'Tidak Hujan' if x == 0 else 'Ringan' if x == 1 else 'Lebat' if x == 2 else 'N/A' if x == -1 else x)
            st.subheader("Tabel Data Aktual")
            st.dataframe(df_act.style.background_gradient(cmap='YlGnBu', axis=None))

    # TAB 4: GRAFIK TREN
    with tab3:
        st.subheader("Grafik Tren Curah Hujan")
        if st.session_state.selected_kabs:
            for kab in st.session_state.selected_kabs:
                if kab in st.session_state.raw_rain_pred:
                    times = sorted(st.session_state.raw_rain_pred[kab].keys())
                    df_chart = pd.DataFrame({
                        'Waktu': times,
                        'Prediksi (mm)': [st.session_state.raw_rain_pred[kab].get(t, 0) for t in times],
                        'Aktual (mm)': [st.session_state.raw_rain_actual[kab].get(t, np.nan) for t in times]
                    }).melt('Waktu', var_name='Tipe', value_name='Curah Hujan (mm)')
                    chart = alt.Chart(df_chart).mark_line(point=True).encode(
                        x='Waktu:Q', y='Curah Hujan (mm):Q', color='Tipe:N', tooltip=['Waktu','Tipe','Curah Hujan (mm)']
                    ).properties(title=f"Tren Curah Hujan di {kab}", width=700, height=400).interactive()
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Pilih kabupaten di sidebar untuk melihat grafik.")

    # TAB 5: EXPORT
    with tab4:
        st.subheader("Export Data & Grafik")
        if pred_data:
            csv = pd.DataFrame.from_dict(pred_data).to_csv(index_label="Kabupaten")
            st.download_button("Download Prediksi (CSV)", csv, f"prediksi_{target_time.strftime('%Y%m%d_%H%M')}.csv", "text/csv")

        if act_data:
                df_act_export = pd.DataFrame.from_dict(act_data, orient='index')
                csv_act = df_act_export.to_csv(index_label='Kabupaten')
                st.download_button(
                    label="Download Data Aktual sebagai CSV",
                    data=csv_act,
                    file_name=f"aktual_{target_time.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

        if st.session_state.selected_kabs:
            for kab in st.session_state.selected_kabs:
                if kab in st.session_state.raw_rain_pred:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    times = sorted(st.session_state.raw_rain_pred[kab].keys())
                    ax.plot(times, [st.session_state.raw_rain_pred[kab].get(t, 0) for t in times], label="Prediksi", marker='o', color='#2563eb')
                    ax.plot(times, [st.session_state.raw_rain_actual[kab].get(t, np.nan) for t in times], label="Aktual", marker='x', color='#22c55e')
                    ax.set_title(f"Curah Hujan di {kab}")
                    ax.set_xlabel("Waktu Relatif (menit)")
                    ax.set_ylabel("Curah Hujan (mm)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    st.download_button(f"Download Grafik {kab}", buf.getvalue(), f"grafik_{kab}.png", "image/png")

        st.markdown("""
        <div style="background:#e0f2fe; padding:20px; border-radius:12px; margin-top:30px; border-left:6px solid #0ea5e9;">
            <p style="font-size:16px; color:#0c4a6e; margin:0;">
            <strong>Terima kasih banyak</strong> atas kunjungan dan kepercayaannya menggunakan Dashboard prediksi curah hujan Jawa Timur.<br><br>
            Semoga data yang Anda unduh bermanfaat. <strong>Tetap waspada</strong> terhadap potensi hujan lebat dan banjir, terutama saat beraktivitas di luar ruangan.<br><br>
            Salam hangat dan selamat beraktivitas!
            </p>
        </div>
        """, unsafe_allow_html=True)