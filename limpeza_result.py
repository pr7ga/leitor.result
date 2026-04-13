import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import hashlib
import time
import uuid

# -----------------------------
# Funções de leitura e processamento
# -----------------------------

def read_df(file):
    """Lê arquivo .Result e retorna DataFrame."""
    name = getattr(file, 'name', 'arquivo')

    try:
        raw = file.read()
    except Exception as e:
        st.warning(f"Não foi possível ler {name}: {e}")
        return pd.DataFrame()

    content = None
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            content = raw.decode(enc)
            break
        except Exception:
            pass

    if content is None:
        st.warning(f"Não foi possível decodificar {name}.")
        return pd.DataFrame()

    lines = content.splitlines()

    start_index = None
    for i, line in enumerate(lines):
        if "[TableValues]" in line:
            start_index = i + 1
            break

    if start_index is None:
        st.warning(f"[TableValues] não encontrado em {name}.")
        return pd.DataFrame()

    table_data = lines[start_index:]
    table_data = [line.strip() for line in table_data if line.strip()]
    table_raw = [line.split() for line in table_data]

    num_cols = None
    for row in table_raw:
        if len(row) >= 6:
            num_cols = len(row)
            break

    if num_cols is None:
        st.warning(f"Nenhuma linha válida encontrada em {name}.")
        return pd.DataFrame()

    if num_cols == 6:
        col_names = [
            'Frequency',
            'Average-dBμV/m',
            'Height',
            'Polarization',
            'Azimuth',
            'Attenuation'
        ]
    elif num_cols == 7:
        col_names = [
            'Frequency',
            'MaxPeak-dBμV/m',
            'Average-dBμV/m',
            'Height',
            'Polarization',
            'Azimuth',
            'Attenuation'
        ]
    elif num_cols == 8:
        col_names = [
            'Frequency',
            'MaxPeak-dBμV/m',
            'Average-dBμV/m',
            'Height',
            'Polarization',
            'Azimuth',
            'Attenuation',
            'Comment'
        ]
    else:
        st.error(f"Número inesperado de colunas ({num_cols}) em {name}.")
        return pd.DataFrame()

    table = [row for row in table_raw if len(row) == num_cols]

    if not table:
        st.warning(f"Nenhuma linha válida em {name}.")
        return pd.DataFrame()

    df = pd.DataFrame(table, columns=col_names)

    for col in [
        'Frequency',
        'Average-dBμV/m',
        'MaxPeak-dBμV/m',
        'Height',
        'Azimuth',
        'Attenuation'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Average-dBμV/m' in df.columns and df['Average-dBμV/m'].notna().any():
        df['dBμV/m'] = df['Average-dBμV/m']
    elif 'MaxPeak-dBμV/m' in df.columns and df['MaxPeak-dBμV/m'].notna().any():
        df['dBμV/m'] = df['MaxPeak-dBμV/m']
    else:
        df['dBμV/m'] = np.nan

    return df


def filter_by_frequency(df, freq_mhz, tol=0.001):
    if 'Frequency' not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')

    return df[np.abs(df['Frequency'] - freq_mhz) <= tol]


# -----------------------------
# CORRIGIDO AQUI
# -----------------------------
def clean_and_convert(df):
    def strip_chars(x):
        s = str(x)
        for ch in ('[', ']', "'", '"'):
            s = s.replace(ch, '')
        return s

    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].map(strip_chars)

    for col in df_cleaned.columns:
        try:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col])
        except Exception:
            pass

    return df_cleaned


def normalize_clwr(df):
    df = df.copy()

    if 'dBμV/m' in df.columns and not df['dBμV/m'].isna().all():
        max_val = df['dBμV/m'].max()
        df['Normalized-values'] = df['dBμV/m'] - max_val
    else:
        df['Normalized-values'] = np.nan

    return df


def rotate_azimuth(df, offset_degrees):
    df = df.copy()

    if 'Azimuth' in df.columns:
        df['Azimuth'] = (df['Azimuth'] - offset_degrees) % 360

    return df


def convert_to_dBm(df, antenna_gain):
    df = df.copy()

    if 'dBμV/m' in df.columns:
        df['Power-dBm'] = df['dBμV/m'] - 115.8 + antenna_gain
    else:
        df['Power-dBm'] = np.nan

    return df


def plot_polar(
    df,
    show_beamwidth=True,
    antenna_name="Antena XYZ",
    subtitle="",
    min_db=-50,
    title_fontsize=14,
    base_fontsize=10,
    font_family='sans-serif'
):
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': base_fontsize,
        'axes.titlesize': title_fontsize
    })

    if 'Azimuth' not in df.columns or df['Azimuth'].isna().all():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Sem dados')
        return fig

    df = df.copy()
    df['Azimuth'] = df['Azimuth'].round(3)

    grouped = df.groupby("Azimuth", as_index=False)["Normalized-values"].mean()
    grouped = grouped.sort_values("Azimuth")

    angles_deg = grouped["Azimuth"].values
    values_db = grouped["Normalized-values"].values

    if len(angles_deg) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Sem dados')
        return fig

    if angles_deg[0] != 0 or angles_deg[-1] != 360:
        angles_deg = np.append(angles_deg, angles_deg[0] + 360)
        values_db = np.append(values_db, values_db[0])

    interp_az = np.linspace(0, 360, 3600)
    interp_db = np.interp(interp_az, angles_deg, values_db)

    angles_rad = np.deg2rad(interp_az)

    fig, ax = plt.subplots(
        subplot_kw={'projection': 'polar'},
        figsize=(6, 6)
    )

    ax.plot(angles_rad, interp_db, linewidth=2)
    ax.fill(angles_rad, interp_db, alpha=0.3)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    fig.suptitle(antenna_name, fontsize=title_fontsize, y=1.05)

    if subtitle:
        fig.text(0.5, 0.98, subtitle, ha='center', color='gray')

    ax.set_ylim([min_db, 0])

    return fig


# -----------------------------
# UI
# -----------------------------

st.set_page_config("Leitor .Result", layout="centered")
st.title("📊 Leitor de Arquivos .Result")

if 'files' not in st.session_state:
    st.session_state['files'] = {}

if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0


with st.expander("Configurações", expanded=True):

    antenna_name = st.text_input("Nome da Antena", "Antena XYZ")
    antenna_subtitle = st.text_input("Subtítulo", "")
    antenna_gain = st.number_input("Ganho (dBi)", value=0.0)
    freq_input = st.number_input("Frequência (MHz)", format="%.3f")
    azimuth_offset = st.number_input("Rotação (graus)", value=0.0)
    min_db = st.number_input("Mínimo dB", value=-50)
    show_beamwidth = st.checkbox("Mostrar largura de feixe", True)


if st.button("🗑️ Limpar arquivos"):
    st.session_state['files'] = {}
    st.session_state['uploader_key'] += 1
    st.rerun()


uploader_key = f"upload_{st.session_state['uploader_key']}"

uploads = st.file_uploader(
    "Arquivos .Result",
    type=["Result"],
    accept_multiple_files=True,
    key=uploader_key
)

if uploads:
    for f in uploads:
        raw = f.read()
        h = hashlib.sha256(raw).hexdigest()

        if h not in st.session_state['files']:
            st.session_state['files'][h] = {
                'name': f.name,
                'bytes': raw,
                'added_at': time.time()
            }


df_final = pd.DataFrame(
    columns=[
        'dBμV/m',
        'Polarization',
        'Azimuth',
        'Filename',
        'Power-dBm'
    ]
)

files_values = sorted(
    list(st.session_state['files'].values()),
    key=lambda x: x['added_at']
)

if files_values and freq_input:

    for item in files_values:

        buf = io.BytesIO(item['bytes'])
        buf.name = item['name']

        df = read_df(buf)

        if df.empty:
            continue

        df_filtered = filter_by_frequency(df, freq_input)

        if not df_filtered.empty:
            row = df_filtered.iloc[0]

            df_final.loc[len(df_final)] = [
                row.get('dBμV/m', np.nan),
                row.get('Polarization', ''),
                row.get('Azimuth', np.nan),
                item['name'],
                None
            ]


if not df_final.empty:

    df_final = clean_and_convert(df_final)
    df_final = rotate_azimuth(df_final, azimuth_offset)
    df_final = normalize_clwr(df_final)
    df_final = convert_to_dBm(df_final, antenna_gain)

    fig = plot_polar(
        df_final,
        show_beamwidth,
        antenna_name,
        antenna_subtitle,
        min_db
    )

    st.pyplot(fig)

    st.subheader("Tabela")
    st.dataframe(df_final)

    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=300)

    st.download_button(
        "📥 Baixar PNG",
        data=img.getvalue(),
        file_name="grafico.png",
        mime="image/png"
    )

    csv = df_final.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Baixar CSV",
        data=csv,
        file_name="dados.csv",
        mime="text/csv"
    )
