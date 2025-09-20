import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

# -----------------------------
# Funções de leitura e processamento
# -----------------------------

def read_df(file):
    """Lê um arquivo .Result a partir de um file-like (tem .read() e .name).
    Tenta várias decodificações e procura a seção [TableValues]. Retorna DataFrame.
    """
    name = getattr(file, 'name', 'arquivo')
    try:
        raw = file.read()
    except Exception as e:
        st.warning(f"Não foi possível ler {name}: {e}")
        return pd.DataFrame()

    # tenta decodificar em várias codificações comuns
    content = None
    for enc in ('utf-16', 'utf-8', 'latin-1'):
        try:
            content = raw.decode(enc)
            break
        except Exception:
            content = None

    if content is None:
        try:
            # fallback: representar bytes como string
            content = str(raw)
        except Exception:
            st.warning(f"Não foi possível decodificar {name} com encodings comuns.")
            return pd.DataFrame()

    lines = content.splitlines()

    start_index = None
    for i, line in enumerate(lines):
        if '[TableValues]' in line:
            start_index = i + 1
            break

    if start_index is None:
        st.warning(f"[TableValues] não encontrado em {name}.")
        return pd.DataFrame()

    table_data = lines[start_index:]
    table_data = [line.strip() for line in table_data if line.strip()]
    table_raw = [line.split() for line in table_data]

    # determina número de colunas (procura linha válida com pelo menos 6 colunas)
    num_cols = None
    for row in table_raw:
        if len(row) >= 6:
            num_cols = len(row)
            break
    if num_cols is None:
        st.warning(f"Nenhuma linha válida encontrada em {name}.")
        return pd.DataFrame()

    if num_cols == 6:
        col_names = ['Frequency', 'Average-dBμV/m', 'Height', 'Polarization', 'Azimuth', 'Attenuation']
    elif num_cols == 7:
        col_names = ['Frequency', 'MaxPeak-dBμV/m', 'Average-dBμV/m', 'Height', 'Polarization', 'Azimuth', 'Attenuation']
    elif num_cols == 8:
        col_names = ['Frequency', 'MaxPeak-dBμV/m', 'Average-dBμV/m', 'Height', 'Polarization', 'Azimuth', 'Attenuation', 'Comment']
    else:
        st.error(f"Número inesperado de colunas ({num_cols}) em {name}.")
        return pd.DataFrame()

    table = [row for row in table_raw if len(row) == num_cols]
    if not table:
        st.warning(f"Nenhuma linha com o número esperado de colunas ({num_cols}) em {name}.")
        return pd.DataFrame()

    df = pd.DataFrame(table, columns=col_names)

    # converte colunas numéricas conhecidas
    for col in ['Frequency', 'Average-dBμV/m', 'MaxPeak-dBμV/m', 'Height', 'Azimuth', 'Attenuation']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # define coluna dBμV/m preferindo Average, senão MaxPeak
    if 'Average-dBμV/m' in df.columns and df['Average-dBμV/m'].notna().any():
        df['dBμV/m'] = df['Average-dBμV/m']
    elif 'MaxPeak-dBμV/m' in df.columns and df['MaxPeak-dBμV/m'].notna().any():
        df['dBμV/m'] = df['MaxPeak-dBμV/m']
    else:
        df['dBμV/m'] = np.nan

    return df


def filter_by_frequency(df, freq_mhz, tol=0.001):
    df = df.copy()
    if 'Frequency' in df.columns:
        df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
        return df[np.abs(df['Frequency'] - freq_mhz) <= tol]
    else:
        return pd.DataFrame()


def clean_and_convert(df):
    # remove caracteres indesejados sem usar regex complexo
    def strip_chars(x):
        s = str(x)
        for ch in ('[', ']', "'", '"'):
            s = s.replace(ch, '')
        return s

    df_cleaned = df.applymap(strip_chars)
    for col in df_cleaned.columns:
        try:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col])
        except Exception:
            pass
    return df_cleaned


def normalize_clwr(df):
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
    if 'dBμV/m' in df.columns:
        df['Power-dBm'] = df['dBμV/m'] - 115.8 + antenna_gain
    else:
        df['Power-dBm'] = np.nan
    return df


def plot_polar(df, show_beamwidth=True, antenna_name="Antena XYZ", min_db=-50,
               title_fontsize=14, base_fontsize=10, font_family='sans-serif'):
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': base_fontsize,
        'axes.titlesize': title_fontsize,
        'axes.labelsize': base_fontsize,
        'xtick.labelsize': base_fontsize,
        'ytick.labelsize': base_fontsize
    })

    if 'Azimuth' not in df.columns or df['Azimuth'].isna().all():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Sem dados de azimute para plotagem', ha='center')
        return fig

    df['Azimuth'] = df['Azimuth'].round(3)
    df_grouped = df.groupby("Azimuth", as_index=False)["Normalized-values"].mean()
    df_grouped = df_grouped.sort_values(by="Azimuth")

    angles_deg = df_grouped["Azimuth"].values
    values_db = df_grouped["Normalized-values"].values

    if len(angles_deg) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Sem dados para plotagem', ha='center')
        return fig

    if angles_deg[0] != 0 or angles_deg[-1] != 360:
        angles_deg = np.append(angles_deg, angles_deg[0] + 360)
        values_db = np.append(values_db, values_db[0])

    interp_az = np.linspace(0, 360, 3600)
    try:
        interp_db = np.interp(interp_az, angles_deg, values_db)
    except Exception:
        interp_db = np.zeros_like(interp_az)

    peak_idx = np.argmax(interp_db)
    peak_angle = interp_az[peak_idx]

    beamwidth = None
    angle1 = angle2 = None

    if show_beamwidth:
        above_3db = interp_db >= -3
        edges = np.where(np.diff(above_3db.astype(int)) != 0)[0]
        angles_3db = []
        for idx in edges:
            x1, x2 = interp_az[idx], interp_az[idx + 1]
            y1, y2 = interp_db[idx], interp_db[idx + 1]
            if (y1 >= -3 and y2 <= -3) or (y1 <= -3 and y2 >= -3):
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if slope != 0:
                    x_cross = x1 + (-3 - y1) / slope
                    angles_3db.append(x_cross)

        angles_3db = np.array(angles_3db)
        if angles_3db.size >= 2:
            diffs = np.abs((angles_3db - peak_angle + 180) % 360 - 180)
            idx_sort = np.argsort(diffs)
            angle1, angle2 = np.sort(angles_3db[idx_sort[:2]])
            beamwidth = (angle2 - angle1) % 360
            if beamwidth > 180:
                beamwidth = 360 - beamwidth

    angles_rad = np.deg2rad(interp_az)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    ax.plot(angles_rad, interp_db, linewidth=2)
    ax.fill(angles_rad, interp_db, alpha=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    fig.suptitle(f"{antenna_name}", fontsize=title_fontsize, y=1.02)
    ax.set_title("Diagrama de Radiação Normalizado", fontsize=base_fontsize, pad=30, color="gray")

    ax.set_rticks([-100, -90, -80, -70, -60, -50, -40, -30, -20, -10])
    ax.tick_params(axis='y', length=0, labelsize=0, colors='gray')
    for level in [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10]:
        ax.text(np.deg2rad(0), level, f"{level}", ha='left', fontsize=base_fontsize, color='brown', va='center')

    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{angle}°" for angle in np.arange(0, 360, 30)])

    if show_beamwidth and angle1 is not None and angle2 is not None:
        for angle in [angle1, angle2]:
            ax.plot([np.deg2rad(angle)] * 2, [min_db, 0], linestyle='--', color='red')
        fig.text(0.5, 0.95, f"Largura do feixe @ -3 dB: {beamwidth:.1f}°", ha='center', fontsize=base_fontsize, color='red')

    ax.set_ylim([min_db, 0])
    return fig


# -----------------------------
# Inicializa session_state
# -----------------------------

def init_state():
    if 'files' not in st.session_state:
        st.session_state['files'] = []  # cada item: {'name': str, 'bytes': bytes}


init_state()


# -----------------------------
# Interface Streamlit
# -----------------------------

st.set_page_config("Leitor .Result", layout="centered")
st.title("📊 Leitor de Arquivos .result (Software EMC32)")
st.markdown("Faça upload de **múltiplos arquivos .Result** e informe a **frequência alvo em MHz**.")

# Configurações de entrada
with st.expander("📥 Configurações de Entrada"):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        antenna_name = st.text_input("Nome da Antena", value="Antena XYZ")
    with col2:
        antenna_gain = st.number_input("Ganho da antena (dBi)", value=0.0, step=0.1)
    with col3:
        freq_input = st.number_input("Frequência de trabalho (MHz)", format="%.3f")
    with col4:
        tolerance = st.number_input("Tolerância (MHz)", min_value=0.0001, value=0.001, step=0.0001)

    col5, col6 = st.columns([1, 1])
    with col5:
        azimuth_offset = st.number_input("Ajuste de rotação (graus)", value=0.0, step=1.0)
    with col6:
        min_db = st.number_input("Valor mínimo de intensidade (dB)", min_value=-100, max_value=0, value=-50, step=1)

    show_beamwidth = st.checkbox("📐 Mostrar largura de feixe a -3 dB", value=True)
    col7, col8, col9 = st.columns([1, 1, 1])
    with col7:
        title_fontsize = st.number_input("Tamanho do título", value=14)
    with col8:
        base_fontsize = st.number_input("Tamanho base da fonte", value=10)
    with col9:
        title_font = st.selectbox("Fonte do gráfico", ["sans-serif", "serif", "monospace", "Arial", "Times New Roman"])

# -----------------------------
# Botões: Limpar e Upload
# -----------------------------
colA, colB = st.columns([1, 3])
with colA:
    if st.button("🗑️ Limpar todos os arquivos"):
        st.session_state['files'] = []
        # rerun para atualizar imediatamente
        try:
            st.rerun()
        except Exception:
            # fallback caso versão antiga do streamlit
            st.experimental_rerun()

with colB:
    st.write("")  # apenas para alinhamento

with st.expander("🔍 Processamento dos Arquivos", expanded=True):
    new_uploads = st.file_uploader("Arquivos .Result:", type=["Result"], accept_multiple_files=True, key="uploader")

    if new_uploads:
        added = 0
        for f in new_uploads:
            # evita duplicatas por nome; se quiser, troca por hash
            if not any(x['name'] == f.name for x in st.session_state['files']):
                try:
                    st.session_state['files'].append({'name': f.name, 'bytes': f.read()})
                    added += 1
                except Exception as e:
                    st.error(f"Falha ao ler {f.name}: {e}")
        if added > 0:
            st.success(f"{added} arquivo(s) adicionados à fila.")

    # --- Processamento dos arquivos armazenados em session_state ---
    df_final = pd.DataFrame(columns=['dBμV/m', 'Polarization', 'Azimuth', 'Filename', 'Power-dBm'])

    if st.session_state['files'] and freq_input:
        st.info(f"Buscando valores próximos de {freq_input:.3f} MHz (±{tolerance:.4f} MHz)")
        st.success(f"{len(st.session_state['files'])} arquivo(s) na fila.")

        for item in st.session_state['files']:
            try:
                buf = io.BytesIO(item['bytes'])
                buf.name = item['name']
                df = read_df(buf)
            except Exception as e:
                st.error(f"Erro ao processar {item['name']}: {e}")
                continue

            if df.empty:
                continue

            df_filtered = filter_by_frequency(df, freq_input, tol=tolerance)
            if not df_filtered.empty:
                row = df_filtered.iloc[0]
                df_final.loc[len(df_final)] = [
                    row.get('dBμV/m', np.nan),
                    row.get('Polarization', ''),
                    row.get('Azimuth', np.nan),
                    item['name'],
                    None
                ]

# -----------------------------
# Exibição do gráfico, downloads e tabela
# -----------------------------
if not df_final.empty:
    df_final = clean_and_convert(df_final)
    df_final = rotate_azimuth(df_final, azimuth_offset)
    df_final = normalize_clwr(df_final)
    df_final = convert_to_dBm(df_final, antenna_gain)

    fig = plot_polar(df_final, show_beamwidth, antenna_name, min_db,
                     title_fontsize=title_fontsize, base_fontsize=base_fontsize, font_family=title_font)
    st.pyplot(fig)

    # PNG
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="📥 Baixar gráfico (PNG)",
        data=img_bytes.getvalue(),
        file_name=f"{antenna_name}.png",
        mime="image/png"
    )

    # PDF
    pdf_bytes = io.BytesIO()
    fig.savefig(pdf_bytes, format="pdf", bbox_inches="tight")
    st.download_button(
        label="📥 Baixar gráfico (PDF)",
        data=pdf_bytes.getvalue(),
        file_name=f"{antenna_name}.pdf",
        mime="application/pdf"
    )

    # CSV
    csv_bytes = df_final.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar resultados (CSV)",
        data=csv_bytes,
        file_name=f"{antenna_name}.csv",
        mime="text/csv"
    )

    st.subheader("📄 Tabela de Resultados")
    st.dataframe(df_final[['Filename', 'Polarization', 'Azimuth', 'dBμV/m', 'Normalized-values', 'Power-dBm']])

else:
    if st.session_state['files']:
        st.warning("Nenhum dado correspondente à frequência informada foi encontrado.")
    else:
        st.info("📂 Nenhum arquivo na fila. Faça upload de arquivos .Result para processar.")
