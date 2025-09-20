import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

# --- Funções de leitura e processamento ---

# (mantém read_df, filter_by_frequency, etc. inalterados)
# ... (mesmo conteúdo das funções que você já tem acima)

# --- Inicializa session_state ---

def init_state():
    if 'files' not in st.session_state:
        st.session_state['files'] = []  # cada item: {'name': str, 'bytes': bytes}

init_state()

# --- Interface Streamlit ---

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

# --- Botões de upload / limpeza ---
colA, colB = st.columns([1, 3])
with colA:
    if st.button("🗑️ Limpar todos os arquivos"):
        st.session_state['files'] = []
        st.rerun()  # substitui experimental_rerun()

with colB:
    st.write("")  # somente para layout

# --- Upload: adiciona arquivos ao session_state['files'] sem perder os anteriores ---
with st.expander("🔍 Processamento dos Arquivos", expanded=True):
    new_uploads = st.file_uploader("Arquivos .Result:", type=["Result"], accept_multiple_files=True, key="uploader")

    if new_uploads:
        added = 0
        for f in new_uploads:
            if not any(x['name'] == f.name for x in st.session_state['files']):
                try:
                    st.session_state['files'].append({'name': f.name, 'bytes': f.read()})
                    added += 1
                except Exception as e:
                    st.error(f"Falha ao ler {f.name}: {e}")
        if added > 0:
            st.success(f"{added} arquivo(s) adicionados à fila.")

    # mostra lista de arquivos atualmente armazenados
    if st.session_state['files']:
        st.write("**Arquivos na fila:**")
        for i, item in enumerate(list(st.session_state['files'])):
            cols = st.columns([6, 1])
            cols[0].write(item['name'])
            if cols[1].button("X", key=f"rm_{i}"):
                st.session_state['files'].pop(i)
                st.rerun()

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

# --- Exibição do gráfico e tabela (fora do expander) ---
if not df_final.empty:
    df_final = clean_and_convert(df_final)
    df_final = rotate_azimuth(df_final, azimuth_offset)
    df_final = normalize_clwr(df_final)
    df_final = convert_to_dBm(df_final, antenna_gain)

    fig = plot_polar(df_final, show_beamwidth, antenna_name, min_db,
                     title_fontsize=title_fontsize, base_fontsize=base_fontsize, font_family=title_font)
    st.pyplot(fig)

    # --- Botões para download da imagem ---
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="📥 Baixar gráfico (PNG)",
        data=img_bytes.getvalue(),
        file_name=f"{antenna_name}.png",
        mime="image/png"
    )

    pdf_bytes = io.BytesIO()
    fig.savefig(pdf_bytes, format="pdf", bbox_inches="tight")
    st.download_button(
        label="📥 Baixar gráfico (PDF)",
        data=pdf_bytes.getvalue(),
        file_name=f"{antenna_name}.pdf",
        mime="application/pdf"
    )

    # --- Botão para download do CSV ---
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
