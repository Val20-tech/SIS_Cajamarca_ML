"""
09_dashboard.py — SIS Cajamarca: Predictor de Demanda Asistencial
Ejecutar: streamlit run notebooks/09_dashboard.py
"""

import os, pickle, concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Configuracion de pagina ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SIS Cajamarca — Demanda Asistencial",
    page_icon="hospital",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, 'modelo_rf.pkl')

# ── Carga del modelo (cacheado) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo Random Forest...")
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

artifact = load_artifact()

if artifact is None:
    st.error("No se encontro modelo_rf.pkl. Ejecuta primero: `python notebooks/08_modelo_random_forest.py`")
    st.stop()

model    = artifact['model']
encoders = artifact['encoders']
features = artifact['features']
cat_cols = artifact['cat_cols']
metrics  = artifact['metrics']

FEAT_LABELS = {
    'anio': 'Año', 'mes': 'Mes', 'provincia': 'Provincia',
    'ipress': 'IPRESS', 'nivel_eess': 'Nivel EESS',
    'cod_servicio': 'Servicio', 'sexo': 'Sexo', 'grupo_edad': 'Grupo Edad',
}

# ── SHAP Explainer (cacheado una vez por sesion) ─────────────────────────────
@st.cache_resource
def load_shap_explainer(_model):
    try:
        import shap
        return shap.TreeExplainer(_model), True
    except Exception:
        return None, False

_explainer, _shap_ok = load_shap_explainer(model)

def _fallback_contributions(row_vals, feats, encs, cat_feats, importances):
    """Aproximacion: importancia x valor normalizado en [-1, 1]."""
    refs = {'anio': (2023.5, 0.5), 'mes': (6.5, 5.5)}
    contribs = []
    for i, f in enumerate(feats):
        v = float(row_vals[i])
        if f in refs:
            ref, rng = refs[f]
        else:
            n = len(encs[f].classes_)
            ref, rng = (n - 1) / 2.0, max((n - 1) / 2.0, 1.0)
        contribs.append(float(importances[i]) * (v - ref) / rng)
    return np.array(contribs)

MES_NOMBRES = {
    1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril',
    5:'Mayo', 6:'Junio', 7:'Julio', 8:'Agosto',
    9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre',
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("SIS Cajamarca — Predictor de Demanda Asistencial")
st.caption(
    f"Modelo: **Random Forest Regressor** | "
    f"Datos: {metrics['n_train'] + metrics['n_test']:,} combinaciones IPRESS/servicio/mes | "
    f"Entrenado con {metrics['n_train']:,} filas | "
    f"Evaluado con {metrics['n_test']:,} filas"
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Metricas del Modelo",
    "Importancia de Variables",
    "Predictor Interactivo",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — METRICAS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Rendimiento del modelo en el conjunto de prueba (20%)")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label="R² (varianza explicada)",
        value=f"{metrics['r2']:.4f}",
        delta=f"{metrics['r2']*100:.1f}%",
    )
    c2.metric(
        label="MAE",
        value=f"{metrics['mae']:.2f}",
        help="Error Absoluto Medio — atenciones",
    )
    c3.metric(
        label="RMSE",
        value=f"{metrics['rmse']:.2f}",
        help="Raiz del Error Cuadratico Medio — atenciones",
    )
    c4.metric(
        label="Filas de prueba",
        value=f"{metrics['n_test']:,}",
    )

    st.divider()

    # Interpretacion
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### Interpretacion de metricas")
        st.markdown(f"""
| Metrica | Valor | Interpretacion |
|---|---|---|
| **R²** | `{metrics['r2']:.4f}` | El modelo explica el **{metrics['r2']*100:.1f}%** de la varianza en demanda |
| **MAE** | `{metrics['mae']:.2f}` atenciones | Error tipico por combinacion IPRESS/servicio/mes |
| **RMSE** | `{metrics['rmse']:.2f}` atenciones | Penaliza errores grandes (outliers con max=3,125) |
        """)

    with col_b:
        st.markdown("### Distribucion del target")
        stats_df = pd.DataFrame({
            "Estadistico": ["Minimo", "Maximo", "Media", "Mediana", "Filas totales"],
            "Valor": ["1", "3,125", "8.29", "4", "1,586,610"],
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    st.info(
        f"**Contexto:** Con una mediana de 4 atenciones por grupo, un MAE de "
        f"{metrics['mae']:.1f} representa un error relativo razonable. "
        f"El RMSE mayor refleja que algunos servicios concentrados (hospitales / especialidades) "
        f"tienen alta varianza y son mas dificiles de predecir."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMPORTANCIA DE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Contribucion de cada variable al modelo")

    imp_df = (
        pd.DataFrame({'Variable': features, 'Importancia': model.feature_importances_})
        .sort_values('Importancia', ascending=True)
        .reset_index(drop=True)
    )

    # Grafico de barras horizontal
    fig_imp = px.bar(
        imp_df,
        x='Importancia',
        y='Variable',
        orientation='h',
        color='Importancia',
        color_continuous_scale='Blues',
        text=imp_df['Importancia'].map(lambda v: f'{v:.2%}'),
        title='Importancia relativa de variables — Random Forest',
        labels={'Importancia': 'Importancia relativa', 'Variable': ''},
    )
    fig_imp.update_traces(textposition='outside')
    fig_imp.update_layout(
        height=420,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=60, t=50, b=20),
        xaxis=dict(tickformat='.0%'),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Tabla de importancias
    imp_tabla = imp_df.sort_values('Importancia', ascending=False).reset_index(drop=True)
    imp_tabla.index = imp_tabla.index + 1
    imp_tabla['Importancia (%)'] = imp_tabla['Importancia'].map(lambda v: f'{v*100:.2f}%')
    imp_tabla['Barra'] = imp_tabla['Importancia'].map(lambda v: '#' * int(v * 60))

    st.markdown("### Ranking de variables")
    st.dataframe(
        imp_tabla[['Variable', 'Importancia (%)', 'Barra']],
        use_container_width=True,
    )

    st.markdown("### Conclusiones")
    top3 = imp_tabla.head(3)
    for _, row in top3.iterrows():
        st.markdown(f"- **`{row['Variable']}`** explica el {row['Importancia (%)']}"
                    f" de las predicciones — es el factor mas determinante en su grupo.")
    st.markdown(
        "- `anio` tiene baja importancia (0.71%), lo que sugiere que el patron estacional "
        "es mas relevante que la tendencia interanual en el periodo 2023-2024."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTOR INTERACTIVO
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Estimar demanda para una combinacion especifica")
    st.markdown(
        "Selecciona las caracteristicas y obtendra la prediccion del modelo "
        "entrenado sobre datos reales SIS 2023-2024."
    )

    col1, col2 = st.columns(2)

    with col1:
        anio = st.selectbox(
            "Año",
            options=[2023, 2024, 2025, 2026],
            index=1,
        )
        mes = st.selectbox(
            "Mes",
            options=list(range(1, 13)),
            format_func=lambda m: f"{m:02d} — {MES_NOMBRES[m]}",
            index=0,
        )
        provincia = st.selectbox(
            "Provincia",
            options=sorted(encoders['provincia'].classes_.tolist()),
        )
        ipress = st.selectbox(
            "IPRESS (escribe para buscar)",
            options=sorted(encoders['ipress'].classes_.tolist()),
        )

    with col2:
        nivel_eess = st.selectbox(
            "Nivel EESS",
            options=sorted(encoders['nivel_eess'].classes_.tolist()),
        )
        cod_servicio = st.selectbox(
            "Codigo de servicio (escribe para buscar)",
            options=sorted(encoders['cod_servicio'].classes_.tolist()),
        )
        sexo = st.selectbox(
            "Sexo",
            options=sorted(encoders['sexo'].classes_.tolist()),
        )
        grupo_edad = st.selectbox(
            "Grupo de edad",
            options=sorted(encoders['grupo_edad'].classes_.tolist()),
        )

    st.divider()

    if st.button("Predecir demanda", type="primary", use_container_width=True):

        inputs = {
            'anio': anio, 'mes': mes,
            'provincia': provincia, 'ipress': ipress,
            'nivel_eess': nivel_eess, 'cod_servicio': cod_servicio,
            'sexo': sexo, 'grupo_edad': grupo_edad,
        }

        # Codificar con LabelEncoders
        row = []
        for feat in features:
            if feat in cat_cols:
                val_str = str(inputs[feat])
                if val_str in encoders[feat].classes_:
                    val = int(encoders[feat].transform([val_str])[0])
                else:
                    val = 0
            else:
                val = int(inputs[feat])
            row.append(val)

        X_new = np.array([row], dtype=np.float32)
        pred  = float(model.predict(X_new)[0])
        pred_int = max(0, round(pred))
        mae      = metrics['mae']

        # Resultados
        r1, r2, r3 = st.columns(3)
        r1.metric("Atenciones estimadas", f"{pred_int:,}")
        r2.metric("Valor exacto del modelo", f"{pred:.2f}")
        r3.metric(
            f"Intervalo de confianza (+/-MAE)",
            f"{max(0, pred_int - int(mae)):,} – {pred_int + int(mae):,}",
        )

        st.success(
            f"Para el servicio **{cod_servicio}** en **{ipress}** ({provincia}) | "
            f"{MES_NOMBRES[mes]} {anio} | Sexo: {sexo} | Grupo edad: {grupo_edad} | "
            f"Nivel: {nivel_eess}  →  **{pred_int} atenciones estimadas**"
        )

        # Mini gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_int,
            title={'text': "Atenciones estimadas"},
            gauge={
                'axis': {'range': [0, max(100, pred_int * 2)]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 10],   'color': "#e8f4fd"},
                    {'range': [10, 50],  'color': "#aed6f1"},
                    {'range': [50, 200], 'color': "#5dade2"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': pred_int,
                },
            },
            number={'suffix': " atenciones"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Explicabilidad local ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Explicabilidad de esta prediccion")

        use_shap  = False
        contribs  = None

        if _shap_ok and _explainer is not None:
            with st.spinner("Calculando contribuciones SHAP..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(_explainer.shap_values, X_new)
                    try:
                        _sv      = _fut.result(timeout=30)
                        contribs = _sv[0] if not isinstance(_sv, list) else _sv[0][0]
                        use_shap = True
                    except concurrent.futures.TimeoutError:
                        st.info("SHAP tardo mas de 30 s — usando aproximacion por importancia.")

        if contribs is None:
            contribs = _fallback_contributions(
                row, features, encoders, cat_cols, model.feature_importances_
            )

        labels = [FEAT_LABELS.get(f, f) for f in features]
        colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in contribs]
        method = "SHAP (TreeExplainer)" if use_shap else "Aproximacion (importancia × valor)"

        # Grafico de barras de contribucion
        fig_contrib = go.Figure(go.Bar(
            x=labels,
            y=contribs,
            marker_color=colors,
            text=[f'{c:+.4f}' for c in contribs],
            textposition='outside',
        ))
        fig_contrib.update_layout(
            title=f"Contribucion de cada variable a esta prediccion  ({method})",
            xaxis_title="Variable",
            yaxis_title="Contribucion",
            height=420,
            margin=dict(l=20, r=20, t=60, b=40),
            plot_bgcolor='white',
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Mapa de calor de contribucion
        fig_heat = go.Figure(go.Heatmap(
            z=[list(contribs)],
            x=labels,
            y=['Contribucion'],
            colorscale=[[0, '#e74c3c'], [0.5, 'white'], [1, '#2ecc71']],
            zmid=0,
            text=[[f'{c:+.4f}' for c in contribs]],
            texttemplate='%{text}',
            showscale=True,
        ))
        fig_heat.update_layout(
            title="Mapa de calor de contribucion",
            height=200,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption(
            f"**Como leer esto:** barras verdes = variables que aumentan la prediccion, "
            f"rojas = que la reducen. Metodo: {method}."
        )

    st.markdown(
        "> **Nota:** El modelo fue entrenado con datos SIS Cajamarca 2023-2024 "
        "(1.27M combinaciones). Para años futuros (2025-2026) la prediccion asume "
        "patrones similares a los historicos."
    )
