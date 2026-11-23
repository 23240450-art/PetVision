
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="PetVision",
    page_icon="🐾",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    return YOLO("yolo11n.pt")

st.title("🐾 PetVision")
st.subheader("Detección de perros y gatos con YOLO11")

st.write(
    "Identifica perros y gatos mediante inteligencia artificial "
    "y visión artificial."
)

try:
    modelo = cargar_modelo()
    st.success("Modelo YOLO11 cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar YOLO11: {e}")
    st.stop()

def detectar(imagen):
    resultados = modelo.predict(
        imagen,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    resultado = resultados[0]
    imagen_anotada = resultado.plot()

    # YOLO devuelve BGR; Streamlit necesita RGB
    imagen_anotada = imagen_anotada[:, :, ::-1]

    detecciones = []

    for caja in resultado.boxes:
        clase = int(caja.cls[0])
        if clase not in [15, 16]:
            continue

        nombre = modelo.names[clase]
        confianza = float(caja.conf[0])

        detecciones.append({
            "Animal": nombre.capitalize(),
            "Confianza": f"{confianza * 100:.2f}%"
        })

    return imagen_anotada, detecciones

pestana1, pestana2 = st.tabs([
    "📁 Subir imágenes",
    "📸 Cámara"
])

with pestana1:
    archivos = st.file_uploader(
        "Selecciona fotografías",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if archivos:
        for archivo in archivos:
            imagen = Image.open(archivo).convert("RGB")

            st.markdown(f"### {archivo.name}")

            if st.button(
                f"Detectar animales en {archivo.name}",
                key=archivo.name
            ):
                resultado, detecciones = detectar(
                    np.array(imagen)
                )

                st.image(
                    resultado,
                    caption="Resultado YOLO11",
                    use_container_width=True
                )

                if detecciones:
                    st.dataframe(
                        detecciones,
                        use_container_width=True
                    )
                else:
                    st.warning(
                        "No se encontraron perros ni gatos."
                    )

with pestana2:
    foto = st.camera_input("Toma una fotografía")

    if foto:
        imagen = Image.open(foto).convert("RGB")

        resultado, detecciones = detectar(
            np.array(imagen)
        )

        st.image(
            resultado,
            caption="Detección de cámara",
            use_container_width=True
        )

        if detecciones:
            st.dataframe(
                detecciones,
                use_container_width=True
            )
        else:
            st.warning(
                "No se encontraron perros ni gatos."
            )

st.divider()
st.caption("PetVision | YOLO11 | Detección inteligente")
