import streamlit as st

def tira_basura():
    import streamlit as st
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_hub as hub
    from streamlit_extras.let_it_rain import rain


    class_names = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plastico', 'Basura'] #For model 85

    @st.cache_resource
    def create_model(path):
        model = keras.models.load_model(path,custom_objects={'KerasLayer':hub.KerasLayer})
        return model
    count = 0
    if count == 0:
        modelo = create_model("modelo_85.h5")
        count = 1


    st.title("Tira tu basura aquí:",)

    st.markdown(
        """
        Abre la camara y toma una foto a tu basura, nuestro algoritmo de clasificación
        de imagenes utilizando ``Tensorflow`` nos permite poder decidir que tipo de basura
        tienes.

        """
    )

    img_file_buffer = st.camera_input("Toma una foto",label_visibility="hidden")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_res = cv2.resize(cv2_img, (224,224))
        img_res = img_res/255.0
        imagenFinal = tf.expand_dims(img_res, axis=0)

        prediccion = modelo.predict(imagenFinal)
        prediccion = tf.squeeze(prediccion)

        if max(prediccion) > 0.70:

            maxIndex = tf.argmax(prediccion).numpy()

            btn = st.download_button(
                    label="Guardar",
                    data=bytes_data,
                    file_name="Basura.png",
                    mime="image/png"
                )
            
            st.title(f"Tu residuo fue clasificado como: :green[{class_names[maxIndex]}]")

            text = str()

            if maxIndex == 0:
                emoji = "📦"
                text = "El cartón, utilizado en el embalaje y en la fabricación de cajas y envases, puede tener impactos ambientales negativos. Su producción requiere la tala de árboles, lo que puede llevar a la deforestación y la pérdida de hábitats naturales. Además, el proceso de fabricación del cartón consume grandes cantidades de agua y energía, contribuyendo a la escasez de recursos y a la emisión de gases de efecto invernadero. Cuando el cartón se desecha incorrectamente, ya sea en vertederos o incineradoras, puede contribuir a la contaminación del suelo, del agua y del aire. Sin embargo, el cartón es un material altamente reciclable y su reciclaje adecuado puede reducir significativamente su impacto ambiental."
            if maxIndex == 1:
                emoji = "🍾"
                text = "A diferencia del plástico, el vidrio es un material que no genera contaminantes durante su producción. Sin embargo, su fabricación requiere grandes cantidades de energía y emite dióxido de carbono (CO2). Cuando el vidrio se desecha incorrectamente, puede convertirse en un problema ambiental. Los desechos de vidrio pueden tardar miles de años en descomponerse en la naturaleza y ocupan espacio en los vertederos. Si se queman en incineradoras, pueden liberar gases contaminantes al aire. Además, el vidrio roto puede representar un peligro para la vida silvestre y los seres humanos si no se maneja adecuadamente. Sin embargo, el vidrio es altamente reciclable y su reciclaje reduce significativamente su impacto ambiental."
            if maxIndex == 2:
                text = "El metal puede tener diversos impactos ambientales negativos a lo largo de su ciclo de vida. La extracción de minerales metálicos implica la destrucción de ecosistemas, deforestación y contaminación del suelo y el agua. El proceso de refinado y fabricación de metales requiere grandes cantidades de energía y emite gases de efecto invernadero, contribuyendo al cambio climático. Además, muchos metales contienen sustancias tóxicas, como el plomo o el mercurio, que pueden contaminar el suelo, el agua y la vida silvestre. La disposición inadecuada de los desechos de metal también puede generar contaminación del suelo y del agua. Sin embargo, el reciclaje del metal es fundamental para reducir la necesidad de extracción y minimizar su impacto ambiental, conservando recursos naturales y evitando la contaminación asociada a su producción."
                emoji = "⚙️"
            if maxIndex == 3:
                text = "La producción y el uso del papel pueden tener impactos ambientales negativos. La fabricación de papel requiere la tala de árboles, lo que puede llevar a la deforestación y la pérdida de biodiversidad. Además, el proceso de fabricación consume grandes cantidades de agua, energía y productos químicos, contribuyendo a la contaminación del agua y la emisión de gases de efecto invernadero. El desecho de papel inadecuado también es problemático, ya que la mayoría de los residuos de papel terminan en vertederos, donde se descomponen y liberan metano, un gas de efecto invernadero potente. Sin embargo, el papel es altamente reciclable y su reciclaje adecuado puede reducir significativamente su impacto ambiental, conservando recursos naturales y reduciendo la contaminación."
                emoji = "📄"
            if maxIndex == 4:
                text = "El plástico contamina el medio ambiente de varias formas. Su producción a partir de combustibles fósiles emite gases de efecto invernadero, contribuyendo al cambio climático. Además, el plástico es duradero y puede tardar cientos de años en descomponerse, lo que lleva a la acumulación de desechos en vertederos y océanos. Cuando el plástico se descompone en microplásticos, puede ser ingerido por animales marinos, causando daños en su salud y en toda la cadena alimentaria. La quema de plástico libera sustancias tóxicas en el aire. La gestión inadecuada de los residuos plásticos también puede causar contaminación del suelo y del agua, impactando negativamente en los ecosistemas y la vida humana."
                emoji = "🔫"
            if maxIndex == 5:
                text = "La basura en general tiene diversos impactos ambientales negativos. Cuando la basura se acumula en vertederos, produce la liberación de gases de efecto invernadero, como metano, que contribuyen al cambio climático. La gestión inadecuada de la basura puede contaminar el suelo y el agua, afectando los ecosistemas naturales y la vida marina. Además, la incineración de residuos produce emisiones tóxicas y contribuye a la contaminación del aire. La basura también representa una gran demanda de recursos naturales, como energía y agua, para su producción y eliminación. Para mitigar estos impactos, es crucial promover la reducción, reutilización, reciclaje y una gestión adecuada de la basura."
                emoji = "🗑️"

            st.markdown(text)

            rain(
                emoji=emoji,
                font_size=54,
                falling_speed=1,
                animation_length=1,
            )
        else:
            st.title(f"Lo siento:( no pudimos clasificar tu :green[residuo]")

page_names_to_funcs = {
    "Tira tu basura 🗑️": tira_basura,
}

demo_name = st.sidebar.selectbox("¿Qué quieres hacer?:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
