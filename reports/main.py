
from turtle import color
import streamlit as st
import pandas as pd 
import altair as alt
import numpy as np
from PIL import Image
import streamlit.components.v1 as components
from numpy import squeeze
from joblib import load
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
client_credentials_manager = SpotifyClientCredentials(client_id="e3aa2b0ba2664fc384f7081a2121f66c",
                                                    client_secret="c2123cf1707f4a95a384f7a3a7b570c0")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

global saved_pipeline 
saved_pipeline = load('../data/raw/song_streams.model')


def predict_streams(uri):
	data=pd.DataFrame(sp.audio_features(uri),columns=['duration_ms','loudness'])
	streams=saved_pipeline.predict(data).squeeze()
	return streams

def main():
    st.title("Factores de éxito en canciones para ser tendencia en México 2021")

    data=pd.read_parquet("C:../data/processed/Streamsfeatures.parquet")
    df=pd.DataFrame(data)

    st.subheader('Introducción')
    st.markdown(
        """
        Spotify es una empresa que ofrece servicios multimedia originaria de Suecia fundada en 2006.
        El producto ***principal*** es su página que es empleada para la reproducción de música vía streaming.
        Además, Spotify cuenta con una API, esta permite obtener características de las canciones como: danceability, 
        tempo, valence, energy, speechiness, entre otras.

        Naturalmente surge la siguiente pregunta:

        ***¿Será posible explicar el número de reproducciones de una canción por medio de estas variables?***.
        
        Para responder a esta se propuso un modelo de regresión lineal múltiple, motivado en explicar el número de reproducciones de una canción durante el año 2020 en México. No obstante, el procedimiento permite repetirlo en cualquier tiempo deseado.
        """
    )

    st.subheader('Extracción de datos')

    st.markdown(
        """
        Los datos utilizados provienen de dos fuentes distintas:
        La fuente charts.spotify.com contiene las canciones más escuchadas semanalmente en México,
        proporcionándonos como muestra los datos del 01/05/2020 hasta 01/10/2020 por lo tanto nuestra muestra
        estaba compuesta por más de 4000 datos.

        Por medio de un programa en Python permitió conocer las canciones que se mostraban
        repetidas disminuyendo nuestra muestra aproximadamente a 400 canciones.
        
        Los datos pueden ser cargados utilizando pandas con las líneas:
        ```python
        import pandas as pd
        data = pd.read_csv('../data/interim/datos_sum.csv')
        ```
        Al desplegar los datos deberías ver una tabla como la siguiente:
        """
    )
    data=pd.read_csv('../data/interim/datos_sum.csv')
    st.dataframe(data)
    st.markdown(
        """    
        Con el identificador de las canciones(uri) y la Api de Spotify adquirimos las características de estas junto con
        el programa spoti.py desarrollado por el equipo, así logrando juntar todo en un mismo archivo(figura 2.1).
        
        Los datos pueden ser cargados utilizando pandas con las líneas:
        ```python
        import pandas as pd
        data = pd.read_parquet('../data/processed/Streamsfeatures.parquet')
        ```
        Al desplegar los datos deberías ver una tabla como la siguiente:
        """
        )
    data=pd.read_parquet('C:../data/processed/Streamsfeatures.parquet')
    st.dataframe(data)

    st.subheader('Análisis exploratorio de datos')

    st.markdown(
        """
        Ya con los datos realize una visualizacion general de los datos por medio de la libreria sweetviz:
        ```python
        import sweetviz as sv
        import pandas as pd
        music=pd.read_parquet('../data/processed/Streamsfeatures.parquet')
        music=music.drop(columns=['uri'])
        my_report = sv.analyze(music)
        #generar reporte en notebook
        my_report.show_notebook(w=None, 
                        h=None, 
                        scale=None,
                        layout='widescreen',
                        filepath=None)
        ```
        Al desplegar el reporte deberías ver lo siguiente:
        """
    )
    HtmlFile = open("SWEETVIZ_REPORT.html", 'r')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600,scrolling=True)


    st.subheader('ANALISIS DE CORRELACION')
    st.markdown(
        """
        Para encontrar las variables adecuadas y relevantes en nuestro modelo obtenemos la correlación de estas
        por medio del siguiente programa  dando como resultado la siguiente figura.
        
        El mapa puede ser cargado utilizando:
        ```python
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        archivo = "../data/interim/Streamsfeatures.parquet"
        df = pd.read_parquet(archivo,columns=['danceability','energy','loudness','speechiness','liveness','valence','tempo','acousticness','duration_ms'])
        df=pd.DataFrame(df)

        upp_mat = np.triu(df.corr())
        fig=sns.heatmap(df.corr(), vmin = -1, vmax = +1, annot = True, cmap = 'coolwarm', mask = upp_mat)
        plt.show() 
        ```
        Al desplegar el mapa deberías ver lo siguiente:
        """
    )
    image = Image.open('../reports/figures/CorrelationMap.png')
    st.image(image, caption='Mapa de correlación')

    st.subheader('Analizando tendencias')
    st.markdown(
        """
        Realize visualizacion de los datos en altair para ver si existen cierta tendencia en los datos
        antes de implementar el modelo de regresion multiple con el siguiente codigo y similar para las otras graficas:.
        ```python
        import pandas as pd 
        import altair as alt
        import numpy as np
        data=pd.read_parquet("../data/processed/Streamsfeatures.parquet")

        df=pd.DataFrame(data,columns=['streams','danceability','energy','duration_ms','loudness','speechiness','instrumentalness','liveness','valence','tempo','acousticness'])

        alt.Chart(df).mark_circle(size=60).encode(
            x='danceability',
            y='streams',
            tooltip=['danceability','streams']
        ).interactive()
        ```
        """
    )

    c= alt.Chart(df).mark_circle(size=60).encode(
        x='danceability',
        y='streams',
        
        tooltip=[ 'danceability','streams']
    ).interactive()
    st.altair_chart(c,use_container_width=True)

    c= alt.Chart(df).mark_circle(size=60).encode(
        x='loudness',
        y='streams',
        
        tooltip=[ 'loudness','streams']
    ).interactive()
    st.altair_chart(c,use_container_width=True)

    c= alt.Chart(df).mark_circle(size=60).encode(
        x='duration_ms',
        y='streams',
        
        tooltip=[ 'duration_ms','streams']
    ).interactive()
    st.altair_chart(c,use_container_width=True)

    st.markdown(
        """
        Se observa que en algunas variables hay cierta tendencia 
        ya que a mayor reproducciones las caracteristicas tienden 
        a estar mas a la derecha, entonces procedí a proponer modelos 
        de regresión múltiple y me quede con el siguiente modelo:
        """
    )
    st.latex(r'''ln(Streams)=\beta_0+\beta_2\ ln(Duration-ms)+\beta_2\ (Loudness)+u ''')

    st.subheader('Modelo de Regresión Lineal Múltiple')
    st.subheader('Escalamiento de datos')
    st.markdown(
        """
        En este caso por propiedad de la regresión se pueden cambiar las unidades 
        de medición y por eso decidi aplicar el logaritmo a la columna 'streams'
        y 'duration_ms' ya que acorta las distancias y los cambios se ven porcentualmente.

        Antes de crear el pipeline y entrenar el modelo cambie la columna streams por logstreams con el siguiente codigo:
        ```python
        from sklearn.model_selection import train_test_split
        import pandas as pd 
        import numpy as np
        songs=pd.read_parquet("../data/processed/Streamsfeatures.parquet")
        songs['logstreams'] = np.log(songs['streams'])
        songs=songs.drop(columns=['streams'])
        #si no estas en notebook: print(songs)
        songs
        ```
        Al desplegar el codigo deberías ver los siguientes datos:
        """
    )
    data=pd.read_parquet('C:../data/processed/StreamsfeaturesModel.parquet')
    st.dataframe(data)

    st.markdown(
        """
        Ya en el pipeline se cambia la columna 'duration_ms' por medio del codigo:
        
        ```python
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.compose import ColumnTransformer
        import numpy as np
        transformer = FunctionTransformer(np.log)
        log_encoding = ColumnTransformer([
        ('log_encoding',transformer, ["duration_ms"])
        ]) 
        ```
        Despues solo se deja pasar la columna loudness
        ```python
        #Just pass loudness
        passthrough= ColumnTransformer ([
        ('pass','passthrough', ['loudness'])
        ]) 
        ```
        Ensamblando el pipeline 
        ```python
        from sklearn.pipeline import Pipeline
        from sklearn.pipeline import FeatureUnion
        #Ensambla todo el pipeline
        pipe = Pipeline([
        ('features',
            FeatureUnion([
                ('log_encoding', log_encoding),
                ('just_passs', passthrough)
            ])
            )
        ])
        ```
        Agregando la regresion lineal
        
        ```python
        from sklearn.linear_model import LinearRegression
        lr=LinearRegression()
        predicting_pipeline = Pipeline([
            ('feature_engineering', pipe),
            ('esimator', lr)
        ])
        ```
        Al desplegar el pipeline se debería ver ver así:
        """
    )
    image = Image.open('../reports/figures/pipeline.png')
    st.image(image, caption='Pipeline',use_column_width=True)  

    st.subheader('Grafica de datos Reales vs predichos')

    st.markdown(
        """
        Ya entrenado el modelo pasé a revisar el modelo por medio de la siguiente grafica:
        Visualizando la grafica de valores reales vs predichos. Lo mejor seria que se visualizara
        una linea recta sin embargo es normal ver los datos algo dispersos ya que el modelo no pretende ser perfecto.
        """
    )
    image = Image.open('../reports/figures/actvspred.png')
    st.image(image)
    st.subheader('Interpretación de los estimadores')

    st.markdown(
        """
        Teniendo el modelo entrenado se obtienen los estimadores con el siguiente codigo
        ```python
        print("coeficientes de la regresion:B1=",lr.coef_[0],"B2=",lr.coef_[1],"B0:",lr.intercept_)
        ```
        """
    )
    st.latex(r'''\ln{\left(Streams\right)}=8.96+0.6In\left(Duration_ms\right)+0.1\left(Loudness\right)+u ''')
    st.markdown(
        """
        Cuando duration_ms aumenta 1%, los streams (reproducciones) aumentan 60%, manteniendo la variable
        loudness constante.

        loudness (volumen) aumenta en 1 unidad, los streams (reproducciones) 
        aumentan 10%, manteniendo la duración constante.

        En este caso el intercepto no tiene interpretación ya que es ilógico pensar que si la canción no 
        tiene volumen o duración la canción vaya a tener reproducciones.

        Para la  R^2:
        ```python
        from sklearn.metrics import r2_score
        r2_score(train['logstreams'], train_pred)
        ```
        """
    )
    st.latex(r'''R^2\approx0.399''')

    st.markdown(
        """
        La R^2 significa que en nuestra muestra de canciones el log(duration_ms) y loudness, explican de manera aproximada 3.99% 
        de la variación en el promedio de reproducciones en el top 200 de México. Puede que esto no sea un porcentaje muy alto, 
        pero se debe tener en cuenta que hay muchos otros factores que afectan a las reproducciones.
        """
    )


    st.write('Prediccion con escalamiento a logaritmo:')
    st.write('La uri se obtiene de spotify')

    uri=st.text_input("Inserta la Uri, ejemplo: spotify:track:7EUvcSFkyVB73zrblhQmEL","Type Here")
    if st.button("Predict"):
        st.markdown(predict_streams(uri))
		

if __name__ == '__main__':
	main()