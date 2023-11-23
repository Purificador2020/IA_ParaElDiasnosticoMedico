#Inicialzamos flask
from flask import Flask, render_template, request, render_template_string
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
#from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
#obtenemos el objeto app
app = Flask(__name__)
#iniciamos un servidor y rutas
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/consulta")
def encuesta():
    return render_template("consulta.html")
@app.route("/creadores")
def creadores():
    return render_template("creadores.html")
@app.route("/resultado")
def resultado():
    return render_template("resultado.html")
#Ruta para guardar los datos del formulario
@app.route('/realizar_diagnostico', methods=['POST', 'GET'])
#----meodo-----#
def realizar_diagnostico():
    if request.method == 'POST':

        #Cargamos el dataset en la variable de nombre BD
        BD = pd.read_csv('Enfermedades.csv')
        #Procesamo los datos del dataset en binarios
        BD['Fever']=BD['Fever'].map({'Yes':1, 'No':0})
        BD['Cough']=BD['Cough'].map({'Yes':1, 'No':0})
        BD['Fatigue']=BD['Fatigue'].map({'Yes':1, 'No':0})
        BD['Difficulty Breathing']=BD['Difficulty Breathing'].map({'Yes':1, 'No':0})
        BD['Gender']=BD['Gender'].map({'Female':1, 'Male':0})
        BD['Blood Pressure']=BD['Blood Pressure'].map({'High':3, 'Low':1, 'Normal':2})
        BD['Cholesterol Level']=BD['Cholesterol Level'].map({'High':3, 'Low':1, 'Normal':2})
        BD['Outcome Variable']=BD['Outcome Variable'].map({'Positive':1, 'Negative':0})
        #definimos la variable predictorias y las variables a predecir
        Y = BD['Outcome Variable']
        X = BD[['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age','Gender', 'Blood Pressure', 'Cholesterol Level']]
        #Dividiremos nuestros datos en las tablas de pruebas (testing) y de entrenamiento
        #(training) donde asignaremos el 25% de los datos de la base de datos BD para el
        #entrenamiento y 75% para la predicción.
        X_train, X_test, Y_train, Y_text = train_test_split(X, Y, train_size=0.2,
        random_state=42)
        #X_train.info()
        #Ajuste de hiperparámetros para tener lo mejores valores 
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'max_depth':[3, 5, 7, 10],
            'min_samples_split':[2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_
        print(best_params)
        #Creamos el modelo de árbol de decisión
        arbol = DecisionTreeClassifier()
        arbol_evaluar = arbol.fit(X_train, Y_train)
        #Predicciòn en el conjunto de pruebas
        Y_pred = arbol.predict(X_test)
        #Precision del modelo
        accuracy = accuracy_score(Y_text, Y_pred)
        print("Precisión del modelo:", accuracy)
        #Matriz de confusión
        conf_matrix = confusion_matrix(Y_text, Y_pred)
        print("Matriz de confusión:")
        print(conf_matrix)
        #---------pendiete_fina-------------#
        # Creas una lista vacía para almacenar los datos del formulario
        datos_del_formulario = []
        # Obtienes los datos del formulario uno a uno y los agregas a la lista
        #datos_del_formulario = request.form.to_dict(flat=False)
        #print("Los datos del formulario son: ", datos_del_formulario)
       
        Age = int(request.form['Edad'])
        Gender = request.form['Género']
        Fever = request.form['Fiebre']
        Cough = request.form['Tos']
        Fatigue = request.form['Fatiga']
        DifficultyBreathing = request.form['Dif_res']
        BloodPressure = request.form['Pre_art']
        CholesterolLevel = request.form['Niv_Col']
        #mapeando los datos segun su equivalente
        datos_del_formulario =[Fever, Cough, Fatigue, DifficultyBreathing, Age, Gender, BloodPressure, CholesterolLevel]
        caracteristicas = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
        #creando el arreglo con Numpy
        nuevos_datos = np.array([datos_del_formulario])
        #Nos aseguramos que los datos sean bidimencionales
        nuevos_datos = nuevos_datos.reshape(1, -1)
        #Asignamos los nombres a las columnas del arrego
        nuevos_datos = pd.DataFrame(nuevos_datos, columns = caracteristicas)
        #Procesamo los datos capturado del fomulario para que sean compatible con el Árbol de decisiones
        nuevos_datos['Fever']=nuevos_datos['Fever'].map({'Yes':1, 'No':0})
        nuevos_datos['Cough']=nuevos_datos['Cough'].map({'Yes':1, 'No':0})
        nuevos_datos['Fatigue']=nuevos_datos['Fatigue'].map({'Yes':1, 'No':0})
        nuevos_datos['Difficulty Breathing']=nuevos_datos['Difficulty Breathing'].map({'Yes':1, 'No':0})
        nuevos_datos['Gender']=nuevos_datos['Gender'].map({'Female':1, 'Male':0})
        nuevos_datos['Blood Pressure']=nuevos_datos['Blood Pressure'].map({'High':3, 'Low':1, 'Normal':2})
        nuevos_datos['Cholesterol Level']=nuevos_datos['Cholesterol Level'].map({'High':3, 'Low':1, 'Normal':2})




        # Captura Nombre y Apellido del formulario
        nombre = request.form['Nombre']
        apellido = request.form['Apellido']

        
        



        

        result = arbol.predict(nuevos_datos)
        
        if result == 1:
            resultadoMostrar = 'Enfermo'
        else:
            resultadoMostrar = 'Sano'            
         
        return render_template('resultado.html', 
                               Nombre=nombre, #pasar nombre a resultado.html
                               Apellido=apellido, #pasar apellido a resultado.html
                               resultado = resultadoMostrar
                               )
        #Realizamos el diagnostico con lo datos ingresados
    #return realizar_diagnostico("consulta.html")
   
#consultamos si aun esta en uso
if __name__ == '__main__':
    app.run(debug=True)