Este código funciona en Python 3.X, usa las librerias de Opencv, yolov4_tiny y numpy, para evitar problemas con el reconocimiento de
librerías debes crear un entorno de trabajo, aquí hay un código de ejemplo:

python -m venv nombre_del_entorno
nombre_del_entorno\Scripts\activate


También se puede usar otras versiones mas potentes de yolocv(como yolov3), dependiendo la potencia de tu dispositivo.

Pasando a la ejecución del código, hay un par de observaciones en el código respecto al ajuste de la ventana en la que se reproduce la cámara o el video y como ajustar la fuente de video, en este caso el código esta escrito para ser usado con una cámara, por lo que si se necesita reproducir un video o una foto, se le tiene que hacer un cambio en la linea 8 de main.py y tenerlos en el mismo directorio que el entorno virtual de trabajo.

Si se quiere profundizar mas en el tema e interpretación del código voy a adjuntar el GitHub en donde esta el código original.