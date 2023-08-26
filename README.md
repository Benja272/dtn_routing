# Tesina Benjamin Martinez Picech
## Derivacion de planes de enrutamiento para redes tolerantes a demora
Implementacion de algoritmo de enruteo para redes tolerantes a demora con probabilidad de fallo en cada contacto.  
Esta implementacion tiene la particularidad de configurar prioridades para multiobjetivos, se puede optimizar con  
un orden de prioridades dentro de los siguientes objetivos: Probabilidad de Exito en la entrega(1), Energia estimada de entrega(2)  
y Tiempo estimado de entrega(3), ordenando estos objetivos segun se desee se pueden obtener distintos planes de enrutamiento.  
## Algoritmo
El algoritmo se basa en una tecnica de programacion dinamica, dado que el problema de enrutamiento esta dado en un rango discreto  
de tiempo, como las decisiones relacionadas a los primeros instantes de tiempo dependen de las decisiones posteriores, se arma una  
matriz de enrutamiento de cada fuente a cada destino para cada instante de tiempo y se la completa de "atras" para adelante,  
es decir que se completan las ultimas decisiones primero.  
## Estructura
El parser de los planes de contacto precalculados se encuentra en el archivo contact_plan.py, luego dentro del archivo network se encuentra  
la implementacion de el algoritmo dervidar de las decisiones de cada nodo hacia cada nodo en el intervalo de tiempo requerido.  
Por ultimo en la carpeta use_cases se encuentran los casos de usos utilizados para el testeo del algoritmo.

## Uso
Ejecutando el archivo main con el comando `python3 main.py 123` estamos ejecutando el ejemplo simple_case.txt de un plan de contacto  
y calculando las decisiones optimas siguiendo las prioridades sdp > energy > delay. Es decir el numero 1 reprecenta el sdp, 2 la energia  
y 3 el tiempo estimado de entrega. Otro ejemplo seria priorizar la energia antes que el sdp, las prioridades en ese caso deberian  
configurarse como `213`



