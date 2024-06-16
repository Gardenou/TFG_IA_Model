Model base de CNN per classificacio d'audio. 

Forma part del TFG d'Intel·ligència Artificial de la UOC.

Muntat sobre un entorn virtual venv i executat en local en PyCharm.
Les carpetes amb els conjunts de dades desades en local al disc dur. 

A les carpetes /logs tenim els logs  de resultats en .csv més significatius amb el nombre d'èpoques i el retorn de la funció de pèrdua i l'accuracy per cada època.

A la carpeta /weights guardem els arxius dels pesos resultants dels entrenaments més significatius en format .pth. Es poden carregar en el model des de la classe run.

El conjunt ESC-50 es pot descarregar directament des de https://github.com/karolpiczak/ESC-50 .

El conjunt UrbanSound8K es pot descarregar (previ registre) des de la web: https://urbansounddataset.weebly.com/download-urbansound8k.html .

Jordi Garriga Muñoz (jgarrigamoon@uoc.edu)
Grau d'Enginyeria Informàtica UOC
