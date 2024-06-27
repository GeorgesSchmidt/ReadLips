**Read Lips**
IA de détection de la sémantique d'une phrase par analyse du mouvement des lèvres. 

L'objectif de ce repo est, dans un premier temps, de récolter et d'analyser les données nécessaires à créer ce modèle. 

On analyse alors le mouvent des lèvres sur l'image par rapport au temps (déroulé du film) en utilisant des modules de vision par ordinateur (InsightFace) et le son par des modules de NLP (interprétation du son). 

![Lips Points](figure_frame_1.jpg)

**Data extraction**

Le module createData.py recoit un film en entrée et applique une détection de visage à l'aide d'Insight Face afin d'avoir les points du visage dont les lèvres. 
InsightFace prédit 60 points de visage, toujours classés dans le meême ordre. 

Les lèvres sont les points de la liste entre les numéros 48 et 68. 

Ce module sépare les images du son et crée, en sortie, un fichier .npy qui contient les coordonnées de ces points à chaque image soit une matrice de dimensions (nb_images, nb_points, 3), de type int ainsi qu'un fichier .wav qui contient le son. 

La détection du visage avec InsightFace prend un certain temps, c'est pourquoi on extrait donc les données de cette manière afin de ne plus avoir à relancer 
l'extraction de la donnée. 
Ces fichiers sont stockés dans le repéretoire 'Datas'. 

**Data cleaning Face**

Le module readLips.py récupère les datas afin d'y appliquer une évaluation du mouvement des lèvres durant le film. La matrice de points du fichier .npy a une dimension fixe dans sa largeur (60 points) mais une dimension variable dans sa hauteur (le nombre d'images du film). 

***1 data extraction x et y des coordonnées lèvres***

On extrait les points des lèvres (lips = pts[48:68]) et calcule le centre de la bouche center = (np.mean(x), np.mean(y)). 

On split dans un premier temps les coordonnées x et y afin des points afin d'obtenir 2 matrices mat_x et mat_y, représentant respectivement les variations en x et en y pendant le film (temps).  

Ces coordonnées sont visualisées avec Matplotlib 3D avec pour les axes :  
- en  X = la variable tau (2*np.pi) du module math. 
- en Y = le temps (numéro d'image). 
- en Z = la variation des coordonnées x ou y par rapport au centre de la bouche.

En dessinant des lignes continues entre les points des lèvres, on remarque que la bouche est réprésentée par une double boucle. 

La première boucle défini les points du bord supérieur de la lèvre, la deuxième défini les points du bord inférieur de la lèvre. 

La courbe en y par rapport à 2*np.pi (variation de l'angle) présente donc 4 pics (min max de chaque boucle). 

Mat_x et mat_y représente les variations des amplitudes durant le temps (nb d'images). 

En effet, le son sera caractérisé par des 'mots' (amplitudes fortes) et des 'silences' (amplitudes faibles). 

***2 data interpolation : correction des dimensions***

On applique à ces 2 matrices (mat_x et mat_y) une interpolation afin de créer une matrice de dimension constante en h comme en w. 
On peut ainsi choisir les dimensions de la data points lips par rapport au temps aussi bien par rapport à son nombre de points (w) qu'au nombre d'images (h). 

Le module qui traite le son extrait les mots du fichier wav, cette interpolation permet donc de corréler les phases de mouvement des lèvres par rapport au nombre de mots. 

L'interpolation se fait d'abord sur le nombre de points en passant de 20 points initialement à 100 points, cela permettra d'obtenir une courbe plus lisse en sortie. Cette première interpolation s'effectue avec la fonction np.interp. 
Pour l'interpolation en h, on utilise gridData qui est une méthode fournie par Scipy. On passe alors de 1280 images à nb_mots (ici 100 pour l'exemple). 

On peut visualiser sous MatplotLib 3D le résultat de cette interpolation. 

***3 data interpretation : 2D-DFT***

Les amplitude de mouvement des lèvres peuvent être représentée aisément avec une transformée de Fourier en 2 dimensions (2D-DFT) qui est très utilisée en traitement d'image. 

Comme il s'agit d'une transformée de fourier, elle est particul!èrement adaptée à la modélisation mathématique des amplitudes de mouvement. 

On travaille alors avec des nombres complexes et Python est particulièremet adapté à ce type de variable. 

La partie réelle représente la variaiton en x, la partie imaginaire la variation en y. 

En calculant les amplitudes de ces nombres, on affiche l'image spectrale ce qui permet d'observer les amplitudes fortes (au centre de l'image spectrale) et les amplitudes faibles (au bords de l'image spectrale). 

En appliquant un masque à cette image, on supprime les amplitudes faibles. 

En sortie, on obtient un pourtour des lèvres lissé donc plus propre shématiquement. Ce lissage est défini par l'ordre choisi pour la tranformée de Fourier, autrement dit le nombre de nombre complexes que l'on choisi de garder. 

On obtient alors en sortie une tranformée de Fourier représentant les amplitudes de mouvement des lèvres par rapport au temps mais cette fois, les dimensions de cette matrice sont fixes : il s'agit de l'ordre choisi pour la transformée

Ce qui est intéressant avec cette interpretation par DFT est que l'on peut définir une constante, quelque soit le film, qui soit satisfaisante pour représenter le mouvement des lèvres. 

Le fait de fixer le nombre de dimensions de la sortie permet aussi de préparer la data pour un apprentissage en deep-learning. 

**Data cleaning sound**

Le module sound.py permet d'analyser le fichier .wav. 

Il extrait les données du son et le représente par rapport au temps (durée du film) et l'affiche à l'écran. 

Il interprete ce son (speechRecoginition) afin de créer une string soit une chaine de caractère qui donne la sémantique de ce qui a été dit sur cette bande son. 

En splitant cette chaine de caractère par rapport aux espaces entre les mots, on défini le nombre de 'mots' et le nombre de silences ('esp'). 

En appliquant une interpolation par np.interp sur le graphique du son, on obient les phases de son par chacun des mots et des silences de la phrase prononcée. 

Enfin, une analyse par transformée de Fourier (FFT) permet de visualiser l'image spectrale du son. 

On visualise aini les sons forts (les mots) et les sons faibles (les silences). 

En supprimant les amplitudes faibles, on défini aussi un ordre constant quelque soit le fichier son. 














