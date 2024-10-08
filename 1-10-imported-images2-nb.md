---
jupytext:
  cell_metadata_json: true
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++

# TP images (2/2)

merci à Wikipedia et à stackoverflow

**le but de ce TP n'est pas d'apprendre le traitement d'image  
on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe ça se voit)**

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
import random as rd
```

+++ {"tags": ["framed_cell"]}

````{admonition} → **notions intervenant dans ce TP**

* sur les tableaux `numpy.ndarray`
  * `reshape()`, masques booléens, *ufunc*, agrégation, opérations linéaires
  * pour l'exercice `patchwork`:  
    on peut le traiter sans, mais l'exercice se prête bien à l'utilisation d'une [indexation d'un tableau par un tableau - voyez par exemple ceci](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
  * pour l'exercice `sepia`:  
    ici aussi on peut le faire "naivement" mais l'utilisation de `np.dot()` peut rendre le code beaucoup plus court
* pour la lecture, l'écriture et l'affichage d'images
  * utilisez `plt.imread()`, `plt.imshow()`
  * utilisez `plt.show()` entre deux `plt.imshow()` si vous affichez plusieurs images dans une même cellule

  ```{admonition} **note à propos de l'affichage**
  :class: seealso dropdown admonition-small

  * nous utilisons les fonctions d'affichage d'images de `pyplot` par souci de simplicité
  * nous ne signifions pas là du tout que ce sont les meilleures!  
    par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
    alors que la fonction `save` de `PIL` le permet
  * vous êtes libres d'utiliser une autre librairie comme `opencv`  
    si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte...
  ```
````

+++

## Création d'un patchwork

+++

1. Le fichier `data/rgb-codes.txt` contient une table de couleurs:
```
AliceBlue 240 248 255
AntiqueWhite 250 235 215
Aqua 0 255 255
.../...
YellowGreen 154 205 50
```
Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.

```{code-cell} ipython3
filename='data/rgb-codes.txt'
dico=dict()
with open(filename,'r') as file:
    for line in file:
        col,*l=line.split()
        dico[col]=np.array([int(e) for e in l])
```

2. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
`'Red'`, `'Lime'`, `'Blue'`

```{code-cell} ipython3
print(dico['Red'],dico['Lime'],dico['Blue'])
```

3. Faites une fonction `patchwork` qui  

   * prend une liste de couleurs et la structure donnant le code des couleurs RGB
   * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
   * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
   si besoin de compléter l'image mettez du blanc

+++

````{admonition} indices
:class: dropdown
  
* sont potentiellement utiles pour cet exo:
  * la fonction `np.indices()`
  * [l'indexation d'un tableau par un tableau](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
* aussi, ça peut être habile de couper le problème en deux, et de commencer par écrire une fonction `rectangle_size(n)` qui vous donne la taille du patchwork en fonction du nombre de couleurs  
  ```{admonition} et pour calculer la taille au plus juste
  :class: tip dropdown

  en version un peu brute, on pourrait utiliser juste la racine carrée;
  par exemple avec 5 couleurs créer un carré 3x3 - mais 3x2 c'est quand même mieux !

  voici pour vous aider à calculer le rectangle qui contient n couleurs

  n | rect | n | rect | n | rect | n | rect |
  -|-|-|-|-|-|-|-|
  1 | 1x1 | 5 | 2x3 | 9 | 3x3 | 14 | 4x4 |
  2 | 1x2 | 6 | 2x3 | 10 | 3x4 | 15 | 4x4 |
  3 | 2x2 | 7 | 3x3 | 11 | 3x4 | 16 | 4x4 |
  4 | 2x2 | 8 | 3x3 | 12 | 3x4 | 17 | 4x5 |
  ```
````

```{code-cell} ipython3
def func(l,dic):
    q,r=len(l)//10,len(l)%10
    l=l+['White']*(10-r)
    print(len(l)%10)
    im=np.zeros(shape=(1,len(l),3),dtype=np.uint8)
    for k in range (len(l)) : 
        im[0][k]=dic[l[k]]
    im=im.reshape(10,len(l)//10,3)
    return im
# j'ai choisi un tableau faisant 10 lignes et je complète avec du blanc
plt.imshow(func(['AliceBlue', 'AntiqueWhite', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'Black', 'BlanchedAlmond', 'Blue', 'BlueViolet', 'Brown', 'BurlyWood', 'CadetBlue', 'Chartreuse', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkBlue', 'DarkCyan', 'DarkGoldenrod', 'DarkGray', 'DarkGreen', 'DarkKhaki', 'DarkMagenta', 'DarkOliveGreen', 'DarkOrange', 'DarkOrchid', 'DarkRed', 'DarkSalmon', 'DarkSeaGreen', 'DarkSlateBlue', 'DarkSlateGray', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DimGray', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'Goldenrod', 'Gray', 'Green', 'GreenYellow', 'HoneyDew', 'HotPink', 'IndianRed', 'Indigo', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenrodYellow', 'LightGray', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'Maroon', 'MediumAquamarine', 'MediumBlue', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MidnightBlue', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'Navy', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenrod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'RebeccaPurple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Salmon', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sepia', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'Snow', 'SpringGreen', 'SteelBlue', 'Tan', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen','AliceBlue', 'AntiqueWhite', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'Black', 'BlanchedAlmond', 'Blue', 'BlueViolet', 'Brown', 'BurlyWood', 'CadetBlue', 'Chartreuse', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkBlue', 'DarkCyan', 'DarkGoldenrod', 'DarkGray', 'DarkGreen', 'DarkKhaki', 'DarkMagenta', 'DarkOliveGreen', 'DarkOrange', 'DarkOrchid', 'DarkRed', 'DarkSalmon', 'DarkSeaGreen', 'DarkSlateBlue', 'DarkSlateGray', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DimGray', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'Goldenrod', 'Gray', 'Green', 'GreenYellow', 'HoneyDew', 'HotPink', 'IndianRed', 'Indigo', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenrodYellow', 'LightGray', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'Maroon', 'MediumAquamarine', 'MediumBlue', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MidnightBlue', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'Navy', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenrod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'RebeccaPurple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Salmon', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sepia', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'Snow', 'SpringGreen', 'SteelBlue', 'Tan', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen','AliceBlue', 'AntiqueWhite', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'Black', 'BlanchedAlmond', 'Blue', 'BlueViolet', 'Brown', 'BurlyWood', 'CadetBlue', 'Chartreuse', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkBlue', 'DarkCyan', 'DarkGoldenrod', 'DarkGray', 'DarkGreen', 'DarkKhaki', 'DarkMagenta', 'DarkOliveGreen', 'DarkOrange', 'DarkOrchid', 'DarkRed', 'DarkSalmon', 'DarkSeaGreen', 'DarkSlateBlue', 'DarkSlateGray', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DimGray', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'Goldenrod', 'Gray', 'Green', 'GreenYellow', 'HoneyDew', 'HotPink', 'IndianRed', 'Indigo', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenrodYellow', 'LightGray', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'Maroon', 'MediumAquamarine', 'MediumBlue', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MidnightBlue', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'Navy', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenrod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'RebeccaPurple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Salmon', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sepia', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'Snow', 'SpringGreen', 'SteelBlue', 'Tan', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'],dico))
        
        
        
```

4. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.

```{code-cell} ipython3
L=[]
couleur=list(dico.keys())

for _ in range (100):
    ind=rd.randint(0,len(couleur)-1)
    L.append(couleur[ind])
plt.imshow(func(L,dico))
```

5. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
même chose pour des jaunes

```{code-cell} ipython3
blanc=[]
for coul in dico.keys():
    if 'White' in coul : 
        blanc.append(coul)
plt.imshow(func(blanc,dico))

jaune=[]
for coul in dico.keys():
    if 'Yellow' in coul : 
        jaune.append(coul)
plt.imshow(func(jaune,dico))
```

6. Appliquez la fonction à toutes les couleurs du fichier  
et sauver ce patchwork dans le fichier `patchwork.png` avec `plt.imsave`

```{code-cell} ipython3
l=list(dico.keys()) 
patch=func(l,dico)
#plt.imshow(patch)
plt.imsave('patchwork.png',patch)
#help(plt.imsave)
```

7. Relisez et affichez votre fichier  
   attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

vous devriez obtenir quelque chose comme ceci

```{image} media/patchwork-all.jpg
:align: center
```

```{code-cell} ipython3
patchenregistre=plt.imread('patchwork.png')
plt.imshow(patchenregistre)
```

## Somme dans une image & overflow

+++

0. Lisez l'image `data/les-mines.jpg`

```{code-cell} ipython3
mines=plt.imread('data/les-mines.jpg')
```

1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
plus=mines[:,:,0]+mines[:,:,1]+mines[:,:,2]
plt.imshow(plus)
#print(mines,plus)
```

2. Regardez le type de cette image-somme, et son maximum; que remarquez-vous?  
   Affichez cette image-somme; comme elle ne contient qu'un canal il est habile de l'afficher en "niveaux de gris" (normalement le résultat n'est pas terrible ...)


   ```{admonition} niveaux de gris ?
   :class: dropdown tip

   cherchez sur google `pyplot imshow cmap gray`
   ```

```{code-cell} ipython3
print(type(plus))
print(np.max(plus))
#on reste dans l'intervalle [0,255]
plt.imshow(plus, cmap='gray', vmin=0, vmax=255)
```

3. Créez un nouveau tableau `numpy.ndarray` en sommant mais cette fois **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
tab=np.array(mines)
somme=np.sum(tab,2)
#print(tab[0],somme[0])
```

4. Comme dans le 2., regardez son maximum et son type, et affichez la

```{code-cell} ipython3
type(somme)
np.max(somme)
# on remarque que le maximum ne correspond plus à un code RGB puisque >255
plt.imshow(somme, cmap='gray', vmin=0, vmax=np.max(somme))
#on retrouve l'image en noir et blanc
```

5. Les deux images sont de qualité très différente, pourquoi cette différence ? Utilisez le help `np.sum?`

```{code-cell} ipython3
help(np.sum)
""" quand on somme avec l'operateur + l'ordinateur remet automatiquement la somme entre 0 et 255 en soustrayant à la somme un multiple de 256 alors qu'avec sum la somme reste tel quel"""
```

6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
(de la manière que vous préférez)

```{code-cell} ipython3
huitbit = somme.astype(np.uint8)

huitbit.dtype
```

7. Remplacez dans l'image en niveaux de gris,  
les valeurs >= à 127 par 255 et celles inférieures par 0  
Affichez l'image avec une carte des couleurs des niveaux de gris  
vous pouvez utilisez la fonction `numpy.where`

```{code-cell} ipython3
somme[somme>=127] = 255
somme[somme<127] = 0
plt.imshow(somme, cmap='gray', vmin=0, vmax=255)
```

8. avec la fonction `numpy.unique`  
regardez les valeurs différentes que vous avez dans votre image en noir et blanc

```{code-cell} ipython3
np.unique(somme)
```

## Image en sépia

+++

Pour passer en sépia les valeurs R, G et B d'un pixel  
(encodées ici sur un entier non-signé 8 bits)  

1. on transforme les valeurs `R`, `G` et `B` par la transformation  
`0.393 * R + 0.769 * G + 0.189 * B`  
`0.349 * R + 0.686 * G + 0.168 * B`  
`0.272 * R + 0.534 * G + 0.131 * B`  
(attention les calculs doivent se faire en flottants pas en uint8  
pour ne pas avoir, par exemple, 256 devenant 0)  
1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
1. naturellement l'image doit être ensuite remise dans un format correct  
(uint8 ou float entre 0 et 1)

+++

````{tip}
jetez un coup d'oeil à la fonction `np.dot` 
qui est si on veut une généralisation du produit matriciel

dont voici un exemple d'utilisation:
````

```{code-cell} ipython3
# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

print(f"en partant des dimensions {A.shape} et {B.shape}")
print(f"on obtient un résultat de dimension {C.shape}")
print(f"et le nombre de termes dans chaque `sum()` est {A.shape[-1]} == {B.shape[-2]}")
```

**Exercice**

+++

1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
la fonction `numpy.dot` peut être utilisée si besoin, voir l'exemple ci-dessus

```{code-cell} ipython3
def sepia(im):
    coef = np.array([0.393,0.349,0.272, 0.769, 0.686, 0.534,0.189,0.168, 0.131]).reshape(3,3)
    sep = np.dot(im,coef)
    nb=np.size(im)
    h,l,g=np.shape(im)
    sep.reshape(1,nb)
    sep[sep  > 255] = int(255) #ne fonctionne pas
    sep.reshape(h,l,g)
    sep=sep/255
    #print(np.shape(sep),np.shape(im),sep.max())
    return sep
plt.imshow(mines)
plt.imshow(sepia(mines))
```

2. Passez votre patchwork de couleurs en sépia  
Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso

```{code-cell} ipython3
#patchsepia=plt.imread('patchwork-all.jpg')
plt.imshow(sepia(patch))
```

3. Passez l'image `data/les-mines.jpg` en sépia

```{code-cell} ipython3
plt.imshow(sepia(mines))
```

## Exemple de qualité de compression

+++

1. Importez la librairie `Image`de `PIL` (pillow)  
(vous devez peut être installer PIL dans votre environnement)

```{code-cell} ipython3
from PIL import Image
```

2. Quelle est la taille du fichier `data/les-mines.jpg` sur disque ?

```{code-cell} ipython3
file = "data/les-mines.jpg"
```

```{code-cell} ipython3
#fonction pas trouvée
```

3. Lisez le fichier 'data/les-mines.jpg' avec `Image.open` et avec `plt.imread`

```{code-cell} ipython3
imagePIL=Image.open(file)
imageplt=plt.imread(file)

#Image.open ouvre l'image en elle même alors que plt.imread ouvre le ndarray qui code l'image
```

4. Vérifiez que les valeurs contenues dans les deux objets sont proches

```{code-cell} ipython3

```

5. Sauvez (toujours avec de nouveaux noms de fichiers)  
l'image lue par `imread` avec `plt.imsave`  
l'image lue par `Image.open` avec `save` et une `quality=100`  
(`save` s'applique à l'objet créé par `Image.open`)

```{code-cell} ipython3

```

```{code-cell} ipython3
plt.imsave('imageplt.png',imageplt)
save('imagePIL.png', imagePIL,quality=100)
```

6. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
Que constatez-vous ?

```{code-cell} ipython3
np.itemsize(imagePIL)
```

7. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence

```{code-cell} ipython3
plt.imshow(imagePIL)
```

```{code-cell} ipython3
plt.imshow(imageplt)
```

```{code-cell} ipython3

```
