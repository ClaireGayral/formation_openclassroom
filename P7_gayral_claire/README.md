## Projet 7 : Stage

Ce stage a été réalisé pour l'entreprise [UXVizer](https://www.uxvizer.com/), sous la direction de Camille Saumard. 
UXvizer est une solution qui évalue la qualité de l'expérience utilisateur sur tout support numérique. En particulier, elle s'intéresse aux problématiques d'accessibilité visuelle d'un site web, d'une application ou d'un logiciel. En particulier, certaines images d'écran sont extraites par l'outil afin d'analyser visuellement le rendu utilisateur du contenu du site. Ce format procure une universalité des supports (applications, logiciels, sites internets), mais il introduit cependant des contraintes techniques fortes. Ces images ont formé le support principal de mon travail.

L’objectif de ce stage fut de produire une méthode de détection de paragraphes dans une image afin d’en automatiser la vérification de la norme ISO 9241-3.

Après avoir fait un travail bibliographique de la question, j’ai exploré deux pistes distinctes, sur un ensemble d’apprentissage que j’ai construit. La première méthode s’appuie sur des méthodes de transfert d’apprentissage, à partir de réseaux de neurones adaptés. La deuxième approche est plus géométrique : sachant que l’entreprise utilise un réseau de neurones pour détecter les mots sur les images, et que ce réseau est particulièrement performant, il suffit d'élargir ces boı̂tes et de les fusionner lorsqu’elles se superposent, jusqu’à ce qu’elles atteignent la taille du ”paragraphe”. 
