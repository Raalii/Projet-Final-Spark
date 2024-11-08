# Analyse de données de films avec Spark

Ce projet analyse un ensemble de données de films en utilisant PySpark. Il fait partie d'un exercice d'évaluation pour Data's Madness, répondant aux besoins de l'entreprise movieMax.

## Prérequis

- Python 3.x
- PySpark
- Un fichier de données `film.csv` contenant les informations des films

## Structure des données

Le fichier CSV contient les colonnes suivantes :

- `Year` (INT) : Année du film
- `Length` (INT) : Durée en minutes
- `Title` (STRING) : Titre du film
- `Subject` (STRING) : Genre
- `Actor` (STRING) : Acteur principal
- `Actress` (STRING) : Actrice principale
- `Director` (STRING) : Réalisateur
- `Popularity` (INT) : Score de popularité
- `Awards` (STRING) : Présence de récompenses (Yes/No)

## Fonctionnalités

Le script permet de :

1. Trier les films par :
   - Ancienneté
   - Popularité
   - Durée (> 2h)
2. Analyser les films par genre
3. Identifier :
   - L'acteur avec le plus de films
   - Le directeur avec le plus de récompenses
   - Le film le plus populaire récompensé
4. Établir des classements :
   - Par genre
   - Global (Top 10)

## Utilisation

1. Placer le fichier `film.csv` dans le même répertoire que le script
2. Exécuter le script dans un environnement Jupyter Notebook
3. Les résultats s'affichent progressivement avec des descriptions claires

## Notes

- Le script nettoie automatiquement les données (suppression ligne des types et colonne Image)
- Les résultats sont limités pour une meilleure lisibilité
- Les classements utilisent la fonction `dense_rank` de Spark
