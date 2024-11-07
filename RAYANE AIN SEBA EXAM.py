from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, desc
from pyspark.sql.types import *

# Création de la session Spark
print("Initialisation de la session Spark...")
spark = SparkSession.builder \
    .appName("MovieProcessing") \
    .getOrCreate()

# Création du schéma
schema = StructType([
    StructField("Year", IntegerType(), True),
    StructField("Length", IntegerType(), True),
    StructField("Title", StringType(), True),
    StructField("Subject", StringType(), True),
    StructField("Actor", StringType(), True),
    StructField("Actress", StringType(), True),
    StructField("Director", StringType(), True),
    StructField("Popularity", IntegerType(), True),
    StructField("Awards", StringType(), True),
    StructField("*Image", StringType(), True)
])

# Lecture et nettoyage initial
print("\n1. Lecture et nettoyage du fichier CSV...")
df = spark.read.option("delimiter", ";").option("header", "true").schema(schema).csv("film.csv")
df_clean_pre = df.filter(df.Title != "STRING")   # Suppression de la ligne des types
df_clean = df_clean_pre.filter(df.Subject != "NULL")   # Suppression des genres inconnus (on aurait pu mettre "Unknow")

print("Nombre de lignes après nettoyage:", df_clean.count())

# Suppression de la colonne Image
df_no_image = df_clean.drop("*Image")
print("\nColonnes du DataFrame:", df_no_image.columns)

# Création de la colonne Credits
print("\n2. Création et affichage de la colonne Credits (3 exemples):")
df_with_credits = df_no_image.withColumn("Credits", 
    expr("concat(Title, ' : a ', Actor, ' and ', Actress, ' film''s, directed by ', Director)"))
df_with_credits.select("Credits").show(3, False)

# Premiers tris demandés
print("\n3. Films les plus anciens (5 premiers):")
df_with_credits.orderBy("Year").select("Year", "Title").show(5)

print("\n4. Films les plus populaires (5 premiers):")
df_with_credits.orderBy(desc("Popularity"), "Title").select("Title", "Popularity").show(5)

# 5. Films de plus de 2 heures (120 minutes)
print("\n5. Films de plus de 2 heures (5 premiers):")
long_movies = df_with_credits.filter(col("Length") > 120)
print(f"Nombre total de films de plus de 2 heures: {long_movies.count()}")
long_movies.select("Title", "Length").orderBy(desc("Length")).show(5)

# 6. Films par genre (2 genres au choix : Drama et Comedy)
print("\n6. Films par genre:")
print("\nFilms Drama (10 premiers):")
drama_movies = df_with_credits.filter(col("Subject") == "Drama")
drama_movies.select("Title", "Subject", "Year").show(10)

print("\nFilms Comedy (10 premiers):")
comedy_movies = df_with_credits.filter(col("Subject") == "Comedy")
comedy_movies.select("Title", "Subject", "Year").show(10)

# 7. Acteur ayant tourné le plus de films
print("\n7. Top 5 des acteurs ayant tourné le plus de films:")
actor_counts = df_with_credits.groupBy("Actor") \
    .count() \
    .orderBy(desc("count"))
actor_counts.show(5)

# 8. Directeur ayant remporté le plus de récompenses
print("\n8. Top 5 des directeurs ayant remporté le plus de récompenses:")
director_awards = df_with_credits.filter(col("Awards") == "Yes") \
    .groupBy("Director") \
    .count() \
    .orderBy(desc("count"))
director_awards.show(5)

# 9. Film le plus populaire ayant remporté un prix
print("\n9. Film le plus populaire ayant remporté un prix:")
popular_awarded = df_with_credits.filter(col("Awards") == "Yes") \
    .orderBy(desc("Popularity"))
popular_awarded.select("Title", "Popularity", "Awards").show(1)

# 10. Genres n'ayant obtenu aucune récompense
print("\n10. Genres n'ayant obtenu aucune récompense:")
genres_no_awards = df_with_credits.groupBy("Subject") \
    .agg(
        F.sum(F.when(F.col("Awards") == "Yes", 1).otherwise(0)).alias("awards_count"),
        F.count("*").alias("total_movies")
    ) \
    .filter(F.col("awards_count") == 0)
genres_no_awards.select("Subject", "total_movies").show()


## Partie 2
print("\n---- PARTIE 2 ----")
# 1. Classement des films par popularité pour chaque genre
print("\n1. Classement des films par popularité par genre:")
window_spec = Window.partitionBy("Subject").orderBy(desc("Popularity"))
ranked_movies = df_with_credits.withColumn("ranking", 
    dense_rank().over(window_spec))

# Affichons un exemple pour quelques genres
print("\nExemple pour le genre 'Action' (10 premiers):")
ranked_movies.filter(col("Subject") == "Action") \
    .select("Subject", "Title", "Popularity", "ranking") \
    .show(10)

print("\nExemple pour le genre 'Drama' (10 premiers):")
ranked_movies.filter(col("Subject") == "Drama") \
    .select("Subject", "Title", "Popularity", "ranking") \
    .show(10)

print("\n. Top 10 des films les plus populaires (tous genres confondus):")
global_ranking = df_with_credits \
    .withColumn("global_rank", dense_rank().over(Window.orderBy(desc("Popularity")))) \
    .select("Title", "Subject", "Popularity", "global_rank") \
    .filter(col("global_rank") <= 10) \
    .orderBy("global_rank")

global_ranking.show(10)

# 2. Nombre de films de chaque directeur
print("\n2. Nombre de films par directeur (top 5):")
director_counts = df_with_credits.filter(col("Director").isNotNull()) \
    .groupBy("Director") \
    .count() \
    .orderBy(desc("count"))
director_counts.show(5)

# 3. UDF pour convertir le titre en majuscules
print("\n3. Conversion des titres en majuscules (8 exemples):")
upper_udf = udf(lambda x: x.upper() if x else None, StringType())
df_upper = df_with_credits.withColumn("TITLE_UPPER", upper_udf(col("Title")))
df_upper.select("Title", "TITLE_UPPER").show(8)

print("\nTraitement terminé!")