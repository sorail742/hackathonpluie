# 🌧️ Alerte Pluie Guinée - Hackathon IndabaX Guinée 2026

Solution de machine learning de bout en bout pour la prédiction des précipitations à 24 heures en Guinée, développée pour le Hackathon IndabaX Guinée 2026.

---

## 📊 Tableau Récapitulatif des Expériences & Résultats

| N° | Expérience / Architecture | Schéma de Validation | CV Log Loss | Score Public Zindi | Soumission Produite |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Baseline Starter** (Naïve, pas de variables météo) | Validation Aléatoire | ~0.6000 | **0.665337652** | `submission_baseline.csv` |
| **1** | **Features Météo + Rolling 5-Fold** | Stratified 5-Fold | **0.48405** | ~0.7420 | `submission_advanced_lgb.csv` |
| **2** | **Spatio-Temporel Régularisé** (Villes 2024 inconnues) | 4-Fold Spatio-Temporel Purge 24h | **0.62292 ± 0.027** | ~0.7610 | `submission_spatio_temporal.csv` |
| **3** | **Ensemble Multi-Seed (3 Seeds) + Calibration** | Spatio-Temporel + Calib ($T=1.005$) | **0.62124** | **0.762224978** | `submission_final_ensemble.csv` |
| **4** | **Blend Multiclasse + Ordinal** (3 seuils de pluie) | 4-Fold Spatio-Temporel Purge 24h | **0.61881** | **0.765841572** | `submission_mc_ordinal_blend.csv` |
| **5** | **Ultimate 3-Way Blend** (LGB MC + LGB Ordinal + XGBoost, $T=1.02$) | 4-Fold Spatio-Temporel Purge 24h | **0.61790** | **En cours (Cible 0.782+)** | **`submission_ultimate_3way_blend.csv`** |

> **Analyse du Leaderboard & Décomposition par Classe** :
> - **Métrique Zindi** : $\text{Score} = 1 - \frac{\text{LogLoss}_{\text{multiclass}}}{4} \iff \text{LogLoss} = 4 \times (1 - \text{Score})$.
> - La classe **`faible`** (1 à 10 mm) représente la principale source d'erreur ($\text{NLL} \approx 0.338$ chez le leader), suivie de **`aucune`** ($\text{NLL} \approx 0.263$).
> - L'approche **Ordinale** (3 classificateurs binaires cumulatifs pour $P(\ge 1\text{mm}), P(\ge 10\text{mm}), P(\ge 30\text{mm})$ avec monotonicité) force les arbres à discriminer les frontières `aucune` vs `faible` et `faible` vs `moderee`.
> - **Arrêt optimal des arbres** : La validation croisée a montré que 170 à 220 arbres suffisent (au lieu de 450). Au-delà, les arbres sur-apprenaient le bruit 2022-2024.
> - **Calibration de Température ($T = 1.02$)** et clipping sécurisé $[10^{-4}, 1 - 10^{-4}]$ évitent toute pénalité logarithmique extrême lors de faux négatifs sur classes rares.

---

## 🛠️ Règles Anti-Fuite & Causalité Strictes

1. **Causalité Stricte** :
   - Tous les calculs glissants sont effectués avec `closed='left'` sur fenêtres temporelles réelles (`'6h'`, `'12h'`, `'24h'`, `'48h'`, `'72h'`).
   - Aucune observation future ou contemporaine n'est utilisée pour prévoir les 24h suivantes.
2. **Absence de Fuite Cible** :
   - Aucun décalage de variable cible (`target`).
   - Le fichier `Reference.csv` est **strictement banni et non référencé**.
3. **Absence d'Index Temporel Brut (Neutralisation de Dérive Adversariale)** :
   - Pas de variable "année", "jour de la semaine" ou index croissant. La saisonnalité est strictement modélisée par des composantes cycliques harmoniques ($\sin / \cos$ sur l'heure, le mois et le jour de l'année).
4. **Généralisation Spatiale (Villes Non Vues : Labé et Siguiri)** :
   - Les villes non vues au test (**Labé** et **Siguiri**) sont représentées par leurs caractéristiques géographiques et climatiques objectives : latitude, longitude, altitude, distance à la côte atlantique, région naturelle, indice orographique et pluie annuelle normale.
5. **Contexte Régional Causal** :
   - Agrégation des conditions météorologiques récentes au niveau de la région naturelle (Moyenne Guinée, Haute Guinée, Guinée Maritime, Guinée Forestière) permettant aux villes inédites d'hériter de la dynamique de leurs voisines.

---

## 🗂️ Structure du Projet

```
predire la pluie/
├── data/
│   ├── train.csv                      # 155 287 relevés (2022-2024, 18 villes)
│   ├── test.csv                       # 38 936 relevés (2025, 20 villes dont Labé et Siguiri)
│   ├── sample_submission.csv           # 1 820 IDs notés
│   └── VariableDefinitions.csv        # Dictionnaire officiel des variables
├── src/
│   ├── config.py                      # Constantes, chemins, alias de normalisation des 20 villes
│   ├── stations_metadata.py           # Métadonnées géographiques et orographiques des 20 villes
│   ├── data.py                        # Chargement, vérification de schéma et tri chronologique
│   ├── cleaning.py                    # Détection sentinelles (-999), conversion °F->°C, filtrage RH=0
│   ├── features.py                    # 83 variables thermodynamiques, vent, ITCZ et rolling causaux
│   ├── cv.py                          # Schémas de validation spatio-temporelle avec purge 24h
│   ├── models.py                      # LightGBM régularisé et wrappers multi-seed
│   └── calibration.py                 # Scaling de température, clipping [1e-4, 1-1e-4], métriques
├── scripts/
│   ├── train_ultimate_blend.py        # Pipeline de production : Triple Blend (LGB MC + Ordinal + XGB)
│   ├── train_blend.py                 # Double Blend LightGBM Multiclasse + Ordinal
│   └── ablation_study.py              # Étude d'ablation systématique sur les plis spatio-temporels
├── tests/
│   ├── test_cleaning.py               # Tests unitaires du nettoyage et des 18+2 villes
│   ├── test_leakage.py                # Test unitaire de non-fuite causale
│   └── test_submission.py             # Test d'intégrité de la soumission (1820 IDs, somme=1, sans NaN)
├── submissions/
│   ├── submission_ultimate_3way_blend.csv # 🏆 SOUMISSION FINALE OPTIMALE (LGB MC + Ordinal + XGBoost)
│   ├── submission_mc_ordinal_blend.csv    # Soumission Multiclasse + Ordinal (0.7658 sur Zindi)
│   ├── submission_final_ensemble.csv      # Soumission précédente (0.7622 sur Zindi)
│   └── submission_spatio_temporal.csv     # Soumission spatio-temporelle de base
├── notebook_reproductible.ipynb       # Notebook unique autonome pour reproduction du jury
├── requirements.txt                   # Dépendances épinglées
└── README.md
```

---

## 🚀 Reproduction Rapide

### 1. Lancer les tests unitaires
```powershell
.venv\Scripts\python.exe -m pytest -v tests/
```

### 2. Entraîner et Générer la Soumission Finale Ultime
```powershell
.venv\Scripts\python.exe scripts/train_ultimate_blend.py
```
Le fichier généré `submissions/submission_ultimate_3way_blend.csv` est prêt à être téléversé sur Zindi.
