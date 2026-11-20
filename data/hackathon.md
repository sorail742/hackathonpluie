Pouvez-vous prédire les précipitations des prochaines 24 heures pour aider les agriculteurs à protéger leurs récoltes ?
Hackathon IndabaX Guinée 2026, organisé par IndabaX Guinée en collaboration avec l'Université de Labé et l'Université Mercure International de Guinée. Du 12 novembre 2026 à 15:00 au 14 novembre 2026 à 15:00 (GMT), en présentiel uniquement, à l'Université de Labé.

En Guinée, la grande majorité des agriculteurs, qu'ils cultivent du riz, du maïs, du fonio, de l'arachide, du café ou du cacao, dépendent directement de la pluie. Pourtant, peu d'entre eux disposent d'une prévision fiable pour les prochaines 24 heures, à l'échelle de leur ville.

Cette incertitude coûte cher. Un engrais épandu la veille d'une forte pluie est emporté, ce qui représente une perte importante pour un petit producteur. Un semis fait juste avant une averse violente est perdu. Le café, le cacao ou le riz qui sèchent dehors pourrissent lorsqu'ils sont surpris par la pluie. Dans les villes comme Conakry, Kankan ou Siguiri, les fortes pluies inondent des quartiers et coupent des routes et des pistes.

Le pays compte des stations et des flux de données météo, mais ces données sont imparfaites : capteurs en panne, valeurs manquantes, erreurs d'unité, relevés irréguliers. Les transformer en alertes utiles demande de vraies compétences en science des données.

Objectif du challenge

À partir des relevés d'une station météo à un instant t (température, humidité, pression, vent, nuages, pluie récente) et de son historique, les participants doivent prédire la pluie cumulée sur les 24 heures suivantes, en quatre classes :

- aucune (moins de 1 mm) : on peut épandre, sécher et récolter sans risque ;

- faible (1 à 10 mm) : bon moment pour semer ;

- moderee (10 à 30 mm) : prudence pour l'engrais et le séchage ;

- forte (plus de 30 mm) : alerte, risque d'inondation.

Les participants soumettent une probabilité pour chaque classe. Le classement utilise la multi-class log loss, qui récompense les modèles bien calibrés : un système d'alerte a besoin de probabilités fiables, pas seulement d'un oui ou d'un non.

Les données couvrent 20 villes guinéennes. Elles sont accompagnées de fichiers additionnels : caractéristiques des stations, indicateurs régionaux (harmattan, humidité du sol, végétation) et indicateurs climatiques nationaux (Front Inter-Tropical, oscillation de Madden-Julian). Le test porte sur une année non vue et sur deux villes absentes de l'entraînement. Il faut donc des modèles qui généralisent dans le temps et dans l'espace.

Pertinence de la solution

La solution gagnante servira de moteur à un système d'alerte agricole envoyé la veille par SMS ou WhatsApp, par exemple : « Kindia, demain : risque de pluie forte. Reportez l'épandage d'engrais. » Un tel outil aide les agriculteurs à réduire leurs pertes d'engrais, de semences et de récoltes, et aide les autorités locales à anticiper les inondations.

Les données du challenge sont simulées à partir d'un vrai extrait du flux d'ingestion météo guinéen et de la climatologie du pays (saisons, harmattan, mousson, relief du Fouta Djallon, lignes de grains). Leurs colonnes reprennent celles du flux réel. La méthode gagnante pourra donc être réentraînée sur les données réelles accumulées après le hackathon.

Enfin, le challenge forme une nouvelle génération de data scientists guinéens à des compétences concrètes : nettoyage de données réelles, séries temporelles, validation rigoureuse et calibration des probabilités, au service d'un problème local.

Rules
Règlement

- Langages et outils :Vous ne pouvez utiliser que des langages et des outils open source pour développer des modèles dans le cadre de ce challenge.

- Qui peut participer : Ouvert à tous.

- Limites de soumission : 10 soumissions par jour, 300 soumissions au total.

- Taille des équipes : 4 membres maximum par équipe.

- Répartition public-privé : Zindi maintient un classement public et un classement privé pour chaque challenge. Le classement public comprend environ 30 % du jeu de données de test. Le classement privé sera révélé à la clôture du challenge et contient les 70 % restants du jeu de test.

- Partage des données : Licence CC-BY SA 4.0.

- Vérification du code : Les 10 premiers du classement privé recevront un e-mail demandant leur code à la clôture du challenge. Vous disposerez de 48 heures pour soumettre votre code.

- Partage de code : L'utilisation de plusieurs comptes ou le partage de code et d'informations entre des comptes qui ne font pas partie de la même équipe est interdit et entraînera une disqualification.

LA PARTICIPATION À CE CHALLENGE CONSTITUE VOTRE ACCEPTATION DU PRÉSENT RÈGLEMENT OFFICIEL DU CHALLENGE.

Règlement complet du challenge

Ce challenge est ouvert à tous.

Équipes et collaboration

Vous pouvez participer aux challenges individuellement ou en équipe de quatre personnes maximum. Lors de la création d'une équipe, le nombre total de soumissions effectuées par ses membres doit être inférieur ou égal au nombre maximal de soumissions autorisées à la date de formation de l'équipe. Une équipe sera autorisée à effectuer le nombre maximal de soumissions prévu pour le challenge, moins le nombre total de soumissions déjà effectuées par ses membres au moment de sa formation. Les prix sont transférés uniquement aux participants individuels ou au chef d'équipe.

Il est interdit à un utilisateur de posséder plusieurs comptes, ainsi que de collaborer avec plusieurs équipes ou d'appartenir à plusieurs équipes. Les personnes utilisant plusieurs comptes, ainsi que leurs soumissions provenant de ces comptes, seront immédiatement disqualifiées de la plateforme.

Le code ne doit pas être partagé en privé en dehors d'une équipe. Tout code partagé doit être mis à la disposition de tous les participants au challenge par l'intermédiaire de la plateforme (c'est-à-dire sur les forums de discussion).

Le data scientist Zindi qui crée une équipe en devient le chef d'équipe par défaut, mais il peut transférer cette responsabilité à un autre data scientist de l'équipe. Le chef d'équipe peut inviter d'autres data scientists à rejoindre son équipe. Les data scientists invités peuvent accepter ou refuser les invitations. Tant qu'un deuxième data scientist n'a pas accepté une invitation à rejoindre l'équipe, le data scientist qui a créé l'équipe reste considéré comme participant individuel dans le classement. Aucun membre supplémentaire ne peut être ajouté aux équipes au cours des 5 derniers jours du challenge ou de la dernière heure d'un hackathon.

Le chef d'équipe peut initier une fusion avec une autre équipe. Seul le chef de la deuxième équipe peut accepter l'invitation. Le chef d'équipe par défaut sera celui de l'équipe ayant initié l'invitation. Les équipes ne peuvent fusionner que si le nombre total de membres est inférieur ou égal à la taille maximale autorisée pour le challenge.

Une équipe peut être dissoute si elle n'a pas encore effectué de soumission. Une fois qu'une soumission a été effectuée, les membres individuels ne peuvent plus quitter l'équipe.

Tous les membres de l'équipe reçoivent les points associés à leur classement dans le challenge, sans partage ni division des points entre les membres.

Jeux de données, packages et principes généraux

La solution doit utiliser uniquement des packages open source accessibles au public.

Vous ne pouvez utiliser que les jeux de données fournis pour ce challenge ainsi que des jeux de données climatiques accessibles au public.

Vous pouvez utiliser des modèles préentraînés, à condition qu'ils soient librement accessibles à tous.

Les outils d'apprentissage automatique automatisé tels qu'AutoML ne sont pas autorisés.

Si la métrique d'évaluation exige la soumission de probabilités, ne définissez pas de seuils (et n'arrondissez pas vos probabilités) dans le but d'améliorer votre position dans le classement. Afin de garantir que le client reçoive la meilleure solution, Zindi aura besoin des probabilités brutes. Cela permettra aux clients de définir des seuils en fonction de leurs propres besoins.

Vous êtes autorisé à accéder aux données du challenge, à les utiliser et à les partager à des fins commerciales, non commerciales, de recherche ou d'enseignement, conformément à la licence CC-BY SA 4.0.

Vous devez informer immédiatement Zindi dès que vous avez connaissance d'une transmission non autorisée ou d'un accès non autorisé aux données du challenge, et collaborer avec Zindi pour remédier à toute transmission ou tout accès non autorisé.

Votre solution ne doit pas porter atteinte aux droits de tiers et vous devez être légalement habilité à céder à Zindi l'ensemble des droits d'auteur relatifs au code de la solution gagnante.

Soumissions et sélection des gagnants

Vous pouvez effectuer un maximum de 10 soumissions par jour.

Vous pouvez effectuer un maximum de 300 soumissions pour ce challenge.

Avant la fin du challenge, vous devez sélectionner 2 soumissions qui seront évaluées sur le classement privé. Si vous n'effectuez aucune sélection, vos 2 meilleures soumissions du classement public seront utilisées pour établir votre score sur le classement privé.

Pendant le challenge, votre meilleur score public sera affiché, quelles que soient les soumissions que vous avez sélectionnées. À la clôture du challenge, votre meilleur score privé parmi les 2 soumissions sélectionnées sera affiché.

Zindi maintient un classement public et un classement privé pour chaque challenge. Le classement public comprend environ 20 % du jeu de données de test. Pendant que le challenge est ouvert, le classement public classe les solutions soumises selon le score de précision qu'elles obtiennent. À la clôture du challenge, le classement privé, qui couvre les 80 % restants du jeu de données de test, sera rendu public et constituera le classement final du challenge.

Veuillez noter que, pour être prise en compte, votre soumission doit d'abord passer l'étape de traitement. Si votre soumission échoue pendant cette étape, elle ne sera pas comptabilisée et ne recevra aucun score ; elle ne sera pas non plus déduite de votre limite quotidienne de soumissions. Si vous rencontrez des problèmes avec votre fichier de soumission, la meilleure solution consiste à demander conseil sur la page du challenge.

Si vous figurez parmi les 10 premiers au moment de la clôture du classement, nous vous enverrons un e-mail pour vous demander votre code. À compter de la réception de cet e-mail, vous disposerez de 48 heures pour répondre et soumettre votre code conformément aux directives relatives à la reproductibilité du code soumis, détaillées ci-dessous. L'absence de réponse entraînera une disqualification.

Si votre solution se classe à la 1re, 2e ou 3e place du classement final, vous devrez nous soumettre le code de votre solution gagnante pour vérification et vous acceptez, par conséquent, de céder à Zindi l'ensemble des droits d'auteur mondiaux relatifs à cette solution gagnante.

Si deux solutions obtiennent des scores identiques dans le classement, elles seront départagées selon la date et l'heure de leur soumission (la solution soumise en premier sera gagnante).

Les gagnants seront payés par virement bancaire, par PayPal si le paiement est inférieur ou égal à 100 $, ou par une autre plateforme de transfert d'argent international. Les frais de transfert international seront déduits du montant total du prix, sauf si le montant du prix est inférieur à 500 $, auquel cas les frais de transfert international seront pris en charge par Zindi. Dans tous les cas, les gagnants sont responsables de tous les autres frais appliqués par leur propre banque ou tout autre établissement pour la réception du prix. Toutes les taxes imposées sur les prix sont à la charge exclusive des gagnants. Les gagnants les mieux classés ou les chefs d'équipe devront fournir à Zindi une pièce d'identité, un justificatif de domicile et une lettre de leur banque confirmant leurs coordonnées bancaires. Les gagnants seront payés en dollars américains (USD) ou dans la devise du challenge. Si votre compte ne peut pas recevoir de dollars américains ou la devise du challenge, votre banque devra en fournir la preuve et Zindi essaiera de trouver une solution adaptée.

Veuillez noter qu'en raison du conflit en cours entre la Russie et l'Ukraine, nous ne sommes actuellement pas en mesure de verser des prix aux gagnants situés en Russie. Nous nous excusons pour les désagréments que cela pourrait occasionner et traiterons les éventuels problèmes au cas par cas.

Le paiement sera effectué après la vérification du code et la validation définitive du classement.

Vous reconnaissez et acceptez que Zindi puisse, sans aucune obligation de le faire, exclure ou disqualifier une personne, une équipe ou un compte si Zindi estime que cette personne, cette équipe ou ce compte enfreint le présent règlement. La participation à ce challenge constitue votre acceptation du présent règlement officiel du challenge.

Zindi s'engage à fournir des solutions de valeur à ses clients et partenaires. À cette fin, nous nous réservons le droit de disqualifier votre soumission pour des raisons d'utilisabilité ou de valeur. Cela inclut, sans s'y limiter, l'utilisation de fuites de données ou toute autre pratique que nous estimons compromettre la valeur intrinsèque de votre solution.

Zindi se réserve également le droit de vous disqualifier et/ou de disqualifier vos soumissions de tout challenge si nous estimons que vous avez enfreint les règles ou l'esprit du challenge ou de la plateforme de quelque manière que ce soit. Les disqualifications sont indépendantes de votre position dans le classement et relèvent entièrement de la discrétion de Zindi.

Veuillez consulter la FAQ et les Conditions d'utilisation pour connaître les règles supplémentaires pouvant s'appliquer à ce challenge. Nous nous réservons le droit de modifier ces règles à tout moment.

## Reproductibilité du code soumis

Si le code que vous avez soumis ne permet pas de reproduire votre score dans le classement, nous nous réservons le droit d'ajuster votre rang en fonction du score obtenu avec le code soumis.

Si votre code ne s'exécute pas, vous serez retiré du top 10. Veuillez vous assurer que votre code fonctionne avant de soumettre votre solution.

Définissez toujours la graine aléatoire (*seed*). La réexécution de votre modèle doit toujours vous placer à la même position dans le classement. Si, lors de l'exécution de votre solution, le caractère aléatoire vous fait descendre dans le classement, nous nous réservons le droit d'ajuster votre rang au score le plus proche que votre soumission permet de reproduire.

Les packages personnalisés dans votre notebook de soumission ne seront pas acceptés.

Vous ne pouvez utiliser que des outils accessibles à tous, c'est-à-dire aucun service payant ni aucune période d'essai gratuite nécessitant une carte bancaire.

Lisez [cet article](https://zindi.africa/learn/documentation-guideline) sur la préparation de votre documentation et [cet article](https://zindi.africa/learn/how-to-ensure-success-when-submitting-your-code-for-review) sur la manière de garantir la réussite de la vérification de votre code.

**Conséquences de toute violation des règles du challenge ou des directives de soumission :**

**Première infraction :** Aucun prix pendant 6 mois et 2 000 points seront retirés de votre profil (période probatoire). Si vous êtes pris en train de tricher, toutes les personnes impliquées seront disqualifiées du ou des challenges concernés, et vous ne pourrez remporter aucun prix dans les challenges pendant les six mois suivants. De plus, 2 000 points seront retirés de votre profil. Si votre profil comporte moins de 2 000 points, votre solde sera ramené à 0.

**Deuxième infraction :** Bannissement de la plateforme. Si vous êtes pris en infraction une deuxième fois, votre compte Zindi sera désactivé et vous serez disqualifié de toute possibilité de gagner des prix ou des points Zindi en utilisant un autre compte.

Les équipes dont certains membres ont été pris en train de tricher ne pourront recevoir ni prix ni points dans le challenge concerné, que les autres membres aient eu connaissance ou non de l'infraction ou y aient participé.

Les équipes comprenant des personnes ayant déjà commis une infraction ne pourront prétendre à aucun prix dans les challenges pendant la période probatoire de 6 mois.

Surveillance des soumissions

Nous examinerons les 10 meilleures solutions de chaque challenge à sa clôture.

Nous nous réservons le droit de demander le code de n'importe quel utilisateur à tout moment pendant un challenge. Vous disposerez de 24 heures pour soumettre votre code conformément aux règles de vérification du code (voir ci-dessus). Zindi se réserve le droit de ne pas expliquer les raisons de sa demande de code. Si vous ne soumettez pas votre code dans les 24 heures, vous serez disqualifié de toute possibilité de gagner des prix ou des points Zindi pendant les six mois suivants. Si vous faites de nouveau l'objet de soupçons, que votre code vous est demandé et que vous ne le soumettez pas dans les 24 heures, votre compte Zindi sera désactivé et vous serez disqualifié de toute possibilité de gagner des prix ou des points Zindi avec tout autre compte.

Evaluation
LEADERBOARD ET SOUMISSIONS

Métrique : Log Loss (multi-classe : aucune, faible, moderee, forte)

Prizes
PRIX (en francs guinéens)

- 1er : 1 000 000 GNF

- 2e : 600 000 GNF

- 3e : 400 000 GNF

Les prix sont remis par les organisateurs, sur place.

Les 5 meilleures équipes doivent fournir leur code reproductible avant le 14 novembre à 16:00 GMT et présenter leur démarche au jury.

2 soumissions sélectionnées par équipe pour le classement final.

Organization
IndabaX Guinée, en collaboration avec l'Université de Labé et l'Université Mercure International de Guinée.

Contact : Abdoulaye Balde – abdoulahi.pro96@gmail.com