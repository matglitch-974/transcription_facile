================================================================================
                    SAISIE VOCALE INTELLIGENTE v2.0
                           Guide d'utilisation
================================================================================

DESCRIPTION
-----------
Script Python de transcription vocale qui convertit automatiquement la parole
en texte. Utilise Whisper (OpenAI) quand internet est disponible, ou Vosk
en mode hors-ligne.


PREREQUIS
---------
- Python 3.8 ou superieur
- Windows 10/11
- Microphone fonctionnel

Dependances Python a installer :
    pip install openai-whisper numpy sounddevice vosk


FONCTIONNALITES
---------------
- Transcription haute precision avec Whisper (avec internet)
- Mode hors-ligne avec Vosk (sans internet)
- Indicateur de niveau audio en temps reel
- Choix du microphone au demarrage
- Sauvegarde automatique sur le bureau en .txt
- Possibilite de continuer un fichier existant
- Copie automatique dans le presse-papier


UTILISATION
-----------
1. Lancer le script :
   python saisie_vocale.py

2. Choisir le microphone :
   [0] Par defaut (systeme)
   [1] Premier microphone
   [2] Deuxieme microphone
   ...
   Taper 0 ou ENTREE pour utiliser le micro par defaut.

3. Menu principal :
   [1] Nouvelle saisie    -> Enregistre et cree un nouveau fichier
   [2] Continuer fichier  -> Enregistre et ajoute a un fichier existant
   [3] Quitter

4. Pendant l'enregistrement :
   - Parlez normalement face au microphone
   - Une barre de niveau audio s'affiche en temps reel :
     Niveau: [##########--------------------] [OK]
   - Appuyez sur N'IMPORTE QUELLE TOUCHE pour arreter

5. Apres l'enregistrement :
   - Le texte transcrit s'affiche
   - Il est copie automatiquement dans le presse-papier
   - Entrez le nom du fichier (sans .txt) pour sauvegarder


INDICATEUR DE NIVEAU AUDIO
--------------------------
   [bas]  -> Parlez plus fort ou rapprochez-vous du micro
   [OK]   -> Niveau correct
   [FORT] -> Trop fort, eloignez-vous un peu


FICHIERS GENERES
----------------
- Les transcriptions sont sauvegardees sur le bureau
- Format : nom_choisi.txt
- Si le fichier existe deja : nom_choisi_1.txt, nom_choisi_2.txt, etc.


CONFIGURATION AVANCEE
---------------------
Modifier les constantes au debut du script :

WHISPER_MODEL = "small"
   Options : "tiny", "base", "small", "medium", "large"
   - tiny/base : rapide mais moins precis
   - small : bon compromis (recommande)
   - medium/large : tres precis mais lent

SAMPLE_RATE = 16000
   Taux d'echantillonnage audio (ne pas modifier sauf probleme)


MODELES TELECHARGES
-------------------
Au premier lancement, les modeles sont telechargés automatiquement :
- Whisper "small" : ~500 Mo (stocke dans le cache utilisateur)
- Vosk francais : ~1.4 Go (stocke dans ./modeles_vocaux/)


RESOLUTION DE PROBLEMES
-----------------------
1. "Aucun texte detecte"
   - Verifiez que le microphone fonctionne
   - Parlez plus fort
   - Verifiez le niveau audio affiche

2. Le script se ferme immediatement
   - Lancez depuis un terminal (pas double-clic)
   - Verifiez les dependances installees

3. Mauvaise qualite de transcription
   - Reduisez le bruit ambiant
   - Parlez clairement et pas trop vite
   - Essayez un modele Whisper plus grand (medium/large)

4. Erreur microphone
   - Verifiez que le micro n'est pas utilise par une autre application
   - Essayez un autre peripherique d'entree


RACCOURCIS
----------
- Menu : ENTREE = option 1 (nouvelle saisie)
- Microphone : 0 ou ENTREE = peripherique par defaut
- Enregistrement : N'importe quelle touche = arreter
- Nom fichier : ENTREE = "transcription.txt"


================================================================================
                              Bonne utilisation !
================================================================================
