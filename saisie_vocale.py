"""
Script de saisie vocale intelligent - Version amelioree
- Avec internet: utilise Whisper (haute precision)
- Sans internet: utilise Vosk (hors-ligne)
- Indicateur de niveau audio en temps reel
- Choix du microphone
- Appuyez sur n'importe quelle touche pour arreter l'enregistrement
"""

import os
import sys
import json
import queue
import tempfile
import wave
import zipfile
import urllib.request
import socket
import threading
import time
import msvcrt  # Pour detecter les touches sur Windows

import numpy as np
import sounddevice as sd


# === CONFIGURATION ===
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeles_vocaux")
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip"
VOSK_MODEL_NAME = "vosk-model-fr-0.22"
SAMPLE_RATE = 16000

# Configuration Whisper (small = meilleure qualite, base = plus rapide)
WHISPER_MODEL = "small"


def verifier_internet(timeout=3):
    """Verifie si une connexion internet est disponible."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def lister_microphones():
    """Retourne la liste des microphones (peripheriques d'entree uniquement)."""
    mics = []
    default_id = sd.default.device[0]

    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            is_default = (i == default_id)
            mics.append({
                'id': i,
                'name': device['name'],
                'default': is_default
            })
    return mics


def choisir_microphone():
    """Permet a l'utilisateur de choisir un microphone."""
    mics = lister_microphones()

    print("\nPeripheriques d'entree (microphones):")
    print("  [0] Par defaut (systeme)")

    for idx, mic in enumerate(mics, 1):
        marker = " <-- DEFAUT" if mic['default'] else ""
        print(f"  [{idx}] {mic['name']}{marker}")

    print()
    choix = input("Choix (0 ou ENTREE = defaut): ").strip()

    # 0 ou vide = defaut
    if not choix or choix == "0":
        for mic in mics:
            if mic['default']:
                print(f"-> {mic['name']}")
                return mic['id']
        if mics:
            print(f"-> {mics[0]['name']}")
            return mics[0]['id']
        return None

    try:
        num = int(choix)
        if 1 <= num <= len(mics):
            mic = mics[num - 1]
            print(f"-> {mic['name']}")
            return mic['id']
        print("Numero invalide, utilisation du micro par defaut.")
        return None
    except ValueError:
        print("Utilisation du micro par defaut.")
        return None


def vider_buffer_clavier():
    """Vide le buffer clavier pour eviter les faux positifs."""
    while msvcrt.kbhit():
        msvcrt.getch()


def attendre_touche(stop_event):
    """Thread qui attend que l'utilisateur appuie sur une touche."""
    # Attendre un peu avant de commencer a detecter
    time.sleep(0.3)

    while not stop_event.is_set():
        if msvcrt.kbhit():
            msvcrt.getch()  # Consommer la touche
            stop_event.set()
            break
        time.sleep(0.05)


def calculer_niveau_db(audio_data):
    """Calcule le niveau en dB d'un bloc audio."""
    if len(audio_data) == 0:
        return -100
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    if rms > 0:
        return 20 * np.log10(rms / 32768)
    return -100


def afficher_niveau(niveau_db):
    """Affiche une barre de niveau audio."""
    # Normaliser entre -60 dB et 0 dB
    niveau_norm = max(0, min(1, (niveau_db + 60) / 60))
    longueur = int(niveau_norm * 30)
    barre = "#" * longueur + "-" * (30 - longueur)

    # Couleur selon le niveau
    if niveau_db > -10:
        indicateur = "[FORT]"
    elif niveau_db > -30:
        indicateur = "[OK]  "
    else:
        indicateur = "[bas] "

    print(f"\r  Niveau: [{barre}] {indicateur}", end="", flush=True)


def normaliser_audio(audio):
    """Normalise l'audio pour un meilleur resultat."""
    if len(audio) == 0:
        return audio

    # Supprimer le silence au debut et a la fin
    seuil = 0.01
    indices = np.where(np.abs(audio) > seuil)[0]
    if len(indices) > 0:
        debut = max(0, indices[0] - 1600)  # 0.1s de marge
        fin = min(len(audio), indices[-1] + 1600)
        audio = audio[debut:fin]

    # Normaliser le volume
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    return audio


# === WHISPER (AVEC INTERNET) ===

class MoteurWhisper:
    def __init__(self, model_size=WHISPER_MODEL, device_id=None):
        import whisper
        print(f"Chargement de Whisper ({model_size})...")
        print("(Premier lancement: telechargement du modele)")
        self.model = whisper.load_model(model_size)
        self.sample_rate = SAMPLE_RATE
        self.device_id = device_id
        print("Whisper pret!")

    def enregistrer(self):
        """Enregistre jusqu'a ce que l'utilisateur appuie sur une touche."""
        print("\n" + "=" * 50)
        print("   ENREGISTREMENT EN COURS")
        print("   Appuyez sur une TOUCHE pour terminer")
        print("=" * 50)

        # Vider le buffer clavier
        vider_buffer_clavier()

        frames = []
        stop_event = threading.Event()

        # Thread pour attendre une touche
        thread = threading.Thread(target=attendre_touche, args=(stop_event,))
        thread.daemon = True
        thread.start()

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())
            # Afficher le niveau audio
            niveau = calculer_niveau_db(indata * 32767)
            afficher_niveau(niveau)

        kwargs = {
            'samplerate': self.sample_rate,
            'channels': 1,
            'dtype': 'float32',
            'blocksize': 1024,
            'callback': callback
        }
        if self.device_id is not None:
            kwargs['device'] = self.device_id

        with sd.InputStream(**kwargs):
            while not stop_event.is_set():
                time.sleep(0.05)

        print("\r" + " " * 50 + "\r", end="")
        print("Enregistrement termine.")

        if not frames:
            return None

        audio = np.concatenate(frames).flatten()
        return normaliser_audio(audio)

    def transcrire(self, audio, langue="fr"):
        """Transcrit l'audio avec parametres optimises."""
        print("Transcription en cours...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            with wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes((audio * 32767).astype(np.int16).tobytes())

        try:
            # Parametres optimises pour le francais
            result = self.model.transcribe(
                temp_path,
                language=langue,
                fp16=False,
                initial_prompt="Transcription en francais avec ponctuation.",
                condition_on_previous_text=True,
                temperature=0.0,  # Plus deterministe
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
            )
            return result["text"].strip()
        finally:
            os.unlink(temp_path)

    def ecouter(self, langue="fr"):
        """Ecoute et transcrit une phrase."""
        audio = self.enregistrer()
        if audio is None or len(audio) < self.sample_rate * 0.3:
            return None
        return self.transcrire(audio, langue)


# === VOSK (SANS INTERNET) ===

def telecharger_modele_vosk():
    """Telecharge le modele Vosk si necessaire."""
    model_path = os.path.join(MODELS_DIR, VOSK_MODEL_NAME)

    if os.path.exists(model_path):
        return model_path

    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(MODELS_DIR, f"{VOSK_MODEL_NAME}.zip")

    print("Telechargement du modele Vosk francais (~1.4 Go)...")
    print("(Modele complet pour meilleure precision)")

    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        downloaded = count * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        print(f"\r  Progression: {percent}% ({downloaded:.1f}/{total:.1f} Mo)", end="", flush=True)

    urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path, progress)
    print()

    print("Extraction...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(MODELS_DIR)
    os.remove(zip_path)

    print("Modele Vosk installe!")
    return model_path


class MoteurVosk:
    def __init__(self, device_id=None):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)  # Desactiver les logs verbeux

        model_path = telecharger_modele_vosk()
        print("Chargement de Vosk...")
        self.model = Model(model_path)
        self.KaldiRecognizer = KaldiRecognizer
        self.sample_rate = SAMPLE_RATE
        self.audio_queue = queue.Queue()
        self.device_id = device_id
        print("Vosk pret!")

    def callback(self, indata, frames, time_info, status):
        self.audio_queue.put(bytes(indata))
        # Afficher le niveau
        niveau = calculer_niveau_db(np.frombuffer(indata, dtype=np.int16))
        afficher_niveau(niveau)

    def ecouter(self, langue=None):
        """Ecoute et transcrit une phrase."""
        recognizer = self.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        print("\n" + "=" * 50)
        print("   ENREGISTREMENT EN COURS")
        print("   Appuyez sur une TOUCHE pour terminer")
        print("=" * 50)

        # Vider le buffer clavier
        vider_buffer_clavier()

        stop_event = threading.Event()
        thread = threading.Thread(target=attendre_touche, args=(stop_event,))
        thread.daemon = True
        thread.start()

        # Vider la queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        kwargs = {
            'samplerate': self.sample_rate,
            'blocksize': 4000,
            'dtype': 'int16',
            'channels': 1,
            'callback': self.callback
        }
        if self.device_id is not None:
            kwargs['device'] = self.device_id

        with sd.RawInputStream(**kwargs):
            resultats = []

            while not stop_event.is_set():
                try:
                    data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    texte = result.get("text", "")
                    if texte:
                        resultats.append(texte)

            # Vider la queue restante
            while not self.audio_queue.empty():
                try:
                    data = self.audio_queue.get_nowait()
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        if result.get("text"):
                            resultats.append(result["text"])
                except queue.Empty:
                    break

            final = json.loads(recognizer.FinalResult())
            if final.get("text"):
                resultats.append(final["text"])

            print("\r" + " " * 50 + "\r", end="")
            print("Enregistrement termine.")

            texte_final = " ".join(resultats) if resultats else None

            # Ajouter la ponctuation basique
            if texte_final:
                texte_final = ajouter_ponctuation(texte_final)

            return texte_final


def ajouter_ponctuation(texte):
    """Ajoute une ponctuation basique au texte."""
    if not texte:
        return texte

    # Premiere lettre en majuscule
    texte = texte[0].upper() + texte[1:] if len(texte) > 1 else texte.upper()

    # Point final si absent
    if texte and texte[-1] not in '.!?':
        texte += '.'

    return texte


# === MOTEUR PRINCIPAL ===

class SaisieVocale:
    def __init__(self, device_id=None):
        self.moteur = None
        self.mode = None
        self.device_id = device_id
        self._initialiser()

    def _initialiser(self):
        """Choisit le meilleur moteur selon la connexion."""
        print("\nVerification de la connexion...")

        if verifier_internet():
            print("Internet detecte -> Whisper (haute precision)\n")
            self.mode = "whisper"
            try:
                self.moteur = MoteurWhisper(WHISPER_MODEL, self.device_id)
            except ImportError:
                print("Whisper non installe, bascule vers Vosk...")
                self._initialiser_vosk()
        else:
            print("Pas d'internet -> Vosk (hors-ligne)\n")
            self._initialiser_vosk()

    def _initialiser_vosk(self):
        """Initialise Vosk."""
        self.mode = "vosk"
        try:
            self.moteur = MoteurVosk(self.device_id)
        except ImportError as e:
            print(f"Erreur: Vosk non installe. {e}")
            sys.exit(1)

    def ecouter(self):
        """Ecoute et retourne le texte."""
        return self.moteur.ecouter()


def main():
    print("=" * 50)
    print("     SAISIE VOCALE INTELLIGENTE v2.0")
    print("=" * 50)

    # Choix du microphone
    try:
        device_id = choisir_microphone()
    except Exception as e:
        print(f"Erreur microphones: {e}")
        device_id = None

    try:
        saisie = SaisieVocale(device_id)
    except Exception as e:
        print(f"Erreur d'initialisation: {e}")
        input("\nAppuyez sur ENTREE pour fermer...")
        return

    print(f"\nMoteur actif: {saisie.mode.upper()}")
    if saisie.mode == "whisper":
        print(f"Modele: {WHISPER_MODEL}")

    bureau = os.path.join(os.path.expanduser("~"), "Desktop")

    while True:
        print("\n" + "-" * 50)
        print("[1] Nouvelle saisie")
        print("[2] Continuer un fichier")
        print("[3] Quitter")
        choix = input("Choix: ").strip()

        if choix == "3" or choix.lower() == "q":
            print("Au revoir!")
            break

        elif choix == "2":
            # Demander le nom du fichier a continuer
            nom_fichier = input("Nom du fichier (sans .txt): ").strip()
            if not nom_fichier:
                print("Nom invalide.")
                continue

            chemin_fichier = os.path.join(bureau, f"{nom_fichier}.txt")

            if not os.path.exists(chemin_fichier):
                print(f"Fichier non trouve: {chemin_fichier}")
                continue

            print(f"Fichier selectionne: {nom_fichier}.txt")

            # Faire la saisie
            texte = saisie.ecouter()
            if texte:
                print("\n" + "=" * 50)
                print("RESULTAT:")
                print(texte)
                print("=" * 50)

                # Copier dans le presse-papier
                try:
                    import subprocess
                    subprocess.run(['clip'], input=texte.encode('utf-8'), check=True)
                    print("(Copie dans le presse-papier)")
                except:
                    pass

                # Ajouter au fichier
                with open(chemin_fichier, "a", encoding="utf-8") as f:
                    f.write("\n\n" + texte)
                print(f"Transcription ajoutee a: {chemin_fichier}")
            else:
                print("\nAucun texte detecte.")

        elif choix == "1" or choix == "":
            texte = saisie.ecouter()
            if texte:
                print("\n" + "=" * 50)
                print("RESULTAT:")
                print(texte)
                print("=" * 50)

                # Copier dans le presse-papier
                try:
                    import subprocess
                    subprocess.run(['clip'], input=texte.encode('utf-8'), check=True)
                    print("(Copie dans le presse-papier)")
                except:
                    pass

                # Nouveau fichier
                nom_fichier = input("\nNom du fichier (sans .txt): ").strip()
                if not nom_fichier:
                    nom_fichier = "transcription"

                chemin_fichier = os.path.join(bureau, f"{nom_fichier}.txt")

                # Eviter d'ecraser un fichier existant
                compteur = 1
                chemin_base = chemin_fichier
                while os.path.exists(chemin_fichier):
                    chemin_fichier = chemin_base.replace(".txt", f"_{compteur}.txt")
                    compteur += 1

                with open(chemin_fichier, "w", encoding="utf-8") as f:
                    f.write(texte)

                print(f"Transcription sauvegardee: {chemin_fichier}")
            else:
                print("\nAucun texte detecte.")

    input("\nAppuyez sur ENTREE pour fermer...")


if __name__ == "__main__":
    main()
