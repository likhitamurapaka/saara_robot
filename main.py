"""
SAARA - Offline Voice Assistant
================================
FLOW:
  1. Sleeping — waiting for wake word "Saara"
  2. Wake word heard — eyes open
  3. Face check — greet by name OR say "Hi! How may I help you?"
  4. Listen for voice commands
  5. "Sleep" — goes back to sleep

Works on Windows AND Raspberry Pi (auto-detected)
Fullscreen 1024x768
"""
import robot_control
import os, sys, json, queue, pickle, threading, datetime
import subprocess, webbrowser, time
import cv2, dlib, pygame, psutil, numpy as np
import sounddevice as sd
import pywhatkit
from vosk import Model, KaldiRecognizer

# ── Paths ────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
VMODEL  = os.path.join(BASE, "model", "vosk-model-small-en-us-0.15")
ENCFILE = os.path.join(BASE, "encodings.pkl")
DMODELS = os.path.join(BASE, "dlib_models")
PRED    = os.path.join(DMODELS, "shape_predictor_68_face_landmarks.dat")
RECOG   = os.path.join(DMODELS, "dlib_face_recognition_resnet_model_v1.dat")
IS_PI   = sys.platform != "win32"

# ── Audio ────────────────────────────────────────────────
SR, BS  = 16000, 8000

# ── Screen ───────────────────────────────────────────────
W, H    = 1024, 768

# ── Colours ──────────────────────────────────────────────
WHITE  = (255,255,255); BLACK  = (0,0,0)
PINK   = (255,160,180); BLUSH  = (255,100,140)
CYAN   = (0,220,255);   BG     = (20,20,30)
GREEN  = (0,220,100);   YELLOW = (255,220,50)
GREY   = (100,100,115); PURPLE = (180,100,255)
SHADOW = (35,35,50)

# ── Shared state ─────────────────────────────────────────
S = {
    "sleeping":   True,
    "blush":      False,
    "wink":       False,
    "listening":  False,
    "speaking":   False,
    "blink_open": False,
    "status":     "z z z   sleeping   z z z",
    "last_cmd":   "",
    "greeted":    "",
    "voltage":    12.0,
}
LOCK = threading.Lock()

# ── Queues ───────────────────────────────────────────────
# NOTE: audio queue is NOT bounded — we never drop audio frames
# We only skip processing while speaking, not drop frames
AQ = queue.Queue()
SQ = queue.Queue()

# ── Face recognition ─────────────────────────────────────
THRESHOLD     = 0.52   # Lowered from 0.62 for stricter matching
GREET_COOLDOWN = 300
greeted_times  = {}

# ── Wake words ───────────────────────────────────────────
WAKE_WORDS = {
    "saara","sara","sarah","sarra","sora",
    "hey saara","hey sara","ok saara","hi saara",
    "namaste saara","namaste sara","namasthe saara","namasthe sara",
    "now my state", "now let's stay", "now this day", "no this damn", "now now now"
}

# ════════════════════════════════════════════════════════
#  STATE HELPERS
# ════════════════════════════════════════════════════════

def gst(*keys):
    with LOCK:
        return S[keys[0]] if len(keys)==1 else tuple(S[k] for k in keys)

def sst(**kw):
    with LOCK: S.update(kw)

def is_wake(text):
    t = text.lower().strip()
    return any(w in t for w in WAKE_WORDS)

# ════════════════════════════════════════════════════════
#  TTS — blocking, runs in its own thread
# ════════════════════════════════════════════════════════

def _say(text):
    safe = text.replace('"','').replace("'","")
    if IS_PI:
        subprocess.call(["espeak","-v","en+f3","-s","150",safe],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:
        subprocess.call(
            f'powershell -Command "Add-Type -AssemblyName System.Speech;'
            f'$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            f"$s.SelectVoice('Microsoft Zira Desktop');"
            f'$s.Rate=1;$s.Volume=100;$s.Speak(\\"{safe}\\");"',
            shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def tts_worker():
    while True:
        text = SQ.get()
        if text is None: break
        sst(speaking=True)
        _say(text)
        sst(speaking=False)

def speak(text):
    print(f"[SAARA] {text}")
    SQ.put(text)

def speak_wait(text):
    """Speak and wait until finished before returning."""
    speak(text)
    while gst("speaking"):
        time.sleep(0.1)
    time.sleep(0.3)  # small gap after speaking

# ════════════════════════════════════════════════════════
#  AUDIO — never drop frames
# ════════════════════════════════════════════════════════

def audio_cb(indata, frames, t, status):
    AQ.put(bytes(indata))

# ════════════════════════════════════════════════════════
#  FACE RECOGNITION — single snapshot, not continuous loop
#  Called once after wake word to identify who is there
# ════════════════════════════════════════════════════════

_face_det  = None
_face_pred = None
_face_rec  = None
_face_cap  = None
_face_encs = []
_face_names= []

def init_face():
    global _face_det, _face_pred, _face_rec, _face_cap
    global _face_encs, _face_names

    if not os.path.isfile(ENCFILE):
        print("[FACE] No encodings.pkl"); return False
    if not os.path.isfile(PRED):
        print("[FACE] No dlib models"); return False

    data = pickle.load(open(ENCFILE,"rb"))
    d = {}
    for enc, name in zip(data["encodings"], data["names"]):
        base = name.rstrip("0123456789_")
        d.setdefault(base,[]).append(enc)
    _face_encs  = [np.mean(v,axis=0) for v in d.values()]
    _face_names = list(d.keys())
    print(f"[FACE] Loaded: {_face_names}")

    _face_det  = dlib.get_frontal_face_detector()
    _face_pred = dlib.shape_predictor(PRED)
    _face_rec  = dlib.face_recognition_model_v1(RECOG)

    _face_cap = None
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            _face_cap = cap
            print(f"[CAM] Opened index {i}")
            break

    return _face_cap is not None

def identify_face():
    """
    Take a few frames from camera and try to identify who is there.
    Returns name string or None if not recognised.
    """
    if _face_cap is None or not _face_encs:
        return None

    # Try up to 10 frames to get a good detection
    for _ in range(10):
        ok, frame = _face_cap.read()
        if not ok: continue
        rgb = np.ascontiguousarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        dets = _face_det(rgb, 0)
        if not dets: continue
        try:
            enc   = np.array(_face_rec.compute_face_descriptor(rgb, _face_pred(rgb,dets[0])))
            dists = [np.linalg.norm(e-enc) for e in _face_encs]
            best  = int(np.argmin(dists))
            print(f"[FACE] Best: {_face_names[best]} dist={dists[best]:.3f}")
            if dists[best] < THRESHOLD:
                return _face_names[best]
            else:
                return None
        except Exception as e:
            print(f"[FACE] Error: {e}")
    return None

# ════════════════════════════════════════════════════════
#  COMMANDS
# ════════════════════════════════════════════════════════

def handle(cmd):
    cmd = cmd.lower().strip()
    sst(last_cmd=cmd, blush=False, wink=False)
    print(f"[CMD] {cmd}")

    if not cmd:
        speak("I didn't catch that. Still listening.")
        return True
    elif any(w in cmd for w in ("sleep","go to sleep")):
        speak("Going to sleep. Say my name to wake me!")
        return False
    elif any(w in cmd for w in ("exit","quit","bye","goodbye","see you","take care")):
        g = gst("greeted")
        msg = f"Bye {g}! Have a great day!" if g else "Bye! Have a great day!"
        speak_wait(msg)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        return True
    elif "happy birthday" in cmd:
        speak("Thank you! Same to you!")
        return True
    elif "bye" in cmd:
        speak("Enjoy the day!")
        return False
    elif any(w in cmd for w in ("hello","hi","hey")):
        speak("Hello! How can I help?")
    elif "your name" in cmd:
        speak("I am Saara, your assistant!")
    elif "how are you" in cmd:
        speak("I am doing great, thank you!")
    elif any(w in cmd for w in ("and now know", "yellow", "ela unna","alone now")):
        speak("bavunnanu, meeru ela unnaru.")
    elif "time" in cmd:
        speak(f"It is {datetime.datetime.now().strftime('%I:%M %p')}")
    elif any(w in cmd for w in ("it put a time to poop", "he put them to boot", "he put them", "ippudu time")):
        speak(f"ippudu time {datetime.datetime.now().strftime('%I %M %p')}")
    elif "date" in cmd or "today" in cmd:
        speak(f"Today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}")
    elif "youtube" in cmd:
        speak("Opening YouTube.")
        if "play" in cmd:
            song = cmd.replace("youtube", "").replace("play", "").strip()
            if not song:
                song = "popular songs"
            speak(f"Playing {song} on YouTube.")
            pywhatkit.playonyt(song)
        else:
            webbrowser.open("https://youtube.com")
    elif "play" in cmd and "song" in cmd:
        song = cmd.replace("play", "").replace("a song", "").replace("song", "").strip()
        if not song:
            song = "trending english songs"
        speak(f"Playing {song} on YouTube.")
        pywhatkit.playonyt(song)
    elif any(w in cmd for w in ("those were open say you", "does it happen to you", "the develop into you", "that open to you", "browser open")):
        speak("Opening browser."); webbrowser.open("https://google.com")
    elif "terminal" in cmd:
        speak("Opening terminal.")
        os.system("start cmd" if not IS_PI else "lxterminal &")
    elif "cpu" in cmd:
        speak(f"CPU usage is {psutil.cpu_percent(interval=1)} percent.")
    elif "memory" in cmd or "ram" in cmd:
        m = psutil.virtual_memory()
        speak(f"Memory is {m.percent} percent used.")
    elif "music" in cmd:
        speak("Opening music.")
        path = os.path.expanduser("~/Music")
        os.startfile(path) if not IS_PI else subprocess.Popen(["xdg-open",path])
    elif "shutdown" in cmd or "shut down" in cmd:
        speak("Shutting down!")
        time.sleep(2)
        os.system("sudo shutdown -h now" if IS_PI else "shutdown /s /t 1")
    elif "restart" in cmd or "reboot" in cmd:
        speak("Restarting!")
        time.sleep(2)
        os.system("sudo reboot" if IS_PI else "shutdown /r /t 1")
    elif "love you" in cmd:
        speak("Aww, thank you! You are sweet!")
        sst(blush=True, wink=True)
    elif "my name is" in cmd:
        name = cmd.split("my name is")[-1].strip().capitalize()
        speak(f"Nice to meet you, {name}!")
    elif "joke" in cmd:
        speak("Why do programmers prefer dark mode? Light attracts bugs!")
    elif "who am i" in cmd or "do you know me" in cmd:
        g = str(gst("greeted") or "")
        speak(f"Yes! You are {g}!" if g else "I am not sure. Look at the camera!")
    elif "forward" in cmd or "go forward" in cmd:
        robot_control.forward()
        speak("Moving forward")
    elif "backward" in cmd or "go back" in cmd:
        robot_control.backward()
        speak("Moving backward")
    elif "stop" in cmd:
        robot_control.stop()
        speak("Stopping")
    else:
        # Check if we were just asking Prasad sir about his bachelor life
        g = str(gst("greeted") or "")
        if g and "prasad" in g.lower() and "sir" in g.lower():
            speak("Okay, have a nice day!")
        else:
            speak("Sorry, I didn't understand. Please try again.")
    return True

# ════════════════════════════════════════════════════════
#  MAIN VOICE LOOP
#  Clean simple flow:
#    wake word → face check → greet → commands → sleep
# ════════════════════════════════════════════════════════

def voice_worker():
    model = Model(VMODEL)
    print("[VOSK] Ready.")

    while True:
        # ── SLEEPING: wait for wake word ──────────────
        rec = KaldiRecognizer(model, SR)
        sst(sleeping=True, listening=False, blink_open=False,
            status="z z z   sleeping   z z z", greeted="", last_cmd="")
        print("[MODE] Sleeping — waiting for wake word...")

        # flush stale audio
        while not AQ.empty():
            try: AQ.get_nowait()
            except: break

        while True:
            data = AQ.get()
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text","").lower().strip()
                print(f"[WAKE?] '{text}'")
                if is_wake(text):
                    just_woke_with_namasthe = any(w in text for w in ("namaste","namasthe","now my state","now let's stay","now this day","no this damn","now now now"))
                    break
            else:
                partial = json.loads(rec.PartialResult()).get("partial","").lower()
                if is_wake(partial):
                    just_woke_with_namasthe = any(w in partial for w in ("namaste","namasthe","now my state","now let's stay","now this day","no this damn","now now now"))
                    break

        # ── WOKEN: open eyes ──────────────────────────
        print("[MODE] Wake word detected!")
        sst(sleeping=False, blink_open=True, status="Identifying...")

        # flush audio collected during wake detection
        while not AQ.empty():
            try: AQ.get_nowait()
            except: break

        # ── FACE CHECK: who is it? ────────────────────
        name = identify_face()
        if name:
            sst(greeted=name, status="Listening...")
            greeting_pfx = "Namasthe" if just_woke_with_namasthe else "Hello"
            
            # Custom Greeting Logic
            if "prasad" in name.lower() and "sir" in name.lower():
                speak_wait(f"{greeting_pfx} Prasad Sir! How is your bachelor life going on?")
            elif "mam" in name.lower() or "madam" in name.lower():
                speak_wait(f"{greeting_pfx} {name}! You are beautiful today!")
            else:
                speak_wait(f"{greeting_pfx} {name}! Great to see you! Welcome to BSCs ASCII Fest!")
        else:
            sst(greeted="", status="Listening...")
            if just_woke_with_namasthe:
                speak_wait("Namasthe! Welcome to BSCs ASCII Fest! How may I help you?")
            else:
                speak_wait("Hi there! Welcome to BSCs ASCII Fest! How may I help you?")

        sst(listening=True, status="Listening...")

        # ── COMMAND MODE: listen until sleep/exit ─────
        print("[MODE] Command mode — listening...")
        rec = KaldiRecognizer(model, SR)
        cmd_timer = time.time()
        CMD_TIMEOUT = 10.0

        # flush audio again before listening for commands
        while not AQ.empty():
            try: AQ.get_nowait()
            except: break

        last_face_check = time.time()

        while True:
            # Face check every 3 seconds while in command mode
            if time.time() - last_face_check > 3.0:
                last_face_check = time.time()
                current_face = identify_face()
                if current_face and current_face != gst("greeted"):
                    sst(greeted=current_face)
                    speak_wait(f"Oh, hello {current_face}! What can I do for you?")
                    cmd_timer = time.time()
                    # flush audio
                    while not AQ.empty():
                        try: AQ.get_nowait()
                        except: break

            try:
                data = AQ.get(timeout=1.0)
            except queue.Empty:
                if time.time() - cmd_timer > CMD_TIMEOUT:
                    speak_wait("I am still here. What would you like?")
                    cmd_timer = time.time()
                continue

            # skip audio while speaking
            if gst("speaking"):
                continue

            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text","").lower().strip()
                if not text:
                    continue
                print(f"[HEARD] '{text}'")
                cmd_timer = time.time()
                stay = handle(text)
                # wait for speech to finish before listening again
                while gst("speaking"):
                    time.sleep(0.05)
                # flush any audio captured during speech
                while not AQ.empty():
                    try: AQ.get_nowait()
                    except: break
                rec = KaldiRecognizer(model, SR)
                if not stay:
                    break  # go back to sleep
            else:
                partial = json.loads(rec.PartialResult()).get("partial","").lower()
                if partial:
                    sst(status=f"Hearing: {partial}...")

# ════════════════════════════════════════════════════════
#  DRAW FACE
# ════════════════════════════════════════════════════════

def draw(screen, fs, fm, s):
    sl  = s["sleeping"]
    bl  = s["blush"]
    wk  = s["wink"]
    bo  = s["blink_open"]
    li  = s["listening"]
    st  = s["status"]
    lc  = s["last_cmd"]
    gr  = s["greeted"]
    vol = s.get("voltage", 12.0)

    screen.fill(BG)
    pygame.draw.rect(screen,SHADOW,(5,5,1014,710),border_radius=55)
    pygame.draw.rect(screen,WHITE, (10,10,1004,700),border_radius=50)

    # Draw ESC Button (Top-Left)
    pygame.draw.rect(screen, GREY, (30, 30, 100, 50), border_radius=10)
    pygame.draw.rect(screen, SHADOW, (30, 30, 100, 50), width=3, border_radius=10)
    esc_surf = fs.render("ESC", True, BLACK)
    screen.blit(esc_surf, esc_surf.get_rect(center=(80, 55)))

    # Draw Battery (Top-Right)
    pct = max(0, min(100, int((vol - 9.0) / (12.0 - 9.0) * 100)))
    col = GREEN if vol > 10.0 else (YELLOW if vol > 9.0 else (255, 50, 50))
    bat_text = f"BAT: {vol:.1f}V ({pct}%)"
    bat_surf = fm.render(bat_text, True, col)
    # The status dot is at (W-28, 28). We place battery info to its left.
    screen.blit(bat_surf, (W - 300, 16))

    # Low battery warning
    if vol <= 9.0:
        warn_surf = fs.render("LOW BATTERY WARNING! <= 9V", True, (255, 50, 50))
        warn_rect = warn_surf.get_rect(center=(W//2, 45))
        pygame.draw.rect(screen, WHITE, warn_rect.inflate(20, 20), border_radius=10)
        pygame.draw.rect(screen, (255, 50, 50), warn_rect.inflate(20, 20), width=3, border_radius=10)
        screen.blit(warn_surf, warn_rect)

    # Eyes
    if sl or not bo:
        pygame.draw.line(screen,BLACK,(210,310),(390,310),10)
        pygame.draw.line(screen,BLACK,(634,310),(814,310),10)
        if sl:
            screen.blit(fs.render("z  z  z",True,PURPLE),(830,130))
    else:
        # Left eye
        pygame.draw.circle(screen,BLACK,(300,310),90)
        pygame.draw.circle(screen,WHITE,(268,272),24)
        # Right eye or wink
        if wk:
            pygame.draw.arc(screen,BLACK,(634,298,180,45),0,3.14,9)
        else:
            pygame.draw.circle(screen,BLACK,(724,310),90)
            pygame.draw.circle(screen,WHITE,(692,272),24)
        # Eyelashes
        for p in [((258,224),(238,190)),((300,218),(300,182)),((342,224),(362,190)),
                  ((682,224),(662,190)),((724,218),(724,182)),((766,224),(786,190))]:
            pygame.draw.line(screen,BLACK,p[0],p[1],5)

    # Blush
    col,r = (BLUSH,72) if bl else (PINK,55)
    pygame.draw.circle(screen,col,(148,530),r)
    pygame.draw.circle(screen,col,(876,530),r)

    # Mouth
    pygame.draw.arc(screen,BLACK,(362,460,300,180),3.14,6.28,8)

    # Status dot
    if li:       dc,dr = GREEN,18
    elif not sl: dc,dr = YELLOW,15
    else:        dc,dr = GREY,12
    pygame.draw.circle(screen,dc,(W-28,28),dr)

    # Greeted name
    if gr and not sl:
        surf = fs.render(f"Hello  {gr} !",True,GREEN)
        screen.blit(surf, surf.get_rect(center=(W//2,90)))

    # Status text
    surf = fs.render(st,True,CYAN)
    screen.blit(surf, surf.get_rect(center=(W//2,H-30)))

    # Last command
    if lc and not sl:
        surf = fm.render(f'"{lc}"',True,YELLOW)
        screen.blit(surf, surf.get_rect(center=(W//2,H-65)))

    # Name plate
    surf = fm.render("S  A  A  R  A",True,GREY)
    screen.blit(surf, surf.get_rect(center=(W//2,H-8)))

# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def main():
    pygame.init()
    screen = pygame.display.set_mode((W,H), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    clock  = pygame.time.Clock()
    fs     = pygame.font.SysFont("monospace",32,bold=True)
    fm     = pygame.font.SysFont("monospace",24)

    # Init face recognition
    init_face()

    # Start TTS thread
    threading.Thread(target=tts_worker, daemon=True).start()

    # Start voice thread
    threading.Thread(target=voice_worker, daemon=True).start()

    # Start audio stream
    try:
        stream = sd.RawInputStream(
            samplerate=SR, blocksize=BS,
            dtype="int16", channels=1,
            latency="low", callback=audio_cb)
        stream.start()
        print("[MIC] Ready.")
    except Exception as e:
        print(f"[MIC ERROR] {e}"); stream=None

    print("[SAARA] Running. Say 'Saara' to wake. ESC to exit.")

    # Blink timer
    bt, bo = pygame.time.get_ticks(), True
    last_vol_time = 0

    while True:
        now = pygame.time.get_ticks()

        # Update battery every 5 seconds
        if now - last_vol_time > 5000:
            last_vol_time = now
            try:
                if hasattr(robot_control, 'get_voltage'):
                    sst(voltage=robot_control.get_voltage())
            except Exception as e:
                print(f"[BATTERY ERROR] {e}")

        for ev in pygame.event.get():
            # ESC touch button handler
            if ev.type == pygame.MOUSEBUTTONDOWN:
                x, y = ev.pos
                if 30 <= x <= 130 and 30 <= y <= 80:
                    if stream: stream.stop(); stream.close()
                    SQ.put(None)
                    pygame.quit()
                    sys.exit()

            if ev.type == pygame.QUIT or (
               ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                if stream: stream.stop(); stream.close()
                SQ.put(None)
                pygame.quit()
                sys.exit()

        sl = gst("sleeping")
        if sl:
            bo = False; bt = now
        else:
            if bo:
                if now-bt > 2000: bo=False; bt=now
            else:
                if now-bt > 200:  bo=True;  bt=now
        sst(blink_open=bo)

        with LOCK: snap = dict(S)
        draw(screen, fs, fm, snap)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()