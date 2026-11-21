"""Starea comuna a detectorului (citita de face_basic.py, trimisa catre dashboard).

new_state() intoarce un dict nou, cu valori neutre, la fiecare cadru.
face_basic.py suprascrie cheile pe care le poate calcula.
"""


def new_state():
    return {
        "face_found": False,        # a fost gasita cel putin o fata
        "eyes_closed_sec": 0.0,     # de cat timp sunt ochii inchisi (secunde)
        "perclos": 0.0,             # fractiunea de timp cu ochii inchisi, ultimele 60 s
        "looking_away_sec": 0.0,    # de cat timp priveste in alta parte (secunde)
        "hand_near_face": False,    # mana in dreptul fetei (proxy pentru telefon)
        "ear": 0.30,                # eye aspect ratio (deschiderea ochilor)
        "mar": 0.25,                # mouth aspect ratio (deschiderea gurii)
        "yawning": False,           # cascat sustinut
        "pitch": 0,                 # unghi cap sus/jos (grade)
        "yaw": 0,                   # unghi cap stanga/dreapta (grade)
    }