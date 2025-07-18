import base64
import json
import logging
import os
import pickle
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

import cv2
import face_recognition
import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image, ImageTk
from dotenv import load_dotenv

load_dotenv()

ENCODINGS_FILE = "known_faces.pkl"
REGISTRO_DIR = "registros"
FRAMES_DIR = "frames"
COMPARISON_DIR = os.path.join(FRAMES_DIR, "comparisons")
LANDMARKS_DIR = os.path.join(FRAMES_DIR, "landmarks")

os.makedirs(REGISTRO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


def carregar_faces_salvas():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return [], []


def salvar_faces(known_faces, known_names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_faces, known_names), f)


def desenhar_box_e_pontos(frame, locations, landmarks):
    for (top, right, bottom, left) in locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    for face in landmarks:
        for feature in face.values():
            for point in feature:
                cv2.circle(frame, point, 1, (0, 0, 255), -1)


def salvar_landmarks_json(landmarks, nome, timestamp, tipo):
    path = os.path.join(LANDMARKS_DIR, f"{tipo}_{nome}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(landmarks, f, indent=2)


def salvar_comparacao(known_img_path, captured_frame_bgr, result, distance, name, timestamp):
    try:
        known_bgr = cv2.imread(known_img_path)
        known_rgb = cv2.cvtColor(known_bgr, cv2.COLOR_BGR2RGB)
        captured_rgb = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2RGB)
        known_locations = face_recognition.face_locations(known_rgb)
        captured_locations = face_recognition.face_locations(captured_rgb)
        known_landmarks = face_recognition.face_landmarks(known_rgb)
        captured_landmarks = face_recognition.face_landmarks(captured_rgb)
        desenhar_box_e_pontos(known_rgb, known_locations, known_landmarks)
        desenhar_box_e_pontos(captured_rgb, captured_locations, captured_landmarks)
        salvar_landmarks_json(known_landmarks, name, timestamp, "known")
        salvar_landmarks_json(captured_landmarks, name, timestamp, "captured")
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(known_rgb)
        axes[0].set_title(f"{name} (Registrado)")
        axes[0].axis("off")
        axes[1].imshow(captured_rgb)
        axes[1].set_title(f"Match: {result} - Dist: {distance:.2f}")
        axes[1].axis("off")
        plt.tight_layout()
        file_path = os.path.join(COMPARISON_DIR, f"cmp_{name}_{timestamp}.png")
        plt.savefig(file_path)
        plt.close()
    except Exception as e:
        print(f"[ERRO] ao salvar comparação visual: {e}")


def imagem_cv2_para_b64(imagem):
    _, buffer = cv2.imencode(".jpg", imagem)
    return base64.b64encode(buffer).decode("utf-8")


def comparar_via_ia(nome, imagem_registrada, imagem_capturada):
    prompt = (
        f"A imagem a seguir foi reconhecida automaticamente como sendo do usuário '{nome}', "
        "com distância facial X. Você acha razoável essa correspondência, com base em semelhança visual geral? "
        "Considere aparência, barba, óculos, posição do rosto."
    )

    try:
        registered_b64 = imagem_cv2_para_b64(imagem_registrada)
        captured_b64 = imagem_cv2_para_b64(imagem_capturada)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um especialista em verificação de identidade visual."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{registered_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{captured_b64}"}}
                    ]
                }
            ],
            max_tokens=10,
            temperature=0,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        import traceback
        logging.error("Erro ao chamar OpenAI API:\n" + traceback.format_exc())
        return f"Erro: {e}"


class FaceAuthApp:
    def __init__(self, base_gui):
        self.base_gui = base_gui
        self.base_gui.title("Reconhecimento Facial")
        self.ultimo_nome = None
        self.ultimo_bbox = []
        self.known_faces, self.known_names = carregar_faces_salvas()
        self.base_gui.columnconfigure(0, weight=1)
        self.base_gui.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(self.base_gui)
        main_frame.grid(column=0, row=0, sticky="nsew", padx=10, pady=10)
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(column=0, row=0, columnspan=3)
        self.nome_entry = ttk.Entry(main_frame, width=30)
        self.nome_entry.grid(column=0, row=1, padx=5, pady=5, sticky="w")
        self.nome_entry.insert(0, "Digite o nome do usuário")
        self.status_label = ttk.Label(main_frame, text="Status: aguardando...", foreground="blue")
        self.status_label.grid(column=1, row=1, sticky="w")
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(column=2, row=1, padx=10, sticky="e")
        style = ttk.Style()
        style.configure("TButton", padding=6, font=('Segoe UI', 10, 'bold'))
        ttk.Button(btn_frame, text="📸 Registrar", command=self.registrar).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔍 Autenticar", command=self.autenticar).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🚪 Sair", command=self.base_gui.quit).pack(side=tk.LEFT, padx=5)
        self.log_area = scrolledtext.ScrolledText(main_frame, height=10, width=100, state="disabled")
        self.log_area.grid(column=0, row=2, columnspan=3, pady=10)
        self.cap = cv2.VideoCapture(0)
        self.processar_video()

    def log(self, msg, color="black"):
        print(msg)
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.tag_add(color, "end-1l", "end-1c")
        self.log_area.tag_config(color, foreground=color)
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")

    def processar_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.base_gui.after(10, self.processar_video)
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small)
        face_landmarks = face_recognition.face_landmarks(small)
        locations = [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]
        landmarks = [{k: [(x * 2, y * 2) for (x, y) in v] for k, v in f.items()} for f in face_landmarks]
        desenhar_box_e_pontos(frame, locations, landmarks)
        if self.ultimo_nome and self.ultimo_bbox:
            for (top, right, bottom, left) in self.ultimo_bbox:
                cv2.putText(frame, self.ultimo_nome, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.base_gui.after(10, self.processar_video)

    def registrar(self):
        nome = self.nome_entry.get().strip()
        if not nome:
            self.status_label.config(text="Digite um nome válido", foreground="red")
            return
        _, frame = self.cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            self.status_label.config(text="Nenhum rosto detectado", foreground="red")
            return
        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            self.status_label.config(text="Falha ao codificar rosto", foreground="red")
            return
        known_faces, known_names = carregar_faces_salvas()
        known_faces.append(encodings[0])
        known_names.append(nome)
        salvar_faces(known_faces, known_names)
        self.known_faces, self.known_names = known_faces, known_names
        cv2.imwrite(f"{REGISTRO_DIR}/{nome}.jpg", frame)
        self.status_label.config(text=f"{nome} registrado com sucesso", foreground="green")
        self.log(f"[REGISTRO] {nome} registrado com sucesso.", "green")

    def autenticar(self):
        nome = self.nome_entry.get().strip()
        if not nome:
            self.status_label.config(text="Digite o nome esperado para autenticação", foreground="red")
            return

        known_path = os.path.join(REGISTRO_DIR, f"{nome}.jpg")
        if not os.path.exists(known_path):
            self.status_label.config(text="Usuário não registrado", foreground="red")
            return

        _, frame = self.cap.read()
        known_img = cv2.imread(known_path)
        result = comparar_via_ia(nome, known_img, frame)

        timestamp = int(time.time())
        salvar_comparacao(known_path, frame, result, 0, nome, timestamp)

        if "sim" in result.lower():
            self.ultimo_nome = nome
            locations = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.ultimo_bbox = locations
            self.status_label.config(text=f"Reconhecido: {nome}", foreground="green")
            self.log(f"[IA] {nome} reconhecido pela IA", "green")
        else:
            self.status_label.config(text="Rosto não reconhecido pela IA", foreground="red")
            self.log(f"[ERRO] {nome} não reconhecido pela IA.", "red")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAuthApp(root)
    root.mainloop()
