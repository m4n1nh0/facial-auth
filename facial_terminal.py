import cv2
import face_recognition
import os
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import json

ENCODINGS_FILE = "known_faces.pkl"
REGISTRO_DIR = "registros"
FRAMES_DIR = "frames"
COMPARISON_DIR = os.path.join(FRAMES_DIR, "comparisons")
LANDMARKS_DIR = os.path.join(FRAMES_DIR, "landmarks")

os.makedirs(REGISTRO_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)


def carregar_faces_salvas():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return [], []


def salvar_faces(known_faces, known_names):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_faces, known_names), f)


def desenhar_box(img, location, label=None):
    top, right, bottom, left = location
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    if label:
        cv2.putText(img, label, (left + 2, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def desenhar_landmarks(img, face_landmarks):
    for face in face_landmarks:
        for feature, points in face.items():
            for point in points:
                cv2.circle(img, point, 1, (0, 0, 255), -1)


def salvar_landmarks_json(face_landmarks, nome, timestamp, tipo):
    path = os.path.join(LANDMARKS_DIR, f"{tipo}_{nome}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(face_landmarks, f, indent=2)


def salvar_comparacao(known_img_path, captured_frame_bgr, result, distance, name, timestamp):
    try:
        known_img_bgr = cv2.imread(known_img_path)
        known_img_rgb = cv2.cvtColor(known_img_bgr, cv2.COLOR_BGR2RGB)
        captured_img_rgb = cv2.cvtColor(captured_frame_bgr, cv2.COLOR_BGR2RGB)

        known_locations = face_recognition.face_locations(known_img_rgb)
        captured_locations = face_recognition.face_locations(captured_img_rgb)
        known_landmarks = face_recognition.face_landmarks(known_img_rgb)
        captured_landmarks = face_recognition.face_landmarks(captured_img_rgb)

        if known_locations:
            desenhar_box(known_img_rgb, known_locations[0], "Registrado")
        if captured_locations:
            desenhar_box(captured_img_rgb, captured_locations[0], "Capturado")

        desenhar_landmarks(known_img_rgb, known_landmarks)
        desenhar_landmarks(captured_img_rgb, captured_landmarks)

        salvar_landmarks_json(known_landmarks, name, timestamp, "known")
        salvar_landmarks_json(captured_landmarks, name, timestamp, "captured")

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(known_img_rgb)
        axes[0].set_title(f"{name} (Registrado)")
        axes[0].axis("off")

        axes[1].imshow(captured_img_rgb)
        axes[1].set_title(f"Match: {result} - Dist: {distance:.2f}")
        axes[1].axis("off")

        plt.tight_layout()
        file_path = os.path.join(COMPARISON_DIR, f"cmp_{name}_{timestamp}.png")
        plt.savefig(file_path)
        plt.close()
    except Exception as e:
        print(f"Erro ao salvar comparação visual: {e}")


def registrar_usuario(name: str):
    cam = cv2.VideoCapture(0)
    print(f"Olhe para a câmera, {name}. Pressione 'r' para registrar.")

    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_landmarks = face_recognition.face_landmarks(rgb_frame)

        for location in face_locations:
            desenhar_box(frame, location)
        desenhar_landmarks(frame, face_landmarks)

        cv2.imshow("Registro", frame)
        key = cv2.waitKey(1)

        if frame_count < 3:
            frame_path = f"{FRAMES_DIR}/registro_{name}_{frame_count + 1}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        if key == ord("r"):
            if not face_locations:
                print("Nenhum rosto detectado.")
                continue

            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if encodings:
                encoding = encodings[0]
                known_faces, known_names = carregar_faces_salvas()
                known_faces.append(encoding)
                known_names.append(name)
                salvar_faces(known_faces, known_names)
                cv2.imwrite(f"{REGISTRO_DIR}/{name}.jpg", frame)
                print(f"{name} registrado com sucesso!")
                break
            else:
                print("Falha ao codificar rosto.")
        elif key == ord("q"):
            print("Registro cancelado.")
            break

    cam.release()
    cv2.destroyAllWindows()


def autenticar():
    known_faces, known_names = carregar_faces_salvas()

    if not known_faces:
        print("Nenhuma face cadastrada.")
        return

    cam = cv2.VideoCapture(0)
    print("Reconhecimento facial contínuo iniciado. Pressione 'q' para sair.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_faces, face_encoding)
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)

            name = "Desconhecido"
            color = (0, 0, 255)

            if True in matches:
                best_match_index = np.argmin(distances)
                name = f"{known_names[best_match_index]} ({distances[best_match_index]:.2f})"
                color = (0, 255, 0)

                timestamp = int(time.time())
                known_img_path = os.path.join(REGISTRO_DIR, known_names[best_match_index] + ".jpg")
                salvar_comparacao(known_img_path, frame, True, distances[best_match_index], known_names[best_match_index], timestamp)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Reconhecimento Facial Contínuo", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Encerrando reconhecimento.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Sistema de Autenticação Facial")
    print("1. Registrar novo usuário")
    print("2. Iniciar reconhecimento facial")
    print("3. Sair do sistema")

    while True:
        opcao = input("Escolha (1/2/3): ")
        if opcao == "1":
            nome = input("Digite o nome do usuário: ")
            registrar_usuario(nome)
        elif opcao == "2":
            autenticar()
        elif opcao == "3":
            break
        else:
            print("Opção inválida.")
