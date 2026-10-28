import time

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

model = YOLO(r"runs\detect\cookie_detector-4\weights\best.pt")
CONFIDENCE = 0.7


def find_object(class_name):
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    results = model(image, verbose=False)

    for result in results:
        for box in result.boxes:
            detected_class = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if detected_class == class_name and confidence >= CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0]
                return int((x1 + x2) / 2), int((y1 + y2) / 2)

    return None


def wait_for_object(class_name, timeout=5):
    deadline = time.time() + timeout

    while time.time() < deadline:
        position = find_object(class_name)

        if position is not None:
            return position

        time.sleep(0.2)

    print(f"{class_name} не найден за {timeout} секунд.")
    return None


# Основной цикл автоматизации
while True:
    print("\n--- Начало нового цикла поиска ---")

    # 1. Находит game один раз и кликает по нему 4 раза.
    position_game = wait_for_object("game", timeout=10) # увеличили таймаут для старта, если игра долго грузится
    if position_game is not None:
        x, y = position_game
        for _ in range(4):
            pyautogui.click(x, y)
            time.sleep(0.3)
    else:
        # Если даже главный экран игры не найден, ждем немного и пробуем снова
        time.sleep(1)
        continue

    # 2. Ищет taper до 5 секунд и кликает один раз, если найдёт.
    position_taper = wait_for_object("taper")
    if position_taper is not None:
        pyautogui.click(*position_taper)
        time.sleep(0.5)

    # 3. Ищет reklama до 5 секунд.
    position_reklama = wait_for_object("reklama")
    if position_reklama is not None:
        print("Реклама найдена, кликаю.")
        pyautogui.click(*position_reklama)
        time.sleep(1)  # Даем время на закрытие/переход после рекламы
    else:
        print("Рекламы нет. Возвращаюсь к началу (поиск game), избегая зацикливания.")
        # Если рекламу не нашли — принудительно переходим на следующую итерацию True-цикла
        continue 

    print("Последовательность успешно завершена.")
    time.sleep(1)  # Небольшая пауза перед следующим полным кругом
