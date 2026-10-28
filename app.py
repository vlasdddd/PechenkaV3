import time

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO


model = YOLO(r"runs\detect\cookie_detector-4\weights\best.pt")
CONFIDENCE = 0.7


def find_and_click(class_name):
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    results = model(image, verbose=False)

    for result in results:
        for box in result.boxes:
            detected_class = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if detected_class == class_name and confidence >= CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0]
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)

                pyautogui.click(x, y)
                print(f"Клик по {class_name}: ({x}, {y})")
                return True

    return False


def click_with_timeout(class_name, timeout=5):
    deadline = time.time() + timeout

    while time.time() < deadline:
        if find_and_click(class_name):
            time.sleep(0.5)
            return True
        time.sleep(0.2)

    print(f"{class_name} не найден за {timeout} секунд — перехожу дальше.")
    return False


if __name__ == "__main__":
    while True:
        # Пытается нажать game 4 раза.
        # Если game не найден, пропускает оставшиеся клики и идёт к taper.
        for _ in range(4):
            if not click_with_timeout("game", timeout=5):
                break

        # Если taper не найден за 5 секунд, всё равно идёт к reklama.
        click_with_timeout("taper", timeout=5)

        # Если reklama не найдена за 5 секунд, цикл начинается сначала.
        click_with_timeout("reklama", timeout=5)
