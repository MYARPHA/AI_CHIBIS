import os
import subprocess
import sys


def create_and_setup_venv():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_dir = os.path.join(base_dir, "env")
    setup_dir = os.path.dirname(__file__)
    requirements_file = os.path.join(setup_dir, "requirements.txt")

    # Создание виртуального окружения
    print("Создание виртуальное окружение")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    if not os.path.exists(venv_dir):
        print("Ошибка: виртуальное окружение не создано.")
        return

    # Установка библиотек
    print("Установка необходимых библиотек в виртуальное окружение")
    if os.path.exists(requirements_file):
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")  # Windows
        if not os.path.exists(pip_path):  # Для Linux/Mac
            pip_path = os.path.join(venv_dir, "bin", "pip")
        subprocess.check_call([pip_path, "install", "-r", requirements_file])
    else:
        print(f"Файл {requirements_file} не найден в {setup_dir}.")
        return
    print("Настройка завершена! Виртуальное окружение готово.")


if __name__ == "__main__":
    create_and_setup_venv()
