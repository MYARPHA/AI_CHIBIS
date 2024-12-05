import tkinter as tk

# Создание окна
root = tk.Tk()
root.title("Комментарий к заказу")

# Создание поля для ввода комментария
comment_label = tk.Label(root, text="Введите комментарий:")
comment_label.pack()

comment_entry = tk.Entry(root)
comment_entry.pack()

# Создание чекбоксов для выбора типа комментария
comment_type = tk.StringVar()
comment_type.set("positive")

positive_checkbox = tk.Checkbutton(root, text="Положительный комментарий", variable=comment_type, onvalue="positive", offvalue="negative")
positive_checkbox.pack()

negative_checkbox = tk.Checkbutton(root, text="Отрицательный комментарий", variable=comment_type, onvalue="negative", offvalue="positive")
negative_checkbox.pack()

# Создание кнопки для публикации комментария
def publish_comment():
    comment = comment_entry.get()
    comment_type_value = comment_type.get()
    print(f"Комментарий: {comment}, Тип комментария: {comment_type_value}")
    # Здесь можно добавить логику для отправки комментария на сервер или в базу данных

publish_button = tk.Button(root, text="Опубликовать", command=publish_comment)
publish_button.pack()

# Запуск главного цикла
root.mainloop()

