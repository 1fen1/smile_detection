import cv2
import datetime
cap = cv2.VideoCapture(0) #Захват веб-камеры
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #каскад лиц
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') #каскад улыбок
while True: #во время работы веб-камеры
    _, frame = cap.read() #чтение кадра
    original_frame = frame.copy() #копирование кадра
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #преобразование кадра в черно-белый
    face = face_cascade.detectMultiScale(gray, 1.3, 5) #настройка параметров определения лица на черно-белом кадре, используя каскад лиц
    for x, y, w, h in face: #цикл для работы с рамкой лица (левый верхний угол рамки, ширина и высота)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2) #создание рамки лица на кадре
        face_roi = frame[y:y+h, x:x+w] #выделение изображения внутри рамки из целого кадры для последующей идентификации улыбки
        gray_roi = gray[y:y+h, x:x+w] #черно-белый вариант изображения внутри рамки
        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25) #настройка параметров определения улыбки на черно-белом кадре, используя каскад улыбок
        for x1, y1, w1, h1 in smile: #цикл для работы с рамкой улыбки
            cv2.rectangle(face_roi, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2) #создание рамки улыбки
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') #определение нынешней даты и времени в формате "гггг-мм-дд-чч-мм-сс"
            file_name = f'selfie-{time_stamp}.png' #создание png файла с названием, включающим нынешние дату и время
            cv2.imwrite(file_name, original_frame) #запись файла изображения с улыбкой в папку проекта
    cv2.imshow('camera', frame) #отображение веб-камеры с рамками, определяющими лицо и улыбку
    if cv2.waitKey(10) == ord('q'): #выключение веб-камеры при нажатии клавиши "q" через 10 милисекунд
        break