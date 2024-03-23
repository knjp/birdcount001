import PySimpleGUI as sg
from ultralytics import YOLO
import datetime
import os
import cv2
import glob
import shutil
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
from datetime import timedelta
import time
import fnmatch
import tkinter as tk
from tkinter import filedialog, ttk

dirbase = '/home/kiyoshi/Simulations/Jupyter/yolobird'
kizu_folder_path = []

images_train_dir = []
images_val_dir = []
labels_train_dir = []
labels_val_dir = []

png_files = []
txt_files = []

png_files_after = []
txt_files_after = []

txt_file_name_list = []

predict_folder = []


def save_frame_range(video_path, start_frame, stop_frame, step_frame,
                     dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    for n in range(start_frame, stop_frame, step_frame):

        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format( base_path, str(n).zfill(digit), ext), frame)
        else:
            return

    print("seisiga done")


def select_file():
    root = tk.Tk()
    root.title('yolov8 tori')
    root.geometry("700x350")

    source_path = ''

    def get_file_path():
        nonlocal source_path
        source_file = filedialog.askopenfilename()
        if source_file:
            source_path = os.path.abspath(source_file)
            quit()

    def quit():
        root.destroy()

    label = tk.Label(root, text="Click the Button to browse the Files", font=('Georgia 13'))
    label.pack(pady=10)
    tk.Button(root, text="select source", command=get_file_path).pack(pady=20)

    root.mainloop()

    return source_path


def distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # d = ds.cityblock(p1, p2)
    return d


def txt_files_change(folder_path, source_path):
    source_name = source_path.split('\\')[-1]
    txt_name = source_name.split('.mp4')[0]
    replace_name = ''.join([txt_name, '_'])

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            new_filename = filename.replace(replace_name, '')
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))


def get_latest_folder_contents(folder_path):
    latest_txt_paths = []
    latest_txt_paths_ex = []
    time_now = str(time.time())

    items = os.listdir(folder_path)
    items = sorted(items, key=lambda fn: os.path.getmtime(os.path.join(folder_path, fn)))

    for i in range(len(items)):
        #if items[i].__contains__('predict'):
        if items[i].__contains__('track'):
            predict_folder.append(items[i])

    latest_predict_folder = predict_folder[-1]
    latest_folder_path = os.path.join(folder_path, latest_predict_folder, "labels")

    for filename in os.listdir(latest_folder_path):
        if filename.endswith('.txt'):
            txt_path = os.path.join(latest_folder_path, filename)
            txt_paths_ex = os.path.join(latest_folder_path, time_now, filename)
            txt_paths_ex_without_name = os.path.join(latest_folder_path, time_now)

            if not os.path.exists(txt_paths_ex_without_name):
                os.makedirs(txt_paths_ex_without_name)

            latest_txt_paths.append(txt_path)
            latest_txt_paths_ex.append(txt_paths_ex)

    for i in range(len(latest_txt_paths)):
        shutil.copy2(latest_txt_paths[i], latest_txt_paths_ex[i])

    latest_folder_path_ex = os.path.join(latest_folder_path, time_now)
    return latest_folder_path_ex


def bird_counts(folder_path, results_len):
    flag = 0
    # folder_path = "C:/Users/mhakk/PycharmProjects/birdcount_MATLAB_rewritten/DataFolder"

    folder_path_list = os.listdir(folder_path)
    folder_data = []
    file_names = []

    # file_names_numbers = []
    for file_full_name in folder_path_list:
        if file_full_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_full_name)
            file_name = file_full_name.split(".txt")[0]
            file_name = file_name.split('_')[-1]
            file_name_number = int(file_name)

            print("file_name_number : " + str(file_name_number) + '(' + file_name + ')')
            with open(file_path, "r", encoding="utf-8") as file:
                # content = file.read().replace("\n", " ").split(" ")
                contents = file.read().split("\n")
                for content in contents:
                    content_data = content.split(" ")
                    mlen = len(content_data)
                    if len(content_data) > 5:
                        content_data.append(file_name_number)
                        folder_data.append(content_data)
                    else:
                        content_data.append("0")
                        content_data.append(file_name_number)
                        folder_data.append(content_data)
    # ile_names.append(file_name)

    # for file_name in file_names:
    #     file_names_number = int(file_name)
    #     file_names_numbers.append(file_names_number)
    #
    # df_name = pd.DataFrame(file_names_numbers)
    #
    # df_txt = pd.DataFrame(folder_data)
    # df_txt.dropna(axis='columns', inplace=True)
    #
    # df_datas = pd.concat([df_name, df_txt], axis=1, ignore_index=True)
    # print(df_datas.shape)
    #
    # np_datas = df_datas.to_numpy()
    #
    # np_datas = np_datas[np.argsort(np_datas[:, 0])]
    # print(np_datas)
    #
    # m = len(np_datas)
    # print(m)
    #
    # J = np.zeros(shape=(m, 7))
    # print(J)
    #
    # for i in range(m):
    #     np_data = np_datas[i, :]
    #     print(np_data)

    df_datas = pd.DataFrame(folder_data)

    df_datas.dropna(axis=0, inplace=True)

    np_datas = df_datas.to_numpy()
    np_datas = np_datas[np.argsort(np_datas[:, 6])]
    print(np_datas)

    # m = int(np.max(np_datas[:, 6]))
    m = results_len
    print(m)
    print("np_datas[6]: " + str(np_datas[:, 6]))

    J = np.zeros(shape=(m, 7))
    J1 = np.zeros(shape=(m, 7))
    B = np.zeros(shape=(m, 2))
    C = np.zeros(shape=(m, 2))
    for i in range(m):
        float_i = float(i + 1)
        np_data = np_datas[np_datas[:, 6] == float_i]
        if len(np_data) != 0:
            J[i, 0] = len(np_data)

        if len(np_data) == 1:
            J[i, 1] = np_data[0, 5]
            J[i, 2] = np_data[0, 1]
            J[i, 3] = np_data[0, 2]

        elif len(np_data) == 2:
            if np_data[0, 5] < np_data[1, 5]:
                J[i, 1] = np_data[0, 5]
                J[i, 2] = np_data[0, 1]
                J[i, 3] = np_data[0, 2]

                J[i, 4] = np_data[1, 5]
                J[i, 5] = np_data[1, 1]
                J[i, 6] = np_data[1, 2]
            else:
                J[i, 1] = np_data[1, 5]
                J[i, 2] = np_data[1, 1]
                J[i, 3] = np_data[1, 2]

                J[i, 4] = np_data[0, 5]
                J[i, 5] = np_data[0, 1]
                J[i, 6] = np_data[0, 2]

    # for i in range(2, m):
    # 	d = J[i, 0] - J[i - 1, 0]
    # 	if d >= 2:
    # 		H = np.zeros(shape=(d - 1, 7))
    # 		H[0:d - 1, 0] = range(J[i - 1, 0] + 1, J[i, 0] - 1)
    #
    # 		a = J[0:i - 1, :]
    # 		b = J[i:, :]
    #
    # 		J1 = np.concatenate((a, H), axis=1)
    # 		J1 = np.concatenate((J1, b), axis=1)
    # 		J = J1
    # m = m + d - 1
    #print(J)

    figflag2 = 1
    figflag3 = 1
    figflag4 = 1

    m2 = len(J)
    #print(m2)

    s1 = J[0, 1:4]
    s2 = J[0, 4:7]
    B[0, 0] = J[0, 2]
    B[0, 1] = J[0, 3]
    C[0, 0] = J[0, 5]
    C[0, 1] = J[0, 6]

    for i in range(1, m2):
        L = []
        d1 = distance(s1[1:3], J[i, 2: 4])
        d2 = distance(s1[1:3], J[i, 5: 7])
        d3 = distance(s2[1:3], J[i, 2: 4])
        d4 = distance(s2[1:3], J[i, 5: 7])
        L.append(d1)
        L.append(d2)
        L.append(d3)
        L.append(d4)
        loc = L.index(np.min(L))
        if loc == 0 or loc == 3:
            s1 = J[i, 1:4]
            s2 = J[i, 4:7]
            if np.any(J[:, 5]) > 0:
                B[i, 0] = J[i, 2]
                B[i, 1] = J[i, 3]
                C[i, 0] = J[i, 5]
                C[i, 1] = J[i, 6]
            else:
                B[i, 0] = J[i, 2]
                B[i, 1] = J[i, 3]
        else:
            s1 = J[i, 4:7]
            s2 = J[i, 1:4]

            if np.any(J[:, 5]) > 0:
                B[i, 0] = J[i, 5]
                B[i, 1] = J[i, 6]
                C[i, 0] = J[i, 2]
                C[i, 1] = J[i, 3]
            else:
                B[i, 0] = J[i, 2]
                B[i, 1] = J[i, 3]

    #print(B)
    #print(C)

    L3 = np.zeros(shape=(m, 4))
    t2 = np.arange(1, m + 1)
    L3[:, 0] = t2
    for i in range(len(t2)):
        if B[i, 0] > 0:
            L3[i, 1] = 1
        else:
            L3[i, 1] = 0

        if C[i, 0] > 0:
            L3[i, 2] = 1
        else:
            L3[i, 2] = 0

    L3[:, 3] = L3[:, 1] + L3[:, 2]
    L3[:, 3] = medfilt(L3[:, 3], kernel_size=5)

    if figflag4 == 1:
        plt.figure(1)
        plt.plot(L3[:, 0], L3[:, 3], '.-', label="Total")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("bird_count_with_medfilt")
        plt.axis([0, m, 0, 3])

    if figflag2 == 1:
        plt.figure(2)
        plt.plot(B[:, 0], 1 - B[:, 1], '.', label="bird1")
        plt.plot(C[:, 0], 1 - C[:, 1], '.', label="bird2")
        plt.legend()
        plt.axis([0, 1, 0, 1])

    flag1 = 0
    b = {}
    for i in range(m):
        if B[i, 0] > 0:
            b[i] = 1
            flag1 = 1
        elif (i > 0 and B[i - 1, 0] > 0 and B[i, 0] == 0 and B[i - 1, 0] < 0.6 and B[i - 2, 1] < 0.5) and flag1 == 1:
            b[i] = 1
            flag1 = 1
            B[i, :] = B[i - 1, :]
        else:
            b[i] = 0
            flag1 = 0

    flag2 = 0
    c = {}
    for i in range(m):
        if C[i, 0] > 0:
            c[i] = 1
            flag2 = 1
        elif (i > 0 and C[i - 1, 0] > 0 and C[i, 0] == 0 and C[i - 1, 0] < 0.6 and C[i - 2, 1] < 0.5) and flag2 == 1:
            c[i] = 1
            flag2 = 1
            C[i, :] = C[i - 1, :]
        else:
            c[i] = 0
            flag2 = 0

    L = np.zeros(shape=(m, 4))
    L1 = np.zeros(shape=(m, 4), dtype=str)
    t = np.arange(1, m + 1)

    L[:, 0] = t

    # median_filter(B, size=3, cval=0, mode='constant')
    # median_filter(C, size=3, cval=0, mode='constant')

    #print(len(t))
    for i in range(len(t)):
        if B[i, 0] > 0:
            L[i, 1] = 1
        else:
            L[i, 1] = 0

        if C[i, 0] > 0:
            L[i, 2] = 1
        else:
            L[i, 2] = 0

    # L[:, 1] = median_filter(L[:, 1], size=3, mode='constant')
    # L[:, 2] = median_filter(L[:, 2], size=3, mode='constant')

    L[:, 3] = L[:, 1] + L[:, 2]

    # median_filter(C, size=3, cval=0, mode='constant')

    t1 = []
    t1_str = []
    for i in range(len(t)):
        second = t[i] * 0.1
        td = timedelta(seconds=second)
        t_str = str(td)
        t1.append(t_str)
        t1_str.append(t_str)

    L1[:, 0] = t1
    #print(L1)
    L1[:, 1:4] = L[:, 1: 4]
    print(L1)

    t1_str_pd = pd.DataFrame(t1_str)
    L1_pd = pd.DataFrame(L1[:, 1:4])
    L1_data_pd = pd.concat([t1_str_pd, L1_pd], axis=1)
    L1_data_pd.to_csv('05result.csv', header=False, index=False)

    # with open('result.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 	writer = csv.writer(file_obj)
    # 	for data in L1_data_pd:
    # 		writer.writerow(data)

    if figflag3 == 1:
        plt.figure(3)
        plt.plot(L[:, 0], L[:, 1] + 0.03, '.-', label="bird1")
        plt.plot(L[:, 0], L[:, 2], '.-', label="bird2")
        plt.plot(L[:, 0], L[:, 3], '.-', label="Total")
        plt.legend()
        plt.xlabel("frame")
        plt.ylabel("bird_count")
        plt.axis([0, m, 0, 3])
        plt.show()


def make_main():
    main_layout = [
        [sg.Button("video_split")],
        [sg.Button("detection")],
        [sg.Button("Analysis")],
        [sg.Button("seisiga")],
        [sg.Button("dataset_split")],
        [sg.Button("model Create")],
        [sg.Button("Exit")]]

    return sg.Window("YoloV8 TORI GUI ", main_layout, finalize=True)


def make_video_split():
    video_split_layout = [[sg.Text('GUI split video with YoloV8')],
                          [sg.FolderBrowse(button_text='FILE CHOOSE', target='video_split_file_path'),
                           sg.In(default_text="folder choose", key='video_split_file_path')],
                          [sg.Button('video_split_Run'), sg.Button('video_split_Close')]
                          ]

    return sg.Window('YoloV8 video split', video_split_layout, finalize=True)


def make_detection():
    default_model = dirbase + "/move/best.pt"
    detection_layout = [
        [sg.Text('GUI Object Detection with YoloV8')],
        [sg.FileBrowse(button_text='FILE CHOOSE', target='detection_model_name'),
         sg.In(default_text= default_model, key='detection_model_name')],
        [sg.Text('iou_thres'), sg.InputText(default_text="0.70", key='detection_iou_thres')],
        [sg.Text('conf_thres'), sg.InputText(default_text="0.25", key='detection_conf_thres')],
        [sg.Button('detection_Run'), sg.Button('detection_Close')],
        [sg.Image(filename='', key='image')]]

    return sg.Window('YoloV8 Detection', detection_layout, finalize=True)


def make_seisiga():
    # start_frame, stop_frame, step_frame,
    seisiga_layout = [
        [sg.FileBrowse(button_text='FILE CHOOSE', target='seisiga_name'),
         sg.In(default_text= dirbase + "/data_detect/bk",
               key='seisiga_name')],
        [sg.Text('start_frame'), sg.InputText(default_text="1", key='start_frame')],
        [sg.Text('stop_frame'), sg.InputText(default_text="50", key='stop_frame')],
        [sg.Text('step_frame'), sg.InputText(default_text="1", key='step_frame')],
        [sg.Button('seisiga_Run'), sg.Button('seisiga_Stop')],
        [sg.Image(filename='', key='image')]]

    return sg.Window('YoloV8 Detection', seisiga_layout, finalize=True)


def make_model():
    model_layout = [
        [sg.Text('GUI Object Detection with YoloV8')],
        [sg.FileBrowse(button_text='Model CHOOSE', target='model_name'),
         sg.In(default_text=dirbase + "/datasets/yolov8x.yaml", key='model_name')],
        [sg.FileBrowse(button_text='Data CHOOSE', target='data_name'),
         sg.In(default_text=dirbase + "/datasets/bird.yaml", key='data_name')],

        [sg.Text('epochs'), sg.InputText(default_text="1000", key='epochs')],
        [sg.Text('batch'), sg.InputText(default_text="-1", key='batch')],
        [sg.Text('imgsz'), sg.InputText(default_text="640", key='imgsz')],
        [sg.Text('device'), sg.InputText(default_text="0", key='device')],

        [sg.Button('model_Run'), sg.Button('model_Stop'), sg.Button('model_Close')],
        [sg.Image(filename='', key='image')]]
    return sg.Window('YoloV8 Model Create', model_layout, finalize=True)


def make_dataset():
    dataset_layout = [
        [sg.Text('GUI Object Detection with YoloV8')],
        [sg.FolderBrowse(button_text='Folder Choose', target='kizu_folder_path'),
         sg.In(default_text=dirbase + "/datasets", key='kizu_folder_path'),
         sg.Text('Proportion of val'), sg.InputText(default_text="0.2", key='val_set'),
         sg.Button('dataset_Run'), sg.Button('dataset_Close')]]
    return sg.Window('YoloV8 dataset split', dataset_layout, finalize=True)


def split_videos(unsplit_videos_directory):
    delete_flag = False
    files = [file_name for file_name in os.listdir(unsplit_videos_directory) if file_name.endswith(".mp4")]

    for file_name in files:
        try:
            input_video_path = os.path.join(unsplit_videos_directory, file_name)
            input_video_path = input_video_path.replace('\\', '/')
            print(input_video_path)
            clip = VideoFileClip(input_video_path)

            current_duration = clip.duration
            if current_duration >= 2000:
                divide_into_count = 2
                single_duration = current_duration / divide_into_count

                while current_duration > single_duration:
                    subclip = clip.subclip(current_duration - single_duration, current_duration)

                    file_base_name, file_extension = os.path.splitext(file_name)
                    first_half_filename = os.path.join(unsplit_videos_directory, f"{file_base_name}_02{file_extension}")
                    file_check = first_half_filename
                    second_half_filename = os.path.join(unsplit_videos_directory,
                                                        f"{file_base_name}_01{file_extension}")

                    subclip.to_videofile(first_half_filename, codec="libx264", audio_codec="aac")
                    subclip.close()

                    subclip = clip.subclip(current_duration - 2 * single_duration, current_duration - single_duration)
                    subclip.to_videofile(second_half_filename, codec="libx264", audio_codec="aac")

                    subclip.close()

                    current_duration -= single_duration

                    print(f"Video split and saved: {first_half_filename} and {second_half_filename}")
                    clip.close()
                    delete_flag = True

            else:
                print("No need to split")
                clip.close()

        finally:
            if os.path.exists(input_video_path) and delete_flag is True:
                os.remove(input_video_path)


if __name__ == "__main__":
    window = make_main()
    run_model = False
    modelCreated_run_model = False
    seisiga_run_model = False
    video_split_run_model = False

    while True:
        event, values = window.read(timeout=0)

        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "video_split":
            window.close()
            window = make_video_split()

        elif event == "video_split_Run":
            input_video_path = values['video_split_file_path']

            split_videos(input_video_path)
            video_split_run_model = True

        elif event == "video_split_Close":
            video_split_run_model = False

            window.close()
            window = make_main()

        elif event == "detection":

            window.close()
            window = make_detection()

        elif event == "Close":

            window.close()
            window = make_model()

        elif event == 'detection_Run':

            # Set up model and parameter
            detection_model = YOLO(values['detection_model_name'])

            source_path = select_file()
            time1 = time.time()
            # source_path = "D:\\ultralytics-main\\data_detect\\03.mp4"
            results = detection_model.track(source=source_path, device=0, save=True,
                                            conf=float(values['detection_conf_thres']),
                                            iou=float(values['detection_iou_thres']),
                                            show=False,
                                            save_txt=True)
            time2 = time.time()
            print(time2 - time1)

            if len(results) > 0:
                folder_path = dirbase + "/runs/detect"

                latest_folder_path_ex = get_latest_folder_contents(folder_path)
                is_dir = os.path.isdir(latest_folder_path_ex)
                if is_dir:
                    txt_files_change(latest_folder_path_ex, source_path)
                    results_len = len(results)
                    bird_counts(latest_folder_path_ex, results_len)
                    window.close()
                    window = make_main()
                else:
                    pass

        elif event == "Analysis":

            window.close()
            folder_path = dirbase + "/runs/detect"
            latest_folder_path_ex = get_latest_folder_contents(folder_path)
            is_dir = os.path.isdir(latest_folder_path_ex)
            if is_dir:
                numfiles = len(fnmatch.filter(os.listdir(latest_folder_path_ex), '*.txt'))
                results_len = numfiles
                print('num files: ' + str(numfiles))
                bird_counts(latest_folder_path_ex, results_len)
                window.close()
                window = make_main()
            else:
                pass

        elif event == 'detection_Close':

            window.close()
            window = make_main()

        elif event == "seisiga":

            window.close()
            window = make_seisiga()



        elif event == "seisiga_Run":
            print("seisiga start")

            time_seisiga = datetime.datetime.now()
            time_str = time_seisiga.strftime("%Y_%m_%d")

            # start_frame, stop_frame, step_frame,
            save_frame_range(values['seisiga_name'],
                             int(float(values['start_frame'])), int(float(values['stop_frame'])),
                             int(float(values['step_frame'])),
                             dirbase + '/images', time_str)

            # cap = cv2.VideoCapture(values['seisiga_name'])

            seisiga_run_model = True

        elif event == "seisiga_Stop":

            window.close()
            window = make_main()

            if seisiga_run_model:
                seisiga_run_model = False
                # cap.release()

                print("seisiga end")



        elif event == "dataset_split":

            window.close()
            window = make_dataset()

        elif event == 'dataset_Run':
            # ############################################################################################################### folder check
            kizu_folder_path = values['kizu_folder_path']
            print(kizu_folder_path)

            images_dir = os.path.join(kizu_folder_path, "images")
            print(images_dir)

            labels_dir = os.path.join(kizu_folder_path, "labels")
            print(labels_dir)

            if os.path.exists(labels_dir) and os.path.exists(images_dir):

                images_train_dir = os.path.join(images_dir, "train")
                images_val_dir = os.path.join(images_dir, "val")

                if os.path.exists(images_train_dir) is not True:
                    os.makedirs(images_train_dir)

                if os.path.exists(images_val_dir) is not True:
                    os.makedirs(images_val_dir)

                labels_train_dir = os.path.join(labels_dir, "train")
                labels_val_dir = os.path.join(labels_dir, "val")

                if os.path.exists(labels_train_dir) is not True:
                    os.makedirs(labels_train_dir)

                if os.path.exists(labels_val_dir) is not True:
                    os.makedirs(labels_val_dir)

            elif os.path.exists(labels_dir) is not True and os.path.exists(images_dir) is not True:
                os.makedirs(images_dir)
                os.makedirs(labels_dir)

                images_train_dir = os.path.join(images_dir, "train")
                images_val_dir = os.path.join(images_dir, "val")
                os.makedirs(images_train_dir)
                os.makedirs(images_val_dir)

                labels_train_dir = os.path.join(labels_dir, "train")
                labels_val_dir = os.path.join(labels_dir, "val")
                os.makedirs(labels_train_dir)
                os.makedirs(labels_val_dir)

            elif os.path.exists(labels_dir) and os.path.exists(images_dir) is not True:
                os.makedirs(images_dir)
                images_train_dir = os.path.join(images_dir, "train")
                images_val_dir = os.path.join(images_dir, "val")
                os.makedirs(images_train_dir)
                os.makedirs(images_val_dir)

            elif os.path.exists(labels_dir) is not True and os.path.exists(images_dir):

                os.makedirs(labels_dir)
                labels_train_dir = os.path.join(labels_dir, "train")
                labels_val_dir = os.path.join(labels_dir, "val")
                os.makedirs(labels_train_dir)
                os.makedirs(labels_val_dir)


            else:
                print(
                    "The internal format of the folder is incorrect. Please delete all the contents of the folder.")
            # ###############################################################################################################

            txt_files = glob.glob(kizu_folder_path + "/*.txt")

            txt_files = [x for x in txt_files if "classes.txt" not in x]

            for txt_file in txt_files:
                txt_files_after.append(eval(repr(txt_file).replace('\\', '/')).replace('//', '/'))

            png_files = glob.glob(kizu_folder_path + "/*.png")
            for png_file in png_files:
                png_files_after.append(eval(repr(png_file).replace('\\', '/')).replace('//', '/'))

            for txt_file in txt_files_after:
                txt_file_path = os.path.join(txt_file)
                txt_file_name = os.path.splitext(os.path.basename(txt_file_path))[0]

                if txt_file_name != 'classes':
                    txt_file_name_list.append(txt_file_name)
                else:
                    txt_files_after.pop()

            # x for x in l if "2" not in x and "3" not in x

            print(txt_file_name_list)

            length = len(png_files_after)
            png_array = np.empty(shape=(length, 2), dtype=np.chararray)

            for i in range(length):
                png_file_path = os.path.join(png_files_after[i])
                png_file_name = os.path.splitext(os.path.basename(png_file_path))[0]
                png_array[i, 0] = png_file_path
                png_array[i, 1] = png_file_name

            png_exist_check = png_array

            for txt_file_name in txt_file_name_list:
                index = np.where(png_exist_check[:, 1] == txt_file_name)[0]
                png_exist_check = np.delete(png_exist_check, index, axis=0)

            png_not_exist_array = png_exist_check

            for png_not_exist in png_not_exist_array[:, 0]:
                if os.path.exists(png_not_exist):
                    os.remove(png_not_exist)
                    print(f" {png_not_exist} delete success！")
                else:
                    print(f" {png_not_exist} not exist。")
            # ###############################################################################################################

            txt_files_length = len(txt_files_after)

            png_txt_array = np.empty(shape=(txt_files_length, 2), dtype=np.chararray)

            png_txt_array[:, 1] = txt_files_after

            for i in range(txt_files_length):
                png_txt_array[i, 0] = eval(repr(png_txt_array[i, 1]).replace('.txt', '.png'))

            print(png_txt_array)

            val_set = float(values['val_set'])

            split_number = round(val_set * txt_files_length)
            print(split_number)

            val_data_set_list = np.random.choice(png_txt_array[:, 1], size=split_number)
            val_data_set_list = val_data_set_list.reshape((split_number, 1))

            val_data_set_index = np.where(png_txt_array[:, 1] == val_data_set_list)[1]
            print(val_data_set_index)

            train_datas = png_txt_array
            val_datas = png_txt_array[val_data_set_index, :]

            for i in range(txt_files_length):
                shutil.copy(str(train_datas[i, 0]), str(images_train_dir))
                shutil.copy(str(train_datas[i, 1]), str(labels_train_dir))

            for i in range(split_number):
                shutil.copy(str(val_datas[i, 0]), str(images_val_dir))
                shutil.copy(str(val_datas[i, 1]), str(labels_val_dir))

            datasets_images_train_dir = dirbase + "/datasets/images/train"
            datasets_labels_train_dir = dirbase + "/datasets/labels/train"
            datasets_images_val_dir = dirbase + "/datasets/images/val"
            datasets_labels_val_dir = dirbase + "/datasets/labels/val"
            for i in range(txt_files_length):
                shutil.copy(str(train_datas[i, 0]), str(datasets_images_train_dir))
                shutil.copy(str(train_datas[i, 1]), str(datasets_labels_train_dir))

            for i in range(split_number):
                shutil.copy(str(val_datas[i, 0]), str(datasets_images_val_dir))
                shutil.copy(str(val_datas[i, 1]), str(datasets_labels_val_dir))

            # copy

            print("split done")

            window.close()
            window = make_main()

        # if os.path.exists(path) and os.path.isfile(path):
        # 			os.remove(path)
        # 			print("deleteFile:" + path

        elif event == 'dataset_Close':
            window.close()
            window = make_main()

        elif event == "model Create":

            window.close()
            window = make_model()

        elif event == 'model_Run':
            model2 = YOLO(values['model_name'])
            results = model2.train(data=values['data_name'], epochs=int(values['epochs']),
                                   batch=int(values['batch']),
                                   save=True,
                                   imgsz=int(values['imgsz']), device=int(values['device']), optimizer='auto')
            print(results)
            modelCreated_run_model = True

        elif event in ('model_Stop', sg.WIN_CLOSED):

            if modelCreated_run_model:
                modelCreated_run_model = False

                window['image'].update(filename='')

                window.close()
                window = make_main()

        # When close window or press Close
        elif event == 'model_Close':

            window.close()
            window = make_main()

    if video_split_run_model:
        input_video_path = values['video_split_file_path']
        split_videos(input_video_path)

    # i = 1
    # while seisiga_run_model:
    #
    #     ret, frame = cap.read()
    #
    #     print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    #
    #     if ret:
    #
    #         dirname = 'D:\\ultralytics-main\\images'
    #         time = datetime.datetime.now()
    #         # %Y%m%d%H%M%S
    #         timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
    #
    #         cv2.imwrite(os.path.join(dirname, str(timestr) + str("_") + str(i).zfill(digit) + ".png"), frame)
    #
    #         print(str(timestr) + str("_") + str(i).zfill(digit) + ".png")
    #         i = i + 1
    #
    #     else:
    #         cap.release()
    #         seisiga_run_model = False

    window.close()
