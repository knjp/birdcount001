from ultralytics import YOLO 
import time
import os
import csv

os.chdir('yolo')
dirbase = './'
detection_model = YOLO('model/best.pt')
#source_path = 'video/abc.mp4'
source_path = 'video/a_007.mp4'
time1 = time.time()
# source_path = "D:\\ultralytics-main\\data_detect\\03.mp4"
results = detection_model.track(source=source_path, save=True,
                                #, device=0, save=False,
#                                conf=float(values['detection_conf_thres']),
#                                iou=float(values['detection_iou_thres']),
#                                show=False,
                                conf=0.25,
                                iou=0.7,
                                save_txt=True,
                                save_frames=False,
                                persist=True,
#                                save_crop=True
                                )
time2 = time.time()
print(time2 - time1)
# c = 0
# for r in results:
#     #print(r.names)
#     #print(r.save_dir)
#     c+=1
#     m = len(r.boxes)
#     if m != 1:
#         print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
#         print('c: ' + str(c))
#         print('m = ' + str(m))
#         #print(r)
#         #print(r.boxes)
#         print(r.boxes.xywh)
#         print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
#     #print('-----------------------------------------------------')

with open('result.txt', 'w', encoding='utf-8', newline='') as tfile:
    c = 0;
    for rone in results:
        c += 1
        cstr = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' + 'c = ' + str(c) + '\n'
        tfile.write(cstr)
        for object in rone.boxes:
            rstr = str(object) + '\n'
            tfile.write(rstr)


with open('result.csv', 'w', encoding='utf-8', newline='') as cfile:
    cwriter = csv.writer(cfile)
    c = 0;
    for rone in results:
        c += 1
        for object in rone.boxes:
            print("object.numpy():")
            print(object.xywh.cpu().numpy())
            xywh = object.xywhn.cpu().numpy()
            x = xywh[0,0]
            y = xywh[0,1]
            w = xywh[0,2]
            h = xywh[0,3]
            if object.id != None:
                id = int(object.id.cpu().numpy()[0])
            else:
                id = 0
            ob = [c, id, x, y, w, h]
            cwriter.writerow(ob)

with open('result2.csv', 'w', encoding='utf-8', newline='') as cfile:
    cwriter = csv.writer(cfile)
    c = 0;
    for rone in results:
        c += 1
        for object in rone.boxes:
            print("object.numpy():")
            print(object.xyxyn.cpu().numpy())
            xyxy = object.xyxyn.cpu().numpy()
            x1 = xyxy[0,0]
            y1 = xyxy[0,1]
            x2 = xyxy[0,2]
            y2 = xyxy[0,3]
            if object.id != None:
                id = int(object.id.cpu().numpy()[0])
            else:
                id = int(0)
            ob = [c, id, x1, y1, x2, y2]
            cwriter.writerow(ob)



#     latest_folder_path_ex = get_latest_folder_contents(folder_path)
#     is_dir = os.path.isdir(latest_folder_path_ex)
#     if is_dir:
#         txt_files_change(latest_folder_path_ex, source_path)
#         results_len = len(results)
#         bird_counts(latest_folder_path_ex, results_len)
#         window.close()
#         window = make_main()
#     else:
#         pass
