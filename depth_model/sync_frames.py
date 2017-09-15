import sys
import glob
import os
import re
import pickle
# frameList("/storage/nyu_v2/dataset/bathroom_0001/")

class Frame:
    time_diff = 0
    depth_filename = ""
    rgb_filename = ""
    accel_filename = ""


if __name__ == "__main__":

    assert (len(sys.argv) == 2)

    main_path = sys.argv[1]
    paths = os.listdir(main_path)

    for path in paths:
        cur_path = main_path + path
        types = [cur_path + "/*.dump", cur_path + "/*.pgm", cur_path + "/*.ppm"]
        files_list = []
        for t in types:
            files_list.extend(glob.glob(t))

        for i in range(len(files_list)):
            files_list[i] = os.path.basename(files_list[i])

        r_re = re.compile('^r-*')
        r_files = list(filter(r_re.match, files_list))
        d_re = re.compile('^d-*')
        d_files = list(filter(d_re.match, files_list))
        a_re = re.compile('^a-*')
        a_files = list(filter(a_re.match, files_list))

        frames_list = []
        for i in range(len(d_files)):
            frame = Frame()
            frame.depth_filename = d_files[i]
            frames_list.append(frame)

        r_i = 0
        a_i = 0
        for d_i in range(len(d_files)):
            time_depth = float(frames_list[d_i].depth_filename.split("-")[1])
            time_rgb = float(r_files[r_i].split("-")[1])
            time_accel = float(a_files[a_i].split("-")[1])

            t_diff = abs(time_depth - time_rgb)

            while r_i < len(r_files) - 1:
                time_rgb = float(r_files[r_i + 1].split("-")[1])

                tmp_diff = abs(time_depth - time_rgb)
                if tmp_diff > t_diff:
                    break

                t_diff = tmp_diff

                r_i += 1

            frames_list[d_i].time_diff = t_diff

            t_diff = abs(time_depth-time_accel)

            while a_i < len(a_files) - 1:
                time_accel = float(a_files[a_i + 1].split("-")[1])

                tmp_diff = abs(time_depth - time_accel);
                if tmp_diff > t_diff:
                    break
                    t_diff = tmp_diff

                a_i += 1

            frames_list[d_i].depth_filename = cur_path + "/" + frames_list[d_i].depth_filename
            frames_list[d_i].rgb_filename = cur_path + "/" + r_files[r_i]
            frames_list[d_i].accel_filename = cur_path + "/" + a_files[a_i]

        with open("test_" + path + "_frames_list.pkl", 'wb') as f:
            pickle.dump(frames_list, f, pickle.HIGHEST_PROTOCOL)

        # with open("frames_list.pkl", 'rb') as f:
        #     data = pickle.load(f)
        #
            # for i in range(len(data)):
            #     print(str(data[i].time_diff))
            #     print(str(data[i].depth_filename))
            #     print(str(data[i].rgb_filename))
            #     print(str(data[i].accel_filename))
