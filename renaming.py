import os

name_list = [13,
             14,
             15,
             16, 17, 18, 19, 20, 21, 22]

sizes = [512]
for size in sizes:
    num = 3
    for name in name_list:
        directory = 'data/final_numpy/'

        old_name = directory + str(size) + "_" + str(name) + "_300_data.npy"
        new_name = directory + str(size) + "_" + str(num) + "_300_data.npy"
        os.rename(old_name, new_name)
        old_name = directory + str(size) + "_" + str(name) + "_500_data.npy"
        new_name = directory + str(size) + "_" + str(num) + "_500_data.npy"
        os.rename(old_name, new_name)
        old_name = directory + str(size) + "_" + str(name) + "_800_data.npy"
        new_name = directory + str(size) + "_" + str(num) + "_800_data.npy"
        os.rename(old_name, new_name)

        print("Renaming Complete for "+str(name)+' '+str(size))
        num += 1
