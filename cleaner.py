import os
'''
li_dir = os.listdir(path="truck-link")
for dir in li_dir:
    if len(dir) > 20:
        os.replace("truck-link/{0}".format(dir), "truck-link/{0}".format(dir[:20]))
'''
age_dir = os.listdir(path="truck-link")
for d in age_dir:
    nat_dir = os.listdir(path="truck-link/{}".format(d))
    for t_dir in nat_dir:
        li_dir = os.listdir(path="truck-link/{0}/{1}".format(d, t_dir))
        for dir in li_dir:
            files = os.listdir(path="truck-link/{0}/{1}/{2}".format(d, t_dir, dir))
            print(files)
            for file in files:
                size = os.stat("truck-link/{0}/{1}/{2}/{3}".format(d,t_dir, dir, file)).st_size
                if size < 25000:
                    os.remove("truck-link/{0}/{1}/{2}/{3}".format(d, t_dir, dir, file))