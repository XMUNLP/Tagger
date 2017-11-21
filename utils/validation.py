# validation.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import time
import subprocess


def get_current_model(filename):
    try:
        with open(filename) as fd:
            line = fd.readline()
            if not line:
                return None

            name = line.strip().split(":")[1]
            return name.strip()[1:-1]
    except:
        return None


def read_record(filename):
    record = []

    try:
        with open(filename) as fd:
            for line in fd:
                line = line.strip().split(":")
                val = float(line[0])
                name = line[1].strip()[1:-1]
                record.append((val, name))
    except:
        pass

    return record


def write_record(filename, record):
    # sort
    sorted_record = sorted(record, key=lambda x: -x[0])

    with open(filename, "w") as fd:
        for item in sorted_record:
            val, name = item
            fd.write("%f: \"%s\"\n" % (val, name))


def write_checkpoint(filename, record):
    # sort
    sorted_record = sorted(record, key=lambda x: -x[0])

    with open(filename, "w") as fd:
        fd.write("model_checkpoint_path: \"%s\"\n" % sorted_record[0][1])
        for item in sorted_record:
            val, name = item
            fd.write("all_model_checkpoint_paths: \"%s\"\n" % name)


def add_to_record(record, item, capacity):
    added = None
    removed = None
    models = {}

    for (val, name) in record:
        models[name] = val

    if len(record) < capacity:
        if item[1] not in models:
            added = item[1]
            record.append(item)
    else:
        sorted_record = sorted(record, key=lambda x: -x[0])
        worst_score = sorted_record[-1][0]
        current_score = item[0]

        if current_score >= worst_score:
            if item[1] not in models:
                added = item[1]
                removed = sorted_record[-1][1]
                record = sorted_record[:-1] + [item]

    return added, removed, record


def validate(params):
    time.sleep(100)
    best_dir = params.model_dir + "/best"

    # create directory
    if not os.path.exists(best_dir):
        os.mkdir(best_dir)
        record = []
    else:
        record = read_record(best_dir + "/top")

    while True:
        try:
            time.sleep(params.frequency)
            model_name = get_current_model(params.model_dir + "/checkpoint")

            if model_name is None:
                continue

            # prediction and evaluation
            child = subprocess.Popen("sh %s" % params.script,
                                     shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            info = child.communicate()[0]

            if not info:
                continue

            info = info.strip().split("\n")
            overall = None

            for line in info[::-1]:
                if line.find("Overall") > 0:
                    overall = line
                    break

            if not overall:
                continue

            # run evaliation script
            f_score = float(overall.strip().split()[-1])

            # save best model
            item = (f_score, model_name)
            added, removed, record = add_to_record(record, item,
                                                   params.keep_top_k)
            log_fd = open(best_dir + "/log", "a")
            log_fd.write("%s: %f\n" % (model_name, f_score))
            log_fd.close()

            if added is not None:
                model_path = params.model_dir + "/" + model_name + "*"
                # copy model
                os.system("cp %s %s" % (model_path, best_dir))
                # update checkpoint
                write_record(best_dir + "/top", record)
                write_checkpoint(best_dir + "/checkpoint", record)

            if removed is not None:
                # remove old model
                model_name = params.model_dir + "/best/" + removed + "*"
                os.system("rm %s" % model_name)
        except:
            pass
