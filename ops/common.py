# common.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn


def check_data_format(data_format):
    if data_format in ["NCHW", "NHWC", "nchw", "nhwc"]:
        return data_format.upper()
    elif data_format in ["NCW", "ncw"]:
        return "NCW"
    elif data_format in ["NWC", "nwc"]:
        return "NWC"
    else:
        raise ValueError("Unknown data_format: %s" % data_format)
