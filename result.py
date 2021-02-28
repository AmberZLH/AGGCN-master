import re
with open("log/aggcn_lap14_val.txt", 'r') as f:
    for lines in f.readlines():
        line = re.split('repeat|: |max_test_acc|, |max_test_f1|', lines)
        while '' in line:
            line.remove('')
        line = list(map(float, line))
        print(line)
        f1 = []
        for i in line:
            if i < 0.8:
                f1.append(i)
                line.remove(i)
        acc = sum(x for x in line if isinstance(x, float))
        f1 = sum(x for x in f1 if isinstance(x, float))
        b = sum(range(1, 21))
        acc_avg = (acc-b)/20
        f1 = f1/20
        print("max_test_acc_avg: {0}    "   "max_test_f1_avg: {1}      ".format(acc_avg, f1))
