import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import multiprocessing
import threading
import pandas as pd
from optparse import OptionParser

par = OptionParser(
    usage="python x.py -d directory name or -j json file"
)

par.add_option("-d", "--dir", action="store",
               type="str", dest="dir", default="",
               help="directory path")

par.add_option("-j", "--json", action="store",
               type="str", dest="json", default="",
               help="json file path")

(options, args) = par.parse_args()


def cv(nock):
    with open("text", "a") as txt:
        for vol in nock:
            txt.write(vol)


def prs(pee):
    data = json.load(open(pee))
    rd = pd.DataFrame(data["Reviews"])
    cv(list(rd["Content"]))


def nice(li):
    for come in range(len(li)):
        print(f"[{come + 1}] {li[come]}")


def main():
    fl = open("text", "r")
    tty = fl.read()
    fol = nltk.word_tokenize(tty)
    r_ls = []
    bitc = []
    stp = stopwords.words("english")
    for k in fol:
        if k.isalpha():
            r_ls.append(k.lower())
        elif k.isdigit() or k.isdecimal():
            r_ls.append("NUM")

    for bi in r_ls:
        if bi not in stp:
            bitc.append(bi)
    fne = []
    prt = PorterStemmer()
    for w in bitc:
        fne.append(prt.stem(w))

    bigram = nltk.ngrams(fne, 2)
    freq_dist = nltk.FreqDist(bigram)
    prob_dist = nltk.MLEProbDist(freq_dist)
    get_all = list()
    num = 0
    for ttg in freq_dist.most_common():
        if "decent" in ttg[0]:
            get_all.append(ttg)
            num += 1
            if num == 10:
                break
    #print("\n[>] top 10:\n")
    nice(get_all)


if __name__ == "__main__":
    if options.dir != "":
        threads = list()
        lsi = os.listdir(options.dir)
        #print("[+] parsing data......", end=" ")
        for i in lsi:
            gt = options.dir + "/" + i
            x = threading.Thread(target=prs, args=(gt,))
            threads.append(x)
            x.start()

        for i, thread in enumerate(threads):
            thread.join()
        #print("[done]")
        #print(f"\n[+] {len(lsi)} json file found!")
    if options.json != "":
        #print("[+] parsing json file......", end=" ")
        prs(options.json)
        #print("[done]")
    #print('\n[>] working......')
    p = multiprocessing.Process(target=main, args=())
    p.start()
    p.join()
    os.remove("text")
