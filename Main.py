from time import time
from Search import search

def main():
    result = search("Respawn has dropped a brief update")
    if result is False:
        print("Not found")
    else:
        for i in result:
            print(f"{i}.txt")


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print("Total Times : %.4f" % (end_time-start_time))