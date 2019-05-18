from time import time
from Search import search, new_search

def main():
    keyword = "data science data"
    result = search(keyword)
    if result is False:
        print("Not found")
    else:
        for i in result:
            print(f"{i}.txt")

    # # # new search cosine similarity # # #
    print("="*20, "new", "="*20)
    # # # new search cosine similarity # # #

    result = new_search(keyword)
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
