import os


def main():
    files = os.listdir("./dataset/test/yolo-train")

    # Create papers directory if it doesn't exist
    if not os.path.exists(os.getcwd() + "/dataset/test/annotations"):
        os.mkdir(os.getcwd() + "/dataset/test/annotations")

    for f in files:
        # Create text file with annotations
        with open("./dataset/test/annotations/" + f.replace(".jpg", "") + ".txt", "w") as f_txt:
            if "thumbs-up" in f:
                f_txt.write("0 0.5 0.5 1 1")
            elif "ok-sign" in f:
                f_txt.write("1 0.5 0.5 1 1")
            elif "l-sign" in f:
                f_txt.write("2 0.5 0.5 1 1")
            elif "victory-sign" in f:
                f_txt.write("3 0.5 0.5 1 1")
            elif "fingers-crossed" in f:
                f_txt.write("4 0.5 0.5 1 1")

    # with open("test.txt", "w") as f_txt:
    #     f_txt.write("0 0.5 0.5 1 1")


if __name__ == "__main__":
    main()