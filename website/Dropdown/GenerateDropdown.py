import sys
import os

def main():
    dir = "../HeatMaps"
    all_dirs = sorted(os.listdir(dir), reverse = True)

    html_output = open("./Dropdown.html", "w")

    for dir in all_dirs:
        html_output.write("<option value=\"" + dir + "\">" + dir + "</option>\n")

    html_output.close()

    return 0

if __name__ == "__main__":
    main()
